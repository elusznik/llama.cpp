#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "ggml-turboq.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

struct turboq_matrix_cache {
    float *  d_M  = nullptr;
    int      d    = 0;
    uint64_t seed = 0;
};

static turboq_matrix_cache g_turboq_rotation_cache[GGML_CUDA_MAX_DEVICES];
static turboq_matrix_cache g_turboq_projection_cache[GGML_CUDA_MAX_DEVICES];

static const float * turboq_get_matrix_device(
        turboq_matrix_cache * caches,
        int d,
        uint64_t seed,
        const float * (*get_host_matrix)(int64_t, uint64_t)) {
    GGML_ASSERT(d > 0);

    const int device = ggml_cuda_get_device();
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);

    auto & cache = caches[device];
    if (cache.d_M != nullptr && cache.d == d && cache.seed == seed) {
        return cache.d_M;
    }

    if (cache.d_M != nullptr) {
        CUDA_CHECK(cudaFree(cache.d_M));
        cache.d_M = nullptr;
        cache.d   = 0;
    }

    const float * M_host = get_host_matrix(d, seed);
    const size_t size    = size_t(d) * size_t(d) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&cache.d_M, size));
    CUDA_CHECK(cudaMemcpy(cache.d_M, M_host, size, cudaMemcpyHostToDevice));

    cache.d    = d;
    cache.seed = seed;

    return cache.d_M;
}

static const float * tbq_get_rotation_device(int d, uint64_t seed, cudaStream_t stream) {
    GGML_UNUSED(stream);
    return turboq_get_matrix_device(g_turboq_rotation_cache, d, seed, turboq_get_rotation);
}

static const float * tbq_get_projection_device(int d, uint64_t seed, cudaStream_t stream) {
    GGML_UNUSED(stream);
    return turboq_get_matrix_device(g_turboq_projection_cache, d, seed, turboq_get_projection);
}

static __device__ __forceinline__ uint8_t quantize_tbq2_scalar(float x) {
    if (x >= 0.0000f) return x >= 0.9816f ? 3 : 2;
    return x >= -0.9816f ? 1 : 0;
}

static __device__ __forceinline__ uint8_t quantize_tbq3_scalar(float x) {
    if (x >= 1.7480f) return 7;
    if (x >= 1.0500f) return 6;
    if (x >= 0.5006f) return 5;
    if (x >= 0.0000f) return 4;
    if (x >= -0.5006f) return 3;
    if (x >= -1.0500f) return 2;
    if (x >= -1.7480f) return 1;
    return 0;
}

static __device__ __forceinline__ uint8_t quantize_tbq4_scalar(float x) {
    if (x >=  2.4008f) return 15;
    if (x >=  1.8435f) return 14;
    if (x >=  1.4371f) return 13;
    if (x >=  1.0993f) return 12;
    if (x >=  0.7996f) return 11;
    if (x >=  0.5225f) return 10;
    if (x >=  0.2583f) return 9;
    if (x >=  0.0000f) return 8;
    if (x >= -0.2583f) return 7;
    if (x >= -0.5225f) return 6;
    if (x >= -0.7996f) return 5;
    if (x >= -1.0993f) return 4;
    if (x >= -1.4371f) return 3;
    if (x >= -1.8435f) return 2;
    if (x >= -2.4008f) return 1;
    return 0;
}

static __device__ __forceinline__ float tbq2_codebook_value(uint8_t idx) {
    switch (idx) {
        case 0: return -1.5104f;
        case 1: return -0.4529f;
        case 2: return  0.4529f;
        default: return  1.5104f;
    }
}

static __device__ __forceinline__ float tbq3_codebook_value(uint8_t idx) {
    switch (idx) {
        case 0: return -2.1520f;
        case 1: return -1.3440f;
        case 2: return -0.7560f;
        case 3: return -0.2451f;
        case 4: return  0.2451f;
        case 5: return  0.7560f;
        case 6: return  1.3440f;
        default: return  2.1520f;
    }
}

template <typename idx_t>
static __global__ void k_set_rows_tbq3(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_tbq3_0 * __restrict__ dst,
        const int64_t ne_rows,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const int64_t nc,
        const float * __restrict__ Q,
        const uint3 ne01_fd,
        const uint3 ne02_fd,
        const uint3 ne11_fd,
        const uint3 ne12_fd) {
    const int64_t row_id = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_id >= ne_rows) {
        return;
    }

    uint32_t tmp = (uint32_t) row_id;
    uint2 div_mod = fast_div_modulo(tmp, ne01_fd);
    const int64_t i01 = div_mod.y;
    tmp = div_mod.x;

    div_mod = fast_div_modulo(tmp, ne02_fd);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_tbq3_0 * dst_row_ptr = (block_tbq3_0 *) ((char *) dst + dst_row*s1 + i02*s2 + i03*s3);

    extern __shared__ unsigned char smem[];
    float * s_row = (float *) smem;
    float * s_reduce = s_row + nc;
    uint8_t * s_idx = (uint8_t *) (s_reduce + blockDim.x);

    float norm_sq = 0.0f;
    for (int64_t i = tid; i < nc; i += blockDim.x) {
        const float v = src0_row[i];
        s_row[i] = v;
        norm_sq += v * v;
    }

    s_reduce[tid] = norm_sq;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float norm = sqrtf(s_reduce[0]);
        s_reduce[0] = norm < 1e-10f ? 1e-10f : norm;
    }
    __syncthreads();

    const float norm = s_reduce[0];
    const float inv_norm = 1.0f / norm;
    const float scale_up = sqrtf((float) nc);
    const int64_t nb = nc / QK_K;

    for (int64_t i = tid; i < nc; i += blockDim.x) {
        s_row[i] *= inv_norm;
    }
    __syncthreads();

    for (int64_t i = tid; i < nc; i += blockDim.x) {
        const float * q_row = Q + i*nc;
        float sum = 0.0f;
        for (int64_t j = 0; j < nc; ++j) {
            sum += q_row[j] * s_row[j];
        }
        s_idx[i] = quantize_tbq3_scalar(sum * scale_up);
    }
    __syncthreads();

    for (int64_t b = tid; b < nb; b += blockDim.x) {
        dst_row_ptr[b].d = (b == 0) ? __float2half(norm) : __float2half(0.0f);
    }

    constexpr int TBQ3_GROUP = 8;
    constexpr int TBQ3_GROUPS_PER_BLOCK = QK_K / TBQ3_GROUP;
    for (int64_t group_idx = tid; group_idx < nb * TBQ3_GROUPS_PER_BLOCK; group_idx += blockDim.x) {
        const int64_t block_idx = group_idx / TBQ3_GROUPS_PER_BLOCK;
        const int64_t group_in_block = group_idx % TBQ3_GROUPS_PER_BLOCK;
        const int64_t base = block_idx * QK_K + group_in_block * TBQ3_GROUP;

        uint32_t bits = 0;
        #pragma unroll
        for (int j = 0; j < TBQ3_GROUP; ++j) {
            bits |= uint32_t(s_idx[base + j] & 0x7u) << (j * 3);
        }

        const int64_t byte_offset = group_in_block * 3;
        dst_row_ptr[block_idx].qs[byte_offset + 0] = uint8_t(bits & 0xffu);
        dst_row_ptr[block_idx].qs[byte_offset + 1] = uint8_t((bits >> 8) & 0xffu);
        dst_row_ptr[block_idx].qs[byte_offset + 2] = uint8_t((bits >> 16) & 0xffu);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template <typename idx_t>
static __global__ void k_set_rows_tbq4(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_tbq4_0 * __restrict__ dst,
        const int64_t ne_rows,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const int64_t nc,
        const float * __restrict__ Q,
        const uint3 ne01_fd,
        const uint3 ne02_fd,
        const uint3 ne11_fd,
        const uint3 ne12_fd) {
    const int64_t row_id = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_id >= ne_rows) {
        return;
    }

    uint32_t tmp = (uint32_t) row_id;
    uint2 div_mod = fast_div_modulo(tmp, ne01_fd);
    const int64_t i01 = div_mod.y;
    tmp = div_mod.x;

    div_mod = fast_div_modulo(tmp, ne02_fd);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_tbq4_0 * dst_row_ptr = (block_tbq4_0 *) ((char *) dst + dst_row*s1 + i02*s2 + i03*s3);

    extern __shared__ unsigned char smem[];
    float * s_row = (float *) smem;
    float * s_reduce = s_row + nc;
    uint8_t * s_idx = (uint8_t *) (s_reduce + blockDim.x);

    float norm_sq = 0.0f;
    for (int64_t i = tid; i < nc; i += blockDim.x) {
        const float v = src0_row[i];
        s_row[i] = v;
        norm_sq += v * v;
    }

    s_reduce[tid] = norm_sq;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float norm = sqrtf(s_reduce[0]);
        s_reduce[0] = norm < 1e-10f ? 1e-10f : norm;
    }
    __syncthreads();

    const float norm = s_reduce[0];
    const float inv_norm = 1.0f / norm;
    const float scale_up = sqrtf((float) nc);
    const int64_t nb = nc / QK_K;

    for (int64_t i = tid; i < nc; i += blockDim.x) {
        s_row[i] *= inv_norm;
    }
    __syncthreads();

    for (int64_t i = tid; i < nc; i += blockDim.x) {
        const float * q_row = Q + i*nc;
        float sum = 0.0f;
        for (int64_t j = 0; j < nc; ++j) {
            sum += q_row[j] * s_row[j];
        }
        s_idx[i] = quantize_tbq4_scalar(sum * scale_up);
    }
    __syncthreads();

    for (int64_t b = tid; b < nb; b += blockDim.x) {
        dst_row_ptr[b].d = (b == 0) ? __float2half(norm) : __float2half(0.0f);
    }

    for (int64_t byte_idx = tid; byte_idx < nb * (QK_K / 2); byte_idx += blockDim.x) {
        const int64_t block_idx = byte_idx / (QK_K / 2);
        const int64_t byte_in_block = byte_idx % (QK_K / 2);
        const int64_t base = block_idx * QK_K + byte_in_block * 2;
        const uint8_t lo = s_idx[base + 0] & 0x0fu;
        const uint8_t hi = s_idx[base + 1] & 0x0fu;
        dst_row_ptr[block_idx].qs[byte_in_block] = lo | (hi << 4);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template <typename idx_t>
static __global__ void k_set_rows_tbqp3(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_tbqp3_0 * __restrict__ dst,
        const int64_t ne_rows,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const int64_t nc,
        const float * __restrict__ Q,
        const float * __restrict__ S,
        const uint3 ne01_fd,
        const uint3 ne02_fd,
        const uint3 ne11_fd,
        const uint3 ne12_fd) {
    const int64_t row_id = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_id >= ne_rows) {
        return;
    }

    uint32_t tmp = (uint32_t) row_id;
    uint2 div_mod = fast_div_modulo(tmp, ne01_fd);
    const int64_t i01 = div_mod.y;
    tmp = div_mod.x;

    div_mod = fast_div_modulo(tmp, ne02_fd);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_tbqp3_0 * dst_row_ptr = (block_tbqp3_0 *) ((char *) dst + dst_row*s1 + i02*s2 + i03*s3);

    extern __shared__ unsigned char smem[];
    float * s_row = (float *) smem;
    float * s_tmp = s_row + nc;
    float * s_reduce = s_tmp + nc;
    uint8_t * s_idx = (uint8_t *) (s_reduce + blockDim.x);

    float norm_sq = 0.0f;
    for (int64_t i = tid; i < nc; i += blockDim.x) {
        const float v = src0_row[i];
        s_row[i] = v;
        norm_sq += v * v;
    }

    s_reduce[tid] = norm_sq;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float norm = sqrtf(s_reduce[0]);
        s_reduce[0] = norm < 1e-10f ? 1e-10f : norm;
    }
    __syncthreads();

    const float norm = s_reduce[0];
    const float inv_norm = 1.0f / norm;
    const float scale_up = sqrtf((float) nc);
    const float scale_down = 1.0f / scale_up;
    const int64_t nb = nc / QK_K;

    for (int64_t i = tid; i < nc; i += blockDim.x) {
        s_row[i] *= inv_norm;
    }
    __syncthreads();

    for (int64_t i = tid; i < nc; i += blockDim.x) {
        const float * q_row = Q + i*nc;
        float sum = 0.0f;
        for (int64_t j = 0; j < nc; ++j) {
            sum += q_row[j] * s_row[j];
        }
        s_idx[i] = quantize_tbq2_scalar(sum * scale_up);
        s_tmp[i] = tbq2_codebook_value(s_idx[i]) * scale_down;
    }
    __syncthreads();

    float gamma_sq = 0.0f;
    for (int64_t i = tid; i < nc; i += blockDim.x) {
        float mse_sum = 0.0f;
        for (int64_t j = 0; j < nc; ++j) {
            mse_sum += Q[j*nc + i] * s_tmp[j];
        }
        const float residual = s_row[i] - mse_sum;
        s_row[i] = residual;
        gamma_sq += residual * residual;
    }

    s_reduce[tid] = gamma_sq;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_reduce[0] = sqrtf(s_reduce[0]);
    }
    __syncthreads();

    const float gamma = s_reduce[0];

    for (int64_t i = tid; i < nc; i += blockDim.x) {
        const float * s_proj_row = S + i*nc;
        float qjl_sum = 0.0f;
        for (int64_t j = 0; j < nc; ++j) {
            qjl_sum += s_proj_row[j] * s_row[j];
        }
        s_tmp[i] = qjl_sum;
    }
    __syncthreads();

    for (int64_t b = tid; b < nb; b += blockDim.x) {
        if (b == 0) {
            dst_row_ptr[b].d = __float2half(norm);
            dst_row_ptr[b].gamma = __float2half(gamma);
        } else {
            dst_row_ptr[b].d = __float2half(0.0f);
            dst_row_ptr[b].gamma = __float2half(0.0f);
        }
    }

    for (int64_t byte_idx = tid; byte_idx < nb * (QK_K / 4); byte_idx += blockDim.x) {
        const int64_t block_idx = byte_idx / (QK_K / 4);
        const int64_t byte_in_block = byte_idx % (QK_K / 4);
        const int64_t base = block_idx * QK_K + byte_in_block * 4;
        uint8_t packed = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            packed |= (s_idx[base + j] & 0x3u) << (j * 2);
        }
        dst_row_ptr[block_idx].qs[byte_in_block] = packed;
    }

    for (int64_t byte_idx = tid; byte_idx < nb * (QK_K / 8); byte_idx += blockDim.x) {
        const int64_t block_idx = byte_idx / (QK_K / 8);
        const int64_t byte_in_block = byte_idx % (QK_K / 8);
        const int64_t base = block_idx * QK_K + byte_in_block * 8;
        uint8_t packed = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (s_tmp[base + j] >= 0.0f) {
                packed |= 1u << j;
            }
        }
        dst_row_ptr[block_idx].signs[byte_in_block] = packed;
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template <typename idx_t>
static __global__ void k_set_rows_tbqp4(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_tbqp4_0 * __restrict__ dst,
        const int64_t ne_rows,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const int64_t nc,
        const float * __restrict__ Q,
        const float * __restrict__ S,
        const uint3 ne01_fd,
        const uint3 ne02_fd,
        const uint3 ne11_fd,
        const uint3 ne12_fd) {
    const int64_t row_id = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_id >= ne_rows) {
        return;
    }

    uint32_t tmp = (uint32_t) row_id;
    uint2 div_mod = fast_div_modulo(tmp, ne01_fd);
    const int64_t i01 = div_mod.y;
    tmp = div_mod.x;

    div_mod = fast_div_modulo(tmp, ne02_fd);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_tbqp4_0 * dst_row_ptr = (block_tbqp4_0 *) ((char *) dst + dst_row*s1 + i02*s2 + i03*s3);

    extern __shared__ unsigned char smem[];
    float * s_row = (float *) smem;
    float * s_tmp = s_row + nc;
    float * s_reduce = s_tmp + nc;
    uint8_t * s_idx = (uint8_t *) (s_reduce + blockDim.x);

    float norm_sq = 0.0f;
    for (int64_t i = tid; i < nc; i += blockDim.x) {
        const float v = src0_row[i];
        s_row[i] = v;
        norm_sq += v * v;
    }

    s_reduce[tid] = norm_sq;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float norm = sqrtf(s_reduce[0]);
        s_reduce[0] = norm < 1e-10f ? 1e-10f : norm;
    }
    __syncthreads();

    const float norm = s_reduce[0];
    const float inv_norm = 1.0f / norm;
    const float scale_up = sqrtf((float) nc);
    const float scale_down = 1.0f / scale_up;
    const int64_t nb = nc / QK_K;

    for (int64_t i = tid; i < nc; i += blockDim.x) {
        s_row[i] *= inv_norm;
    }
    __syncthreads();

    for (int64_t i = tid; i < nc; i += blockDim.x) {
        const float * q_row = Q + i*nc;
        float sum = 0.0f;
        for (int64_t j = 0; j < nc; ++j) {
            sum += q_row[j] * s_row[j];
        }
        s_idx[i] = quantize_tbq3_scalar(sum * scale_up);
        s_tmp[i] = tbq3_codebook_value(s_idx[i]) * scale_down;
    }
    __syncthreads();

    float gamma_sq = 0.0f;
    for (int64_t i = tid; i < nc; i += blockDim.x) {
        float mse_sum = 0.0f;
        for (int64_t j = 0; j < nc; ++j) {
            mse_sum += Q[j*nc + i] * s_tmp[j];
        }
        const float residual = s_row[i] - mse_sum;
        s_row[i] = residual;
        gamma_sq += residual * residual;
    }

    s_reduce[tid] = gamma_sq;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_reduce[0] = sqrtf(s_reduce[0]);
    }
    __syncthreads();

    const float gamma = s_reduce[0];

    for (int64_t i = tid; i < nc; i += blockDim.x) {
        const float * s_proj_row = S + i*nc;
        float qjl_sum = 0.0f;
        for (int64_t j = 0; j < nc; ++j) {
            qjl_sum += s_proj_row[j] * s_row[j];
        }
        s_tmp[i] = qjl_sum;
    }
    __syncthreads();

    for (int64_t b = tid; b < nb; b += blockDim.x) {
        if (b == 0) {
            dst_row_ptr[b].d = __float2half(norm);
            dst_row_ptr[b].gamma = __float2half(gamma);
        } else {
            dst_row_ptr[b].d = __float2half(0.0f);
            dst_row_ptr[b].gamma = __float2half(0.0f);
        }
    }

    constexpr int TBQ3_GROUP = 8;
    constexpr int TBQ3_GROUPS_PER_BLOCK = QK_K / TBQ3_GROUP;
    for (int64_t group_idx = tid; group_idx < nb * TBQ3_GROUPS_PER_BLOCK; group_idx += blockDim.x) {
        const int64_t block_idx = group_idx / TBQ3_GROUPS_PER_BLOCK;
        const int64_t group_in_block = group_idx % TBQ3_GROUPS_PER_BLOCK;
        const int64_t base = block_idx * QK_K + group_in_block * TBQ3_GROUP;

        uint32_t bits = 0;
        #pragma unroll
        for (int j = 0; j < TBQ3_GROUP; ++j) {
            bits |= uint32_t(s_idx[base + j] & 0x7u) << (j * 3);
        }

        const int64_t byte_offset = group_in_block * 3;
        dst_row_ptr[block_idx].qs[byte_offset + 0] = uint8_t(bits & 0xffu);
        dst_row_ptr[block_idx].qs[byte_offset + 1] = uint8_t((bits >> 8) & 0xffu);
        dst_row_ptr[block_idx].qs[byte_offset + 2] = uint8_t((bits >> 16) & 0xffu);
    }

    for (int64_t byte_idx = tid; byte_idx < nb * (QK_K / 8); byte_idx += blockDim.x) {
        const int64_t block_idx = byte_idx / (QK_K / 8);
        const int64_t byte_in_block = byte_idx % (QK_K / 8);
        const int64_t base = block_idx * QK_K + byte_in_block * 8;
        uint8_t packed = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (s_tmp[base + j] >= 0.0f) {
                packed |= 1u << j;
            }
        }
        dst_row_ptr[block_idx].signs[byte_in_block] = packed;
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template <typename idx_t>
static void set_rows_cuda_tbq3(
        const float * src0_d, const idx_t * src1_d, block_tbq3_0 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t ne_rows = ne01 * ne02 * ne03;

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_rows > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint64_t seed = turboq_seed_from_row(0);
        const float * d_Q = tbq_get_rotation_device((int) ne00, seed, stream);

        const size_t shared_bytes = size_t(ne00) * sizeof(float) + size_t(CUDA_SET_ROWS_BLOCK_SIZE) * sizeof(float) + size_t(ne00) * sizeof(uint8_t);
        GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_tbq3<<<ne_rows, CUDA_SET_ROWS_BLOCK_SIZE, shared_bytes, stream>>>(
                src0_d, src1_d, dst_d, ne_rows, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00,
                d_Q, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <typename idx_t>
static void set_rows_cuda_tbq4(
        const float * src0_d, const idx_t * src1_d, block_tbq4_0 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t ne_rows = ne01 * ne02 * ne03;

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_rows > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint64_t seed = turboq_seed_from_row(0);
        const float * d_Q = tbq_get_rotation_device((int) ne00, seed, stream);

        const size_t shared_bytes = size_t(ne00) * sizeof(float) + size_t(CUDA_SET_ROWS_BLOCK_SIZE) * sizeof(float) + size_t(ne00) * sizeof(uint8_t);
        GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_tbq4<<<ne_rows, CUDA_SET_ROWS_BLOCK_SIZE, shared_bytes, stream>>>(
                src0_d, src1_d, dst_d, ne_rows, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00,
                d_Q, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <typename idx_t>
static void set_rows_cuda_tbqp3(
        const float * src0_d, const idx_t * src1_d, block_tbqp3_0 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t ne_rows = ne01 * ne02 * ne03;

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_rows > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint64_t seed = turboq_seed_from_row(0);
        const float * d_Q = tbq_get_rotation_device((int) ne00, seed, stream);
        const float * d_S = tbq_get_projection_device((int) ne00, seed, stream);

        const size_t shared_bytes =
            2 * size_t(ne00) * sizeof(float) +
            size_t(CUDA_SET_ROWS_BLOCK_SIZE) * sizeof(float) +
            size_t(ne00) * sizeof(uint8_t);
        GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_tbqp3<<<ne_rows, CUDA_SET_ROWS_BLOCK_SIZE, shared_bytes, stream>>>(
                src0_d, src1_d, dst_d, ne_rows, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00,
                d_Q, d_S, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <typename idx_t>
static void set_rows_cuda_tbqp4(
        const float * src0_d, const idx_t * src1_d, block_tbqp4_0 * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t ne_rows = ne01 * ne02 * ne03;

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_rows > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint64_t seed = turboq_seed_from_row(0);
        const float * d_Q = tbq_get_rotation_device((int) ne00, seed, stream);
        const float * d_S = tbq_get_projection_device((int) ne00, seed, stream);

        const size_t shared_bytes =
            2 * size_t(ne00) * sizeof(float) +
            size_t(CUDA_SET_ROWS_BLOCK_SIZE) * sizeof(float) +
            size_t(ne00) * sizeof(uint8_t);
        GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_tbqp4<<<ne_rows, CUDA_SET_ROWS_BLOCK_SIZE, shared_bytes, stream>>>(
                src0_d, src1_d, dst_d, ne_rows, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00,
                d_Q, d_S, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * __restrict__ src0,
                                  const idx_t * __restrict__ src1,
                                  dst_t * __restrict__ dst,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows<<<grid_size, block_size, 0, stream>>>(src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
                                                         s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
                                                         ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TBQ3_0) {
        set_rows_cuda_tbq3(
            src0_d, src1_d, (block_tbq3_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TBQ4_0) {
        set_rows_cuda_tbq4(
            src0_d, src1_d, (block_tbq4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TBQP3_0) {
        set_rows_cuda_tbqp3(
            src0_d, src1_d, (block_tbqp3_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TBQP4_0) {
        set_rows_cuda_tbqp4(
            src0_d, src1_d, (block_tbqp4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}
