#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "ggml-turboq.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TURBOQ_KV_DIM 128

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
    float * s_reduce = s_row + QK_K;
    uint8_t * s_idx = (uint8_t *) (s_reduce + blockDim.x);

    const float scale_up = sqrtf((float) QK_K);
    const int64_t nb = nc / QK_K;

    for (int64_t b = 0; b < nb; ++b) {
        const int64_t base = b * QK_K;
        const float v = src0_row[base + tid];
        s_row[tid] = v;
        s_reduce[tid] = v * v;
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
            dst_row_ptr[b].d = __float2half(s_reduce[0]);
        }
        __syncthreads();

        const float inv_norm = 1.0f / s_reduce[0];
        s_row[tid] *= inv_norm;
        __syncthreads();

        const int half = tid / TURBOQ_KV_DIM;
        const int col  = tid % TURBOQ_KV_DIM;
        const float * q_col = Q + col * TURBOQ_KV_DIM;
        const float * x_half = s_row + half * TURBOQ_KV_DIM;

        float sum = 0.0f;
        for (int j = 0; j < TURBOQ_KV_DIM; ++j) {
            sum += q_col[j] * x_half[j];
        }
        s_idx[tid] = quantize_tbq3_scalar(sum * scale_up);
        __syncthreads();

        constexpr int TBQ3_GROUP = 8;
        constexpr int TBQ3_GROUPS_PER_BLOCK = QK_K / TBQ3_GROUP;
        if (tid < TBQ3_GROUPS_PER_BLOCK) {
            const int64_t group_in_block = tid;
            const int64_t group_base = group_in_block * TBQ3_GROUP;

            uint32_t bits = 0;
            #pragma unroll
            for (int j = 0; j < TBQ3_GROUP; ++j) {
                bits |= uint32_t(s_idx[group_base + j] & 0x7u) << (j * 3);
            }

            const int64_t byte_offset = group_in_block * 3;
            dst_row_ptr[b].qs[byte_offset + 0] = uint8_t(bits & 0xffu);
            dst_row_ptr[b].qs[byte_offset + 1] = uint8_t((bits >> 8) & 0xffu);
            dst_row_ptr[b].qs[byte_offset + 2] = uint8_t((bits >> 16) & 0xffu);
        }
        __syncthreads();
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
    float * s_reduce = s_row + QK_K;
    uint8_t * s_idx = (uint8_t *) (s_reduce + blockDim.x);

    const float scale_up = sqrtf((float) QK_K);
    const int64_t nb = nc / QK_K;

    for (int64_t b = 0; b < nb; ++b) {
        const int64_t base = b * QK_K;
        const float v = src0_row[base + tid];
        s_row[tid] = v;
        s_reduce[tid] = v * v;
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
            dst_row_ptr[b].d = __float2half(s_reduce[0]);
        }
        __syncthreads();

        const float inv_norm = 1.0f / s_reduce[0];
        s_row[tid] *= inv_norm;
        __syncthreads();

        const int half = tid / TURBOQ_KV_DIM;
        const int col  = tid % TURBOQ_KV_DIM;
        const float * q_col = Q + col * TURBOQ_KV_DIM;
        const float * x_half = s_row + half * TURBOQ_KV_DIM;

        float sum = 0.0f;
        for (int j = 0; j < TURBOQ_KV_DIM; ++j) {
            sum += q_col[j] * x_half[j];
        }
        s_idx[tid] = quantize_tbq4_scalar(sum * scale_up);
        __syncthreads();

        if (tid < QK_K / 2) {
            const int idx0 = 2 * tid;
            const uint8_t lo = s_idx[idx0 + 0] & 0x0fu;
            const uint8_t hi = s_idx[idx0 + 1] & 0x0fu;
            dst_row_ptr[b].qs[tid] = lo | (hi << 4);
        }
        __syncthreads();
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// TBQ34_0: mixed 3.5-bit - 64 regular channels (3-bit) + 64 outlier channels (4-bit) per 128-wide half
template <typename idx_t>
static __global__ void k_set_rows_tbq34(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_tbq34_0 * __restrict__ dst,
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
    block_tbq34_0 * dst_row_ptr = (block_tbq34_0 *) ((char *) dst + dst_row*s1 + i02*s2 + i03*s3);

    extern __shared__ unsigned char smem[];
    float * s_row = (float *) smem;
    float * s_reduce = s_row + QK_K;
    float * s_rot = s_reduce + blockDim.x;
    uint8_t * s_q4 = (uint8_t *) (s_rot + QK_K);

    const float scale_up = sqrtf((float) QK_K);
    const int64_t nb = nc / QK_K;

    for (int64_t b = 0; b < nb; ++b) {
        const int64_t base = b * QK_K;
        const float v = src0_row[base + tid];
        s_row[tid] = v;
        s_reduce[tid] = v * v;
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
            dst_row_ptr[b].d = __float2half(s_reduce[0]);
        }
        __syncthreads();

        const float inv_norm = 1.0f / s_reduce[0];

        // Compute rotation and store in s_rot
        const int half = tid / TURBOQ_KV_DIM;
        const int col  = tid % TURBOQ_KV_DIM;
        const float * q_col = Q + col * TURBOQ_KV_DIM;
        const float * x_half = s_row + half * TURBOQ_KV_DIM;

        float sum = 0.0f;
        for (int j = 0; j < TURBOQ_KV_DIM; ++j) {
            sum += q_col[j] * x_half[j];
        }
        s_rot[tid] = sum * scale_up;
        __syncthreads();

        // Quantize: 4-bit for all values first (for outlier channels)
        s_q4[tid] = quantize_tbq4_scalar(s_rot[tid]);
        __syncthreads();

        // Pack 4-bit values for outlier channels (64-127 in each half) into qs_hi
        // Each thread packs 2 values: thread t handles values (2t) and (2t+1) in its half
        if (tid < QK_K / 2) {
            const int val_idx = 2 * tid + half * TURBOQ_KV_DIM + TURBOQ_KV_DIM; // outlier offset
            const uint8_t lo = s_q4[val_idx - 1] & 0x0fu;  // value at outlier index - 1
            const uint8_t hi = s_q4[val_idx] & 0x0fu;      // value at outlier index
            dst_row_ptr[b].qs_hi[tid + half * (QK_K/2)] = lo | (hi << 4);
        }
        __syncthreads();

        // Now quantize and pack 3-bit values for regular channels (0-63 in each half)
        // Each thread recomputes its 3-bit quantization
        if (col < 64) {
            s_q4[tid] = quantize_tbq3_scalar(s_rot[tid]);  // reuse s_q4 for 3-bit indices
        }
        __syncthreads();

        // Pack 3-bit values for regular channels into qs_lo
        // Layout: half0 bytes 0-23, half1 bytes 24-47 (48 bytes total for 128 values)
        // 8 values per 3 bytes
        if (col < 64) {
            const int lo_half_off = half * 24;  // byte offset for this half
            const int val_idx = col;  // 0-63
            const int group_idx = val_idx / 8;  // which group (0-7)
            const int byte_base = lo_half_off + group_idx * 3;  // starting byte for this group

            // Compute 3-bit packed value for all 8 values in this group
            uint32_t packed = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int vi = group_idx * 8 + i;  // value index within half (0-63)
                packed |= uint32_t(s_q4[half * TURBOQ_KV_DIM + vi] & 0x7u) << (i * 3);
            }

            // Write 3 bytes - only threads with col < 8 write (one per group)
            if (col < 8) {
                dst_row_ptr[b].qs_lo[byte_base + 0] = uint8_t(packed & 0xffu);
                dst_row_ptr[b].qs_lo[byte_base + 1] = uint8_t((packed >> 8) & 0xffu);
                dst_row_ptr[b].qs_lo[byte_base + 2] = uint8_t((packed >> 16) & 0xffu);
            }
        }
        __syncthreads();

        GGML_UNUSED(ne10);
        GGML_UNUSED(ne11);
        GGML_UNUSED(ne12);
        GGML_UNUSED(ne13);
    }
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
    float * s_tmp = s_row + QK_K;
    float * s_reduce = s_tmp + QK_K;
    uint8_t * s_idx = (uint8_t *) (s_reduce + blockDim.x);

    const int64_t nb = nc / QK_K;
    const float scale_up = sqrtf((float) TURBOQ_KV_DIM);
    const float scale_down = 1.0f / scale_up;

    // Load all 256 elements and compute per-block norms
    for (int64_t block_idx = 0; block_idx < nb; ++block_idx) {
        const int64_t base = block_idx * QK_K;
        float norm_sq = 0.0f;
        for (int64_t i = tid; i < QK_K; i += blockDim.x) {
            const float v = src0_row[base + i];
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

        // Normalize
        for (int64_t i = tid; i < QK_K; i += blockDim.x) {
            s_row[i] *= inv_norm;
        }
        __syncthreads();

        // Blockwise 128x128 forward rotation: Q @ x for each half
        for (int h = 0; h < 2; ++h) {
            const int col = tid % TURBOQ_KV_DIM;
            float sum = 0.0f;
            const float * q_col = Q + col * TURBOQ_KV_DIM;
            const float * x_half = s_row + h * TURBOQ_KV_DIM;
            for (int j = 0; j < TURBOQ_KV_DIM; ++j) {
                sum += q_col[j] * x_half[j];
            }
            const int out_idx = h * TURBOQ_KV_DIM + col;
            s_idx[out_idx] = quantize_tbq2_scalar(sum * scale_up);
            s_tmp[out_idx] = tbq2_codebook_value(s_idx[out_idx]) * scale_down;
        }
        __syncthreads();

        // Blockwise 128x128 MSE residual: Q^T @ s_tmp - x
        for (int h = 0; h < 2; ++h) {
            const int col = tid % TURBOQ_KV_DIM;
            float mse_sum = 0.0f;
            for (int j = 0; j < TURBOQ_KV_DIM; ++j) {
                mse_sum += Q[j*TURBOQ_KV_DIM + col] * s_tmp[h * TURBOQ_KV_DIM + j];
            }
            const int out_idx = h * TURBOQ_KV_DIM + col;
            const float residual = s_row[out_idx] - mse_sum;
            s_row[out_idx] = residual;
        }
        __syncthreads();

        // Compute gamma (residual norm) - all threads participate
        {
            float gamma_sq = 0.0f;
            for (int64_t i = tid; i < QK_K; i += blockDim.x) {
                gamma_sq += s_row[i] * s_row[i];
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
        }

        const float gamma = s_reduce[0];

        // Blockwise 128x128 QJL projection: S @ residual
        for (int h = 0; h < 2; ++h) {
            const int col = tid % TURBOQ_KV_DIM;
            float qjl_sum = 0.0f;
            const float * s_col = S + col * TURBOQ_KV_DIM;
            const float * res_half = s_row + h * TURBOQ_KV_DIM;
            for (int j = 0; j < TURBOQ_KV_DIM; ++j) {
                qjl_sum += s_col[j] * res_half[j];
            }
            const int out_idx = h * TURBOQ_KV_DIM + col;
            s_tmp[out_idx] = qjl_sum;
        }
        __syncthreads();

        // Store norm/gamma (block 0 only)
        if (tid == 0) {
            if (block_idx == 0) {
                dst_row_ptr[block_idx].d = __float2half(norm);
                dst_row_ptr[block_idx].gamma = __float2half(gamma);
            } else {
                dst_row_ptr[block_idx].d = __float2half(0.0f);
                dst_row_ptr[block_idx].gamma = __float2half(0.0f);
            }
        }
        __syncthreads();

        // Pack 2-bit indices (96 bytes for 256 elements)
        constexpr int PACK_GROUPS = QK_K / 4;
        for (int64_t group_idx = tid; group_idx < PACK_GROUPS; group_idx += blockDim.x) {
            const int64_t byte_in_block = group_idx;
            const int64_t base = group_idx * 4;
            uint8_t packed = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                packed |= (s_idx[base + j] & 0x3u) << (j * 2);
            }
            dst_row_ptr[block_idx].qs[byte_in_block] = packed;
        }
        __syncthreads();

        // Pack signs (32 bytes for 256 elements)
        constexpr int SIGN_BYTES = QK_K / 8;
        for (int64_t byte_idx = tid; byte_idx < SIGN_BYTES; byte_idx += blockDim.x) {
            const int64_t base = byte_idx * 8;
            uint8_t packed = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (s_tmp[base + j] >= 0.0f) {
                    packed |= 1u << j;
                }
            }
            dst_row_ptr[block_idx].signs[byte_idx] = packed;
        }
        __syncthreads();
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
    float * s_tmp = s_row + QK_K;
    float * s_reduce = s_tmp + QK_K;
    uint8_t * s_idx = (uint8_t *) (s_reduce + blockDim.x);

    const int64_t nb = nc / QK_K;
    const float scale_up = sqrtf((float) TURBOQ_KV_DIM);
    const float scale_down = 1.0f / scale_up;

    // Load all 256 elements and compute per-block norms
    for (int64_t block_idx = 0; block_idx < nb; ++block_idx) {
        const int64_t base = block_idx * QK_K;
        float norm_sq = 0.0f;
        for (int64_t i = tid; i < QK_K; i += blockDim.x) {
            const float v = src0_row[base + i];
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

        // Normalize
        for (int64_t i = tid; i < QK_K; i += blockDim.x) {
            s_row[i] *= inv_norm;
        }
        __syncthreads();

        // Blockwise 128x128 forward rotation: Q @ x for each half
        for (int h = 0; h < 2; ++h) {
            const int col = tid % TURBOQ_KV_DIM;
            float sum = 0.0f;
            const float * q_col = Q + col * TURBOQ_KV_DIM;
            const float * x_half = s_row + h * TURBOQ_KV_DIM;
            for (int j = 0; j < TURBOQ_KV_DIM; ++j) {
                sum += q_col[j] * x_half[j];
            }
            const int out_idx = h * TURBOQ_KV_DIM + col;
            s_idx[out_idx] = quantize_tbq3_scalar(sum * scale_up);
            s_tmp[out_idx] = tbq3_codebook_value(s_idx[out_idx]) * scale_down;
        }
        __syncthreads();

        // Blockwise 128x128 MSE residual: Q^T @ s_tmp - x
        for (int h = 0; h < 2; ++h) {
            const int col = tid % TURBOQ_KV_DIM;
            float mse_sum = 0.0f;
            for (int j = 0; j < TURBOQ_KV_DIM; ++j) {
                mse_sum += Q[j*TURBOQ_KV_DIM + col] * s_tmp[h * TURBOQ_KV_DIM + j];
            }
            const int out_idx = h * TURBOQ_KV_DIM + col;
            const float residual = s_row[out_idx] - mse_sum;
            s_row[out_idx] = residual;
        }
        __syncthreads();

        // Compute gamma (residual norm) - all threads participate
        {
            float gamma_sq = 0.0f;
            for (int64_t i = tid; i < QK_K; i += blockDim.x) {
                gamma_sq += s_row[i] * s_row[i];
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
        }

        const float gamma = s_reduce[0];

        // Blockwise 128x128 QJL projection: S @ residual
        for (int h = 0; h < 2; ++h) {
            const int col = tid % TURBOQ_KV_DIM;
            float qjl_sum = 0.0f;
            const float * s_col = S + col * TURBOQ_KV_DIM;
            const float * res_half = s_row + h * TURBOQ_KV_DIM;
            for (int j = 0; j < TURBOQ_KV_DIM; ++j) {
                qjl_sum += s_col[j] * res_half[j];
            }
            const int out_idx = h * TURBOQ_KV_DIM + col;
            s_tmp[out_idx] = qjl_sum;
        }
        __syncthreads();

        // Store norm/gamma (block 0 only)
        if (tid == 0) {
            if (block_idx == 0) {
                dst_row_ptr[block_idx].d = __float2half(norm);
                dst_row_ptr[block_idx].gamma = __float2half(gamma);
            } else {
                dst_row_ptr[block_idx].d = __float2half(0.0f);
                dst_row_ptr[block_idx].gamma = __float2half(0.0f);
            }
        }
        __syncthreads();

        // Pack 3-bit indices (96 bytes for 256 elements) - 8 groups of 8 elements = 24 bytes per block
        constexpr int TBQ4_GROUP = 8;
        constexpr int TBQ4_GROUPS_PER_BLOCK = QK_K / TBQ4_GROUP;
        for (int64_t group_idx = tid; group_idx < TBQ4_GROUPS_PER_BLOCK; group_idx += blockDim.x) {
            const int64_t base = group_idx * TBQ4_GROUP;

            uint32_t bits = 0;
            #pragma unroll
            for (int j = 0; j < TBQ4_GROUP; ++j) {
                bits |= uint32_t(s_idx[base + j] & 0x7u) << (j * 3);
            }

            const int64_t byte_offset = group_idx * 3;
            dst_row_ptr[block_idx].qs[byte_offset + 0] = uint8_t(bits & 0xffu);
            dst_row_ptr[block_idx].qs[byte_offset + 1] = uint8_t((bits >> 8) & 0xffu);
            dst_row_ptr[block_idx].qs[byte_offset + 2] = uint8_t((bits >> 16) & 0xffu);
        }
        __syncthreads();

        // Pack signs (32 bytes for 256 elements)
        constexpr int SIGN_BYTES = QK_K / 8;
        for (int64_t byte_idx = tid; byte_idx < SIGN_BYTES; byte_idx += blockDim.x) {
            const int64_t base = byte_idx * 8;
            uint8_t packed = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (s_tmp[base + j] >= 0.0f) {
                    packed |= 1u << j;
                }
            }
            dst_row_ptr[block_idx].signs[byte_idx] = packed;
        }
        __syncthreads();
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// TBQP34_0: mixed Q_prod 3.625-bit - 64 regular channels (2-bit) + 64 outlier channels (3-bit) per 128-wide half
// Plus 1-bit QJL signs for all 256 elements
template <typename idx_t>
static __global__ void k_set_rows_tbqp34(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_tbqp34_0 * __restrict__ dst,
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
    block_tbqp34_0 * dst_row_ptr = (block_tbqp34_0 *) ((char *) dst + dst_row*s1 + i02*s2 + i03*s3);

    extern __shared__ unsigned char smem[];
    float * s_row = (float *) smem;
    float * s_tmp = s_row + QK_K;
    float * s_reduce = s_tmp + QK_K;
    uint8_t * s_idx_lo = (uint8_t *) (s_reduce + blockDim.x);
    uint8_t * s_idx_hi = s_idx_lo + QK_K;

    const int64_t nb = nc / QK_K;
    const float scale_up = sqrtf((float) TURBOQ_KV_DIM);
    const float scale_down = 1.0f / scale_up;

    for (int64_t block_idx = 0; block_idx < nb; ++block_idx) {
        const int64_t base = block_idx * QK_K;
        float norm_sq = 0.0f;
        for (int64_t i = tid; i < QK_K; i += blockDim.x) {
            const float v = src0_row[base + i];
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

        // Normalize
        for (int64_t i = tid; i < QK_K; i += blockDim.x) {
            s_row[i] *= inv_norm;
        }
        __syncthreads();

        const int half = tid / TURBOQ_KV_DIM;
        const int in_half = tid % TURBOQ_KV_DIM;  // 0-127

        // Blockwise 128x128 forward rotation: Q @ x for each half with mixed precision
        // Regular (0-63): 2-bit, Outlier (64-127): 3-bit
        {
            const int col = in_half;
            float sum = 0.0f;
            const float * q_col = Q + col * TURBOQ_KV_DIM;
            const float * x_half = s_row + half * TURBOQ_KV_DIM;
            for (int j = 0; j < TURBOQ_KV_DIM; ++j) {
                sum += q_col[j] * x_half[j];
            }
            const int out_idx = half * TURBOQ_KV_DIM + col;
            if (in_half < 64) {
                s_idx_lo[out_idx] = quantize_tbq2_scalar(sum * scale_up);
                s_tmp[out_idx] = tbq2_codebook_value(s_idx_lo[out_idx]) * scale_down;
            } else {
                s_idx_hi[out_idx] = quantize_tbq3_scalar(sum * scale_up);
                s_tmp[out_idx] = tbq3_codebook_value(s_idx_hi[out_idx]) * scale_down;
            }
        }
        __syncthreads();

        // Blockwise 128x128 MSE residual: Q^T @ s_tmp - x
        for (int h = 0; h < 2; ++h) {
            const int col = tid % TURBOQ_KV_DIM;
            float mse_sum = 0.0f;
            for (int j = 0; j < TURBOQ_KV_DIM; ++j) {
                mse_sum += Q[j*TURBOQ_KV_DIM + col] * s_tmp[h * TURBOQ_KV_DIM + j];
            }
            const int out_idx = h * TURBOQ_KV_DIM + col;
            const float residual = s_row[out_idx] - mse_sum;
            s_row[out_idx] = residual;
        }
        __syncthreads();

        // Compute gamma (residual norm) - all threads participate
        {
            float gamma_sq = 0.0f;
            for (int64_t i = tid; i < QK_K; i += blockDim.x) {
                gamma_sq += s_row[i] * s_row[i];
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
        }

        const float gamma = s_reduce[0];

        // Blockwise 128x128 QJL projection: S @ residual
        for (int h = 0; h < 2; ++h) {
            const int col = tid % TURBOQ_KV_DIM;
            float qjl_sum = 0.0f;
            const float * s_col = S + col * TURBOQ_KV_DIM;
            const float * res_half = s_row + h * TURBOQ_KV_DIM;
            for (int j = 0; j < TURBOQ_KV_DIM; ++j) {
                qjl_sum += s_col[j] * res_half[j];
            }
            const int out_idx = h * TURBOQ_KV_DIM + col;
            s_tmp[out_idx] = qjl_sum;
        }
        __syncthreads();

        // Store norm/gamma (block 0 only)
        if (tid == 0) {
            if (block_idx == 0) {
                dst_row_ptr[block_idx].d = __float2half(norm);
                dst_row_ptr[block_idx].gamma = __float2half(gamma);
            } else {
                dst_row_ptr[block_idx].d = __float2half(0.0f);
                dst_row_ptr[block_idx].gamma = __float2half(0.0f);
            }
        }
        __syncthreads();

        // Pack 2-bit indices for regular channels into qs_lo (32 bytes for 128 values)
        // Layout: half0 bytes 0-15, half1 bytes 16-31
        if (in_half < 64) {
            constexpr int TBQ2_GROUP = 4;
            constexpr int TBQ2_GROUPS_PER_BLOCK = QK_K / TBQ2_GROUP;
            for (int64_t group_idx = tid; group_idx < TBQ2_GROUPS_PER_BLOCK; group_idx += blockDim.x) {
                const int h = group_idx / 32;  // 0 or 1 for half
                const int g = group_idx % 32;  // group within half
                const int64_t base = h * 32 * TBQ2_GROUP + g * TBQ2_GROUP;

                uint32_t bits = 0;
                #pragma unroll
                for (int j = 0; j < TBQ2_GROUP; ++j) {
                    bits |= uint32_t(s_idx_lo[base + j] & 0x3u) << (j * 2);
                }

                const int lo_half_off = h * 16;
                const int64_t byte_offset = lo_half_off + g * 2;
                dst_row_ptr[block_idx].qs_lo[byte_offset + 0] = uint8_t(bits & 0xffu);
                dst_row_ptr[block_idx].qs_lo[byte_offset + 1] = uint8_t((bits >> 8) & 0xffu);
            }
        }
        __syncthreads();

        // Pack 3-bit indices for outlier channels into qs_hi (48 bytes for 128 values)
        // Layout: half0 bytes 0-23, half1 bytes 24-47
        if (in_half >= 64) {
            constexpr int TBQ3_GROUP = 8;
            constexpr int TBQ3_GROUPS_PER_BLOCK = QK_K / TBQ3_GROUP;
            for (int64_t group_idx = tid; group_idx < TBQ3_GROUPS_PER_BLOCK; group_idx += blockDim.x) {
                const int h = group_idx / 16;  // 0 or 1 for half
                const int g = group_idx % 16;  // group within half
                const int64_t base = h * 16 * TBQ3_GROUP + g * TBQ3_GROUP;

                uint32_t bits = 0;
                #pragma unroll
                for (int j = 0; j < TBQ3_GROUP; ++j) {
                    bits |= uint32_t(s_idx_hi[base + j] & 0x7u) << (j * 3);
                }

                const int hi_half_off = h * 24;
                const int64_t byte_offset = hi_half_off + g * 3;
                dst_row_ptr[block_idx].qs_hi[byte_offset + 0] = uint8_t(bits & 0xffu);
                dst_row_ptr[block_idx].qs_hi[byte_offset + 1] = uint8_t((bits >> 8) & 0xffu);
                dst_row_ptr[block_idx].qs_hi[byte_offset + 2] = uint8_t((bits >> 16) & 0xffu);
            }
        }
        __syncthreads();

        // Pack signs (32 bytes for 256 elements)
        constexpr int SIGN_BYTES = QK_K / 8;
        for (int64_t byte_idx = tid; byte_idx < SIGN_BYTES; byte_idx += blockDim.x) {
            const int64_t base = byte_idx * 8;
            uint8_t packed = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (s_tmp[base + j] >= 0.0f) {
                    packed |= 1u << j;
                }
            }
            dst_row_ptr[block_idx].signs[byte_idx] = packed;
        }
        __syncthreads();
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
        const float * d_Q = tbq_get_rotation_device(TURBOQ_KV_DIM, seed, stream);

        const size_t shared_bytes = size_t(QK_K) * sizeof(float) + size_t(CUDA_SET_ROWS_BLOCK_SIZE) * sizeof(float) + size_t(QK_K) * sizeof(uint8_t);
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
        const float * d_Q = tbq_get_rotation_device(TURBOQ_KV_DIM, seed, stream);

        const size_t shared_bytes = size_t(QK_K) * sizeof(float) + size_t(CUDA_SET_ROWS_BLOCK_SIZE) * sizeof(float) + size_t(QK_K) * sizeof(uint8_t);
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
static void set_rows_cuda_tbq34(
        const float * src0_d, const idx_t * src1_d, block_tbq34_0 * dst_d,
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
        const float * d_Q = tbq_get_rotation_device(TURBOQ_KV_DIM, seed, stream);

        const size_t shared_bytes =
            size_t(QK_K) * sizeof(float) +
            size_t(CUDA_SET_ROWS_BLOCK_SIZE) * sizeof(float) +
            2 * size_t(QK_K) * sizeof(uint8_t);
        GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_tbq34<<<ne_rows, CUDA_SET_ROWS_BLOCK_SIZE, shared_bytes, stream>>>(
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
        const float * d_Q = tbq_get_rotation_device(TURBOQ_KV_DIM, seed, stream);
        const float * d_S = tbq_get_projection_device(TURBOQ_KV_DIM, seed, stream);

        const size_t shared_bytes =
            2 * size_t(QK_K) * sizeof(float) +
            size_t(CUDA_SET_ROWS_BLOCK_SIZE) * sizeof(float) +
            size_t(QK_K) * sizeof(uint8_t);
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
        const float * d_Q = tbq_get_rotation_device(TURBOQ_KV_DIM, seed, stream);
        const float * d_S = tbq_get_projection_device(TURBOQ_KV_DIM, seed, stream);

        const size_t shared_bytes =
            2 * size_t(QK_K) * sizeof(float) +
            size_t(CUDA_SET_ROWS_BLOCK_SIZE) * sizeof(float) +
            size_t(QK_K) * sizeof(uint8_t);
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

template <typename idx_t>
static void set_rows_cuda_tbqp34(
        const float * src0_d, const idx_t * src1_d, block_tbqp34_0 * dst_d,
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
        const float * d_Q = tbq_get_rotation_device(TURBOQ_KV_DIM, seed, stream);
        const float * d_S = tbq_get_projection_device(TURBOQ_KV_DIM, seed, stream);

        const size_t shared_bytes =
            2 * size_t(QK_K) * sizeof(float) +
            size_t(CUDA_SET_ROWS_BLOCK_SIZE) * sizeof(float) +
            2 * size_t(QK_K) * sizeof(uint8_t);
        GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_tbqp34<<<ne_rows, CUDA_SET_ROWS_BLOCK_SIZE, shared_bytes, stream>>>(
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
    } else if (dst->type == GGML_TYPE_TBQ34_0) {
        set_rows_cuda_tbq34(
            src0_d, src1_d, (block_tbq34_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TBQP34_0) {
        set_rows_cuda_tbqp34(
            src0_d, src1_d, (block_tbqp34_0*)dst->data,
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
