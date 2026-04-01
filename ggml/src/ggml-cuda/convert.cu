#include "convert.cuh"
#include "dequantize.cuh"
#include "ggml-turboq.h"

#include <cstdint>

#define CUDA_Q8_0_NE_ALIGN 2048

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
        const float * (*get_host_matrix)(int64_t, uint64_t)) {
    GGML_ASSERT(d > 0);

    const int device = ggml_cuda_get_device();
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);

    const uint64_t seed = turboq_seed_from_row(0);
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

static const float * tbq_get_rotation_device(int d) {
    return turboq_get_matrix_device(g_turboq_rotation_cache, d, turboq_get_rotation);
}

static const float * tbq_get_projection_device(int d) {
    return turboq_get_matrix_device(g_turboq_projection_cache, d, turboq_get_projection);
}

static __device__ __forceinline__ float tbq2_codebook_value(uint8_t idx) {
    switch (idx) {
        case 0: return -1.5104f;
        case 1: return -0.4528f;
        case 2: return  0.4528f;
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

static __device__ __forceinline__ float tbq4_codebook_value(uint8_t idx) {
    switch (idx) {
        case 0:  return -2.7326f;
        case 1:  return -2.0690f;
        case 2:  return -1.6180f;
        case 3:  return -1.2562f;
        case 4:  return -0.9424f;
        case 5:  return -0.6568f;
        case 6:  return -0.3881f;
        case 7:  return -0.1284f;
        case 8:  return  0.1284f;
        case 9:  return  0.3881f;
        case 10: return  0.6568f;
        case 11: return  0.9424f;
        case 12: return  1.2562f;
        case 13: return  1.6180f;
        case 14: return  2.0690f;
        default: return  2.7326f;
    }
}

template<typename dst_t>
static __global__ void dequantize_row_tbq3_nc(
        const block_tbq3_0 * __restrict__ vx,
        dst_t * __restrict__ y,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne0203,
        const uint3 ne02,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const float * __restrict__ Q) {
    const int64_t row_id = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_id >= ne01 * ne0203) {
        return;
    }

    const uint2 dm = fast_div_modulo((uint32_t) (row_id / ne01), ne02);
    const int64_t i01 = row_id % ne01;
    const int64_t i02 = dm.y;
    const int64_t i03 = dm.x;

    const block_tbq3_0 * row = vx + i03*s03 + i02*s02 + i01*s01;
    dst_t * out = y + row_id * ne00;

    extern __shared__ float smem[];
    float * s_rot = smem;
    const float scale_down = 1.0f / sqrtf((float) QK_K);
    const int64_t nb = ne00 / QK_K;

    for (int64_t block_idx = 0; block_idx < nb; ++block_idx) {
        const int64_t base = block_idx * QK_K;
        const int64_t in_block = tid;
        const int64_t group = in_block / 8;
        const int64_t shift = (in_block % 8) * 3;
        const uint8_t * qs = row[block_idx].qs + group * 3;
        const uint32_t bits = uint32_t(qs[0]) | (uint32_t(qs[1]) << 8) | (uint32_t(qs[2]) << 16);
        const uint8_t idx = (bits >> shift) & 0x7u;

        s_rot[tid] = tbq3_codebook_value(idx) * scale_down;
        __syncthreads();

        const int half = tid / TURBOQ_KV_DIM;
        const int col  = tid % TURBOQ_KV_DIM;
        float sum = 0.0f;
        for (int j = 0; j < TURBOQ_KV_DIM; ++j) {
            sum += Q[j*TURBOQ_KV_DIM + col] * s_rot[half * TURBOQ_KV_DIM + j];
        }

        const float norm = __half2float(row[block_idx].d);
        out[base + tid] = ggml_cuda_cast<dst_t>(sum * norm);
        __syncthreads();
    }
}

template<typename dst_t>
static __global__ void dequantize_row_tbq4_nc(
        const block_tbq4_0 * __restrict__ vx,
        dst_t * __restrict__ y,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne0203,
        const uint3 ne02,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const float * __restrict__ Q) {
    const int64_t row_id = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_id >= ne01 * ne0203) {
        return;
    }

    const uint2 dm = fast_div_modulo((uint32_t) (row_id / ne01), ne02);
    const int64_t i01 = row_id % ne01;
    const int64_t i02 = dm.y;
    const int64_t i03 = dm.x;

    const block_tbq4_0 * row = vx + i03*s03 + i02*s02 + i01*s01;
    dst_t * out = y + row_id * ne00;

    extern __shared__ float smem[];
    float * s_rot = smem;
    const float scale_down = 1.0f / sqrtf((float) QK_K);
    const int64_t nb = ne00 / QK_K;

    for (int64_t block_idx = 0; block_idx < nb; ++block_idx) {
        const int64_t base = block_idx * QK_K;
        const uint8_t packed = row[block_idx].qs[tid / 2];
        const uint8_t idx = (tid & 1) == 0 ? (packed & 0x0fu) : ((packed >> 4) & 0x0fu);

        s_rot[tid] = tbq4_codebook_value(idx) * scale_down;
        __syncthreads();

        const int half = tid / TURBOQ_KV_DIM;
        const int col  = tid % TURBOQ_KV_DIM;
        float sum = 0.0f;
        for (int j = 0; j < TURBOQ_KV_DIM; ++j) {
            sum += Q[j*TURBOQ_KV_DIM + col] * s_rot[half * TURBOQ_KV_DIM + j];
        }

        const float norm = __half2float(row[block_idx].d);
        out[base + tid] = ggml_cuda_cast<dst_t>(sum * norm);
        __syncthreads();
    }
}

template<typename dst_t>
static __global__ void dequantize_row_tbqp3_nc(
        const block_tbqp3_0 * __restrict__ vx,
        dst_t * __restrict__ y,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne0203,
        const uint3 ne02,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const float * __restrict__ Q,
        const float * __restrict__ S) {
    const int64_t row_id = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_id >= ne01 * ne0203) {
        return;
    }

    const uint2 dm = fast_div_modulo((uint32_t) (row_id / ne01), ne02);
    const int64_t i01 = row_id % ne01;
    const int64_t i02 = dm.y;
    const int64_t i03 = dm.x;

    const block_tbqp3_0 * row = vx + i03*s03 + i02*s02 + i01*s01;
    dst_t * out = y + row_id * ne00;

    extern __shared__ float smem[];
    float * s_mse_rot = smem;
    float * s_signs = s_mse_rot + QK_K;

    const float scale_down = 1.0f / sqrtf((float) QK_K);
    const int64_t nb = ne00 / QK_K;

    for (int64_t block_idx = 0; block_idx < nb; ++block_idx) {
        const int64_t base = block_idx * QK_K;
        const float norm = __half2float(row[block_idx].d);
        const float gamma = __half2float(row[block_idx].gamma);
        const float qjl_f = sqrtf((float) M_PI / 2.0f) * gamma / (float) QK_K;

        // Load MSE codebook values and signs for this block
        const int64_t in_block = tid;
        const uint8_t idx = (row[block_idx].qs[in_block / 4] >> ((in_block % 4) * 2)) & 0x3u;
        s_mse_rot[tid] = tbq2_codebook_value(idx) * scale_down;
        s_signs[tid] = ((row[block_idx].signs[tid / 8] >> (tid % 8)) & 1u) ? 1.0f : -1.0f;
        __syncthreads();

        // Full QK_K x QK_K matvecs: Q^T @ mse_rot and S @ signs
        float mse_sum = 0.0f;
        float qjl_sum = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            mse_sum += Q[tid*QK_K + j] * s_mse_rot[j];
            qjl_sum += S[tid*QK_K + j] * s_signs[j];
        }
        out[base + tid] = ggml_cuda_cast<dst_t>(norm * (mse_sum + qjl_f * qjl_sum));
        __syncthreads();
    }
}

template<typename dst_t>
static __global__ void dequantize_row_tbqp3_mse_nc(
        const block_tbqp3_0 * __restrict__ vx,
        dst_t * __restrict__ y,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne0203,
        const uint3 ne02,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const float * __restrict__ Q) {
    const int64_t row_id = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_id >= ne01 * ne0203) {
        return;
    }

    const uint2 dm = fast_div_modulo((uint32_t) (row_id / ne01), ne02);
    const int64_t i01 = row_id % ne01;
    const int64_t i02 = dm.y;
    const int64_t i03 = dm.x;

    const block_tbqp3_0 * row = vx + i03*s03 + i02*s02 + i01*s01;
    dst_t * out = y + row_id * ne00;

    extern __shared__ float s_mse_rot[];
    const float scale_down = 1.0f / sqrtf((float) QK_K);
    const int64_t nb = ne00 / QK_K;

    for (int64_t block_idx = 0; block_idx < nb; ++block_idx) {
        const int64_t base = block_idx * QK_K;
        const float norm = __half2float(row[block_idx].d);

        const int64_t in_block = tid;
        const uint8_t idx = (row[block_idx].qs[in_block / 4] >> ((in_block % 4) * 2)) & 0x3u;
        s_mse_rot[tid] = tbq2_codebook_value(idx) * scale_down;
        __syncthreads();

        float mse_sum = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            mse_sum += Q[tid*QK_K + j] * s_mse_rot[j];
        }
        out[base + tid] = ggml_cuda_cast<dst_t>(norm * mse_sum);
        __syncthreads();
    }
}

template<typename dst_t>
static __global__ void dequantize_row_tbqp4_nc(
        const block_tbqp4_0 * __restrict__ vx,
        dst_t * __restrict__ y,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne0203,
        const uint3 ne02,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const float * __restrict__ Q,
        const float * __restrict__ S) {
    const int64_t row_id = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_id >= ne01 * ne0203) {
        return;
    }

    const uint2 dm = fast_div_modulo((uint32_t) (row_id / ne01), ne02);
    const int64_t i01 = row_id % ne01;
    const int64_t i02 = dm.y;
    const int64_t i03 = dm.x;

    const block_tbqp4_0 * row = vx + i03*s03 + i02*s02 + i01*s01;
    dst_t * out = y + row_id * ne00;

    extern __shared__ float smem[];
    float * s_mse_rot = smem;
    float * s_signs = s_mse_rot + QK_K;

    const float scale_down = 1.0f / sqrtf((float) QK_K);
    const int64_t nb = ne00 / QK_K;

    for (int64_t block_idx = 0; block_idx < nb; ++block_idx) {
        const int64_t base = block_idx * QK_K;
        const float norm = __half2float(row[block_idx].d);
        const float gamma = __half2float(row[block_idx].gamma);
        const float qjl_f = sqrtf((float) M_PI / 2.0f) * gamma / (float) QK_K;

        // Load MSE codebook values and signs for this block
        const int64_t in_block = tid;
        const int64_t group = in_block / 8;
        const int64_t shift = (in_block % 8) * 3;
        const uint8_t * qs = row[block_idx].qs + group * 3;
        const uint32_t bits = uint32_t(qs[0]) | (uint32_t(qs[1]) << 8) | (uint32_t(qs[2]) << 16);
        const uint8_t idx = (bits >> shift) & 0x7u;
        s_mse_rot[tid] = tbq3_codebook_value(idx) * scale_down;
        s_signs[tid] = ((row[block_idx].signs[tid / 8] >> (tid % 8)) & 1u) ? 1.0f : -1.0f;
        __syncthreads();

        // Full QK_K x QK_K matvecs: Q^T @ mse_rot and S @ signs
        float mse_sum = 0.0f;
        float qjl_sum = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            mse_sum += Q[tid*QK_K + j] * s_mse_rot[j];
            qjl_sum += S[tid*QK_K + j] * s_signs[j];
        }
        out[base + tid] = ggml_cuda_cast<dst_t>(norm * (mse_sum + qjl_f * qjl_sum));
        __syncthreads();
    }
}

template<typename dst_t>
static __global__ void dequantize_row_tbqp4_mse_nc(
        const block_tbqp4_0 * __restrict__ vx,
        dst_t * __restrict__ y,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne0203,
        const uint3 ne02,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const float * __restrict__ Q) {
    const int64_t row_id = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_id >= ne01 * ne0203) {
        return;
    }

    const uint2 dm = fast_div_modulo((uint32_t) (row_id / ne01), ne02);
    const int64_t i01 = row_id % ne01;
    const int64_t i02 = dm.y;
    const int64_t i03 = dm.x;

    const block_tbqp4_0 * row = vx + i03*s03 + i02*s02 + i01*s01;
    dst_t * out = y + row_id * ne00;

    extern __shared__ float s_mse_rot[];
    const float scale_down = 1.0f / sqrtf((float) QK_K);
    const int64_t nb = ne00 / QK_K;

    for (int64_t block_idx = 0; block_idx < nb; ++block_idx) {
        const int64_t base = block_idx * QK_K;
        const float norm = __half2float(row[block_idx].d);

        const int64_t in_block = tid;
        const int64_t group = in_block / 8;
        const int64_t shift = (in_block % 8) * 3;
        const uint8_t * qs = row[block_idx].qs + group * 3;
        const uint32_t bits = uint32_t(qs[0]) | (uint32_t(qs[1]) << 8) | (uint32_t(qs[2]) << 16);
        const uint8_t idx = (bits >> shift) & 0x7u;
        s_mse_rot[tid] = tbq3_codebook_value(idx) * scale_down;
        __syncthreads();

        float mse_sum = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            mse_sum += Q[tid*QK_K + j] * s_mse_rot[j];
        }
        out[base + tid] = ggml_cuda_cast<dst_t>(norm * mse_sum);
        __syncthreads();
    }
}

template<typename dst_t>
static void dequantize_row_tbq3_nc_cuda(
        const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t ne0203 = ne02 * ne03;
    const int64_t nrows = ne01 * ne0203;
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    const float * d_Q = tbq_get_rotation_device(TURBOQ_KV_DIM);
    const size_t shared_bytes = size_t(QK_K) * sizeof(float);

    GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);
    GGML_ASSERT(nrows < UINT_MAX);

    dequantize_row_tbq3_nc<<<nrows, CUDA_DEQUANTIZE_BLOCK_SIZE, shared_bytes, stream>>>(
            (const block_tbq3_0 *) vx, y, ne00, ne01, ne0203, ne02_fd, s01, s02, s03, d_Q);
}

template<typename dst_t>
static void dequantize_row_tbq4_nc_cuda(
        const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t ne0203 = ne02 * ne03;
    const int64_t nrows = ne01 * ne0203;
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    const float * d_Q = tbq_get_rotation_device(TURBOQ_KV_DIM);
    const size_t shared_bytes = size_t(QK_K) * sizeof(float);

    GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);
    GGML_ASSERT(nrows < UINT_MAX);

    dequantize_row_tbq4_nc<<<nrows, CUDA_DEQUANTIZE_BLOCK_SIZE, shared_bytes, stream>>>(
            (const block_tbq4_0 *) vx, y, ne00, ne01, ne0203, ne02_fd, s01, s02, s03, d_Q);
}

// TBQ34_0: mixed 3.5-bit - 64 regular channels (3-bit) + 64 outlier channels (4-bit) per 128-wide half
template<typename dst_t>
static __global__ void dequantize_row_tbq34_nc(
        const block_tbq34_0 * __restrict__ vx,
        dst_t * __restrict__ y,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne0203,
        const uint3 ne02,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const float * __restrict__ Q) {
    const int64_t row_id = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_id >= ne01 * ne0203) {
        return;
    }

    const uint2 dm = fast_div_modulo((uint32_t) (row_id / ne01), ne02);
    const int64_t i01 = row_id % ne01;
    const int64_t i02 = dm.y;
    const int64_t i03 = dm.x;

    const block_tbq34_0 * row = vx + i03*s03 + i02*s02 + i01*s01;
    dst_t * out = y + row_id * ne00;

    extern __shared__ float smem[];
    float * s_rot = smem;
    const float scale_down = 1.0f / sqrtf((float) QK_K);
    const int64_t nb = ne00 / QK_K;

    // Each 256-element block: 2 halves of 128 elements
    // Half 0 (indices 0-63 regular, 64-127 outlier), Half 1 (indices 128-191 regular, 192-255 outlier)
    for (int64_t block_idx = 0; block_idx < nb; ++block_idx) {
        const int64_t base = block_idx * QK_K;
        const int half = tid / TURBOQ_KV_DIM;  // 0 or 1
        const int in_half = tid % TURBOQ_KV_DIM;  // 0-127

        // Dequantize based on position in half:
        // - indices 0-63 (regular): 3-bit from qs_lo
        // - indices 64-127 (outlier): 4-bit from qs_hi
        float val;
        if (in_half < 64) {
            // Regular channel: 3-bit from qs_lo.
            // Each half stores 64 regular values in 24 bytes, packed as 8 values / 3 bytes.
            const int lo_half_off = half * 24;
            const int group = in_half / 8;
            const int bit_shift = (in_half % 8) * 3;
            const uint8_t * qs = row[block_idx].qs_lo + lo_half_off + group * 3;
            const uint32_t bits = uint32_t(qs[0]) | (uint32_t(qs[1]) << 8) | (uint32_t(qs[2]) << 16);
            const uint8_t idx = (bits >> bit_shift) & 0x7u;
            val = tbq3_codebook_value(idx) * scale_down;
        } else {
            // Outlier channel: 4-bit from qs_hi
            // qs_hi layout: half0 bytes 0-31, half1 bytes 32-63 (64 bytes total for 128 4-bit values)
            const int hi_half_off = half * 32;  // byte offset for this half
            const int val_idx = in_half - 64;  // 0-63
            const int byte_idx = hi_half_off + val_idx / 2;
            const uint8_t bits = row[block_idx].qs_hi[byte_idx];
            const uint8_t idx = (val_idx & 1) == 0 ? (bits & 0x0fu) : ((bits >> 4) & 0x0fu);
            val = tbq4_codebook_value(idx) * scale_down;
        }

        // All 256 threads write their dequantized value
        s_rot[tid] = val;
        __syncthreads();

        // Blockwise 128x128 inverse rotation: Q^T @ s_rot
        // Thread i computes output position i using s_rot[half*128 : half*128+127]
        const int col = tid % TURBOQ_KV_DIM;
        float sum = 0.0f;
        for (int j = 0; j < TURBOQ_KV_DIM; ++j) {
            sum += Q[j*TURBOQ_KV_DIM + col] * s_rot[half * TURBOQ_KV_DIM + j];
        }

        const float norm = __half2float(row[block_idx].d);
        out[base + tid] = ggml_cuda_cast<dst_t>(sum * norm);
        __syncthreads();
    }
}

template<typename dst_t>
static void dequantize_row_tbq34_nc_cuda(
        const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t ne0203 = ne02 * ne03;
    const int64_t nrows = ne01 * ne0203;
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    const float * d_Q = tbq_get_rotation_device(TURBOQ_KV_DIM);
    const size_t shared_bytes = size_t(QK_K) * sizeof(float);

    GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);
    GGML_ASSERT(nrows < UINT_MAX);

    dequantize_row_tbq34_nc<<<nrows, QK_K, shared_bytes, stream>>>(
            (const block_tbq34_0 *) vx, y, ne00, ne01, ne0203, ne02_fd, s01, s02, s03, d_Q);
}

template<typename dst_t>
static void dequantize_row_tbqp3_nc_cuda(
        const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t ne0203 = ne02 * ne03;
    const int64_t nrows = ne01 * ne0203;
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    const float * d_Q = tbq_get_rotation_device(QK_K);
    const float * d_S = tbq_get_projection_device(QK_K);
    const size_t shared_bytes = 2 * size_t(QK_K) * sizeof(float);

    GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);
    GGML_ASSERT(nrows < UINT_MAX);

    dequantize_row_tbqp3_nc<<<nrows, QK_K, shared_bytes, stream>>>(
            (const block_tbqp3_0 *) vx, y, ne00, ne01, ne0203, ne02_fd, s01, s02, s03, d_Q, d_S);
}

template<typename dst_t>
static void dequantize_row_tbqp4_nc_cuda(
        const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t ne0203 = ne02 * ne03;
    const int64_t nrows = ne01 * ne0203;
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    const float * d_Q = tbq_get_rotation_device(QK_K);
    const float * d_S = tbq_get_projection_device(QK_K);
    const size_t shared_bytes = 2 * size_t(QK_K) * sizeof(float);

    GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);
    GGML_ASSERT(nrows < UINT_MAX);

    dequantize_row_tbqp4_nc<<<nrows, QK_K, shared_bytes, stream>>>(
            (const block_tbqp4_0 *) vx, y, ne00, ne01, ne0203, ne02_fd, s01, s02, s03, d_Q, d_S);
}

template<typename dst_t>
static void dequantize_row_tbqp3_mse_nc_cuda(
        const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t ne0203 = ne02 * ne03;
    const int64_t nrows = ne01 * ne0203;
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    const float * d_Q = tbq_get_rotation_device(QK_K);
    const size_t shared_bytes = size_t(QK_K) * sizeof(float);

    GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);
    GGML_ASSERT(nrows < UINT_MAX);

    dequantize_row_tbqp3_mse_nc<<<nrows, QK_K, shared_bytes, stream>>>(
            (const block_tbqp3_0 *) vx, y, ne00, ne01, ne0203, ne02_fd, s01, s02, s03, d_Q);
}

template<typename dst_t>
static void dequantize_row_tbqp4_mse_nc_cuda(
        const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t ne0203 = ne02 * ne03;
    const int64_t nrows = ne01 * ne0203;
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    const float * d_Q = tbq_get_rotation_device(QK_K);
    const size_t shared_bytes = size_t(QK_K) * sizeof(float);

    GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);
    GGML_ASSERT(nrows < UINT_MAX);

    dequantize_row_tbqp4_mse_nc<<<nrows, QK_K, shared_bytes, stream>>>(
            (const block_tbqp4_0 *) vx, y, ne00, ne01, ne0203, ne02_fd, s01, s02, s03, d_Q);
}

// TBQP34_0: mixed Q_prod 3.625-bit - 64 regular channels (2-bit) + 64 outlier channels (3-bit) per 128-wide half
// Plus 1-bit QJL signs for all 256 elements
template<typename dst_t>
static __global__ void dequantize_row_tbqp34_nc(
        const block_tbqp34_0 * __restrict__ vx,
        dst_t * __restrict__ y,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne0203,
        const uint3 ne02,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const float * __restrict__ Q,
        const float * __restrict__ S) {
    const int64_t row_id = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_id >= ne01 * ne0203) {
        return;
    }

    const uint2 dm = fast_div_modulo((uint32_t) (row_id / ne01), ne02);
    const int64_t i01 = row_id % ne01;
    const int64_t i02 = dm.y;
    const int64_t i03 = dm.x;

    const block_tbqp34_0 * row = vx + i03*s03 + i02*s02 + i01*s01;
    dst_t * out = y + row_id * ne00;

    extern __shared__ float smem[];
    float * s_mse_rot = smem;
    float * s_signs = s_mse_rot + QK_K;

    const float scale_down = 1.0f / sqrtf((float) QK_K);
    const int64_t nb = ne00 / QK_K;

    // Each 256-element block: 2 halves of 128 elements
    // Half 0 (indices 0-63 regular 2-bit, 64-127 outlier 3-bit), Half 1 (128-191 regular, 192-255 outlier)
    for (int64_t block_idx = 0; block_idx < nb; ++block_idx) {
        const int64_t base = block_idx * QK_K;
        const int half = tid / TURBOQ_KV_DIM;  // 0 or 1
        const int in_half = tid % TURBOQ_KV_DIM;  // 0-127
        const float norm = __half2float(row[block_idx].d);
        const float gamma = __half2float(row[block_idx].gamma);
        const float qjl_f = sqrtf((float) M_PI / 2.0f) * gamma / (float) QK_K;

        // Dequantize MSE values and load signs
        float mse_val;
        if (in_half < 64) {
            // Regular channel: 2-bit from qs_lo
            // qs_lo layout: half0 bytes 0-15, half1 bytes 16-31 (32 bytes total for 128 2-bit values)
            // 2 bits per value, 4 values per byte
            const int lo_half_off = half * 16;  // byte offset for this half
            const int val_idx = in_half;  // 0-63
            const int byte_idx = lo_half_off + val_idx / 4;
            const int bit_shift = (val_idx % 4) * 2;
            const uint8_t bits = row[block_idx].qs_lo[byte_idx];
            const uint8_t idx = (bits >> bit_shift) & 0x3u;
            mse_val = tbq2_codebook_value(idx) * scale_down;
        } else {
            // Outlier channel: 3-bit from qs_hi
            // qs_hi layout: half0 bytes 0-23, half1 bytes 24-47 (48 bytes total for 128 3-bit values)
            // 3 bits per value, 8 values per 3 bytes
            const int hi_half_off = half * 24;  // byte offset for this half
            const int val_idx = in_half - 64;  // 0-63
            const int byte_idx = hi_half_off + (val_idx * 3) / 8;
            const int bit_shift = (val_idx * 3) % 8;
            const uint8_t bits = row[block_idx].qs_hi[byte_idx];
            const uint8_t idx = (bits >> bit_shift) & 0x7u;
            mse_val = tbq3_codebook_value(idx) * scale_down;
        }
        s_mse_rot[tid] = mse_val;
        s_signs[tid] = ((row[block_idx].signs[tid / 8] >> (tid % 8)) & 1u) ? 1.0f : -1.0f;
        __syncthreads();

        // Full QK_K x QK_K matvecs: Q^T @ mse_rot and S @ signs
        float mse_sum = 0.0f;
        float qjl_sum = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            mse_sum += Q[tid*QK_K + j] * s_mse_rot[j];
            qjl_sum += S[tid*QK_K + j] * s_signs[j];
        }
        out[base + tid] = ggml_cuda_cast<dst_t>(norm * (mse_sum + qjl_f * qjl_sum));
        __syncthreads();
    }
}

template<typename dst_t>
static __global__ void dequantize_row_tbqp34_mse_nc(
        const block_tbqp34_0 * __restrict__ vx,
        dst_t * __restrict__ y,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne0203,
        const uint3 ne02,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const float * __restrict__ Q) {
    const int64_t row_id = blockIdx.x;
    const int tid = threadIdx.x;

    if (row_id >= ne01 * ne0203) {
        return;
    }

    const uint2 dm = fast_div_modulo((uint32_t) (row_id / ne01), ne02);
    const int64_t i01 = row_id % ne01;
    const int64_t i02 = dm.y;
    const int64_t i03 = dm.x;

    const block_tbqp34_0 * row = vx + i03*s03 + i02*s02 + i01*s01;
    dst_t * out = y + row_id * ne00;

    extern __shared__ float s_mse_rot[];
    const float scale_down = 1.0f / sqrtf((float) QK_K);
    const int64_t nb = ne00 / QK_K;

    for (int64_t block_idx = 0; block_idx < nb; ++block_idx) {
        const int64_t base = block_idx * QK_K;
        const int half = tid / TURBOQ_KV_DIM;
        const int in_half = tid % TURBOQ_KV_DIM;
        const float norm = __half2float(row[block_idx].d);

        float mse_val;
        if (in_half < 64) {
            const int lo_half_off = half * 16;
            const int val_idx = in_half;
            const int byte_idx = lo_half_off + val_idx / 4;
            const int bit_shift = (val_idx % 4) * 2;
            const uint8_t bits = row[block_idx].qs_lo[byte_idx];
            const uint8_t idx = (bits >> bit_shift) & 0x3u;
            mse_val = tbq2_codebook_value(idx) * scale_down;
        } else {
            const int hi_half_off = half * 24;
            const int val_idx = in_half - 64;
            const int byte_idx = hi_half_off + (val_idx * 3) / 8;
            const int bit_shift = (val_idx * 3) % 8;
            const uint8_t bits = row[block_idx].qs_hi[byte_idx];
            const uint8_t idx = (bits >> bit_shift) & 0x7u;
            mse_val = tbq3_codebook_value(idx) * scale_down;
        }
        s_mse_rot[tid] = mse_val;
        __syncthreads();

        float mse_sum = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            mse_sum += Q[tid*QK_K + j] * s_mse_rot[j];
        }
        out[base + tid] = ggml_cuda_cast<dst_t>(norm * mse_sum);
        __syncthreads();
    }
}

template<typename dst_t>
static void dequantize_row_tbqp34_nc_cuda(
        const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t ne0203 = ne02 * ne03;
    const int64_t nrows = ne01 * ne0203;
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    const float * d_Q = tbq_get_rotation_device(QK_K);
    const float * d_S = tbq_get_projection_device(QK_K);
    const size_t shared_bytes = 2 * size_t(QK_K) * sizeof(float);

    GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);
    GGML_ASSERT(nrows < UINT_MAX);

    dequantize_row_tbqp34_nc<<<nrows, QK_K, shared_bytes, stream>>>(
            (const block_tbqp34_0 *) vx, y, ne00, ne01, ne0203, ne02_fd, s01, s02, s03, d_Q, d_S);
}

template<typename dst_t>
static void dequantize_row_tbqp34_mse_nc_cuda(
        const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t ne0203 = ne02 * ne03;
    const int64_t nrows = ne01 * ne0203;
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    const float * d_Q = tbq_get_rotation_device(QK_K);
    const size_t shared_bytes = size_t(QK_K) * sizeof(float);

    GGML_ASSERT(shared_bytes <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);
    GGML_ASSERT(nrows < UINT_MAX);

    dequantize_row_tbqp34_mse_nc<<<nrows, QK_K, shared_bytes, stream>>>(
            (const block_tbqp34_0 *) vx, y, ne00, ne01, ne0203, ne02_fd, s01, s02, s03, d_Q);
}

} // namespace

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = 2 * (int64_t(blockDim.x)*blockIdx.x + threadIdx.x);

    if (i00 >= ne00) {
        return;
    }

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

            const int64_t ib = ibx0 + i00/qk; // block index
            const int64_t iqs = (i00%qk)/qr; // quant index
            const int64_t iybs = i00 - i00%qk; // y block start index
            const int64_t y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(vx, ib, iqs, v);

            const int64_t iy0 = (i0203*ne01 + i01)*ne00 + iybs + iqs;
            y[iy0 + 0]        = ggml_cuda_cast<dst_t>(v.x);
            y[iy0 + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= k*sizeof(block_q8_0)/QK8_0) {
            break;
        }

        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED_VARS(vx, y, k);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d.x * (q[l] & 0xF) + d.y;
        y[l+16] = d.x * (q[l] >>  4) + d.y;
    }
}

//================================== k-quants

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i = blockIdx.x;
    const block_q3_K * x = (const block_q3_K *) vx;

    const int64_t r = threadIdx.x/4;
    const int64_t tid = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(threadIdx.x%4);
    const int64_t n = tid / 4;
    const int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l +32] = d2 * (q[l] >>  4) - m2;
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_s * x = (const block_iq2_s *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t  * q3 = x[i].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq3_s * x = (const block_iq3_s *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * qs = x[i].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    const float d = (float)x[i].d * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[i].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const float delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_m * x = (const block_iq1_m  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = (float)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq4_nl * x = (const block_iq4_nl *) vx + i*(QK_K/QK4_NL);

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = (float)x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = (float)x[i].d * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

template<typename dst_t>
static __global__ void dequantize_block_mxfp4(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_mxfp4 * x = (const block_mxfp4 *) vx + i*(QK_K/QK_MXFP4);

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_mxfp4[q4[j] & 0xf]*0.5f;
        y[j+16] = d * kvalues_mxfp4[q4[j] >>  4]*0.5f;
    }
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE), (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cont_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    dequantize_block_cuda<qk, qr, dequantize_kernel, dst_t>(vx, y, k, 1, 1, 1, k/qk, k/qk, k/qk, stream);
}

static void dequantize_block_q8_0_f16_cuda(const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
        const bool need_check = false;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    } else {
        const bool need_check = true;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    }
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_1<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_m<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_mxfp4_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_mxfp4<<<nb, 32, 0, stream>>>(vx, y);
}

template <typename dst_t>
static __global__ void dequantize_block_nvfp4(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t i = blockIdx.x;
    const int     tid = threadIdx.x;

    const int64_t base = i * QK_NVFP4;
    if (base >= ne) {
        return;
    }

    const block_nvfp4 * x = (const block_nvfp4 *) vx;
    const block_nvfp4 & xb = x[i];

    const int sub = tid / (QK_NVFP4_SUB / 2);
    const int j = tid % (QK_NVFP4_SUB / 2);

    const float d = ggml_cuda_ue4m3_to_fp32(xb.d[sub]);
    const uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    const int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    const int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}

template <typename dst_t>
static void dequantize_row_nvfp4_cuda(
        const void * vx,
        dst_t * y,
        const int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nb = k / QK_NVFP4;
    dequantize_block_nvfp4<<<nb, 32, 0, stream>>>(vx, y, k);
}
template <typename src_t, typename dst_t>
static __global__ void convert_unary(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i00 >= ne00) {
        return;
    }

    const src_t * x = (const src_t *) vx;

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ix = i03*s03 + i02*s02 + i01*s01 + i00;
            const int64_t iy = (i0203*ne01 + i01)*ne00 + i00;
            y[iy] = ggml_cuda_cast<dst_t>(x[ix]);
        }
    }
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    convert_unary<src_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <typename src_t, typename dst_t>
static void convert_unary_cont_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    convert_unary_cuda<src_t>(vx, y, k, 1, 1, 1, k, k, k, stream);
}

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        default:
            return nullptr;
    }
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (fp16_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
                return dequantize_block_q8_0_f16_cuda;
            }
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float>;
        case GGML_TYPE_TBQ3_0:
            return dequantize_row_tbq3_nc_cuda;
        case GGML_TYPE_TBQ4_0:
            return dequantize_row_tbq4_nc_cuda;
        case GGML_TYPE_TBQ34_0:
            return dequantize_row_tbq34_nc_cuda;
        case GGML_TYPE_TBQP3_0:
            return dequantize_row_tbqp3_nc_cuda;
        case GGML_TYPE_TBQP4_0:
            return dequantize_row_tbqp4_nc_cuda;
        case GGML_TYPE_TBQP34_0:
            return dequantize_row_tbqp34_nc_cuda;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda_tbqp_mse(ggml_type type) {
    switch (type) {
        case GGML_TYPE_TBQP3_0:
            return dequantize_row_tbqp3_mse_nc_cuda;
        case GGML_TYPE_TBQP4_0:
            return dequantize_row_tbqp4_mse_nc_cuda;
        case GGML_TYPE_TBQP34_0:
            return dequantize_row_tbqp34_mse_nc_cuda;
        default:
            return nullptr;
    }
}

to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float, nv_bfloat16>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, float>;
        case GGML_TYPE_TBQ3_0:
            return dequantize_row_tbq3_nc_cuda;
        case GGML_TYPE_TBQ4_0:
            return dequantize_row_tbq4_nc_cuda;
        case GGML_TYPE_TBQ34_0:
            return dequantize_row_tbq34_nc_cuda;
        case GGML_TYPE_TBQP3_0:
            return dequantize_row_tbqp3_nc_cuda;
        case GGML_TYPE_TBQP4_0:
            return dequantize_row_tbqp4_nc_cuda;
        case GGML_TYPE_TBQP34_0:
            return dequantize_row_tbqp34_nc_cuda;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16, float>;
        default:
            return nullptr;
    }
}
