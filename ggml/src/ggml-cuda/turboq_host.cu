// C-callable TurboQuant HIP wrappers.

#include "turboq.cuh"
#include "turboq_host.h"
#include "ggml-turboq.h"

#include <hip/hip_runtime.h>

static constexpr int TURBOQ_D = 128;

namespace {

struct {
    float * d_Q = nullptr;
    float * d_x = nullptr;
    float * d_y = nullptr;
    int d_capacity = 0;
    bool initialized = false;
    hipStream_t stream = hipStream_t(0);
} g_turboq_dev;

static void matvec_cpu(float * y, const float * Q, const float * x, int d) {
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += Q[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}

} // anonymous namespace

void turboq_cuda_init(void) {
    if (g_turboq_dev.initialized) return;

    g_turboq_dev.stream = 0;
    g_turboq_dev.d_capacity = TURBOQ_D;

    CUDA_CHECK(cudaMalloc(&g_turboq_dev.d_Q, TURBOQ_D * TURBOQ_D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_turboq_dev.d_x, TURBOQ_D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_turboq_dev.d_y, TURBOQ_D * sizeof(float)));

    g_turboq_dev.initialized = true;
}

void turboq_cuda_free(void) {
    if (!g_turboq_dev.initialized) return;

    CUDA_CHECK(cudaFree(g_turboq_dev.d_Q));
    CUDA_CHECK(cudaFree(g_turboq_dev.d_x));
    CUDA_CHECK(cudaFree(g_turboq_dev.d_y));
    g_turboq_dev.d_Q = nullptr;
    g_turboq_dev.d_x = nullptr;
    g_turboq_dev.d_y = nullptr;
    g_turboq_dev.initialized = false;
}

bool turboq_cuda_available(void) {
    int device = -1;
    hipError_t err = hipGetDevice(&device);
    return err == hipSuccess && device >= 0;
}

void turboq_cuda_set_rotation(const float * Q_host, int d, uint64_t seed) {
    if (!g_turboq_dev.initialized) turboq_cuda_init();
    if (d != TURBOQ_D) {
        return;
    }
    (void)seed;
    CUDA_CHECK(cudaMemcpy(g_turboq_dev.d_Q, Q_host, d * d * sizeof(float), cudaMemcpyHostToDevice));
}

void turboq_cuda_matvec_forward(float * y, const float * Q, const float * x, int d) {
    if (!g_turboq_dev.initialized) turboq_cuda_init();

    if (d == TURBOQ_D) {
        CUDA_CHECK(cudaMemcpy(g_turboq_dev.d_x, x, d * sizeof(float), cudaMemcpyHostToDevice));

        GGML_CUDA_TURBOQ_NAMESPACE::turboq_matvec_forward_cuda<128>(
            g_turboq_dev.d_y,
            g_turboq_dev.d_Q,
            g_turboq_dev.d_x,
            g_turboq_dev.stream
        );

        CUDA_CHECK(cudaMemcpy(y, g_turboq_dev.d_y, d * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        matvec_cpu(y, Q, x, d);
    }
}

void turboq_cuda_matvec_inverse(float * x, const float * Q, const float * y, int d) {
    if (!g_turboq_dev.initialized) turboq_cuda_init();

    if (d == TURBOQ_D) {
        CUDA_CHECK(cudaMemcpy(g_turboq_dev.d_x, y, d * sizeof(float), cudaMemcpyHostToDevice));

        GGML_CUDA_TURBOQ_NAMESPACE::turboq_matvec_inverse_cuda<128>(
            g_turboq_dev.d_y,
            g_turboq_dev.d_Q,
            g_turboq_dev.d_x,
            g_turboq_dev.stream
        );

        CUDA_CHECK(cudaMemcpy(x, g_turboq_dev.d_y, d * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        for (int j = 0; j < d; j++) {
            float sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += Q[i * d + j] * y[i];
            }
            x[j] = sum;
        }
    }
}

static constexpr int TBQ_QK = 256;
static constexpr int TBQ_D = 128;

template<int QK, int D>
__global__ void k_tbq3_quant_kernel(
    block_tbq3_0 * __restrict__ dst,
    const float * __restrict__ src,
    const float * __restrict__ Q,
    int64_t nblocks) {

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;

    if (block_id >= nblocks) return;

    const int64_t src_offset = block_id * QK;
    const int64_t dst_offset = block_id;

    __shared__ float s_sq[QK];
    __shared__ float s_unit[QK];
    __shared__ float s_rot[QK];
    __shared__ uint8_t s_idx[QK];
    __shared__ float s_norm;

    const float val = src[src_offset + tid];
    s_sq[tid] = val * val;
    __syncthreads();

    for (int stride = QK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sq[tid] += s_sq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_norm = sqrtf(s_sq[0] + 1e-10f);
    }
    __syncthreads();

    const float norm = s_norm;
    s_unit[tid] = norm > 0.0f ? (val / norm) : 0.0f;
    __syncthreads();

    if (tid < D) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < D; j++) {
            sum += Q[tid * D + j] * s_unit[j];
        }
        s_rot[tid] = sum;
    } else if (tid < 2 * D) {
        float sum = 0.0f;
        const int half_idx = 0;  // Same Q rows (0-127) for second half of unit vector
        #pragma unroll
        for (int j = 0; j < D; j++) {
            sum += Q[half_idx * D + j] * s_unit[D + j];
        }
        s_rot[tid] = sum;
    }
    __syncthreads();

    const float scaled = s_rot[tid] * sqrtf((float)QK);

    uint8_t idx = 0;
    if (scaled >= 1.7480f) idx = 7;
    else if (scaled >= 1.0500f) idx = 6;
    else if (scaled >= 0.5006f) idx = 5;
    else if (scaled >= 0.0000f) idx = 4;
    else if (scaled >= -0.5006f) idx = 3;
    else if (scaled >= -1.0500f) idx = 2;
    else if (scaled >= -1.7480f) idx = 1;
    else idx = 0;
    s_idx[tid] = idx;
    __syncthreads();

    if (tid == 0) {
        dst[dst_offset].d = __float2half(norm);
    }
    if (tid < QK / 8) {
        const int base = tid * 8;
        uint32_t bits = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            bits |= uint32_t(s_idx[base + j] & 0x7) << (j * 3);
        }
        dst[dst_offset].qs[tid * 3 + 0] = uint8_t(bits & 0xff);
        dst[dst_offset].qs[tid * 3 + 1] = uint8_t((bits >> 8) & 0xff);
        dst[dst_offset].qs[tid * 3 + 2] = uint8_t((bits >> 16) & 0xff);
    }
}

void turboq_cuda_quantize_tbq3(block_tbq3_0 * dst, const float * src, int nblocks) {
    if (!g_turboq_dev.initialized) turboq_cuda_init();

    uint64_t seed = turboq_seed_from_row(0);
    const float * Q_host = turboq_get_rotation(TBQ_D, seed);
    turboq_cuda_set_rotation(Q_host, TBQ_D, seed);

    const int block_size = TBQ_QK;
    const int grid_size = nblocks;

    k_tbq3_quant_kernel<TBQ_QK, TBQ_D><<<grid_size, block_size, 0, g_turboq_dev.stream>>>(
        dst, src, g_turboq_dev.d_Q, nblocks);
    CUDA_CHECK(cudaGetLastError());
}

template<int QK, int D>
__global__ void k_tbq4_quant_kernel(
    block_tbq4_0 * __restrict__ dst,
    const float * __restrict__ src,
    const float * __restrict__ Q,
    int64_t nblocks) {

    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;

    if (block_id >= nblocks) return;

    const int64_t src_offset = block_id * QK;
    const int64_t dst_offset = block_id;

    __shared__ float s_sq[QK];
    __shared__ float s_unit[QK];
    __shared__ float s_rot[QK];
    __shared__ uint8_t s_idx[QK];
    __shared__ float s_norm;

    const float val = src[src_offset + tid];
    s_sq[tid] = val * val;
    __syncthreads();

    for (int stride = QK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sq[tid] += s_sq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_norm = sqrtf(s_sq[0] + 1e-10f);
    }
    __syncthreads();

    const float norm = s_norm;
    s_unit[tid] = norm > 0.0f ? (val / norm) : 0.0f;
    __syncthreads();

    if (tid < D) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < D; j++) {
            sum += Q[tid * D + j] * s_unit[j];
        }
        s_rot[tid] = sum;
    } else if (tid < 2 * D) {
        float sum = 0.0f;
        const int half_idx = 0;  // Same Q rows (0-127) for second half of unit vector
        #pragma unroll
        for (int j = 0; j < D; j++) {
            sum += Q[half_idx * D + j] * s_unit[D + j];
        }
        s_rot[tid] = sum;
    }
    __syncthreads();

    const float scaled = s_rot[tid] * sqrtf((float)QK);

    uint8_t idx = 0;
    if (scaled >=  2.4008f) idx = 15;
    else if (scaled >=  1.8435f) idx = 14;
    else if (scaled >=  1.4371f) idx = 13;
    else if (scaled >=  1.0993f) idx = 12;
    else if (scaled >=  0.7996f) idx = 11;
    else if (scaled >=  0.5225f) idx = 10;
    else if (scaled >=  0.2583f) idx = 9;
    else if (scaled >=  0.0000f) idx = 8;
    else if (scaled >= -0.2583f) idx = 7;
    else if (scaled >= -0.5225f) idx = 6;
    else if (scaled >= -0.7996f) idx = 5;
    else if (scaled >= -1.0993f) idx = 4;
    else if (scaled >= -1.4371f) idx = 3;
    else if (scaled >= -1.8435f) idx = 2;
    else if (scaled >= -2.4008f) idx = 1;
    else idx = 0;
    s_idx[tid] = idx;
    __syncthreads();

    if (tid == 0) {
        dst[dst_offset].d = __float2half(norm);
    }

    if (tid < QK / 2) {
        const uint8_t lo = s_idx[2 * tid + 0] & 0x0f;
        const uint8_t hi = s_idx[2 * tid + 1] & 0x0f;
        dst[dst_offset].qs[tid] = lo | (hi << 4);
    }
}

void turboq_cuda_quantize_tbq4(block_tbq4_0 * dst, const float * src, int nblocks) {
    if (!g_turboq_dev.initialized) turboq_cuda_init();

    // Generate Q on CPU if not already cached, then upload to device
    uint64_t seed = turboq_seed_from_row(0);
    const float * Q_host = turboq_get_rotation(TBQ_D, seed);
    turboq_cuda_set_rotation(Q_host, TBQ_D, seed);

    const int block_size = TBQ_QK;
    const int grid_size = nblocks;

    k_tbq4_quant_kernel<TBQ_QK, TBQ_D><<<grid_size, block_size, 0, g_turboq_dev.stream>>>(
        dst, src, g_turboq_dev.d_Q, nblocks);
    CUDA_CHECK(cudaGetLastError());
}
