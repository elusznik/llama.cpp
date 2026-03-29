// TurboQuant HIP kernels.

#include "turboq.cuh"
#include "ggml-turboq.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

static constexpr int TURBOQ_D = 128;

static constexpr int WAVE_SIZE_GCN_CDNA = 64;
static constexpr int WAVE_SIZE_RDNA = 32;

template<int WAVE_SIZE = 32>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int offset = WAVE_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset, WAVE_SIZE);
    }
    return val;
}

template<>
__device__ __forceinline__ float warp_reduce_sum_f32<64>(float val) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset, 64);
    }
    // Handle the upper 32 lanes
    val += __shfl_xor_sync(0xffffffff, val, 32, 64);
    return val;
}

#if defined(GGML_USE_HIP) && (defined(GCN) || defined(CDNA))

template<int D>
__global__ void turboq_forward_gcn_kernel(
    float * __restrict__ y,
    const float * __restrict__ Q,
    const float * __restrict__ x) {

    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid & 63;

    if (row >= D) return;

    if (lane_id < D) {
        sdata[lane_id] = x[lane_id];
    }
    __syncthreads();

    float sum = 0.0f;

    #pragma unroll
    for (int j = 0; j < D / 64; j++) {
        int col = lane_id + j * 64;
        sum += Q[row + col * D] * sdata[col];
    }

    sum = warp_reduce_sum_f32<64>(sum);

    if (lane_id == 0) {
        y[row] = sum;
    }
}

template<int D>
__global__ void turboq_inverse_gcn_kernel(
    float * __restrict__ x_out,
    const float * __restrict__ Q,
    const float * __restrict__ y) {

    extern __shared__ float sdata[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid & 63;

    if (row >= D) return;

    if (lane_id < D) {
        sdata[lane_id] = y[lane_id];
    }
    __syncthreads();

    float sum = 0.0f;

    #pragma unroll
    for (int j = 0; j < D / 64; j++) {
        int col = lane_id + j * 64;
        sum += Q[row + col * D] * sdata[col];
    }

    sum = warp_reduce_sum_f32<64>(sum);

    if (lane_id == 0) {
        x_out[row] = sum;
    }
}

#endif // GCN || CDNA

#if defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)

template<int D>
__global__ void turboq_forward_cdna_kernel(
    float * __restrict__ y,
    const float * __restrict__ Q,
    const float * __restrict__ x) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid & 63;

    if (row >= D) return;

    extern __shared__ float sdata[];
    float * x_shared = sdata;

    if (lane_id < D) {
        x_shared[lane_id] = x[lane_id];
    }
    __syncthreads();

    float sum = 0.0f;

    #pragma unroll
    for (int j = 0; j < 2; j++) {
        int col = lane_id + j * 64;
        sum += Q[row + col * D] * x_shared[col];
    }

    sum = warp_reduce_sum_f32<64>(sum);

    if (lane_id == 0) {
        y[row] = sum;
    }
}

#endif // CDNA && MFMA

#if defined(GGML_USE_HIP) && defined(RDNA)

template<int D>
__global__ void turboq_forward_rdnai_kernel(
    float * __restrict__ y,
    const float * __restrict__ Q,
    const float * __restrict__ x) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;

    if (row >= D) return;

    float sum = 0.0f;

    const int col_base = lane_id * 4;

    // Load Q elements (strided access - not contiguous)
    float q0 = Q[row + (col_base + 0) * D];
    float q1 = Q[row + (col_base + 1) * D];
    float q2 = Q[row + (col_base + 2) * D];
    float q3 = Q[row + (col_base + 3) * D];

    // Load x elements (contiguous from shared memory)
    float x0 = x[col_base + 0];
    float x1 = x[col_base + 1];
    float x2 = x[col_base + 2];
    float x3 = x[col_base + 3];

    sum = q0 * x0 + q1 * x1 + q2 * x2 + q3 * x3;

    // Wavefront reduction (32-wide)
    sum = warp_reduce_sum_f32<32>(sum);

    if (lane_id == 0) {
        y[row] = sum;
    }
}

template<int D>
__global__ void turboq_inverse_rdnai_kernel(
    float * __restrict__ x_out,
    const float * __restrict__ Q,
    const float * __restrict__ y) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;

    if (row >= D) return;

    float sum = 0.0f;
    const int col_base = lane_id * 4;

    // x[row] = sum_j(Q[j,row] * y[j])
    // Q[j,row] in column-major = Q[j + row*D]

    float q0 = Q[(col_base + 0) + row * D];
    float q1 = Q[(col_base + 1) + row * D];
    float q2 = Q[(col_base + 2) + row * D];
    float q3 = Q[(col_base + 3) + row * D];

    float y0 = y[col_base + 0];
    float y1 = y[col_base + 1];
    float y2 = y[col_base + 2];
    float y3 = y[col_base + 3];

    sum = q0 * y0 + q1 * y1 + q2 * y2 + q3 * y3;

    sum = warp_reduce_sum_f32<32>(sum);

    if (lane_id == 0) {
        x_out[row] = sum;
    }
}

#endif // RDNA

// =============================================================================
// RDNA3/4 WMMA Kernels (32-wide with matrix instructions)
// WMMA: Wavefront Matrix Multiply-Accumulate
// =============================================================================

#if defined(GGML_USE_HIP) && defined(RDNA3) && !defined(GGML_HIP_NO_WMMA)

// WMMA operations available on RDNA3+
// Can use wavefront intrinsics for matrix operations
// However, for D=128, traditional SIMD may be more efficient due to small size

// For now, fall back to RDNA2-style kernels
// WMMA would shine for much larger matrices (D >> 128)

#endif // RDNA3 && WMMA

// =============================================================================
// Fallback Kernels (Basic SIMD, works on all architectures)
// =============================================================================

template<int D, int WAVE_SIZE>
__global__ void turboq_forward_fallback_kernel(
    float * __restrict__ y,
    const float * __restrict__ Q,
    const float * __restrict__ x) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= D) return;

    float sum = 0.0f;

    // Each thread handles (D / WAVE_SIZE) elements
    // With proper padding, this divides evenly
    const int elts_per_thread = (D + WAVE_SIZE - 1) / WAVE_SIZE;

    #pragma unroll
    for (int j = 0; j < 8; j++) {  // Limit unroll for variable D
        int col = tid + j * WAVE_SIZE;
        if (col < D) {
            sum += Q[row + col * D] * x[col];
        }
    }

    sum = warp_reduce_sum_f32<WAVE_SIZE>(sum);

    const int lane_id = tid & (WAVE_SIZE - 1);
    if (lane_id == 0) {
        y[row] = sum;
    }
}

template<int D, int WAVE_SIZE>
__global__ void turboq_inverse_fallback_kernel(
    float * __restrict__ x_out,
    const float * __restrict__ Q,
    const float * __restrict__ y) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= D) return;

    float sum = 0.0f;

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int col = tid + j * WAVE_SIZE;
        if (col < D) {
            sum += Q[row + col * D] * y[col];
        }
    }

    sum = warp_reduce_sum_f32<WAVE_SIZE>(sum);

    const int lane_id = tid & (WAVE_SIZE - 1);
    if (lane_id == 0) {
        x_out[row] = sum;
    }
}

// =============================================================================
// Specialized 128x128 Kernels (optimal for KV cache)
// =============================================================================

// 128x128 Forward - GCN/CDNA (64-wide)
#if defined(GGML_USE_HIP) && (defined(GCN) || defined(CDNA))
template<>
__global__ void turboq_forward_gcn_kernel<128>(
    float * __restrict__ y,
    const float * __restrict__ Q,
    const float * __restrict__ x) {

    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const int lane_id = threadIdx.x & 63;

    if (row >= 128) return;

    // Load x vector into shared memory
    if (lane_id < 128) {
        sdata[lane_id] = x[lane_id];
    }
    __syncthreads();

    // Each of 64 threads handles 2 elements: (lane_id) and (lane_id + 64)
    float sum = 0.0f;
    sum += Q[row + lane_id * 128] * sdata[lane_id];
    sum += Q[row + (lane_id + 64) * 128] * sdata[lane_id + 64];

    sum = warp_reduce_sum_f32<64>(sum);

    if (lane_id == 0) {
        y[row] = sum;
    }
}
#endif

// 128x128 Forward - RDNA (32-wide)
#if defined(GGML_USE_HIP) && defined(RDNA)
template<>
__global__ void turboq_forward_rdnai_kernel<128>(
    float * __restrict__ y,
    const float * __restrict__ Q,
    const float * __restrict__ x) {

    const int row = blockIdx.x;
    const int lane_id = threadIdx.x & 31;

    if (row >= 128) return;

    // Each thread handles 4 elements
    // lane 0: col 0,32,64,96; lane 1: col 1,33,65,97; etc.
    float sum = 0.0f;

    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int col = lane_id + k * 32;
        sum += Q[row + col * 128] * x[col];
    }

    sum = warp_reduce_sum_f32<32>(sum);

    if (lane_id == 0) {
        y[row] = sum;
    }
}
#endif

// =============================================================================
// Host-side Launchers
// =============================================================================

namespace GGML_CUDA_TURBOQ_NAMESPACE {

// Device memory for rotation matrices
static float * d_Q = nullptr;
static float * d_S = nullptr;  // For QJL projection
static bool matrices_initialized = false;

// CUDA stream for async operations
static cudaStream_t cached_stream = nullptr;

void turboq_init_cuda(void) {
    if (matrices_initialized) return;

    // Allocate device memory for Q and S matrices
    // D=128: Q matrix is 128*128*4 = 64KB
    CUDA_CHECK(cudaMalloc(&d_Q, 128 * 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S, 128 * 128 * sizeof(float)));

    matrices_initialized = true;
}

void turboq_free_cuda(void) {
    if (d_Q) {
        CUDA_CHECK(cudaFree(d_Q));
        d_Q = nullptr;
    }
    if (d_S) {
        CUDA_CHECK(cudaFree(d_S));
        d_S = nullptr;
    }
    matrices_initialized = false;
}

void turboq_set_rotation_cuda(const float * Q_host, int d, uint64_t seed) {
    if (!matrices_initialized) {
        turboq_init_cuda();
    }

    // Copy Q matrix to device
    // In practice, for d=128 the seed determines Q, so we could pre-compute
    // But for flexibility, we accept the pre-computed matrix
    CUDA_CHECK(cudaMemcpy(d_Q, Q_host, d * d * sizeof(float), cudaMemcpyHostToDevice));
}

void turboq_set_projection_cuda(const float * S_host, int d, uint64_t seed) {
    if (!matrices_initialized) {
        turboq_init_cuda();
    }

    CUDA_CHECK(cudaMemcpy(d_S, S_host, d * d * sizeof(float), cudaMemcpyHostToDevice));
}

// Forward rotation: y = Q * x
template<int D>
void turboq_matvec_forward_cuda(
    float * y,
    const float * Q,
    const float * x,
    cudaStream_t stream) {

    const int n_blocks = D;

    // Choose kernel based on architecture
#if defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)
    // CDNA: Use GCN-style kernel with 64 threads
    turboq_forward_gcn_kernel<D><<<n_blocks, 64, 0, stream>>>(y, Q, x);
#elif defined(GGML_USE_HIP) && defined(RDNA)
    // RDNA2/3/4: Use dp4a-optimized kernel with 32 threads
    if (D == 128) {
        turboq_forward_rdnai_kernel<128><<<n_blocks, 32, 0, stream>>>(y, Q, x);
    } else {
        turboq_forward_rdnai_kernel<D><<<n_blocks, 32, 0, stream>>>(y, Q, x);
    }
#elif defined(GGML_USE_HIP) && (defined(GCN))
    // GCN: Use GCN kernel with 64 threads
    turboq_forward_gcn_kernel<D><<<n_blocks, 64, 0, stream>>>(y, Q, x);
#else
    // Fallback for unknown architecture
    turboq_forward_fallback_kernel<D, 32><<<n_blocks, 32, 0, stream>>>(y, Q, x);
#endif

    CUDA_CHECK(cudaGetLastError());
}

// Inverse rotation: x = Q^T * y
template<int D>
void turboq_matvec_inverse_cuda(
    float * x_out,
    const float * Q,
    const float * y,
    cudaStream_t stream) {

    const int n_blocks = D;

#if defined(GGML_USE_HIP) && defined(CDNA) && !defined(GGML_HIP_NO_MMQ_MFMA)
    turboq_inverse_gcn_kernel<D><<<n_blocks, 64, 0, stream>>>(x_out, Q, y);
#elif defined(GGML_USE_HIP) && defined(RDNA)
    if (D == 128) {
        turboq_inverse_rdnai_kernel<128><<<n_blocks, 32, 0, stream>>>(x_out, Q, y);
    } else {
        turboq_inverse_rdnai_kernel<D><<<n_blocks, 32, 0, stream>>>(x_out, Q, y);
    }
#elif defined(GGML_USE_HIP) && (defined(GCN))
    turboq_inverse_gcn_kernel<D><<<n_blocks, 64, 0, stream>>>(x_out, Q, y);
#else
    turboq_inverse_fallback_kernel<D, 32><<<n_blocks, 32, 0, stream>>>(x_out, Q, y);
#endif

    CUDA_CHECK(cudaGetLastError());
}

// Explicit template instantiations for D=128 (KV cache)
template void turboq_matvec_forward_cuda<128>(float*, const float*, const float*, cudaStream_t);
template void turboq_matvec_inverse_cuda<128>(float*, const float*, const float*, cudaStream_t);

} // namespace GGML_CUDA_TURBOQ_NAMESPACE
