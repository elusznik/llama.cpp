#pragma once

// TurboQuant HIP kernels.

#include "common.cuh"

#include <cstdint>

namespace GGML_CUDA_TURBOQ_NAMESPACE {

template<int D>
void turboq_matvec_forward_cuda(
    float * y,
    const float * Q,
    const float * x,
    cudaStream_t stream);

template<int D>
void turboq_matvec_inverse_cuda(
    float * x_out,
    const float * Q,
    const float * y,
    cudaStream_t stream);

void turboq_init_cuda(void);
void turboq_free_cuda(void);

void turboq_set_rotation_cuda(const float * Q_host, int d, uint64_t seed);
void turboq_set_projection_cuda(const float * S_host, int d, uint64_t seed);

} // namespace GGML_CUDA_TURBOQ_NAMESPACE
