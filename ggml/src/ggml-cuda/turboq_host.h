#pragma once
// C-callable TurboQuant HIP wrappers.

#ifdef __cplusplus
extern "C" {
#endif

bool turboq_cuda_available(void);

void turboq_cuda_init(void);
void turboq_cuda_free(void);

void turboq_cuda_set_rotation(const float * Q_host, int d, uint64_t seed);

void turboq_cuda_matvec_forward(float * y, const float * Q, const float * x, int d);
void turboq_cuda_matvec_inverse(float * x, const float * Q, const float * y, int d);

void turboq_cuda_quantize_tbq3(block_tbq3_0 * dst, const float * src, int nblocks);
void turboq_cuda_quantize_tbq4(block_tbq4_0 * dst, const float * src, int nblocks);

#ifdef __cplusplus
}
#endif
