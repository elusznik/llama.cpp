#pragma once

// TurboQuant helpers used by the CPU quantizers.

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void turboq_rotate_forward(float * y, const float * x, int64_t d, uint64_t seed);

void turboq_rotate_inverse(float * x, const float * y, int64_t d, uint64_t seed);

uint64_t turboq_seed_from_row(int64_t row_idx);

const float * turboq_get_rotation(int64_t d, uint64_t seed);
const float * turboq_get_projection(int64_t d, uint64_t seed);

void turboq_dequantize_slice_tbq3_0_f32(const void * vx, float * y, int64_t k, int64_t offset, int64_t n);
void turboq_dequantize_slice_tbq4_0_f32(const void * vx, float * y, int64_t k, int64_t offset, int64_t n);
void turboq_dequantize_slice_tbq34_0_f32(const void * vx, float * y, int64_t k, int64_t offset, int64_t n);
void turboq_dequantize_slice_tbqp3_0_f32(const void * vx, float * y, int64_t k, int64_t offset, int64_t n);
void turboq_dequantize_slice_tbqp4_0_f32(const void * vx, float * y, int64_t k, int64_t offset, int64_t n);
void turboq_dequantize_slice_tbqp34_0_f32(const void * vx, float * y, int64_t k, int64_t offset, int64_t n);
void turboq_dequantize_slice_tbqp3_0_mse_f32(const void * vx, float * y, int64_t k, int64_t offset, int64_t n);
void turboq_dequantize_slice_tbqp4_0_mse_f32(const void * vx, float * y, int64_t k, int64_t offset, int64_t n);
void turboq_dequantize_slice_tbqp34_0_mse_f32(const void * vx, float * y, int64_t k, int64_t offset, int64_t n);

float turboq_vec_dot_tbqp3_0_f32(int n, const void * vx, const float * q);
float turboq_vec_dot_tbqp4_0_f32(int n, const void * vx, const float * q);
float turboq_vec_dot_tbqp34_0_f32(int n, const void * vx, const float * q);

float turboq_vec_dot_tbqp3_0_q8_K(int n, const void * vx, const void * vy);
float turboq_vec_dot_tbqp4_0_q8_K(int n, const void * vx, const void * vy);
float turboq_vec_dot_tbqp34_0_q8_K(int n, const void * vx, const void * vy);

#ifdef __cplusplus
}
#endif
