// TurboQuant quantize/dequant. Built with arch flags, and the rotation matvec
// reuses ggml_vec_dot_f32 so it vectorizes on every supported CPU arch.

#include "quants.h"
#include "vec.h"
#include "ggml-turboq-impl.h"

static void matvec_row(float * y, const float * M, const float * x, int64_t d) {
    for (int64_t i = 0; i < d; ++i) {
        ggml_vec_dot_f32(d, &y[i], 0, M + i * d, 0, x, 0, 1);
    }
}

static void matvec_t(float * y, const float * M, const float * x, int64_t d) {
    for (int64_t j = 0; j < d; ++j) {
        ggml_vec_dot_f32(d, &y[j], 0, M + j * d, 0, x, 0, 1);
    }
}

void quantize_row_tbq3_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    turboq_quantize_tbq3_0(x, y, k);
}

void quantize_row_tbq4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    turboq_quantize_tbq4_0(x, y, k);
}

void ggml_cpu_dequantize_row_tbq3_0(const block_tbq3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turboq_dequantize_tbq3_0(x, y, k);
}

void ggml_cpu_dequantize_row_tbq4_0(const block_tbq4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turboq_dequantize_tbq4_0(x, y, k);
}
