// TurboQuant scalar reference. The arch-built SIMD dequant lives in ggml-cpu/turboq.c.

#include "ggml-quants.h"
#include "ggml-turboq-impl.h"

static void matvec_row(float * y, const float * M, const float * x, int64_t d) {
    for (int64_t i = 0; i < d; ++i) {
        float sum = 0.0f;
        for (int64_t j = 0; j < d; ++j) {
            sum += M[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}

static void matvec_t(float * y, const float * M, const float * x, int64_t d) {
    for (int64_t j = 0; j < d; j++) {
        float sum = 0.0f;
        for (int64_t i = 0; i < d; ++i) {
            sum += M[j * d + i] * x[i];
        }
        y[j] = sum;
    }
}

void quantize_row_tbq3_0_ref(const float * GGML_RESTRICT x, block_tbq3_0 * GGML_RESTRICT y, int64_t k) {
    turboq_quantize_tbq3_0(x, y, k);
}

void dequantize_row_tbq3_0(const block_tbq3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turboq_dequantize_tbq3_0(x, y, k);
}

size_t quantize_tbq3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_K == 0);

    const int64_t nb_per_row = n_per_row / QK_K;
    const size_t row_size = nb_per_row * sizeof(block_tbq3_0);

    for (int64_t row = 0; row < nrows; row++) {
        const float * row_src = src + row * n_per_row;
        block_tbq3_0 * row_dst = (block_tbq3_0 *)((char *)dst + row * row_size);
        quantize_row_tbq3_0_ref(row_src, row_dst, n_per_row);
    }
    return nrows * row_size;
}

void quantize_row_tbq4_0_ref(const float * GGML_RESTRICT x, block_tbq4_0 * GGML_RESTRICT y, int64_t k) {
    turboq_quantize_tbq4_0(x, y, k);
}

void dequantize_row_tbq4_0(const block_tbq4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turboq_dequantize_tbq4_0(x, y, k);
}

size_t quantize_tbq4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_K == 0);

    const int64_t nb_per_row = n_per_row / QK_K;
    const size_t row_size = nb_per_row * sizeof(block_tbq4_0);

    for (int64_t row = 0; row < nrows; row++) {
        const float * row_src = src + row * n_per_row;
        block_tbq4_0 * row_dst = (block_tbq4_0 *)((char *)dst + row * row_size);
        quantize_row_tbq4_0_ref(row_src, row_dst, n_per_row);
    }
    return nrows * row_size;
}
