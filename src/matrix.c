#include "matrix.h"

#include <assert.h>
#include <string.h>
#include <math.h>

#include <nyoravim/mem.h>

#include <log.h>

matrix_t* mat_alloc(const struct nv_allocator* alloc, uint32_t rows, uint32_t columns) {
    log_trace("allocating %ux%u matrix %s an allocator", rows, columns, alloc ? "with" : "without");

    size_t meta_size = sizeof(matrix_t);
    size_t data_size = sizeof(float) * rows * columns;
    size_t block_size = meta_size + data_size;

    matrix_t* mat;
    if (alloc) {
        mat = alloc->alloc(alloc->user, block_size);
    } else {
        mat = nv_alloc(block_size);
    }

    if (!mat) {
        return NULL;
    }

    mat->rows = rows;
    mat->columns = columns;
    mat->data = (void*)mat + meta_size;

    return mat;
}

void mat_free(const struct nv_allocator* alloc, matrix_t* mat) {
    if (!mat) {
        return;
    }

    if (!alloc) {
        nv_free(mat);
    } else if (alloc->free) {
        alloc->free(alloc->user, mat);
    }
}

void mat_copy(matrix_t* dst, const matrix_t* src) {
    assert(dst->rows == src->rows);
    assert(dst->columns == src->columns);

    memcpy(dst->data, src->data, sizeof(float) * dst->rows * dst->columns);
}

void mat_zero(matrix_t* mat) {
    size_t data_size = sizeof(float) * mat->rows * mat->columns;
    memset(mat->data, 0, data_size);
}

void mat_mul(matrix_t* result, const matrix_t* lhs, const matrix_t* rhs, uint32_t flags) {
    bool transpose_lhs = flags & MAT_MUL_TRANSPOSE_LHS;
    bool transpose_rhs = flags & MAT_MUL_TRANSPOSE_RHS;

    uint32_t lhs_rows = transpose_lhs ? lhs->columns : lhs->rows;
    uint32_t lhs_columns = transpose_lhs ? lhs->rows : lhs->columns;

    uint32_t rhs_rows = transpose_rhs ? rhs->columns : rhs->rows;
    uint32_t rhs_columns = transpose_rhs ? rhs->rows : rhs->columns;

    assert(lhs_columns == rhs_rows);
    assert(result->rows == lhs_rows);
    assert(result->columns == rhs_columns);

    if (flags & MAT_MUL_ZERO_RESULT) {
        mat_zero(result);
    }

    for (uint32_t m = 0; m < lhs_rows; m++) {
        for (uint32_t n = 0; n < rhs_columns; n++) {
            uint32_t result_index = m * rhs_columns + n;

            /* can also be rhs_rows */
            for (uint32_t x = 0; x < lhs_columns; x++) {
                uint32_t lhs_index = transpose_lhs ? x * lhs_rows + m : m * lhs_rows * x;
                uint32_t rhs_index = transpose_rhs ? n * rhs_rows + x : x * rhs_rows + n;

                result->data[result_index] += lhs->data[lhs_index] * rhs->data[rhs_index];
            }
        }
    }
}

void mat_scale(matrix_t* mat, float scalar) {
    uint32_t total = mat->rows * mat->columns;
    for (uint32_t i = 0; i < total; i++) {
        mat->data[i] *= scalar;
    }
}

static float relu(float x) { return x > 0 ? x : 0.f; }
static float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

void mat_relu(matrix_t* output, const matrix_t* input) {
    assert(output->rows == input->rows);
    assert(output->columns == input->columns);

    uint32_t total = output->rows * output->columns;
    for (uint32_t i = 0; i < total; i++) {
        float in = input->data[i];
        output->data[i] = in > 0 ? in : 0.f;
    }
}

void mat_sigmoid(matrix_t* output, const matrix_t* input) {
    assert(output->rows == input->rows);
    assert(output->columns == input->columns);

    uint32_t total = output->rows * output->columns;
    for (uint32_t i = 0; i < total; i++) {
        output->data[i] = sigmoid(input->data[i]);
    }
}

void mat_softmax(matrix_t* output, const matrix_t* input) {
    assert(output->rows == input->rows);
    assert(output->columns == input->columns);

    uint32_t total = output->rows * output->columns;
    float sum = 0.f;

    for (uint32_t i = 0; i < total; i++) {
        float expf_in = expf(input->data[i]);
        output->data[i] = expf_in;

        sum += expf_in;
    }

    mat_scale(output, 1.f / sum);
}
