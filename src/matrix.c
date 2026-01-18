#include "matrix.h"

#include <assert.h>
#include <string.h>

#include <nyoravim/mem.h>

#include <log.h>

matrix_t* mat_alloc(uint32_t rows, uint32_t columns) {
    size_t meta_size = sizeof(matrix_t);
    size_t data_size = sizeof(float) * rows * columns;

    matrix_t* mat = nv_alloc(meta_size + data_size);
    assert(mat);

    mat->rows = rows;
    mat->columns = columns;
    mat->data = (void*)mat + meta_size;

    mat_zero(mat);
    return mat;
}

void mat_free(matrix_t* mat) { nv_free(mat); }

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
