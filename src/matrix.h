#ifndef _MATRIX_H
#define _MATRIX_H

#include <stdint.h>
#include <stdbool.h>

typedef struct matrix {
    uint32_t rows, columns;
    float* data;
} matrix_t;

matrix_t* mat_alloc(uint32_t rows, uint32_t columns);
void mat_free(matrix_t* mat);

void mat_zero(matrix_t* mat);

enum {
    MAT_MUL_TRANSPOSE_LHS = (1 << 0),
    MAT_MUL_TRANSPOSE_RHS = (1 << 1),
    MAT_MUL_ZERO_RESULT = (1 << 2),
};

void mat_mul(matrix_t* result, const matrix_t* lhs, const matrix_t* rhs, uint32_t flags);

#endif
