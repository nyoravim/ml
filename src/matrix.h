#ifndef _MATRIX_H
#define _MATRIX_H

#include <stdint.h>
#include <stdbool.h>

typedef struct matrix {
    uint32_t rows, columns;
    float* data;
} matrix_t;

struct nv_allocator;

matrix_t* mat_alloc(const struct nv_allocator* alloc, uint32_t rows, uint32_t columns);
void mat_free(const struct nv_allocator* alloc, matrix_t* mat);

void mat_copy(matrix_t* dst, const matrix_t* src);

void mat_zero(matrix_t* mat);

enum {
    MAT_MUL_TRANSPOSE_LHS = (1 << 0),
    MAT_MUL_TRANSPOSE_RHS = (1 << 1),
    MAT_MUL_ZERO_RESULT = (1 << 2),
};

void mat_mul(matrix_t* result, const matrix_t* lhs, const matrix_t* rhs, uint32_t flags);

void mat_scale(matrix_t* mat, float scalar);

void mat_relu(matrix_t* output, const matrix_t* input);
void mat_sigmoid(matrix_t* output, const matrix_t* input);
void mat_softmax(matrix_t* output, const matrix_t* input);

#endif
