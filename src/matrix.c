#include "matrix.h"

#include "prng.h"

#include <assert.h>
#include <string.h>
#include <math.h>

#include <nyoravim/mem.h>
#include <nyoravim/log.h>

matrix_t* mat_alloc(uint32_t rows, uint32_t columns) {
    NV_LOG_TRACE("allocating %ux%u matrix", rows, columns);

    size_t meta_size = sizeof(matrix_t);
    size_t data_size = sizeof(float) * rows * columns;
    size_t block_size = meta_size + data_size;

    matrix_t* mat = nv_alloc(block_size);
    if (!mat) {
        return NULL;
    }

    mat->rows = rows;
    mat->columns = columns;
    mat->data = (void*)mat + meta_size;

    return mat;
}

void mat_free(matrix_t* mat) { nv_free(mat); }

void mat_copy(matrix_t* dst, const matrix_t* src) {
    assert(dst->rows == src->rows);
    assert(dst->columns == src->columns);

    memcpy(dst->data, src->data, sizeof(float) * dst->rows * dst->columns);
}

void mat_zero(matrix_t* mat) {
    size_t data_size = sizeof(float) * mat->rows * mat->columns;
    memset(mat->data, 0, data_size);
}

void mat_randomize(struct prng* rng, matrix_t* mat) {
    uint32_t total = mat->rows * mat->columns;
    for (uint32_t i = 0; i < total; i++) {
        uint32_t value = rng ? prng_rand(rng) : prng_rand_g();
        mat->data[i] = (float)value / (float)UINT32_MAX;
    }
}

void mat_mul(matrix_t* result, const matrix_t* lhs, const matrix_t* rhs, uint32_t flags) {
    bool transpose_lhs = flags & MAT_TRANSPOSE_LHS;
    bool transpose_rhs = flags & MAT_TRANSPOSE_RHS;

    uint32_t lhs_rows = transpose_lhs ? lhs->columns : lhs->rows;
    uint32_t lhs_columns = transpose_lhs ? lhs->rows : lhs->columns;

    uint32_t rhs_rows = transpose_rhs ? rhs->columns : rhs->rows;
    uint32_t rhs_columns = transpose_rhs ? rhs->rows : rhs->columns;

    assert(lhs_columns == rhs_rows);
    assert(result->rows == lhs_rows);
    assert(result->columns == rhs_columns);

    if (flags & MAT_ZERO_RESULT) {
        mat_zero(result);
    }

    for (uint32_t m = 0; m < lhs_rows; m++) {
        for (uint32_t n = 0; n < rhs_columns; n++) {
            uint32_t result_index = m * rhs_columns + n;

            /* can also be rhs_rows */
            for (uint32_t x = 0; x < lhs_columns; x++) {
                uint32_t lhs_index = transpose_lhs ? x * lhs_columns + m : m * lhs_columns * x;
                uint32_t rhs_index = transpose_rhs ? n * rhs_columns + x : x * rhs_columns + n;

                result->data[result_index] += lhs->data[lhs_index] * rhs->data[rhs_index];
            }
        }
    }
}

void mat_add(matrix_t* lhs, const matrix_t* rhs, uint32_t flags) {
    assert(!(flags & ~(uint32_t)MAT_TRANSPOSE_RHS));
    bool transpose_rhs = flags & MAT_TRANSPOSE_RHS;

    uint32_t rhs_rows = transpose_rhs ? rhs->columns : rhs->rows;
    uint32_t rhs_columns = transpose_rhs ? rhs->rows : rhs->columns;

    assert(lhs->rows == rhs_rows);
    assert(lhs->columns == rhs_columns);

    for (uint32_t m = 0; m < lhs->rows; m++) {
        for (uint32_t n = 0; n < lhs->columns; n++) {
            uint32_t lhs_index = m * lhs->columns + n;
            uint32_t rhs_index = transpose_rhs ? n * lhs->columns + m : m * lhs->columns + n;

            lhs->data[lhs_index] += rhs->data[rhs_index];
        }
    }
}

void mat_scale(matrix_t* mat, float scalar) {
    uint32_t total = mat->rows * mat->columns;
    for (uint32_t i = 0; i < total; i++) {
        mat->data[i] *= scalar;
    }
}

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

void mat_cross_entropy(matrix_t* output, const matrix_t* actual, const matrix_t* expected) {
    assert(output->rows == actual->rows);
    assert(output->columns == actual->columns);

    assert(actual->rows == expected->rows);
    assert(actual->columns == expected->columns);

    uint32_t total = output->rows * output->columns;
    for (uint32_t i = 0; i < total; i++) {
        float x = expected->data[i];
        float y = actual->data[i];

        output->data[i] = x == 0.f ? 0.f : x * -logf(y);
    }
}

void mat_relu_gradient(matrix_t* output, const matrix_t* input) {
    assert(output->rows == output->columns);
    assert(output->rows == input->rows);
    assert(input->columns == 1);

    mat_zero(output);

    for (uint32_t i = 0; i < input->rows; i++) {
        float x = input->data[i];

        uint32_t output_index = i * (output->columns + 1);
        output->data[output_index] = x > 0.f ? 1.f : 0.f;
    }
}

void mat_sigmoid_gradient(matrix_t* output, const matrix_t* input) {
    assert(output->rows == output->columns);
    assert(output->rows == input->rows);
    assert(input->columns == 1);

    mat_zero(output);

    for (uint32_t i = 0; i < input->rows; i++) {
        float x = input->data[i];
        float sig = sigmoid(x);

        uint32_t output_index = i * (output->columns + 1);
        output->data[output_index] = sig * (1.f - sig);
    }
}

/*
 * input is vector of dc/da_n where n is the row
 * therefore softmax gradient gotta be a matrix of da_m/dz_n
 * where n is row, m is column
 */

void mat_softmax_gradient(matrix_t* output, const matrix_t* input) {
    /*
     * d/dx f(x)/g(x) = (f'(x)g(x) - f(x)g'(x))/(g^2(x))
     * S(x) = (e^x)/sum
     * S'(x) = (e^x * sum - e^x * e^x)/(sum ^ 2) = (e^x)/sum - (e^2x)/(sum^2) = S(x) - S^2(x) = S(x)
     * * (1 - S(x))
     *
     * da_n/dz_n = a_n * (1 - a_n)
     * da_m/dz_n { m != n } = -(e^z_m)(e^z_n)/sum^2 = -a_m * a_n
     */

    assert(output->rows == output->columns);
    assert(output->rows == input->rows);
    assert(input->columns == 1);

    float a[input->rows];
    float sum = 0.f;

    for (uint32_t i = 0; i < input->rows; i++) {
        float expf_in = expf(input->data[i]);
        a[i] = expf_in;

        sum += expf_in;
    }

    for (uint32_t i = 0; i < input->rows; i++) {
        a[i] /= sum;
    }

    for (uint32_t n = 0; n < output->rows; n++) {
        for (uint32_t m = 0; m < output->columns; m++) {
            uint32_t output_index = n * output->columns + m;
            float a_n = a[n];

            if (n == m) {
                output->data[output_index] = a_n * (1.f - a_n);
            } else {
                float a_m = a[m];
                output->data[output_index] = -a_m * a_n;
            }
        }
    }
}

void mat_cross_entropy_gradient(matrix_t* output, const matrix_t* actual,
                                const matrix_t* expected) {
    assert(output->rows == actual->rows);
    assert(output->columns == actual->columns);

    assert(actual->rows == expected->rows);
    assert(actual->columns == expected->columns);

    uint32_t total = output->rows * output->columns;
    for (uint32_t i = 0; i < total; i++) {
        /* cross entropy is -xlny
         * d/dy = -x/y */

        float x = expected->data[i];
        float y = actual->data[i];

        output->data[i] = -x / y;
    }
}
