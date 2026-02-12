#ifndef _FUNCTION_H
#define _FUNCTION_H

#include <stdint.h>

#include "matrix.h"

typedef enum {
    FUNCTION_OP_NOOP = 0,
    FUNCTION_OP_ZERO = 1,
    FUNCTION_OP_COPY = 2,

    FUNCTION_OP_ADD = 3,
    FUNCTION_OP_DOT = 4,

    FUNCTION_OP_RELU = 5,
    FUNCTION_OP_SIGMOID = 6,
    FUNCTION_OP_SOFTMAX = 7,
    FUNCTION_OP_CROSS_ENTROPY = 8, /* actual, expected */

    /* these three ops expect input and existing gradient.
     * output is in same dimensions as input and gradient */
    FUNCTION_OP_RELU_GRADIENT = 9,
    FUNCTION_OP_SIGMOID_GRADIENT = 10,
    FUNCTION_OP_SOFTMAX_GRADIENT = 11,

    FUNCTION_OP_CROSS_ENTROPY_GRADIENT = 12, /* actual, expected */
} function_op_id;

typedef enum {
    PARAMETER_SOURCE_DATA = 0,
    PARAMETER_SOURCE_WEIGHTS = 1,
} parameter_source;

struct function_op_parameter {
    uint32_t index;
    parameter_source source;
};

struct function_op {
    function_op_id id;
    uint32_t flags;

    uint32_t parameter_count;
    const struct function_op_parameter* parameters;

    uint32_t output_index;
};

typedef struct function function_t;

function_t* function_compile(uint32_t operation_count, const struct function_op* operations);
void function_free(function_t* func);

struct function_context {
    matrix_t* const* data;
    const matrix_t* const* weights;
};

void function_evaluate(const function_t* func, const struct function_context* ctx);

#endif
