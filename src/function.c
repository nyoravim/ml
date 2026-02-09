#include "function.h"

#include <assert.h>
#include <string.h>

#include <nyoravim/mem.h>
#include <nyoravim/log.h>

typedef struct function {
    /* parameter & op data is part of the same block as function; we dont need to worry about
     * freeing it */
    uint32_t op_count;
    const struct function_op* ops;
} function_t;

function_t* function_compile(uint32_t operation_count, const struct function_op* operations) {
    NV_LOG_TRACE("compiling function");

    if (operation_count == 0) {
        NV_LOG_ERROR("no operations passed to function_compile");
        return NULL;
    }

    uint32_t parameter_count = 0;
    for (uint32_t i = 0; i < operation_count; i++) {
        const struct function_op* op = &operations[i];
        parameter_count += op->parameter_count;
    }

    size_t function_size = sizeof(function_t) + operation_count * sizeof(struct function_op) +
                           parameter_count * sizeof(struct function_op_parameter);

    /* basically just arranging memory in a way that is convenient to access at runtime */
    function_t* func = nv_alloc(function_size);
    assert(func);

    struct function_op* ops = (void*)func + sizeof(function_t);
    struct function_op_parameter* parameters =
        (void*)ops + operation_count * sizeof(struct function_op);

    func->op_count = operation_count;
    func->ops = ops;

    uint32_t parameter_index = 0;
    for (uint32_t i = 0; i < operation_count; i++) {
        const struct function_op* src_op = &operations[i];
        struct function_op* dst_op = &ops[i];

        struct function_op_parameter* dst_params = parameters + parameter_index;
        parameter_index += src_op->parameter_count;

        memcpy(dst_params, src_op->parameters,
               src_op->parameter_count * sizeof(struct function_op_parameter));

        memcpy(dst_op, src_op, sizeof(struct function_op));
        dst_op->parameters = dst_params;
    }

    return func;
}

/* see function_t definition above */
void function_free(function_t* func) { nv_free(func); }

static void function_op_evaluate(const struct function_op* op, const matrix_t* const* params_data,
                                 matrix_t* output) {
    switch (op->id) {
    case FUNCTION_OP_NOOP:
        NV_LOG_TRACE("no-op");
        assert(op->parameter_count == 0);

        break;
    case FUNCTION_OP_ZERO:
        NV_LOG_TRACE("zero");
        assert(op->parameter_count == 0);

        mat_zero(output);
        break;
    case FUNCTION_OP_COPY:
        NV_LOG_TRACE("copy");
        assert(op->parameter_count == 1);

        mat_copy(output, params_data[0]);
        break;
    case FUNCTION_OP_ADD:
        NV_LOG_TRACE("add");
        assert(op->parameter_count == 1);

        mat_add(output, params_data[0], op->flags);
        break;
    case FUNCTION_OP_DOT:
        NV_LOG_TRACE("dot");
        assert(op->parameter_count == 2);

        mat_mul(output, params_data[0], params_data[1], op->flags);
        break;
    case FUNCTION_OP_RELU:
        NV_LOG_TRACE("relu");
        assert(op->parameter_count == 1);

        mat_relu(output, params_data[0]);
        break;
    case FUNCTION_OP_SIGMOID:
        NV_LOG_TRACE("sigmoid");
        assert(op->parameter_count == 1);

        mat_relu(output, params_data[0]);
        break;
    case FUNCTION_OP_SOFTMAX:
        NV_LOG_TRACE("softmax");
        assert(op->parameter_count == 1);

        mat_softmax(output, params_data[0]);
        break;
    case FUNCTION_OP_CROSS_ENTROPY:
        NV_LOG_TRACE("cross entropy");
        assert(op->parameter_count == 2);

        mat_cross_entropy(output, params_data[0], params_data[1]);
        break;
    case FUNCTION_OP_RELU_GRADIENT:
        NV_LOG_TRACE("relu gradient");
        assert(op->parameter_count == 1);

        mat_relu_gradient(output, params_data[0]);
        break;
    case FUNCTION_OP_SIGMOID_GRADIENT:
        NV_LOG_TRACE("sigmoid gradient");
        assert(op->parameter_count == 1);

        mat_sigmoid_gradient(output, params_data[0]);
        break;
    case FUNCTION_OP_SOFTMAX_GRADIENT:
        NV_LOG_TRACE("softmax gradient");
        assert(op->parameter_count == 1);

        mat_softmax_gradient(output, params_data[0]);
        break;
    case FUNCTION_OP_CROSS_ENTROPY_GRADIENT:
        NV_LOG_TRACE("cross entropy gradient");
        assert(op->parameter_count == 2);

        mat_cross_entropy_gradient(output, params_data[0], params_data[1]);
        break;
    default:
        NV_LOG_ERROR("invalid op id: %u", (uint32_t)op->id);
        break;
    }
}

void function_evaluate(const function_t* func, const struct function_context* ctx) {
    NV_LOG_TRACE("function eval");

    uint32_t params_capacity = 0;
    const matrix_t** params_data = NULL;

    for (uint32_t i = 0; i < func->op_count; i++) {
        const struct function_op* op = &func->ops[i];
        NV_LOG_TRACE("evaluating op %u (%u params)", i, op->parameter_count);

        if (params_capacity < op->parameter_count) {
            NV_LOG_TRACE("reallocating temp buffer from %u to %u pointers", params_capacity,
                         op->parameter_count);

            params_data = nv_realloc(params_data, op->parameter_count * sizeof(const matrix_t*));
            params_capacity = op->parameter_count;
        }

        for (uint32_t i = 0; i < op->parameter_count; i++) {
            const struct function_op_parameter* param = &op->parameters[i];
            NV_LOG_TRACE("retrieving param %u", i);

            switch (param->source) {
            case PARAMETER_SOURCE_DATA:
                NV_LOG_TRACE("retrieving data matrix %u", param->index);
                params_data[i] = ctx->data[param->index];

                break;
            case PARAMETER_SOURCE_WEIGHTS:
                NV_LOG_TRACE("retrieving weight matrix %u", param->index);
                params_data[i] = ctx->weights[param->index];

                break;
            default:
                NV_LOG_WARN("invalid parameter source: %u (index %u)", (uint32_t)param->source,
                            param->index);

                params_data[i] = NULL;
                break;
            }
        }

        NV_LOG_TRACE("output index: %u", op->output_index);
        matrix_t* output = ctx->data[op->output_index];

        function_op_evaluate(op, params_data, output);
    }

    nv_free(params_data);
}
