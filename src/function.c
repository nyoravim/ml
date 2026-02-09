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

function_t* function_compile(const struct nv_list* operations) {
    NV_LOG_TRACE("compiling function");

    uint32_t op_count = 0;
    uint32_t parameter_count = 0;

    if (operations) {
        for (struct nv_list_node* node = operations->head; node != NULL; node = node->next) {
            op_count++;

            const struct function_op* op = node->value;
            parameter_count += op->parameter_count;
        }
    }

    if (op_count == 0) {
        NV_LOG_ERROR("no operations passed to function_compile");
        return NULL;
    }

    size_t function_size = sizeof(function_t) + op_count * sizeof(struct function_op) +
                           parameter_count * sizeof(struct function_op_parameter);

    /* basically just arranging memory in a way that is convenient to access at runtime */
    function_t* func = nv_alloc(function_size);
    assert(func);

    struct function_op* ops = (void*)func + sizeof(function_t);
    struct function_op_parameter* parameters = (void*)ops + op_count * sizeof(struct function_op);

    func->op_count = op_count;
    func->ops = ops;

    uint32_t op_index = 0;
    uint32_t parameter_index = 0;

    for (struct nv_list_node* node = operations->head; node != NULL; node = node->next) {
        const struct function_op* src_op = node->value;
        struct function_op* dst_op = &ops[op_index++];

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

function_t* function_compile_gradient(const function_t* source, uint32_t gradient_offset) {
    NV_LOG_ERROR("i dont want to think about implementing this");
    return NULL;
}

static void function_op_evaluate(const struct function_op* op, const matrix_t* const* params_data,
                                 matrix_t* output) {
    switch (op->id) {
    case FUNCTION_OP_NOOP:
        NV_LOG_TRACE("noop");
        break;
    case FUNCTION_OP_ZERO:
        assert(op->parameter_count == 0);

        mat_zero(output);
        break;
    case FUNCTION_OP_COPY:
        assert(op->parameter_count == 1);

        mat_copy(output, params_data[0]);
        break;
    case FUNCTION_OP_ADD:
        assert(op->parameter_count == 1);

        mat_add(output, params_data[0], op->flags);
        break;
    case FUNCTION_OP_DOT:
        assert(op->parameter_count == 2);

        mat_mul(output, params_data[0], params_data[1], op->flags);
        break;
    case FUNCTION_OP_RELU:
        assert(op->parameter_count == 1);

        mat_relu(output, params_data[0]);
        break;
    case FUNCTION_OP_SIGMOID:
        assert(op->parameter_count == 1);

        mat_relu(output, params_data[0]);
        break;
    case FUNCTION_OP_SOFTMAX:
        assert(op->parameter_count == 1);

        mat_softmax(output, params_data[0]);
        break;
    case FUNCTION_OP_CROSS_ENTROPY:
        assert(op->parameter_count == 2);

        mat_cross_entropy(output, params_data[0], params_data[1]);
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
        if (params_capacity < op->parameter_count) {
            params_data = nv_realloc(params_data, op->parameter_count * sizeof(const matrix_t*));
            params_capacity = op->parameter_count;
        }

        for (uint32_t i = 0; i < op->parameter_count; i++) {
            const struct function_op_parameter* param = &op->parameters[i];
            switch (param->source) {
            case PARAMETER_SOURCE_DATA:
                params_data[i] = ctx->data[param->index];
                break;
            case PARAMETER_SOURCE_WEIGHTS:
                params_data[i] = ctx->weights[param->index];
                break;
            default:
                params_data[i] = NULL;

                NV_LOG_WARN("invalid parameter source: %u", (uint32_t)param->source);
                break;
            }
        }

        matrix_t* output = ctx->data[op->output_index];
        function_op_evaluate(op, params_data, output);
    }

    nv_free(params_data);
}
