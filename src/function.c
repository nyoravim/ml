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

void function_evaluate(const function_t* func, const struct function_context* ctx) {
    NV_LOG_ERROR("todo implement");
}
