#ifndef _MODEL_H
#define _MODEL_H

#include <stdint.h>
#include <stdbool.h>

/* from matrix.h */
typedef struct matrix matrix_t;

typedef enum {
    LAYER_OP_NONE = 0,
    LAYER_OP_RELU = 1,
    LAYER_OP_SIGMOID = 2,
    LAYER_OP_SOFTMAX = 3,
} layer_op;

struct model_layer_spec {
    layer_op op;
    uint32_t size;
};

typedef struct model model_t;
typedef struct model_gradients model_gradients_t;
typedef struct eval_context eval_context_t;

model_t* model_alloc(uint32_t input_size, uint32_t num_layers,
                     const struct model_layer_spec* layers);

void model_free(model_t* model);

/* from prng.h */
struct prng;

void model_randomize(struct prng* rng, model_t* model);

model_t* model_read_from_path(const char* path);
bool model_write_to_path(const model_t* model, const char* path);

uint32_t model_get_layer_count(const model_t* model);

uint32_t model_get_input_count(const model_t* model);
uint32_t model_get_output_count(const model_t* model);

/* from nyoravim/arena.h */
typedef struct nv_arena nv_arena_t;

model_gradients_t* model_gradients_alloc(nv_arena_t* arena, const model_t* model);

void model_gradients_zero(model_gradients_t* gradients);
void model_gradients_flush(const model_gradients_t* gradients, float learning_rate,
                           uint32_t batch_size, model_t* model);

enum {
    /* eval does nothing */
    EVAL_LEVEL_NONE = 0,

    /* eval only evaluates neural net (forwardpropagation) */
    EVAL_LEVEL_EVAL = 1,

    /* eval runs backpropagation */
    EVAL_LEVEL_BACKPROP = 2,
};

eval_context_t* eval_context_allocate(nv_arena_t* arena, model_t* model, uint32_t level);

uint32_t eval_context_get_level(const eval_context_t* ctx);

void eval_context_set_input(eval_context_t* ctx, const matrix_t* input);
void eval_context_set_expected(eval_context_t* ctx, const matrix_t* expected);

const matrix_t* eval_context_get_output(const eval_context_t* ctx);
const matrix_t* eval_context_get_cost(const eval_context_t* ctx);

bool eval_context_add_gradients(const eval_context_t* ctx, model_gradients_t* gradients);

void eval_context_eval(eval_context_t* ctx);

#endif
