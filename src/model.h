#ifndef _MODEL_H
#define _MODEL_H

#include <stdint.h>
#include <stdbool.h>

/* from matrix.h */
typedef struct matrix matrix_t;

enum {
    LAYER_OP_NONE = 0,
    LAYER_OP_RELU = 1,
    LAYER_OP_SIGMOID = 2,
    LAYER_OP_SOFTMAX = 3,
};

struct model_layer_spec {
    uint32_t op;
    uint32_t size;
};

struct model_layer {
    uint32_t op;
    matrix_t* weights;
    matrix_t* biases;
};

/* from function.h */
typedef struct function function_t;

struct compiled_model {
    function_t* forwardprop;

    function_t* backprop;
    uint32_t gradient_offset;
};

typedef struct model {
    uint32_t num_layers;
    struct model_layer* layers;

    struct compiled_model compiled;
} model_t;

model_t* model_alloc(uint32_t input_size, uint32_t num_layers,
                     const struct model_layer_spec* layers);

void model_free(model_t* model);

/* from prng.h */
struct prng;

void model_randomize(struct prng* rng, model_t* model);

model_t* model_read_from_path(const char* path);
bool model_write_to_path(const model_t* model, const char* path);

#endif
