#include "model.h"

#include "matrix.h"
#include "function.h"

#include <assert.h>
#include <string.h>
#include <stdio.h>

#include <nyoravim/mem.h>
#include <nyoravim/log.h>
#include <nyoravim/arena.h>

struct model_layer {
    layer_op op;
    matrix_t* weights;
    matrix_t* biases;
};

struct compiled_model {
    uint32_t data_matrix_count;

    function_t* forwardprop;
    uint32_t expected_index, input_index, output_index, cost_index;

    function_t* backprop;
    uint32_t gradient_offset;
};

typedef struct model {
    uint32_t num_layers;
    struct model_layer* layers;

    struct compiled_model compiled;
} model_t;

struct op_array {
    nv_arena_t* arena;

    uint32_t capacity, count;
    struct function_op* ops;
};

static void op_array_append(struct op_array* ops, const struct function_op* op) {
    uint32_t index = ops->count++;
    if (ops->count > ops->capacity) {
        if (ops->capacity > 0) {
            ops->capacity *= 2;
        } else {
            /* arbitrary starting capacity */
            ops->capacity = 16;
        }

        ops->ops =
            nv_arena_realloc(ops->arena, ops->ops, ops->capacity * sizeof(struct function_op));
    }

    struct function_op* dst = &ops->ops[index];
    memcpy(dst, op, sizeof(struct function_op));

    size_t params_size = op->parameter_count * sizeof(struct function_op_parameter);
    struct function_op_parameter* params = nv_arena_alloc(ops->arena, params_size);

    memcpy(params, op->parameters, params_size);
    dst->parameters = params;
}

static void get_layer_matrix_indices(uint32_t offset, uint32_t layer, uint32_t* weights,
                                     uint32_t* biases) {
    uint32_t layer_offset = offset + layer * 2; /* weights & biases */

    if (weights) {
        *weights = layer_offset + 0; /* weights are first */
    }

    if (biases) {
        *biases = layer_offset + 1; /* biases are second */
    }
}

static void get_layer_data_indices(uint32_t offset, uint32_t layer, uint32_t* z, uint32_t* a) {
    uint32_t layer_offset = offset + layer * 2; /* z & a */

    if (z) {
        *z = layer_offset + 0; /* z is first */
    }

    if (a) {
        *a = layer_offset + 1; /* a is second */
    }
}

static void compile_layer_forwardprop(uint32_t working_data_offset, struct op_array* ops,
                                      layer_op activation_function, uint32_t layer_index) {
    uint32_t z_1, a_1;
    get_layer_data_indices(working_data_offset, layer_index, &z_1, &a_1);

    /* last data matrix written to is layer input */
    assert(z_1 > 0);
    uint32_t a_0 = z_1 - 1;

    uint32_t w_1, b_1;
    get_layer_matrix_indices(0, layer_index, &w_1, &b_1);

    struct function_op_parameter params[2];
    memset(params, 0, sizeof(params));

    struct function_op op;
    memset(&op, 0, sizeof(struct function_op));
    op.parameters = params;

    /* copy from layer bias to z */
    params[0].source = PARAMETER_SOURCE_WEIGHTS;
    params[0].index = b_1;

    op.id = FUNCTION_OP_COPY;
    op.parameter_count = 1;
    op.output_index = z_1;

    op_array_append(ops, &op);

    /* dot from weights and previous activations and add to z */
    params[0].source = PARAMETER_SOURCE_WEIGHTS;
    params[0].index = w_1;
    params[1].source = PARAMETER_SOURCE_DATA;
    params[1].index = a_0;

    op.id = FUNCTION_OP_DOT;
    op.parameter_count = 2;
    op.output_index = z_1;

    op_array_append(ops, &op);

    /* finally, activation function */
    params[0].source = PARAMETER_SOURCE_DATA;
    params[0].index = z_1;

    op.parameter_count = 1;
    op.output_index = a_1;

    switch (activation_function) {
    case LAYER_OP_NONE:
        op.id = FUNCTION_OP_COPY;
        break;
    case LAYER_OP_RELU:
        op.id = FUNCTION_OP_RELU;
        break;
    case LAYER_OP_SIGMOID:
        op.id = FUNCTION_OP_SIGMOID;
        break;
    case LAYER_OP_SOFTMAX:
        op.id = FUNCTION_OP_SOFTMAX;
        break;
    }

    op_array_append(ops, &op);
}

static void compile_cost_op(uint32_t working_data_offset, struct op_array* ops, model_t* model) {
    assert(working_data_offset > 0);

    model->compiled.cost_index = working_data_offset;
    model->compiled.output_index = working_data_offset - 1; /* last activation matrix */

    struct function_op_parameter params[2];
    params[0].source = PARAMETER_SOURCE_DATA;
    params[0].index = model->compiled.output_index;
    params[1].source = PARAMETER_SOURCE_DATA;
    params[1].index = model->compiled.expected_index;

    struct function_op op;
    memset(&op, 0, sizeof(struct function_op));

    op.id = FUNCTION_OP_CROSS_ENTROPY;
    op.output_index = model->compiled.cost_index;
    op.parameter_count = 2;
    op.parameters = params;

    op_array_append(ops, &op);
}

static void compile_forwardprop(struct op_array* ops, model_t* model) {
    /* first is expected & input matrix */
    model->compiled.expected_index = 0;
    model->compiled.input_index = 1;
    uint32_t working_data_offset = 2;

    /* then go through layers */
    for (uint32_t i = 0; i < model->num_layers; i++) {
        compile_layer_forwardprop(working_data_offset, ops, model->layers[i].op, i);
    }

    working_data_offset += model->num_layers * 2;
    compile_cost_op(working_data_offset, ops, model);

    model->compiled.forwardprop = function_compile(ops->count, ops->ops);
    assert(model->compiled.forwardprop);
}

static void compile_backprop(struct op_array* ops, model_t* model) {
    /* todo: compile backprop? im so tired */
    model->compiled.backprop = NULL;
}

static void compile_model(model_t* model) {
    nv_arena_t* temp = nv_arena_create(4 * 1024 * 1024); /* 4 mb */

    struct op_array ops;
    memset(&ops, 0, sizeof(struct op_array));
    ops.arena = temp;

    compile_forwardprop(&ops, model);

    /* clear by resetting length */
    ops.count = 0;

    compile_forwardprop(&ops, model);

    nv_arena_destroy(temp);
}

model_t* model_alloc(uint32_t input_size, uint32_t num_layers,
                     const struct model_layer_spec* layers) {
    if (num_layers < 1) {
        NV_LOG_ERROR("each network must have at least 1 layer!");
        return NULL;
    }

    NV_LOG_TRACE("allocating model with %u layers", num_layers);
    size_t model_size = sizeof(model_t) + num_layers * sizeof(struct model_layer);

    model_t* model = nv_alloc(model_size);
    assert(model);

    model->num_layers = num_layers;
    model->layers = (void*)model + sizeof(model_t);

    for (uint32_t i = 0; i < num_layers; i++) {
        /* layer sizes have the input layer at the front hence the +1 offset */
        uint32_t previous_size = i > 0 ? layers[i - 1].size : input_size;
        uint32_t current_size = layers[i].size;

        struct model_layer* layer = &model->layers[i];
        layer->op = layers[i].op;

        NV_LOG_DEBUG("layer %u: %u>%u, op %u", i, previous_size, current_size, layer->op);

        layer->biases = mat_alloc(current_size, 1);
        layer->weights = mat_alloc(current_size, previous_size);
    }

    compile_model(model);
    return model;
}

void model_free(model_t* model) {
    if (!model) {
        return;
    }

    for (uint32_t i = 0; i < model->num_layers; i++) {
        const struct model_layer* layer = &model->layers[i];

        mat_free(layer->biases);
        mat_free(layer->weights);
    }

    function_free(model->compiled.forwardprop);
    function_free(model->compiled.backprop);

    nv_free(model);
}

void model_randomize(struct prng* rng, model_t* model) {
    for (uint32_t i = 0; i < model->num_layers; i++) {
        struct model_layer* layer = &model->layers[i];

        mat_randomize(rng, layer->biases);
        mat_randomize(rng, layer->weights);
    }
}

static bool read_chunk_from_file(FILE* f, void* buffer, size_t size) {
    while (size > 0) {
        size_t bytes_read = fread(buffer, 1, size, f);
        if (bytes_read == 0) {
            /* EOF */
            NV_LOG_WARN("failed to read entire chunk from file! (%zu bytes missing)", size);
            return false;
        }

        assert(bytes_read <= size);

        buffer += bytes_read;
        size -= bytes_read;
    }

    return true;
}

struct initial_header {
    uint32_t layer_count;
    uint32_t input_size;
};

static model_t* create_model_from_header(FILE* f) {
    struct initial_header initial_header;
    if (!read_chunk_from_file(f, &initial_header, sizeof(struct initial_header))) {
        NV_LOG_ERROR("failed to read initial header from model file!");
        return NULL;
    }

    NV_LOG_DEBUG("layers: %u", initial_header.layer_count);
    NV_LOG_DEBUG("input size: %u", initial_header.input_size);

    struct model_layer_spec* layer_specs =
        nv_alloc(initial_header.layer_count * sizeof(struct model_layer_spec));
    assert(layer_specs);

    if (!read_chunk_from_file(f, layer_specs,
                              initial_header.layer_count * sizeof(struct model_layer_spec))) {
        NV_LOG_ERROR("failed to read layer specs from model file!");

        nv_free(layer_specs);
        return NULL;
    }

    model_t* model =
        model_alloc(initial_header.input_size, initial_header.layer_count, layer_specs);

    nv_free(layer_specs);
    if (!model) {
        return NULL;
    }

    return model;
}

static bool read_matrix_from_file(matrix_t* mat, FILE* f) {
    size_t total_size = sizeof(float) * mat->rows * mat->columns;
    return read_chunk_from_file(f, mat->data, total_size);
}

static bool read_layer_from_file(struct model_layer* layer, FILE* f) {
    /* biases before weights */
    NV_LOG_TRACE("biases: %ux%u", layer->biases->rows, layer->biases->columns);
    if (!read_matrix_from_file(layer->biases, f)) {
        NV_LOG_ERROR("failed to read layer biases!");
        return false;
    }

    NV_LOG_TRACE("weights: %ux%u", layer->weights->rows, layer->weights->columns);
    if (!read_matrix_from_file(layer->weights, f)) {
        NV_LOG_ERROR("failed to read layer weights!");
        return false;
    }

    return true;
}

model_t* model_read_from_path(const char* path) {
    NV_LOG_DEBUG("reading model from path: %s", path);

    FILE* f = fopen(path, "rb");
    if (!f) {
        NV_LOG_ERROR("failed to open model at path: %s", path);
        return NULL;
    }

    model_t* model = create_model_from_header(f);
    if (!model) {
        NV_LOG_ERROR("failed to allocate model from file header!");

        fclose(f);
        return NULL;
    }

    for (uint32_t i = 0; i < model->num_layers; i++) {
        NV_LOG_TRACE("reading layer %u", i);

        struct model_layer* layer = &model->layers[i];
        if (!read_layer_from_file(layer, f)) {
            NV_LOG_ERROR("failed to read layer %u from file!", i);

            fclose(f);
            model_free(model);

            return NULL;
        }
    }

    return model;
}

static bool write_chunk_to_file(FILE* f, const void* data, size_t size) {
    while (size > 0) {
        size_t bytes_written = fwrite(data, 1, size, f);
        if (bytes_written == 0) {
            NV_LOG_ERROR("failed to write complete chunk to file!");
            return false;
        }

        assert(bytes_written <= size);

        data += bytes_written;
        size -= bytes_written;
    }

    return true;
}

static bool write_matrix_to_file(FILE* f, const matrix_t* mat) {
    size_t total_size = sizeof(float) * mat->rows * mat->columns;
    return write_chunk_to_file(f, mat->data, total_size);
}

static bool serialize_model(const model_t* model, FILE* f) {
    assert(model->num_layers > 0);

    /* initial header data */
    struct initial_header initial_header;
    initial_header.layer_count = model->num_layers;
    initial_header.input_size = model->layers[0].weights->columns;

    if (!write_chunk_to_file(f, &initial_header, sizeof(struct initial_header))) {
        NV_LOG_ERROR("failed to write initial header to file!");
        return false;
    }

    /* layer sizes and operations */
    for (uint32_t i = 0; i < model->num_layers; i++) {
        const struct model_layer* layer = &model->layers[i];

        struct model_layer_spec spec;
        spec.op = layer->op;
        spec.size = layer->weights->rows;

        if (!write_chunk_to_file(f, &spec, sizeof(struct model_layer_spec))) {
            NV_LOG_ERROR("failed to write layer spec to file!");
            return false;
        }
    }

    /* layer data */
    for (uint32_t i = 0; i < model->num_layers; i++) {
        const struct model_layer* layer = &model->layers[i];

        /* biases before weights */
        if (!write_matrix_to_file(f, layer->biases)) {
            NV_LOG_ERROR("failed to write layer biases to file!");
            return false;
        }

        if (!write_matrix_to_file(f, layer->weights)) {
            NV_LOG_ERROR("failed to write layer weights to file!");
            return false;
        }
    }

    return true;
}

bool model_write_to_path(const model_t* model, const char* path) {
    NV_LOG_DEBUG("writing model to path: %s", path);

    FILE* f = fopen(path, "wb");
    if (!f) {
        NV_LOG_ERROR("failed to write to model at path: %s", path);
        return NULL;
    }

    bool success = serialize_model(model, f);
    fclose(f);

    return success;
}
