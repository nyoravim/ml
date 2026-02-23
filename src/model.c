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

/*
 * layout:
 *  data matrices:
 *   - expected values
 *   - input data
 *   - layers:
 *    - z values
 *    - activations
 *   - cost vector
 *   - delta gradients:
 *    - weight matrix
 *    - bias vector
 *   - activation gradients
 *   - cost to last activation gradient
 *  weight matrices:
 *   - pairs of weight & bias matrices
 */
struct compiled_model {
    uint32_t data_matrix_count;

    function_t* forwardprop;
    uint32_t activations_offset;
    uint32_t expected_index, input_index, output_index, cost_index;

    function_t* backprop;
    uint32_t gradient_offset, activation_gradient_offset;
    uint32_t cost_gradient_index;
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

static void compile_forwardprop_layer(uint32_t working_data_offset, struct op_array* ops,
                                      layer_op activation_function, uint32_t layer_index) {
    NV_LOG_TRACE("compiling forwardprop layer %u", layer_index);

    uint32_t z_1, a_1;
    get_layer_data_indices(working_data_offset, layer_index, &z_1, &a_1);
    NV_LOG_TRACE("z_1: %u; a_1: %u", z_1, a_1);

    /* last data matrix written to is layer input */
    assert(z_1 > 0);
    uint32_t a_0 = z_1 - 1;

    uint32_t w_1, b_1;
    get_layer_matrix_indices(0, layer_index, &w_1, &b_1);
    NV_LOG_TRACE("w_1: %u; b_1: %u", w_1, b_1);

    struct function_op_parameter params[2];
    memset(params, 0, sizeof(params));

    struct function_op op;
    memset(&op, 0, sizeof(struct function_op));
    op.parameters = params;

    /* copy from layer bias to z */
    params[0].source = PARAMETER_SOURCE_WEIGHTS;
    params[0].index = b_1;

    op.label = "z_0 = b";
    op.id = FUNCTION_OP_COPY;
    op.parameter_count = 1;
    op.output_index = z_1;

    op_array_append(ops, &op);

    /* dot from weights and previous activations and add to z */
    params[0].source = PARAMETER_SOURCE_WEIGHTS;
    params[0].index = w_1;
    params[1].source = PARAMETER_SOURCE_DATA;
    params[1].index = a_0;

    op.label = "z = w * a_prev + z_0";
    op.id = FUNCTION_OP_DOT;
    op.parameter_count = 2;
    op.output_index = z_1;

    op_array_append(ops, &op);

    /* finally, activation function */
    params[0].source = PARAMETER_SOURCE_DATA;
    params[0].index = z_1;

    op.label = "activation function";
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

static void compile_cost_op(struct op_array* ops, model_t* model) {
    NV_LOG_TRACE("compiling cost op");

    struct function_op_parameter params[2];
    params[0].source = PARAMETER_SOURCE_DATA;
    params[0].index = model->compiled.output_index;
    params[1].source = PARAMETER_SOURCE_DATA;
    params[1].index = model->compiled.expected_index;

    struct function_op op;
    memset(&op, 0, sizeof(struct function_op));

    op.label = "cost function";
    op.id = FUNCTION_OP_CROSS_ENTROPY;
    op.output_index = model->compiled.cost_index;
    op.parameter_count = 2;
    op.parameters = params;

    op_array_append(ops, &op);
}

static void compile_forwardprop(struct op_array* ops, model_t* model) {
    NV_LOG_TRACE("compiling forwardprop function");

    /* first is expected & input matrix */
    model->compiled.expected_index = 0;
    model->compiled.input_index = 1;

    model->compiled.activations_offset = 2;
    NV_LOG_TRACE("activations start at %u", model->compiled.activations_offset);

    /* then go through layers */
    for (uint32_t i = 0; i < model->num_layers; i++) {
        compile_forwardprop_layer(model->compiled.activations_offset, ops, model->layers[i].op, i);
    }

    get_layer_data_indices(model->compiled.activations_offset, model->num_layers - 1, NULL,
                           &model->compiled.output_index);

    model->compiled.cost_index = model->compiled.activations_offset + model->num_layers * 2;
    compile_cost_op(ops, model);

    NV_LOG_TRACE("cost index: %u", model->compiled.cost_index);

    model->compiled.data_matrix_count = model->compiled.cost_index + 1;
    NV_LOG_TRACE("data matrices used after forwardprop compilation: %u",
                 model->compiled.data_matrix_count);

    model->compiled.forwardprop = function_compile(ops->count, ops->ops);
    assert(model->compiled.forwardprop);
}

static void compute_final_activation_gradient(struct op_array* ops, model_t* model) {
    /* the very last activation gradient */
    model->compiled.cost_gradient_index =
        model->compiled.activation_gradient_offset + model->num_layers - 1;

    /* actual, expected */
    struct function_op_parameter params[2];
    memset(params, 0, sizeof(struct function_op_parameter) * 2);

    assert(model->compiled.cost_index > 0);

    /* actual output (right before cost vector) */
    params[0].index = model->compiled.cost_index - 1;
    params[0].source = PARAMETER_SOURCE_DATA;

    /* expected output (labeled by dataset) */
    params[1].index = model->compiled.expected_index;
    params[1].source = PARAMETER_SOURCE_DATA;

    struct function_op op;
    memset(&op, 0, sizeof(struct function_op));

    op.label = "cost function gradient";
    op.id = FUNCTION_OP_CROSS_ENTROPY_GRADIENT;
    op.parameter_count = 2;
    op.parameters = params;

    /* last activation gradient */
    op.output_index = model->compiled.cost_gradient_index;

    op_array_append(ops, &op);
}

static uint32_t compute_activation_gradient(struct op_array* ops, const model_t* model,
                                            uint32_t layer_index) {
    assert(layer_index < model->num_layers - 1);

    uint32_t next_layer = layer_index + 1;

    uint32_t w_2;
    get_layer_matrix_indices(0, next_layer, &w_2, NULL);

    /* bias gradient is dc/dz_2 */
    uint32_t dc_dz_2;
    get_layer_matrix_indices(model->compiled.gradient_offset, next_layer, NULL, &dc_dz_2);

    struct function_op_parameter params[2];
    memset(params, 0, sizeof(struct function_op_parameter) * 2);

    params[0].index = w_2;
    params[0].source = PARAMETER_SOURCE_WEIGHTS;

    params[1].index = dc_dz_2;
    params[1].source = PARAMETER_SOURCE_DATA;

    uint32_t gradient_index = model->compiled.activation_gradient_offset + layer_index;

    struct function_op op;
    memset(&op, 0, sizeof(struct function_op));

    op.label = "activation gradient";
    op.id = FUNCTION_OP_DOT;
    op.output_index = gradient_index;
    op.parameter_count = 2;
    op.parameters = params;

    /* transpose to map columns to next row, and rows to current row */
    op.flags = MAT_ZERO_RESULT | MAT_TRANSPOSE_LHS;

    op_array_append(ops, &op);
    return gradient_index;
}

static void compile_backprop_layer(struct op_array* ops, model_t* model, uint32_t layer_index) {
    NV_LOG_TRACE("compiling backprop layer %u", layer_index);

    uint32_t dc_da_1;
    if (layer_index < model->num_layers - 1) {
        dc_da_1 = compute_activation_gradient(ops, model, layer_index);
    } else {
        compute_final_activation_gradient(ops, model);
        dc_da_1 = model->compiled.cost_gradient_index;
    }

    NV_LOG_TRACE("dc/da_1: %u", dc_da_1);

    uint32_t z_1;
    get_layer_data_indices(model->compiled.activations_offset, layer_index, &z_1, NULL);
    NV_LOG_TRACE("z_1: %u", z_1);

    struct function_op_parameter params[2];
    memset(params, 0, sizeof(struct function_op_parameter) * 2);

    /* z values */
    params[0].index = z_1;
    params[1].source = PARAMETER_SOURCE_DATA;

    /* existing gradient */
    params[1].index = dc_da_1;
    params[1].source = PARAMETER_SOURCE_DATA;

    struct function_op op;
    memset(&op, 0, sizeof(struct function_op));

    uint32_t dc_dw_1, dc_db_1;
    get_layer_matrix_indices(model->compiled.gradient_offset, layer_index, &dc_dw_1, &dc_db_1);
    NV_LOG_TRACE("dc/dw_1: %u; dc/db_1: %u", dc_dw_1, dc_db_1);

    /* identical matrices; biases are just offsets */
    uint32_t dc_dz_1 = dc_db_1;

    op.label = "biases gradient";
    op.output_index = dc_dz_1;
    op.parameter_count = 2;
    op.parameters = params;

    switch (model->layers[layer_index].op) {
    case LAYER_OP_NONE:
        /* da/dz is an identity */
        op.id = FUNCTION_OP_COPY;

        op.parameter_count = 1;
        op.parameters = &params[1];

        break;
    case LAYER_OP_RELU:
        op.id = FUNCTION_OP_RELU_GRADIENT;
        break;
    case LAYER_OP_SIGMOID:
        op.id = FUNCTION_OP_SIGMOID_GRADIENT;
        break;
    case LAYER_OP_SOFTMAX:
        op.id = FUNCTION_OP_SOFTMAX_GRADIENT;
        break;
    }

    op_array_append(ops, &op);

    /* dc/dw = dc/dz * dz/dw
     * dz/dw = a_0 (kind of)
     * dc/dw = dc/dz * transpose(a_0) */

    /* if first layer, will grab input
     * otherwise, will grab previous activations */
    uint32_t a_0 = z_1 - 1;

    params[0].index = dc_dz_1;
    params[0].source = PARAMETER_SOURCE_DATA;

    params[1].index = a_0;
    params[1].source = PARAMETER_SOURCE_DATA;

    op.label = "weights gradient";
    op.id = FUNCTION_OP_DOT;
    op.flags = MAT_ZERO_RESULT | MAT_TRANSPOSE_RHS;
    op.parameter_count = 2;
    op.parameters = params;
    op.output_index = dc_dw_1;

    op_array_append(ops, &op);
}

static void compile_backprop(struct op_array* ops, model_t* model) {
    NV_LOG_TRACE("compiling backprop function");

    model->compiled.gradient_offset = model->compiled.cost_index + 1;
    model->compiled.activation_gradient_offset =
        model->compiled.gradient_offset + model->num_layers * 2;

    NV_LOG_TRACE("output gradients start at %u", model->compiled.gradient_offset);
    NV_LOG_TRACE("activation gradients start at %u", model->compiled.activation_gradient_offset);

    for (uint32_t i = 0; i < model->num_layers; i++) {
        uint32_t layer_index = model->num_layers - (i + 1);
        compile_backprop_layer(ops, model, layer_index);
    }

    /* cost gradient matrix is last data matrix */
    model->compiled.data_matrix_count = model->compiled.cost_gradient_index + 1;
    NV_LOG_TRACE("%u matrices used after backprop compilation", model->compiled.data_matrix_count);

    model->compiled.backprop = function_compile(ops->count, ops->ops);
    assert(model->compiled.backprop);
}

static void compile_model(model_t* model) {
    NV_LOG_TRACE("compiling model (%u layers)", model->num_layers);
    nv_arena_t* temp = nv_arena_create(4 * 1024 * 1024); /* 4 mb */

    struct op_array ops;
    memset(&ops, 0, sizeof(struct op_array));
    ops.arena = temp;

    compile_forwardprop(&ops, model);

    /* clear by resetting length */
    ops.count = 0;

    /* todo: add nv_arena_reset or something idk */

    compile_backprop(&ops, model);

    NV_LOG_DEBUG("%u-layer model compiled", model->num_layers);
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
    assert(model);

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

uint32_t model_get_layer_count(const model_t* model) { return model->num_layers; }
uint32_t model_get_input_count(const model_t* model) { return model->layers[0].weights->columns; }

static matrix_t* arena_allocate_matrix(nv_arena_t* arena, uint32_t rows, uint32_t columns) {
    assert(arena);

    size_t meta_size = sizeof(matrix_t);
    size_t data_size = sizeof(float) * rows * columns;

    matrix_t* mat = nv_arena_alloc(arena, meta_size + data_size);
    if (!mat) {
        return NULL;
    }

    mat->rows = rows;
    mat->columns = columns;
    mat->data = (void*)mat + meta_size;

    return mat;
}

typedef struct model_gradients {
    uint32_t num_layers;
    matrix_t** biases;
    matrix_t** weights;
} model_gradients_t;

model_gradients_t* model_gradients_alloc(nv_arena_t* arena, const model_t* model) {
    assert(arena && model);

    model_gradients_t* gradients = nv_arena_alloc(arena, sizeof(model_gradients_t));
    assert(gradients);

    gradients->num_layers = model->num_layers;
    gradients->weights = nv_arena_alloc(arena, sizeof(matrix_t*) * gradients->num_layers);
    gradients->biases = nv_arena_alloc(arena, sizeof(matrix_t*) * gradients->num_layers);

    for (uint32_t i = 0; i < gradients->num_layers; i++) {
        const struct model_layer* layer = &model->layers[i];
        uint32_t layer_size = layer->weights->rows;
        uint32_t prev_size = layer->weights->columns;

        gradients->weights[i] = arena_allocate_matrix(arena, layer_size, prev_size);
        gradients->biases[i] = arena_allocate_matrix(arena, layer_size, 1);
    }

    return gradients;
}

void model_gradients_zero(model_gradients_t* gradients) {
    for (uint32_t i = 0; i < gradients->num_layers; i++) {
        mat_zero(gradients->weights[i]);
        mat_zero(gradients->biases[i]);
    }
}

void model_gradients_flush(const model_gradients_t* gradients, float learning_rate,
                           uint32_t batch_size, model_t* model) {
    assert(gradients->num_layers == model->num_layers);

    /*
     * negative: gradient descent, not ascent
     * learning_rate: we dont want to take big clumsy steps
     * batch_size: average across batch
     */

    float scalar = -learning_rate / batch_size;
    for (uint32_t i = 0; i < model->num_layers; i++) {
        struct model_layer* layer = &model->layers[i];

        mat_add_scaled(layer->weights, gradients->weights[i], scalar, 0);
        mat_add_scaled(layer->biases, gradients->biases[i], scalar, 0);
    }
}

uint32_t model_get_output_count(const model_t* model) {
    uint32_t last_layer = model->num_layers - 1;
    return model->layers[last_layer].weights->rows;
}

typedef struct eval_context {
    nv_arena_t* arena;

    model_t* model;
    uint32_t level;

    matrix_t** data_matrices;
    struct function_context function_context;
} eval_context_t;

static void map_weight_matrices(eval_context_t* ctx) {
    assert(ctx->model);

    size_t ptr_array_size = sizeof(const matrix_t*) * ctx->model->num_layers * 2;
    const matrix_t** matrix_ptrs = nv_arena_alloc(ctx->arena, ptr_array_size);
    assert(matrix_ptrs);

    for (uint32_t i = 0; i < ctx->model->num_layers; i++) {
        const struct model_layer* layer = &ctx->model->layers[i];

        uint32_t weights, biases;
        get_layer_matrix_indices(0, i, &weights, &biases);

        matrix_ptrs[biases] = layer->biases;
        matrix_ptrs[weights] = layer->weights;
    }

    ctx->function_context.weights = matrix_ptrs;
}

/* see struct compiled_model for data layout */

static void allocate_forwardprop_matrices(eval_context_t* ctx) {
    assert(ctx->model->compiled.forwardprop && ctx->data_matrices);

    uint32_t output_count = ctx->model->layers[ctx->model->num_layers - 1].biases->rows;
    matrix_t* expected_matrix = arena_allocate_matrix(ctx->arena, output_count, 1);

    uint32_t input_count = ctx->model->layers[0].weights->columns;
    matrix_t* input_matrix = arena_allocate_matrix(ctx->arena, input_count, 1);

    const struct compiled_model* compiled = &ctx->model->compiled;
    ctx->data_matrices[compiled->expected_index] = expected_matrix;
    ctx->data_matrices[compiled->input_index] = input_matrix;

    for (uint32_t i = 0; i < ctx->model->num_layers; i++) {
        uint32_t layer_size = ctx->model->layers[i].biases->rows;

        matrix_t* z = arena_allocate_matrix(ctx->arena, layer_size, 1);
        matrix_t* a = arena_allocate_matrix(ctx->arena, layer_size, 1);

        uint32_t z_index, a_index;
        get_layer_data_indices(ctx->model->compiled.activations_offset, i, &z_index, &a_index);

        ctx->data_matrices[z_index] = z;
        ctx->data_matrices[a_index] = a;
    }

    matrix_t* cost_matrix = arena_allocate_matrix(ctx->arena, output_count, 1);
    ctx->data_matrices[ctx->model->compiled.cost_index] = cost_matrix;
}

static void allocate_backprop_matrices(eval_context_t* ctx) {
    assert(ctx->model->compiled.backprop && ctx->data_matrices);

    for (uint32_t i = 0; i < ctx->model->num_layers; i++) {
        const struct model_layer* layer = &ctx->model->layers[i];

        uint32_t layer_size = layer->weights->rows;
        uint32_t previous_size = layer->weights->columns;

        matrix_t* weights_gradient = arena_allocate_matrix(ctx->arena, layer_size, previous_size);
        matrix_t* biases_gradient = arena_allocate_matrix(ctx->arena, layer_size, 1);
        matrix_t* activation_gradient = arena_allocate_matrix(ctx->arena, layer_size, 1);

        uint32_t weights_index, biases_index;
        get_layer_matrix_indices(ctx->model->compiled.gradient_offset, i, &weights_index,
                                 &biases_index);

        ctx->data_matrices[weights_index] = weights_gradient;
        ctx->data_matrices[biases_index] = biases_gradient;

        uint32_t activation_gradient_index = ctx->model->compiled.activation_gradient_offset + i;
        ctx->data_matrices[activation_gradient_index] = activation_gradient;
    }

    uint32_t output_count = ctx->model->layers[ctx->model->num_layers - 1].weights->rows;
    matrix_t* cost_gradient = arena_allocate_matrix(ctx->arena, output_count, 1);

    ctx->data_matrices[ctx->model->compiled.cost_gradient_index] = cost_gradient;
}

static void allocate_data_matrices(eval_context_t* ctx) {
    assert(ctx->model);

    size_t data_array_size = ctx->model->compiled.data_matrix_count * sizeof(matrix_t*);
    ctx->data_matrices = nv_arena_alloc(ctx->arena, data_array_size);
    assert(ctx->data_matrices);

    if (ctx->level >= EVAL_LEVEL_EVAL) {
        allocate_forwardprop_matrices(ctx);
    }

    if (ctx->level >= EVAL_LEVEL_BACKPROP) {
        allocate_backprop_matrices(ctx);
    }

    ctx->function_context.data = ctx->data_matrices;
}

eval_context_t* eval_context_allocate(nv_arena_t* arena, model_t* model, uint32_t level) {
    if (!model) {
        NV_LOG_ERROR("no model passed to eval_context_allocate");
        return NULL;
    }

    eval_context_t* ctx = nv_arena_alloc(arena, sizeof(eval_context_t));
    assert(ctx);

    ctx->arena = arena;
    ctx->model = model;
    ctx->level = level;

    map_weight_matrices(ctx);
    allocate_data_matrices(ctx);

    return ctx;
}

uint32_t eval_context_get_level(const eval_context_t* ctx) {
    if (!ctx) {
        return EVAL_LEVEL_NONE;
    }

    return ctx->level;
}

void eval_context_set_input(eval_context_t* ctx, const matrix_t* input) {
    if (ctx->level < EVAL_LEVEL_EVAL) {
        return;
    }

    uint32_t input_index = ctx->model->compiled.input_index;
    matrix_t* input_matrix = ctx->data_matrices[input_index];

    mat_copy(input_matrix, input);
}

void eval_context_set_expected(eval_context_t* ctx, const matrix_t* expected) {
    if (ctx->level < EVAL_LEVEL_EVAL) {
        return;
    }

    uint32_t expected_index = ctx->model->compiled.expected_index;
    matrix_t* expected_matrix = ctx->data_matrices[expected_index];

    mat_copy(expected_matrix, expected);
}

const matrix_t* eval_context_get_output(const eval_context_t* ctx) {
    if (ctx->level < EVAL_LEVEL_EVAL) {
        return NULL;
    }

    uint32_t index;
    get_layer_data_indices(ctx->model->compiled.activations_offset, ctx->model->num_layers - 1,
                           NULL, &index);

    return ctx->data_matrices[index];
}

const matrix_t* eval_context_get_cost(const eval_context_t* ctx) {
    if (ctx->level < EVAL_LEVEL_EVAL) {
        return NULL;
    }

    uint32_t cost_index = ctx->model->compiled.cost_index;
    return ctx->data_matrices[cost_index];
}

bool eval_context_add_gradients(const eval_context_t* ctx, model_gradients_t* gradients) {
    if (ctx->level < EVAL_LEVEL_BACKPROP) {
        return false;
    }

    assert(gradients->num_layers == ctx->model->num_layers);
    for (uint32_t i = 0; i < ctx->model->num_layers; i++) {
        uint32_t weights_index, biases_index;
        get_layer_matrix_indices(ctx->model->compiled.gradient_offset, i, &weights_index,
                                 &biases_index);

        const matrix_t* weights_gradient = ctx->data_matrices[weights_index];
        mat_add(gradients->weights[i], weights_gradient, 0);

        const matrix_t* biases_gradient = ctx->data_matrices[biases_index];
        mat_add(gradients->biases[i], biases_gradient, 0);
    }

    return true;
}

void eval_context_eval(eval_context_t* ctx) {
    NV_LOG_TRACE("evaluating model");

    if (ctx->level >= EVAL_LEVEL_EVAL) {
        NV_LOG_TRACE("evaluating model forwardprop");
        function_evaluate(ctx->model->compiled.forwardprop, &ctx->function_context);
    }

    if (ctx->level >= EVAL_LEVEL_BACKPROP) {
        NV_LOG_TRACE("evaluating model backprop");
        function_evaluate(ctx->model->compiled.backprop, &ctx->function_context);
    }
}
