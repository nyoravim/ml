#include "model.h"

#include "matrix.h"

#include <assert.h>
#include <string.h>
#include <stdio.h>

#include <log.h>

#include <nyoravim/mem.h>

model_t* model_alloc(const struct nv_allocator* alloc, uint32_t input_size, uint32_t num_layers,
                     const struct model_layer_spec* layers) {
    if (num_layers < 1) {
        log_error("each network must have at least 1 layer!");
        return NULL;
    }

    log_trace("allocating model with %u layers", num_layers);
    size_t model_size = sizeof(model_t) + num_layers * sizeof(struct model_layer);

    model_t* model;
    if (alloc) {
        /* allocate model + allocator at end */
        model = alloc->alloc(alloc->user, model_size + sizeof(struct nv_allocator));
        assert(model);

        model->alloc = (void*)model + model_size;
        memcpy(model->alloc, alloc, sizeof(struct nv_allocator));
    } else {
        model = nv_alloc(model_size);
        assert(model);
    }

    model->num_layers = num_layers;
    model->layers = (void*)model + sizeof(model_t);

    for (uint32_t i = 0; i < num_layers; i++) {
        /* layer sizes have the input layer at the front hence the +1 offset */
        uint32_t previous_size = i > 0 ? layers[i - 1].size : input_size;
        uint32_t current_size = layers[i].size;

        struct model_layer* layer = &model->layers[i];
        layer->op = layers[i].op;

        log_debug("layer %u: %u after %u, op %u", i, current_size, previous_size, layer->op);

        layer->biases = mat_alloc(alloc, current_size, 1);
        layer->weights = mat_alloc(alloc, current_size, previous_size);
    }

    return model;
}

void model_free(model_t* model) {
    if (!model) {
        return;
    }

    for (uint32_t i = 0; i < model->num_layers; i++) {
        const struct model_layer* layer = &model->layers[i];
        mat_free(model->alloc, layer->biases);
        mat_free(model->alloc, layer->weights);
    }

    if (!model->alloc) {
        nv_free(model);
    } else {
        struct nv_allocator alloc;
        memcpy(&alloc, model->alloc, sizeof(struct nv_allocator));

        if (alloc.free) {
            alloc.free(alloc.user, model);
        }
    }
}

static void layer_forwardprop(const struct model_layer* layer, const matrix_t* input,
                              struct forwardprop_layer_output* output) {
    /* z_1 = w_1 * a_0 + b_1 */
    mat_copy(output->z, layer->biases);
    mat_mul(output->z, layer->weights, input, 0);

    /* a = A(z) */
    switch (layer->op) {
    case LAYER_OP_RELU:
        mat_relu(output->activations, output->z);
        break;
    case LAYER_OP_SIGMOID:
        mat_relu(output->activations, output->z);
        break;
    case LAYER_OP_SOFTMAX:
        mat_softmax(output->activations, output->z);
        break;
    default:
        if (layer->op != LAYER_OP_NONE) {
            log_warn("unknown layer op %u; assuming LAYER_OP_NONE", layer->op);
        }

        /* copy as is */
        mat_copy(output->activations, output->z);
        break;
    }
}

void model_forwardprop(const model_t* model, const matrix_t* input,
                       struct forwardprop_layer_output* output) {
    assert(input);

    for (uint32_t i = 0; i < model->num_layers; i++) {
        const matrix_t* layer_input = i > 0 ? output[i - 1].activations : input;
        layer_forwardprop(&model->layers[i], layer_input, &output[i]);
    }
}

static bool read_chunk_from_file(FILE* f, void* buffer, size_t size) {
    while (size > 0) {
        size_t bytes_read = fread(buffer, 1, size, f);
        if (bytes_read == 0) {
            /* EOF */
            log_warn("failed to read entire chunk from file! (%zu bytes missing)", size);
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

static model_t* create_model_from_header(const struct nv_allocator* alloc, FILE* f) {
    struct initial_header initial_header;
    if (!read_chunk_from_file(f, &initial_header, sizeof(struct initial_header))) {
        log_error("failed to read initial header from model file!");
        return NULL;
    }

    log_debug("layers: %u", initial_header.layer_count);
    log_debug("input size: %u", initial_header.input_size);

    struct model_layer_spec* layer_specs =
        nv_alloc(initial_header.layer_count * sizeof(struct model_layer_spec));
    assert(layer_specs);

    if (!read_chunk_from_file(f, layer_specs,
                              initial_header.layer_count * sizeof(struct model_layer_spec))) {
        log_error("failed to read layer specs from model file!");

        nv_free(layer_specs);
        return NULL;
    }

    model_t* model =
        model_alloc(alloc, initial_header.input_size, initial_header.layer_count, layer_specs);

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
    log_trace("biases: %ux%u", layer->biases->rows, layer->biases->columns);
    if (!read_matrix_from_file(layer->biases, f)) {
        log_error("failed to read layer biases!");
        return false;
    }

    log_trace("weights: %ux%u", layer->weights->rows, layer->weights->columns);
    if (!read_matrix_from_file(layer->weights, f)) {
        log_error("failed to read layer weights!");
        return false;
    }

    return true;
}

model_t* model_read_from_path(const struct nv_allocator* alloc, const char* path) {
    log_debug("reading model from path: %s", path);

    FILE* f = fopen(path, "rb");
    if (!f) {
        log_error("failed to open model at path: %s", path);
        return NULL;
    }

    model_t* model = create_model_from_header(alloc, f);
    if (!model) {
        log_error("failed to allocate model from file header!");

        fclose(f);
        return NULL;
    }

    for (uint32_t i = 0; i < model->num_layers; i++) {
        log_trace("reading layer %u", i);

        struct model_layer* layer = &model->layers[i];
        if (!read_layer_from_file(layer, f)) {
            log_error("failed to read layer %u from file!", i);
            
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
            log_error("failed to write complete chunk to file!");
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
        log_error("failed to write initial header to file!");
        return false;
    }

    /* layer sizes and operations */
    for (uint32_t i = 0; i < model->num_layers; i++) {
        const struct model_layer* layer = &model->layers[i];

        struct model_layer_spec spec;
        spec.op = layer->op;
        spec.size = layer->weights->rows;

        if (!write_chunk_to_file(f, &spec, sizeof(struct model_layer_spec))) {
            log_error("failed to write layer spec to file!");
            return false;
        }
    }

    /* layer data */
    for (uint32_t i = 0; i < model->num_layers; i++) {
        const struct model_layer* layer = &model->layers[i];

        /* biases before weights */
        if (!write_matrix_to_file(f, layer->biases)) {
            log_error("failed to write layer biases to file!");
            return false;
        }

        if (!write_matrix_to_file(f, layer->weights)) {
            log_error("failed to write layer weights to file!");
            return false;
        }
    }

    return true;
}

bool model_write_to_path(const model_t* model, const char* path) {
    log_debug("writing model to path: %s");

    FILE* f = fopen(path, "wb");
    if (!f) {
        log_error("failed to write to model at path: %s", path);
        return NULL;
    }

    bool success = serialize_model(model, f);
    fclose(f);

    return success;
}
