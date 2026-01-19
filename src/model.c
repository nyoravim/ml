#include "model.h"

#include "matrix.h"

#include <assert.h>
#include <string.h>

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
