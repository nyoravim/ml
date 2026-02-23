#include "matrix.h"
#include "model.h"
#include "trainer.h"

#include "data/dataset.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <nyoravim/mem.h>
#include <nyoravim/map.h>
#include <nyoravim/log.h>
#include <nyoravim/util.h>
#include <nyoravim/list.h>

/* for access(2) */
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include <tickit.h>

#include "ui/layout.h"
#include "ui/training_menu.h"

static void draw_matrix(const matrix_t* mat) {
    /* over rows */
    for (uint32_t y = 0; y < mat->rows; y++) {

        /* over columns */
        for (uint32_t x = 0; x < mat->columns; x++) {
            float value = mat->data[y * mat->columns + x];

            /* 24 steps w/ offset of 232. see
             * https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797 */
            uint8_t color = 232 + (uint8_t)(value * 23.f);

            /* two spaces to make the pixel fairly square */
            printf("\x1b[48;5;%hhum  ", color);
        }

        /* next row */
        printf("\n");
    }

    /* reset output style */
    printf("\x1b[0m");
}

enum {
    DATASET_TRAINING = 0,
    DATASET_TESTING,

    DATASET_COUNT,
};

static struct dataset* load_dataset_by_id(uint32_t id) {
    const char* labels;
    const char* images;
    const char* name;

    switch (id) {
    case DATASET_TRAINING:
        labels = "data/train-labels-idx1-ubyte.gz";
        images = "data/train-images-idx3-ubyte.gz";
        name = "training";

        break;
    case DATASET_TESTING:
        labels = "data/t10k-labels-idx1-ubyte.gz";
        images = "data/t10k-images-idx3-ubyte.gz";
        name = "testing";

        break;
    default:
        NV_LOG_WARN("invalid dataset id: %u", id);
        return NULL;
    }

    NV_LOG_DEBUG("loading %s dataset", name);

    struct dataset* data = dataset_load(labels, images);
    if (!data) {
        NV_LOG_ERROR("failed to load %s dataset!", name);
        return NULL;
    }

    NV_LOG_INFO("loaded %s dataset", name);
    return data;
}

static void free_dataset(void* user, void* value) { dataset_free(value); }

static nv_map_t* load_datasets() {
    NV_LOG_TRACE("loading datasets");

    struct nv_map_callbacks callbacks;
    memset(&callbacks, 0, sizeof(struct nv_map_callbacks));

    callbacks.free_value = free_dataset;

    nv_map_t* datasets = nv_map_alloc(8, &callbacks);
    assert(datasets);

    for (uint32_t id = 0; id < DATASET_COUNT; id++) {
        struct dataset* data = load_dataset_by_id(id);
        if (!data) {
            continue;
        }

        nv_map_insert(datasets, (void*)(size_t)id, data);
    }

    return datasets;
}

static bool is_file_writable(const char* path) {
    int ret = access(path, W_OK);
    return ret == 0 || errno != EACCES;
}

static bool file_exists(const char* path) {
    int ret = access(path, F_OK);
    return ret == 0 || errno != ENOENT;
}

static model_t* create_model(const char* path) {
    if (!is_file_writable(path)) {
        NV_LOG_ERROR("cannot write to path %s; aborting", path);
        return NULL;
    }

    static const uint32_t layer_count = 3;
    struct model_layer_spec layers[layer_count];

    layers[0].op = LAYER_OP_SIGMOID;
    layers[0].size = 128;

    layers[1].op = LAYER_OP_SIGMOID;
    layers[1].size = 64;

    layers[2].op = LAYER_OP_SOFTMAX;
    layers[2].size = 10;

    NV_LOG_DEBUG("manually allocating model with %u layers", layer_count);

    model_t* model = model_alloc(28 * 28, layer_count, layers);
    if (!model) {
        NV_LOG_ERROR("failed to manually allocate model!");
        return NULL;
    }

    NV_LOG_TRACE("randomizing model");
    model_randomize(NULL, model);

    if (!model_write_to_path(model, path)) {
        NV_LOG_ERROR("failed to write model to path %s", path);

        model_free(model);
        return NULL;
    }

    return model;
}

static model_t* open_model(const char* path) {
    if (file_exists(path)) {
        NV_LOG_INFO("file %s exists; reading", path);
        return model_read_from_path(path);
    } else {
        NV_LOG_INFO("file %s does not exist; creating new model and writing", path);
        return create_model(path);
    }
}

static void print_help(const char* program) { printf("usage: %s [model path]\n", program); }

struct model_context {
    nv_map_t* datasets;
    trainer_t* trainer;

    model_t* model;
    const char* model_path;
};

static void cleanup_context(const struct model_context* ctx) {
    nv_map_free(ctx->datasets);
    model_free(ctx->model);
}

static int render_test(TickitWindow* win, TickitEventFlags flags, void* info, void* data) {
    TickitExposeEventInfo* expose = info;
    TickitRenderBuffer* rb = expose->rb;

    tickit_renderbuffer_goto(rb, 0, 0);
    tickit_renderbuffer_textf(rb, "Hello %u!", (uint32_t)(size_t)data);

    return 1;
}

int main(int argc, const char** argv) {
    if (argc > 1 && strcmp(argv[1], "--help") == 0) {
        print_help(argv[0]);
    }

    struct nv_logger_sink stdout_sink;
    nv_create_stdout_sink(&stdout_sink);
    stdout_sink.level = NV_LOG_LEVEL_TRACE;

    struct nv_logger logger;
    logger.level = NV_LOG_LEVEL_TRACE;
    logger.sink_count = 0; /* temp */
    logger.sinks = &stdout_sink;

    nv_set_default_logger(&logger);

    struct model_context ctx;
    memset(&ctx, 0, sizeof(struct model_context));

    ctx.datasets = load_datasets();
    if (nv_map_size(ctx.datasets) < DATASET_COUNT) {
        cleanup_context(&ctx);
        return 1;
    }

    ctx.model_path = argc > 1 ? argv[1] : "model.bin";
    ctx.model = open_model(ctx.model_path);

    struct trainer_spec trainer_spec;
    trainer_spec.batch_size = 100;
    trainer_spec.learning_rate = 0.1f;

    nv_map_get(ctx.datasets, (void*)DATASET_TRAINING, (void**)&trainer_spec.training_data);
    nv_map_get(ctx.datasets, (void*)DATASET_TESTING, (void**)&trainer_spec.test_data);

    ctx.trainer = trainer_new(ctx.model, ctx.model_path, &trainer_spec);

    Tickit* t = tickit_new_stdtty();
    if (!t) {
        cleanup_context(&ctx);
        return 1;
    }

    TickitWindow* root = tickit_get_rootwin(t);
    if (!root) {
        cleanup_context(&ctx);
        return 1;
    }

    struct layout_spec spec[2];
    spec[0].split = tickit_window_cols(root) / 2;
    spec[0].type = LAYOUT_HORIZONTAL;
    spec[1].split = tickit_window_lines(root) / 2;
    spec[1].type = LAYOUT_VERTICAL;

    struct layout* layouts[2];
    layouts[0] = layout_create(root, &spec[0]);
    layouts[1] = layout_create(layouts[0]->children[0], &spec[1]);

    TickitWindow* training_window = layouts[1]->children[0];
    create_training_menu(training_window, ctx.trainer);

    tickit_run(t);
    tickit_window_close(root);

    tickit_unref(t);

    cleanup_context(&ctx);
    return 0;
}
