#include "matrix.h"
#include "model.h"

#include "prng.h"

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

static model_t* create_model(const struct nv_allocator* alloc, const char* path) {
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

    model_t* model = model_alloc(alloc, 28 * 28, layer_count, layers);
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

static model_t* open_model(const struct nv_allocator* alloc, const char* path) {
    if (file_exists(path)) {
        NV_LOG_INFO("file %s exists; reading", path);
        return model_read_from_path(alloc, path);
    } else {
        NV_LOG_INFO("file %s does not exist; creating new model and writing", path);
        return create_model(alloc, path);
    }
}

enum { MODE_TRAINING, MODE_EVAL };

struct program_params {
    uint32_t mode;
    char* model_path;
    uint32_t cluster_size;
    float training_threshold;
};

static bool parse_program_mode(const char* name, uint32_t* mode) {
    if (strcmp(name, "training") == 0) {
        NV_LOG_DEBUG("training selected");

        *mode = MODE_TRAINING;
        return true;
    }

    if (strcmp(name, "eval") == 0) {
        NV_LOG_DEBUG("eval selected");

        *mode = MODE_EVAL;
        return true;
    }

    NV_LOG_ERROR("invalid mode: %s", name);
    return false;
}

static void print_help(const char* program) {
    printf("usage: %s [training|eval] [options]\n"
           "options:\n"
           "\t-c, --cluster\tcluster size\n"
           "\t-m, --model\tmodel path\n"
           "\t-t, --threshold\ttraining threshold\n",
           program);
}

static bool parse_params(int argc, const char** argv, struct program_params* params) {
    if (argc >= 2 && strcmp(argv[1], "--help") == 0) {
        print_help(argv[0]);
        exit(0);
    }

    struct nv_list arguments;
    nv_list_init(&arguments);

    for (int i = 1; i < argc; i++) {
        const char* param = argv[i];
    }
    if (argc < 2) {
        NV_LOG_DEBUG("no mode passed; assuming training");
        params->mode = MODE_TRAINING;
    } else if (!parse_program_mode(argv[1], &params->mode)) {
        return false;
    }

    nv_list_clear(&arguments, NULL, NULL);
    return true;
}

struct model_context {
    nv_map_t* datasets;

    model_t* model;
    const char* model_path;

    struct program_params params;
};

static void cleanup_context(const struct model_context* ctx) {
    nv_free(ctx->params.model_path);

    nv_map_free(ctx->datasets);
    model_free(ctx->model);
}

static float train_on_cluster(struct model_context* ctx, const dataset_t* data,
                              const uint32_t* indices) {
    struct model_layer* deltas = model_alloc_deltas(ctx->model);

    NV_LOG_INFO("todo: train on cluster");

    model_free_deltas(deltas);
    return 0.f;
}

/* generates a random uint32_t in the range [a, b) */
static uint32_t rand_between(uint32_t a, uint32_t b) {
    uint32_t r = prng_rand_g();
    return (r % (b - a)) + a;
}

static float run_training_phase(struct model_context* ctx, const dataset_t* data) {
    uint32_t num_images = dataset_get_image_count(data);
    uint32_t num_labels = dataset_get_label_count(data);
    uint32_t num_entries = num_images < num_labels ? num_images : num_labels;

    uint32_t num_clusters = num_entries / ctx->params.cluster_size;
    NV_LOG_DEBUG("beginning training phase %ux%u", num_clusters, ctx->params.cluster_size);

    /* shuffle indices */
    uint32_t total_entries = num_clusters * ctx->params.cluster_size;
    uint32_t indices[total_entries];

    for (uint32_t i = 0; i < total_entries; i++) {
        indices[i] = i;
    }

    for (uint32_t i = 0; i < total_entries - 1; i++) {
        uint32_t j = rand_between(i + 1, total_entries);

        uint32_t swap = indices[i];
        indices[i] = indices[j];
        indices[j] = swap;
    }

    float avg = 0.f;
    for (uint32_t i = 0; i < num_clusters; i++) {
        NV_LOG_DEBUG("training on cluster %u", i);

        uint32_t offset = i * ctx->params.cluster_size;
        const uint32_t* cluster_indices = indices + offset;

        float cost = train_on_cluster(ctx, data, cluster_indices);
        avg += cost / num_clusters;
    }

    return avg;
}

static void train_for_threshold(struct model_context* ctx, const dataset_t* data) {
    while (true) {
    }
}

static void run_training(struct model_context* ctx) {
    NV_LOG_INFO("beginning training cycle");

    while (true) {
        dataset_t* data;
        if (nv_map_get(ctx->datasets, (void*)DATASET_TRAINING, (void**)&data)) {
            run_training_phase(ctx, data);
        } else {
            NV_LOG_INFO("no training dataset; exiting out of training cycle");
            break;
        }

        /* todo: eval */
    }
}

int main(int argc, const char** argv) {
    struct nv_logger_sink stdout_sink;
    nv_create_stdout_sink(&stdout_sink);
    stdout_sink.level = NV_LOG_LEVEL_TRACE;

    struct nv_logger logger;
    logger.level = NV_LOG_LEVEL_TRACE;
    logger.sink_count = 1;
    logger.sinks = &stdout_sink;

    nv_set_default_logger(&logger);

    struct model_context ctx;
    memset(&ctx, 0, sizeof(struct model_context));

    if (!parse_params(argc, argv, &ctx.params)) {
        cleanup_context(&ctx);
    }

    ctx.datasets = load_datasets();
    if (nv_map_size(ctx.datasets) < DATASET_COUNT) {
        cleanup_context(&ctx);
        return 1;
    }

    ctx.model_path = ctx.params.model_path ? ctx.params.model_path : "model.bin";
    ctx.model = open_model(NULL, ctx.model_path);

    switch (ctx.params.mode) {
    case MODE_TRAINING:
        run_training(&ctx);
        break;
    }

    cleanup_context(&ctx);
    return 0;
}
