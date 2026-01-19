#include "matrix.h"
#include "data/dataset.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <log.h>

#include <nyoravim/map.h>

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
        log_warn("invalid dataset id: %u", id);
        return NULL;
    }

    log_debug("loading %s dataset", name);

    struct dataset* data = dataset_load(labels, images);
    if (!data) {
        log_error("failed to load %s dataset!", name);
        return NULL;
    }

    log_info("loaded %s dataset", name);
    return data;
}

static void free_dataset(void* user, void* value) { dataset_free(value); }

static nv_map_t* load_datasets() {
    log_trace("loading datasets");

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

int main(int argc, const char** argv) {
    nv_map_t* datasets = load_datasets();
    if (nv_map_size(datasets) < DATASET_COUNT) {
        nv_map_free(datasets);
        return 1;
    }

    nv_map_free(datasets);
    return 0;
}
