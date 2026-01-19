#include "dataset.h"
#include "mnist.h"

#include "../matrix.h"

#include <assert.h>
#include <string.h>

#include <log.h>

#include <nyoravim/mem.h>

struct label_data {
    uint32_t num;
    struct mnist* data;
};

struct image_data {
    uint32_t num;
    uint32_t width, height;

    struct mnist* data;
};

typedef struct dataset {
    struct label_data labels;
    struct image_data images;
} dataset_t;

static bool load_labels(const char* label_path, struct label_data* data) {
    log_info("loading label file: %s", label_path);

    struct mnist* labels = mnist_load(label_path);
    if (!labels) {
        return false;
    }

    if (labels->num_dimensions != 1) {
        log_error("label file has more than one dimension!");

        mnist_free(labels);
        return false;
    }

    data->data = labels;
    data->num = labels->dimensions[0];

    return true;
}

static bool load_images(const char* image_path, struct image_data* data) {
    log_info("loading image file: %s", image_path);

    struct mnist* images = mnist_load(image_path);
    if (!images) {
        return false;
    }

    if (images->num_dimensions != 3) {
        log_error("label file must have 3 dimensions: entries, rows, columns");

        mnist_free(images);
        return false;
    }

    data->data = images;

    /* entries, rows, columns respectively */
    data->num = images->dimensions[0];
    data->height = images->dimensions[1];
    data->width = images->dimensions[2];

    return true;
}

dataset_t* dataset_load(const char* label_path, const char* image_path) {
    log_trace("loading dataset");

    dataset_t* dataset = nv_alloc(sizeof(dataset_t));
    assert(dataset);
    memset(dataset, 0, sizeof(dataset_t));

    if (!load_labels(label_path, &dataset->labels)) {
        log_error("failed to load label file!");

        dataset_free(dataset);
        return NULL;
    }

    if (!load_images(image_path, &dataset->images)) {
        log_error("failed to load image file!");

        dataset_free(dataset);
        return NULL;
    }

    if (dataset->images.num != dataset->labels.num) {
        log_warn("images & labels do not match in number! (%u vs %u)", dataset->images.num,
                 dataset->labels.num);
    }

    return dataset;
}

void dataset_free(dataset_t* data) {
    if (!data) {
        return;
    }

    mnist_free(data->labels.data);
    mnist_free(data->images.data);

    nv_free(data);
}

uint32_t dataset_get_image_count(const dataset_t* data) { return data->images.num; }
uint32_t dataset_get_label_count(const dataset_t* data) { return data->labels.num; }

uint32_t dataset_get_entry(const dataset_t* data, uint32_t index, const struct nv_allocator* alloc,
                           struct dataset_entry* entry) {
    uint32_t flags = 0;

    if (index < data->images.num) {
        uint32_t offsets[] = { index, 0, 0 };
        const uint8_t* image = mnist_get_data(data->images.data, offsets);

        entry->image = mat_alloc(alloc, data->images.height, data->images.width);
        assert(entry->image);

        uint32_t total = data->images.width * data->images.height;
        for (uint32_t i = 0; i < total; i++) {
            entry->image->data[i] = (float)image[i] / 255;
        }

        flags |= DATASET_ENTRY_HAS_IMAGE;
    }

    if (index < data->labels.num) {
        entry->label = data->labels.data->data[index];
        flags |= DATASET_ENTRY_HAS_LABEL;
    }

    return flags;
}
