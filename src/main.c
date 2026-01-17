#include "data/dataset.h"

#include <stdio.h>

#include <log.h>

static void draw_mnist_digit(const struct dataset_entry* entry) {
    /* over rows */
    for (uint32_t y = 0; y < entry->height; y++) {

        /* over columns */
        for (uint32_t x = 0; x < entry->width; x++) {
            uint8_t value = entry->image[y * entry->width + x];

            /* from 256 steps down to 24 with an offset of 232. see
             * https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797 */
            float lightness = (float)value / 255.f;
            uint8_t color = 232 + (uint8_t)(lightness * 23.f);

            /* two spaces to make the pixel fairly square */
            printf("\x1b[48;5;%hhum  ", color);
        }

        /* next row */
        printf("\n");
    }

    /* reset output style */
    printf("\x1b[0m");
}

int main(int argc, const char** argv) {
    struct dataset* data =
        dataset_load("data/train-labels-idx1-ubyte.gz", "data/train-images-idx3-ubyte.gz");

    if (!data) {
        return 1;
    }

    log_info("images: %u", dataset_get_image_count(data));
    log_info("labels: %u", dataset_get_label_count(data));

    struct dataset_entry entry;
    uint32_t flags = dataset_get_entry(data, 0, &entry);

    log_info("entry 0 begin");
    if (flags & DATASET_ENTRY_HAS_LABEL) {
        log_info("label: %hhu", entry.label);
    }

    if (flags & DATASET_ENTRY_HAS_IMAGE) {
        draw_mnist_digit(&entry);
    }

    log_info("entry 0 end");

    dataset_free(data);
    return 0;
}
