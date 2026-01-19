#ifndef _DATASET_H
#define _DATASET_H

#include <stdint.h>
#include <stdbool.h>

typedef struct dataset dataset_t;

/* from ../matrix.h */
typedef struct matrix matrix_t;

struct dataset_entry {
    matrix_t* image;
    uint8_t label;
};

dataset_t* dataset_load(const char* label_path, const char* image_path);
void dataset_free(dataset_t* data);

uint32_t dataset_get_image_count(const dataset_t* data);
uint32_t dataset_get_label_count(const dataset_t* data);

enum {
    DATASET_ENTRY_HAS_IMAGE = (1 << 0),
    DATASET_ENTRY_HAS_LABEL = (1 << 1),
    DATASET_ENTRY_HAS_ALL = DATASET_ENTRY_HAS_IMAGE | DATASET_ENTRY_HAS_LABEL,
};

struct nv_allocator;

/* returns flags (DATASET_ENTRY_HAS_*) */
uint32_t dataset_get_entry(const dataset_t* data, uint32_t index, const struct nv_allocator* alloc,
                           struct dataset_entry* entry);

#endif
