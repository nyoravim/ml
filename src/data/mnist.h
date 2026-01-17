#ifndef _MNIST_H
#define _MNIST_H

#include <stddef.h>
#include <stdint.h>

struct mnist {
    uint8_t num_dimensions;
    uint32_t* dimensions;

    uint8_t* data;
};

struct mnist* mnist_load(const char* path);

void mnist_free(struct mnist* data);

const uint8_t* mnist_get_data(const struct mnist* data, const uint32_t* offsets);

#endif
