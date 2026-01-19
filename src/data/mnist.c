#include "mnist.h"

#include <stdbool.h>
#include <string.h>
#include <assert.h>

#include <arpa/inet.h>

#include <zlib.h>

#include <nyoravim/mem.h>
#include <nyoravim/log.h>

struct mnist_parse_context {
    struct mnist* data;
    size_t header_values_read;
    size_t values_read;

    size_t bytes_consumed;
};

static size_t get_data_size(const struct mnist* data) {
    if (data->num_dimensions == 0) {
        return 0;
    }

    size_t size = 1;
    for (uint8_t i = 0; i < data->num_dimensions; i++) {
        size *= data->dimensions[i];
    }

    return size;
}

static bool is_data_complete(const struct mnist_parse_context* ctx) {
    if (!ctx->data) {
        return false;
    }

    if (ctx->header_values_read < ctx->data->num_dimensions + 1) {
        return false;
    }

    size_t data_size = get_data_size(ctx->data);
    if (ctx->values_read < data_size) {
        return false;
    }

    return true;
}

static bool consume_data(const void* data, size_t size, struct mnist_parse_context* ctx,
                         size_t* bytes_processed) {
    if (ctx->header_values_read < ctx->data->num_dimensions + 1) {
        /* header is yet incomplete */

        if (size < sizeof(uint32_t)) {
            *bytes_processed = 0;
            return true;
        }

        uint32_t be = *(uint32_t*)data;
        uint32_t ne = ntohl(be);

        if (ctx->header_values_read == 0) {
            /* magic number */

            /* is this 0x08xx? */
            if ((ne & ~(uint32_t)0xFF) != 0x800) {
                NV_LOG_ERROR("invalid magic number: 0x%X", ne);
                return false;
            }

            ctx->data->num_dimensions = ne & 0xFF;
            if (ctx->data->num_dimensions == 0) {
                NV_LOG_ERROR("dimension byte set as 0!");
                return false;
            }

            NV_LOG_DEBUG("%hhu matrix dimensions", ctx->data->num_dimensions);

            ctx->data->dimensions = nv_alloc(ctx->data->num_dimensions * sizeof(uint32_t));
            assert(ctx->data->dimensions);
        } else {
            /* dimension */
            size_t dimension_index = ctx->header_values_read - 1;

            ctx->data->dimensions[dimension_index] = ne;
            NV_LOG_DEBUG("dimension %zu: %u", dimension_index, ne);
        }

        *bytes_processed = sizeof(uint32_t);
        ctx->header_values_read++;
    } else {
        /* rest is data */

        size_t total_size = get_data_size(ctx->data);
        if (!ctx->data->data) {
            ctx->data->data = nv_alloc(total_size);
            assert(ctx->data->data);
        }

        size_t remaining = total_size - ctx->values_read;
        size_t to_copy = size > remaining ? remaining : size; /* min */

        if (to_copy > 0) {
            memcpy(ctx->data->data + ctx->values_read, data, to_copy);
            ctx->values_read += to_copy;
        }

        *bytes_processed = to_copy;
    }

    return true;
}

/* reads all data it can from provided buffer. on success (true), returns consumed bytes in
 * ctx->bytes_consumed */
static bool read_chunk(const void* data, size_t size, struct mnist_parse_context* ctx) {
    size_t offset = 0;
    while (true) {
        size_t bytes_processed;
        if (!consume_data(data + offset, size - offset, ctx, &bytes_processed)) {
            return false;
        }

        if (bytes_processed == 0) {
            break; /* had nothing to do */
        }

        offset += bytes_processed;
    }

    ctx->bytes_consumed = offset;
    return true;
}

struct mnist* mnist_load(const char* path) {
    gzFile file = gzopen(path, "rb");
    if (!file) {
        NV_LOG_ERROR("failed to open gz file for reading: %s", path);
        return NULL;
    }

    struct mnist_parse_context ctx;
    memset(&ctx, 0, sizeof(struct mnist_parse_context));

    ctx.data = nv_alloc(sizeof(struct mnist));
    memset(ctx.data, 0, sizeof(struct mnist));

    uint8_t buffer[256];
    size_t bytes_read, offset = 0;

    while (true) {
        bytes_read = gzfread(buffer + offset, 1, sizeof(buffer) - offset, file);
        if (bytes_read == 0) {
            break;
        }

        if (!read_chunk(buffer, offset + bytes_read, &ctx)) {
            break;
        }

        if (ctx.bytes_consumed < bytes_read) {
            size_t remaining = bytes_read - ctx.bytes_consumed;

            void* remaining_start = buffer + offset + ctx.bytes_consumed;
            memcpy(buffer, remaining_start, remaining);

            offset = remaining;
        }
    }

    gzclose(file);
    if (is_data_complete(&ctx)) {
        return ctx.data;
    } else {
        NV_LOG_WARN("data not complete; discarding");

        mnist_free(ctx.data);
        return NULL;
    }
}

void mnist_free(struct mnist* data) {
    if (!data) {
        return;
    }

    nv_free(data->dimensions);
    nv_free(data->data);
    nv_free(data);
}

const uint8_t* mnist_get_data(const struct mnist* data, const uint32_t* offsets) {
    size_t offset = 0;
    for (uint8_t i = 0; i < data->num_dimensions; i++) {
        size_t byte_offset = offsets[i];
        for (uint8_t j = i + 1; j < data->num_dimensions; j++) {
            byte_offset *= data->dimensions[j];
        }

        offset += byte_offset;
    }

    return data->data + offset;
}
