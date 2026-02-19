#ifndef _LAYOUT_H
#define _LAYOUT_H

#include <tickit.h>

typedef enum {
    LAYOUT_HORIZONTAL,
    LAYOUT_VERTICAL,
} layout_type;

struct layout_spec {
    layout_type type;
    uint32_t split;
};

struct layout {
    struct layout_spec spec;
    TickitWindow* window;

    TickitWindow* children[2];
};

struct layout* layout_create(TickitWindow* window, const struct layout_spec* spec);
void layout_invalidate(struct layout* layout);

#endif
