#include "layout.h"

#include <nyoravim/mem.h>

#include <assert.h>
#include <string.h>

static void generate_horizontal_layout(const struct layout* layout, TickitRect* geometry) {
    uint32_t width = tickit_window_cols(layout->window);
    uint32_t height = tickit_window_lines(layout->window);
    assert(layout->spec.split > 0 && layout->spec.split < width - 1);

    /* first window is left half */
    geometry[0].top = 0;
    geometry[0].left = 0;
    geometry[0].cols = layout->spec.split;
    geometry[0].lines = height;

    /* second is right half */
    uint32_t right_start = layout->spec.split + 1;

    geometry[1].top = 0;
    geometry[1].left = right_start;
    geometry[1].cols = width - right_start;
    geometry[1].lines = height;
}

static void generate_vertical_layout(const struct layout* layout, TickitRect* geometry) {
    uint32_t width = tickit_window_cols(layout->window);
    uint32_t height = tickit_window_lines(layout->window);
    assert(layout->spec.split > 0 && layout->spec.split < height - 1);

    /* first window is top half */
    geometry[0].top = 0;
    geometry[0].left = 0;
    geometry[0].cols = width;
    geometry[0].lines = layout->spec.split;

    /* second is bottom half */
    uint32_t bottom_start = layout->spec.split + 1;

    geometry[1].top = bottom_start;
    geometry[1].left = 0;
    geometry[1].cols = width;
    geometry[1].lines = height - bottom_start;
}

static void generate_layout_geometry(const struct layout* layout, TickitRect* geometry) {
    assert(layout && geometry);

    switch (layout->spec.type) {
    case LAYOUT_HORIZONTAL:
        generate_horizontal_layout(layout, geometry);
        break;
    case LAYOUT_VERTICAL:
        generate_vertical_layout(layout, geometry);
        break;
    }
}

static int change_layout_geometry(TickitWindow* window, TickitEventFlags flags, void* info,
                                  void* user) {
    TickitGeomchangeEventInfo* geomchange = info;
    struct layout* layout = user;

    uint32_t size;
    switch (layout->spec.type) {
    case LAYOUT_HORIZONTAL:
        size = geomchange->rect.cols;
        break;
    case LAYOUT_VERTICAL:
        size = geomchange->rect.lines;
        break;
    }

    uint32_t max_split = size - 2;
    if (layout->spec.split > max_split) {
        layout->spec.split = max_split;
    }

    layout_invalidate(layout);
    return 1;
}

static int render_layout(TickitWindow* window, TickitEventFlags flags, void* info, void* user) {
    TickitExposeEventInfo* expose = info;
    TickitRenderBuffer* rb = expose->rb;

    uint32_t right = tickit_window_cols(window) - 1;
    uint32_t bottom = tickit_window_lines(window) - 1;

    struct layout* layout = user;
    switch (layout->spec.type) {
    case LAYOUT_HORIZONTAL:
        tickit_renderbuffer_vline_at(rb, 0, bottom, layout->spec.split, TICKIT_LINE_SINGLE, 0);
        break;
    case LAYOUT_VERTICAL:
        tickit_renderbuffer_hline_at(rb, layout->spec.split, 0, right, TICKIT_LINE_SINGLE, 0);
        break;
    }

    return 1;
}

static int destroy_layout(TickitWindow* window, TickitEventFlags flags, void* info, void* user) {
    nv_free(user); /* isnt really any data to free beyond this */

    return 1;
}

struct layout* layout_create(TickitWindow* window, const struct layout_spec* spec) {
    assert(window && spec);

    struct layout* layout = nv_alloc(sizeof(struct layout));
    assert(layout);

    layout->window = window;
    memcpy(&layout->spec, spec, sizeof(struct layout_spec));

    TickitRect geometry[2];
    generate_layout_geometry(layout, geometry);

    for (uint32_t i = 0; i < 2; i++) {
        TickitWindow* child = tickit_window_new(window, geometry[i], 0);
        assert(child);

        layout->children[i] = child;
    }

    tickit_window_bind_event(window, TICKIT_WINDOW_ON_GEOMCHANGE, 0, change_layout_geometry,
                             layout);

    tickit_window_bind_event(window, TICKIT_WINDOW_ON_EXPOSE, 0, render_layout, layout);
    tickit_window_bind_event(window, TICKIT_WINDOW_ON_DESTROY, 0, destroy_layout, layout);

    return layout;
}

void layout_invalidate(struct layout* layout) {
    TickitRect geometry[2];
    generate_layout_geometry(layout, geometry);

    for (uint32_t i = 0; i < 2; i++) {
        tickit_window_set_geometry(layout->children[i], geometry[i]);
    }
}
