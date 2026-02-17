#ifndef PANEL_H
#define PANEL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include <nyoravim/arena.h>

typedef struct ui_context ui_context_t;

struct panel_vtable {
    void* (*init)(void* user, nv_arena_t* arena);
    void (*cleanup)(void* data, nv_arena_t* arena);

    void (*update)(ui_context_t* context);
    void (*render)(ui_context_t* context);
};

struct panel {
    struct panel_vtable vtable;

    const char* id;
    void* user;
};

struct panel_bounds {
    uint32_t x, y;
    uint32_t width, height;
};

struct panel_state {
    void* data;
    const struct panel* panel;
};

enum {
    PANEL_VISIBLE = (1 << 0),
};

struct active_panel {
    const struct panel_state* state;

    struct panel_bounds bounds;
    uint32_t flags;
    bool focused;
};

ui_context_t* ui_context_alloc();
void ui_context_free(ui_context_t* ctx);

int ui_context_loop(ui_context_t* ctx, const struct panel* root);

void ui_context_push_panel(ui_context_t* ctx, const struct panel* panel,
                           const struct panel_bounds* bounds, uint32_t flags);

bool ui_context_pop_panel(ui_context_t* ctx);

const struct active_panel* ui_context_get_top_panel(const ui_context_t* ctx);

bool ui_context_update_top(ui_context_t* ctx);
bool ui_context_render_top(ui_context_t* ctx);

void ui_context_focus_panel(ui_context_t* ctx, const char* panel_id);

void ui_context_render_char(ui_context_t* ctx, uint32_t x, uint32_t y, char c);
void ui_context_render_string(ui_context_t* ctx, uint32_t x, uint32_t y, const char* str);

#endif
