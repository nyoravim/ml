#include "panel.h"

#include <ncurses.h>

#include <nyoravim/mem.h>
#include <nyoravim/map.h>
#include <nyoravim/util.h>

#include <assert.h>
#include <string.h>
#include <time.h>

/* usleep */
#include <unistd.h>

static uint32_t ncurses_refs = 0;

static void ncurses_ref() {
    if (ncurses_refs++ > 0) {
        return;
    }

    initscr();
    cbreak();
    noecho();

    intrflush(stdscr, FALSE);
    keypad(stdscr, TRUE);
    nodelay(stdscr, TRUE);
    curs_set(1);
}

static void ncurses_unref() {
    if (--ncurses_refs > 0) {
        return;
    }

    endwin();
}

struct panel_stack_node {
    struct panel_stack_node* lower;
    struct active_panel panel;
};

typedef struct ui_context {
    nv_arena_t* arena;

    nv_map_t* panel_state;

    struct panel_stack_node* top_panel;
    char* focused_panel;

    uint32_t screen_width, screen_height;
    char* screen_buffer;
} ui_context_t;

static bool key_equals(void* user, const void* lhs, const void* rhs) {
    return strcmp(lhs, rhs) == 0;
}

static size_t hash_key(void* user, const void* key) { return nv_hash_string(key); }

static void free_id_key(void* user, void* key) { nv_arena_free(user, key); }

static void free_panel_state(void* user, void* value) {
    struct panel_state* state = value;
    const struct panel* panel = state->panel;

    if (panel->vtable.cleanup) {
        panel->vtable.cleanup(state->data, user);
    }

    nv_arena_free(user, state);
}

static void get_screen_size(uint32_t* width, uint32_t* height) {
    if (width) {
        *width = (uint32_t)getmaxx(stdscr);
    }

    if (height) {
        *height = (uint32_t)getmaxy(stdscr);
    }
}

static void resize_buffer(ui_context_t* ctx, uint32_t width, uint32_t height) {
    ctx->screen_width = width;
    ctx->screen_height = height;

    ctx->screen_buffer = nv_arena_realloc(ctx->arena, ctx->screen_buffer, width * height);
}

ui_context_t* ui_context_alloc() {
    ncurses_ref();

    /* 4 mb */
    nv_arena_t* arena = nv_arena_create(4 * 1024 * 1024);

    ui_context_t* ctx = nv_arena_alloc(arena, sizeof(ui_context_t));
    assert(ctx);

    ctx->arena = arena;

    /* none pushed */
    ctx->top_panel = NULL;

    /* none focused */
    ctx->focused_panel = NULL;

    struct nv_map_callbacks callbacks;
    callbacks.equals = key_equals;
    callbacks.hash = hash_key;
    callbacks.free_key = free_id_key;
    callbacks.free_value = free_panel_state;
    callbacks.user = arena;

    /* todo: add arena param to nv_map_alloc */
    ctx->panel_state = nv_map_alloc(16, &callbacks);
    assert(ctx->panel_state);

    uint32_t width, height;
    get_screen_size(&width, &height);

    ctx->screen_buffer = NULL;
    resize_buffer(ctx, width, height);

    return ctx;
}

void ui_context_free(ui_context_t* ctx) {
    if (!ctx) {
        return;
    }

    nv_map_free(ctx->panel_state);

    /* will wipe out associated data */
    nv_arena_destroy(ctx->arena);

    ncurses_unref();
}

static void validate_buffer_size(ui_context_t* ctx) {
    uint32_t width, height;
    get_screen_size(&width, &height);

    if (ctx->screen_width != width || ctx->screen_height != height) {
        nv_free(ctx->screen_buffer);
        resize_buffer(ctx, width, height);
    }
}

static void flush_buffer(const ui_context_t* ctx) {
    clear();

    char buffer[ctx->screen_width + 1];
    buffer[ctx->screen_width] = '\0';

    for (uint32_t i = 0; i < ctx->screen_height; i++) {
        const char* line = ctx->screen_buffer + i * ctx->screen_width;
        mvaddnstr(i, 0, line, ctx->screen_width);
    }

    refresh();
}

static void time_diff(const struct timespec* t1, const struct timespec* t0,
                      struct timespec* delta) {
    assert(t1->tv_sec >= t0->tv_sec);
    delta->tv_sec = t1->tv_sec - t0->tv_sec;

    uint64_t t1_nsec = t1->tv_nsec;
    if (t1->tv_nsec < t0->tv_nsec) {
        assert(delta->tv_sec > 1);
        delta->tv_sec--;

        t1_nsec += 1e9;
    }

    delta->tv_nsec = t1_nsec - t0->tv_nsec;
}

int ui_context_loop(ui_context_t* ctx, const struct panel* root) {
    assert(ctx && root);

    struct timespec t0, t1, delta;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t0);

    while (true) {
        validate_buffer_size(ctx);

        /* todo: handle input */

        struct panel_bounds bounds;
        bounds.x = bounds.y = 0;
        bounds.width = ctx->screen_width;
        bounds.height = ctx->screen_height;

        ui_context_push_panel(ctx, root, &bounds, PANEL_VISIBLE);
        ui_context_update_top(ctx);

        memset(ctx->screen_buffer, ' ', ctx->screen_width * ctx->screen_height);
        ui_context_render_top(ctx);

        ui_context_pop_panel(ctx);
        flush_buffer(ctx);

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        time_diff(&t1, &t0, &delta);
        memcpy(&t0, &t1, sizeof(struct timespec));

        uint64_t max_frame_time_nsec = 1e9 / 60;
        if (delta.tv_nsec < max_frame_time_nsec) {
            uint64_t sleep_time_nsec = max_frame_time_nsec - delta.tv_nsec;
            uint64_t sleep_time_usec = sleep_time_nsec / 1e3;

            usleep(sleep_time_usec);
        }
    }
}

static bool is_panel_focused(ui_context_t* ctx, const char* id) {
    if (!id || !ctx->focused_panel) {
        return false;
    }

    return strcmp(id, ctx->focused_panel) == 0;
}

static struct panel_state* get_panel_state(ui_context_t* ctx, const struct panel* panel) {
    assert(ctx);

    if (!panel || !panel->id) {
        return NULL;
    }

    struct panel_state* state;
    if (!nv_map_get(ctx->panel_state, panel->id, (void**)&state)) {
        state = nv_arena_alloc(ctx->arena, sizeof(struct panel_state));
        state->panel = panel;

        state->data = NULL;
        if (panel->vtable.init) {
            state->data = panel->vtable.init(panel->user, ctx->arena);
        }

        char* id = nv_strdup(panel->id);
        assert(id);

        nv_map_insert(ctx->panel_state, id, state);
    }

    return state;
}

void ui_context_push_panel(ui_context_t* ctx, const struct panel* panel,
                           const struct panel_bounds* bounds, uint32_t flags) {
    assert(ctx);

    struct panel_stack_node* node = nv_arena_alloc(ctx->arena, sizeof(struct panel_stack_node));
    node->lower = ctx->top_panel;

    memcpy(&node->panel.bounds, bounds, sizeof(struct panel_bounds));

    node->panel.state = get_panel_state(ctx, panel);
    assert(node->panel.state);

    node->panel.flags = flags;
    node->panel.focused = is_panel_focused(ctx, panel->id);

    ctx->top_panel = node;
}

bool ui_context_pop_panel(ui_context_t* ctx) {
    assert(ctx);

    if (!ctx->top_panel) {
        return false;
    }

    struct panel_stack_node* lower = ctx->top_panel->lower;
    nv_arena_free(ctx->arena, ctx->top_panel);

    ctx->top_panel = lower;
    return true;
}

const struct active_panel* ui_context_get_top_panel(const ui_context_t* ctx) {
    assert(ctx);
    if (!ctx->top_panel) {
        return NULL;
    }

    return &ctx->top_panel->panel;
}

bool ui_context_update_top(ui_context_t* ctx) {
    assert(ctx);

    const struct active_panel* top = ui_context_get_top_panel(ctx);
    if (!top) {
        return false;
    }

    const struct panel* panel = top->state->panel;
    if (panel->vtable.update) {
        panel->vtable.update(ctx);
    }

    return true;
}

bool ui_context_render_top(ui_context_t* ctx) {
    assert(ctx);

    const struct active_panel* top = ui_context_get_top_panel(ctx);
    if (!top) {
        return false;
    }

    if (!(top->flags & PANEL_VISIBLE)) {
        return true; /* shouldnt render if panel isnt visible */
    }

    const struct panel* panel = top->state->panel;
    if (panel->vtable.render) {
        panel->vtable.render(ctx);
    }

    return true;
}

void ui_context_focus_panel(ui_context_t* ctx, const char* panel_id) {
    assert(ctx);

    nv_free(ctx->focused_panel);
    ctx->focused_panel = panel_id ? nv_strdup(panel_id) : NULL;
}

void ui_context_render_char(ui_context_t* ctx, uint32_t x, uint32_t y, char c) {
    assert(ctx);

    const struct active_panel* top = ui_context_get_top_panel(ctx);
    assert(top);

    if (x >= top->bounds.width || y >= top->bounds.height) {
        return;
    }

    uint32_t abs_x = x + top->bounds.x;
    uint32_t abs_y = y + top->bounds.y;

    uint32_t index = abs_y * ctx->screen_width + abs_x;
    ctx->screen_buffer[index] = c;
}

void ui_context_render_string(ui_context_t* ctx, uint32_t x, uint32_t y, const char* str) {
    size_t len = strlen(str);
    for (size_t i = 0; i < len; i++) {
        ui_context_render_char(ctx, x + (uint32_t)i, y, str[i]);
    }
}
