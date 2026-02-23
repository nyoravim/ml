#include "training_menu.h"

#include "../trainer.h"

#include <nyoravim/mem.h>

#include <tickit.h>

#include <assert.h>
#include <string.h>

static int training_menu_on_expose(TickitWindow* window, TickitEventFlags flags, void* info,
                                   void* data) {
    TickitExposeEventInfo* expose = info;
    TickitRenderBuffer* rb = expose->rb;

    TickitRect rect = tickit_window_get_geometry(window);
    tickit_renderbuffer_eraserect(rb, &rect);

    struct training_menu* menu = data;

    bool running = trainer_is_running(menu->trainer);
    const char* action_text = running ? "Stop" : "Start";

    tickit_renderbuffer_textf_at(rb, 0, 0, "%s: F5", action_text);

    struct trainer_spec spec;
    trainer_get_spec(menu->trainer, &spec);

    tickit_renderbuffer_textf_at(rb, 2, 0, "Learning rate: %f", spec.learning_rate);
    tickit_renderbuffer_textf_at(rb, 3, 0, "Batch size: %u", spec.batch_size);

    float batch_cost = trainer_get_batch_cost(menu->trainer);
    tickit_renderbuffer_textf_at(rb, 5, 0, "Training batch cost: %f", batch_cost);

    float eval_cost = trainer_get_eval_cost(menu->trainer);
    tickit_renderbuffer_textf_at(rb, 6, 0, "Testing/eval cost: %f", eval_cost);

    return 1;
}

static const char* substring_after_last(char c, const char* str) {
    /* keep track of two pointers; "base" keeps track of the start of the substring; "cursor" moves
     * through string, and when it reaches the delimiter, it moves base to cursor */

    const char* base = str;
    const char* cursor = base;

    char current;
    while ((current = *cursor) != '\0') {
        cursor++;

        if (current == c) {
            base = cursor;
        }
    }

    return base;
}

static bool keys_match(const TickitKeyEventInfo* event, const char* test, int mods) {
    int present_mods = event->mod & mods;
    if (present_mods != mods) {
        return false;
    }

    static const char delimiter = '-';
    const char* expected = substring_after_last(delimiter, test);
    const char* actual = substring_after_last(delimiter, event->str);

    return strcmp(expected, actual) == 0;
}

static void update_trainer_status(struct training_menu* menu) {
    if (trainer_is_running(menu->trainer)) {
        trainer_stop(menu->trainer);
    } else {
        trainer_start(menu->trainer);
    }
}

static int training_menu_on_key(TickitWindow* window, TickitEventFlags flags, void* info,
                                void* data) {
    TickitKeyEventInfo* key = info;
    struct training_menu* menu = data;

    if (key->type != TICKIT_KEYEV_KEY) {
        return 0;
    }

    if (keys_match(key, "F5", 0)) {
        update_trainer_status(menu);
        return 1;
    }

    return 0;
}

static int training_menu_on_destroy(TickitWindow* window, TickitEventFlags flags, void* info,
                                    void* data) {
    nv_free(data);
    return 1;
}

struct training_menu* create_training_menu(TickitWindow* window, trainer_t* trainer) {
    assert(window && trainer);

    struct training_menu* menu = nv_alloc(sizeof(struct training_menu));
    assert(menu);

    menu->window = window;
    menu->trainer = trainer;

    tickit_window_bind_event(window, TICKIT_WINDOW_ON_EXPOSE, 0, training_menu_on_expose, menu);
    tickit_window_bind_event(window, TICKIT_WINDOW_ON_KEY, 0, training_menu_on_key, menu);
    tickit_window_bind_event(window, TICKIT_WINDOW_ON_DESTROY, 0, training_menu_on_destroy, menu);

    return menu;
}
