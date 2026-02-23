#ifndef _TRAINING_MENU_H
#define _TRAINING_MENU_H

#include <stdint.h>

/* from ../trainer.h */
typedef struct trainer trainer_t;

/* from tickit.h */
typedef struct TickitWindow TickitWindow;

struct training_menu {
    trainer_t* trainer;
    TickitWindow* window;
};

struct training_menu* create_training_menu(TickitWindow* window, trainer_t* trainer);

#endif
