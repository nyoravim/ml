#ifndef _TRAINER_H
#define _TRAINER_H

typedef struct trainer trainer_t;

/* from data/dataset.h */
typedef struct dataset dataset_t;

/* from model.h */
typedef struct model model_t;

typedef enum {
    TRAINER_PHASE_TRAINING,
    TRAINER_PHASE_TESTING,
} trainer_phase;

struct trainer_output {
};

trainer_t* trainer_new(model_t* model, dataset_t* training_data, dataset_t* test_data);
void trainer_destroy(trainer_t* trainer);

void trainer_start(trainer_t* trainer);
void trainer_stop(trainer_t* trainer);

#endif
