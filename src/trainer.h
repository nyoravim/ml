#ifndef _TRAINER_H
#define _TRAINER_H

#include <stdint.h>

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
    trainer_phase phase;

    float eval_cost;

    uint32_t batch_index;
    uint32_t num_batches;

    uint32_t num_workers;
    const uint32_t* training_entry_indices;
};

struct trainer_spec {
    dataset_t* training_data;
    dataset_t* test_data;

    uint32_t batch_size;
    float learning_rate;
};

trainer_t* trainer_new(model_t* model, dataset_t* training_data, dataset_t* test_data);
void trainer_destroy(trainer_t* trainer);

const struct trainer_spec* trainer_get_spec(const trainer_t* trainer);

void trainer_get_output(struct trainer_output* output);

void trainer_start(trainer_t* trainer);
void trainer_stop(trainer_t* trainer);

#endif
