#ifndef _TRAINER_H
#define _TRAINER_H

#include <stdint.h>
#include <stdbool.h>

typedef struct trainer trainer_t;

/* from data/dataset.h */
typedef struct dataset dataset_t;
struct dataset_entry;

/* from model.h */
typedef struct model model_t;

typedef enum {
    TRAINER_PHASE_TRAINING,
    TRAINER_PHASE_EVAL,
} trainer_phase;

struct trainer_spec {
    dataset_t* training_data;
    dataset_t* test_data;

    uint32_t batch_size;
    float learning_rate;
};

trainer_t* trainer_new(model_t* model, const char* disk_path, const struct trainer_spec* spec);
void trainer_destroy(trainer_t* trainer);

void trainer_get_spec(const trainer_t* trainer, struct trainer_spec* spec);
bool trainer_set_spec(trainer_t* trainer, const struct trainer_spec* spec);

trainer_phase trainer_get_phase(const trainer_t* trainer);
uint32_t trainer_get_working_entries(const trainer_t* trainer, uint32_t max_entries,
                                     struct dataset_entry* entries);

void trainer_start(trainer_t* trainer);
void trainer_stop(trainer_t* trainer);

bool trainer_is_running(const trainer_t* trainer);

#endif
