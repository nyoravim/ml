#include "trainer.h"

#include "thread_pool.h"
#include "model.h"
#include "matrix.h"
#include "prng.h"

#include "data/dataset.h"

#include <nyoravim/log.h>
#include <nyoravim/arena.h>

#include <assert.h>
#include <string.h>
#include <math.h>

#include <pthread.h>

/* "manager thread" manages a thread pool of workers. manager cycles phases, sends jobs to thread
 * pool. main thread talks to thread pool through mutexes etc */

struct training_phase {
    uint32_t entry_count;

    uint32_t index_capacity;
    uint32_t* shuffled_indices;

    uint32_t batch_count, batch_index;

    pthread_mutex_t gradients_mutex;
    model_gradients_t* gradients;

    float batch_cost;
};

struct training_data {
    trainer_phase phase;
    pthread_mutex_t mutex;

    struct training_phase training;
};

typedef struct trainer {
    nv_arena_t* arena;
    thread_pool_t* pool;

    model_t* model;
    char* disk_path;

    struct trainer_spec spec;

    uint32_t num_workers;
    eval_context_t** training_contexts;
    eval_context_t** eval_contexts;

    bool running, should_continue;
    pthread_t manager_thread;
    pthread_mutex_t manager_mutex;

    struct training_data phase;
} trainer_t;

struct async_job {
    dataset_t* dataset;
    uint32_t worker_index, worker_entries;
    const uint32_t* batch_indices;

    eval_context_t* eval_context;
    float cost;
};

static void prepare_eval_context(eval_context_t* ctx, uint32_t output_count,
                                 const struct dataset_entry* entry) {
    /* convert the label (integer) to vector */
    float expected[output_count];
    memset(expected, 0, output_count * sizeof(float));
    expected[entry->label] = 1.f;

    /* represent input as a flat vector */
    matrix_t input_matrix;
    input_matrix.rows = entry->image->rows * entry->image->columns;
    input_matrix.columns = 1;
    input_matrix.data = entry->image->data;

    /* wrap expected */
    matrix_t expected_matrix;
    expected_matrix.rows = output_count;
    expected_matrix.columns = 1;
    expected_matrix.data = expected;

    eval_context_set_input(ctx, &input_matrix);
    eval_context_set_expected(ctx, &expected_matrix);
}

static void trainer_job(void* user, void* job) {
    trainer_t* trainer = user;
    struct async_job* data = job;

    uint32_t level = eval_context_get_level(data->eval_context);
    if (level < EVAL_LEVEL_EVAL) {
        return; /* nothing to do */
    }

    uint32_t output_count = model_get_output_count(trainer->model);
    for (uint32_t i = 0; i < data->worker_entries; i++) {
        uint32_t batch_index = i * trainer->num_workers + data->worker_index;
        uint32_t entry_index = data->batch_indices[batch_index];

        struct dataset_entry entry;
        uint32_t flags = dataset_get_entry(data->dataset, entry_index, &entry);

        /* we need an image */
        assert(flags & DATASET_ENTRY_HAS_IMAGE);

        /* also need a label */
        assert(flags & DATASET_ENTRY_HAS_LABEL);

        prepare_eval_context(data->eval_context, output_count, &entry);
        mat_free(entry.image);

        /* pull the trigger */
        eval_context_eval(data->eval_context);

        const matrix_t* cost_matrix = eval_context_get_cost(data->eval_context);
        for (uint32_t i = 0; i < cost_matrix->rows * cost_matrix->columns; i++) {
            data->cost += fabsf(cost_matrix->data[i]);
        }

        if (level >= EVAL_LEVEL_BACKPROP) {
            pthread_mutex_lock(&trainer->phase.training.gradients_mutex);
            eval_context_add_gradients(data->eval_context, trainer->phase.training.gradients);
            pthread_mutex_unlock(&trainer->phase.training.gradients_mutex);
        }
    }
}

static void shuffle_indices(uint32_t* indices, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        indices[i] = i;
    }

    for (uint32_t i = 0; i < count - 1; i++) {
        uint32_t min = i + 1;
        uint32_t range = count - min;
        uint32_t j = min + (prng_rand_g() % range);

        uint32_t temp = indices[j];
        indices[j] = indices[i];
        indices[i] = temp;
    }
}

static void trainer_start_training(trainer_t* trainer) {
    struct training_data* phase = &trainer->phase;
    phase->phase = TRAINER_PHASE_TRAINING;

    assert(trainer->spec.training_data);
    uint32_t image_count = dataset_get_image_count(trainer->spec.training_data);
    uint32_t label_count = dataset_get_label_count(trainer->spec.training_data);
    uint32_t raw_entry_count = image_count > label_count ? label_count : image_count;

    struct training_phase* training = &phase->training;
    training->entry_count = raw_entry_count - (raw_entry_count % trainer->spec.batch_size);
    training->batch_count = raw_entry_count / trainer->spec.batch_size;
    training->batch_index = 0;

    if (training->entry_count > training->index_capacity) {
        training->index_capacity = training->entry_count;
        training->shuffled_indices = nv_arena_realloc(trainer->arena, training->shuffled_indices,
                                                      sizeof(uint32_t) * training->entry_count);
    }

    shuffle_indices(training->shuffled_indices, training->entry_count);
    NV_LOG_DEBUG("starting training phase with %u batches (%u entries per batch)",
                 training->batch_count, trainer->spec.batch_size);
}

static void trainer_train_on_batch(trainer_t* trainer) {
    struct training_phase* phase = &trainer->phase.training;
    model_gradients_zero(phase->gradients);

    uint32_t batch_size = trainer->spec.batch_size;
    struct async_job jobs[trainer->num_workers];

    for (uint32_t i = 0; i < trainer->num_workers; i++) {
        struct async_job* job = &jobs[i];

        job->eval_context = trainer->training_contexts[i];
        job->dataset = trainer->spec.training_data;
        job->worker_index = i;
        job->cost = 0.f;
        job->batch_indices = phase->shuffled_indices + phase->batch_index * batch_size;

        job->worker_entries = batch_size / trainer->num_workers;
        if (i < batch_size % trainer->num_workers) {
            job->worker_entries++;
        }

        thread_pool_push_job(trainer->pool, job);
    }

    thread_pool_wait_idle(trainer->pool);
    model_gradients_flush(phase->gradients, trainer->spec.learning_rate, trainer->spec.batch_size,
                          trainer->model);

    float batch_cost = 0.f;
    for (uint32_t i = 0; i < trainer->num_workers; i++) {
        batch_cost += jobs[i].cost;
    }

    uint32_t output_count = model_get_output_count(trainer->model);
    phase->batch_cost = batch_cost / (output_count * batch_size);

    if (trainer->disk_path) {
        NV_LOG_DEBUG("writing model to file at path %s", trainer->disk_path);
        if (!model_write_to_path(trainer->model, trainer->disk_path)) {
            NV_LOG_WARN("failed to write! model may be out of date");
        }
    }

    if (++phase->batch_index >= phase->batch_count) {
        trainer->phase.phase = TRAINER_PHASE_EVAL;
    }
}

static bool should_manager_continue(trainer_t* trainer) {
    pthread_mutex_lock(&trainer->manager_mutex);
    bool should_continue = trainer->should_continue;
    pthread_mutex_unlock(&trainer->manager_mutex);

    return should_continue;
}

static void* manager_routine(void* param) {
    trainer_t* trainer = param;

    while (should_manager_continue(trainer)) {
        switch (trainer->phase.phase) {
        case TRAINER_PHASE_TRAINING:
            trainer_train_on_batch(trainer);
            break;
        case TRAINER_PHASE_EVAL:
            NV_LOG_WARN("eval not implemented yet; going back to training");

            trainer_start_training(trainer);
            break;
        }
    }

    return NULL;
}

static bool validate_trainer_spec(const struct trainer_spec* spec) {
    if (!spec->training_data || !spec->test_data) {
        return false;
    }

    if (spec->batch_size == 0) {
        return false;
    }

    return true;
}

trainer_t* trainer_new(model_t* model, const char* disk_path, const struct trainer_spec* spec) {
    assert(model && spec);

    if (!validate_trainer_spec(spec)) {
        NV_LOG_ERROR("invalid trainer spec!");
        return NULL;
    }

    /* 32 mb */
    nv_arena_t* arena = nv_arena_create(32 * 1024 * 1024);

    trainer_t* trainer = nv_arena_alloc(arena, sizeof(trainer_t));
    assert(trainer);

    /* no nv_arena_strdup */
    if (disk_path) {
        size_t path_len = strlen(disk_path);
        size_t buffer_len = path_len + 1;

        trainer->disk_path = nv_arena_alloc(arena, buffer_len);
        strncpy(trainer->disk_path, disk_path, buffer_len);
    } else {
        trainer->disk_path = NULL;
    }

    trainer->pool = thread_pool_new(trainer_job, trainer);
    trainer->model = model;

    memcpy(&trainer->spec, spec, sizeof(struct trainer_spec));

    trainer->num_workers = thread_pool_get_num_workers(trainer->pool);
    assert(trainer->num_workers > 0);

    trainer->eval_contexts = nv_arena_alloc(arena, sizeof(eval_context_t*) * trainer->num_workers);
    trainer->training_contexts =
        nv_arena_alloc(arena, sizeof(eval_context_t*) * trainer->num_workers);
    assert(trainer->training_contexts && trainer->eval_contexts);

    for (uint32_t i = 0; i < trainer->num_workers; i++) {
        trainer->training_contexts[i] = eval_context_allocate(arena, model, EVAL_LEVEL_BACKPROP);
        trainer->eval_contexts[i] = eval_context_allocate(arena, model, EVAL_LEVEL_EVAL);
    }

    trainer->running = false;
    trainer->should_continue = false;

    pthread_mutex_init(&trainer->manager_mutex, NULL);
    pthread_mutex_init(&trainer->phase.mutex, NULL);
    pthread_mutex_init(&trainer->phase.training.gradients_mutex, NULL);

    trainer->phase.training.index_capacity = 0;
    trainer->phase.training.shuffled_indices = NULL;
    trainer->phase.training.batch_cost = 0.f;

    trainer->phase.training.gradients = model_gradients_alloc(arena, model);

    return trainer;
}

void trainer_destroy(trainer_t* trainer) {
    if (!trainer) {
        return;
    }

    trainer_stop(trainer);
    thread_pool_destroy(trainer->pool);

    pthread_mutex_destroy(&trainer->phase.training.gradients_mutex);
    pthread_mutex_destroy(&trainer->phase.mutex);
    pthread_mutex_destroy(&trainer->manager_mutex);

    /* will wipe out everything else owned by trainer */
    nv_arena_destroy(trainer->arena);
}

void trainer_get_spec(const trainer_t* trainer, struct trainer_spec* spec) {
    memcpy(spec, &trainer->spec, sizeof(struct trainer_spec));
}

bool trainer_set_spec(trainer_t* trainer, const struct trainer_spec* spec) {
    if (trainer->running) {
        NV_LOG_ERROR("failed to set spec; trainer running");
        return false;
    }

    if (!validate_trainer_spec(spec)) {
        NV_LOG_ERROR("invalid trainer spec!");
        return false;
    }

    memcpy(&trainer->spec, spec, sizeof(struct trainer_spec));
    return true;
}

void trainer_start(trainer_t* trainer) {
    if (trainer->running) {
        NV_LOG_WARN("failed to start; trainer already running");
        return;
    }

    NV_LOG_INFO("starting trainer with %u parallel workers", trainer->num_workers);

    trainer->running = true;
    trainer->should_continue = true;
    trainer_start_training(trainer);

    pthread_create(&trainer->manager_thread, NULL, manager_routine, trainer);
    pthread_detach(trainer->manager_thread);

    NV_LOG_DEBUG("trainer started");
}

void trainer_stop(trainer_t* trainer) {
    if (!trainer->running) {
        return;
    }

    NV_LOG_INFO("stopping trainer");

    pthread_mutex_lock(&trainer->manager_mutex);
    trainer->should_continue = false;
    pthread_mutex_unlock(&trainer->manager_mutex);

    pthread_join(trainer->manager_thread, NULL);
    trainer->running = false;

    NV_LOG_DEBUG("trainer stopped");
}

bool trainer_is_running(const trainer_t* trainer) { return trainer->running; }
