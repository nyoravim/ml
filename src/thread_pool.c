#include "thread_pool.h"

#include <stdbool.h>
#include <stdint.h>

#include <assert.h>

#include <nyoravim/mem.h>
#include <nyoravim/list.h>

#include <pthread.h>
#include <unistd.h>

struct worker {
    pthread_t thread_id;
    thread_pool_t* pool;
};

typedef struct thread_pool {
    pthread_mutex_t mutex;
    pthread_cond_t wake_signal;

    uint32_t num_workers;
    struct worker* workers;

    uint32_t num_idle;
    pthread_cond_t thread_idle_signal;
    struct nv_list job_queue;

    bool stopping;
    uint32_t num_stopped;
    pthread_cond_t thread_stopped_signal;

    thread_pool_callback callback;
    void* user;
} thread_pool_t;

/* assumes this thread owns the mutex lock. be careful! */
static void worker_work(thread_pool_t* pool) {
    while (pool->job_queue.head) {
        void* job = pool->job_queue.head->value;
        nv_list_remove(&pool->job_queue, pool->job_queue.head);

        /* wont be written to during runtime but better safe than sorry */
        thread_pool_callback callback = pool->callback;
        void* user = pool->user;

        /* stay unlocked while job is executing so that we dont hold up the pool coordination */
        pthread_mutex_unlock(&pool->mutex);

        callback(user, job);

        /* lock again to read */
        pthread_mutex_lock(&pool->mutex);
    }
}

static void* worker_routine(void* user) {
    struct worker* worker = user;
    thread_pool_t* pool = worker->pool;

    /* hold on to a lock for when this thread is not waiting */
    pthread_mutex_lock(&pool->mutex);

    while (!pool->stopping) {
        pool->num_idle++;
        pthread_cond_signal(&pool->thread_idle_signal);
        pthread_cond_wait(&pool->wake_signal, &pool->mutex);
        pool->num_idle--;

        worker_work(pool);
    }

    pool->num_stopped++;
    pthread_cond_signal(&pool->thread_stopped_signal);

    pthread_mutex_unlock(&pool->mutex);
    return NULL;
}

thread_pool_t* thread_pool_new(thread_pool_callback callback, void* user) {
    assert(callback);

    thread_pool_t* pool = nv_alloc(sizeof(thread_pool_t));
    assert(pool);

    pool->callback = callback;
    pool->user = user;

    pool->stopping = false;
    pool->num_stopped = 0;
    pool->num_idle = 0;

    nv_list_init(&pool->job_queue);

    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->wake_signal, NULL);
    pthread_cond_init(&pool->thread_idle_signal, NULL);
    pthread_cond_init(&pool->thread_stopped_signal, NULL);

    pool->num_workers = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);
    pool->workers = nv_alloc(sizeof(struct worker) * pool->num_workers);
    assert(pool->workers);

    for (uint32_t i = 0; i < pool->num_workers; i++) {
        struct worker* worker = &pool->workers[i];
        worker->pool = pool;

        pthread_create(&worker->thread_id, NULL, worker_routine, worker);
        pthread_detach(worker->thread_id);
    }

    return pool;
}

static void stop_pool(thread_pool_t* pool) {
    pthread_mutex_lock(&pool->mutex);

    pool->stopping = true;
    pthread_cond_broadcast(&pool->wake_signal);

    /* wait for threads to stop */
    while (pool->num_stopped < pool->num_workers) {
        pthread_cond_wait(&pool->thread_stopped_signal, &pool->mutex);
    }

    /* all threads stopped now */
    pthread_mutex_unlock(&pool->mutex);
}

void thread_pool_destroy(thread_pool_t* pool) {
    if (!pool) {
        return;
    }

    stop_pool(pool);

    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->wake_signal);
    pthread_cond_destroy(&pool->thread_idle_signal);
    pthread_cond_destroy(&pool->thread_stopped_signal);

    nv_free(pool->workers);
    nv_free(pool);
}

void thread_pool_push_job(thread_pool_t* pool, void* job) {
    pthread_mutex_lock(&pool->mutex);

    nv_list_push_back(&pool->job_queue, job);
    pthread_cond_signal(&pool->wake_signal);

    pthread_mutex_unlock(&pool->mutex);
}

void thread_pool_wait_idle(thread_pool_t* pool) {
    pthread_mutex_lock(&pool->mutex);

    while (pool->num_idle < pool->num_stopped) {
        pthread_cond_wait(&pool->thread_idle_signal, &pool->mutex);
    }

    pthread_mutex_unlock(&pool->mutex);
}

uint32_t thread_pool_get_num_workers(const thread_pool_t* pool) { return pool->num_workers; }
