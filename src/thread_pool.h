#ifndef _THREAD_POOL_H
#define _THREAD_POOL_H

typedef void (*thread_pool_callback)(void* user, void* job);
typedef struct thread_pool thread_pool_t;

thread_pool_t* thread_pool_new(thread_pool_callback callback, void* user);
void thread_pool_destroy(thread_pool_t* pool);

void thread_pool_push_job(thread_pool_t* pool, void* job);
void thread_pool_wait_idle(thread_pool_t* pool);

#endif
