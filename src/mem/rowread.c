#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>

#include "substrate/mem/rowread.h"

#define ROWREAD_TAKE 16 /* rows a worker takes at a time: few trips to the shared counter, the ends of a batch even */

struct Rowread
{
  pthread_mutex_t mu;
  pthread_cond_t go, done;
  pthread_t th[ROWREAD_MAX_THREADS];
  int n_th, busy;
  uint64_t batch; /* a call's number: a worker's cue that there is a new one */
  bool stop;
  int fd;
  const uint64_t *offset;
  uint32_t n, bytes;
  uint8_t *dst;
  atomic_uint next;
  atomic_bool failed;
};

static void read_rows(Rowread *r)
{
  for (;;)
  {
    const uint32_t a = atomic_fetch_add(&r->next, ROWREAD_TAKE);
    if (a >= r->n)
      return;
    const uint32_t z = a + ROWREAD_TAKE < r->n ? a + ROWREAD_TAKE : r->n;
    for (uint32_t i = a; i < z; i++)
      if (pread(r->fd, r->dst + (uint64_t)i * r->bytes, r->bytes, (off_t)r->offset[i]) != (ssize_t)r->bytes)
        atomic_store(&r->failed, true);
  }
}

static void *work(void *arg)
{
  Rowread *r = arg;
  uint64_t seen = 0;
  pthread_mutex_lock(&r->mu);
  for (;;)
  {
    while (!r->stop && r->batch == seen)
      pthread_cond_wait(&r->go, &r->mu);
    if (r->stop)
      break;
    seen = r->batch;
    r->busy++;
    pthread_mutex_unlock(&r->mu);
    read_rows(r);
    pthread_mutex_lock(&r->mu);
    if (--r->busy == 0)
      pthread_cond_signal(&r->done);
  }
  pthread_mutex_unlock(&r->mu);
  return NULL;
}

Rowread *rowread_create(int threads)
{
  if (threads < 1 || threads > ROWREAD_MAX_THREADS)
    return NULL;
  Rowread *r = calloc(1, sizeof *r);
  if (!r)
    return NULL;
  pthread_mutex_init(&r->mu, NULL);
  pthread_cond_init(&r->go, NULL);
  pthread_cond_init(&r->done, NULL);
  for (int i = 0; i < threads; i++)
    if (pthread_create(&r->th[i], NULL, work, r) == 0)
      r->n_th++;
    else
    {
      rowread_destroy(r);
      return NULL;
    }
  return r;
}

void rowread_destroy(Rowread *r)
{
  if (!r)
    return;
  pthread_mutex_lock(&r->mu);
  r->stop = true;
  pthread_cond_broadcast(&r->go);
  pthread_mutex_unlock(&r->mu);
  for (int i = 0; i < r->n_th; i++)
    pthread_join(r->th[i], NULL);
  pthread_cond_destroy(&r->go);
  pthread_cond_destroy(&r->done);
  pthread_mutex_destroy(&r->mu);
  free(r);
}

bool rowread(Rowread *r, int fd, const uint64_t *offset, uint32_t n, uint32_t bytes, uint8_t *dst)
{
  if (!r || n == 0)
    return r != NULL;
  pthread_mutex_lock(&r->mu);
  r->fd = fd, r->offset = offset, r->n = n, r->bytes = bytes, r->dst = dst;
  atomic_store(&r->next, 0);
  atomic_store(&r->failed, false);
  r->batch++;
  pthread_cond_broadcast(&r->go);
  /* done when every row is taken and no worker is still in one: a worker that never woke for this batch finds
     nothing left to take */
  while (r->busy > 0 || atomic_load(&r->next) < n)
  {
    pthread_mutex_unlock(&r->mu);
    read_rows(r); /* the caller reads too, rather than only wait */
    pthread_mutex_lock(&r->mu);
    while (r->busy > 0)
      pthread_cond_wait(&r->done, &r->mu);
  }
  pthread_mutex_unlock(&r->mu);
  return !atomic_load(&r->failed);
}
