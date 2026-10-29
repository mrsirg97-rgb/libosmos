#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "substrate/mem/fetchq.h"

#define FETCH_SCRATCH (1u << 20)
#define FETCH_MAX_THREADS 64

typedef struct Job
{
  uint32_t unit;
  PoolRange part[POOL_PARTS];
  int n;
} Job;

typedef struct Worker
{
  struct Fetchq *q;
  pthread_t thread;
  uint8_t *scratch;
} Worker;

struct Fetchq
{
  pthread_mutex_t mu;
  pthread_cond_t cv;
  Job *jobs;
  size_t head;
  size_t len;
  size_t cap;
  bool stop;
  fetch_done_fn done;
  void *done_arg;
  Worker w[FETCH_MAX_THREADS];
  int n_workers;
};

static bool read_part(const PoolRange *r, uint8_t *scratch)
{
  uint64_t got = 0;
  while (got < r->bytes)
  {
    const uint64_t want = r->bytes - got < FETCH_SCRATCH ? r->bytes - got : FETCH_SCRATCH;
    const ssize_t n = pread(r->fd, scratch, (size_t)want, (off_t)(r->offset + got));
    if (n <= 0)
      return false;
    got += (uint64_t)n;
  }
  return true;
}

static void *work(void *arg)
{
  Worker *w = arg;
  Fetchq *q = w->q;
  for (;;)
  {
    pthread_mutex_lock(&q->mu);
    while (!q->stop && q->len == 0)
      pthread_cond_wait(&q->cv, &q->mu);
    if (q->stop)
    {
      pthread_mutex_unlock(&q->mu);
      return NULL;
    }
    const Job job = q->jobs[q->head];
    q->head = (q->head + 1) % q->cap;
    q->len--;
    pthread_mutex_unlock(&q->mu);
    bool ok = true;
    for (int k = 0; k < job.n && ok; k++)
      ok = read_part(&job.part[k], w->scratch);
    q->done(job.unit, ok, q->done_arg);
  }
}

static void stop(Fetchq *q)
{
  pthread_mutex_lock(&q->mu);
  q->stop = true;
  pthread_cond_broadcast(&q->cv);
  pthread_mutex_unlock(&q->mu);
  for (int i = 0; i < q->n_workers; i++)
    pthread_join(q->w[i].thread, NULL);
  q->n_workers = 0;
}

Fetchq *fetchq_create(int threads, fetch_done_fn done, void *done_arg)
{
  if (threads < 1 || threads > FETCH_MAX_THREADS || !done)
    return NULL;
  Fetchq *q = calloc(1, sizeof *q);
  if (!q)
    return NULL;
  pthread_mutex_init(&q->mu, NULL);
  pthread_cond_init(&q->cv, NULL);
  q->done = done;
  q->done_arg = done_arg;
  q->cap = 256;
  q->jobs = malloc(q->cap * sizeof *q->jobs);
  bool ok = q->jobs != NULL;
  for (int i = 0; ok && i < threads; i++)
    ok = (q->w[i].scratch = malloc(FETCH_SCRATCH)) != NULL;
  for (int i = 0; ok && i < threads; i++)
  {
    q->w[i].q = q;
    ok = pthread_create(&q->w[i].thread, NULL, work, &q->w[i]) == 0;
    if (ok)
      q->n_workers++;
  }
  if (!ok)
  {
    fetchq_destroy(q);
    return NULL;
  }
  return q;
}

void fetchq_destroy(Fetchq *q)
{
  if (!q)
    return;
  stop(q);
  for (int i = 0; i < FETCH_MAX_THREADS; i++)
    free(q->w[i].scratch);
  free(q->jobs);
  pthread_cond_destroy(&q->cv);
  pthread_mutex_destroy(&q->mu);
  free(q);
}

/* the consumer's half: a job queued, a worker woken. a queue that cannot grow reports the read as failed, at once */
static void fetch(uint32_t unit, const PoolRange *parts, int n, void *ctx)
{
  Fetchq *q = ctx;
  Job job = {unit, {{0}}, n < POOL_PARTS ? n : POOL_PARTS};
  memcpy(job.part, parts, (size_t)job.n * sizeof *parts);
  pthread_mutex_lock(&q->mu);
  if (q->len == q->cap)
  {
    Job *grown = malloc(q->cap * 2 * sizeof *grown);
    if (!grown)
    {
      pthread_mutex_unlock(&q->mu);
      q->done(unit, false, q->done_arg);
      return;
    }
    for (size_t i = 0; i < q->len; i++)
      grown[i] = q->jobs[(q->head + i) % q->cap];
    free(q->jobs);
    q->jobs = grown;
    q->head = 0;
    q->cap *= 2;
  }
  q->jobs[(q->head + q->len) % q->cap] = job;
  q->len++;
  pthread_cond_signal(&q->cv);
  pthread_mutex_unlock(&q->mu);
}

Fetcher fetchq_fetcher(Fetchq *q) { return (Fetcher){fetch, q}; }
