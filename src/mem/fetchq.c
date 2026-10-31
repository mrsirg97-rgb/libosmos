#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "substrate/mem/fetchq.h"

#define FETCH_MAX_THREADS 64
#define FETCH_RING_START 256 /* a ring's first jobs: a decode's reads are a few dozen, a load's grow it */

/* a unit's read: its chunks left, and whether any failed - the last chunk's worker calls done and frees it */
typedef struct Unit
{
  uint32_t unit;
  atomic_int left;
  atomic_bool bad;
} Unit;

/* a job: one chunk of one part of a unit (FETCHQ_CHUNK bytes, the part's last chunk the rest) */
typedef struct Job
{
  Unit *u;
  PoolRange r;
  uint8_t *dst;
} Job;

typedef struct Worker
{
  struct Fetchq *q;
  pthread_t thread;
} Worker;

/* a queue of jobs, oldest first (a ring) */
typedef struct Ring
{
  Job *jobs;
  size_t head;
  size_t len;
  size_t cap;
} Ring;

/* two queues: the reads a pin waits on (urgent) and those read ahead (pool_want's), the workers taking the urgent
   first - a decode's miss is not read behind the guesses queued before it */
struct Fetchq
{
  pthread_mutex_t mu;
  pthread_cond_t cv;
  Ring urgent, ahead;
  bool stop;
  fetch_done_fn done;
  void *done_arg;
  Worker w[FETCH_MAX_THREADS];
  int n_workers;
};

/* the ring grown to hold `more` jobs; false if it cannot */
static bool room(Ring *r, size_t more)
{
  if (r->len + more <= r->cap)
    return true;
  size_t cap = r->cap ? r->cap : FETCH_RING_START;
  while (r->len + more > cap)
    cap *= 2;
  Job *grown = malloc(cap * sizeof *grown);
  if (!grown)
    return false;
  for (size_t i = 0; i < r->len; i++)
    grown[i] = r->jobs[(r->head + i) % r->cap];
  free(r->jobs);
  r->jobs = grown, r->head = 0, r->cap = cap;
  return true;
}

static Job *at(Ring *r, size_t i) { return &r->jobs[(r->head + i) % r->cap]; }

static Job take(Ring *r)
{
  const Job j = r->jobs[r->head];
  r->head = (r->head + 1) % r->cap, r->len--;
  return j;
}

static bool read_part(const PoolRange *r, uint8_t *dst)
{
  uint64_t got = 0;
  while (got < r->bytes)
  {
    const ssize_t n = pread(r->fd, dst + got, (size_t)(r->bytes - got), (off_t)(r->offset + got));
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
    while (!q->stop && !q->urgent.len && !q->ahead.len)
      pthread_cond_wait(&q->cv, &q->mu);
    if (q->stop)
    {
      pthread_mutex_unlock(&q->mu);
      return NULL;
    }
    const Job job = take(q->urgent.len ? &q->urgent : &q->ahead);
    pthread_mutex_unlock(&q->mu);
    if (!read_part(&job.r, job.dst))
      atomic_store(&job.u->bad, true);
    if (atomic_fetch_sub(&job.u->left, 1) == 1) /* the unit's last chunk */
    {
      const uint32_t unit = job.u->unit;
      const bool ok = !atomic_load(&job.u->bad);
      free(job.u);
      q->done(unit, ok, q->done_arg);
    }
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
  bool ok = room(&q->urgent, FETCH_RING_START) && room(&q->ahead, FETCH_RING_START);
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
  Ring *rings[2] = {&q->urgent, &q->ahead};
  for (int k = 0; k < 2; k++) /* the chunks dropped: a unit freed once its last is, no done */
  {
    for (size_t i = 0; i < rings[k]->len; i++)
    {
      Unit *u = at(rings[k], i)->u;
      if (atomic_fetch_sub(&u->left, 1) == 1)
        free(u);
    }
    free(rings[k]->jobs);
  }
  pthread_cond_destroy(&q->cv);
  pthread_mutex_destroy(&q->mu);
  free(q);
}

/* the consumer's half: a unit's jobs queued side by side, on the urgent queue or the ahead one, and the workers
   woken. when its queue is idle (fewer jobs than workers: a lone unit, a decode's miss) its parts go a FETCHQ_CHUNK a
   job, the workers reading one unit at once - at the SSD's speed, not a worker's; when reads are queued (a load, a
   prompt's) each part is one job, read whole by one worker - chunks of many units side by side defeat the file's
   readahead (measured: a load 16 -> 22 s). an urgent read's queue is the urgent one alone: the ahead reads wait for
   it. a queue that cannot grow reports the read as failed, at once */
static void fetch(uint32_t unit, const PoolRange *parts, int n, void *const dst[POOL_PARTS], bool urgent, void *ctx)
{
  Fetchq *q = ctx;
  n = n < POOL_PARTS ? n : POOL_PARTS;
  pthread_mutex_lock(&q->mu);
  const size_t queued = q->urgent.len + (urgent ? 0 : q->ahead.len);
  const uint64_t chunk = queued < (size_t)q->n_workers ? FETCHQ_CHUNK : UINT64_MAX;
  pthread_mutex_unlock(&q->mu);
  int chunks = 0;
  for (int k = 0; k < n; k++)
    chunks += (int)(chunk == UINT64_MAX ? parts[k].bytes > 0 : (parts[k].bytes + chunk - 1) / chunk);
  if (!chunks) /* nothing to read: done, ok */
  {
    q->done(unit, true, q->done_arg);
    return;
  }
  Unit *u = malloc(sizeof *u);
  Ring *r = urgent ? &q->urgent : &q->ahead;
  pthread_mutex_lock(&q->mu);
  if (!u || !room(r, (size_t)chunks))
  {
    pthread_mutex_unlock(&q->mu);
    free(u);
    q->done(unit, false, q->done_arg);
    return;
  }
  u->unit = unit;
  atomic_init(&u->left, chunks);
  atomic_init(&u->bad, false);
  for (int k = 0; k < n; k++)
    for (uint64_t off = 0; off < parts[k].bytes; off += chunk)
    {
      const uint64_t len = parts[k].bytes - off < chunk ? parts[k].bytes - off : chunk;
      *at(r, r->len++) = (Job){u, {parts[k].fd, parts[k].offset + off, len}, (uint8_t *)dst[k] + off};
    }
  pthread_cond_broadcast(&q->cv);
  pthread_mutex_unlock(&q->mu);
}

/* a read ahead now waited on: its jobs not yet begun moved to the urgent queue's end, in their order (the ahead
   queue is a decode's few dozen: a scan). a unit not queued (read, being read, never asked) is nothing to do */
static void hurry(uint32_t unit, void *ctx)
{
  Fetchq *q = ctx;
  pthread_mutex_lock(&q->mu);
  size_t kept = 0, moved = 0;
  for (size_t i = 0; i < q->ahead.len; i++)
    moved += at(&q->ahead, i)->u->unit == unit;
  if (moved && room(&q->urgent, moved))
  {
    for (size_t i = 0; i < q->ahead.len; i++)
    {
      const Job j = *at(&q->ahead, i);
      if (j.u->unit == unit)
        *at(&q->urgent, q->urgent.len++) = j;
      else
        *at(&q->ahead, kept++) = j;
    }
    q->ahead.len = kept;
  }
  pthread_mutex_unlock(&q->mu);
}

Fetcher fetchq_fetcher(Fetchq *q) { return (Fetcher){fetch, q, hurry}; }
