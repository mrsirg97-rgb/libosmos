#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>

#include "libevt/runtime/context.h"
#include "libevt/runtime/event.h"
#include "libevt/runtime/queue.h"
#include "osmos/mem/fetchq.h"

#define FETCH_MAX_THREADS 64
#define FETCH_QUEUE_START 256 /* the heap's first jobs: a decode's reads are a few dozen, a load's grow it */

typedef struct Unit Unit;

/* a job: one chunk of one part of a unit (FETCHQ_CHUNK bytes, the part's last chunk the rest) - its event while it
   is queued, NULL once a worker took it */
typedef struct Job
{
  Unit *u;
  PoolRange r;
  uint8_t *dst;
  Event *ev;
} Job;

/* a unit's read: its jobs, those left to read, whether any failed, those still queued - the last job's worker calls
   done and frees it */
struct Unit
{
  uint32_t unit;
  atomic_int left;
  atomic_bool bad;
  int n_jobs, queued;
  Job jobs[];
};

typedef struct Worker
{
  struct Fetchq *q;
  pthread_t thread;
} Worker;

/* the jobs on libevt's heap by their read's need; a unit with jobs queued in `queued` (fetchq_need finds them) */
struct Fetchq
{
  pthread_mutex_t mu;
  pthread_cond_t cv;
  Queue *jobs;
  Unit **queued;
  uint32_t n_units;
  uint64_t given; /* the events' ids: the order given, a tie's order */
  bool stop;
  fetch_done_fn done;
  void *done_arg;
  Worker w[FETCH_MAX_THREADS];
  int n_workers;
  uint64_t ahead_limit, ahead_out; /* the bound, and the reads ahead's bytes in flight (fetchq_ahead_limit) */
  FetchqStats stats;
};

/* the heap pops the highest: need 0 at the top, need n below it by n (clamped at 0) */
static int priority(uint64_t need)
{
  if (!need)
    return INT_MAX;
  const uint64_t past = need - 1, most = (uint64_t)INT_MAX - 1;
  return INT_MAX - 1 - (int)(past < most ? past : most);
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

/* a job off the heap (the consumer's lock held): no longer findable by fetchq_need */
static Job *take(Fetchq *q, Event *ev)
{
  Job *j = ev->ctx->scope;
  j->ev = NULL;
  if (--j->u->queued == 0)
    q->queued[j->u->unit] = NULL;
  event_destroy(ev);
  return j;
}

/* the queue's first job a read ahead the bound keeps queued (the lock held): reads ahead are in flight and its bytes
   would pass the bound. a waited job is the heap's first whenever one is queued, so it is never behind this */
static bool bound(const Fetchq *q)
{
  const Event *ev = queue_peek(q->jobs);
  if (!q->ahead_limit || !q->ahead_out || !ev || ev->priority == INT_MAX)
    return false;
  return q->ahead_out + ((const Job *)ev->ctx->scope)->r.bytes > q->ahead_limit;
}

static void *work(void *arg)
{
  Worker *w = arg;
  Fetchq *q = w->q;
  for (;;)
  {
    pthread_mutex_lock(&q->mu);
    for (bool counted = false; !q->stop && (!q->jobs->len || bound(q)); counted = q->jobs->len != 0)
    {
      q->stats.ahead_held += q->jobs->len && !counted; /* once a stay, not once a wake */
      pthread_cond_wait(&q->cv, &q->mu);
    }
    if (q->stop)
    {
      pthread_mutex_unlock(&q->mu);
      return NULL;
    }
    Event *ev = NULL;
    queue_pop(q->jobs, &ev);
    const bool ahead = ev->priority != INT_MAX; /* a read ahead: its bytes in flight until it is read */
    Job *j = take(q, ev);
    if (ahead)
    {
      q->ahead_out += j->r.bytes;
      q->stats.ahead_peak = q->ahead_out > q->stats.ahead_peak ? q->ahead_out : q->stats.ahead_peak;
    }
    pthread_mutex_unlock(&q->mu);
    Unit *u = j->u;
    if (!read_part(&j->r, j->dst))
      atomic_store(&u->bad, true);
    if (ahead)
    {
      pthread_mutex_lock(&q->mu);
      q->ahead_out -= j->r.bytes;
      pthread_cond_broadcast(&q->cv); /* a worker that left a read ahead for the bound may take it now */
      pthread_mutex_unlock(&q->mu);
    }
    if (atomic_fetch_sub(&u->left, 1) == 1) /* the unit's last job */
    {
      const uint32_t unit = u->unit;
      const bool ok = !atomic_load(&u->bad);
      free(u);
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

Fetchq *fetchq_create(int threads, uint32_t n_units, fetch_done_fn done, void *done_arg)
{
  if (threads < 1 || threads > FETCH_MAX_THREADS || !n_units || !done)
    return NULL;
  Fetchq *q = calloc(1, sizeof *q);
  if (!q)
    return NULL;
  pthread_mutex_init(&q->mu, NULL);
  pthread_cond_init(&q->cv, NULL);
  q->done = done, q->done_arg = done_arg, q->n_units = n_units;
  bool ok = (q->jobs = queue_create(FETCH_QUEUE_START)) && (q->queued = calloc(n_units, sizeof *q->queued));
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
  for (Event *ev = NULL; q->jobs && q->jobs->len;) /* the jobs dropped: a unit freed once its last is, no done */
  {
    queue_pop(q->jobs, &ev);
    Unit *u = take(q, ev)->u;
    if (atomic_fetch_sub(&u->left, 1) == 1)
      free(u);
  }
  queue_destroy(q->jobs);
  free(q->queued);
  pthread_cond_destroy(&q->cv);
  pthread_mutex_destroy(&q->mu);
  free(q);
}

/* a unit's jobs on the heap at its need, side by side (consecutive ids), and the workers woken. split (a lone unit, a
   decode's miss): its parts a FETCHQ_CHUNK a job, the workers reading one unit at once - at the SSD's speed, not a
   worker's; else each part one job, read whole by one worker - chunks of many units side by side defeat the file's
   readahead (measured: a load 16 -> 22 s). which it is, the giver knows (reads.h) */
bool fetchq_read(Fetchq *q, uint32_t unit, const PoolRange *parts, int n, void *const dst[POOL_PARTS], bool split,
                 uint64_t need)
{
  if (!q || !parts || n < 0 || unit >= q->n_units)
    return false;
  n = n < POOL_PARTS ? n : POOL_PARTS;
  const uint64_t chunk = split ? FETCHQ_CHUNK : UINT64_MAX;
  int chunks = 0;
  for (int k = 0; k < n; k++)
    chunks += (int)(chunk == UINT64_MAX ? parts[k].bytes > 0 : (parts[k].bytes + chunk - 1) / chunk);
  if (!chunks) /* nothing to read: done, ok */
  {
    q->done(unit, true, q->done_arg);
    return true;
  }
  Unit *u = malloc(sizeof *u + (size_t)chunks * sizeof(Job));
  if (!u)
    return false;
  u->unit = unit, u->n_jobs = 0, u->queued = 0;
  atomic_init(&u->bad, false);
  for (int k = 0; k < n; k++)
    for (uint64_t off = 0; off < parts[k].bytes; off += chunk)
    {
      const uint64_t len = parts[k].bytes - off < chunk ? parts[k].bytes - off : chunk;
      u->jobs[u->n_jobs++] = (Job){u, {parts[k].fd, parts[k].offset + off, len}, (uint8_t *)dst[k] + off, NULL};
    }
  const int prio = priority(need);
  pthread_mutex_lock(&q->mu);
  if (q->queued[unit]) /* a read of it queued already: the pool never asks twice - refused */
  {
    pthread_mutex_unlock(&q->mu);
    free(u);
    return false;
  }
  for (int i = 0; i < u->n_jobs; i++)
  {
    Context *c = context_create(&u->jobs[i], NULL);
    Event *ev = c ? event_create(++q->given, prio, c) : NULL;
    if (!ev || !queue_push(q->jobs, ev))
    {
      if (ev)
        event_destroy(ev);
      else
        context_destroy(c);
      break;
    }
    u->jobs[i].ev = ev, u->queued++;
  }
  if (!u->queued) /* none could be queued: nothing out, the caller fails it closed */
  {
    pthread_mutex_unlock(&q->mu);
    free(u);
    return false;
  }
  if (u->queued < u->n_jobs) /* refused partway: the read ends failed once its queued jobs are read */
    atomic_store(&u->bad, true);
  atomic_init(&u->left, u->queued);
  q->queued[unit] = u;
  pthread_cond_broadcast(&q->cv);
  pthread_mutex_unlock(&q->mu);
  return true;
}

void fetchq_ahead_limit(Fetchq *q, uint64_t bytes)
{
  if (!q)
    return;
  pthread_mutex_lock(&q->mu);
  q->ahead_limit = bytes;
  pthread_cond_broadcast(&q->cv); /* a wider bound: the workers that left a read ahead look again */
  pthread_mutex_unlock(&q->mu);
}

void fetchq_stats(Fetchq *q, FetchqStats *s)
{
  if (!q || !s)
    return;
  pthread_mutex_lock(&q->mu);
  *s = q->stats;
  pthread_mutex_unlock(&q->mu);
}

void fetchq_need(Fetchq *q, uint32_t unit, uint64_t need)
{
  if (!q || unit >= q->n_units)
    return;
  const int prio = priority(need);
  pthread_mutex_lock(&q->mu);
  Unit *u = q->queued[unit];
  for (int i = 0; u && i < u->n_jobs; i++)
    if (u->jobs[i].ev)
      queue_update(q->jobs, u->jobs[i].ev, prio);
  pthread_cond_broadcast(&q->cv); /* a job the bound kept queued may be a waited one now */
  pthread_mutex_unlock(&q->mu);
}
