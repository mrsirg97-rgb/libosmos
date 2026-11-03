#include <stdlib.h>
#include <time.h>

#include "forward_impl.h"

/* a fetchq worker's read is done: the unit goes to the inbox, for the forward to hand the pool */
void fwd_on_read(uint32_t unit, bool ok, void *arg)
{
  Inbox *in = arg;
  pthread_mutex_lock(&in->mu);
  if (in->n == in->cap)
  {
    const size_t cap = in->cap ? in->cap * 2 : 64;
    Read *grown = realloc(in->r, cap * sizeof *grown);
    if (grown)
      in->r = grown, in->cap = cap;
  }
  if (in->n < in->cap)
    in->r[in->n++] = (Read){unit, ok};
  else
    in->broken = true;
  pthread_cond_signal(&in->cv);
  pthread_mutex_unlock(&in->mu);
}

/* the reads finished so far handed to the pool, with no wait: a read begun ahead (pool_want) and never waited for is
   READING until it is - and a reading unit's memory cannot be made room of. each layer's routing calls it, so the
   prefetched reads land whether or not the forward ever waits on one (with the resident set following the text, it
   seldom does) */
bool fwd_drain(Forward *f, char *err)
{
  pthread_mutex_lock(&f->in.mu);
  const bool broken = f->in.broken;
  Read got[64];
  for (;;)
  {
    size_t n = 0;
    while (f->in.n && n < sizeof got / sizeof *got)
      got[n++] = f->in.r[--f->in.n];
    if (!n)
      break;
    pthread_mutex_unlock(&f->in.mu);
    for (size_t i = 0; i < n; i++)
      pool_ready(f->pool, got[i].unit, got[i].ok);
    pthread_mutex_lock(&f->in.mu);
  }
  pthread_mutex_unlock(&f->in.mu);
  return !broken || fwd_fail(err, "a finished read the inbox could not hold");
}

/* at least one read's end waited for, and every finished read handed to the pool */
bool fwd_wait_read(Forward *f, char *err)
{
  pthread_mutex_lock(&f->in.mu);
  while (!f->in.n && !f->in.broken)
    pthread_cond_wait(&f->in.cv, &f->in.mu);
  const bool broken = f->in.broken;
  Read got[64];
  size_t n = 0;
  while (f->in.n && n < sizeof got / sizeof *got)
    got[n++] = f->in.r[--f->in.n];
  pthread_mutex_unlock(&f->in.mu);
  if (broken)
    return fwd_fail(err, "a finished read the inbox could not hold");
  for (size_t i = 0; i < n; i++)
    pool_ready(f->pool, got[i].unit, got[i].ok);
  return true;
}

/* the unit held and pinned: its reads handed to the pool as they finish, until it is (or its read failed) */
static uint64_t now_ns(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

bool fwd_pinned(Forward *f, uint32_t unit, const void *dev[POOL_PARTS], char *err)
{
  if (pool_state(f->pool, unit) == POOL_READING)
  {
    const int k = f->guessed && f->guessed[unit] == f->visits; /* (the guess's arrays: a decode's alone) */
    const uint64_t t0 = now_ns();
    while (pool_state(f->pool, unit) == POOL_READING)
      if (!fwd_wait_read(f, err))
        return false;
    f->w_n[k]++, f->w_ns[k] += now_ns() - t0;
  }
  return pool_dev(f->pool, unit, dev) || fwd_fail(err, "an expert's read from its file failed");
}

void fwd_release_held(Forward *f)
{
  for (uint32_t k = 0; k < f->n_held; k++) /* the layer they came from next visited a whole round on */
  {
    pool_release(f->pool, f->held[k], f->visits - 1 + (uint64_t)f->n_layers);
    f->is_held[f->held[k]] = false;
  }
  f->n_held = 0;
  f->held_layer = -1;
}
