#include <stdlib.h>

#include "forward_impl.h"

/* a fetchq worker's read is done: the unit goes to the inbox, for the forward to hand the pool */
void fwd_on_read(uint32_t unit, bool ok, void *arg)
{
  (void)ok; /* a failed read is the pin's own read to do: the pool reads through the registration */
  Inbox *in = arg;
  pthread_mutex_lock(&in->mu);
  if (in->n == in->cap)
  {
    const size_t cap = in->cap ? in->cap * 2 : 64;
    uint32_t *grown = realloc(in->unit, cap * sizeof *grown);
    if (grown)
      in->unit = grown, in->cap = cap;
  }
  if (in->n < in->cap)
    in->unit[in->n++] = unit;
  else
    in->broken = true;
  pthread_cond_signal(&in->cv);
  pthread_mutex_unlock(&in->mu);
}

/* the unit held and pinned: its reads handed to the pool as they finish, until it is (or is refused) */
bool fwd_pinned(Forward *f, uint32_t unit, const void *dev[POOL_PARTS], char *err)
{
  while (pool_state(f->pool, unit) == POOL_READING)
  {
    pthread_mutex_lock(&f->in.mu);
    while (!f->in.n && !f->in.broken)
      pthread_cond_wait(&f->in.cv, &f->in.mu);
    const bool broken = f->in.broken;
    uint32_t got[64];
    size_t n = 0;
    while (f->in.n && n < sizeof got / sizeof *got)
      got[n++] = f->in.unit[--f->in.n];
    pthread_mutex_unlock(&f->in.mu);
    if (broken)
      return fwd_fail(err, "a finished read the inbox could not hold");
    for (size_t i = 0; i < n; i++)
      pool_ready(f->pool, got[i]);
  }
  return pool_dev(f->pool, unit, dev) || fwd_fail(err, "a unit the pool could not pin (its budget)");
}

void fwd_release_held(Forward *f)
{
  for (uint32_t k = 0; k < f->n_held; k++) /* the layer they came from next visited a whole round on */
    pool_release(f->pool, f->held[k], f->visits - 1 + (uint64_t)f->n_layers);
  f->n_held = 0;
}
