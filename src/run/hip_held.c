#include <stdlib.h>
#include <time.h>

#include "forward_impl.h"

/* the reads finished so far handed to the pool, with no wait: a read begun ahead (pool_want) and never waited for is
   READING until it is - and a reading unit's memory cannot be made room of. each layer's routing calls it, so the
   prefetched reads land whether or not the forward ever waits on one (with the resident set following the text, it
   seldom does) - and each end frees a slot, the next read needed soonest given out (reads.h) */
bool fwd_drain(Forward *f, char *err)
{
  char rerr[READS_ERR];
  return reads_pump(f->reads, f->pool, false, rerr) || fwd_fail(err, rerr);
}

/* at least one read's end waited for, and every finished read handed to the pool */
bool fwd_wait_read(Forward *f, char *err)
{
  char rerr[READS_ERR];
  return reads_pump(f->reads, f->pool, true, rerr) || fwd_fail(err, rerr);
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
