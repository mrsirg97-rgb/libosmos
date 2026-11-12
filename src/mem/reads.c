#include "osmos/mem/reads.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#include "libevt/runtime/context.h"
#include "libevt/scheduler/engine.h"
#include "osmos/mem/fetchq.h"

#define READS_PRIO 1 /* the engine carries one kind of event: a read's end */

typedef struct Read
{
  struct Reads *r;
  uint32_t unit;
  bool out, pin; /* given to the readers (its end not yet pumped); waited on (need 0) */
  int pool;      /* the pool it is out for (reads_fetcher's) */
  bool ok;       /* written by the reader before it posts the end; read by the consumer after the engine hands it over */
} Read;

struct Reads
{
  Engine *evt;
  Fetchq *q;
  Read *rd;
  uint32_t n_units;
  int out, pins; /* the reads out; of them, the pins */
  uint64_t split;
  Pool *pools[READS_POOLS]; /* the pump's: pool_ready's */
  struct ReadsFor
  {
    struct Reads *r;
    int pool;
  } fors[READS_POOLS]; /* each pool's Fetcher context */
  atomic_bool broken;
};

static void ended(void *scope);

/* a reader's end (any thread): an event for the consumer. an end the engine cannot take stops it, so a consumer waiting
   in reads_pump wakes and fails closed rather than waiting for a read that never comes */
static void post_end(uint32_t unit, bool ok, void *arg)
{
  Reads *r = arg;
  Read *rd = &r->rd[unit];
  rd->ok = ok;
  Context *ctx = context_create(rd, ended);
  if (ctx && engine_add(r->evt, ctx, READS_PRIO))
    return;
  context_destroy(ctx);
  atomic_store(&r->broken, true);
  engine_stop(r->evt);
}

/* the consumer, in the pump: the read's end to the pool */
static void ended(void *scope)
{
  Read *rd = scope;
  Reads *r = rd->r;
  r->out--, r->pins -= rd->pin;
  rd->out = false, rd->pin = false;
  Pool *p = r->pools[rd->pool];
  if (!p) /* the pump not given the pool it was read for: fail closed */
  {
    atomic_store(&r->broken, true);
    engine_stop(r->evt);
    return;
  }
  pool_ready(p, rd->unit, rd->ok);
}

/* to the readers at once, at its need: a pin split unless another pin is out, a read ahead split only alone (none other
   out) - reads side by side read whole. one the readers refuse ends failed, never lost */
static void fetch(uint32_t unit, const PoolRange *parts, int n, void *const dst[POOL_PARTS], uint64_t need, void *ctx)
{
  const struct ReadsFor *fr = ctx;
  Reads *r = fr->r;
  if (unit >= r->n_units || r->rd[unit].out) /* not the pool's contract: the forward fails closed */
  {
    atomic_store(&r->broken, true);
    engine_stop(r->evt);
    return;
  }
  Read *rd = &r->rd[unit];
  rd->out = true, rd->pin = need == 0, rd->pool = fr->pool, r->out++, r->pins += rd->pin;
  const bool split = rd->pin ? r->pins == 1 : r->out == 1;
  r->split += split;
  if (!fetchq_read(r->q, unit, parts, n, dst, split, need))
    post_end(unit, false, r);
}

/* a read's need moved (a want's next corrected, or now waited on): its jobs not yet begun move with it */
static void need(uint32_t unit, uint64_t need, void *ctx)
{
  Reads *r = ((const struct ReadsFor *)ctx)->r;
  if (unit >= r->n_units || !r->rd[unit].out)
    return;
  Read *rd = &r->rd[unit];
  if (need == 0 && !rd->pin)
    rd->pin = true, r->pins++;
  fetchq_need(r->q, unit, need);
}

Reads *reads_create(uint32_t n_units, int threads, char err[READS_ERR])
{
  if (!n_units)
    return snprintf(err, READS_ERR, "reads: no units"), NULL;
  Reads *r = calloc(1, sizeof *r);
  if (!r || !(r->rd = calloc(n_units, sizeof *r->rd)))
  {
    free(r);
    return snprintf(err, READS_ERR, "reads: no memory for %u reads", n_units), NULL;
  }
  r->n_units = n_units;
  atomic_init(&r->broken, false);
  for (uint32_t i = 0; i < n_units; i++)
    r->rd[i] = (Read){.r = r, .unit = i};
  for (int k = 0; k < READS_POOLS; k++)
    r->fors[k] = (struct ReadsFor){r, k};
  r->evt = engine_create(NULL, NULL, 0);
  r->q = r->evt ? fetchq_create(threads, n_units, post_end, r) : NULL;
  if (!r->q)
  {
    snprintf(err, READS_ERR, "reads: the readers (%d) or the engine could not start", threads);
    reads_destroy(r);
    return NULL;
  }
  return r;
}

void reads_destroy(Reads *r)
{
  if (!r)
    return;
  fetchq_destroy(r->q); /* no end is posted after: the queued and the posted ones are dropped, never resolved */
  engine_destroy(r->evt);
  free(r->rd);
  free(r);
}

Fetcher reads_fetcher(Reads *r, int pool)
{
  return pool >= 0 && pool < READS_POOLS ? (Fetcher){fetch, &r->fors[pool], need} : (Fetcher){0};
}

bool reads_pump(Reads *r, Pool *const pools[READS_POOLS], bool wait, char err[READS_ERR])
{
  if (!r || !pools || !(pools[0] || pools[1]))
    return snprintf(err, READS_ERR, "reads: no reads or no pool to pump into"), false;
  for (int k = 0; k < READS_POOLS; k++)
    r->pools[k] = pools[k];
  if (wait && !atomic_load(&r->broken))
    engine_pump(r->evt, -1);
  while (!atomic_load(&r->broken) && engine_pump(r->evt, 0))
    ;
  if (atomic_load(&r->broken))
    return snprintf(err, READS_ERR, "reads: a read's end could not be posted (no memory)"), false;
  return true;
}

int reads_out(const Reads *r) { return r ? r->out : 0; }
uint64_t reads_split(const Reads *r) { return r ? r->split : 0; }

void reads_ahead_limit(Reads *r, uint64_t bytes)
{
  if (r)
    fetchq_ahead_limit(r->q, bytes);
}
