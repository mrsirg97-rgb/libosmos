#include "osmos/mem/reads.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#include "libevt/runtime/context.h"
#include "libevt/scheduler/cap.h"
#include "libevt/scheduler/engine.h"
#include "osmos/mem/fetchq.h"

#define READS_PRIO 1 /* the engine carries one kind of event: a read's end */

typedef enum
{
  READ_IDLE = 0,
  READ_WAITING = 1, /* asked, waiting in the cap for a slot */
  READ_OUT = 2,     /* given to the readers: its end comes back as an event */
} ReadState;

typedef struct Read
{
  struct Reads *r;
  uint32_t unit;
  ReadState st;
  uint64_t need, asked;
  PoolRange parts[POOL_PARTS];
  void *dst[POOL_PARTS];
  int n;
  bool ok; /* written by the reader before it posts the end; read by the consumer after the engine hands it over */
} Read;

struct Reads
{
  Engine *evt;
  Cap *cap;
  Fetchq *q;
  Read *rd;
  uint32_t n_units;
  int inflight, out;
  size_t waiting;
  uint64_t split;
  uint64_t asked;
  Pool *pool; /* the pump's: pool_ready's */
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

/* the cap gave the read a slot: to the readers - split across them when it is alone (none other out or waiting: a
   decode's miss), whole when others go with it (the batch: a prompt's next layer). one they refuse ends failed (its
   slot freed at that end) */
static void give_out(void *scope)
{
  Read *rd = scope;
  Reads *r = rd->r;
  rd->st = READ_OUT, r->out++, r->waiting--;
  const bool alone = r->out == 1 && r->waiting == 0;
  r->split += alone;
  if (!fetchq_read(r->q, rd->unit, rd->parts, rd->n, rd->dst, alone))
    post_end(rd->unit, false, r);
}

/* the consumer, in the pump: the pool told, the slot freed - the next read needed soonest given out in it */
static void ended(void *scope)
{
  Read *rd = scope;
  Reads *r = rd->r;
  rd->st = READ_IDLE, r->out--;
  pool_ready(r->pool, rd->unit, rd->ok);
  cap_release(r->cap);
}

static bool sooner(void *a, void *b, void *arg)
{
  (void)arg;
  const Read *x = a, *y = b;
  return x->need < y->need || (x->need == y->need && x->asked < y->asked);
}

static void fetch(uint32_t unit, const PoolRange *parts, int n, void *const dst[POOL_PARTS], uint64_t need, void *ctx)
{
  Reads *r = ctx;
  if (unit >= r->n_units || r->rd[unit].st != READ_IDLE || n < 0 || n > POOL_PARTS)
  {
    post_end(unit < r->n_units ? unit : 0, false, r);
    return;
  }
  Read *rd = &r->rd[unit];
  rd->n = n, rd->need = need, rd->asked = ++r->asked, rd->st = READ_WAITING;
  for (int k = 0; k < n; k++)
    rd->parts[k] = parts[k], rd->dst[k] = dst[k];
  r->waiting++;
  Context *c = context_create(rd, give_out);
  if (c && cap_submit(r->cap, c))
    return;
  context_destroy(c); /* the cap could not hold it: it ends failed, never lost */
  rd->st = READ_OUT, r->out++, r->waiting--;
  post_end(unit, false, r);
}

static void need(uint32_t unit, uint64_t need, void *ctx)
{
  Reads *r = ctx;
  if (unit < r->n_units && r->rd[unit].st == READ_WAITING)
    r->rd[unit].need = need;
}

Reads *reads_create(uint32_t n_units, int threads, int inflight, char err[READS_ERR])
{
  if (!n_units || inflight < 1)
    return snprintf(err, READS_ERR, "reads: no units, or nothing in flight"), NULL;
  Reads *r = calloc(1, sizeof *r);
  if (!r || !(r->rd = calloc(n_units, sizeof *r->rd)))
  {
    free(r);
    return snprintf(err, READS_ERR, "reads: no memory for %u reads", n_units), NULL;
  }
  r->n_units = n_units, r->inflight = inflight;
  atomic_init(&r->broken, false);
  for (uint32_t i = 0; i < n_units; i++)
    r->rd[i] = (Read){.r = r, .unit = i};
  r->evt = engine_create(NULL, NULL, 0);
  r->cap = r->evt ? cap_create(inflight, sooner, NULL) : NULL;
  r->q = r->cap ? fetchq_create(threads, post_end, r) : NULL;
  if (!r->q)
  {
    snprintf(err, READS_ERR, "reads: the readers (%d), the engine or the cap could not start", threads);
    reads_destroy(r);
    return NULL;
  }
  return r;
}

void reads_destroy(Reads *r)
{
  if (!r)
    return;
  fetchq_destroy(r->q); /* no end is posted after: the waiting and posted ones are dropped below, never resolved */
  cap_destroy(r->cap);
  engine_destroy(r->evt);
  free(r->rd);
  free(r);
}

Fetcher reads_fetcher(Reads *r) { return (Fetcher){fetch, r, need}; }

bool reads_pump(Reads *r, Pool *p, bool wait, char err[READS_ERR])
{
  if (!r || !p)
    return snprintf(err, READS_ERR, "reads: no reads or no pool to pump into"), false;
  r->pool = p;
  if (wait && !atomic_load(&r->broken))
    engine_pump(r->evt, -1);
  while (!atomic_load(&r->broken) && engine_pump(r->evt, 0))
    ;
  if (atomic_load(&r->broken))
    return snprintf(err, READS_ERR, "reads: a read's end could not be posted (no memory)"), false;
  return true;
}

int reads_inflight(const Reads *r) { return r ? r->out : 0; }
size_t reads_waiting(const Reads *r) { return r ? r->waiting : 0; }
uint64_t reads_split(const Reads *r) { return r ? r->split : 0; }
