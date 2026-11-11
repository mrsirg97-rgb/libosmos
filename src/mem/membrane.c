#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "osmos/mem/membrane.h"
#include "osmos/mem/reads.h"

struct Membrane
{
  PoolUnit *units;
  uint32_t n_units, n_layers;
  Reads *reads; /* both pools' (fetcher 0 the membrane's, 1 the stream's) */
  Pool *pool, *stream;
  membrane_copy_fn copy;
  void *copy_ctx;
  uint32_t *warm;
  size_t n_warm;
  uint64_t visits;  /* the layers' expert steps so far: pool_release's clock */
  int64_t layer;    /* the layer being visited, -1: none */
  bool passing;
  uint32_t *held; /* the visit's holds, 2 a unit at most (a promotion's: the stream's copy and the membrane's) */
  Pool **held_in;
  uint32_t n_held;
  bool *is_held;
  uint32_t *pend; /* membrane_begin's units still reading */
  uint32_t n_pend;
  uint64_t *guessed; /* the visit a unit was guessed for */
  uint32_t *failed;  /* promotions whose copy failed: pinned in the membrane's pool, their bytes never written -
                        every call refused until membrane_end drops them */
  uint32_t n_failed;
  MembraneStats stats;
};

static bool fail(char *err, const char *why)
{
  if (err)
    snprintf(err, MEMBRANE_ERR, "%s", why);
  return false;
}

static const char kBroken[] = "the membrane refused: a promotion's copy failed - membrane_end, the GPU past it, first";

static uint64_t now_ns(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

/* the pool a unit is pinned or reading in - the membrane's first, else the stream's; NULL in neither */
static Pool *pool_in(const Membrane *m, uint32_t unit)
{
  return pool_state(m->pool, unit) != POOL_COLD ? m->pool : pool_state(m->stream, unit) != POOL_COLD ? m->stream : NULL;
}

/* where a unit is, or - cold - where this visit reads it */
static Pool *pool_for(const Membrane *m, uint32_t unit, bool passing)
{
  Pool *in = pool_in(m, unit);
  return in ? in : passing ? m->stream : m->pool;
}

static bool pump(Membrane *m, bool wait, char *err)
{
  char rerr[READS_ERR] = "";
  return reads_pump(m->reads, (Pool *[READS_POOLS]){m->pool, m->stream}, wait, rerr) || fail(err, rerr);
}

static void held(Membrane *m, Pool *in, uint32_t unit) { m->held_in[m->n_held] = in, m->held[m->n_held++] = unit; }

Membrane *membrane_create(const MembraneOptions *o, char err[MEMBRANE_ERR])
{
  if (!o || !o->units || !o->n_layers || !o->per_layer || !o->copy || (o->n_warm && !o->warm))
    return fail(err, "the membrane: no options, units, layers or copy"), NULL;
  const uint64_t n = (uint64_t)o->n_layers * o->per_layer;
  if (n > UINT32_MAX)
    return fail(err, "the membrane: more units than it numbers"), NULL;
  for (size_t i = 0; i < o->n_warm; i++)
    if (o->warm[i] >= n)
      return fail(err, "the membrane: a warm unit past its units"), NULL;
  Membrane *m = calloc(1, sizeof *m);
  if (!m)
    return fail(err, "out of memory"), NULL;
  m->n_units = (uint32_t)n, m->n_layers = o->n_layers, m->copy = o->copy, m->copy_ctx = o->copy_ctx, m->layer = -1;
  m->n_warm = o->n_warm;
  char perr[POOL_ERR] = "";
  const bool ok = (m->units = malloc(n * sizeof *m->units)) && (m->held = malloc(2 * n * sizeof *m->held)) &&
                  (m->held_in = malloc(2 * n * sizeof *m->held_in)) && (m->is_held = calloc(n, sizeof *m->is_held)) &&
                  (m->pend = malloc(n * sizeof *m->pend)) && (m->failed = malloc(n * sizeof *m->failed)) && (m->guessed = calloc(n, sizeof *m->guessed)) &&
                  (!o->n_warm || (m->warm = malloc(o->n_warm * sizeof *m->warm)));
  if (!ok)
    return membrane_destroy(m), fail(err, "out of memory"), NULL;
  memcpy(m->units, o->units, n * sizeof *m->units);
  if (o->n_warm)
    memcpy(m->warm, o->warm, o->n_warm * sizeof *m->warm);
  if (!(m->reads = reads_create(m->n_units, o->readers, perr)) ||
      !(m->pool = pool_create(m->units, m->n_units, o->room, o->slots, reads_fetcher(m->reads, 0), o->read_align, NULL,
                              NULL, perr)) ||
      !(m->stream = pool_create(m->units, m->n_units, o->stream_room, o->stream_slots, reads_fetcher(m->reads, 1),
                                o->read_align, NULL, NULL, perr)))
    return membrane_destroy(m), fail(err, perr[0] ? perr : "the membrane's pools refused"), NULL;
  return m;
}

void membrane_destroy(Membrane *m)
{
  if (!m)
    return;
  reads_destroy(m->reads); /* its readers stopped before the pools they feed go */
  pool_destroy(m->pool), pool_destroy(m->stream);
  free(m->units), free(m->held), free(m->held_in), free(m->is_held), free(m->pend), free(m->failed), free(m->guessed), free(m->warm);
  free(m);
}

bool membrane_warm(Membrane *m, char err[MEMBRANE_ERR])
{
  if (m->n_failed)
    return fail(err, kBroken);
  const size_t n = m->n_warm;
  m->n_warm = 0; /* once, read or refused */
  for (size_t k = 0; k < n; k++)
    if (pool_pin(m->pool, m->warm[k], (const void *[POOL_PARTS]){0}) == POOL_FULL)
      return fail(err, "an expert the budget holds the pool could not");
  while (n && pool_reading(m->pool))
    if (!pump(m, true, err))
      return false;
  for (size_t k = 0; k < n; k++)
    if (pool_state(m->pool, m->warm[k]) != POOL_PINNED)
      return fail(err, "an expert's read from its file failed");
  for (size_t k = n; k-- > 0;)
    pool_release(m->pool, m->warm[k], 0);
  return true;
}

/* the visit's holds ended */
static void end_holds(Membrane *m)
{
  for (uint32_t k = 0; k < m->n_held; k++) /* the layer they came from next visited a whole round on */
  {
    if (m->passing && m->held_in[k] == m->pool) /* a prompt's hit in the membrane: its order kept */
      pool_pass(m->pool, m->held[k]);
    else
      pool_release(m->held_in[k], m->held[k], m->visits - 1 + (uint64_t)m->n_layers);
    m->is_held[m->held[k]] = false;
  }
  m->n_held = 0;
  m->layer = -1;
}

void membrane_end(Membrane *m)
{
  end_holds(m);
  uint32_t keep = 0;
  for (uint32_t k = 0; k < m->n_failed; k++) /* back to cold: read from its file when next picked */
    if (!pool_drop(m->pool, m->failed[k]))
      m->failed[keep++] = m->failed[k];
  m->n_failed = keep;
}

bool membrane_visit(Membrane *m, uint32_t layer, bool passing, char err[MEMBRANE_ERR])
{
  if (m->n_failed)
    return fail(err, kBroken);
  if (layer >= m->n_layers)
    return fail(err, "the membrane: a layer past its layers");
  if (!pump(m, false, err))
    return false;
  if (m->layer == (int64_t)layer)
    return true;
  end_holds(m);
  m->visits++, m->layer = layer, m->passing = passing;
  pool_now(m->pool, m->visits), pool_now(m->stream, m->visits);
  return true;
}

void membrane_score(Membrane *m, uint32_t unit)
{
  const Pool *in = pool_in(m, unit);
  if (!(in && pool_state(in, unit) == POOL_PINNED))
    m->stats.streamed++, m->stats.guessed += m->guessed[unit] == m->visits;
}

void membrane_guess(Membrane *m, uint32_t unit)
{
  m->guessed[unit] = m->visits + 1;
  if (pool_in(m, unit) == m->stream)
    return;
  const bool cold = pool_state(m->pool, unit) == POOL_COLD;
  pool_want(m->pool, unit, m->visits + 1);
  m->stats.guess_read += cold && pool_state(m->pool, unit) != POOL_COLD;
}

void membrane_ahead(Membrane *m, uint32_t unit)
{
  if (pool_in(m, unit) != m->pool) /* one in the membrane a hit, its order left alone */
    pool_want(m->stream, unit, m->visits + 1);
}

/* a decode's unit the stream holds copied into the membrane's memory, so a prompt's later reads cannot push it out:
   the stream's copy held until the visit's end (the copy reads it). *done false, the unit not yet held for the
   visit, where the stream has it not pinned or the membrane no room */
static bool promote(Membrane *m, uint32_t unit, bool *done, char *err)
{
  const void *src[POOL_PARTS], *dst[POOL_PARTS];
  *done = false;
  if (pool_state(m->stream, unit) != POOL_PINNED || pool_pin(m->stream, unit, src) != POOL_OK)
    return true;
  held(m, m->stream, unit);
  if (pool_place(m->pool, unit, dst) != POOL_OK)
    return true;
  held(m, m->pool, unit), m->is_held[unit] = true;
  for (int k = 0; k < m->units[unit].n_parts; k++)
    if (!m->copy((void *)dst[k], src[k], m->units[unit].part[k].bytes, m->copy_ctx))
      return m->failed[m->n_failed++] = unit, fail(err, "an expert's copy into the membrane failed");
  *done = true;
  return true;
}

bool membrane_hold(Membrane *m, uint32_t unit, char err[MEMBRANE_ERR])
{
  if (m->n_failed)
    return fail(err, kBroken);
  if (m->is_held[unit])
    return true;
  bool promoted = false;
  if (!m->passing && pool_in(m, unit) == m->stream && !promote(m, unit, &promoted, err))
    return false;
  if (promoted)
    return true;
  Pool *in = pool_for(m, unit, m->passing);
  PoolPin pin;
  while ((pin = pool_pin(in, unit, (const void *[POOL_PARTS]){0})) == POOL_FULL && pool_reading(in))
    if (!pump(m, true, err))
      return false;
  if (pin == POOL_FULL)
    return fail(err, "an expert the pool cannot hold (its room, or GPU memory refused)");
  held(m, in, unit), m->is_held[unit] = true;
  return true;
}

bool membrane_ready(const Membrane *m, uint32_t unit, const void *dev[POOL_PARTS])
{
  const Pool *in = pool_in(m, unit);
  return !m->n_failed && in && pool_dev(in, unit, dev);
}

bool membrane_wait(Membrane *m, uint32_t unit, const void *dev[POOL_PARTS], char err[MEMBRANE_ERR])
{
  if (m->n_failed)
    return fail(err, kBroken);
  Pool *p = pool_in(m, unit);
  if (p && pool_state(p, unit) == POOL_READING)
  {
    const int k = m->guessed[unit] == m->visits;
    const uint64_t t0 = now_ns();
    while (pool_state(p, unit) == POOL_READING)
      if (!pump(m, true, err))
        return false;
    m->stats.waits[k]++, m->stats.wait_ns[k] += now_ns() - t0;
  }
  return (p && pool_dev(p, unit, dev)) || fail(err, "an expert's read from its file failed");
}

bool membrane_in_memory(const Membrane *m, uint32_t unit)
{
  const Pool *in = pool_in(m, unit);
  return !m->n_failed && in && pool_state(in, unit) == POOL_PINNED;
}

bool membrane_begin(Membrane *m, uint32_t unit)
{
  if (m->n_failed)
    return false;
  if (m->is_held[unit])
    return true;
  Pool *in = pool_for(m, unit, false);
  const PoolPin pin = pool_pin(in, unit, (const void *[POOL_PARTS]){0});
  if (pin == POOL_FULL)
    return false;
  if (pin == POOL_WAIT)
    m->pend[m->n_pend++] = unit;
  else
    held(m, in, unit);
  m->is_held[unit] = true;
  return true;
}

void membrane_settle(Membrane *m)
{
  uint32_t keep = 0;
  for (uint32_t k = 0; k < m->n_pend; k++)
  {
    const uint32_t u = m->pend[k];
    Pool *in = pool_in(m, u);
    if (in && pool_state(in, u) == POOL_READING)
      m->pend[keep++] = u;
    else
      held(m, in ? in : m->pool, u);
  }
  m->n_pend = keep;
}

uint64_t membrane_clock(const Membrane *m) { return m->visits; }

void membrane_stats(const Membrane *m, MembraneStats *s) { *s = m->stats; }
