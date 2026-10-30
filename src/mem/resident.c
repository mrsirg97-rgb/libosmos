#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "substrate/mem/fetchq.h"
#include "substrate/mem/resident.h"

struct Resident
{
  PoolUnit *u;
  uint8_t **dev; /* each unit's memory: its parts at dev + off */
  uint64_t (*off)[POOL_PARTS];
  size_t n;
  uint8_t **arena;
  size_t n_arena;
  uint64_t bytes;
  Slots slots;
  pthread_mutex_t mu;
  pthread_cond_t cv;
  size_t done, failed;
};

static uint64_t up(uint64_t b) { return (b + POOL_ALIGN - 1) / POOL_ALIGN * POOL_ALIGN; }

/* a worker's read is done: its file pages dropped (the copy is the one kept), counted */
static void on_read(uint32_t unit, bool ok, void *arg)
{
  Resident *r = arg;
  const PoolUnit *u = &r->u[unit];
  for (int k = 0; ok && k < u->n_parts; k++)
    (void)posix_fadvise(u->part[k].fd, (off_t)u->part[k].offset, (off_t)u->part[k].bytes, POSIX_FADV_DONTNEED);
  pthread_mutex_lock(&r->mu);
  r->done++, r->failed += !ok;
  pthread_cond_signal(&r->cv);
  pthread_mutex_unlock(&r->mu);
}

static bool check(const PoolUnit *u, size_t i, uint64_t *size, char *err)
{
  if (u->n_parts < 1 || u->n_parts > POOL_PARTS)
    return snprintf(err, POOL_ERR, "unit %zu: %d parts (1 to %d)", i, u->n_parts, POOL_PARTS), false;
  *size = 0;
  for (int k = 0; k < u->n_parts; k++)
  {
    const PoolRange *p = &u->part[k];
    struct stat st;
    if (fstat(p->fd, &st) != 0 || p->bytes == 0 || p->offset >= (uint64_t)st.st_size ||
        p->bytes > (uint64_t)st.st_size - p->offset)
      return snprintf(err, POOL_ERR, "unit %zu part %d: 0 bytes, or past its file", i, k), false;
    *size += up(p->bytes);
  }
  return true;
}

/* every unit its place, arena by arena: a unit never spans two */
static bool lay_out(Resident *r, char *err)
{
  uint64_t used = RESIDENT_ARENA, cap = 0;
  for (size_t i = 0; i < r->n; i++)
  {
    uint64_t size;
    if (!check(&r->u[i], i, &size, err))
      return false;
    if (used + size > cap)
    {
      cap = size > RESIDENT_ARENA ? size : RESIDENT_ARENA;
      uint8_t *a = r->slots.alloc(cap, r->slots.ctx);
      if (!a)
        return snprintf(err, POOL_ERR, "GPU memory refused (%llu bytes held)", (unsigned long long)r->bytes), false;
      r->arena[r->n_arena++] = a, r->bytes += cap, used = 0;
    }
    r->dev[i] = r->arena[r->n_arena - 1] + used;
    for (int k = 0, at = 0; k < r->u[i].n_parts; k++)
      r->off[i][k] = (uint64_t)at, at += (int)up(r->u[i].part[k].bytes);
    used += size;
  }
  return true;
}

Resident *resident_create(const PoolUnit *units, size_t n_units, Slots slots, int threads, char err[POOL_ERR])
{
  err[0] = 0;
  Resident *r = calloc(1, sizeof *r);
  if (!r)
    return snprintf(err, POOL_ERR, "out of memory"), NULL;
  pthread_mutex_init(&r->mu, NULL);
  pthread_cond_init(&r->cv, NULL);
  r->slots = slots, r->n = n_units;
  const size_t m = n_units ? n_units : 1;
  r->u = malloc(m * sizeof *r->u), r->dev = calloc(m, sizeof *r->dev), r->off = calloc(m, sizeof *r->off);
  r->arena = calloc(m, sizeof *r->arena); /* at most an arena a unit */
  if (!r->u || !r->dev || !r->off || !r->arena)
  {
    snprintf(err, POOL_ERR, "out of memory");
    resident_destroy(r);
    return NULL;
  }
  for (size_t i = 0; i < n_units; i++)
    r->u[i] = units[i];
  Fetchq *q = NULL;
  bool ok = lay_out(r, err);
  if (ok && n_units && !(q = fetchq_create(threads, on_read, r)))
  {
    snprintf(err, POOL_ERR, "the readers could not start");
    ok = false;
  }
  for (size_t i = 0; ok && i < n_units; i++)
  {
    void *dst[POOL_PARTS] = {0};
    for (int k = 0; k < r->u[i].n_parts; k++)
      dst[k] = r->dev[i] + r->off[i][k];
    fetchq_fetcher(q).fetch((uint32_t)i, r->u[i].part, r->u[i].n_parts, dst, q);
  }
  pthread_mutex_lock(&r->mu);
  while (ok && r->done < n_units)
    pthread_cond_wait(&r->cv, &r->mu);
  pthread_mutex_unlock(&r->mu);
  fetchq_destroy(q);
  if (ok && r->failed)
  {
    snprintf(err, POOL_ERR, "%zu of %zu reads failed", r->failed, n_units);
    ok = false;
  }
  if (!ok)
  {
    resident_destroy(r);
    return NULL;
  }
  return r;
}

void resident_destroy(Resident *r)
{
  if (!r)
    return;
  for (size_t i = 0; r->arena && i < r->n_arena; i++)
    r->slots.free(r->arena[i], r->slots.ctx);
  free(r->arena), free(r->u), free(r->dev), free(r->off);
  pthread_cond_destroy(&r->cv);
  pthread_mutex_destroy(&r->mu);
  free(r);
}

bool resident_dev(const Resident *r, size_t unit, const void *dev[POOL_PARTS])
{
  if (!r || unit >= r->n)
    return false;
  for (int k = 0; k < r->u[unit].n_parts; k++)
    dev[k] = r->dev[unit] + r->off[unit][k];
  return true;
}

uint64_t resident_bytes(const Resident *r) { return r ? r->bytes : 0; }
