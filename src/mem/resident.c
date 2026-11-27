#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "osmos/mem/fetchq.h"
#include "osmos/mem/resident.h"
#include "osmos/err.h"

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
    if (u->part[k].fd >= 0)
      (void)posix_fadvise(u->part[k].fd, (off_t)u->part[k].offset, (off_t)u->part[k].bytes, POSIX_FADV_DONTNEED);
  pthread_mutex_lock(&r->mu);
  r->done++, r->failed += !ok;
  pthread_cond_signal(&r->cv);
  pthread_mutex_unlock(&r->mu);
}

typedef struct FdSize
{
  int fd;
  uint64_t size;
} FdSize;

static bool fd_size(int fd, FdSize *cache, size_t *n_cache, size_t cap, uint64_t *out)
{
  for (size_t i = 0; i < *n_cache; i++)
    if (cache[i].fd == fd)
      return *out = cache[i].size, true;
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size < 0)
    return false;
  if (*n_cache < cap)
    cache[(*n_cache)++] = (FdSize){fd, (uint64_t)st.st_size};
  *out = (uint64_t)st.st_size;
  return true;
}

static bool check(const PoolUnit *u, size_t i, uint64_t *size, FdSize *cache, size_t *n_cache, size_t cap, char *err)
{
  if (u->n_parts < 1 || u->n_parts > POOL_PARTS)
    return osmos_fail(err, "unit %zu: %d parts (1 to %d)", i, u->n_parts, POOL_PARTS), false;
  *size = 0;
  for (int k = 0; k < u->n_parts; k++)
  {
    const PoolRange *p = &u->part[k];
    uint64_t fsz = 0;
    if (p->bytes == 0 || (p->fd < 0 && p->offset != 0) ||
        (p->fd >= 0 && (!fd_size(p->fd, cache, n_cache, cap, &fsz) || p->offset >= fsz || p->bytes > fsz - p->offset)))
      return osmos_fail(err, "unit %zu part %d: 0 bytes, or past its file", i, k), false;
    *size += up(p->bytes);
  }
  return true;
}

/* every unit its place, arena by arena: a unit never spans two; each arena is clamped to the remaining units that fit
   within RESIDENT_ARENA so the final arena does not over-allocate up to 2 GiB */
static bool lay_out(Resident *r, char *err)
{
  uint64_t *sizes = r->n ? malloc(r->n * sizeof *sizes) : NULL;
  if (r->n && !sizes)
    return osmos_fail(err, "out of memory"), false;
  FdSize fcache[64];
  size_t n_fcache = 0;
  for (size_t i = 0; i < r->n; i++)
    if (!check(&r->u[i], i, &sizes[i], fcache, &n_fcache, 64, err))
      return free(sizes), false;
  uint64_t used = 0, cap = 0;
  for (size_t i = 0; i < r->n; i++)
  {
    const uint64_t size = sizes[i];
    if (!cap || used + size > cap)
    {
      uint64_t need = size;
      if (need < RESIDENT_ARENA)
        for (size_t j = i + 1; j < r->n && need + sizes[j] <= RESIDENT_ARENA; j++)
          need += sizes[j];
      cap = need;
      uint8_t *a = r->slots.alloc(cap, r->slots.ctx);
      if (!a)
        return free(sizes), osmos_fail(err, "GPU memory refused (%llu bytes held)", (unsigned long long)r->bytes), false;
      r->arena[r->n_arena++] = a, r->bytes += cap, used = 0;
    }
    r->dev[i] = r->arena[r->n_arena - 1] + used;
    uint64_t at = 0; /* (not an int: a first part of 2 GiB or more wraps one) */
    for (int k = 0; k < r->u[i].n_parts; k++)
      r->off[i][k] = at, at += up(r->u[i].part[k].bytes);
    used += size;
  }
  free(sizes);
  return true;
}

Resident *resident_create(const PoolUnit *units, size_t n_units, Slots slots, int threads, char err[OSMOS_ERR])
{
  err[0] = 0;
  Resident *r = calloc(1, sizeof *r);
  if (!r)
    return osmos_fail(err, "out of memory"), NULL;
  pthread_mutex_init(&r->mu, NULL);
  pthread_cond_init(&r->cv, NULL);
  r->slots = slots, r->n = n_units;
  const size_t m = n_units ? n_units : 1;
  r->u = malloc(m * sizeof *r->u), r->dev = calloc(m, sizeof *r->dev), r->off = calloc(m, sizeof *r->off);
  r->arena = calloc(m, sizeof *r->arena); /* at most an arena a unit */
  if (!r->u || !r->dev || !r->off || !r->arena)
  {
    osmos_fail(err, "out of memory");
    resident_destroy(r);
    return NULL;
  }
  for (size_t i = 0; i < n_units; i++)
    r->u[i] = units[i];
  Fetchq *q = NULL;
  bool ok = lay_out(r, err);
  if (ok && n_units && !(q = fetchq_create(threads, (uint32_t)n_units, on_read, r)))
  {
    osmos_fail(err, "the readers could not start");
    ok = false;
  }
  for (size_t i = 0; ok && i < n_units; i++)
  {
    void *dst[POOL_PARTS] = {0};
    PoolRange read[POOL_PARTS];
    int n = 0;
    for (int k = 0; k < r->u[i].n_parts; k++) /* a room part (no file) is memory only: nothing read */
      if (r->u[i].part[k].fd >= 0)
        read[n] = r->u[i].part[k], dst[n++] = r->dev[i] + r->off[i][k];
    if (!fetchq_read(q, (uint32_t)i, read, n, dst, false, 1)) /* the queue could not grow: the unit's read failed */
      on_read((uint32_t)i, false, r);
  }
  pthread_mutex_lock(&r->mu);
  while (ok && r->done < n_units)
    pthread_cond_wait(&r->cv, &r->mu);
  pthread_mutex_unlock(&r->mu);
  fetchq_destroy(q);
  if (ok && r->failed)
  {
    osmos_fail(err, "%zu of %zu reads failed", r->failed, n_units);
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
