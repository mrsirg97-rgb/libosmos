#include "osmos/mem/arena.h"

#include <stdlib.h>
#include <string.h>

/* an extent of grains: [at, at + n) from the arena's base */
typedef struct Ext
{
  uint64_t at, n;
} Ext;

/* extents in address order, grown as needed */
typedef struct List
{
  Ext *e;
  size_t len, cap;
} List;

struct Arena
{
  uint8_t *base;
  uint64_t grains, used;
  List free, given;
};

static bool grow(List *l)
{
  if (l->len < l->cap)
    return true;
  const size_t cap = l->cap ? l->cap * 2 : 64;
  Ext *e = realloc(l->e, cap * sizeof *e);
  if (!e)
    return false;
  l->e = e, l->cap = cap;
  return true;
}

/* the first index whose extent starts at or after `at` */
static size_t find(const List *l, uint64_t at)
{
  size_t lo = 0, hi = l->len;
  while (lo < hi)
  {
    const size_t mid = (lo + hi) / 2;
    if (l->e[mid].at < at)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

static bool insert(List *l, size_t i, Ext x)
{
  if (!grow(l))
    return false;
  memmove(l->e + i + 1, l->e + i, (l->len - i) * sizeof *l->e);
  l->e[i] = x, l->len++;
  return true;
}

static void erase(List *l, size_t i)
{
  memmove(l->e + i, l->e + i + 1, (l->len - i - 1) * sizeof *l->e);
  l->len--;
}

Arena *arena_create(void *base, uint64_t bytes)
{
  if (!base || !bytes || (uintptr_t)base % ARENA_GRAIN || bytes % ARENA_GRAIN)
    return NULL;
  Arena *a = calloc(1, sizeof *a);
  if (!a)
    return NULL;
  a->base = base, a->grains = bytes / ARENA_GRAIN;
  if (!insert(&a->free, 0, (Ext){0, a->grains}))
  {
    arena_destroy(a);
    return NULL;
  }
  return a;
}

void arena_destroy(Arena *a)
{
  if (!a)
    return;
  free(a->free.e);
  free(a->given.e);
  free(a);
}

void *arena_alloc(Arena *a, uint64_t bytes)
{
  if (!a || !bytes)
    return NULL;
  const uint64_t n = (bytes + ARENA_GRAIN - 1) / ARENA_GRAIN;
  for (size_t i = 0; i < a->free.len; i++)
  {
    Ext *f = &a->free.e[i];
    if (f->n < n)
      continue;
    const Ext blk = {f->at, n};
    if (!grow(&a->given)) /* the bookkeeping first: nothing changed if it cannot grow */
      return NULL;
    if (!insert(&a->given, find(&a->given, blk.at), blk))
      return NULL;
    if (f->n == n)
      erase(&a->free, i);
    else
      f->at += n, f->n -= n;
    a->used += n * ARENA_GRAIN;
    return a->base + blk.at * ARENA_GRAIN;
  }
  return NULL;
}

bool arena_free(Arena *a, void *p)
{
  if (!a || !p || (uint8_t *)p < a->base)
    return false;
  const uint64_t off = (uint64_t)((uint8_t *)p - a->base);
  if (off % ARENA_GRAIN)
    return false;
  const uint64_t at = off / ARENA_GRAIN;
  const size_t g = find(&a->given, at);
  if (g == a->given.len || a->given.e[g].at != at)
    return false;
  const Ext blk = a->given.e[g];
  size_t i = find(&a->free, at); /* the free extent after it (if any) is at i, the one before at i - 1 */
  const bool before = i > 0 && a->free.e[i - 1].at + a->free.e[i - 1].n == blk.at;
  const bool after = i < a->free.len && blk.at + blk.n == a->free.e[i].at;
  if (!before && !after && !insert(&a->free, i, blk))
    return false; /* no memory to record it: still given, nothing changed */
  if (before && after)
    a->free.e[i - 1].n += blk.n + a->free.e[i].n, erase(&a->free, i);
  else if (before)
    a->free.e[i - 1].n += blk.n;
  else if (after)
    a->free.e[i].at = blk.at, a->free.e[i].n += blk.n;
  erase(&a->given, g);
  a->used -= blk.n * ARENA_GRAIN;
  return true;
}

uint64_t arena_used(const Arena *a) { return a ? a->used : 0; }

uint64_t arena_largest(const Arena *a)
{
  uint64_t most = 0;
  for (size_t i = 0; a && i < a->free.len; i++)
    most = a->free.e[i].n > most ? a->free.e[i].n : most;
  return most * ARENA_GRAIN;
}
