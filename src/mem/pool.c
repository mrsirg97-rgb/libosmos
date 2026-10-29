#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

#include "pages.h"
#include "substrate/mem/pool.h"

typedef struct Map
{
  int fd;
  uint8_t *base;
  size_t bytes;
} Map;

typedef struct Slot
{
  PoolUnit u;
  uint8_t *host[POOL_PARTS];
  PoolState state;
  int holds;
  uint64_t released;
} Slot;

struct Pool
{
  Slot *s;
  size_t n;
  Map *maps;
  size_t n_maps;
  Pages *pages;
  uint64_t budget;
  Fetcher fetcher;
  pool_event_fn on_event;
  void *event_arg;
  uint64_t clock;
};

static uintptr_t page_lo(const uint8_t *p) { return (uintptr_t)p / PAGE_BYTES * PAGE_BYTES; }
static uintptr_t page_hi(const uint8_t *p, uint64_t bytes)
{
  return ((uintptr_t)p + bytes + PAGE_BYTES - 1) / PAGE_BYTES * PAGE_BYTES;
}

static void emit(const Pool *p, PoolEventKind kind, uint32_t unit, uint64_t bytes)
{
  if (!p->on_event)
    return;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  const PoolEvent ev = {kind, unit, bytes, p->pages->total, (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec};
  p->on_event(&ev, p->event_arg);
}

/* every part's pages: +-refs, +-held */
static void ref(Pool *p, const Slot *s, int refs, int held)
{
  for (int k = 0; k < s->u.n_parts; k++)
    pages_ref(p->pages, page_lo(s->host[k]), page_hi(s->host[k], s->u.part[k].bytes), refs, held);
}

static uint64_t uncovered(const Pool *p, const Slot *s, bool held_only)
{
  uint64_t need = 0;
  for (int k = 0; k < s->u.n_parts; k++)
    need += pages_uncovered(p->pages, page_lo(s->host[k]), page_hi(s->host[k], s->u.part[k].bytes), held_only);
  return need;
}

static void unpin(Pool *p, Slot *s)
{
  ref(p, s, -1, 0);
  const uint64_t freed = pages_drop(p->pages);
  s->state = POOL_READ;
  emit(p, POOL_EV_UNPIN, (uint32_t)(s - p->s), freed);
}

/* pins a read unit that `holds` will hold: room first (FULL changes nothing), then its pages */
static PoolPin pin_now(Pool *p, Slot *s, bool held)
{
  const uint32_t unit = (uint32_t)(s - p->s);
  if (uncovered(p, s, true) > p->budget - pages_held(p->pages)) /* not even with every unheld unit unpinned */
  {
    emit(p, POOL_EV_REFUSED, unit, uncovered(p, s, false));
    return POOL_FULL;
  }
  while (p->pages->total + uncovered(p, s, false) > p->budget)
  {
    Slot *oldest = NULL;
    for (size_t i = 0; i < p->n; i++)
      if (p->s[i].state == POOL_PINNED && p->s[i].holds == 0 && (!oldest || p->s[i].released < oldest->released))
        oldest = &p->s[i];
    unpin(p, oldest); /* one exists: the check above counted only held units' pages as kept */
  }
  for (int k = 0; k < s->u.n_parts; k++)
    if (!pages_cover(p->pages, page_lo(s->host[k]), page_hi(s->host[k], s->u.part[k].bytes)))
    {
      emit(p, POOL_EV_REFUSED, unit, 0); /* the pinner refused: what this pin registered is gone again */
      return POOL_FULL;
    }
  ref(p, s, 1, held ? 1 : 0);
  s->state = POOL_PINNED;
  emit(p, POOL_EV_PIN, unit, uncovered(p, s, false));
  return POOL_OK;
}

static void begin_read(Pool *p, Slot *s)
{
  uint64_t total = 0;
  for (int k = 0; k < s->u.n_parts; k++)
    total += s->u.part[k].bytes;
  s->state = POOL_READING;
  emit(p, POOL_EV_READ, (uint32_t)(s - p->s), total);
  p->fetcher.fetch((uint32_t)(s - p->s), s->u.part, s->u.n_parts, p->fetcher.ctx);
}

static const Map *map_of(Pool *p, int fd, char *err)
{
  for (size_t i = 0; i < p->n_maps; i++)
    if (p->maps[i].fd == fd)
      return &p->maps[i];
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size <= 0)
  {
    snprintf(err, POOL_ERR, "file descriptor %d: cannot stat it, or it is empty", fd);
    return NULL;
  }
  void *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (base == MAP_FAILED)
  {
    snprintf(err, POOL_ERR, "file descriptor %d: cannot map it", fd);
    return NULL;
  }
  p->maps[p->n_maps] = (Map){fd, base, (size_t)st.st_size};
  return &p->maps[p->n_maps++];
}

Pool *pool_create(const PoolUnit *units, size_t n_units, uint64_t budget, Pinner pinner, Fetcher fetcher,
                  pool_event_fn on_event, void *event_arg, char err[POOL_ERR])
{
  err[0] = 0;
  Pool *p = calloc(1, sizeof *p);
  if (!p)
    return NULL;
  p->budget = budget;
  p->fetcher = fetcher;
  p->on_event = on_event;
  p->event_arg = event_arg;
  p->s = calloc(n_units ? n_units : 1, sizeof *p->s);
  p->maps = calloc(n_units * POOL_PARTS + 1, sizeof *p->maps);
  p->pages = pages_create(pinner);
  if (!p->s || !p->maps || !p->pages)
  {
    snprintf(err, POOL_ERR, "out of memory");
    pool_destroy(p);
    return NULL;
  }
  for (size_t i = 0; i < n_units; i++, p->n = i)
  {
    Slot *s = &p->s[i];
    s->u = units[i];
    if (s->u.n_parts < 1 || s->u.n_parts > POOL_PARTS)
    {
      snprintf(err, POOL_ERR, "unit %zu: %d parts (1 to %d)", i, s->u.n_parts, POOL_PARTS);
      pool_destroy(p);
      return NULL;
    }
    uint64_t pages = 0;
    for (int k = 0; k < s->u.n_parts; k++)
    {
      const PoolRange *r = &s->u.part[k];
      const Map *m = map_of(p, r->fd, err);
      if (!m || r->bytes == 0 || r->offset >= m->bytes || r->bytes > m->bytes - r->offset)
      {
        if (!err[0])
          snprintf(err, POOL_ERR, "unit %zu part %d: 0 bytes, or past its file", i, k);
        pool_destroy(p);
        return NULL;
      }
      s->host[k] = m->base + r->offset;
      pages += page_hi(s->host[k], r->bytes) - page_lo(s->host[k]);
    }
    if (pages > budget)
    {
      snprintf(err, POOL_ERR, "unit %zu: %llu bytes of pages, more than the budget", i, (unsigned long long)pages);
      pool_destroy(p);
      return NULL;
    }
  }
  p->n = n_units;
  return p;
}

void pool_destroy(Pool *p)
{
  if (!p)
    return;
  for (size_t i = 0; p->pages && i < p->n; i++)
    if (p->s[i].state == POOL_PINNED)
      unpin(p, &p->s[i]);
  pages_destroy(p->pages);
  for (size_t i = 0; p->maps && i < p->n_maps; i++)
    munmap(p->maps[i].base, p->maps[i].bytes);
  free(p->maps);
  free(p->s);
  free(p);
}

void pool_want(Pool *p, const uint32_t *units, size_t n)
{
  for (size_t i = 0; p && i < n; i++)
    if (units[i] < p->n && p->s[units[i]].state == POOL_COLD)
      begin_read(p, &p->s[units[i]]);
}

void pool_ready(Pool *p, uint32_t unit)
{
  if (!p || unit >= p->n || p->s[unit].state != POOL_READING)
    return;
  Slot *s = &p->s[unit];
  s->state = POOL_READ;
  emit(p, POOL_EV_READY, unit, 0);
  if (s->holds > 0)
    pin_now(p, s, true);
}

static void fill(const Slot *s, const void *dev[POOL_PARTS])
{
  for (int k = 0; k < s->u.n_parts; k++)
    dev[k] = s->host[k];
}

PoolPin pool_pin(Pool *p, uint32_t unit, const void *dev[POOL_PARTS])
{
  if (!p || unit >= p->n)
    return POOL_FULL;
  Slot *s = &p->s[unit];
  switch (s->state)
  {
  case POOL_PINNED:
    if (s->holds++ == 0)
      ref(p, s, 0, 1);
    fill(s, dev);
    return POOL_OK;
  case POOL_READ:
    if (pin_now(p, s, true) != POOL_OK)
      return POOL_FULL;
    s->holds++;
    fill(s, dev);
    return POOL_OK;
  case POOL_COLD:
    begin_read(p, s);
    s->holds++;
    return POOL_WAIT;
  default:
    s->holds++;
    return POOL_WAIT;
  }
}

bool pool_dev(const Pool *p, uint32_t unit, const void *dev[POOL_PARTS])
{
  if (!p || unit >= p->n || p->s[unit].state != POOL_PINNED)
    return false;
  fill(&p->s[unit], dev);
  return true;
}

void pool_release(Pool *p, uint32_t unit)
{
  if (!p || unit >= p->n || p->s[unit].holds == 0)
    return;
  Slot *s = &p->s[unit];
  if (--s->holds == 0)
  {
    s->released = ++p->clock;
    if (s->state == POOL_PINNED)
      ref(p, s, 0, -1);
  }
}

PoolState pool_state(const Pool *p, uint32_t unit)
{
  return p && unit < p->n ? p->s[unit].state : POOL_COLD;
}

uint64_t pool_pinned(const Pool *p) { return p ? p->pages->total : 0; }
