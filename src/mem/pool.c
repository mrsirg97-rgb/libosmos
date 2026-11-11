#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>

#include "osmos/mem/pool.h"

typedef struct Slot
{
  PoolUnit u;
  uint64_t off[POOL_PARTS];   /* each part's read's place in the unit's memory */
  uint64_t head[POOL_PARTS];  /* the part's slice inside its read (a read alignment's remainder; else 0) */
  PoolRange read[POOL_PARTS]; /* what is read for each part: its range, or the aligned range around it */
  uint64_t size;            /* the memory's bytes */
  uint8_t *dev;             /* its memory, reading or pinned; NULL cold */
  PoolState state;
  int holds;
  uint64_t released, next; /* when its last hold ended (the pool's clock); when it is wanted again (the caller's) */
  bool fresh;              /* read since its last release: no place of its own yet (pool_pass) */
} Slot;

typedef struct Spare
{
  uint8_t *dev;
  uint64_t size;
} Spare;

struct Pool
{
  Slot *s;
  size_t n;
  Spare *spare;
  size_t n_spare, cap_spare;
  uint64_t budget, bytes; /* bytes: every allocation held, units' and spares' */
  Slots slots;
  Fetcher fetcher;
  pool_event_fn on_event;
  void *event_arg;
  uint64_t clock;
  uint64_t now;   /* the caller's clock (pool_now): a unit whose next is before it is stale */
  size_t reading; /* units reading now */
};

static void emit(const Pool *p, PoolEventKind kind, uint32_t unit, uint64_t bytes)
{
  if (!p->on_event)
    return;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  const PoolEvent ev = {kind, unit, bytes, p->bytes, (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec};
  p->on_event(&ev, p->event_arg);
}

static void release_memory(Pool *p, uint8_t *dev, uint64_t size)
{
  p->slots.free(dev, p->slots.ctx);
  p->bytes -= size;
}

/* memory no unit holds, kept for one of its size; freed at once when the list cannot grow */
static void keep_spare(Pool *p, uint8_t *dev, uint64_t size)
{
  if (p->n_spare == p->cap_spare)
  {
    const size_t cap = p->cap_spare ? p->cap_spare * 2 : 64;
    Spare *grown = realloc(p->spare, cap * sizeof *grown);
    if (!grown)
    {
      release_memory(p, dev, size);
      return;
    }
    p->spare = grown, p->cap_spare = cap;
  }
  p->spare[p->n_spare++] = (Spare){dev, size};
}

static uint8_t *take_spare(Pool *p, uint64_t size)
{
  for (size_t i = p->n_spare; i-- > 0;)
    if (p->spare[i].size == size)
    {
      uint8_t *dev = p->spare[i].dev;
      p->spare[i] = p->spare[--p->n_spare];
      return dev;
    }
  return NULL;
}

/* a pinned unit gone: cold, its memory the caller's */
static uint8_t *unpin(Pool *p, Slot *s)
{
  uint8_t *dev = s->dev;
  s->dev = NULL, s->state = POOL_COLD;
  emit(p, POOL_EV_UNPIN, (uint32_t)(s - p->s), s->size);
  return dev;
}

/* how far room may reach: a pin's, every unheld unit; a want's, only those wanted after its time */
typedef struct Reach
{
  bool limited;
  uint64_t after;
} Reach;

/* when a unit is wanted for room's choice: its next, or - past it, not wanted then (a decode's expert not picked again)
   - stale: after every unit still wanted, so the first to go */
static uint64_t when(const Pool *p, const Slot *c) { return c->next >= p->now ? c->next : UINT64_MAX; }

static bool may_go(const Pool *p, const Slot *c, Reach r)
{
  return c->holds == 0 && (!r.limited || when(p, c) > r.after);
}

/* the unit room may take wanted furthest ahead (stale first); of those, the least recently released - of `size`
   bytes alone, or any (0). NULL for none */
static Slot *furthest(Pool *p, Reach r, uint64_t size)
{
  Slot *last = NULL;
  for (size_t i = 0; i < p->n; i++)
  {
    Slot *c = &p->s[i];
    if (c->state == POOL_PINNED && may_go(p, c, r) && (!size || c->size == size) &&
        (!last || when(p, c) > when(p, last) || (when(p, c) == when(p, last) && c->released < last->released)))
      last = c;
  }
  return last;
}

/* the bytes no room can be made from: units reading, or pinned and out of room's reach */
static uint64_t kept(const Pool *p, Reach r)
{
  uint64_t k = 0;
  for (size_t i = 0; i < p->n; i++)
  {
    const Slot *c = &p->s[i];
    k += c->state == POOL_READING || (c->state == POOL_PINNED && !may_go(p, c, r)) ? c->size : 0;
  }
  return k;
}

/* memory for `s`: a spare of its size, else room (spares freed, then the furthest of the units in reach) and the
   allocator's - which, refusing, gets more room the same way until it fits. NULL when it cannot fit (nothing changed)
   or nothing more can go (spares and units may have gone for its room) */
static uint8_t *room(Pool *p, const Slot *s, Reach r)
{
  uint8_t *dev = take_spare(p, s->size);
  if (dev)
    return dev;
  if (p->bytes + s->size <= p->budget && (dev = p->slots.alloc(s->size, p->slots.ctx)))
    return p->bytes += s->size, dev;
  Slot *same = furthest(p, r, s->size); /* its memory reused as it is: no hole to find */
  if (same)
    return unpin(p, same);
  if (p->bytes + s->size > p->budget && s->size > p->budget - kept(p, r))
    return NULL;
  while (p->bytes + s->size > p->budget)
  {
    if (p->n_spare)
    {
      const Spare x = p->spare[--p->n_spare];
      release_memory(p, x.dev, x.size);
      continue;
    }
    Slot *v = furthest(p, r, 0); /* one exists: kept() counted every unit that cannot go */
    uint8_t *freed = unpin(p, v);
    if (v->size == s->size)
      return freed;
    release_memory(p, freed, v->size);
  }
  for (;;) /* the allocator refusing with the budget's room made (its memory in pieces: an arena's): more room made, the
              spares then the units furthest, until it fits or nothing more can go */
  {
    if ((dev = p->slots.alloc(s->size, p->slots.ctx)))
    {
      p->bytes += s->size;
      return dev;
    }
    if (p->n_spare)
    {
      const Spare x = p->spare[--p->n_spare];
      release_memory(p, x.dev, x.size);
      continue;
    }
    Slot *v = furthest(p, r, 0);
    if (!v)
      return NULL;
    release_memory(p, unpin(p, v), v->size);
  }
}

/* a read's need on the Fetcher: a pin's 0 (waited on now), a want's its `next` + 1 - so 0 is a pin's alone */
static uint64_t want_need(uint64_t next) { return next < UINT64_MAX ? next + 1 : next; }

static void begin_read(Pool *p, Slot *s, uint8_t *dev, uint64_t need)
{
  void *dst[POOL_PARTS] = {0};
  for (int k = 0; k < s->u.n_parts; k++)
    dst[k] = dev + s->off[k];
  s->dev = dev, s->state = POOL_READING, s->fresh = true, s->released = 0, p->reading++; /* no place from its past */
  emit(p, POOL_EV_READ, (uint32_t)(s - p->s), s->size);
  p->fetcher.fetch((uint32_t)(s - p->s), s->read, s->u.n_parts, dst, need, p->fetcher.ctx);
}

/* a part's read: its range, or the read alignment's around it; and the memory it takes in the unit (from a POOL_ALIGN
   boundary at least) */
static PoolRange part_read(const PoolRange *r, uint64_t align, uint64_t *head, uint64_t *mem)
{
  const uint64_t a = align > POOL_ALIGN ? align : POOL_ALIGN;
  *head = align ? r->offset % align : 0;
  const uint64_t len = align ? (*head + r->bytes + align - 1) / align * align : r->bytes;
  *mem = (len + a - 1) / a * a;
  return (PoolRange){r->fd, r->offset - *head, len};
}

uint64_t pool_unit_bytes(const PoolUnit *u, uint64_t read_align)
{
  uint64_t b = 0, head, mem;
  for (int k = 0; k < u->n_parts && k < POOL_PARTS; k++)
    part_read(&u->part[k], read_align, &head, &mem), b += mem;
  return b;
}

static bool check_unit(Slot *s, size_t i, uint64_t align, char *err)
{
  if (s->u.n_parts < 1 || s->u.n_parts > POOL_PARTS)
  {
    snprintf(err, POOL_ERR, "unit %zu: %d parts (1 to %d)", i, s->u.n_parts, POOL_PARTS);
    return false;
  }
  for (int k = 0; k < s->u.n_parts; k++)
  {
    const PoolRange *r = &s->u.part[k];
    struct stat st;
    if (fstat(r->fd, &st) != 0 || r->bytes == 0 || r->offset >= (uint64_t)st.st_size ||
        r->bytes > (uint64_t)st.st_size - r->offset)
    {
      snprintf(err, POOL_ERR, "unit %zu part %d: 0 bytes, past its file, or a file that cannot be read", i, k);
      return false;
    }
    uint64_t head, mem;
    const PoolRange rd = part_read(r, align, &head, &mem);
    if (align && rd.offset + rd.bytes > (uint64_t)st.st_size)
    {
      snprintf(err, POOL_ERR, "unit %zu part %d: its %llu-aligned range passes its file's end", i, k,
               (unsigned long long)align);
      return false;
    }
    s->read[k] = rd;
    s->head[k] = head;
    s->off[k] = s->size;
    s->size += mem;
  }
  return true;
}

Pool *pool_create(const PoolUnit *units, size_t n_units, uint64_t budget, Slots slots, Fetcher fetcher,
                  uint64_t read_align, pool_event_fn on_event, void *event_arg, char err[POOL_ERR])
{
  err[0] = 0;
  if (read_align & (read_align - 1))
    return snprintf(err, POOL_ERR, "a read alignment of %llu: not a power of two", (unsigned long long)read_align), NULL;
  Pool *p = calloc(1, sizeof *p);
  if (!p || !(p->s = calloc(n_units ? n_units : 1, sizeof *p->s)))
  {
    free(p);
    snprintf(err, POOL_ERR, "out of memory");
    return NULL;
  }
  p->budget = budget, p->slots = slots, p->fetcher = fetcher, p->on_event = on_event, p->event_arg = event_arg;
  for (size_t i = 0; i < n_units; i++)
  {
    Slot *s = &p->s[i];
    s->u = units[i];
    const bool fits = check_unit(s, i, read_align, err) && s->size <= budget;
    if (!fits && !err[0])
      snprintf(err, POOL_ERR, "unit %zu: %llu bytes, more than the budget", i, (unsigned long long)s->size);
    if (!fits)
    {
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
  for (size_t i = 0; i < p->n; i++)
    if (p->s[i].state != POOL_COLD)
    {
      Slot *s = &p->s[i];
      const uint64_t size = s->size;
      release_memory(p, s->state == POOL_PINNED ? unpin(p, s) : s->dev, size);
    }
  while (p->n_spare)
  {
    const Spare x = p->spare[--p->n_spare];
    release_memory(p, x.dev, x.size);
  }
  free(p->spare);
  free(p->s);
  free(p);
}

static void fill(const Slot *s, const void *dev[POOL_PARTS])
{
  for (int k = 0; k < s->u.n_parts; k++)
    dev[k] = s->dev + s->off[k] + s->head[k];
}

PoolPin pool_pin(Pool *p, uint32_t unit, const void *dev[POOL_PARTS])
{
  if (!p || unit >= p->n)
    return POOL_FULL;
  Slot *s = &p->s[unit];
  if (s->state == POOL_COLD)
  {
    uint8_t *mem = room(p, s, (Reach){false, 0});
    if (!mem)
    {
      emit(p, POOL_EV_REFUSED, unit, s->size);
      return POOL_FULL;
    }
    begin_read(p, s, mem, 0);
  }
  else if (s->state == POOL_READING && p->fetcher.need)
    p->fetcher.need(unit, 0, p->fetcher.ctx);
  s->holds++;
  if (s->state != POOL_PINNED)
    return POOL_WAIT;
  fill(s, dev);
  return POOL_OK;
}

void pool_want(Pool *p, uint32_t unit, uint64_t next)
{
  if (!p || unit >= p->n || p->s[unit].holds > 0)
    return;
  Slot *s = &p->s[unit];
  if (s->state != POOL_COLD)
  {
    s->next = next;
    if (s->state == POOL_READING && p->fetcher.need)
      p->fetcher.need(unit, want_need(next), p->fetcher.ctx);
    return;
  }
  uint8_t *mem = room(p, s, (Reach){true, next});
  if (mem)
    s->next = next, begin_read(p, s, mem, want_need(next));
}

PoolPin pool_place(Pool *p, uint32_t unit, const void *dev[POOL_PARTS])
{
  if (!p || unit >= p->n || p->s[unit].state != POOL_COLD)
    return POOL_FULL;
  Slot *s = &p->s[unit];
  uint8_t *mem = room(p, s, (Reach){false, 0});
  if (!mem)
  {
    emit(p, POOL_EV_REFUSED, unit, s->size);
    return POOL_FULL;
  }
  s->dev = mem, s->state = POOL_PINNED, s->fresh = true, s->released = 0, s->holds = 1;
  emit(p, POOL_EV_PIN, unit, s->size);
  fill(s, dev);
  return POOL_OK;
}

bool pool_drop(Pool *p, uint32_t unit)
{
  if (!p || unit >= p->n || p->s[unit].state != POOL_PINNED || p->s[unit].holds)
    return false;
  Slot *s = &p->s[unit];
  const uint64_t size = s->size;
  keep_spare(p, unpin(p, s), size);
  return true;
}

void pool_ready(Pool *p, uint32_t unit, bool ok)
{
  if (!p || unit >= p->n || p->s[unit].state != POOL_READING)
    return;
  Slot *s = &p->s[unit];
  p->reading--;
  if (ok)
  {
    s->state = POOL_PINNED;
    emit(p, POOL_EV_PIN, unit, s->size);
    return;
  }
  keep_spare(p, s->dev, s->size);
  s->dev = NULL, s->state = POOL_COLD;
  emit(p, POOL_EV_FAILED, unit, s->size);
}

bool pool_dev(const Pool *p, uint32_t unit, const void *dev[POOL_PARTS])
{
  if (!p || unit >= p->n || p->s[unit].state != POOL_PINNED)
    return false;
  fill(&p->s[unit], dev);
  return true;
}

void pool_release(Pool *p, uint32_t unit, uint64_t next)
{
  if (!p || unit >= p->n || p->s[unit].holds == 0)
    return;
  Slot *s = &p->s[unit];
  if (--s->holds == 0)
    s->released = ++p->clock, s->next = next, s->fresh = false;
}

void pool_now(Pool *p, uint64_t now)
{
  if (p)
    p->now = now;
}

void pool_pass(Pool *p, uint32_t unit)
{
  if (!p || unit >= p->n || p->s[unit].holds == 0)
    return;
  Slot *s = &p->s[unit];
  if (--s->holds == 0 && s->fresh) /* read for this hold: stale, the least recently released */
    s->released = 0, s->next = 0, s->fresh = false;
}

PoolState pool_state(const Pool *p, uint32_t unit)
{
  return p && unit < p->n ? p->s[unit].state : POOL_COLD;
}

uint64_t pool_bytes(const Pool *p) { return p ? p->bytes : 0; }
size_t pool_reading(const Pool *p) { return p ? p->reading : 0; }
