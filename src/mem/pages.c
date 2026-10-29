#include <stdlib.h>
#include <string.h>

#include "pages.h"

Pages *pages_create(Pinner pinner)
{
  Pages *pg = calloc(1, sizeof *pg);
  if (!pg)
    return NULL;
  pg->pinner = pinner;
  return pg;
}

void pages_destroy(Pages *pg)
{
  if (!pg)
    return;
  for (size_t i = 0; i < pg->n; i++)
    pg->pinner.unpin((void *)pg->e[i].start, pg->pinner.ctx);
  free(pg->e);
  free(pg);
}

/* the first extent ending after `a` */
static size_t first_after(const Pages *pg, uintptr_t a)
{
  size_t lo = 0, hi = pg->n;
  while (lo < hi)
  {
    const size_t mid = (lo + hi) / 2;
    if (pg->e[mid].start + pg->e[mid].bytes <= a)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

uint64_t pages_uncovered(const Pages *pg, uintptr_t a, uintptr_t b, bool held_only)
{
  uint64_t gap = 0;
  uintptr_t at = a;
  for (size_t i = first_after(pg, a); i < pg->n && pg->e[i].start < b; i++)
  {
    const Extent *x = &pg->e[i];
    if (held_only && x->held <= 0)
      continue;
    if (x->start > at)
      gap += x->start - at;
    if (x->start + x->bytes > at)
      at = x->start + x->bytes;
  }
  return gap + (b > at ? b - at : 0);
}

uint64_t pages_held(const Pages *pg)
{
  uint64_t held = 0;
  for (size_t i = 0; i < pg->n; i++)
    held += pg->e[i].held > 0 ? pg->e[i].bytes : 0;
  return held;
}

static bool insert(Pages *pg, size_t at, uintptr_t start, size_t bytes)
{
  if (pg->n == pg->cap)
  {
    const size_t cap = pg->cap ? pg->cap * 2 : 64;
    Extent *e = realloc(pg->e, cap * sizeof *e);
    if (!e)
      return false;
    pg->e = e;
    pg->cap = cap;
  }
  memmove(&pg->e[at + 1], &pg->e[at], (pg->n - at) * sizeof *pg->e);
  pg->e[at] = (Extent){start, bytes, 0, 0};
  pg->n++;
  pg->total += bytes;
  return true;
}

static void remove_at(Pages *pg, size_t at)
{
  pg->total -= pg->e[at].bytes;
  memmove(&pg->e[at], &pg->e[at + 1], (pg->n - at - 1) * sizeof *pg->e);
  pg->n--;
}

bool pages_cover(Pages *pg, uintptr_t a, uintptr_t b)
{
  uintptr_t at = a;
  size_t i = first_after(pg, a);
  while (at < b)
  {
    const uintptr_t next = i < pg->n && pg->e[i].start < b ? pg->e[i].start : b;
    if (next > at)
    {
      const bool pinned = pg->pinner.pin((void *)at, next - at, pg->pinner.ctx);
      if (!pinned || !insert(pg, i, at, next - at))
      {
        if (pinned)
          pg->pinner.unpin((void *)at, pg->pinner.ctx);
        pages_drop(pg); /* the gaps this call registered are the only extents without a ref: nothing changed */
        return false;
      }
      i++;
    }
    if (next >= b)
      break;
    at = pg->e[i].start + pg->e[i].bytes;
    i++;
  }
  return true;
}

void pages_ref(Pages *pg, uintptr_t a, uintptr_t b, int refs, int held)
{
  for (size_t i = first_after(pg, a); i < pg->n && pg->e[i].start < b; i++)
  {
    pg->e[i].refs += refs;
    pg->e[i].held += held;
  }
}

uint64_t pages_drop(Pages *pg)
{
  uint64_t freed = 0;
  for (size_t i = pg->n; i-- > 0;)
    if (pg->e[i].refs <= 0)
    {
      freed += pg->e[i].bytes;
      pg->pinner.unpin((void *)pg->e[i].start, pg->pinner.ctx);
      remove_at(pg, i);
    }
  return freed;
}
