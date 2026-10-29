#ifndef PAGES_H
#define PAGES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "substrate/mem/pool.h"

#define PAGE_BYTES 4096u

typedef struct Extent
{
  uintptr_t start;
  size_t bytes;
  int refs;
  int held;
} Extent;

typedef struct Pages
{
  Extent *e;
  size_t n;
  size_t cap;
  Pinner pinner;
  uint64_t total;
} Pages;

Pages *pages_create(Pinner pinner);
void pages_destroy(Pages *pg);

uint64_t pages_uncovered(const Pages *pg, uintptr_t a, uintptr_t b, bool held_only);
uint64_t pages_held(const Pages *pg);
bool pages_cover(Pages *pg, uintptr_t a, uintptr_t b);
void pages_ref(Pages *pg, uintptr_t a, uintptr_t b, int refs, int held);
uint64_t pages_drop(Pages *pg);

#endif

/*
Pages
  the pool's registrations: page-aligned, disjoint, sorted by address, each one registered once with the pinner and
  never changed. `refs` counts the pinned units touching an extent, `held` the held ones among them. one address space
  holds every file's mapping, so one list serves them all. the pool's helper, not part of the API.

pages_uncovered
  the bytes of [a, b) (page-aligned) no extent covers - or, `held_only`, none with a held unit: what pinning would
  need once every unheld unit were gone.

pages_held
  the bytes of the extents a held unit touches: what no eviction can free.

pages_cover
  registers the gaps of [a, b) no extent covers, each its own extent (refs 0, until the caller's pages_ref). if the
  pinner refuses one, the ones this call registered are unregistered again and it returns false: nothing changed. it
  relies on the caller's rule: between calls no extent has refs 0 (every unref is followed by pages_drop), so the
  call's own extents are the only ones pages_drop takes.

pages_ref
  every extent overlapping [a, b) gains `refs` and `held` (negative to drop).

pages_drop
  unregisters every extent no pinned unit touches (refs 0); the bytes freed.
*/
