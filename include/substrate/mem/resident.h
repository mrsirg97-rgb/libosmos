#ifndef RESIDENT_H
#define RESIDENT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "substrate/mem/pool.h"

#define RESIDENT_ARENA (1ull << 30) /* an allocation's bytes (or one unit's, if larger): few for the driver to keep */

typedef struct Resident Resident;

Resident *resident_create(const PoolUnit *units, size_t n_units, Slots slots, int threads, char err[POOL_ERR]);
void resident_destroy(Resident *r);

bool resident_dev(const Resident *r, size_t unit, const void *dev[POOL_PARTS]);
uint64_t resident_bytes(const Resident *r);

#endif

/*
Resident
  the units kept in GPU memory for the forward's life: copies, read once from their files straight into memory of
  their own (DESIGN.md, "the memory"), their file pages then dropped from the page cache - RAM holds them once, and
  the page cache is left to the experts the pool streams (pool.h) and Engram's rows. not registered file pages: under
  the page cache's pressure the kernel reclaims those, and the driver's restore of the registrations stalled a long
  prompt's step for minutes; this memory is the driver's own, which reclaim never takes.

resident_create
  `units` (copied) laid out back to back, each part from a POOL_ALIGN boundary, in arenas of RESIDENT_ARENA bytes
  from `slots` (hip_slots.h: GPU memory the CPU writes; malloc in the tests); read by `threads` fetchq workers at once;
  returns when every read is done. NULL, with why in `err`, for a unit with no parts or more than POOL_PARTS, a range
  of 0 bytes or past its file, memory the allocator refused, a worker that cannot start, or a read that failed -
  nothing left allocated. 0 units is an empty set.

resident_destroy
  every arena freed - the caller's part: the GPU is past them. NULL is nothing to do.

resident_dev
  a unit's pointers, in part order; false, `dev` untouched, for a unit past the set.

resident_bytes
  the GPU memory held (the arenas').
*/
