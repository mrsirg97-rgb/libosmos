#ifndef RESIDENT_H
#define RESIDENT_H

#include "osmos/err.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "osmos/mem/pool.h"

#define RESIDENT_ARENA (1ull << 30) /* an allocation's bytes (or one unit's, if larger): few for the driver to keep */

typedef struct Resident Resident;

Resident *resident_create(const PoolUnit *units, size_t n_units, Slots slots, int threads, char err[OSMOS_ERR]);
void resident_destroy(Resident *r);

bool resident_dev(const Resident *r, size_t unit, const void *dev[POOL_PARTS]);
uint64_t resident_bytes(const Resident *r);

#endif

/*
Resident
  the units kept in GPU memory for the forward's life: copies, read once from their files straight into memory of
  their own, their file pages then dropped from the page cache - RAM holds them once. not registered file pages: the
  kernel reclaims those under pressure and the driver's restore of the registrations stalls; this memory is the
  driver's own, which reclaim never takes.

resident_create
  `units` (copied) laid out back to back, each part from a POOL_ALIGN boundary, in arenas of RESIDENT_ARENA bytes
  from `slots` (GPU memory the CPU writes); read by `threads` fetchq workers at once - a part of fd -1 (offset 0) is
  room: its bytes laid out, not read (a repacked layout after the file's bytes); returns when every read is done.
  NULL, with why in `err`, for a unit with no parts or more than POOL_PARTS, a range of 0 bytes or past its file,
  memory the allocator refused, a worker that cannot start, or a read that failed - nothing left allocated. 0 units
  is an empty set.

resident_destroy
  every arena freed - the caller's part: the GPU is past them. NULL is nothing to do.

resident_dev
  a unit's pointers, in part order; false, `dev` untouched, for a unit past the set.

resident_bytes
  the GPU memory held (the arenas').
*/
