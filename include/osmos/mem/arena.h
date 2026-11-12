#ifndef ARENA_H
#define ARENA_H

#include <stdbool.h>
#include <stdint.h>

#define ARENA_GRAIN 4096u /* a block's size and alignment: a direct read's (O_DIRECT on btrfs: its 4 KiB sector) */

typedef struct Arena Arena;

Arena *arena_create(void *base, uint64_t bytes);
void arena_destroy(Arena *a);
void *arena_alloc(Arena *a, uint64_t bytes);
bool arena_free(Arena *a, void *p);
uint64_t arena_used(const Arena *a);
uint64_t arena_largest(const Arena *a);
void *arena_tail(const Arena *a, uint64_t bytes);
void *arena_take(Arena *a, void *p, uint64_t bytes);

#endif

/*
Arena
  memory the caller owns (`base`, `bytes`) handed out in ARENA_GRAIN blocks: the pool's units in host_slots' registered
  memory (DESIGN.md, "0.3.0: energy a token", target 1). first fit over the free extents in address order; a block
  freed joins its free neighbours. host-only, no locks: the pool's consumer is its one caller.

arena_create
  over [base, base + bytes): base ARENA_GRAIN-aligned, bytes a whole number of grains (NULL otherwise, or no memory for
  the bookkeeping). the memory itself is the caller's: destroy does not free it. arena_destroy: NULL nothing to do.

arena_alloc
  `bytes` rounded up to grains, the first free extent that holds them, from its start; NULL when none holds them (the
  pool's room: it frees and asks again, or refuses) or bytes is 0.

arena_free
  a block arena_alloc gave back, joined with its free neighbours. false: not a block of this arena's (never given, or
  freed already) - nothing changed.

arena_used / arena_largest
  the bytes handed out now; the largest free extent (the largest alloc that would succeed).

arena_tail / arena_take
  a block at a place of the caller's choosing - a large one made room for in one window rather than in the holes
  first fit leaves (pool_lend: a long prompt's step buffers in the experts' memory). arena_tail: where the arena's
  last `bytes` (rounded up to grains) begin; NULL for 0 or more than the arena. arena_take: [p, p + bytes) given as a
  block, as arena_alloc's (arena_free gives it back), when one free extent holds all of it; NULL otherwise (nothing
  changed): any of it given, p not a grain of the arena, or bytes 0.
*/
