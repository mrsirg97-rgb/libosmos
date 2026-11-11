#ifndef MEMBRANE_H
#define MEMBRANE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "osmos/mem/pool.h"

#define MEMBRANE_ERR 256

typedef struct Membrane Membrane;

typedef bool (*membrane_copy_fn)(void *dst, const void *src, uint64_t bytes, void *ctx);

typedef struct MembraneOptions
{
  const PoolUnit *units; /* n_layers * per_layer of them: unit = layer * per_layer + expert */
  uint32_t n_layers, per_layer;
  uint64_t room, stream_room; /* bytes: the membrane's pool, the stream's */
  Slots slots, stream_slots;
  uint64_t read_align; /* pool_create's */
  int readers;         /* reads_create's threads */
  const uint32_t *warm; /* units read in at membrane_warm, the most wanted first; or NULL */
  size_t n_warm;
  membrane_copy_fn copy; /* a promotion's bytes, stream's memory to the membrane's */
  void *copy_ctx;
  uint64_t ahead_limit; /* the most bytes of reads ahead (guesses, a prompt's) read at once: reads.h's; 0: no bound */
} MembraneOptions;

typedef struct MembraneStats
{
  uint64_t waits[2], wait_ns[2]; /* reads waited for and their time: [0] not guessed, [1] guessed */
  uint64_t streamed, guessed;    /* scored picks not in memory, and of them those guessed a visit ahead */
  uint64_t guess_read;           /* reads a guess began */
} MembraneStats;

Membrane *membrane_create(const MembraneOptions *o, char err[MEMBRANE_ERR]);
void membrane_destroy(Membrane *m);

bool membrane_warm(Membrane *m, char err[MEMBRANE_ERR]);
bool membrane_visit(Membrane *m, uint32_t layer, bool passing, char err[MEMBRANE_ERR]);
void membrane_score(Membrane *m, uint32_t unit);
void membrane_guess(Membrane *m, uint32_t unit);
void membrane_ahead(Membrane *m, uint32_t unit);
bool membrane_hold(Membrane *m, uint32_t unit, char err[MEMBRANE_ERR]);
bool membrane_ready(const Membrane *m, uint32_t unit, const void *dev[POOL_PARTS]);
bool membrane_wait(Membrane *m, uint32_t unit, const void *dev[POOL_PARTS], char err[MEMBRANE_ERR]);
void membrane_end(Membrane *m);

bool membrane_in_memory(const Membrane *m, uint32_t unit);
bool membrane_begin(Membrane *m, uint32_t unit);
void membrane_settle(Membrane *m);

uint64_t membrane_clock(const Membrane *m);
void membrane_stats(const Membrane *m, MembraneStats *s);

#endif

/*
Membrane (DESIGN.md, "0.5.8: the membrane", "0.6.2: the membrane a module of its own")
  a streamed MoE's experts as one cache: every expert a unit (pool.h) of the membrane's pool - what a decode keeps -
  with a second pool, the stream, that a prompt's step reads its misses and its read ahead into and leaves. the
  membrane knows units, layers and visits, not kernels or routers: the model says which units a step picks and names
  ahead, the membrane says where their bytes are. one thread calls it (the forward's: reads.h's consumer). nothing in
  it touches the GPU: the memory is the caller's Slots, a promotion's copy the caller's function.

membrane_create
  the two pools (room, stream_room; each over its Slots, read_align both), their reads (`readers` threads), the
  units copied. NULL with why for no units, layers or copy, a warm unit out of range, or what pool_create and
  reads_create refuse (a unit larger than a room among them). no read is made: membrane_warm's. ahead_limit: the
  readers' bound on reads ahead in flight (reads_ahead_limit) - a miss waited on then finds readers free and at
  most that many bytes ahead of it; a warm start's reads and a hold's are waited reads, never bound, and a passing
  visit lifts it until the next visit that is not (a prompt's reads ahead are its reads: the SSD's whole).

membrane_destroy
  the readers stopped, then the pools. the caller's part: the GPU past every unit. NULL is nothing to do.

membrane_warm
  the warm units read into the membrane's pool, waited for, then released from the last to the first - the first the
  most recent, the last to go. once, before the first visit; false with why where one cannot be held or read.

membrane_visit
  a layer's expert step begins: the reads that ended are handed to their pools; then, when `layer` is not the layer
  being visited, the last visit's holds end (membrane_end's), the clock moves one visit on and this visit is
  `passing` or not. the same layer again (a step's next sub-batch) only hands the reads in. passing: a prompt's step
  - its holds end by pool_pass in the membrane (their order kept), its cold units are read into the stream. false with
  why when a read's end could not be handed in, or the layer is past n_layers.

membrane_score
  a pick of this visit, before it is held: counted streamed when its unit is not in memory, and guessed when
  membrane_guess named it for this visit.

membrane_guess
  a decode's guess: the unit, of the next layer to be visited, wanted in the membrane's pool at the next visit (its
  read begun if it is cold and room can be made from units wanted later: pool_want) and marked for that visit's
  score. one the stream holds is marked and left where it is.

membrane_ahead
  a prompt's read ahead: the unit wanted in the stream at the next visit, unless the membrane's pool has it.

membrane_hold
  the unit held until the visit's holds end; a unit already held is nothing to do. not passing, and the stream holds
  it pinned: promoted - room and a hold in the membrane's pool (pool_place), its parts copied by `copy`, the
  stream's copy held too (the copy reads it); no room there: the stream's copy held. otherwise pinned in the pool
  that has it, or its read begun - into the stream when passing, else the membrane. a pool full while reads are
  under way is waited on (a reading unit's memory is room once its read ends); full with none reading, or a copy
  that failed: false with why, the unit not held.

a promotion's copy that failed
  leaves its unit pinned in the membrane's pool with bytes never written: until membrane_end, warm, visit, hold,
  ready, wait, in_memory and begin all refuse - no call reads that unit.

membrane_ready, membrane_wait
  the unit's pointers, in part order. ready: true only if it is pinned now. wait: its read waited for first (counted
  in the stats, by whether this visit's guess named it); false with why where the read failed or the unit is in
  neither pool.

membrane_end
  every hold of the visit ended: released wanted again n_layers visits on, or - a passing visit's, in the membrane's
  pool - passed. a step's end, or its failure (the caller first past every kernel that reads them - and, after a
  hold whose copy failed, past the copies queued before it: they write the unit's memory). a failed promotion's
  unit is then dropped (pool_drop): cold, read from its file when next picked, the stream's copy where it was.

membrane_in_memory, membrane_begin, membrane_settle
  for a caller that computes a unit elsewhere while it reads (DeepSeek's CPU lane). in_memory: pinned in either pool.
  begin: a unit not held is pinned without a wait, in the pool that has it or the membrane's - still reading, it is
  pending; false where the pool is full (not held). settle: the pending units whose reads have ended become the
  visit's holds; those still reading stay pending.

membrane_clock, membrane_stats
  the visits so far (a caller's mark of "this visit"); the counters since create.
*/
