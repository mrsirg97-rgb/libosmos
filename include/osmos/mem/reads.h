#ifndef READS_H
#define READS_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/mem/pool.h"

#define READS_ERR 256

typedef struct Reads Reads;

Reads *reads_create(uint32_t n_units, int threads, int inflight, char err[READS_ERR]);
void reads_destroy(Reads *r);

Fetcher reads_fetcher(Reads *r);
bool reads_pump(Reads *r, Pool *p, bool wait, char err[READS_ERR]);
int reads_inflight(const Reads *r);
size_t reads_waiting(const Reads *r);
uint64_t reads_split(const Reads *r);

#endif

/*
Reads
  the pool's reads on libevt (DESIGN.md, "0.2.0: the forward's reads on libevt"): the order they go to the readers in,
  and their ends coming back as events. one thread is the consumer - the forward's, the pool's only caller - and owns
  the order; fetchq's `threads` workers only pread what they were given. a read is a unit's (the pool never has two of
  one unit out), its record the unit's own: nothing is allocated a read but the event that brings its end back.
  - the order: libevt's cap, `inflight` reads given to the readers at once; the rest wait, and when one ends the read
    needed soonest goes next - its `need` (the pool's clock: pool_want's `next`, a pin's 0: now), ties the order they
    were asked in. a read's need may move while it waits (Fetcher.need: a want then a pin - fetchq's old hurry) and
    the order at the next release is the order then.
  - the ends: a reader's last chunk posts an event to the consumer's libevt Engine; reads_pump runs them there -
    pool_ready, then the read's slot freed and the next read given out.
  - the batch: a read given out with none other out or waiting (a decode's miss, alone) is split across the readers,
    FETCHQ_CHUNK a job - the SSD's speed for one expert; one given out with others (a prompt's next layer, reads
    ahead) is read whole by one reader - chunks of many reads side by side defeat the file's readahead.

reads_create
  `n_units`: the pool's units (a read's record each); `threads`: fetchq's readers (1 to 64); `inflight`: the reads
  given out at once (1 or more: the measured depth, hip_forward.c). NULL, the reason in `err`, when a reader, the
  engine or the cap cannot start.

reads_destroy
  the readers stopped (each after the read it is in), then the reads given out and waiting dropped - no pool_ready
  for them, so destroy the pool after, as fetchq before it. NULL is nothing to do.

reads_fetcher
  the Fetcher pool_create is given: fetch queues a unit's read by its need (cap_submit: given out now if a slot is
  free); need tells a waiting read its need moved (a read already given out, or none, is nothing to do). a read cap
  cannot hold, or the readers refuse, fails closed: its end posted as failed (pool_ready's ok false: a FAILED event),
  never dropped.

reads_pump
  the consumer: the reads that ended handed to `p` (pool_ready, in the order they ended) and the next given out. wait
  false: those ended so far, no wait (fwd_drain: each layer's routing); true: at least one, waiting for it
  (fwd_wait_read: a pin waited on). false, the reason in `err`, when an end could not be posted (the engine refused
  it: no memory) - the forward fails closed from then on, as its inbox's `broken` did.

reads_inflight / reads_waiting / reads_split
  the reads given to the readers now (at most `inflight`); those waiting for a slot; those given out split, so far.
*/
