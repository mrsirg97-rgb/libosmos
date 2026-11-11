#ifndef READS_H
#define READS_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/mem/pool.h"

#define READS_ERR 256
#define READS_POOLS 2 /* the pools one Reads reads for: the forward's membrane and its stream (DESIGN.md, "0.5.8") */

typedef struct Reads Reads;

Reads *reads_create(uint32_t n_units, int threads, char err[READS_ERR]);
void reads_destroy(Reads *r);

Fetcher reads_fetcher(Reads *r, int pool);
bool reads_pump(Reads *r, Pool *const pools[READS_POOLS], bool wait, char err[READS_ERR]);
int reads_out(const Reads *r);
uint64_t reads_split(const Reads *r);
void reads_ahead_limit(Reads *r, uint64_t bytes);

#endif

/*
Reads
  the pool's reads on libevt (DESIGN.md, "0.2.0: the forward's reads on libevt"): given to the readers at their need,
  and their ends coming back as events. one thread is the consumer - the forward's, the pool's only caller; fetchq's
  `threads` workers pread. a read is a unit's (the pool never has two of one unit out), its record the unit's own:
  nothing is allocated a read but the event that brings its end back.
  - the order: each read goes to the readers at once with its need (the pool's clock: a pin's 0, a want's next + 1),
    and the readers take the job needed soonest (fetchq: libevt's queue). a need that moves (Fetcher.need: a want then
    a pin) moves the read's jobs not yet begun (fetchq_need: libevt's queue_update) - 0.1.3's hurry, general. no read
    is held back from the readers: holding them back (0.2.0's first build, a cap) left the SSD idle between the
    consumer's pumps.
  - the batch: a pin (waited on) is split across the readers unless another pin is out; a read ahead is split only
    alone (none other out) - reads side by side read whole keep the file's readahead.
  - the ends: a reader's last chunk posts an event to the consumer's libevt Engine; reads_pump runs them there -
    pool_ready, in the order they ended.

reads_create
  `n_units`: the pool's units (a read's record each); `threads`: fetchq's readers (1 to 64). NULL, the reason in
  `err`, when a reader or the engine cannot start.

reads_destroy
  the readers stopped (each after the read it is in), then the reads queued and the ends posted dropped - no
  pool_ready for them, so destroy the pool after. NULL is nothing to do.

reads_fetcher
  the Fetcher pool `pool` (0 to READS_POOLS - 1) is created with: fetch gives a unit's read to the readers at its need,
  its end then that pool's; need moves a read's jobs not yet begun. a read the readers refuse fails closed: its end
  posted as failed (pool_ready's ok false: a FAILED event), never dropped. a unit is out for one pool at a time: a
  fetch of one already out (for either) stops the reads - the pump fails from then on.

reads_pump
  the consumer: the reads that ended handed to their pools (pools[k] pool k's: pool_ready, in the order they ended;
  NULL for a pool not in use - a read's end for one fails the pump closed). wait false: those ended so
  far, no wait (fwd_drain: each layer's routing); true: at least one, waiting for it (fwd_wait_read: a pin waited
  on). false, the reason in `err`, when an end could not be posted (the engine refused it: no memory) - the forward
  fails closed from then on.

reads_out / reads_split
  the reads out now (given to the readers, their ends not yet pumped); those given out split, so far.

reads_ahead_limit
  fetchq_ahead_limit of the readers: the most bytes of reads ahead (a pool's wants, not its pins) being read at
  once; 0: no bound. NULL is nothing to do.
*/
