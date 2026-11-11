#ifndef FETCHQ_H
#define FETCHQ_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/mem/pool.h"

/* a job's bytes when the queue is idle: a lone unit's parts read in chunks of this, by many workers at once */
#define FETCHQ_CHUNK (1u << 20)

typedef void (*fetch_done_fn)(uint32_t unit, bool ok, void *arg);

typedef struct Fetchq Fetchq;

Fetchq *fetchq_create(int threads, uint32_t n_units, fetch_done_fn done, void *done_arg);
void fetchq_destroy(Fetchq *q);

bool fetchq_read(Fetchq *q, uint32_t unit, const PoolRange *parts, int n, void *const dst[POOL_PARTS], bool split,
                 uint64_t need);
void fetchq_need(Fetchq *q, uint32_t unit, uint64_t need);

typedef struct FetchqStats
{
  uint64_t ahead_peak; /* the most bytes of reads ahead being read at once */
  uint64_t ahead_held; /* the times a worker left a read ahead queued for the bound */
} FetchqStats;

void fetchq_ahead_limit(Fetchq *q, uint64_t bytes);
void fetchq_stats(Fetchq *q, FetchqStats *s);

#endif

/*
Fetchq
  the readers: `threads` workers that pread a unit's parts straight into the memory it was given (its `dst`, GPU
  memory the CPU writes) - each part a job, read whole by one worker, or, `split`, FETCHQ_CHUNK bytes a job read by
  many workers at once - and the one that reads a unit's last job calls `done`. measured on this box (DESIGN.md, "the
  fetcher"): 4 workers read 600 cold 3 MiB ranges at 6.85 GB/s, the SSD's ceiling, p50 1.8 ms a range; more add only
  queueing. a worker takes the job needed soonest: the queue is libevt's (a heap by priority), each job's priority its
  read's `need` - 0 (a miss waited on) first, then the smallest - ties in the order given, so a unit's jobs go side by
  side. a need that moves (fetchq_need: a read ahead now waited on) moves the unit's jobs not yet begun in place
  (libevt's queue_update) - the order is where the workers take, never a second queue. the queue is the only state
  the giver and the workers share; nothing else locks.

a range past its file's end
  a job whose range passes its file's end is read to the end and is whole: a direct read's aligned range around a
  file's last bytes (pool.h's read_align) - the caller's part, that what it wants of the range lies in the file.

fetchq_create
  `threads` workers (1 to 64), started now; `n_units`: the units a read may name (0 .. n_units - 1). `done(unit, ok,
  arg)` runs on a worker once a unit's parts are read - ok false if a read failed; the binary's: reads.h posts the end
  to the forward's engine; the resident set's counts its load. NULL when a worker cannot start (those started are
  stopped again), or no units.

fetchq_destroy
  stops the workers once each finishes the read it is in and joins them; the reads still queued are dropped, their
  `done` never called. NULL is nothing to do.

fetchq_read
  queues the unit's parts and their places (copied; the places stay the caller's until `done`) at `need` and
  returns at once; `split`: in FETCHQ_CHUNK jobs (a lone read, at the SSD's speed), else a job a part (reads side by
  side: each whole, the file's readahead kept). false, nothing queued and no `done`, when the queue cannot grow, the
  unit is past n_units, or it has a read queued already: the caller fails it closed. a queue that refuses partway
  ends the read failed (`done`, ok false, once its queued jobs are read). a read of no bytes is done at once (ok).
  priorities: need 0 the highest, need n below it by n (clamped where an int ends - DESIGN.md: ~53M forward steps of
  the visit clock, then reads ahead keep the order given).

fetchq_need
  the unit's jobs not yet begun moved to `need` (their order among themselves kept); a unit with none queued (read,
  being read, never asked) is nothing to do.

fetchq_ahead_limit, fetchq_stats (DESIGN.md, "0.6.5: ... reads ahead in the background")
  the most bytes of reads ahead - jobs at a need past 0 - being read at once; 0 (as created): no bound. a worker
  leaves the queue's first job where it is while it is a read ahead and the reads ahead in flight plus its bytes
  would pass the bound, and takes it once one ends; with none in flight a read ahead always goes (a job larger than
  the bound is read alone, never starved). a waited job (need 0, or moved there by fetchq_need before a worker took
  it) is never held back and counts nothing: a miss finds workers free and at most the bound's bytes ahead of it on
  the SSD. set before the reads it is to bound (a job a worker holds stays as it was counted). stats: since create.
*/
