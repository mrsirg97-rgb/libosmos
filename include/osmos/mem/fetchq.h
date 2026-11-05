#ifndef FETCHQ_H
#define FETCHQ_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/mem/pool.h"

/* a job's bytes when the queue is idle: a lone unit's parts read in chunks of this, by many workers at once */
#define FETCHQ_CHUNK (1u << 20)

typedef void (*fetch_done_fn)(uint32_t unit, bool ok, void *arg);

typedef struct Fetchq Fetchq;

Fetchq *fetchq_create(int threads, fetch_done_fn done, void *done_arg);
void fetchq_destroy(Fetchq *q);

bool fetchq_read(Fetchq *q, uint32_t unit, const PoolRange *parts, int n, void *const dst[POOL_PARTS], bool split);

#endif

/*
Fetchq
  the readers: `threads` workers that pread a unit's parts straight into the memory it was given (its `dst`, GPU
  memory the CPU writes), in the order they were given - each part a job, read whole by one worker, or, `split`, FETCHQ_CHUNK
  bytes a job read by many workers at once (a lone read: the giver decides - reads.h) - and the one that reads a unit's last job
  calls `done`. measured on this box (DESIGN.md, "the fetcher"): 4 workers read 600 cold 3 MiB ranges at 6.85 GB/s,
  the SSD's ceiling, p50 1.8 ms a range; more add only queueing. which read goes first is not the readers' to choose:
  the pool's reads come in the order reads.h gives them out (by need), the resident set's load in its own. the queue
  is the only state the giver and the workers share; nothing else locks.

fetchq_create
  `threads` workers (1 to 64), started now. `done(unit, ok, arg)` runs on a worker once a unit's parts are read - ok
  false if a read failed; the binary's: reads.h posts the end to the forward's engine; the resident set's counts its
  load. NULL when a worker cannot start (those started are stopped again).

fetchq_destroy
  stops the workers once each finishes the read it is in and joins them; the reads still queued are dropped, their
  `done` never called. NULL is nothing to do.

fetchq_read
  queues the unit's parts and their places (copied; the places stay the caller's until `done`) behind those queued
  and returns at once; `split`: in FETCHQ_CHUNK jobs (a lone read, at the SSD's speed), else a job a part (reads side
  by side: each whole, the file's readahead kept). false, nothing queued and no `done`, when the queue cannot grow: the caller fails it closed. a
  read of no bytes is done at once (ok).
*/
