#ifndef FETCHQ_H
#define FETCHQ_H

#include <stdbool.h>
#include <stdint.h>

#include "substrate/mem/pool.h"

/* a job's bytes when the queue is idle: a lone unit's parts read in chunks of this, by many workers at once */
#define FETCHQ_CHUNK (1u << 20)

typedef void (*fetch_done_fn)(uint32_t unit, bool ok, void *arg);

typedef struct Fetchq Fetchq;

Fetchq *fetchq_create(int threads, fetch_done_fn done, void *done_arg);
void fetchq_destroy(Fetchq *q);

Fetcher fetchq_fetcher(Fetchq *q);

#endif

/*
Fetchq
  the pool's and the resident set's reads: `threads` workers that pread a unit's parts straight into the memory it was
  given (its `dst`, GPU memory the CPU writes), each part a job, or, when the queue is idle (a lone unit: a decode's
  miss), FETCHQ_CHUNK bytes a job read by many workers at once - and the one that reads a unit's last job calls
  `done`. measured on this box (DESIGN.md, "the fetcher"): 4 workers read 600 cold 3 MiB ranges at 6.85 GB/s, the
  SSD's ceiling, p50 1.8 ms a range; more add only queueing. two queues: the urgent reads (a pin's, waited on) and
  those read ahead (a want's), a worker taking the urgent first - a decode's miss is not read behind guesses begun
  before it. the queues are the only state the consumer and the workers share; nothing else locks.

fetchq_create
  `threads` workers (1 to 64), started now. `done(unit, ok, arg)` runs on a worker once a unit's parts are read - ok
  false if a read failed (pool_ready's ok: a FAILED event); the binary's `done` posts to the forward's inbox, which
  calls pool_ready. NULL when a worker cannot start (those
  started are stopped again).

fetchq_destroy
  stops the workers once each finishes the read it is in and joins them; the reads still queued are dropped, their
  `done` never called. NULL is nothing to do. destroy it before the pool its fetcher feeds.

fetchq_fetcher
  the Fetcher the pool (and resident.h) is given: queues the unit's parts and their places (copied) on the urgent
  queue or the ahead one and returns at once; the places must stay the caller's until `done`. its hurry moves a
  unit's jobs not yet begun from the ahead queue to the urgent one's end, in their order - a unit not queued is
  nothing to do.
*/
