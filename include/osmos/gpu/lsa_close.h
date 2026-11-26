/* osmos/gpu/lsa_close.h - LSA's closed blocks bucketed beside the GPU, not in its way (0.18.1). a step that reads none
   of the blocks it closes (every one below sparse_from: a ~3K prompt, a decode there) queues them here: their index
   vectors copied on the step's stream into a pinned slot, an event after the copy; the closer's thread waits on the
   event, buckets each block (src/lsa's lsa_block, in block order: the same buckets as close_blocks' own), then copies
   their centroids and members to the sequence's on its own stream - while the GPU goes on with the step. the step's
   GPU never waits for the host: a 3K prompt's 30 drains of the queue gone.

   join before anything reads the buckets or reuses their memory: a step (the forward's every one, whichever
   sequence), a state's save or load, a reset, a sequence's or the forward's destruction. a job that failed (a GPU
   call, lsa_block) is applied at the join, on the joining thread: its sequence's *broken set - the closer's thread
   never writes a sequence's state. */
#ifndef OSMOS_GPU_LSA_CLOSE_H
#define OSMOS_GPU_LSA_CLOSE_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/mem/ledger.h"

typedef struct LsaCloser LsaCloser;

/* slots jobs in flight at most (a step's full layers'), each of at most blocks_max blocks; its pinned memory through
   l. NULL: refused (memory, a thread, a stream) */
LsaCloser *lsa_closer_create(OsmosLedger *l, uint32_t slots, uint32_t blocks_max);
/* joined, then freed (NULL: nothing) */
void lsa_closer_destroy(LsaCloser *c);

/* blocks b0 .. b0 + k - 1 (dev_ik: block b0's first index vector, LSA_WINDOW x LSA_DIM a block) bucketed beside the
   GPU into dev_cent / dev_mem (block b0's: LSA_BUCKETS x LSA_DIM, LSA_BUCKETS x LSA_CAP a block); the copy queued on
   stream after its work. waits for a free slot when every one is in flight. *broken: set at the join if it fails.
   false: nothing queued (k past blocks_max, a GPU call refused) - the caller closes them itself */
bool lsa_closer_queue(LsaCloser *c, const float *dev_ik, uint32_t b0, uint32_t k, float *dev_cent, int32_t *dev_mem,
                      bool *broken, void *stream);
/* every queued job done and its failures applied (their *broken set); false when one failed since the last join */
bool lsa_closer_join(LsaCloser *c);

#endif
