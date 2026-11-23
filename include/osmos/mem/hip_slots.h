#ifndef HIP_SLOTS_H
#define HIP_SLOTS_H

#include "osmos/err.h"
#include "osmos/mem/ledger.h"
#include "osmos/mem/resident.h"

typedef struct HipSlots HipSlots;

HipSlots *hip_slots_create(OsmosLedger *l, char err[OSMOS_ERR]);
void hip_slots_destroy(HipSlots *h);
Slots hip_slots(HipSlots *h);

#endif

/*
hip_slots_create
  the GPU's coarse-grained pool and the CPU agent found (the runtime up first), each allocation from it counted in l
  (osmos/mem/ledger.h: borrowed; NULL counts nothing); NULL, why said, without the memory for it. 0.16.0: the pool
  and agent were a process's static, found at a first use - now its owner's, from create to destroy.
hip_slots_destroy
  gone: its owner's part, after every Pool and Resident given its Slots (their memory released through it). NULL is
  nothing to do.
hip_slots
  the pool's and the resident set's memory on this APU (pool.h, resident.h): HSA's allocations from the GPU's coarse-grained global pool
  (what hipMalloc takes: GTT, ttm's pages, which the kernel's reclaim never takes), the CPU allowed to write them - the
  readers pread a unit's parts straight in (measured: 25 GB/s from the page cache with 4 workers, 6.96 GB/s cold; the
  GPU reads them at 221-239 GB/s, DESIGN.md "the memory"). alloc takes ~0.18 ms; free ~0.11 ms and returns at once -
  hipFree waits for the GPU, this does not, so the caller frees only memory the GPU is past. alloc is refused (NULL,
  nothing left allocated) by the ledger, or without a GPU, its pool or the CPU's access.
*/
