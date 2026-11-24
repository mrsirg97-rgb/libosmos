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
  (borrowed; NULL counts nothing); NULL, why said, without the memory for it. the pool and agent are its owner's, from
  create to destroy.
hip_slots_destroy
  gone: its owner's part, after every Pool and Resident given its Slots (their memory released through it). NULL is
  nothing to do.
hip_slots
  the pool's and the resident set's memory on this APU: HSA's allocations from the GPU's coarse-grained global pool
  (what hipMalloc takes: GTT, ttm's pages, which the kernel's reclaim never takes), the CPU allowed to write them so the
  readers pread a unit's parts straight in. free returns at once - hipFree waits for the GPU, this does not, so the
  caller frees only memory the GPU is past. alloc is refused (NULL, nothing left allocated) by the ledger, or without a
  GPU, its pool or the CPU's access.
*/
