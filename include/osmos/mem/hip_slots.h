#ifndef HIP_SLOTS_H
#define HIP_SLOTS_H

#include "osmos/mem/resident.h"

Slots hip_slots(void);

#endif

/*
hip_slots
  the pool's and the resident set's memory on this APU (pool.h, resident.h): HSA's allocations from the GPU's coarse-grained global pool
  (what hipMalloc takes: GTT, ttm's pages, which the kernel's reclaim never takes), the CPU allowed to write them - the
  readers pread a unit's parts straight in (measured: 25 GB/s from the page cache with 4 workers, 6.96 GB/s cold; the
  GPU reads them at 221-239 GB/s, DESIGN.md "the memory"). alloc takes ~0.18 ms; free ~0.11 ms and returns at once -
  hipFree waits for the GPU, this does not, so the caller frees only memory the GPU is past. alloc is refused (NULL,
  nothing left allocated) without a GPU, its pool or the CPU's access.
*/
