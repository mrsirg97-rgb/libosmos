#ifndef HIP_PINNER_H
#define HIP_PINNER_H

#include "substrate/mem/pool.h"

Pinner hip_pinner(void);

#endif

/*
hip_pinner
  the pool's Pinner on this APU: HSA's hsa_amd_memory_lock_to_pool (the CPU's coarse-grained system pool, mapped for
  the GPUs) and hsa_amd_memory_unlock - what hipHostRegister and hipHostUnregister call, without hipHostUnregister's
  wait for every stream to drain (measured: an unregister with a kernel in flight waits for it, so each of a layer's
  unpins stalled the GPU; an unlock returns at once, and the GPU reads a lock's pages at the same 241 GB/s). coarse,
  not a plain lock's fine-grained pages: those a decode step 4% slower (75 -> 78 ms). the caller's part: a range is
  unpinned only when the GPU is past it - the pool unpins only units no hold is on. a pin is refused - false, nothing
  left locked - without the pool or a GPU, or unless the device's pointer for the range is the host's own address: the
  pool hands the GPU the mapping's addresses, which on gfx1151 is what a lock gives (measured, DESIGN.md "the
  memory"); a device where it is not fails closed instead of reading the wrong bytes.
*/
