#ifndef HOST_SLOTS_H
#define HOST_SLOTS_H

#include <stdint.h>

#include "osmos/mem/pool.h"

typedef struct HostSlots HostSlots;

HostSlots *host_slots_create(uint64_t bytes, uint64_t tail, char err[POOL_ERR]);
bool host_slots_drop_tail(HostSlots *h);
bool host_slots_keep_tail(HostSlots *h);
void host_slots_destroy(HostSlots *h);
Slots host_slots(HostSlots *h);
uint64_t host_slots_bytes(const HostSlots *h);
void *host_slots_base(const HostSlots *h);
uint64_t host_slots_tail(const HostSlots *h);

#endif

/*
HostSlots
  the pool's memory as plain pinned host pages the GPU reads (DESIGN.md, "0.3.0: energy a token", target 1): one
  anonymous mapping, registered with HIP once, its blocks handed out by an Arena (ARENA_GRAIN: a direct read's
  alignment). an O_DIRECT read DMAs straight into it - measured 6.14 GB/s a reader at 13% of its core, against 3.63 at
  76% buffered into hip_slots' memory - and the GPU reads it at its own memory's speed (237.5 against 239.7 GB/s: "in
  place"). registered memory is outside ttm's pages_limit: the forward counts it against its budget instead.

host_slots_create
  `bytes` rounded up to whole pages, mapped, populated and registered - its last `tail` bytes (a multiple of 2 MiB, less
  than bytes; 0: none) registered on their own, to be dropped and kept again; NULL with the reason in `err` when the mapping,
  the registration or the arena is refused, or when the GPU's address of the memory is not the host's (the readers
  and the kernels share one pointer: refused rather than guessed).

host_slots_destroy
  unregistered and unmapped: the caller's part - no unit still in it (pool_destroy first), the GPU past it. NULL is
  nothing to do.

host_slots
  the Slots pool_create is given: alloc a block of the arena (NULL when none holds it: the pool frees and asks again,
  or refuses), free it back.

host_slots_bytes
  the arena's size.

host_slots_base / host_slots_tail
  the arena's first byte (host and GPU address alike: registered, mapped), and its tail's bytes - the NPU's regions
  (npu_region): the pages before the tail always registered; the tail's only while kept.

host_slots_drop_tail / host_slots_keep_tail
  0.6.9 (DESIGN.md, "0.6.9"): the tail's pages given back to the system while a long step's buffers take their place
  in GPU memory of their own (registered pages are coherent: the GPU does not keep its writes there in its cache - a
  step's buffers, written and read again, ran 10% slower in them) - unregistered and dropped (MADV_DONTNEED); kept:
  populated and registered again, its bytes zero. the caller's part: no unit in the tail (pool_lend's block, the arena's
  last bytes), the GPU past it. false: no tail, the tail not in that state, or HIP refused (keep: the tail stays dropped).
*/
