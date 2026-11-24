#ifndef HOST_SLOTS_H
#define HOST_SLOTS_H

#include "osmos/err.h"
#include <stdint.h>

#include "osmos/mem/ledger.h"
#include "osmos/mem/pool.h"

typedef struct HostSlots HostSlots;

HostSlots *host_slots_create(OsmosLedger *l, uint64_t bytes, char err[OSMOS_ERR]);
void host_slots_destroy(HostSlots *h);
Slots host_slots(HostSlots *h);
uint64_t host_slots_bytes(const HostSlots *h);
void *host_slots_base(const HostSlots *h);

#endif

/*
HostSlots
  the pool's memory as plain pinned host pages the GPU reads: one anonymous mapping, registered with HIP once, its
  blocks handed out by an Arena (ARENA_GRAIN: a direct read's alignment), so an O_DIRECT read DMAs straight into it.
  registered memory is outside ttm's pages_limit: the forward counts it against its budget instead.

host_slots_create
  `bytes` rounded up to whole pages, reserved in l (borrowed, given back at destroy; NULL counts nothing) before they
  are mapped, then mapped, populated and registered; NULL with the reason in `err` when the mapping,
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

host_slots_base
  the arena's first byte (host and GPU address alike: registered, mapped) - the NPU's region (npu_region).
*/
