#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>

#include "osmos/mem/ledger_hip.h"
#include "osmos/mem/arena.h"
#include "osmos/mem/host_slots.h"
#include "osmos/err.h"

struct HostSlots
{
  uint8_t *base;
  uint64_t bytes;
  bool registered;
  Arena *arena;
  OsmosLedger *ledger; /* borrowed: what its mapping is counted in */
  bool kept;           /* its mapping in the ledger (released at destroy) */
};

static void *alloc(uint64_t bytes, void *ctx) { return arena_alloc(((HostSlots *)ctx)->arena, bytes); }
static void release(void *dev, void *ctx) { (void)arena_free(((HostSlots *)ctx)->arena, dev); }

HostSlots *host_slots_create(OsmosLedger *l, uint64_t bytes, char err[OSMOS_ERR])
{
  const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE), grain = page > ARENA_GRAIN ? page : ARENA_GRAIN;
  HostSlots *h = calloc(1, sizeof *h);
  if (!h || !bytes)
    return free(h), osmos_fail(err, "host slots: no memory, or none asked for"), NULL;
  h->bytes = (bytes + grain - 1) / grain * grain, h->ledger = l;
  if (!ledger_reserve(l, h->bytes))
  {
    osmos_fail(err, "host slots: refused by the ledger: %s", ledger_why(l));
    free(h);
    return NULL;
  }
  void *m = mmap(NULL, h->bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  if (m == MAP_FAILED)
  {
    osmos_fail(err, "host slots: %llu bytes could not be mapped", (unsigned long long)h->bytes);
    ledger_unreserve(l, h->bytes);
    free(h);
    return NULL;
  }
  h->base = m;
  ledger_keep(l, h->base, h->bytes), h->kept = true;
  void *dev = NULL;
  if (hipHostRegister(h->base, h->bytes, hipHostRegisterMapped) != hipSuccess)
    osmos_fail(err, "host slots: %llu bytes could not be registered with HIP", (unsigned long long)h->bytes);
  else if (h->registered = true, hipHostGetDevicePointer(&dev, h->base, 0) != hipSuccess || dev != h->base)
    osmos_fail(err, "host slots: the GPU's address of the memory is not the host's");
  else if (!(h->arena = arena_create(h->base, h->bytes)))
    osmos_fail(err, "host slots: no memory for the arena");
  else
    return h;
  host_slots_destroy(h);
  return NULL;
}

void host_slots_destroy(HostSlots *h)
{
  if (!h)
    return;
  arena_destroy(h->arena);
  if (h->registered)
    (void)hipHostUnregister(h->base);
  if (h->base)
    munmap(h->base, h->bytes);
  if (h->kept)
    ledger_release(h->ledger, h->base);
  free(h);
}

Slots host_slots(HostSlots *h) { return (Slots){alloc, release, h}; }
uint64_t host_slots_bytes(const HostSlots *h) { return h ? h->bytes : 0; }
/* the pages' first byte: host and GPU address alike (registered, mapped) - the NPU's region (npu_region) */
void *host_slots_base(const HostSlots *h) { return h ? h->base : NULL; }
