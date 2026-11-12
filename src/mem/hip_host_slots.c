#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>

#include "osmos/mem/arena.h"
#include "osmos/mem/host_slots.h"

struct HostSlots
{
  uint8_t *base;
  uint64_t bytes;
  bool registered;
  Arena *arena;
};

static void *alloc(uint64_t bytes, void *ctx) { return arena_alloc(((HostSlots *)ctx)->arena, bytes); }
static void release(void *dev, void *ctx) { (void)arena_free(((HostSlots *)ctx)->arena, dev); }

HostSlots *host_slots_create(uint64_t bytes, char err[POOL_ERR])
{
  const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE), grain = page > ARENA_GRAIN ? page : ARENA_GRAIN;
  HostSlots *h = calloc(1, sizeof *h);
  if (!h || !bytes)
    return free(h), snprintf(err, POOL_ERR, "host slots: no memory, or none asked for"), NULL;
  h->bytes = (bytes + grain - 1) / grain * grain;
  void *m = mmap(NULL, h->bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  if (m == MAP_FAILED)
  {
    snprintf(err, POOL_ERR, "host slots: %llu bytes could not be mapped", (unsigned long long)h->bytes);
    free(h);
    return NULL;
  }
  h->base = m;
  void *dev = NULL;
  if (hipHostRegister(h->base, h->bytes, hipHostRegisterMapped) != hipSuccess)
    snprintf(err, POOL_ERR, "host slots: %llu bytes could not be registered with HIP", (unsigned long long)h->bytes);
  else if (h->registered = true, hipHostGetDevicePointer(&dev, h->base, 0) != hipSuccess || dev != h->base)
    snprintf(err, POOL_ERR, "host slots: the GPU's address of the memory is not the host's");
  else if (!(h->arena = arena_create(h->base, h->bytes)))
    snprintf(err, POOL_ERR, "host slots: no memory for the arena");
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
  free(h);
}

static void *tail(uint64_t bytes, void *ctx) { return arena_tail(((HostSlots *)ctx)->arena, bytes); }
static void *take(void *at, uint64_t bytes, void *ctx) { return arena_take(((HostSlots *)ctx)->arena, at, bytes); }

Slots host_slots(HostSlots *h) { return (Slots){alloc, release, h, tail, take}; }
uint64_t host_slots_bytes(const HostSlots *h) { return h ? h->bytes : 0; }
