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
  uint64_t bytes, tail;
  bool registered, tail_registered;
  Arena *arena;
};

static void *alloc(uint64_t bytes, void *ctx) { return arena_alloc(((HostSlots *)ctx)->arena, bytes); }
static void release(void *dev, void *ctx) { (void)arena_free(((HostSlots *)ctx)->arena, dev); }

/* the tail's GPU address the host's, as the rest's (none: true) */
static bool tail_mapped(const HostSlots *h)
{
  void *dev = NULL, *at = h->base + h->bytes - h->tail;
  return !h->tail || (hipHostGetDevicePointer(&dev, at, 0) == hipSuccess && dev == at);
}

HostSlots *host_slots_create(uint64_t bytes, uint64_t tail, char err[POOL_ERR])
{
  const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE), grain = page > ARENA_GRAIN ? page : ARENA_GRAIN;
  HostSlots *h = calloc(1, sizeof *h);
  if (!h || !bytes)
    return free(h), snprintf(err, POOL_ERR, "host slots: no memory, or none asked for"), NULL;
  h->bytes = (bytes + grain - 1) / grain * grain;
  if (tail % (2u << 20) || tail >= h->bytes)
    return free(h), snprintf(err, POOL_ERR, "host slots: a tail not of 2 MiB steps, or not less than the slots"), NULL;
  h->tail = tail;
  void *m = mmap(NULL, h->bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  if (m == MAP_FAILED)
  {
    snprintf(err, POOL_ERR, "host slots: %llu bytes could not be mapped", (unsigned long long)h->bytes);
    free(h);
    return NULL;
  }
  h->base = m;
  void *dev = NULL;
  if (hipHostRegister(h->base, h->bytes - h->tail, hipHostRegisterMapped) != hipSuccess ||
      (h->registered = true, h->tail && hipHostRegister(h->base + h->bytes - h->tail, h->tail, hipHostRegisterMapped) !=
                                            hipSuccess))
    snprintf(err, POOL_ERR, "host slots: %llu bytes could not be registered with HIP", (unsigned long long)h->bytes);
  else if (h->tail_registered = h->tail > 0, hipHostGetDevicePointer(&dev, h->base, 0) != hipSuccess ||
           dev != h->base || !tail_mapped(h))
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
  if (h->tail_registered)
    (void)hipHostUnregister(h->base + h->bytes - h->tail);
  if (h->base)
    munmap(h->base, h->bytes);
  free(h);
}

static void *tail(uint64_t bytes, void *ctx) { return arena_tail(((HostSlots *)ctx)->arena, bytes); }
static void *take(void *at, uint64_t bytes, void *ctx) { return arena_take(((HostSlots *)ctx)->arena, at, bytes); }

Slots host_slots(HostSlots *h) { return (Slots){alloc, release, h, tail, take}; }
uint64_t host_slots_bytes(const HostSlots *h) { return h ? h->bytes : 0; }
/* the pages' first byte, and the tail's bytes (the last of them; lent and given back: host_slots_drop_tail) */
void *host_slots_base(const HostSlots *h) { return h ? h->base : NULL; }
uint64_t host_slots_tail(const HostSlots *h) { return h ? h->tail : 0; }

#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

bool host_slots_drop_tail(HostSlots *h)
{
  if (!h || !h->tail || !h->tail_registered)
    return false;
  uint8_t *at = h->base + h->bytes - h->tail;
  if (hipHostUnregister(at) != hipSuccess)
    return false;
  h->tail_registered = false;
  (void)madvise(at, h->tail, MADV_DONTNEED); /* refused: the pages stay, the memory not given back - nothing wrong */
  return true;
}

bool host_slots_keep_tail(HostSlots *h)
{
  if (!h || !h->tail || h->tail_registered)
    return false;
  uint8_t *at = h->base + h->bytes - h->tail;
  if (madvise(at, h->tail, MADV_POPULATE_WRITE) != 0) /* an older kernel: touched a page at a time */
    for (uint64_t i = 0; i < h->tail; i += 4096)
      at[i] = 0;
  if (hipHostRegister(at, h->tail, hipHostRegisterMapped) != hipSuccess)
    return false;
  h->tail_registered = true;
  if (tail_mapped(h))
    return true;
  (void)hipHostUnregister(at), h->tail_registered = false;
  return false;
}
