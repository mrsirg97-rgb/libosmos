#include <stdlib.h>

#include <cuda_runtime_api.h>

#include "osmos/mem/hip_slots.h"

/* hip_slots on a discrete card (hip_slots.h; cuda.mk builds this in place of hip_slots.c): memory the CPU writes and
   the GPU reads is pinned host memory, mapped - under CUDA's unified addressing the host pointer is the device's
   too, so the readers pread into it and the kernels read it as they do on the APU. the GPU's reads cross PCIe
   (GPU0 x16 Gen5 ~54 GB/s, DESIGN.md "a second GPU: CUDA": the tiers' host level), not VRAM's ~1.8 TB/s */

struct HipSlots
{
  OsmosLedger *ledger; /* borrowed */
};

static void *alloc(uint64_t bytes, void *ctx)
{
  HipSlots *h = ctx;
  void *p = NULL;
  if (!ledger_reserve(h->ledger, bytes))
    return NULL;
  if (cudaHostAlloc(&p, bytes, cudaHostAllocMapped) != cudaSuccess)
    return ledger_unreserve(h->ledger, bytes), NULL;
  ledger_keep(h->ledger, p, bytes);
  return p;
}

static void release(void *p, void *ctx)
{
  HipSlots *h = ctx;
  ledger_release(h->ledger, p);
  (void)cudaFreeHost(p);
}

HipSlots *hip_slots_create(OsmosLedger *l, char err[OSMOS_ERR])
{
  HipSlots *h = calloc(1, sizeof *h);
  if (!h)
    return osmos_fail(err, "out of memory (the GPU's slots)"), NULL;
  h->ledger = l;
  return h;
}

void hip_slots_destroy(HipSlots *h) { free(h); }

Slots hip_slots(HipSlots *h) { return (Slots){alloc, release, h}; }
