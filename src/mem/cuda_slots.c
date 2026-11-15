#include <cuda_runtime_api.h>

#include "osmos/mem/hip_slots.h"

/* hip_slots on a discrete card (hip_slots.h; cuda.mk builds this in place of hip_slots.c): memory the CPU writes and
   the GPU reads is pinned host memory, mapped - under CUDA's unified addressing the host pointer is the device's
   too, so the readers pread into it and the kernels read it as they do on the APU. the GPU's reads cross PCIe
   (GPU0 x16 Gen5 ~54 GB/s, DESIGN.md "a second GPU: CUDA": the tiers' host level), not VRAM's ~1.8 TB/s */

static void *alloc(uint64_t bytes, void *ctx)
{
  (void)ctx;
  void *p = NULL;
  return cudaHostAlloc(&p, bytes, cudaHostAllocMapped) == cudaSuccess ? p : NULL;
}

static void release(void *p, void *ctx)
{
  (void)ctx;
  (void)cudaFreeHost(p);
}

Slots hip_slots(void)
{
  return (Slots){alloc, release, NULL};
}
