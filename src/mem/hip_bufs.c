/* a forward's step buffers from one table (osmos/gpu/bufs.h) */
#include <hip/hip_runtime_api.h>

#include "osmos/mem/ledger_hip.h"
#include "osmos/gpu/bufs.h"

bool gpu_bufs_alloc(GpuBufs *b, OsmosLedger *l, const GpuWant *want, size_t n)
{
  b->ledger = l;
  for (size_t i = 0; i < n; i++)
  {
    if (b->n == GPU_BUFS_MAX || ledger_malloc(l, (void **)want[i].p, want[i].bytes ? want[i].bytes : 1) != hipSuccess)
      return false;
    b->slot[b->n++] = (void **)want[i].p;
  }
  return true;
}

void gpu_bufs_free(GpuBufs *b)
{
  for (uint32_t i = 0; i < b->n; i++)
    ledger_free(b->ledger, *b->slot[i]), *b->slot[i] = NULL;
  b->n = 0;
}
