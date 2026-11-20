/* a forward's step buffers from one table (osmos/gpu/bufs.h) */
#include <hip/hip_runtime_api.h>

#include "osmos/gpu/bufs.h"

bool gpu_bufs_alloc(GpuBufs *b, const GpuWant *want, size_t n)
{
  for (size_t i = 0; i < n; i++)
  {
    if (b->n == GPU_BUFS_MAX || hipMalloc((void **)want[i].p, want[i].bytes ? want[i].bytes : 1) != hipSuccess)
      return false;
    b->slot[b->n++] = (void **)want[i].p;
  }
  return true;
}

void gpu_bufs_free(GpuBufs *b)
{
  for (uint32_t i = 0; i < b->n; i++)
    hipFree(*b->slot[i]), *b->slot[i] = NULL;
  b->n = 0;
}
