/* a component's buffers recorded as they are made (osmos/gpu/bufs.h) */
#include <stdlib.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/bufs.h"
#include "osmos/mem/ledger_hip.h"

/* room for one more entry (false: no memory for the record - nothing allocated for it) */
static bool room(GpuBufs *b)
{
  if (b->n < b->cap)
    return true;
  const uint32_t cap = b->cap ? 2 * b->cap : 64;
  GpuBuf *s = realloc(b->slot, cap * sizeof *s);
  if (!s)
    return false;
  b->slot = s, b->cap = cap;
  return true;
}

static bool one(GpuBufs *b, OsmosLedger *l, void *p, uint64_t bytes, bool host)
{
  b->ledger = l;
  if (!room(b))
    return false;
  const size_t n = bytes ? (size_t)bytes : 1;
  if ((host ? ledger_host_malloc(l, (void **)p, n, 0) : ledger_malloc(l, (void **)p, n)) != hipSuccess)
    return *(void **)p = NULL, false;
  b->slot[b->n++] = (GpuBuf){(void **)p, host};
  return true;
}

bool gpu_bufs_alloc(GpuBufs *b, OsmosLedger *l, const GpuWant *want, size_t n)
{
  for (size_t i = 0; i < n; i++)
    if (!one(b, l, want[i].p, want[i].bytes, false))
      return false;
  return true;
}

bool gpu_bufs_one(GpuBufs *b, OsmosLedger *l, void *p, uint64_t bytes) { return one(b, l, p, bytes, false); }

bool gpu_bufs_host(GpuBufs *b, OsmosLedger *l, void *p, uint64_t bytes) { return one(b, l, p, bytes, true); }

void gpu_bufs_free(GpuBufs *b)
{
  for (uint32_t i = b->n; i-- > 0;)
  {
    void **p = b->slot[i].p;
    if (*p)
      b->slot[i].host ? ledger_host_free(b->ledger, *p) : ledger_free(b->ledger, *p);
    *p = NULL;
  }
  free(b->slot);
  b->slot = NULL, b->n = b->cap = 0;
}
