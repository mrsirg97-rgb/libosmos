#ifndef OSMOS_GPU_BUFS_H
#define OSMOS_GPU_BUFS_H

/* a component's buffers recorded as they are made and freed from the record (src/mem/hip_bufs.c) - one list a buffer,
   never an allocation's and a teardown's kept in step by hand (0.11.2's rule; 0.16.0: Mini's hand-kept list had
   lost five of its buffers - 11.3 MiB an engine, test_engine_ledger_gpu). device memory or pinned host memory, each
   through one ledger, freed in reverse */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "osmos/mem/ledger.h"

/* every buffer zeroed and GPU_BUFS_PAD bytes past its end - one policy every family's (0.16.0: DeepSeek's zeroed, Inkling's
   padded 16 since its first engine, the others neither) */
#define GPU_BUFS_PAD 16

typedef struct GpuWant
{
  void *p;        /* the pointer's address (a float **, an int32_t **, ...) */
  uint64_t bytes; /* 0: one byte (a buffer every step reads the address of) */
} GpuWant;

typedef struct GpuBuf
{
  void **p;  /* the pointer's address: NULLed when freed */
  bool host; /* pinned host memory (ledger_host_malloc's), else the device's */
} GpuBuf;

typedef struct GpuBufs
{
  GpuBuf *slot; /* the record, grown as it fills */
  uint32_t n, cap;
  OsmosLedger *ledger; /* what they were taken from (borrowed: the build's; every call on one GpuBufs the same) */
} GpuBufs;

/* each wanted buffer allocated on the device through l (osmos/mem/ledger.h; NULL counts nothing) and recorded; false
   at the first refused (those before it recorded: gpu_bufs_free frees them) */
bool gpu_bufs_alloc(GpuBufs *b, OsmosLedger *l, const GpuWant *want, size_t n);
/* one device buffer, or one of pinned host memory, recorded alike */
bool gpu_bufs_one(GpuBufs *b, OsmosLedger *l, void *p, uint64_t bytes);
bool gpu_bufs_host(GpuBufs *b, OsmosLedger *l, void *p, uint64_t bytes);
/* pinned host memory mapped and coherent (the GPU reads what the CPU wrote, uncached: hipHostMallocMapped |
   hipHostMallocCoherent), recorded alike */
bool gpu_bufs_mapped(GpuBufs *b, OsmosLedger *l, void *p, uint64_t bytes);
/* a device buffer made through b's ledger elsewhere (gpu_read_tensor's: p its pointer's address) recorded as if made
   here; false (nothing recorded: the caller frees it) without memory for the record */
bool gpu_bufs_adopt(GpuBufs *b, void *p);
/* every recorded buffer freed, the last made first, its pointer NULL; the record gone - before the memory holding the
   pointers is (the record writes NULL to each) */
void gpu_bufs_free(GpuBufs *b);

#endif
