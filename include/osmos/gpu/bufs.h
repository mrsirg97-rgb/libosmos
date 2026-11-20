#ifndef OSMOS_GPU_BUFS_H
#define OSMOS_GPU_BUFS_H

/* a forward's step buffers allocated from one table and freed from what it recorded (src/mem/hip_bufs.c) - one list
   a buffer, not an allocation's and a teardown's kept in step by hand (0.11.2) */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define GPU_BUFS_MAX 128 /* a forward's table's buffers at most */

typedef struct GpuWant
{
  void *p;        /* the pointer's address (a float **, an int32_t **, ...) */
  uint64_t bytes; /* 0: one byte (a buffer every step reads the address of) */
} GpuWant;

typedef struct GpuBufs
{
  void **slot[GPU_BUFS_MAX];
  uint32_t n;
} GpuBufs;

/* each wanted buffer allocated on the device and recorded; false at the first the GPU refused (those before it
   recorded: gpu_bufs_free frees them) or past GPU_BUFS_MAX */
bool gpu_bufs_alloc(GpuBufs *b, const GpuWant *want, size_t n);
/* every recorded buffer freed, its pointer NULL; the record empty */
void gpu_bufs_free(GpuBufs *b);

#endif
