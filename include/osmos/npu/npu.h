/* the NPU (AMD XDNA2, 0.7.0): a prompt step's hottest routed experts, beside the GPU - their gate and up (D -> FF, two
   matrices a launch) and their down (FF -> D). its kernels (spikes/npu/fused/fused_ds.py) compiled xclbins and
   instruction streams, run through XRT: an expert part of NPU_TOKENS tokens a launch, its weights read in place from
   the expert cache's registered host pages, its A and C in hipMalloc'd memory the GPU writes and reads (npu_pack.h),
   no copy. built with NPU=1 (src/npu/npu_xrt.cpp); else every call fails, npu_built false */
#ifndef OSMOS_NPU_NPU_H
#define OSMOS_NPU_NPU_H

#include <stdbool.h>
#include <stdint.h>

#define NPU_ERR 256

typedef enum NpuKind
{
  NPU_GATE_UP = 0, /* D -> FF, gate and up: 24 rows a core a pass (2304 = 3 passes x 32 cores x 24) */
  NPU_DOWN = 1,    /* FF -> D: 16 rows a core a pass, its tile those rows whole (5120 = 10 x 32 x 16) */
  NPU_KINDS
} NpuKind;

#define NPU_GATE_UP_ROWS 24u
#define NPU_DOWN_ROWS 16u

typedef struct Npu Npu;

typedef struct NpuStats
{
  uint64_t calls; /* npu_run's */
  uint64_t parts; /* launches */
  double wait_us; /* the host's time in npu_run: the launches and their waits */
} NpuStats;

bool npu_built(void);

/* the kernels from dir, a subdirectory a format and kind (iq2xxs/, iq1s/, iq2xs/ gate + up; iq2xxs-down/, iq1s-down/,
   iq2xs-down/, iq2s-down/: each
   final.xclbin, insts.bin - its own hardware context, all live at once; going from one to another ~2.7 ms), for an
   expert of D -> FF -> D; A and C for at most max_parts parts of either kind (hipMalloc'd, their dma-bufs the NPU's).
   none found: NULL */
Npu *npu_open(const char *dir, uint32_t D, uint32_t FF, uint32_t max_parts, char err[NPU_ERR]);
bool npu_has(const Npu *n, NpuKind kind, uint32_t qtype); /* a kernel for a GGUF type id */
void npu_close(Npu *n);

/* a host region the weights are read from: registered with HIP (its device address its own); every expert an
   npu_run names must lie in one */
bool npu_region(Npu *n, void *base, uint64_t bytes, char err[NPU_ERR]);

void *npu_a(const Npu *n); /* the parts' A streams (device memory; npu_pack_a's dst: K D or FF by the kind) */
void *npu_c(const Npu *n); /* their C (npu_unpack_c's c) */

/* parts launches of the kind's qtype kernel - part p's weights w0[p] (gate, or the down) and w1[p] (up; unread by the
   down), w_bytes each, in a region; its A and C the p-th of the kind's size. returns when all are done */
bool npu_run(Npu *n, NpuKind kind, uint32_t qtype, uint32_t parts, const void *const *w0, const void *const *w1,
             uint64_t w_bytes, char err[NPU_ERR]);

NpuStats npu_stats(const Npu *n);

#endif
