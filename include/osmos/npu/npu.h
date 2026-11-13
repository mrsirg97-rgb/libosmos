/* the NPU (AMD XDNA2, 0.7.0): routed experts' gate and up for a prompt step's hottest experts, beside the GPU. its
   kernel (spikes/npu/fused/fused_ds.py) a compiled xclbin and instruction stream, run through XRT: an expert part of
   NPU_TOKENS tokens a launch, its weights read in place from the expert cache's registered host pages, its A and C in
   hipMalloc'd memory the GPU writes and reads (npu_pack.h), no copy. built with NPU=1 (src/npu/npu_xrt.cpp); else
   every call fails, npu_built false */
#ifndef OSMOS_NPU_NPU_H
#define OSMOS_NPU_NPU_H

#include <stdbool.h>
#include <stdint.h>

#define NPU_ERR 256

typedef struct Npu Npu;

typedef struct NpuStats
{
  uint64_t layers;  /* npu_gate_up's calls */
  uint64_t parts;   /* launches */
  double wait_us;   /* the host's time in npu_gate_up: the launches and their waits */
} NpuStats;

bool npu_built(void);

/* the kernel from dir (final.xclbin, insts.bin), for parts of K -> N gate + up; A and C for at most max_parts parts
   (hipMalloc'd, their dma-bufs the NPU's) */
Npu *npu_open(const char *dir, uint32_t K, uint32_t N, uint32_t max_parts, char err[NPU_ERR]);
void npu_close(Npu *n);

/* a host region the weights are read from: registered with HIP (its device address its own); every expert an
   npu_gate_up names must lie in one */
bool npu_region(Npu *n, void *base, uint64_t bytes, char err[NPU_ERR]);

void *npu_a(const Npu *n); /* the parts' A streams (device memory; npu_pack_a's dst) */
void *npu_c(const Npu *n); /* their C (npu_unpack_c's c) */

/* parts launches, part p's gate and up weights at wg[p] and wu[p] (w_bytes each, in a region), its A and C the p-th;
   returns when all are done */
bool npu_gate_up(Npu *n, uint32_t parts, const void *const *wg, const void *const *wu, uint64_t w_bytes,
                 char err[NPU_ERR]);

NpuStats npu_stats(const Npu *n);

#endif
