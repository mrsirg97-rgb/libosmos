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
#define NPU_GATE_UP_ROWS 24u /* the gate + up kernel's rows a core a pass: 2304 = 3 passes x 32 cores x 24 */
#define NPU_DOWN_ROWS 16u    /* the down's: 5120 = 10 x 32 x 16 (its 16 rows whole, a core's tile) */

typedef struct Npu Npu;

typedef struct NpuStats
{
  uint64_t layers;  /* npu_gate_up's calls */
  uint64_t parts;   /* launches */
  double wait_us;   /* the host's time in npu_gate_up: the launches and their waits */
} NpuStats;

bool npu_built(void);

/* the kernels from dir, a subdirectory a format (iq2xxs/, iq1s/: final.xclbin, insts.bin - each its own hardware
   context, all live at once; going from one to another ~2.7 ms), for parts of K -> N gate + up; A and C for at most
   max_parts parts (hipMalloc'd, their dma-bufs the NPU's). none found: NULL */
Npu *npu_open(const char *dir, uint32_t K, uint32_t N, uint32_t max_parts, char err[NPU_ERR]);
bool npu_has(const Npu *n, uint32_t qtype); /* a kernel for a GGUF type id */
void npu_close(Npu *n);

/* a host region the weights are read from: registered with HIP (its device address its own); every expert an
   npu_gate_up names must lie in one */
bool npu_region(Npu *n, void *base, uint64_t bytes, char err[NPU_ERR]);

void *npu_a(const Npu *n); /* the parts' A streams (device memory; npu_pack_a's dst) */
void *npu_c(const Npu *n); /* their C (npu_unpack_c's c) */

/* parts launches of qtype's kernel, part p's gate and up weights at wg[p] and wu[p] (w_bytes each, in a region), its A and C the p-th;
   returns when all are done */
bool npu_gate_up(Npu *n, uint32_t qtype, uint32_t parts, const void *const *wg, const void *const *wu,
                 uint64_t w_bytes, char err[NPU_ERR]);

NpuStats npu_stats(const Npu *n);

#endif
