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

/* the kernels from dir, a subdirectory a format and kind - a model's set (DeepSeek's: iq2xxs/, iq1s/, iq2xs/ gate + up,
   iq2xxs-down/, iq1s-down/, iq2xs-down/, iq2s-down/; Qwen's: iq3xxs/, iq2s/, iq4nl-down/): each
   final.xclbin, insts.bin - its own hardware context, all live at once; going from one to another ~2.7 ms), for an
   expert of D -> FF -> D; A for at most max_parts parts of either kind (hipMalloc'd, its dma-buf the NPU's). none
   found: NULL */
Npu *npu_open(const char *dir, uint32_t D, uint32_t FF, uint32_t max_parts, char err[NPU_ERR]);
bool npu_has(const Npu *n, NpuKind kind, uint32_t qtype); /* a kernel for a GGUF type id */
void npu_close(Npu *n);

/* a host region the weights are read from: registered with HIP (its device address its own) - DeepSeek's expert
   cache; every expert an npu_run names must lie in one of these or an npu_device */
bool npu_region(Npu *n, void *base, uint64_t bytes, char err[NPU_ERR]);

/* GPU memory the kernels read their weights from or write their rows into (a whole hipMalloc: its dma-buf imported) -
   Qwen's resident weights, and every engine's outputs; every output an npu_run names must lie in one */
bool npu_device(Npu *n, void *dev, uint64_t bytes, char err[NPU_ERR]);

void *npu_a(const Npu *n); /* the parts' A streams (device memory; npu_pack_a's dst: K D or FF by the kind) */

/* parts launches of the kind's qtype kernel - part p's weights w0[p] (gate, or the down) and w1[p] (up; unread by the
   down), w_bytes each, in a region; its A the p-th of the kind's size; its rows (NPU_TOKENS of them, row-major, FF wide
   for gate and up, D for the down) written at c0[p] (gate's, or the down's) and c1[p] (up's; the down: unread), each
   in an output. returns when all are done */
bool npu_run(Npu *n, NpuKind kind, uint32_t qtype, uint32_t parts, const void *const *w0, const void *const *w1,
             uint64_t w_bytes, void *const *c0, void *const *c1, char err[NPU_ERR]);

NpuStats npu_stats(const Npu *n);

#endif
