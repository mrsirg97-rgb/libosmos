/* the NPU's routed-expert kernel's input and output on the GPU's side (kernels/npu_pack.hip): an expert's part of
   NPU_TOKENS tokens in, its gate and up out. the layouts are the kernel's (spikes/npu/fused/fused_ds.py) */
#ifndef OSMOS_GPU_NPU_PACK_H
#define OSMOS_GPU_NPU_PACK_H

#include <stdbool.h>
#include <stdint.h>

#define NPU_TOKENS 128u /* a part's tokens: two 64-row halves */
#define NPU_CORES 32u

/* a part's A stream: [K-step of 256][64-row half][K half of 128], each piece 64 x 128 values as bfp16ebs8 (blocks of
   8: an exponent byte, 8 int8 mantissas) in the mmul's sub-tile order (8 rows x 8 values, raster); K * 9 bytes */
static inline uint64_t npu_a_bytes(uint32_t K) { return (uint64_t)NPU_TOKENS * K * 9 / 8; }

/* parts' A streams, one after another: part p's token t the row tok[p * NPU_TOKENS + t] of x (f32, K wide; -1 a zero
   row). each block of 8 rounded to nearest, ties to even, at its largest value's exponent, the exponent raised by
   one where a mantissa would round past 127 (as the core converts) */
bool npu_pack_a(const float *x, uint32_t K, const int32_t *tok, uint32_t parts, uint8_t *dst, void *stream);

/* parts' A from f16 rows (Qwen's swiglu output, its down's input: x16 slots, rows[] each part's), halves: the pieces in
   128-value K halves ([K half][token half]: its down's K 640 no whole 256), else as npu_pack_a's */
bool npu_pack_a_x16(const void *x16, uint32_t K, const int32_t *rows, uint32_t parts, bool halves, uint8_t *dst,
                    void *stream);

/* Qwen's NPU parts (src/q4x: its tables built on the GPU, q4x_expert_tiles): a taken entry's weights (gate, up, down)
   and where its rows go (ga / up at yg floats, y at yd) */
typedef struct NpuPart
{
  const void *gate, *up, *down;
  uint32_t yg, yd;
} NpuPart;
#define NPU_SELECT_MAX 64u /* a layer's most parts (the NPU's share of a 4,096-token step's: ~32 at 10%) */

/* the first `want` whole (NPU_TOKENS) entries of the tables' 128-token run [first, first + count), in table order: their
   counts zeroed (mm_group skips them), each's weights and offsets to sel (the host's), their count to n_sel, their
   tokens' rows (gidx, the gate's and up's input) to tok and their slots (the down's input, x16's) to slot - each
   [part][NPU_TOKENS], the parts past them to want -1 (zero rows: a pack of want parts, before the host knows n_sel).
   ptr the tables' weights ([gate, up, down][entries]); D, ff the tokens' and slots' widths */
bool npu_select_q4x(int32_t *cnt, uint32_t first, uint32_t count, uint32_t want, const void *const *ptr,
                    uint32_t entries, const uint32_t *xg, const uint32_t *yg, const uint32_t *xd, const uint32_t *yd,
                    const int32_t *gidx, uint32_t D, uint32_t ff, NpuPart *sel, uint32_t *n_sel, int32_t *tok,
                    int32_t *slot, void *stream);

#endif
