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
/* a part's C: f32 [matrix][pass][core][half][row octet][column octet][8 m + n] - the gate + up kernel's two matrices
   (24 rows a core a pass), the down's one (16) */
static inline uint64_t npu_c_floats(uint32_t N, uint32_t mats) { return (uint64_t)mats * NPU_TOKENS * N; }

/* parts' A streams, one after another: part p's token t the row tok[p * NPU_TOKENS + t] of x (f32, K wide; -1 a zero
   row). each block of 8 rounded to nearest, ties to even, at its largest value's exponent, the exponent raised by
   one where a mantissa would round past 127 (as the core converts) */
bool npu_pack_a(const float *x, uint32_t K, const int32_t *tok, uint32_t parts, uint8_t *dst, void *stream);

/* parts' C (npu_c_floats(N, mats) each, nr rows a core a pass) into rows: part p's token t's first matrix to
   y0[(row0[p] + t) * N + n], its second (mats 2) to y1's; N a multiple of NPU_CORES * nr, nr of 8 */
bool npu_unpack_c(const float *c, uint32_t N, uint32_t nr, uint32_t mats, uint32_t parts, const uint32_t *row0,
                  float *y0, float *y1, void *stream);

#endif
