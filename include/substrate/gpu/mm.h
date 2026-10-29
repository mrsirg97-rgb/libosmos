#ifndef MM_H
#define MM_H

#include <stdbool.h>
#include <stdint.h>

#define MM_TILE 16
#define MM_MAX_TOKENS 2048
#define MM_MAX_GROUP 65535

bool mm_known(uint32_t type);
bool mm_x16(const float *x, uint32_t xs, uint32_t n, uint32_t cols, void *x16, int32_t *overflow, void *stream);
bool mm_run(uint32_t type, const void *w, uint32_t rows, uint32_t cols, const void *x16, uint32_t n, float *y,
            uint32_t ys, void *stream);
bool mm_x16_rows(const float *x, uint32_t xs, const int32_t *idx, uint32_t slots, uint32_t cols, void *x16,
                 int32_t *overflow, void *stream);
bool mm_group(uint32_t type, const void *const *w, const int32_t *cnt, uint32_t n_mat, uint32_t max_cnt, uint32_t rows,
              uint32_t cols, const void *x16, const uint32_t *xoff, uint32_t xts, float *y, const uint32_t *yoff,
              uint32_t ys, void *stream);

#endif

/*
mm
  the quantized weights' matrix products, the same kernel for 1 to MM_MAX_TOKENS tokens (kernels/mm.h): WMMA over f16
  activations and the weights' exact values rounded to f16, accumulated in F32; a weight slab decoded once for up to
  8 tiles of 16 tokens. the f16 activations (x16) are tiled as the WMMA reads them: token t's column c (rows of xts
  values) at (t / 16) 16 xts + (c / 16) 256 + (t % 16) 16 + c % 16 - each 16 x 16 fragment contiguous. a token's
  results are the same bits whatever the other tokens - so whatever it is batched with, and wherever in the batch.

mm_known
  whether `type` has an mm kernel: Q8_0, Q5_K, Q6_K, IQ1_M, IQ2_XXS, IQ2_XS.

mm_x16
  the activations an mm reads: n rows of cols F32 (row t at x + t * xs) into x16 (n rounded up to a multiple of
  MM_TILE rows of cols f16, nearest, tiled (xts cols); the rows past n zero). a value past f16's range sets *overflow
  (device memory; the caller reads it after the step: a refusal, not a guess). n 1 .. MM_MAX_TOKENS, cols a multiple
  of 16.

mm_run
  y[t * ys + r] = sum over c of W[r, c] * x16[t][c] for t < n, queued on `stream`, not waited for. rows a multiple
  of 16, cols of 128 (a slab). false, nothing queued, for a type without a kernel, a shape it cannot tile, n
  outside 1 .. MM_MAX_TOKENS, or a launch refused.

mm_x16_rows
  slots rows of activations: slot s is row idx[s] of x (row r at x + r * xs; -1: zero) as f16, into x16 (slots rows of
  cols, tiled) - each distinct expert's tokens gathered into slots of its own (whole tiles of MM_TILE). *overflow as
  mm_x16.

mm_group
  n_mat matrices of one type and shape at once: matrix m (w[m]) over the tiled x16 from x16 + xoff[m] (halves; rows
  xts long: its first token at a tile, xoff[m] = tile * 16 xts + its first column * 16), its first cnt[m] (at most
  max_cnt <= MM_MAX_TOKENS) tokens' rows into y + yoff[m] + t * ys - a step's distinct experts (each its own gathered
  tokens),
  or a layer's wo_a groups (each a slice of the same tokens). each tile the same sums as mm_run's: the same bits.
  `w`, `cnt`, `xoff`, `yoff` on the device.
*/
