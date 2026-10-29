#ifndef MM_H
#define MM_H

#include <stdbool.h>
#include <stdint.h>

#define MM_TILE 16
#define MM_MAX_GROUP 65535

bool mm_known(uint32_t type);
bool mm_x16(const float *x, uint32_t xs, uint32_t n, uint32_t cols, void *x16, int32_t *overflow, void *stream);
bool mm_run(uint32_t type, const void *w, uint32_t rows, uint32_t cols, const void *x16, uint32_t n, float *y,
            uint32_t ys, void *stream);
bool mm_x16_rows(const float *x, uint32_t xs, const int32_t *idx, uint32_t n_mat, uint32_t cols, void *x16,
                 int32_t *overflow, void *stream);
bool mm_group(uint32_t type, const void *const *w, const int32_t *cnt, uint32_t n_mat, uint32_t rows, uint32_t cols,
              const void *x16, const uint32_t *xoff, uint32_t xts, float *y, const uint32_t *yoff, uint32_t ys,
              void *stream);

#endif

/*
mm
  the quantized weights' matrix products, the same kernel for one token or sixteen (kernels/mm.h): WMMA over f16
  activations and the weights' exact values rounded to f16, accumulated in F32. a token's results are the same bits
  whatever the other tokens in its tile - so whatever it is batched with.

mm_known
  whether `type` has an mm kernel: Q8_0, Q5_K, Q6_K, IQ1_M, IQ2_XXS, IQ2_XS.

mm_x16
  the activations an mm reads: n rows of cols F32 (row t at x + t * xs) into x16 ([MM_TILE][cols] f16, nearest; the
  rows past n zero). a value past f16's range sets *overflow (device memory; the caller reads it after the step: a
  refusal, not a guess). n 1 .. MM_TILE, cols a multiple of 16.

mm_run
  y[t * ys + r] = sum over c of W[r, c] * x16[t][c] for t < n, queued on `stream`, not waited for. rows a multiple
  of 16, cols of 128 (a slab). false, nothing queued, for a type without a kernel, a shape it cannot tile, n
  outside 1 .. MM_TILE, or a launch refused.

mm_x16_rows
  n_mat matrices' activations: matrix m's slot j is row idx[m * MM_TILE + j] of x (row r at x + r * xs; -1: zero) as
  f16, into x16 ([n_mat][MM_TILE][cols]) - each distinct expert's tokens, gathered. *overflow as mm_x16.

mm_group
  n_mat matrices of one type and shape at once: matrix m (w[m]) over its token t's f16 x at x16 + xoff[m] + t * xts,
  its first cnt[m] tokens' rows into y + yoff[m] + t * ys - a step's distinct experts (each its own gathered tokens),
  or a layer's wo_a groups (each a slice of the same tokens). each tile the same sums as mm_run's: the same bits.
  `w`, `cnt`, `xoff`, `yoff` on the device.
*/
