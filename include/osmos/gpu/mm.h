#ifndef MM_H
#define MM_H

#include <stdbool.h>
#include <stdint.h>

#define MM_TILE 16
#define MM_MAX_TOKENS 2048
#define MM_MAX_GROUP 65535

bool mm_known(uint32_t type);
bool mm_fits(uint32_t type, uint32_t rows, uint32_t cols);
uint32_t mm_col_tiles(uint32_t type, uint32_t n);
bool mm_x16(const float *x, uint32_t xs, uint32_t n, uint32_t cols, void *x16, int32_t *overflow, void *stream);
bool mm_x16_attn(const float *x, uint32_t xs, uint32_t n, uint32_t cols, uint32_t head_dim, uint32_t rope_dim,
                 const float *freqs, const int32_t *pos, void *x16, int32_t *overflow, void *stream);
bool mm_run(uint32_t type, const void *w, uint32_t rows, uint32_t cols, const void *x16, uint32_t n, float *y,
            uint32_t ys, void *stream);
bool mm_tokens(uint32_t type, const void *w0, uint32_t rows0, float *y0, uint32_t ys0, const void *w1, uint32_t rows1,
               float *y1, uint32_t ys1, uint32_t cols, const float *x, uint32_t xs, uint32_t n, void *x16,
               int32_t *overflow, bool ready, void *stream);
bool mm_run2(uint32_t type, const void *w0, uint32_t rows0, float *y0, uint32_t ys0, const void *w1, uint32_t rows1,
             float *y1, uint32_t ys1, uint32_t cols, const void *x16, uint32_t n, void *stream);
#define MM_SILU_MAX_TOKENS 64 /* mm_run_silu's most tokens (past them, the gemm's counts) */
bool mm_run_silu(uint32_t type, const void *w, uint32_t rows, uint32_t cols, const void *x16, uint32_t n, float *y,
                 uint32_t ys, float scale, void *x16_out, int32_t *overflow, void *stream);
#define MM_GATED_MEAN_MAX_HC 8 /* mm_run_gated_mean's most streams: the hyper-connections' most (Q4X_MAX_HC) */
bool mm_run_gated_mean(uint32_t type, const void *w, uint32_t hc, uint32_t dim, uint32_t cols, const void *x16,
                       uint32_t n, const float *xn, float *out, void *out16, uint32_t row0, int32_t *overflow,
                       void *stream);
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
  whether `type` has an mm kernel: Q8_0, Q5_K, Q6_K, IQ1_M, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S; Qwen3.8's IQ3_XXS, IQ4_NL,
  IQ4_XS, Q2_0.

mm_fits
  whether a rows x cols matrix of `type` tiles: rows a multiple of MM_TILE, cols whole blocks and whole slabs (the
  type's for that width) - mm_run and mm_group take it; else the matvec.

mm_x16
  the activations an mm reads: n rows of cols F32 (row t at x + t * xs) into x16 (n rounded up to a multiple of
  MM_TILE rows of cols f16, nearest, tiled (xts cols); the rows past n zero). a value past f16's range sets *overflow
  (device memory; the caller reads it after the step: a refusal, not a guess). n 1 .. MM_MAX_TOKENS, cols a multiple
  of 16.

mm_x16_attn
  mm_x16 of the attention's output, its two roundings on the way: each head's (head_dim values, cols / head_dim of
  them) last rope_dim rotated back (ops_rope's inverse, pos[t] the token's position), then each 32-value block to FP8
  (ops_fp8_round), then f16 - the same bits as the three calls, x read once and not written. also false for cols not
  whole heads, a head not whole blocks, or a rope part not starting at a block.

mm_run
  y[t * ys + r] = sum over c of W[r, c] * x16[t][c] for t < n, queued on `stream`, not waited for. rows a multiple
  of 16, cols of 128 (a slab). false, nothing queued, for a type without a kernel, a shape it cannot tile, n
  outside 1 .. MM_MAX_TOKENS, or a launch refused.

mm_run2
  two matrices of one type and cols on the same x16, in one launch: y0 (rows0, row length ys0) from w0, y1 from w1,
  each the bits mm_run gives it alone (the same tiles; the pair's blocks side by side - more blocks to fill the GPU at
  one token: the shared expert's gate and up, q_a and kv). false as mm_run's for either, or a matrix NULL.

mm_tokens
  a matrix (w1 NULL), or two of one type (mm_run2's pair), over any n tokens of F32 x (row t at x + t * xs): MM_MAX_TOKENS
  a time through the one x16 - mm_x16 of the chunk, then mm_run or mm_run2 into its rows (y + t * ys) - each token's
  sums its own, so the bits of one call whatever the chunks. `ready`: x16 holds this x's f16 already (an op wrote it
  beside x) - taken as it is when n is within one x16, the conversion skipped. both families' dispatch (fwd_mat,
  q4x_mat). false as its calls'.

mm_col_tiles
  the column tiles a block of `type` takes for n tokens (mm_run's, and mm_group's for its max_cnt): a slab decoded
  once for that many tiles' WMMAs - more tiles, more registers and fewer waves to hide the decode. so a group's
  matrices of very different counts are best launched apart, each run of the same tiles its own mm_group (measured:
  a one-tile expert under an 8-tile launch 37% slower). 0 for a type without a kernel.
mm_x16_rows
  slots rows of activations: slot s is row idx[s] of x (row r at x + r * xs; -1: zero) as f16, into x16 (slots rows of
  cols, tiled) - each distinct expert's tokens gathered into slots of its own (whole tiles of MM_TILE). *overflow as
  mm_x16. false - nothing launched - for no slots, cols not a multiple of MM_TILE, xs under cols or not a multiple of
  4, or x not 16-byte aligned (a thread's two 16-byte loads).

mm_run_silu
  mm_run, then each output y = silu(y scale) - q4x_scale_silu's expression, the same bits - written to y and as f16
  into x16_out (tiled as mm_x16 lays it, rows columns wide; the overflow flag set past f16's range): the next matmul's
  activations in the same launch. x16_out must not be x16 (other blocks still read it). false - nothing launched, the
  caller's mm_run and q4x_scale_silu then - for a step the gemm would take (more than 64 tokens of a gemm type) or a
  type it has no epilogue for (Q8_0 alone: the mixers' down).

mm_run_gated_mean
  the hyper-connection mixer's up (hc dim rows: hc streams of dim) with q4x_hc_gated_mean as its epilogue: out[t][d]
  = (sum over s in order of xn[t][s][d] sigmoid(up_t[s dim + d])) / hc - mm_run then q4x_hc_gated_mean, the same
  bits - and its f16 into out16 (dim wide, token t at row row0 + t; NULL: none) for the next matmul; up's F32 output
  never stored. out16 must not be x16 (other blocks still read it). false - nothing launched, the caller's two calls
  then - for a matrix the gemm would take or a type it has no epilogue for (Q8_0 alone: the mixers' up).

mm_group
  n_mat matrices of one type and shape at once: matrix m (w[m]) over the tiled x16 from x16 + xoff[m] (halves; rows
  xts long: its first token at a tile, xoff[m] = tile * 16 xts + its first column * 16), its first cnt[m] (at most
  max_cnt <= MM_MAX_TOKENS) tokens' rows into y + yoff[m] + t * ys - a step's distinct experts (each its own gathered
  tokens),
  or a layer's wo_a groups (each a slice of the same tokens). each tile the same sums as mm_run's: the same bits.
  `w`, `cnt`, `xoff`, `yoff` on the device.
*/
