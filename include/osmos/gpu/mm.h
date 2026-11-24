#ifndef MM_H
#define MM_H

#include <stdbool.h>
#include <stdint.h>

#define MM_TILE 16
#define MM_MAX_TOKENS 2048
#define MM_MAX_GROUP 65535

bool mm_known(uint32_t type);
bool mm_fits(uint32_t type, uint32_t rows, uint32_t cols);
bool mm_gemm(uint32_t type, uint32_t rows, uint32_t cols, uint32_t n);
uint32_t mm_col_tiles(uint32_t type, uint32_t n);
bool mm_x16(const float *x, uint32_t xs, uint32_t n, uint32_t cols, void *x16, int32_t *overflow, void *stream);
bool mm_x16_attn(const float *x, uint32_t xs, uint32_t n, uint32_t cols, uint32_t head_dim, uint32_t rope_dim,
                 const float *freqs, const int32_t *pos, void *x16, int32_t *overflow, void *stream);
bool mm_run(uint32_t type, const void *w, uint32_t rows, uint32_t cols, const void *x16, uint32_t n, float *y,
            uint32_t ys, void *stream);
bool mm_tokens(uint32_t type, const void *w0, uint32_t rows0, float *y0, uint32_t ys0, const void *w1, uint32_t rows1,
               float *y1, uint32_t ys1, uint32_t cols, const float *x, uint32_t xs, uint32_t n, void *x16,
               int32_t *overflow, bool ready, void *stream);
/* mm_tokens on an x16 already holding all n rows (its producer's: a norm's, an activation's f16 beside its F32) - any
   n, MM_MAX_TOKENS rows a launch over x16's slices (its tiles contiguous: x16_at), no conversion; the same bits as
   mm_tokens converting x. x16's buffer the caller's: n rows (whole tiles) of cols */
bool mm_tokens_x16(uint32_t type, const void *w0, uint32_t rows0, float *y0, uint32_t ys0, const void *w1, uint32_t rows1,
                   float *y1, uint32_t ys1, uint32_t cols, const void *x16, uint32_t n, void *stream);
#define MM_X16_ROWS_MAX 65536 /* an x16's rows at most (a producer's: ops_rms_x16) - a family's step, past MM_MAX_TOKENS */
bool mm_tokens_scaled(uint32_t type, const void *w0, uint32_t rows0, float *y0, uint32_t ys0, float s0, const void *w1,
                      uint32_t rows1, float *y1, uint32_t ys1, float s1, uint32_t cols, const float *x, uint32_t xs,
                      uint32_t n, void *x16, int32_t *overflow, bool ready, void *stream);
#define MM_BATCH_MAX 8     /* mm_batch's most matrices */
#define MM_BATCH_TOKENS 64 /* its most tokens in one launch (past them, the gemm's counts: a launch a part) */
typedef struct MmPart
{
  const void *w;     /* the matrix */
  uint32_t rows;     /* its rows */
  const void *x16;   /* its input (mm_x16's) */
  float *y;          /* its output: token t's rows at y + t * ys */
  uint32_t ys;
} MmPart;
bool mm_batch(uint32_t type, uint32_t cols, const MmPart *parts, uint32_t count, uint32_t n, void *stream);
bool mm_run2(uint32_t type, const void *w0, uint32_t rows0, float *y0, uint32_t ys0, const void *w1, uint32_t rows1,
             float *y1, uint32_t ys1, uint32_t cols, const void *x16, uint32_t n, void *stream);
#define MM_SILU_MAX_TOKENS 64 /* mm_run_silu's most tokens (past them, the gemm's counts) */
bool mm_run_silu(uint32_t type, const void *w, uint32_t rows, uint32_t cols, const void *x16, uint32_t n, float *y,
                 uint32_t ys, float scale, void *x16_out, int32_t *overflow, void *stream);
bool mm_run_x16(uint32_t type, const void *w, uint32_t rows, uint32_t cols, const void *x16, uint32_t n, float *y,
                uint32_t ys, void *x16_out, int32_t *overflow, void *stream);
#define MM_GATED_MEAN_MAX_HC 8 /* mm_run_gated_mean's most streams: the hyper-connections' most (Q4X_MAX_HC) */
bool mm_run_gated_mean(uint32_t type, const void *w, uint32_t hc, uint32_t dim, uint32_t cols, const void *x16,
                       uint32_t n, const float *xn, float *out, void *out16, uint32_t row0, int32_t *overflow,
                       void *stream);
bool mm_x16_rows(const float *x, uint32_t xs, const int32_t *idx, uint32_t slots, uint32_t cols, void *x16,
                 int32_t *overflow, void *stream);
bool mm_x16_heads(const float *x, uint32_t xs, uint32_t n, uint32_t heads, uint32_t hs, uint32_t cols, void *x16,
                  int32_t *overflow, void *stream);
bool mm_group(uint32_t type, const void *const *w, const int32_t *cnt, uint32_t n_mat, uint32_t max_cnt, uint32_t rows,
              uint32_t cols, const void *x16, const uint32_t *xoff, uint32_t xts, float *y, const uint32_t *yoff,
              uint32_t ys, void *stream);

#ifdef __cplusplus
extern "C" {
#endif
/* W4A4 on the FP4 matrix cores (kernels/mm_fp4.hip, sm_120a only - mm_fp4_ok false elsewhere): x's n rows (stride
   xs) quantized as ModelOpt's activations under the input scale gs (an E4M3 scale a 16, E2M1) into q (n x cols/2
   bytes) and sc (n x cols/16); then y[t * ys + r] = scale * sum_c W[r, c] xq[t, c], W ggml 40's rows (an NVFP4
   matrix, its tensor scale and the input's in `scale`). cols whole 64s, rows whole 8s */
bool mm_fp4_ok(void);
bool mm_fp4_act(const float *x, uint32_t xs, uint32_t n, uint32_t cols, float gs, void *q, void *sc, void *stream);
bool mm_fp4(const void *w, uint32_t rows, uint32_t cols, const void *q, const void *sc, uint32_t n, float scale,
            float *y, uint32_t ys, void *stream);
/* the same over ModelOpt's layout (packed [rows, cols/2] in k order, the low nibble first; E4M3 scales [rows, cols/16]
   at wscale): the training engine's (incarnate train/fp4.py). the sums times scale (and *scale_dev, a device scale,
   when not NULL); y F32, or (bf16) BF16 rounded to nearest */
bool mm_fp4_mo(const void *packed, const void *wscale, uint32_t rows, uint32_t cols, const void *q, const void *sc,
               uint32_t n, float scale, const float *scale_dev, uint32_t bf16, void *y, uint32_t ys, void *stream);
/* mm_fp4_act's quantization of F32 or (bf16) BF16 rows, the tensor scale gs or the device's *gs_dev; sr_seed 0
   nearest, else stochastic (a gradient's: unbiased), seeded */
bool mm_fp4_quant(const void *x, uint32_t bf16, uint32_t xs, uint32_t n, uint32_t cols, float gs, const float *gs_dev,
                  uint32_t sr_seed, void *q, void *sc, void *stream);
/* a BF16 tensor's NVFP4 tensor scale (amax / 6 / 448) into *gs on the device (bits: a word of scratch) */
bool mm_fp4_amax(const void *x_bf16, uint64_t n, uint32_t *bits, float *gs, void *stream);
/* a MoE's routed experts at 2 bits times NVFP4 activations (W2A4): expert e's codes [rows, cols/4] (4 a byte in k
   order, the lowest bits first: one of the 4 values of an E2M1-subset grid, lut holding code c's E2M1 nibble at bits
   4c), its UE4M3 scales [rows, cols/16] and tensor scale gsw[e] (the experts' back to back: codes [E, rows, cols/4],
   wscale [E, rows, cols/16]); the n tokens sorted by expert, expert e's offsets[e] .. offsets[e + 1] (on the device:
   no host sync), q / sc their mm_fp4_quant. y [n, ys] the sums times scale * gsw[e] (and *scale_dev, the
   activations' device scale, when not NULL), F32 or (bf16) BF16. tile_start E + 1 words of device scratch */
bool mm_2b_grouped(const void *codes, const void *wscale, const float *gsw, uint32_t e, uint32_t lut, uint32_t rows,
                   uint32_t cols, const uint32_t *offsets, uint32_t *tile_start, const void *q, const void *sc,
                   uint32_t n, float scale, const float *scale_dev, uint32_t bf16, void *y, uint32_t ys,
                   void *stream);
/* W2A16: the same experts (codes, wscale, gsw, lut, the grouping) times BF16 activations x [n, xs] on the BF16 matrix
   cores, each code decoded to its grid value times its block's scale (exact in BF16), gsw[e] on the sums. trans 0:
   y [n, rows] = x [n, cols] W^T (the forward); trans 1: y [n, cols] = x [n, rows] W (the input's gradient). y F32 or
   (bf16) BF16. the reduction a multiple of 64 */
bool mm_w2a16_grouped(const void *codes, const void *wscale, const float *gsw, uint32_t e, uint32_t lut, uint32_t rows,
                      uint32_t cols, uint32_t trans, const uint32_t *offsets, uint32_t *tile_start, const void *x,
                      uint32_t xs, uint32_t n, uint32_t bf16, void *y, uint32_t ys, void *stream);
#ifdef __cplusplus
}
#endif

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
  IQ4_XS, Q2_0; Inkling-2's NVFP4 (its blocks: a tensor's own scale applied by its role) and MXFP8.

mm_fits
  whether a rows x cols matrix of `type` tiles: rows a multiple of MM_TILE, cols whole blocks and whole slabs (the
  type's for that width) - mm_run and mm_group take it; else the matvec.

mm_gemm
  whether mm_run of n tokens on a rows x cols matrix of `type` runs on the prompt's gemm (a slab decoded once for 256
  tokens; past 64 tokens, the matrix's shape the tile's) - its bits the tile's either way.

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

mm_batch
  count matrices of one type and cols (at most MM_BATCH_MAX), each on its own x16, in one launch: each part's y the
  bits mm_run gives it alone (the same tiles side by side - a decode's projections of a layer, launches fewer and
  blocks to fill the GPU). n tokens; past MM_BATCH_TOKENS (the gemm's counts) each part its own mm_run. false as
  mm_run's for any part, or count 0 or past MM_BATCH_MAX.

mm_tokens
  a matrix (w1 NULL), or two of one type (mm_run2's pair), over any n tokens of F32 x (row t at x + t * xs):
  MM_MAX_TOKENS a time through the one x16 - mm_x16 of the chunk, then mm_run or mm_run2 into its rows (y + t * ys) -
  each token's sums its own, so the bits of one call whatever the chunks. `ready`: x16 holds this x's f16 already (an op
  wrote it beside x) - taken as it is when n is within one x16, the conversion skipped. both families' dispatch
  (fwd_mat, q4x_mat). false as its calls'.

mm_tokens_scaled
  mm_tokens with each matrix's sums times its scale (s0, s1) as they are stored - a per-tensor scale (NVFP4's) without
  a pass of its own: the bits of mm_tokens then ops_scale (one F32 multiply of the same sum); 1 the plain store.
mm_col_tiles
  the column tiles a block of `type` takes for n tokens (mm_run's, and mm_group's for its max_cnt): a slab decoded
  once for that many tiles' WMMAs - more tiles, more registers and fewer waves to hide the decode. so a group's
  matrices of very different counts are best launched apart, each run of the same tiles its own mm_group. 0 for a
  type without a kernel.
mm_x16_rows
  slots rows of activations: slot s is row idx[s] of x (row r at x + r * xs; -1: zero) as f16, into x16 (slots rows of
  cols, tiled) - each distinct expert's tokens gathered into slots of its own (whole tiles of MM_TILE). *overflow as
  mm_x16. false - nothing launched - for no slots, cols not a multiple of MM_TILE, xs under cols or not a multiple of
  4, or x not 16-byte aligned (a thread's two 16-byte loads).

mm_run_x16
  mm_run, its y also as f16 into x16_out (mm_x16's bytes of that y, rows columns wide) in the same launch: a matmul
  whose output is the next one's input (Inkling's loop control: its down, then its gain and forget). x16_out must not
  be x16. false - nothing launched, the caller's mm_run and mm_x16 then - for a step the gemm would take, more than
  MM_SILU_MAX_TOKENS tokens, or a type other than BF16 and Q8_0.

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

mm_x16_heads
  every head's slice of n rows (row t at x + t xs; head h's cols values at h hs of it) to its own tiled x16, in one
  pass along the rows: head h's at x16 + h n16 cols halves (n16: n in whole tiles) - each as mm_x16(x + h hs, xs, n,
  cols) would put it, the same bits; mm_tokens(..., ready) a head at a time on them. n at most MM_MAX_TOKENS; x 16-
  byte aligned, xs and hs whole float4s.

mm_group
  n_mat matrices of one type and shape at once: matrix m (w[m]) over the tiled x16 from x16 + xoff[m] (halves; rows
  xts long: its first token at a tile, xoff[m] = tile * 16 xts + its first column * 16), its first cnt[m] (at most
  max_cnt <= MM_MAX_TOKENS) tokens' rows into y + yoff[m] + t * ys - a step's distinct experts (each its own gathered
  tokens), or a layer's wo_a groups (each a slice of the same tokens). each tile the same sums as mm_run's: the same
  bits - but for Q6_K under 2,048 columns (mm_run's short shape, 64-value slabs; mm_group's its tuned 128: another K
  split, other bits). a token's bits its own either way.
  false for cols not a multiple of 128 (mm_fits takes some: Q8_0 at 320, Q6_K at 1,344 - mm_run then, not this).
  `w`, `cnt`, `xoff`, `yoff` on the device.
*/
