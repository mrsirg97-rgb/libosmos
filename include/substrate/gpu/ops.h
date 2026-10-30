#ifndef OPS_H
#define OPS_H

#include <stdbool.h>
#include <stdint.h>

#define OPS_MAX_HC 8
#define OPS_MAX_WINDOW 1024
#define OPS_MAX_EXPERTS 1024
#define OPS_MAX_TOPK 16
#define OPS_MAX_SELECT 1024
#define OPS_MAX_RATIO 16
#define OPS_ATTN_ROW(keys) (((keys) + 3) / 4 * 4) /* a score row: the keys, to a float4 */
/* ops_sparse_attn's scores: n_tok tokens of heads heads, a row a head, then a denominator a head */
#define OPS_ATTN_SCRATCH(n_tok, heads, keys) ((uint64_t)(n_tok) * (heads) * (OPS_ATTN_ROW(keys) + 1))

bool ops_embed(uint32_t type, const void *table, uint32_t dim, const int32_t *ids, uint32_t n_tok, uint32_t hc,
               float *h, void *stream);
bool ops_rms(const float *x, uint32_t n, uint32_t x_stride, const float *w, float eps, uint32_t rows, float *y,
             uint32_t y_stride, void *stream);
#define OPS_HC_PARTS 64
bool ops_hc_dot(const float *fn, const float *h, uint32_t hc, uint32_t dim, uint32_t n_tok, float *partial,
                void *stream);
bool ops_hc_mix(const float *partial, const float *scale, const float *base, uint32_t hc, uint32_t dim, float eps,
                float hc_eps, uint32_t iters, uint32_t n_tok, float *pre, float *post, float *comb, void *stream);
bool ops_hc_pre_rms(const float *h, const float *pre, uint32_t hc, uint32_t dim, const float *w, float eps,
                    uint32_t n_tok, float *x, void *stream);
bool ops_hc_pre(const float *h, const float *pre, uint32_t hc, uint32_t dim, uint32_t n_tok, float *x, void *stream);
bool ops_hc_mean(const float *h, uint32_t hc, uint32_t dim, uint32_t n_tok, float *out, uint32_t out_stride,
                 void *stream);
bool ops_hc_post(const float *y, const float *res, const float *post, const float *comb, uint32_t hc, uint32_t dim,
                 uint32_t n_tok, float *out, void *stream);
bool ops_rope(float *x, uint32_t heads, uint32_t head_dim, uint32_t rope_dim, uint32_t stride, const float *freqs,
              const int32_t *pos, bool inverse, uint32_t n_tok, void *stream);
bool ops_fp8_round(float *x, uint64_t n, void *stream);
bool ops_fp4_round(float *x, uint64_t n, uint32_t block, bool e4m3_scale, void *stream);
bool ops_round_bf16(float *x, uint64_t n, void *stream);
bool ops_engram_gate(float *h, const float *kv, const void *q_w, const void *k_w, uint32_t hc, uint32_t dim, float eps,
                     uint32_t n_tok, void *stream);
bool ops_sparse_attn(const float *q, const float *ring, uint32_t slots, const int32_t *pos, uint32_t window,
                     int32_t floor, const float *ckv, const int32_t *sel, const int32_t *n_sel, uint32_t sel_cap,
                     uint32_t max_sel, const float *sink, float scale, uint32_t heads, uint32_t head_dim,
                     uint32_t n_tok, float *scores, float *o, void *stream);
bool ops_compress(const float *kvc, const float *sc, uint32_t slots, const int32_t *pos, uint32_t ratio, uint32_t dim,
                  uint32_t n_tok, float *out, void *stream);
bool ops_put_rows(float *dst, uint32_t slots, const float *src, const int32_t *pos, uint32_t every, uint32_t dim,
                  uint32_t n_tok, void *stream);
bool ops_index_score(const float *q, const float *keys, const float *w, const int32_t *pos, uint32_t ratio,
                     uint32_t max_n, uint32_t heads, uint32_t head_dim, uint32_t n_tok, float *score, uint32_t stride,
                     void *stream);
bool ops_topk(const float *score, uint32_t stride, const int32_t *pos, uint32_t ratio, uint32_t k, uint32_t n_tok,
              int32_t *sel, uint32_t sel_stride, int32_t *n_sel, void *stream);
bool ops_router(const float *logits, const float *bias, uint32_t n_exp, uint32_t k, float scale, uint32_t n_tok,
                int32_t *idx, float *w, void *stream);
bool ops_swiglu(const float *g, const float *u, uint32_t n, float lim, const float *scale, uint32_t n_tok, float *h,
                void *stream);
bool ops_gather_rows(float *dst, const float *src, const int32_t *idx, uint32_t n, uint32_t len, void *stream);
bool ops_gather_sum(float *y, const float *src, const int32_t *idx, uint32_t m, uint32_t n, uint32_t dim,
                    void *stream);
bool ops_scale(float *x, uint64_t n, float s, void *stream);
bool ops_add(float *y, const float *a, uint64_t n, void *stream);

#endif

/*
ops
  the forward's small steps on the GPU, each ref.py's (oracle/ref.py) in F32: the same operations in the same order
  where the order is the model's (a norm's product then weight, a weight's division then scale), reductions in a
  fixed order (a thread's stride in turn, then a fixed tree) so a token's result is the same bits whatever the others
  beside it or the run. each queued on `stream` (a hipStream_t; NULL the default), not waited for; false - nothing
  queued - for arguments it cannot run (a zero size, a size past its limit) or a launch refused. buffers are the
  device's; a token's rows are contiguous ([n_tok][...]) unless a stride says otherwise. `pos` and `ids` are on the
  device, one a token.

ops_embed
  h[t][k][:] = the table's row ids[t], decoded (quant.h) and rounded to BF16 (the original model's embedding type, as
  ref_gguf.py rounds it), for each of the hc streams k.

ops_rms
  y = (x * 1 / sqrt(mean(x^2) + eps)) * w, a row at a time (w NULL: no weight).

ops_hc_dot
  the hyper-connection mixes' inputs: each of the (2 + hc) hc rows of fn (F32, [rows][hc * dim]) dotted with each
  token's stream h, and the stream with itself, in OPS_HC_PARTS fixed parts: partial[t][row][part] (row (2 + hc) hc
  the sum of squares), each a sum over its part in order. ops_hc_mix sums the parts in order.

ops_hc_mix
  the mixes from ops_hc_dot's partial: mix = the row's parts summed, times 1 / sqrt(the squares' parts summed /
  (hc dim) + eps); then pre = sigmoid(mix * scale[0] + base) + hc_eps, post = 2 sigmoid(mix * scale[1] + base), comb a
  softmax a row (+ hc_eps) and Sinkhorn: a column normalisation, then (iters - 1) times a row's and a column's.

ops_hc_pre_rms
  x = rms(sum over k of pre[k] h[k], w) - the stream mixed into the block's input.

ops_hc_pre
  x[t][d] = sum over k of pre[t][k] h[t][k][d], in k's order (ops_hc_pre_rms's sum, not normed): DSpark's collapse of
  its last block's streams.

ops_hc_mean
  out[t * out_stride + d] = the mean of token t's hc streams at d (summed in stream order, then / hc): DSpark's taps
  of the streams entering its target layers.

ops_hc_post
  out[j] = post[j] y + sum over i of comb[i][j] res[i] - the block's output mixed back into the hc streams.

ops_rope
  each of the `heads` heads' last rope_dim values (a head head_dim apart, a token `stride` apart) rotated in place as
  adjacent pairs, pair i by pos[t] * freqs[i] (rope_freqs); `inverse` the other way.

ops_fp8_round
  each 32 values rounded as the reference's act_quant: a power-of-two scale (amax / 448, at least 1e-4, up to the next
  power of two), the values E4M3 (nearest, ties to even) under it. n a multiple of 32.

ops_fp4_round
  each `block` values rounded as the reference's fp4_act_quant: E2M1 values (nearest, ties to the even mantissa) under
  a scale - amax / 6 rounded to E4M3 (e4m3_scale; amax at least 6 * 2^-9), else up to the next power of two (amax
  at least 6 * 2^-126). n a multiple of block.

ops_round_bf16
  each value rounded to BF16 (nearest, ties to even), kept as F32.

ops_engram_gate
  Engram's update of the hc streams h ([n_tok][hc][dim], in place) from kv ([n_tok][hc + 1][dim]: a key a stream,
  then the value) and the BF16 q_w, k_w ([hc][dim]): h_k += sigmoid(signed sqrt(max(|dot|, 1e-6))) * value, dot =
  sum((h_k * q_w k_w) * key_k) * rstd(h_k) * rstd(key_k) * dim^-0.5 (rounded to F32 from double, as python's).

ops_sparse_attn
  for each token and head: q against its keys - the window's (positions max(pos - window + 1, floor) .. pos, at ring
  row position % slots: floor the sequence's first position with the layer's state, 0 for every position), then the
  n_sel[t] compressed rows of ckv that sel[t * sel_cap ..] names (ckv NULL: none) - each a latent that is the key
  and the value; scaled, softmax with the head's sink, the values summed in that order. slots
  at least window + the step's tokens - 1, so no position a token reads is overwritten by the step. max_sel bounds
  n_sel (the model's top k); scores the device's OPS_ATTN_SCRATCH(n_tok, heads, window + max_sel) floats (window
  alone with no ckv); heads a multiple of 64, head_dim of 256. a score a sum over d ascending, an output over the
  keys in their order: a token's bits do not depend on the other tokens, the heads' tiling, or the batch.

ops_compress
  for each token t whose position completes a compressed row ((pos[t] + 1) a multiple of ratio): the row from the
  ratio positions ending at it - position p at row p % slots of kvc and sc ([slots][dim]) - out[t] = sum over them of
  kvc * softmax(sc), the softmax a value at a time over the positions; any other token's out[t] zeros.

ops_put_rows
  each token t's row src[t] (dim floats) into dst: every 1, at row pos[t] % slots (a ring); every r, only for a
  token completing a row of r positions, at row ((pos[t] + 1) / r - 1) % slots. a copy, so the bits.

ops_index_score
  for each token t (its q at q + t heads head_dim, its w at w + t heads) and its visible rows c < (pos[t] + 1) / ratio
  of keys: score[t stride + c] = sum over heads h of relu(q_h . keys_c) * w_h (heads at most 32, head_dim a multiple of
  16; the heads summed as a 32-lane xor tree sums them). max_n bounds the visible rows (at most stride).

ops_topk
  for each token t: the k highest of score[t stride + 0 .. n) (n its visible rows, as ops_index_score's) - ties to
  the lower row - their rows ascending into sel + t sel_stride and how many into n_sel[t] (all n when n <= k; 0 for
  none visible). deterministic: integer counts, a compaction in row order.

ops_router
  scores = sqrt(softplus(logits)); the k experts of the highest score + bias (ties to the lower expert), in that
  order, into idx; w = their scores / (their sum + 1e-20) * scale.

ops_swiglu
  h = silu(min(g, lim)) * clamp(u, -lim, lim), times scale[t] if scale is not NULL (a routed expert's weight: applied
  before the down projection, as ref.py does).

ops_gather_rows
  dst[i][:] = src[idx[i]][:] for the n rows of len: a copy, so the bits.

ops_gather_sum
  y[t][:] = sum over j < m of src[idx[t * m + j]][:], for t < n: from 0, in j's order (a batch's tokens' experts,
  each token's in ascending id - the same sums as a zeroed y and m ops_add calls).

ops_scale
  x *= s.

ops_add
  y += a.
*/
