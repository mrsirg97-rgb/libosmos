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

bool ops_embed(uint32_t type, const void *table, uint32_t dim, const int32_t *ids, uint32_t n_tok, uint32_t hc,
               float *h, void *stream);
bool ops_rms(const float *x, uint32_t n, uint32_t x_stride, const float *w, float eps, uint32_t rows, float *y,
             uint32_t y_stride, void *stream);
bool ops_hc_mix(const float *h, const float *mix, const float *scale, const float *base, uint32_t hc, uint32_t dim,
                float eps, float hc_eps, uint32_t iters, uint32_t n_tok, float *pre, float *post, float *comb,
                void *stream);
bool ops_hc_pre_rms(const float *h, const float *pre, uint32_t hc, uint32_t dim, const float *w, float eps,
                    uint32_t n_tok, float *x, void *stream);
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
                     const float *ckv, const int32_t *sel, const int32_t *n_sel, uint32_t sel_cap, const float *sink,
                     float scale, uint32_t heads, uint32_t head_dim, uint32_t n_tok, float *o, void *stream);
bool ops_compress(const float *kvc, const float *sc, uint32_t ratio, uint32_t dim, float *out, void *stream);
bool ops_index_score(const float *q, const float *keys, const float *w, uint32_t n, uint32_t heads, uint32_t head_dim,
                     float *score, void *stream);
bool ops_topk(const float *score, uint32_t n, uint32_t k, int32_t *sel, int32_t *n_sel, void *stream);
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

ops_hc_mix
  the hyper-connection mixes from mix = fn h (mv: (2 + hc) hc values a token) and h itself: mix * 1 / sqrt(mean(h^2)
  + eps) then pre = sigmoid(mix * scale[0] + base) + hc_eps, post = 2 sigmoid(mix * scale[1] + base), comb a softmax
  a row (+ hc_eps) and Sinkhorn: a column normalisation, then (iters - 1) times a row's and a column's.

ops_hc_pre_rms
  x = rms(sum over k of pre[k] h[k], w) - the stream mixed into the block's input.

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
  for each token and head: q against its keys - the window's (positions pos - window + 1 .. pos, at ring row
  position % slots), then the n_sel[t] compressed rows of ckv that sel[t * sel_cap ..] names (ckv NULL: none) - each
  a latent that is the key and the value; scaled, softmax with the head's sink, the values summed in that order. slots
  at least window + the step's tokens - 1, so no position a token reads is overwritten by the step.

ops_compress
  a compressed row from `ratio` positions' kvc and sc ([ratio][dim]): out = sum over them of kvc * softmax(sc), the
  softmax a value at a time over the positions.

ops_index_score
  score[c] = sum over heads h of relu(q_h . keys_c) * w_h for the rows c < n (heads at most 32: a lane a head).

ops_topk
  the k highest of score[0 .. n) - ties to the lower row - their rows ascending into sel and how many into *n_sel
  (all n when n <= k). deterministic: integer counts, a compaction in row order.

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
