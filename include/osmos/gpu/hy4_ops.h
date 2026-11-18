#ifndef OSMOS_GPU_HY4_OPS_H
#define OSMOS_GPU_HY4_OPS_H

/* Hy4-flash's small steps on the GPU (kernels/hy4.hip): its iHC and Engram's ends, each the reference's
   (modeling_hy_v4_flash.py) in F32 - DESIGN.md, "0.10.0: Hy4-flash (hy4)". ops.h's conventions: queued on `stream`,
   not waited for; false - nothing queued - for arguments they cannot run; a token's rows contiguous; reductions in a
   fixed order, so a token's bits are the same whatever the others beside it */

#include <stdbool.h>
#include <stdint.h>

#define HY4_HC_MAX 4                    /* the streams (hy4/weights.h's HY4_MAX_HC) */
#define HY4_ENGRAM_EPS 1.1920929e-07f   /* Engram's RMSNorms' eps: the reference's eps unset, F32's machine epsilon */
#define HY4_MLA_MAX_KEYS 4096           /* the keys a query reads at most (a window's 512; the indexer's 2,048) */
#define HY4_MLA_MAX_LATENT 512          /* the latent's width at most (Hy4-flash's kv_lora) */

bool hy4_hc_pre(const float *h, const float *fn, const float *scale, const float *base, uint32_t rows, uint32_t hc,
                uint32_t dim, float eps, float hc_eps, float magnitude, const float *norm_w, uint32_t n_tok, float *x,
                float *post, void *stream);
bool hy4_hc_post(float *h, const float *y, const float *post, uint32_t hc, uint32_t dim, uint32_t n_tok,
                 void *stream);
bool hy4_engram_rows(const float *rows, const float *norm_w, uint32_t cols, uint32_t row_dim, uint32_t n_tok,
                     float *emb, void *stream);
bool hy4_engram_update(float *h, const float *kv, const float *kv_b, const float *gate, const float *hc_scale,
                       const float *pre_gamma, uint32_t hc, uint32_t dim, uint32_t n_tok, void *stream);
bool hy4_mla_kv(const float *kva, const float *norm_w, uint32_t latent, uint32_t rope_dim, float eps,
                const float *freqs, const int32_t *pos, uint32_t n_tok, float *cache, void *stream);
bool hy4_mla_q(const float *q, const void *k_b, uint32_t heads, uint32_t nope, uint32_t latent, uint32_t rope_dim,
               const float *freqs, const int32_t *pos, uint32_t n_tok, float *qa, void *stream);
bool hy4_mla_attn(const float *qa, const float *cache, const float *sinks, uint32_t heads, uint32_t latent,
                  uint32_t rope_dim, uint32_t window, const int32_t *sel, const int32_t *n_sel, uint32_t sel_cap,
                  uint32_t pool, float scale, const int32_t *pos, uint32_t n_tok, float *o_lat, void *stream);
bool hy4_mla_out(const float *o_lat, const void *v_b, const float *gate, uint32_t heads, uint32_t latent,
                 uint32_t v_dim, uint32_t n_tok, float *out, void *stream);
bool hy4_ix_pool(const float *keys, const void *w_value, const void *w_gate, const float *ape, const float *norm_w,
                 uint32_t dim, uint32_t pool, uint32_t rope_dim, float eps, const float *freqs, uint32_t g0,
                 uint32_t n_groups, float *groups, void *stream);
bool hy4_ix_query(float *q, float *w, uint32_t heads, uint32_t dim, uint32_t rope_dim, const float *freqs,
                  const int32_t *pos, uint32_t n_tok, void *stream);
bool hy4_ix_select(const float *q, const float *w, const float *groups, uint32_t heads, uint32_t dim, uint32_t pool,
                   uint32_t top, const int32_t *pos, uint32_t n_tok, uint32_t max_groups, float *scores, int32_t *sel,
                   int32_t *n_sel, void *stream);

#endif

/*
hy4_hc_pre
  iHC before a sublayer (HYV4FlashHyperConnection), or the head's collapse (HYV4FlashHyperHead): token t's hc streams
  h[t] ([hc][dim], flat hc * dim) mixed by fn's `rows` rows (F32, [rows][hc * dim]): mix_r = (fn_r . flat) *
  rsqrt(mean(flat^2) + eps); pre_k = sigmoid(mix_k * scale[0] + base[k]) + hc_eps; then x = rms(sum over k of pre_k
  h_k, norm_w) (eps again: the sublayer's norm, HYV4FlashRMSNorm). rows 2 hc: post_k = magnitude sigmoid(mix_{hc+k} *
  scale[1] + base[hc + k]) + hc_eps into post ([n_tok][hc]); rows hc (the head): no post (NULL). hc at most
  HY4_HC_MAX.

hy4_hc_post
  the sublayer's output y ([n_tok][dim]) back into the streams: h_k = post_k y + h_k (the reference's order, F32).

hy4_engram_rows
  Engram's lookups to the projection's input: each of a token's `cols` rows (rows [n_tok][cols][row_dim], the table's,
  dequantised) through the fp8 round trip the reference trains with (_fp8_round_trip: scale = max(amax, 1e-12) / 448,
  E4M3 nearest-even of x / scale clamped to +-448, times the scale; x + (that - x), as it adds it), then its head's
  RMSNorm (HY4_ENGRAM_EPS) times norm_w's row ([cols][row_dim]) - emb [n_tok][cols * row_dim].

hy4_engram_update
  Engram's residual into the streams (HYV4FlashEngramLayer.forward from its projection on): kv the fused projection's
  output ([n_tok][hc + 1][dim]: a key a stream, then the value) plus kv_b ([hc + 1][dim]); gate_k = sigmoid(signed
  sqrt(max(|g|, 1e-6))), g = sum(rms(key_k) rms(h_k) gate_k) / sqrt(dim) (gate [hc][dim]: the file's key and query
  norms' weights multiplied); v = pre_gamma * rms(value) (no weight); h_k = h_k + (gate_k v + hc_scale_k * v)
  (hc_scale [hc][dim]). every norm HY4_ENGRAM_EPS. h in place.

gated MLA, absorbed (HYV4FlashAttention, its kv_b_proj split as the file keeps it: k_b [heads][latent][nope] and v_b
[heads][v_dim][latent], Q8_0 - read as stored). a cache row is latent + rope_dim F32 values a position (the step's
F32 form: the sequence's cache comes with step 2); rope the reference's: rope_dim / 2 adjacent pairs, pair i by
pos * freqs[i] (freqs: the layer kind's base^(-2i / rope_dim)), on the rope part's values in place.

hy4_mla_kv
  the latent and its key from kv_a's output (kva [n_tok][latent + rope_dim]): the latent RMSNorm'd (norm_w, eps), the
  key's rope part roped at pos[t]; the row into cache row pos[t].

hy4_mla_q
  q_b's output (q [n_tok][heads][nope + rope_dim]) to the queries the cache is read with: each head's nope part
  through its k_b (qa_h[j] = sum_i k_b[h][j][i] q_h[i], j < latent) and its rope part roped at pos[t] - qa
  [n_tok][heads][latent + rope_dim].

hy4_mla_attn
  each head's attention over its token's keys: the cache rows at positions max(0, pos[t] - window + 1) .. pos[t]
  (window 0: from 0) - or, sel given (a sparse layer past its dense length), the tokens of the n_sel[t] groups at
  sel[t * sel_cap ..] (ascending; each `pool` positions, the last the token's own, its tokens up to pos[t]): scores
  qa_h . row * scale; the head's sink a logit with no value (softmax over the scores and the sink); o_lat
  [n_tok][heads][latent] = sum over the keys of p row[:latent], in the keys' order - every group selected, the dense
  path's bits. at most HY4_MLA_MAX_KEYS keys, latent at most HY4_MLA_MAX_LATENT.

hy4_mla_out
  each head's latent through its v_b (v_dim values), times sigmoid of the gate's (gate_proj's output, [n_tok][heads *
  v_dim]) - out [n_tok][heads * v_dim], o_proj's input.

the sparse layers' compressed indexer (HYV4FlashCompressedIndexer): a token key a position (rms(ix_k x): ops_rms),
groups of `pool` of them pooled into a key each, a query's heads scoring the complete groups before its own.

hy4_ix_pool
  groups g0 .. g0 + n_groups - 1 of the token keys (keys [position][dim]; group g positions g pool .. g pool + pool -
  1, every one present): each key's value W_v k and gate W_g k + ape[slot] (w_value, w_gate Q8_0 [dim][dim]; ape
  [pool][dim]), a softmax over the group's slots each dimension, the values summed by it, then RMSNorm (norm_w, eps)
  - pool_groups' - and roped on its last rope_dim at the group's first position (freqs NULL: not roped) - groups
  [g][dim] (g from 0: row g0 the first written).

hy4_ix_query
  the indexer's query and its heads' weights in place: q [n_tok][heads][dim] roped on each head's last rope_dim at
  pos[t]; w [n_tok][heads] times heads^-1/2 dim^-1/2.

hy4_ix_select
  each token's groups: C = pos[t] / pool complete groups before its own (group C). C < top: all of them and its own
  (the reference's top-k picks masked groups then, the causal mask dropping them: the same keys). else the top - 1
  with the greatest score_g = sum over the heads of w_h relu(q_h . groups[g]) (each a fixed order), ties to the
  lower group - an exact selection (the score's k-th greatest by its bits), any C - then its own: sel[t * top ..]
  ascending, n_sel[t] how many. scores: scratch for n_tok x max_groups (C at most max_groups).
*/
