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

uint64_t hy4_hc_scratch(uint32_t n_tok); /* hy4_hc_pre's scratch: floats for n_tok tokens */
bool hy4_hc_pre(const float *h, const float *fn, const float *scale, const float *base, uint32_t rows, uint32_t hc,
                uint32_t dim, float eps, float hc_eps, float magnitude, const float *norm_w, uint32_t n_tok, float *x,
                float *post, float *scratch, void *stream);
bool hy4_hc_post(float *h, const float *y, const float *post, uint32_t hc, uint32_t dim, uint32_t n_tok,
                 void *stream);
bool hy4_engram_rows(const float *rows, const float *norm_w, uint32_t cols, uint32_t row_dim, uint32_t n_tok,
                     float *emb, void *stream);
bool hy4_engram_update(float *h, const float *kv, const float *kv_b, const float *gate, const float *hc_scale,
                       const float *pre_gamma, uint32_t hc, uint32_t dim, uint32_t n_tok, void *stream);
uint64_t hy4_mla_row_bytes(uint32_t latent, uint32_t rope_dim, bool q8); /* a cache row's bytes */
bool hy4_mla_kv(const float *kva, const float *norm_w, uint32_t latent, uint32_t rope_dim, float eps,
                const float *freqs, const int32_t *pos, uint32_t n_tok, void *cache, uint32_t slots, bool q8,
                void *stream);
uint64_t hy4_mla_table_bytes(uint32_t n_tok, uint32_t heads); /* hy4_mla_q's and hy4_mla_out's table: bytes */
bool hy4_mla_q(const float *q, const void *k_b, uint32_t heads, uint32_t nope, uint32_t latent, uint32_t rope_dim,
               const float *freqs, const int32_t *pos, uint32_t n_tok, float *qa, void *table, void *stream);
uint64_t hy4_mla_attn_scratch(uint32_t heads); /* hy4_mla_attn's partials: floats */
bool hy4_mla_attn(const float *qa, const void *cache, uint32_t slots, bool q8, const float *sinks, uint32_t heads,
                  uint32_t latent, uint32_t rope_dim, uint32_t window, const int32_t *sel, const int32_t *n_sel, uint32_t sel_cap,
                  uint32_t pool, float scale, const int32_t *pos, uint32_t n_tok, float *o_lat, float *scratch,
                  bool read, void *stream);
/* a decode's expert tables on the GPU (hy4_route below) */
#define HY4_ROUTE_MAX_EXPERTS 1024
#define HY4_ROUTE_MAX_K 64
bool hy4_route(const int32_t *idx, const float *wtok, uint32_t n_tok, uint32_t k, uint32_t n_exp, uint32_t chunk,
               const void *const base[3], const uint64_t bytes[3], int32_t *gidx, int32_t *sidx, float *wsel,
               int32_t *cnt, int32_t *xrow, const void **wptr, int32_t *fault, void *stream);
bool hy4_mla_qrope(const float *q, uint32_t heads, uint32_t qk, uint32_t row, uint32_t rope_dim, const float *freqs,
                   const int32_t *pos, uint32_t n_tok, float *qa, void *stream);
bool hy4_gate_mul(float *out, const float *gate, uint64_t n, void *stream);
bool hy4_mla_out(const float *o_lat, const void *v_b, const float *gate, uint32_t heads, uint32_t latent,
                 uint32_t v_dim, uint32_t n_tok, float *out, void *table, void *stream);
bool hy4_ix_pool(const float *keys, const void *w_value, const void *w_gate, const float *ape, const float *norm_w,
                 uint32_t dim, uint32_t pool, uint32_t rope_dim, float eps, const float *freqs, uint32_t g0,
                 uint32_t n_groups, float *groups, void *stream);
bool hy4_ix_pool_at(const float *keys, uint32_t slots, const void *w_value, const void *w_gate, const float *ape,
                    const float *norm_w, uint32_t dim, uint32_t pool, uint32_t rope_dim, float eps, const float *freqs,
                    const int32_t *pos, uint32_t n_tok, float *groups, void *stream);
bool hy4_rows_at(float *dst, const float *src, const int32_t *pos, uint32_t n, uint32_t len, uint32_t slots,
                 void *stream);
bool hy4_ix_query(float *q, float *w, uint32_t heads, uint32_t dim, uint32_t rope_dim, const float *freqs,
                  const int32_t *pos, uint32_t n_tok, void *stream);
bool hy4_ix_select(const float *q, const float *w, const float *groups, uint32_t heads, uint32_t dim, uint32_t pool,
                   uint32_t top, const int32_t *pos, uint32_t n_tok, uint32_t max_groups, float *scores, int32_t *sel,
                   int32_t *n_sel, bool read, void *stream);

/* DFly2's draft block (0.11.0) */
#define HY4_DFLY_MAX_ROWS 8 /* a draft block's rows at most (the anchor and its masks) */
bool hy4_dfly_sel(const int32_t *snap, const int32_t *snap_n, int32_t anchor, uint32_t pool, uint32_t top,
                  uint32_t rows, uint32_t cap, int32_t *sel, int32_t *n_sel, void *stream);
bool hy4_dfly_pick(const float *logits, uint32_t n, int32_t *id, float *p, void *stream);

#endif

/*
hy4_hc_pre
  iHC before a sublayer (HYV4FlashHyperConnection), or the head's collapse (HYV4FlashHyperHead): token t's hc streams
  h[t] ([hc][dim], flat hc * dim) mixed by fn's `rows` rows (F32, [rows][hc * dim]): mix_r = (fn_r . flat) *
  rsqrt(mean(flat^2) + eps); pre_k = sigmoid(mix_k * scale[0] + base[k]) + hc_eps; then x = rms(sum over k of pre_k
  h_k, norm_w) (eps again: the sublayer's norm, HYV4FlashRMSNorm). rows 2 hc: post_k = magnitude sigmoid(mix_{hc+k} *
  scale[1] + base[hc + k]) + hc_eps into post ([n_tok][hc]); rows hc (the head): no post (NULL). hc HY4_HC_MAX,
  hc * dim a multiple of 32. the dots in 32 fixed slices of the flat streams (a block each), summed in order: a
  token's bits whatever the step. scratch: hy4_hc_scratch(n_tok) floats.

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
  key's rope part roped at pos[t]; the row into cache row pos[t] - or, slots given (a sliding layer's ring), row
  pos[t] % slots. a row hy4_mla_row_bytes: F32 [latent | rope], or q8 the latent Q8_0 (ggml's: an f16 d = amax / 127
  a 32 values, codes round(x / d); latent a multiple of 32) then the rope part F32 - 800 bytes for Hy4's 512 + 64
  against 2,304 (0.11.0: the context's memory a quarter).

hy4_mla_q
  q_b's output (q [n_tok][heads][nope + rope_dim]) to the queries the cache is read with: each head's nope part
  through its k_b (qa_h[j] = sum_i k_b[h][j][i] q_h[i], j < latent) and its rope part roped at pos[t] - qa
  [n_tok][heads][latent + rope_dim]. the absorption a matvec a (token, head): mv_group_tok's (its Q8_0 sums, a
  token's own), its table in `table` (hy4_mla_table_bytes, the device's).

hy4_mla_attn
  each head's attention over its token's keys: the cache rows at positions max(0, pos[t] - window + 1) .. pos[t]
  (window 0: from 0) - or, sel given (a sparse layer past its dense length), the tokens of the n_sel[t] groups at
  sel[t * sel_cap ..] (ascending; each `pool` positions, the last the token's own, its tokens up to pos[t]): scores
  qa_h . row * scale; the head's sink a logit with no value (softmax over the scores and the sink); o_lat
  [n_tok][heads][latent] = sum over the keys of p row[:latent], in the keys' order - every group selected, the dense
  path's bits. a key's row is its position's, or position % slots (slots given: a ring at least the window long);
  q8: rows hy4_mla_kv's q8 (their latent decoded d times a code, exact in F32, before any of the math).
  flash decoding: a block a group of 8 heads (MQA's rows read once for them) over a split of 64 of its token's keys,
  an online softmax in the keys' order; the splits merged in order with the sink - the splits a token's own: its bits
  whatever the step. scratch: hy4_mla_attn_scratch(heads) floats (the partials of 16 tokens at a time). at most HY4_MLA_MAX_KEYS keys, latent at most HY4_MLA_MAX_LATENT.
  read (a prompt's step): on the matrix cores where the shape fits (16 heads a group, a row at most 576) - f16 Q, K, V
  and P, F32 sums (mla_mm_kernel: a token's output its own keys' alone, its bits whatever the step; not the F32 path's
  bits); env HY4_MLA_F32 set: the F32 path's for every step

hy4_route
  a decode's routed picks (idx/wtok: n_tok x k, each token's k experts distinct, < n_exp) as the expert matvecs'
  tables, no host between: the picks by expert ascending, an expert's tokens ascending - row r's token gidx[r] and
  weight wsel[r], pick m's row sidx[m]; the entries (each expert's rows in chunks of `chunk`, experts ascending) cnt,
  xrow (its first row) and wptr (3 tables, n_tok x k apart: base[i] + e * bytes[i] - gate, up, down), then count 0 to
  n_tok x k (mv_group_tok's skipped entries: a launch to that bound). a pick out of range or twice
  in a token: *fault = 1 (the tables then not used). false, nothing queued, for k 0 or past HY4_ROUTE_MAX_K, n_exp
  past HY4_ROUTE_MAX_EXPERTS, chunk 0, n_tok 0.

hy4_mla_qrope / hy4_gate_mul
  hy4_mla_q's and hy4_mla_out's ends alone (a read's absorptions on the matrix cores: mm_group a head): each head's
  rope part of q (its last rope_dim of qk) roped at pos[t] into qa's (its last rope_dim of row); out *= sigmoid(gate).

hy4_mla_out
  each head's latent through its v_b (v_dim values), times sigmoid of the gate's (gate_proj's output, [n_tok][heads *
  v_dim]) - out [n_tok][heads * v_dim], o_proj's input. the matvecs mv_group_tok's, as hy4_mla_q's (its table).

the sparse layers' compressed indexer (HYV4FlashCompressedIndexer): a token key a position (rms(ix_k x): ops_rms),
groups of `pool` of them pooled into a key each, a query's heads scoring the complete groups before its own.

hy4_ix_pool_at / hy4_rows_at (slots: the keys a ring - a position's row its place mod slots, slots a multiple of
  pool: a group's keys read from it while the ring holds them; 0 a row a position)
  a step's own, its positions on the GPU (pos[t], t < n_tok: a decode replayed without the host): hy4_ix_pool of the
  group each token completes (pos[t] + 1 a multiple of pool: group pos[t] / pool; the rest nothing) - hy4_ix_pool's
  bits; dst's row pos[t] = src's row t (len floats a row).

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
  read (a prompt's step) past 16 tokens: the scores on the matrix cores (f16 inputs, F32 sums: a near-tie at the
  511th group may swap; env HY4_IX_F32 set: the F32 loop's)

hy4_dfly_sel
  a DFly2 draft block's rows' groups (hy4_mla_attn's sel / n_sel, cap apart): row j (position anchor + j, j < rows)
  the snapshot's snap_n groups (a target layer's picks for position anchor - 1, ascending, the last its own group),
  then every group after that one through (anchor + j) / pool - the target's keys for the last accepted token and the
  block's own rows up to the row's. false, nothing queued, for anchor < 1, rows 0 or past HY4_DFLY_MAX_ROWS, or a cap
  under top plus the block's groups.

hy4_dfly_pick
  a logits row's greatest (the lowest index among equal values) into *id and its softmax probability into *p (the
  sum of exp(x - max) by a fixed tree) - a draft token and its confidence, one launch.
*/
