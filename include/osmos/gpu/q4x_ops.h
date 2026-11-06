#ifndef OSMOS_GPU_Q4X_OPS_H
#define OSMOS_GPU_Q4X_OPS_H

#include <stdbool.h>
#include <stdint.h>

#define OPS_Q4X_MAX_KEYS 8192 /* a token's keys the dense attention takes (its scores in a block's LDS) */

/* Qwen3.8-Flash-Next's own steps (DESIGN.md, "Qwen's forward, op by op"), F32 on the GPU, queued on `stream` */

bool q4x_hc_norm(const float *h, const float *w, uint32_t n_tok, uint32_t hc, uint32_t dim, float eps, float *xn,
                 void *stream);
bool q4x_scale_silu(float *x, uint64_t n, float scale, void *stream);
bool q4x_hc_gated_mean(const float *xn, const float *gate, uint32_t n_tok, uint32_t hc, uint32_t dim, float *out,
                       void *stream);
bool q4x_hc_combine(float *h, const float *out, const float *inject, uint32_t n_tok, uint32_t hc, uint32_t dim,
                    float inject_scale, void *stream);

bool q4x_conv(const float *x, uint32_t n_tok, uint32_t ch, uint32_t taps, uint32_t dilation, const float *w,
              float *hist, float *y, void *stream);
bool q4x_ssm_gates(const float *alpha, const float *beta, const float *dt, const float *a, uint32_t n_tok,
                   uint32_t heads, float *g, float *b, void *stream);
bool q4x_delta(const float *qkv, const float *g, const float *b, uint32_t n_tok, uint32_t k_heads, uint32_t v_heads,
               uint32_t dim, float eps, float *state, float *out, void *stream);
bool q4x_gated_norm(const float *x, const float *w, const float *z, uint32_t n_tok, uint32_t heads, uint32_t dim,
                    float eps, float *y, void *stream);

bool q4x_qk(const float *qfull, const float *k, uint32_t n_tok, uint32_t heads, uint32_t kv_heads, uint32_t dim,
            uint32_t rope_dim, const float *q_norm, const float *k_norm, float eps, int32_t pos0, float rope_base,
            float *q, float *gate, float *k_out, void *stream);
bool q4x_attn(const float *q, const float *kc, const float *vc, uint32_t n_tok, uint32_t heads, uint32_t kv_heads,
              uint32_t dim, uint32_t pos0, const float *gate, float *out, void *stream);

bool q4x_route(const float *logits, uint32_t n_tok, uint32_t n_exp, uint32_t k, int32_t *ids, float *w,
               int32_t *rows, void *stream);
bool q4x_expert_table(const int32_t *ids, uint32_t n, const void *base, uint64_t bytes, const void **ptr,
                      int32_t *cnt, int32_t *row, void *stream);
bool q4x_gather(const float *src, const int32_t *rows, uint32_t n, uint32_t len, float *dst, void *stream);
bool q4x_swiglu(const float *g, const float *u, uint64_t n, float *y, void *stream);
bool q4x_moe_sum(const float *y, const float *w, uint32_t n_tok, uint32_t k, uint32_t dim, const float *shared,
                 const float *shared_logit, float *out, void *stream);

bool q4x_rows(uint32_t type, const void *table, uint64_t row_bytes, uint32_t row_values, const int32_t *rows,
              uint32_t n, float *out, void *stream);
bool q4x_ple_gate(const float *key, const float *query, const float *value, uint32_t n_tok, uint32_t hc, uint32_t dim,
                  float *gate, float *gated, void *stream);
bool q4x_add2(float *h, const float *a, const float *b, uint64_t n, void *stream);
bool q4x_streams(const float *x, uint32_t n_tok, uint32_t hc, uint32_t dim, float *h, void *stream);

#endif

/*
the residual h: n_tok tokens of hc streams of dim, h[(t hc + s) dim + d]. false, nothing queued, for an empty shape,
hc past 8 or a launch refused.

q4x_hc_norm
  xn[t][s][d] = h[t][s][d] / sqrt(mean over d of h[t][s][]^2 + eps) * w[s dim + d]: an RMS norm a stream, its
  weight a stream's own.

q4x_scale_silu
  x[i] = silu(x[i] * scale), in place: the mixer's rank after its down matrix (scale 1 / hc).

q4x_hc_gated_mean
  out[t][d] = (sum over s, in order, of xn[t][s][d] sigmoid(gate[t][s][d])) / hc: the streams' gated mean, the
  block's input.

q4x_hc_combine
  h[t][s][d] += 2 sigmoid(inject[t hc + s] inject_scale) out[t][d]: the block's output into every stream, each by its
  own weight (2 sigmoid: 1 at an inject of 0, a plain residual add).

the Gated DeltaNet (a layer's, its tokens in order):

q4x_conv
  y[t][c] = silu(sum over k < taps, in order, of w[c taps + k] x'[t + k dilation][c]), x' the last (taps - 1)
  dilation inputs before (hist[i][c], oldest first) and then x: a depthwise causal conv, its current token the last
  tap (the DeltaNet's dilation 1; the n-gram layer's its n-gram size). hist then holds x''s last (taps - 1) dilation
  (the sequence's conv state).

q4x_ssm_gates
  b[t][h] = sigmoid(beta[t][h]); g[t][h] = softplus(alpha[t][h] + dt[h]) a[h] (a = -exp(A_log): g <= 0, a decay).

q4x_delta
  qkv a token's conv output: q (k_heads x dim), k (k_heads x dim), v (v_heads x dim). q and k each l2-normed a head
  (x / sqrt(sum x^2 + eps)); value head h reads key group h % k_heads; its state S (dim x dim, state[h][j][i] = S[i][j])
  a token at a time: S *= exp(g); delta_j = (v_j - sum_i S[i][j] k_i) b; S[i][j] += k_i delta_j; out[t][h][j] =
  sum_i S[i][j] q_i / sqrt(dim). state holds the sequence's S after the last token.

q4x_gated_norm
  y[t][h][d] = x[t][h][d] / sqrt(mean over d of x^2 + eps) w[d] sigmoid(z[t][h][d]): the DeltaNet's output norm a
  head, gated by z.

the sparse attention (a layer's; one sequence's tokens at positions pos0, pos0 + 1, ...):

q4x_qk
  qfull a token's attn_q: each head's q (dim) then its gate (dim), head after head. q[t][h] = rope(rms_norm(q) q_norm),
  gate[t][h] its gate; k_out[t][g] = rope(rms_norm(k[t][g]) k_norm) (k_out may be a cache's row pos0). rope: the first
  rope_dim of a head, pairs (i, i + rope_dim / 2), angle (pos0 + t) theta_i, theta_i = rope_base^(-2i / rope_dim) as
  ggml builds it (a running product); IMRoPE's sections all the token's position (text) - this rotation.

q4x_attn
  out[t][h][d] = sigmoid(gate[t][h][d]) sum over p <= pos0 + t of softmax_p(q[t][h] . kc[p][g] / sqrt(dim)) vc[p][g][d],
  g = h / (heads / kv_heads) (llama.cpp's GQA: a KV head's heads consecutive). kc, vc: the cache, [p][kv_heads][dim],
  rows 0 .. pos0 + n_tok - 1 written. its keys a block's LDS: at most OPS_Q4X_MAX_KEYS (the dense build's reach).

the MoE (a token's k experts of n_exp: entry e = t k + j its j-th):

q4x_route
  softmax over a token's n_exp logits; its k most probable (the greater first, the lower id among equals) into ids
  [t][j]; their weights w[t][j] = p / (sum of the k p's, at least 6.103515625e-5); rows[e] = t (the token an entry
  reads).

q4x_expert_table
  entry e's matrix base + ids[e] bytes (an expert's slice of the layer's tensor), cnt 1, row e: mv_group_tok's tables,
  each entry its own row (its token's input copied there first).

q4x_gather
  dst[e][] = src[rows[e]][] for e < n, len floats a row: an entry's own copy of its token's input.

q4x_swiglu
  y[i] = silu(g[i]) u[i].

q4x_moe_sum
  out[t][d] = (sum over j, in order, of w[t][j] y[t k + j][d]) + sigmoid(shared_logit[t]) shared[t][d]: the experts'
  weighted sum and the shared expert, gated.

the n-gram table (PLE):

q4x_rows
  out[i][] = row rows[i] of the table, decoded (row_values of the type's values, row_bytes apart): a token's heads'
  rows, one after another (its embedding: heads x row_values) - or a token's row of token_embd. IQ4_NL, Q8_0, Q6_K.

q4x_ple_gate
  key, query: [t][s][dim], each normed a stream already. s_ts = (sum over d, in order, of key[t][s][d] query[t][s][d])
  / sqrt(dim); gate[t][s] = sigmoid(sgn(s_ts) sqrt(max(|s_ts|, 1e-6))); gated[t][s][d] = value[t][d] gate[t][s].

q4x_add2
  h[i] += a[i] + b[i] (a + b first, as llama.cpp adds them).

q4x_streams
  h[t][s][d] = x[t][d] for every stream s: the residual a token starts with, its embedding in each.
*/
