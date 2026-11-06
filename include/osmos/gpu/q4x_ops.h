#ifndef OSMOS_GPU_Q4X_OPS_H
#define OSMOS_GPU_Q4X_OPS_H

#include <stdbool.h>
#include <stdint.h>

/* Qwen3.8-Flash-Next's own steps (DESIGN.md, "Qwen's forward, op by op"), F32 on the GPU, queued on `stream` */

bool q4x_hc_norm(const float *h, const float *w, uint32_t n_tok, uint32_t hc, uint32_t dim, float eps, float *xn,
                 void *stream);
bool q4x_scale_silu(float *x, uint64_t n, float scale, void *stream);
bool q4x_hc_gated_mean(const float *xn, const float *gate, uint32_t n_tok, uint32_t hc, uint32_t dim, float *out,
                       void *stream);
bool q4x_hc_combine(float *h, const float *out, const float *inject, uint32_t n_tok, uint32_t hc, uint32_t dim,
                    float inject_scale, void *stream);

bool q4x_conv(const float *x, uint32_t n_tok, uint32_t ch, uint32_t taps, const float *w, float *hist, float *y,
              void *stream);
bool q4x_ssm_gates(const float *alpha, const float *beta, const float *dt, const float *a, uint32_t n_tok,
                   uint32_t heads, float *g, float *b, void *stream);
bool q4x_delta(const float *qkv, const float *g, const float *b, uint32_t n_tok, uint32_t k_heads, uint32_t v_heads,
               uint32_t dim, float eps, float *state, float *out, void *stream);
bool q4x_gated_norm(const float *x, const float *w, const float *z, uint32_t n_tok, uint32_t heads, uint32_t dim,
                    float eps, float *y, void *stream);

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
  y[t][c] = silu(sum over k < taps of w[c taps + k] x'[t + k][c]), x' the last taps - 1 inputs before (hist[k][c],
  oldest first) and then x: a depthwise causal conv, its current token the last tap. hist then holds the last taps - 1
  of x' (the sequence's conv state).

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
*/
