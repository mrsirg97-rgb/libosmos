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
*/
