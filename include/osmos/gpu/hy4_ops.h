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

bool hy4_hc_pre(const float *h, const float *fn, const float *scale, const float *base, uint32_t rows, uint32_t hc,
                uint32_t dim, float eps, float hc_eps, float magnitude, const float *norm_w, uint32_t n_tok, float *x,
                float *post, void *stream);
bool hy4_hc_post(float *h, const float *y, const float *post, uint32_t hc, uint32_t dim, uint32_t n_tok,
                 void *stream);
bool hy4_engram_rows(const float *rows, const float *norm_w, uint32_t cols, uint32_t row_dim, uint32_t n_tok,
                     float *emb, void *stream);
bool hy4_engram_update(float *h, const float *kv, const float *kv_b, const float *gate, const float *hc_scale,
                       const float *pre_gamma, uint32_t hc, uint32_t dim, uint32_t n_tok, void *stream);

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
*/
