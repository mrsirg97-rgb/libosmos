#ifndef OSMOS_GPU_LFM3_OPS_H
#define OSMOS_GPU_LFM3_OPS_H

/* LFM3's own steps (DESIGN.md, "vibes-flash: LFM3"), F32 on the GPU, queued on `stream` */

#include <stdbool.h>
#include <stdint.h>

#include "osmos/gpu/q4x_ops.h"

#define LFM3_CONV_TAPS_MAX 8 /* the short convolution's taps at most (its state taps - 1 rows) */

bool lfm3_shortconv(const float *bcx, uint32_t n_tok, uint32_t dim, uint32_t taps, const float *w, float *ring,
                    uint32_t slots, uint32_t cur, float *out, void *stream);
bool lfm3_kv_turn(void *kc, uint32_t rows, uint32_t kv_heads, uint32_t dim, const Q4xRope *from, const Q4xRope *to,
                  void *stream);

#endif

/*
lfm3_shortconv
  the short convolution's operator between its projections: bcx a token's in_proj output (B, C, x: dim each, a row
  3 dim), u = B * x, out[t][c] = C[t][c] * (sum over k < taps, in order, of w[c taps + k] u'[t + k][c]), u' the
  (taps - 1) rows before (the state: ring slot cur, [taps - 1][dim], oldest first) then the step's u - a depthwise
  causal conv, no activation (Liquid's Lfm3ShortConv). after token t, u''s last taps - 1 rows up to it into slot
  (cur + 1 + t) % slots, for the step's last min(n_tok, slots) tokens (q4x_conv's ring: a rollback reads an earlier
  slot). a token's bits its own, whatever the step.

lfm3_kv_turn
  a full layer's cached keys turned from one rotation to another (a YaRN stage's crossing: DESIGN.md, "vibes-flash:
  2M"): kc's rows 0 .. rows - 1 (a row a position, kv_heads heads of dim, Q4X_KV_BLOCK_BYTES' Q8_0) each dequantized,
  a rotated pair's (i, i + dim / 2) turned back by `from`'s cos and sin at the row's position (rope_cs: their
  magnitude's square divided out) and on by `to`'s, the rest times to's magnitude over from's, then Q8_0 again in
  place (ggml's rounding). dim 128 (Q4X_QK_HEAD_NARROW) alone; the two rotations' dims and pairs the same.
*/
