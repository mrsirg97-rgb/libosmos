// a rotation's pair (i, i + dim / 2) at position pos (include/osmos/gpu/q4x_ops.h's Q4xRope): its cos and sin - the
// plain rotation's own bits (theta_i a running product of theta_scale, ggml's rope cache), or YaRN's interpolation
// and magnitude (llama.cpp's rope_multi and rope_yarn)
#pragma once

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

__device__ inline void rope_cs(const Q4xRope& r, float pos, uint32_t i, float* c, float* s) {
  if (r.ext_factor == 0.0f) {  // the plain rotation: theta_i the running product, as ggml's rope cache builds it
    float theta = pos;
    for (uint32_t j = 0; j < i; j++) theta *= r.theta_scale;
    *c = cosf(theta), *s = sinf(theta);
    return;
  }
  // YaRN's: theta_i pos theta_scale^i as llama.cpp's GPU rope forms it (at a million positions an F32 angle's last
  // bit is ~0.06 rad - the reference's own rounding, not a running product's)
  const float theta = pos * powf(r.theta_scale, (float) i);
  const float y = ((float) i - r.corr_lo) / fmaxf(0.001f, r.corr_hi - r.corr_lo);
  const float mix = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * r.ext_factor;
  const float th = r.freq_scale * theta * (1 - mix) + theta * mix;
  *c = cosf(th) * r.mscale, *s = sinf(th) * r.mscale;
}
