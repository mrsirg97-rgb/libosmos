/* the lane's arithmetic, the same on the GPU and the CPU (hip_lane.c's experts, kernels/ops.hip's lane swiglu): IEEE
   operations only - fmaf, rintf, ldexpf, + - * / - in one order, so the same bits wherever they run (the C side built
   with -ffp-contract=off, as the HIP side is). an exp of its own: the libraries' expf need not agree */
#ifndef LANE_H
#define LANE_H

#include <math.h>

#if defined(__HIPCC__)
#define LANE_FN __host__ __device__ static inline
#else
#define LANE_FN static inline
#endif

#define LANE_EXP_HI 88.72283905f /* past it e^x is past F32: infinity */
#define LANE_LOG2E 1.44269504089f
#define LANE_LN2_HI 0.693145751953125f /* ln 2 split: the high part exact in F32's top bits */
#define LANE_LN2_LO 1.428606765330187e-06f

/* e^x: x = k ln 2 + r (|r| <= ln 2 / 2), e^r by its Taylor series to r^7 (Horner, fmaf), times 2^k. within ~2 ulp of
   e^x; for x below about -87 the result would be subnormal - the lane's swiglu never asks (its argument -min(g, lim)
   >= -lim) */
LANE_FN float lane_exp(float x)
{
  if (x > LANE_EXP_HI)
    return INFINITY;
  const float k = rintf(x * LANE_LOG2E);
  float r = fmaf(-k, LANE_LN2_HI, x);
  r = fmaf(-k, LANE_LN2_LO, r);
  float p = 1.0f / 5040.0f;
  p = fmaf(p, r, 1.0f / 720.0f);
  p = fmaf(p, r, 1.0f / 120.0f);
  p = fmaf(p, r, 1.0f / 24.0f);
  p = fmaf(p, r, 1.0f / 6.0f);
  p = fmaf(p, r, 0.5f);
  p = fmaf(p, r, 1.0f);
  p = fmaf(p, r, 1.0f);
  return ldexpf(p, (int)k);
}

/* the expert's activation: silu(min(g, lim)) clamp(u, -lim, lim), times scale (a routed expert's weight) */
LANE_FN float lane_swiglu(float g, float u, float lim, float scale)
{
  const float gv = fminf(g, lim), uv = fminf(fmaxf(u, -lim), lim);
  return gv / (1.0f + lane_exp(-gv)) * uv * scale;
}

#endif
