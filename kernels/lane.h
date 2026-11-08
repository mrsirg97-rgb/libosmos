/* the lane's arithmetic, the same on the GPU and the CPU (hip_lane.c's experts, kernels/ops.hip's lane swiglu): IEEE
   operations only - fmaf, rintf, ldexpf, + - * / - in one order, so the same bits wherever they run (the C side built
   with -ffp-contract=off, as the HIP side is). an exp of its own: the libraries' expf need not agree */
#ifndef LANE_H
#define LANE_H

#include <math.h>

#include "osmos/model/qtype.h"

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

/* the routed experts' int8 inputs (maya's approach, ds4_iq.cu, MIT; kernels/q8.h, src/cpu/expert.c): a block of
   LANE_Q8_VALUES values rounded to int8 by its largest magnitude, q = rint(x * (127 / amax)), its scale amax / 127;
   a weight's half-block (16 values, one scale) against them an integer dot - exact, in any order - then
   lane_q8_part's products, the block's two halves added, each block's scale times that fmaf'd onto the lane's sum.
   a row's blocks go to LANE_Q8_LANES(cols) lanes (l, l + lanes, ...), then the lanes' xor tree */
#define LANE_Q8_VALUES 32
#define LANE_Q8_MAX 127.0f       /* int8's largest magnitude */
#define LANE_Q8_LONG_COLS 4096   /* a row this long or longer (gate / up, 5,120): 32 lanes; shorter (down, 2,304): 16 */
#define LANE_Q8_LANES(cols) ((cols) >= LANE_Q8_LONG_COLS ? 32 : 16)
#define LANE_Q8_EIGHTH 0.125f    /* the IQ types' (2s + 1) / 8 scales, and IQ1_M's delta 1/8 as codes 8 grid +- 1 */

LANE_FN float lane_q8_inv(float amax) { return amax > 0.0f ? LANE_Q8_MAX / amax : 0.0f; }
LANE_FN float lane_q8_scale(float amax) { return amax / LANE_Q8_MAX; }
LANE_FN int lane_q8_round(float x, float inv) { return (int)rintf(x * inv); }
/* whether an expert's three matrices all have the contract's decode: the table's rows naming QTYPE_Q8 (qtype.h) */
#define LANE_Q8_ROW(name, id, values, bytes, paths) || (t == (id) && ((paths) & QTYPE_Q8))
LANE_FN int lane_q8_type(unsigned t) { return 0 QTYPES(LANE_Q8_ROW); }
#undef LANE_Q8_ROW
LANE_FN int lane_q8_types(unsigned g, unsigned u, unsigned d) { return lane_q8_type(g) && lane_q8_type(u) && lane_q8_type(d); }
/* a half-block's sum: its block scale d (an f16's value), its 2s + 1, its integer dot */
LANE_FN float lane_q8_part(float d, int s21, int isum) { return d * (float)s21 * LANE_Q8_EIGHTH * (float)isum; }
/* a K-quant's half with a minimum (Q2_K): (d sc) its dot - (dmin m) x's sum, each a part's (8 sc, 8 m: the eighth's) */
LANE_FN float lane_q8_part_min(float d, int sc8, int isum, float dmin, int m8, int xsum)
{
  return lane_q8_part(d, sc8, isum) - lane_q8_part(dmin, m8, xsum);
}

/* the expert's activation: silu(min(g, lim)) clamp(u, -lim, lim), times scale (a routed expert's weight) */
LANE_FN float lane_swiglu(float g, float u, float lim, float scale)
{
  const float gv = fminf(g, lim), uv = fminf(fmaxf(u, -lim), lim);
  return gv / (1.0f + lane_exp(-gv)) * uv * scale;
}

#endif
