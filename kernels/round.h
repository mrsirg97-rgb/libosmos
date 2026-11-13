// the reference kernels' roundings as device functions (ops.hip, round.hip): BF16, E4M3, E2M1
#pragma once
#include <hip/hip_runtime.h>

#include <cstdint>

namespace rnd {

constexpr float kFp8Max = 448.0f;   // E4M3's largest
constexpr int kFp8Mant = 3;
constexpr int kFp8MinExp = -6;      // E4M3's smallest normal: 2^-6; below it steps of 2^-9
constexpr float kFp8Floor = 1e-4f;  // act_quant's amax floor
constexpr int kFp8Block = 32;       // act_quant's block: a wave's lanes, a value each
constexpr int kRndWave = 32;

__device__ inline float bf16_round(float x) {
  uint32_t u = __float_as_uint(x);
  if ((u & 0x7fffffffu) > 0x7f800000u) return x;  // NaN as it is
  u += 0x7fffu + ((u >> 16) & 1u);                 // nearest, ties to even
  return __uint_as_float(u & 0xffff0000u);
}

// E4M3 (no infinities), nearest, ties to even; |a| <= 448
__device__ inline float e4m3(float x) {
  const float a = fabsf(x);
  float q;
  if (a < ldexpf(1.0f, kFp8MinExp)) {
    const float step = ldexpf(1.0f, kFp8MinExp - kFp8Mant);
    q = rintf(a / step) * step;
  } else {
    int e;
    frexpf(a, &e);  // a in [2^(e-1), 2^e)
    const float step = ldexpf(1.0f, e - 1 - kFp8Mant);
    q = rintf(a / step) * step;
  }
  return copysignf(q, x);
}

// E2M1 {0, 0.5, 1, 1.5, 2, 3, 4, 6}, nearest, ties to the even mantissa (ref.py _e2m1): 0.25 -> 0, 0.75 -> 1,
// 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4; |v| <= 6
__device__ inline float e2m1(float v) {
  const float a = fabsf(v);
  const float q = a <= 0.25f ? 0.0f : a < 0.75f ? 0.5f : a <= 1.25f ? 1.0f : a < 1.75f ? 1.5f
                : a <= 2.5f ? 2.0f : a < 3.5f ? 3.0f : a <= 5.0f ? 4.0f : 6.0f;
  return copysignf(q, v);
}

// a wave's max, an xor tree: exact in any order, so the same bits as a value at a time
__device__ inline float wave_amax(float m) {
  for (int off = kRndWave / 2; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor(m, off, kRndWave));
  return m;
}

// act_quant's FP8 for a block of kFp8Block values, a lane each (the whole wave calls it): the block's power-of-two
// scale from its max, each value to E4M3 at it
__device__ inline float fp8_block(float v) {
  static_assert(kFp8Block == kRndWave, "a lane a value");
  const float amax = wave_amax(fabsf(v));
  const float s = exp2f(ceilf(log2f(fmaxf(amax, kFp8Floor) / kFp8Max)));
  return e4m3(fminf(fmaxf(v / s, -kFp8Max), kFp8Max)) * s;
}

// the codes a rounded value is kept as (osmos/gpu/kvfmt.h): each the exact value round.h's rounding gave, and
// back - E4M3 (OCP's, no infinities: sign, 4 exponent bits biased 7, 3 mantissa bits; exponent 0 the subnormals,
// steps of 2^-9), E2M1 (sign, the grid's index 0..7: {0, 0.5, 1, 1.5, 2, 3, 4, 6}), UE8M0 (2^e as e + 127). -0 keeps
// its sign
constexpr int kE4m3Bias = 7, kUe8m0Bias = 127;
__device__ inline uint32_t e4m3_code(float q) {
  const uint32_t sign = __float_as_uint(q) >> 31 << 7;
  const float a = fabsf(q);
  if (a < ldexpf(1.0f, kFp8MinExp)) return sign | (uint32_t) (a * ldexpf(1.0f, kFp8Mant - kFp8MinExp));
  int e;
  const float m = frexpf(a, &e);  // a = m 2^e, m in [0.5, 1)
  return sign | (uint32_t) (e - 1 + kE4m3Bias) << kFp8Mant | (uint32_t) ((m * 2.0f - 1.0f) * (1 << kFp8Mant));
}
constexpr uint32_t kE4m3Nan = 0x7F;  // E4M3's NaN (its only one, with the sign): a scale past 448 kept as it
// a code's value from its bits: a normal one's exponent rebiased (127 - 7) and its mantissa moved up (23 - 3 bits);
// a subnormal one m 2^-9 (a small integer times a power of two: exact); the sign its own bit
constexpr uint32_t kE4m3Rebias = (127 - kE4m3Bias) << 23;
// a value's code (never NaN's: rounding clamps to 448, code 0x7E) without a branch: both forms, one selected
__device__ inline float e4m3_value_finite(uint32_t c) {
  const uint32_t mag = c & 0x7F, sign = (c & 0x80) << 24;
  const uint32_t normal = (mag << (23 - kFp8Mant)) + kE4m3Rebias;
  const uint32_t sub = __float_as_uint((float) mag * 0.001953125f);  // m 2^-9
  return __uint_as_float(sign | (mag >> kFp8Mant ? normal : sub));
}
__device__ inline float e4m3_value(uint32_t c) {
  const uint32_t mag = c & 0x7F, sign = (c & 0x80) << 24;
  if (mag == kE4m3Nan) return __builtin_nanf("");
  if (mag >> kFp8Mant)
    return __uint_as_float(sign | ((mag << (23 - kFp8Mant)) + kE4m3Rebias));
  return __uint_as_float(sign | __float_as_uint((float) mag * 0.001953125f));  // m 2^-9
}
// an unsigned E4M3 scale's value as ggml reads one (ggml_ue4m3_to_fp32, before its 0.5): the sign bit not read, the
// code 0x7F zero (E4M3's NaN; 0xFF, read without its sign, is 0x7F's finite neighbour's 480 - as ggml)
__device__ inline float ue4m3_value(uint32_t c) {
  return c == kE4m3Nan ? 0.0f : e4m3_value_finite(c & 0x7F);
}
__device__ inline uint32_t e2m1_code(float q) {
  const float a = fabsf(q);
  const uint32_t i = a == 0.0f ? 0 : a == 0.5f ? 1 : a == 1.0f ? 2 : a == 1.5f ? 3 : a == 2.0f ? 4 : a == 3.0f ? 5
                   : a == 4.0f ? 6 : 7;
  return (__float_as_uint(q) >> 31) << 3 | i;
}
// the grid's value from its bits: index i >= 2 is exponent (i >> 1) - 1 + 127 and mantissa bit i & 1 (1, 1.5, 2, 3,
// 4, 6), i = 1 is 0.5, 0 is 0; the sign its own bit
__device__ inline float e2m1_value(uint32_t c) {
  const uint32_t i = c & 0x7, sign = (c & 0x8) << 28;
  const uint32_t mag = i >= 2 ? ((i >> 1) + 126u) << 23 | (i & 1u) << 22 : i ? 0x3F000000u : 0u;
  return __uint_as_float(sign | mag);
}
__device__ inline uint32_t ue8m0_code(float s) { return (uint32_t) (ilogbf(s) + kUe8m0Bias); }
// 2^(c - 127): F32's own exponent field (c 1 .. 254); 0xFF UE8M0's NaN; 0 the one subnormal, 2^-127
__device__ inline float ue8m0_value(uint32_t c) {
  if (c == 0xFF) return __builtin_nanf("");
  return c ? __uint_as_float(c << 23) : __uint_as_float(1u << 22);
}

// fp8_block, its code and its block's scale code too (the whole wave calls it)
__device__ inline float fp8_block_coded(float v, uint32_t& code, uint32_t& scale) {
  const float amax = wave_amax(fabsf(v));
  const float s = exp2f(ceilf(log2f(fmaxf(amax, kFp8Floor) / kFp8Max)));
  const float q = e4m3(fminf(fmaxf(v / s, -kFp8Max), kFp8Max));
  code = e4m3_code(q), scale = ue8m0_code(s);
  return q * s;
}

}  // namespace rnd
