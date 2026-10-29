// the reference kernels' roundings as device functions (ops.hip, round.hip): BF16, E4M3, E2M1
#pragma once
#include <hip/hip_runtime.h>

#include <cstdint>

namespace rnd {

constexpr float kFp8Max = 448.0f;   // E4M3's largest
constexpr int kFp8Mant = 3;
constexpr int kFp8MinExp = -6;      // E4M3's smallest normal: 2^-6; below it steps of 2^-9

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

}  // namespace rnd
