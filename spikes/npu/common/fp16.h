// fp16 to f32 on an AIE2P core without scalar float (the scalar unit has none: every float op on it is software):
// by the bits, times 2^-shift (a format's power-of-two factor folded in). subnormals normalized by their leading bit -
// IQ3_XXS's scales reach them (13 of 4,096 blocks of a Qwen tensor), a zero d is 0
#pragma once
#include <stdint.h>

static inline float fp16_to_f32_shift(uint16_t h, int shift) {
  const uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  uint32_t b;
  if (e) {
    b = s | ((e + 112 - shift) << 23) | (m << 13);
  } else if (m) { // m 2^-24, m's leading bit p: 1.f 2^(p - 24)
    const int p = 31 - __builtin_clz(m);
    b = s | ((uint32_t)(103 + p - shift) << 23) | ((m << (23 - p)) & 0x7fffff);
  } else {
    b = s;
  }
  float f;
  __builtin_memcpy(&f, &b, 4);
  return f;
}

// fp16 to bf16 times 2^-shift, by the bits alone (no float at all: a scalar float -> bf16 conversion is software too):
// the f32 bits as above, rounded to bf16 nearest-even by integer ops
static inline bfloat16 fp16_to_bf16_shift(uint16_t h, int shift) {
  const uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  uint32_t b;
  if (e) {
    b = s | ((e + 112 - shift) << 23) | (m << 13);
  } else if (m) {
    const int p = 31 - __builtin_clz(m);
    b = s | ((uint32_t)(103 + p - shift) << 23) | ((m << (23 - p)) & 0x7fffff);
  } else {
    b = s;
  }
  const uint16_t r = (uint16_t)((b + 0x7fff + ((b >> 16) & 1)) >> 16);
  bfloat16 x;
  __builtin_memcpy(&x, &r, 2);
  return x;
}
