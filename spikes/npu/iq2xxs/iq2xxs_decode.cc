// 0.7.0 spike: ggml's IQ2_XXS decoded on one AIE2P core - NB blocks of 256 values (66 bytes: d fp16, then 8 groups
// of two 32-bit words: four 8-bit grid indices, then four 7-bit sign indices and a 4-bit scale) to bf16, ggml's
// dequantize_row_iq2_xxs: y = d * (0.5 + s) / 4 * grid[j] * sign. the grid (256 x 8 magnitudes) and the sign
// patterns (128) in the core's own memory; no scalar float (the scalar unit has none). REPS as q2k_decode's.
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "iq2xxs_tables.h"

#ifndef NB
#define NB 16
#endif
#ifndef REPS
#define REPS 1
#endif
#ifndef IQ_VEC
#define IQ_VEC 2 /* 1: bytes unpacked and converted, f32 product; 2: the grid as bf16, one native product */
#endif

// fp16 to f32 by its bits, scaled by 2^-3 (the format's 1/4 and the 2s + 1's halving): integer ops alone; 0 for a
// zero or subnormal d
static inline float half_bits_div8(uint16_t h) {
  const uint32_t e = (h >> 10) & 0x1f;
  const uint32_t b = e ? ((uint32_t)(h & 0x8000) << 16) | ((e + 112 - 3) << 23) | ((uint32_t)(h & 0x3ff) << 13) : 0;
  float f;
  __builtin_memcpy(&f, &b, 4);
  return f;
}

static inline uint32_t u32_at(const uint8_t *p) { // 2-byte aligned: two halves
  const uint16_t *h = (const uint16_t *)p;
  return (uint32_t)h[0] | (uint32_t)h[1] << 16;
}

static inline void block(const uint8_t *blk, bfloat16 *y) {
  const float d8 = half_bits_div8((uint16_t)(blk[0] | blk[1] << 8)); // d / 8: db = d8 * (2s + 1)
  const uint8_t *qs = blk + 2;
  uint32_t a0[8], a1[8];
  aie::vector<int16, 16> s16 = aie::zeros<int16, 16>();
  for (int g = 0; g < 8; g++) {
    a0[g] = u32_at(qs + 8 * g), a1[g] = u32_at(qs + 8 * g + 4);
    s16[g] = (int16)(2 * (a1[g] >> 28) + 1);
  }
  const aie::vector<float, 16> dbv = aie::mul(aie::to_float(s16, 0), aie::broadcast<float, 16>(d8)).to_vector<float>();
#if IQ_VEC == 2
  for (int g = 0; g < 8; g++) {
    uint32_t sg = 0;
    for (int l = 0; l < 4; l++) sg |= (uint32_t)ksigns_iq2xs[(a1[g] >> (7 * l)) & 127] << (8 * l);
    const aie::vector<bfloat16, 32> v = aie::concat(
        aie::load_v<8>((const bfloat16 *)iq2xxs_grid_bf16[a0[g] & 255]), aie::load_v<8>((const bfloat16 *)iq2xxs_grid_bf16[(a0[g] >> 8) & 255]),
        aie::load_v<8>((const bfloat16 *)iq2xxs_grid_bf16[(a0[g] >> 16) & 255]), aie::load_v<8>((const bfloat16 *)iq2xxs_grid_bf16[a0[g] >> 24]));
    // the sign: bf16's top bit flipped where the pattern says, in integer (Peano lowers no bf16 negate)
    const aie::vector<int16, 32> vi = v.cast_to<int16>();
    const aie::vector<bfloat16, 32> sv =
        aie::select(vi, aie::bit_xor(vi, aie::broadcast<int16, 32>((int16)0x8000)), aie::mask<32>::from_uint32(sg)).cast_to<bfloat16>();
    const aie::accum<accfloat, 32> acc = aie::mul(sv, aie::broadcast<bfloat16, 32>((bfloat16)dbv[g]));
    aie::store_v(y, acc.to_vector<bfloat16>());
    y += 32;
  }
#else
  for (int g = 0; g < 8; g++) {
    aie::vector<uint32, 8> rows;
    uint32_t sg = 0;
    for (int l = 0; l < 4; l++) {
      const uint64_t r = iq2xxs_grid[(a0[g] >> (8 * l)) & 255];
      rows[2 * l] = (uint32_t)r, rows[2 * l + 1] = (uint32_t)(r >> 32);
      sg |= (uint32_t)ksigns_iq2xs[(a1[g] >> (7 * l)) & 127] << (8 * l);
    }
    aie::vector<int16, 32> v = aie::unpack(rows.cast_to<uint8>()).cast_to<int16>(); // 8, 25 or 43
    v = aie::select(v, aie::neg(v), aie::mask<32>::from_uint32(sg));
    const aie::accum<accfloat, 32> acc = aie::mul(aie::to_float(v, 0), aie::broadcast<float, 32>(dbv[g]));
    aie::store_v(y, acc.to_vector<bfloat16>());
    y += 32;
  }
#endif
}

extern "C" {
void iq2xxs_decode(const uint8_t *__restrict in, bfloat16 *__restrict out) {
  event0();
  for (int rep = 0; rep < REPS; rep++)
    for (int b = 0; b < NB; b++) block(in + 66 * b, out + 256 * b);
  event1();
}
}
