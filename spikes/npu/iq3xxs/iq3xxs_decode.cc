// 0.7.0 spike: ggml's IQ3_XXS decoded on one AIE2P core - NB blocks of 256 values (98 bytes: d fp16, qs[64] grid
// indices - two an octet - then 8 words: four 7-bit sign indices and a 4-bit scale s) to bf16, ggml's
// dequantize_row_iq3_xxs: y = d (0.5 + s) / 2 * grid[j] * sign. an octet's two 4-value grid rows one aligned 16-byte
// load each from iq3_lo / iq3_hi (the values in lanes 0-3 / 4-7, zeros elsewhere), or'd; a group's 4 octets 32 lanes,
// its signs a mask, its scale one bf16 product; no scalar float. REPS as q2k_decode's.
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "iq3xxs_tables.h"
#include "../common/fp16.h"

#ifndef NB
#define NB 16
#endif
#ifndef REPS
#define REPS 1
#endif

static inline float half_bits_div4(uint16_t h) { // fp16 to f32 / 4 (db = d / 4 * (2s + 1)) by its bits
  const uint32_t e = (h >> 10) & 0x1f;
  const uint32_t b = e ? ((uint32_t)(h & 0x8000) << 16) | ((e + 110) << 23) | ((uint32_t)(h & 0x3ff) << 13) : 0;
  float f;
  __builtin_memcpy(&f, &b, 4);
  return f;
}

static inline uint32_t u32_at(const uint8_t *p) {
  const uint16_t *h = (const uint16_t *)p;
  return (uint32_t)h[0] | (uint32_t)h[1] << 16;
}

static inline void block(const uint8_t *blk, bfloat16 *y) {
  const float d4 = fp16_to_f32_shift((uint16_t)(blk[0] | blk[1] << 8), 2);
  const uint8_t *qs = blk + 2, *ss = blk + 66;
  uint32_t aux[8];
  aie::vector<int16, 16> s16 = aie::zeros<int16, 16>();
  for (int g = 0; g < 8; g++) aux[g] = u32_at(ss + 4 * g), s16[g] = (int16)(2 * (aux[g] >> 28) + 1);
  const aie::vector<float, 16> dbv = aie::mul(aie::to_float(s16, 0), aie::broadcast<float, 16>(d4)).to_vector<float>();
  for (int g = 0; g < 8; g++) {
    const uint8_t *q = qs + 8 * g;
#define LO(l) aie::load_v<8>((const int16 *)iq3_lo[q[2 * (l)]])
#define HI(l) aie::load_v<8>((const int16 *)iq3_hi[q[2 * (l) + 1]])
    // the or at 512 bits (Peano lowers none at 128): the 4 octets' low rows, then their high rows
    const aie::vector<int16, 32> vi = aie::bit_or(aie::concat(LO(0), LO(1), LO(2), LO(3)), aie::concat(HI(0), HI(1), HI(2), HI(3)));
#undef LO
#undef HI
    uint32_t sg = 0;
    for (int l = 0; l < 4; l++) sg |= (uint32_t)ksigns_iq2xs[(aux[g] >> (7 * l)) & 127] << (8 * l);
    const aie::vector<bfloat16, 32> sv =
        aie::select(vi, aie::bit_xor(vi, aie::broadcast<int16, 32>((int16)0x8000)), aie::mask<32>::from_uint32(sg)).cast_to<bfloat16>();
    aie::store_v(y, aie::mul(sv, aie::broadcast<bfloat16, 32>((bfloat16)dbv[g])).to_vector<bfloat16>());
    y += 32;
  }
}

extern "C" {
void iq3xxs_decode(const uint8_t *__restrict in, bfloat16 *__restrict out) {
  event0();
  for (int rep = 0; rep < REPS; rep++)
    for (int b = 0; b < NB; b++) block(in + 98 * b, out + 256 * b);
  event1();
}
}
