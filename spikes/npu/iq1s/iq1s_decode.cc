// 0.7.0 spike: ggml's IQ1_S decoded on one AIE2P core - NB blocks of 256 values (50 bytes: d fp16, qs[32] the
// grid indices' low bytes, qh[8]: a group's three high index bits a row, its scale s (bits 12-14) and its shift's
// sign (bit 15)) to bf16, ggml's dequantize_row_iq1_s: y = d * (2s + 1) * (grid[j] + delta), delta +-0.125. the
// 2,048-row grid of {-1, 0, 1} as bf16 in the core's memory (32 KB); no scalar float. REPS as q2k_decode's.
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "iq1s_tables.h"

#ifndef NB
#define NB 8
#endif
#ifndef REPS
#define REPS 1
#endif

static inline float half_bits(uint16_t h) { // fp16 to f32 by its bits; 0 for a zero or subnormal d
  const uint32_t e = (h >> 10) & 0x1f;
  const uint32_t b = e ? ((uint32_t)(h & 0x8000) << 16) | ((e + 112) << 23) | ((uint32_t)(h & 0x3ff) << 13) : 0;
  float f;
  __builtin_memcpy(&f, &b, 4);
  return f;
}

static inline void block(const uint8_t *blk, bfloat16 *y) {
  const float d = half_bits((uint16_t)(blk[0] | blk[1] << 8));
  const uint8_t *qs = blk + 2;
  const uint16_t *qh = (const uint16_t *)(blk + 34);
  aie::vector<int16, 16> s16 = aie::zeros<int16, 16>(), sg16 = aie::zeros<int16, 16>();
  for (int g = 0; g < 8; g++) {
    s16[g] = (int16)(2 * ((qh[g] >> 12) & 7) + 1);
    sg16[g] = (int16)((qh[g] & 0x8000) ? -1 : 1);
  }
  // a group's scale dl = d (2s + 1), and its shift dl * delta
  const aie::vector<float, 16> dlv = aie::mul(aie::to_float(s16, 0), aie::broadcast<float, 16>(d)).to_vector<float>();
  const aie::vector<float, 16> ddv =
      aie::mul(dlv, aie::mul(aie::to_float(sg16, 0), aie::broadcast<float, 16>(0.125f)).to_vector<float>()).to_vector<float>();
  for (int g = 0; g < 8; g++) {
    const uint32_t h = qh[g];
    const aie::vector<bfloat16, 32> v = aie::concat(
        aie::load_v<8>((const bfloat16 *)iq1s_grid_bf16[qs[4 * g] | ((h & 7) << 8)]),
        aie::load_v<8>((const bfloat16 *)iq1s_grid_bf16[qs[4 * g + 1] | (((h >> 3) & 7) << 8)]),
        aie::load_v<8>((const bfloat16 *)iq1s_grid_bf16[qs[4 * g + 2] | (((h >> 6) & 7) << 8)]),
        aie::load_v<8>((const bfloat16 *)iq1s_grid_bf16[qs[4 * g + 3] | (((h >> 9) & 7) << 8)]));
    aie::accum<accfloat, 32> acc;
    acc.from_vector(aie::broadcast<bfloat16, 32>((bfloat16)ddv[g]));
    acc = aie::mac(acc, v, aie::broadcast<bfloat16, 32>((bfloat16)dlv[g]));
    aie::store_v(y, acc.to_vector<bfloat16>());
    y += 32;
  }
}

extern "C" {
void iq1s_decode(const uint8_t *__restrict in, bfloat16 *__restrict out) {
  event0();
  for (int rep = 0; rep < REPS; rep++)
    for (int b = 0; b < NB; b++) block(in + 50 * b, out + 256 * b);
  event1();
}
}
