// 0.7.0 spike: ggml's IQ2_S decoded on one AIE2P core - NB blocks of 256 values (82 bytes: d fp16, qs[32] the grid
// indices' low bytes, signs[32] (an octet's byte), qh[8] (2 high index bits an octet), scales[8] (4 bits a 16)) to
// bf16, ggml's dequantize_row_iq2_s: y = d (0.5 + s) / 4 * grid[j] * sign. the 1,024-row grid as bf16 (16 KB), an
// octet one 16-byte row load; a group's 4 sign bytes one 32-bit word - its 32-lane mask; its two 16-value scales one
// vector. no scalar float. REPS as q2k_decode's.
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "iq2s_tables.h"
#include "../common/fp16.h"

#ifndef NB
#define NB 16
#endif
#ifndef REPS
#define REPS 1
#endif

static inline uint32_t u32_at(const uint8_t *p) {
  const uint16_t *h = (const uint16_t *)p;
  return (uint32_t)h[0] | (uint32_t)h[1] << 16;
}

static inline void block(const uint8_t *blk, bfloat16 *y) {
  const float d8 = fp16_to_f32_shift((uint16_t)(blk[0] | blk[1] << 8), 3); // d / 8: dl = d / 8 (2s + 1)
  const uint8_t *qs = blk + 2, *sg = blk + 34, *qh = blk + 66, *sc = blk + 74;
  aie::vector<int16, 16> s16 = aie::zeros<int16, 16>();
  for (int k = 0; k < 8; k++) s16[2 * k] = (int16)(2 * (sc[k] & 15) + 1), s16[2 * k + 1] = (int16)(2 * (sc[k] >> 4) + 1);
  const aie::vector<float, 16> dlv = aie::mul(aie::to_float(s16, 0), aie::broadcast<float, 16>(d8)).to_vector<float>();
  for (int g = 0; g < 8; g++) {
    const uint8_t *q = qs + 4 * g;
    const uint32_t h = qh[g];
#define ROW(l) aie::load_v<8>((const bfloat16 *)iq2s_grid_bf16[q[l] | ((h >> (2 * (l))) & 3) << 8])
    const aie::vector<int16, 32> vi = aie::concat(ROW(0), ROW(1), ROW(2), ROW(3)).cast_to<int16>();
#undef ROW
    const aie::vector<bfloat16, 32> sv =
        aie::select(vi, aie::bit_xor(vi, aie::broadcast<int16, 32>((int16)0x8000)), aie::mask<32>::from_uint32(u32_at(sg + 4 * g)))
            .cast_to<bfloat16>();
    const aie::vector<bfloat16, 32> dl = aie::concat(aie::broadcast<bfloat16, 16>((bfloat16)dlv[2 * g]),
                                                     aie::broadcast<bfloat16, 16>((bfloat16)dlv[2 * g + 1]));
    aie::store_v(y, aie::mul(sv, dl).to_vector<bfloat16>());
    y += 32;
  }
}

extern "C" {
void iq2s_decode(const uint8_t *__restrict in, bfloat16 *__restrict out) {
  event0();
  for (int rep = 0; rep < REPS; rep++)
    for (int b = 0; b < NB; b++) block(in + 82 * b, out + 256 * b);
  event1();
}
}
