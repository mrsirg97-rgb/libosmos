// 0.7.0 spike: ggml's IQ4_NL decoded on one AIE2P core - NB blocks of 32 values (18 bytes: d fp16, qs[16]: value j
// the low nibble of qs[j], value j + 16 its high) to bf16: y = d * kvalues_iq4nl[q]. the 16-entry table through the
// core's parallel lookup, indexed by the raw byte twice (iq4nl_lut.h: a byte to its low / high nibble's value) - no
// bit operations, no scalar work a value. REPS as q2k_decode's.
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../common/fp16.h"

#include "iq4nl_lut.h"

#ifndef NB
#define NB 32
#endif
#ifndef REPS
#define REPS 1
#endif

static inline float half_bits(uint16_t h) { return fp16_to_f32_shift(h, 0); } // subnormals too (common/fp16.h)

extern "C" {
void iq4nl_decode(const uint8_t *__restrict in, bfloat16 *__restrict out) {
  event0();
  using lut_t = aie::lut<4, bfloat16, bfloat16>;
  const lut_t llo(256, lut_lo_ab, lut_lo_cd), lhi(256, lut_hi_ab, lut_hi_cd);
  aie::parallel_lookup<uint8, lut_t, aie::lut_oor_policy::truncate> plo(llo, 0), phi(lhi, 0);
  for (int rep = 0; rep < REPS; rep++)
    for (int b = 0; b < NB; b++) {
      const uint8_t *blk = in + 18 * b;
      const bfloat16 d = (bfloat16)half_bits((uint16_t)(blk[0] | blk[1] << 8));
      const aie::vector<uint8, 16> qs = aie::load_unaligned_v<16>(blk + 2);
      const aie::vector<bfloat16, 32> v = aie::concat(plo.fetch(qs), phi.fetch(qs));
      aie::store_v(out + 32 * b, aie::mul(v, aie::broadcast<bfloat16, 32>(d)).to_vector<bfloat16>());
    }
  event1();
}
}
