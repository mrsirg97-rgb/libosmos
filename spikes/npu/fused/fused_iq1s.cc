// 0.7.0 spike: an IQ1_S expert tile on one AIE2P core - 64 rows of 256 (one 50-byte block a row: d fp16, qs[32],
// qh[8]) decoded straight into the native bfp16 matmul's 8x8 sub-tile order, then C (64 x 64) = A (64 x 256) x W^T.
// the grid packed (iq1s_grid2.h: a row's 8 values as 2-bit codes, 4 KB - its 32 KB as bf16 would not fit beside A
// and the decoded tile), expanded in registers: each row's code over its 8 lanes, a lane's two bits picked by a mask,
// two compares and two selects to -1 / 0 / 1. a value: d (2s + 1) (grid + delta), delta +-0.125 (qh bit 15), the
// scale's odd factor from an 8-entry bf16 table, the scale and the shift a band-group's vectors. REPS, DECODE_ONLY
// as fused_iq2xxs.cc's.
#define MATMUL_ONLY
#define DIM_M 64
#define DIM_K 256
#define DIM_N 64
#include "linalg/mm_bfp.cc"

#include "aie_kernel_utils.h"
#include "iq1s_grid2.h"

#ifndef REPS
#define REPS 1
#endif
#ifndef DECODE_ONLY
#define DECODE_ONLY 0
#endif

constexpr int NR = 64, KV = 256, BB = 50;

alignas(64) static uint8_t Bm[NR * KV * 9 / 8];
alignas(64) static const uint16_t kodd8_bf16[8] = {0x3f80, 0x4040, 0x40a0, 0x40e0, 0x4110, 0x4130, 0x4150, 0x4170}; // 1 3 .. 15
alignas(64) static const uint16_t kdelta_bf16[2] = {0x3e00, 0xbe00};                                                 // +-0.125

static inline float half_bits(uint16_t h) {
  const uint32_t e = (h >> 10) & 0x1f;
  const uint32_t b = e ? ((uint32_t)(h & 0x8000) << 16) | ((e + 112) << 23) | ((uint32_t)(h & 0x3ff) << 13) : 0;
  float f;
  __builtin_memcpy(&f, &b, 4);
  return f;
}

// a lane's 2-bit field, and the codes 0 and 2 (-1 and +1) at it: lane 8r + j holds value j of its row
alignas(64) static const int16_t kfield[64] = {
#define F8 3, 12, 48, 192, 768, 3072, 12288, -16384
    F8, F8, F8, F8, F8, F8, F8, F8
#undef F8
};
alignas(64) static const int16_t ktwo[64] = {
#define T8 2, 8, 32, 128, 512, 2048, 8192, -32768
    T8, T8, T8, T8, T8, T8, T8, T8
#undef T8
};

static void decode_tile(const uint8_t *__restrict w) {
  alignas(64) static bfloat16 d[NR];
  for (int n = 0; n < NR; n++) d[n] = (bfloat16)half_bits((uint16_t)(w[BB * n] | w[BB * n + 1] << 8));
  const bfloat16 *ko = (const bfloat16 *)kodd8_bf16, *kd = (const bfloat16 *)kdelta_bf16;
  const aie::vector<int16, 64> field = aie::load_v<64>(kfield), two = aie::load_v<64>(ktwo);
  const aie::vector<int16, 64> zero = aie::zeros<int16, 64>();
  const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>((bfloat16)1.0f), mone = aie::broadcast<bfloat16, 64>((bfloat16)-1.0f);
  const aie::vector<bfloat16, 64> bzero = aie::zeros<bfloat16, 64>();
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bm);
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int j = 0; j < NR / 8; j++) {
    const uint8_t *band = w + BB * 8 * j;
    const aie::vector<bfloat16, 64> dv = aie::concat(
        aie::concat(aie::broadcast<bfloat16, 8>(d[8 * j]), aie::broadcast<bfloat16, 8>(d[8 * j + 1]), aie::broadcast<bfloat16, 8>(d[8 * j + 2]), aie::broadcast<bfloat16, 8>(d[8 * j + 3])),
        aie::concat(aie::broadcast<bfloat16, 8>(d[8 * j + 4]), aie::broadcast<bfloat16, 8>(d[8 * j + 5]), aie::broadcast<bfloat16, 8>(d[8 * j + 6]), aie::broadcast<bfloat16, 8>(d[8 * j + 7])));
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (int g = 0; g < 8; g++) {
      uint32_t h[8];
      AIE_LOOP_UNROLL_FULL
      for (int r = 0; r < 8; r++) h[r] = ((const uint16_t *)(band + BB * r + 34))[g];
#define K(r) aie::broadcast<bfloat16, 8>(ko[(h[r] >> 12) & 7])
#define D(r) aie::broadcast<bfloat16, 8>(kd[h[r] >> 15])
      const aie::vector<bfloat16, 64> kv = aie::concat(aie::concat(K(0), K(1), K(2), K(3)), aie::concat(K(4), K(5), K(6), K(7)));
      const aie::vector<bfloat16, 64> delta = aie::concat(aie::concat(D(0), D(1), D(2), D(3)), aie::concat(D(4), D(5), D(6), D(7)));
#undef K
#undef D
      const aie::vector<bfloat16, 64> sc = aie::mul(dv, kv).to_vector<bfloat16>();
      AIE_LOOP_UNROLL_FULL
      for (int l = 0; l < 4; l++) {
#define C(r) aie::broadcast<int16, 8>((int16)iq1s_grid2[band[BB * (r) + 2 + 4 * g + l] | ((h[r] >> (3 * l)) & 7) << 8])
        const aie::vector<int16, 64> c = aie::concat(aie::concat(C(0), C(1), C(2), C(3)), aie::concat(C(4), C(5), C(6), C(7)));
#undef C
        const aie::vector<int16, 64> t = aie::bit_and(c, field);
        const aie::vector<bfloat16, 64> gv = aie::select(aie::select(bzero, one, aie::eq(t, two)), mone, aie::eq(t, zero));
        const aie::vector<bfloat16, 64> gd = aie::add(gv, delta);
        out.push(aie::mul(gd, sc).to_vector<bfp16ebs8>());
      }
    }
  }
}

extern "C" {
void fused_iq1s(const uint8_t *__restrict w, uint8_t *__restrict a, uint8_t *__restrict c) {
  event0();
  for (int rep = 0; rep < REPS; rep++) {
#if DECODE_ONLY != 2
    decode_tile(w);
#endif
#if DECODE_ONLY != 1
    for (int i = 0; i < 64 * 64 * 9 / 8; i += 64) aie::store_v(c + i, aie::zeros<uint8, 64>());
    matmul_vectorized_2x2_bfp16<DIM_M / 8, DIM_K / 8, DIM_N / 8, 8, 8, 8>((bfp16ebs8 *)a, (bfp16ebs8 *)Bm, (bfp16ebs8 *)c);
#endif
  }
#if DECODE_ONLY == 1
  c[0] = Bm[0];
#endif
  event1();
}
}
