// 0.7.0 spike: an expert's matmul on one AIE2P core with its weights decoded there - a 64-row by 256 weight tile in
// IQ2_XXS (one block a row, 66 bytes) decoded straight into the native bfp16 matmul's operand order (8x8 sub-tiles:
// 8 rows at one 8-value K octet, raster), then C (64 x 64) = A (64 x 256, bfp16, the host's shuffle) x W^T by
// mlir-aie's matmul_vectorized_2x2_bfp16. REPS: the decode + matmul that many times (C zeroed each), its rate.
#define MATMUL_ONLY
#define DIM_M 64
#define DIM_K 256
#define DIM_N 64
#include "linalg/mm_bfp.cc"

#include "iq2xxs_tables.h"

#ifndef REPS
#define REPS 1
#endif
#ifndef DECODE_ONLY
#define DECODE_ONLY 0 /* 1: the decode alone (the matmul skipped); 2: the matmul alone (the decode skipped) */
#endif

constexpr int NR = 64, KV = 256, BB = 66;

alignas(64) static uint8_t Bm[NR * KV * 9 / 8]; // the decoded weights, sub-tile order

static inline float half_bits_div8(uint16_t h) {
  const uint32_t e = (h >> 10) & 0x1f;
  const uint32_t b = e ? ((uint32_t)(h & 0x8000) << 16) | ((e + 109) << 23) | ((uint32_t)(h & 0x3ff) << 13) : 0;
  float f;
  __builtin_memcpy(&f, &b, 4);
  return f;
}

static inline uint32_t u32_at(const uint8_t *p) {
  const uint16_t *h = (const uint16_t *)p;
  return (uint32_t)h[0] | (uint32_t)h[1] << 16;
}

// the tile's 64 rows: each row's 8 groups' index words and scales, then its sub-tiles - for an 8-row band j and a
// group g, the 8 rows' scales one 64-lane vector (8 lanes each), and for each of the group's 4 octets l the 8 rows'
// grid rows at it, signed, scaled, converted, pushed
static void decode_tile(const uint8_t *__restrict w) {
  static uint32_t a0[NR][8], a1[NR][8];
  alignas(64) static bfloat16 db[NR][8];
  for (int n = 0; n < NR; n++) {
    const uint8_t *blk = w + BB * n;
    const float d8 = half_bits_div8((uint16_t)(blk[0] | blk[1] << 8));
    aie::vector<int16, 16> s16 = aie::zeros<int16, 16>();
    for (int g = 0; g < 8; g++) {
      a0[n][g] = u32_at(blk + 2 + 8 * g), a1[n][g] = u32_at(blk + 6 + 8 * g);
      s16[g] = (int16)(2 * (a1[n][g] >> 28) + 1);
    }
    const aie::vector<bfloat16, 16> dv = aie::mul(aie::to_float(s16, 0), aie::broadcast<float, 16>(d8)).to_vector<bfloat16>();
    aie::store_v(db[n], dv.extract<8>(0));
  }
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bm);
  for (int j = 0; j < NR / 8; j++)
    for (int g = 0; g < 8; g++) {
      const bfloat16 *dj = &db[8 * j][0]; // the band's 8 rows' scales for group g: rows 8 apart in db
      const aie::vector<bfloat16, 64> sc = aie::concat(
          aie::concat(aie::broadcast<bfloat16, 8>(dj[g]), aie::broadcast<bfloat16, 8>(dj[8 + g]), aie::broadcast<bfloat16, 8>(dj[16 + g]), aie::broadcast<bfloat16, 8>(dj[24 + g])),
          aie::concat(aie::broadcast<bfloat16, 8>(dj[32 + g]), aie::broadcast<bfloat16, 8>(dj[40 + g]), aie::broadcast<bfloat16, 8>(dj[48 + g]), aie::broadcast<bfloat16, 8>(dj[56 + g])));
      for (int l = 0; l < 4; l++) {
        const uint32_t *w0 = &a0[8 * j][g], *w1 = &a1[8 * j][g]; // row r's word: w0[8 r]
        const int sh = 8 * l, ss = 7 * l;
#define ROW(r) aie::load_v<8>((const bfloat16 *)iq2xxs_grid_bf16[(w0[8 * (r)] >> sh) & 255])
#define SGN(r) ((uint32_t)ksigns_iq2xs[(w1[8 * (r)] >> ss) & 127])
        const aie::vector<bfloat16, 64> v =
            aie::concat(aie::concat(ROW(0), ROW(1), ROW(2), ROW(3)), aie::concat(ROW(4), ROW(5), ROW(6), ROW(7)));
        const uint32_t lo = SGN(0) | SGN(1) << 8 | SGN(2) << 16 | SGN(3) << 24;
        const uint32_t hi = SGN(4) | SGN(5) << 8 | SGN(6) << 16 | SGN(7) << 24;
#undef ROW
#undef SGN
        const aie::vector<int16, 64> vi = v.cast_to<int16>();
        const aie::vector<bfloat16, 64> sv =
            aie::select(vi, aie::bit_xor(vi, aie::broadcast<int16, 64>((int16)0x8000)), aie::mask<64>::from_uint32(lo, hi))
                .cast_to<bfloat16>();
        out.push(aie::mul(sv, sc).to_vector<bfp16ebs8>());
      }
    }
}

extern "C" {
void fused_iq2xxs(const uint8_t *__restrict w, uint8_t *__restrict a, uint8_t *__restrict c) {
  event0();
  for (int rep = 0; rep < REPS; rep++) {
#if DECODE_ONLY != 2
    decode_tile(w);
#endif
#if DECODE_ONLY != 1
    for (int i = 0; i < 64 * 64 * 9 / 8; i += 64) aie::store_v(c + i, aie::zeros<uint8, 64>()); // 4,608 = 72 x 64
    matmul_vectorized_2x2_bfp16<DIM_M / 8, DIM_K / 8, DIM_N / 8, 8, 8, 8>((bfp16ebs8 *)a, (bfp16ebs8 *)Bm, (bfp16ebs8 *)c);
#endif
  }
#if DECODE_ONLY == 1
  c[0] = Bm[0];
#endif
  event1();
}
}
