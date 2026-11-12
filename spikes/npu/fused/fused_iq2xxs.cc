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
#include "aie_kernel_utils.h"

#ifndef FUSED_V
#define FUSED_V 2 /* 1: row-major index words, nested loops; 2: group-major (a band's rows contiguous), one counted loop */
#endif

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

#if FUSED_V == 2
static void decode_tile2(const uint8_t *__restrict w) {
  alignas(64) static uint32_t a0t[8][NR], a1t[8][NR];
  alignas(64) static bfloat16 dbt[8][NR];
  for (int n = 0; n < NR; n++) {
    const uint8_t *blk = w + BB * n;
    const float d8 = half_bits_div8((uint16_t)(blk[0] | blk[1] << 8));
    aie::vector<int16, 16> s16 = aie::zeros<int16, 16>();
    for (int g = 0; g < 8; g++) {
      const uint32_t x0 = u32_at(blk + 2 + 8 * g), x1 = u32_at(blk + 6 + 8 * g);
      a0t[g][n] = x0, a1t[g][n] = x1;
      s16[g] = (int16)(2 * (x1 >> 28) + 1);
    }
    const aie::vector<bfloat16, 16> dv = aie::mul(aie::to_float(s16, 0), aie::broadcast<float, 16>(d8)).to_vector<bfloat16>();
    for (int g = 0; g < 8; g++) dbt[g][n] = dv[g];
  }
#ifdef SETUP_ONLY
  Bm[0] = (uint8_t)a0t[3][5] ^ (uint8_t)(float)dbt[2][7];
  return;
#endif
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bm);
  AIE_LOOP_MIN_ITERATION_COUNT(64)
  for (int jg = 0; jg < (NR / 8) * 8; jg++) {
    const int j = jg >> 3, g = jg & 7;
    const uint32_t *w0 = &a0t[g][8 * j], *w1 = &a1t[g][8 * j];
    const bfloat16 *d = &dbt[g][8 * j];
    const aie::vector<bfloat16, 64> sc = aie::concat(
        aie::concat(aie::broadcast<bfloat16, 8>(d[0]), aie::broadcast<bfloat16, 8>(d[1]), aie::broadcast<bfloat16, 8>(d[2]), aie::broadcast<bfloat16, 8>(d[3])),
        aie::concat(aie::broadcast<bfloat16, 8>(d[4]), aie::broadcast<bfloat16, 8>(d[5]), aie::broadcast<bfloat16, 8>(d[6]), aie::broadcast<bfloat16, 8>(d[7])));
    AIE_LOOP_UNROLL_FULL
    for (int l = 0; l < 4; l++) {
      const int sh = 8 * l, ss = 7 * l;
#define ROW(r) aie::load_v<8>((const bfloat16 *)iq2xxs_grid_bf16[(w0[r] >> sh) & 255])
#define SGN(r) ((uint32_t)ksigns_iq2xs[(w1[r] >> ss) & 127])
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
#endif

#if FUSED_V == 3
// the odd factor 2s + 1 (1 .. 31) of a group's scale, exact in bf16
alignas(64) static const uint16_t kodd_bf16[16] = {0x3f80, 0x4040, 0x40a0, 0x40e0, 0x4110, 0x4130, 0x4150, 0x4170,
                                                   0x4188, 0x4198, 0x41a8, 0x41b8, 0x41c8, 0x41d8, 0x41e8, 0x41f8};

static void decode_tile3(const uint8_t *__restrict w) {
  alignas(64) static bfloat16 d8[NR]; // a row's d / 8: its one setup
  for (int n = 0; n < NR; n++) d8[n] = (bfloat16)half_bits_div8((uint16_t)(w[BB * n] | w[BB * n + 1] << 8));
  const bfloat16 *kb = (const bfloat16 *)kodd_bf16;
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bm);
  AIE_LOOP_MIN_ITERATION_COUNT(8)
  for (int j = 0; j < NR / 8; j++) {
    const uint8_t *band = w + BB * 8 * j;
    const aie::vector<bfloat16, 64> dv = aie::concat(
        aie::concat(aie::broadcast<bfloat16, 8>(d8[8 * j]), aie::broadcast<bfloat16, 8>(d8[8 * j + 1]), aie::broadcast<bfloat16, 8>(d8[8 * j + 2]), aie::broadcast<bfloat16, 8>(d8[8 * j + 3])),
        aie::concat(aie::broadcast<bfloat16, 8>(d8[8 * j + 4]), aie::broadcast<bfloat16, 8>(d8[8 * j + 5]), aie::broadcast<bfloat16, 8>(d8[8 * j + 6]), aie::broadcast<bfloat16, 8>(d8[8 * j + 7])));
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (int g = 0; g < 8; g++) {
      uint32_t w0[8], w1[8];
      AIE_LOOP_UNROLL_FULL
      for (int r = 0; r < 8; r++) w0[r] = u32_at(band + BB * r + 2 + 8 * g), w1[r] = u32_at(band + BB * r + 6 + 8 * g);
#define K(r) aie::broadcast<bfloat16, 8>(kb[w1[r] >> 28])
      const aie::vector<bfloat16, 64> kv = aie::concat(aie::concat(K(0), K(1), K(2), K(3)), aie::concat(K(4), K(5), K(6), K(7)));
#undef K
      const aie::vector<bfloat16, 64> sc = aie::mul(dv, kv).to_vector<bfloat16>();
      AIE_LOOP_UNROLL_FULL
      for (int l = 0; l < 4; l++) {
        const int sh = 8 * l, ss = 7 * l;
#define ROW(r) aie::load_v<8>((const bfloat16 *)iq2xxs_grid_bf16[(w0[r] >> sh) & 255])
#define SGN(r) ((uint32_t)ksigns_iq2xs[(w1[r] >> ss) & 127])
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
}
#endif

extern "C" {
void fused_iq2xxs(const uint8_t *__restrict w, uint8_t *__restrict a, uint8_t *__restrict c) {
  event0();
  for (int rep = 0; rep < REPS; rep++) {
#if DECODE_ONLY != 2
#if FUSED_V == 3
    decode_tile3(w);
#elif FUSED_V == 2
    decode_tile2(w);
#else
    decode_tile(w);
#endif
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
