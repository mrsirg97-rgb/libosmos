// 0.7.0 spike: Qwen UD-IQ3_XXS's expert formats on the whole array (fused32.py, FMT=iq4nl | iq3xxs | iq2s) - a core's
// NR-row W tile, a 256 K-step of it, decoded straight into the native bfp16 matmul's 8x8 sub-tile order as two K
// halves ([kh][j][o]: band j's 8 rows at octet o, 8 values each), then C (2 halves of 64 x NR, f32) += A x W^T a
// quarter at a time (common/mm_f32.h). one format a build: QFMT 0 IQ4_NL, 1 IQ3_XXS, 2 IQ2_S.
#define MATMUL_ONLY
#define DIM_M 64
#define DIM_K 256
#define DIM_N 64
#include "linalg/mm_bfp.cc"

#include "aie_kernel_utils.h"
#include "../common/fp16.h"
#include "../common/c_rows.h"
#ifndef SPLIT
#define SPLIT 0 /* timing only (fused_ds.py SPLIT=): 1 the decode once a pass, 2 no matmul */
#endif
#include "../common/mm_f32.h"

#ifndef QFMT
#define QFMT 0
#endif
#ifndef NROWS
#define NROWS 32
#endif
constexpr int NR = NROWS;

#ifndef ABL
#define ABL 0 /* ablations, timing only: 1 no qh, 2 one sign byte, 4 no group scales */
#endif
#if QFMT == 0
#include "iq4nl_lut.h"
constexpr int BLK = 8 * 18; // a K-step's bytes a row: 8 blocks of 32
#elif QFMT == 1
#include "iq3xxs_tables.h"
constexpr int BLK = 98;
#elif QFMT == 2
#include "iq2s_tables.h"
constexpr int BLK = 82;
#else
#include "iq2s_grid2.h" // QFMT 3 / 4: IQ2_S with its grid packed (2 KB)
constexpr int BLK = 82;
#endif
#ifdef ROWSTRIDE /* a W tile row's bytes (fused_ds.py): a K-step pair 2 BLK, or a row whole (WHOLEK) */
constexpr int RB = ROWSTRIDE;
#else
constexpr int RB = BLK;
#endif

alignas(64) static uint8_t Bm[NR * 256 * 9 / 8];

static inline uint32_t u32_at(const uint8_t *p) {
  const uint16_t *h = (const uint16_t *)p;
  return (uint32_t)h[0] | (uint32_t)h[1] << 16;
}

#define BCAST8(f) aie::concat(aie::concat(f(0), f(1), f(2), f(3)), aie::concat(f(4), f(5), f(6), f(7)))

#if QFMT == 0
// IQ4_NL: block b (of 8) of a K-step holds octets 4b .. 4b + 3: its 16 bytes' low nibbles (values 0-15) then high
// (16-31); a row's two fetches give both, an octet one 8-lane slice of them
static void decode_tile(const uint8_t *__restrict w) {
  using lut_t = aie::lut<4, bfloat16, bfloat16>;
  const lut_t llo(256, lut_lo_ab, lut_lo_cd), lhi(256, lut_hi_ab, lut_hi_cd);
  aie::parallel_lookup<uint8, lut_t, aie::lut_oor_policy::truncate> plo(llo, 0), phi(lhi, 0);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bm);
  for (int kh = 0; kh < 2; kh++)
    for (int j = 0; j < NR / 8; j++)
      for (int b = 4 * kh; b < 4 * kh + 4; b++) {
        aie::vector<bfloat16, 32> v[8];
        bfloat16 d[8];
        AIE_LOOP_UNROLL_FULL
        for (int r = 0; r < 8; r++) {
          const uint8_t *blk = w + RB * (8 * j + r) + 18 * b;
          d[r] = fp16_to_bf16_shift((uint16_t)(blk[0] | blk[1] << 8), 0);
          const aie::vector<uint8, 16> qs = aie::load_unaligned_v<16>(blk + 2);
          v[r] = aie::concat(plo.fetch(qs), phi.fetch(qs));
        }
#define D(r) aie::broadcast<bfloat16, 8>(d[r])
        const aie::vector<bfloat16, 64> sc = BCAST8(D);
#undef D
        AIE_LOOP_UNROLL_FULL
        for (int l = 0; l < 4; l++) {
#define P(r) v[r].extract<8>(l)
          out.push(aie::mul(BCAST8(P), sc).to_vector<bfp16ebs8>());
#undef P
        }
      }
}
#elif QFMT == 1
// IQ3_XXS: octet o = 4 g + l of a row: its grid indices qs[8 g + 2 l], qs[8 g + 2 l + 1] (iq3_lo, iq3_hi or'd), its
// signs ksigns[(aux_g >> 7 l) & 127], its scale d / 4 (2 s + 1), s = aux_g >> 28
alignas(64) static const uint16_t kodd_bf16[16] = {0x3f80, 0x4040, 0x40a0, 0x40e0, 0x4110, 0x4130, 0x4150, 0x4170,
                                                   0x4188, 0x4198, 0x41a8, 0x41b8, 0x41c8, 0x41d8, 0x41e8, 0x41f8};
static void decode_tile(const uint8_t *__restrict w) {
  const bfloat16 *kb = (const bfloat16 *)kodd_bf16;
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bm);
  for (int kh = 0; kh < 2; kh++)
    for (int j = 0; j < NR / 8; j++) {
      const uint8_t *band = w + RB * 8 * j;
#define D(r) aie::broadcast<bfloat16, 8>(fp16_to_bf16_shift((uint16_t)(band[RB * (r)] | band[RB * (r) + 1] << 8), 2))
      const aie::vector<bfloat16, 64> dv = BCAST8(D);
#undef D
      for (int g = 4 * kh; g < 4 * kh + 4; g++) {
        uint32_t aux[8];
        AIE_LOOP_UNROLL_FULL
        for (int r = 0; r < 8; r++) aux[r] = u32_at(band + RB * r + 66 + 4 * g);
#define K(r) aie::broadcast<bfloat16, 8>(kb[aux[r] >> 28])
        const aie::vector<bfloat16, 64> sc = aie::mul(dv, BCAST8(K)).to_vector<bfloat16>();
#undef K
        AIE_LOOP_UNROLL_FULL
        for (int l = 0; l < 4; l++) {
          const uint8_t *q = band + 2 + 8 * g + 2 * l;
#define LO(r) aie::load_v<8>((const int16 *)iq3_lo[q[RB * (r)]])
#define HI(r) aie::load_v<8>((const int16 *)iq3_hi[q[RB * (r) + 1]])
          const aie::vector<int16, 64> vi = aie::bit_or(BCAST8(LO), BCAST8(HI));
#undef LO
#undef HI
#define S(r) ((uint32_t)ksigns_iq2xs[(aux[r] >> (7 * l)) & 127])
          const uint32_t lo = S(0) | S(1) << 8 | S(2) << 16 | S(3) << 24, hi = S(4) | S(5) << 8 | S(6) << 16 | S(7) << 24;
#undef S
          const aie::vector<bfloat16, 64> sv =
              aie::select(vi, aie::bit_xor(vi, aie::broadcast<int16, 64>((int16)0x8000)), aie::mask<64>::from_uint32(lo, hi)).cast_to<bfloat16>();
          out.push(aie::mul(sv, sc).to_vector<bfp16ebs8>());
        }
      }
    }
}
#elif QFMT == 3
// IQ2_S, its grid packed: a row's 16-bit code over its 8 lanes, a lane's 2-bit field masked, 8 / 25 / 43 (bf16 bits)
// by two compares and two selects; the rest as QFMT 2's
alignas(64) static const uint16_t kodd_bf16[16] = {0x3f80, 0x4040, 0x40a0, 0x40e0, 0x4110, 0x4130, 0x4150, 0x4170,
                                                   0x4188, 0x4198, 0x41a8, 0x41b8, 0x41c8, 0x41d8, 0x41e8, 0x41f8};
alignas(64) static const int16_t kfield[64] = {
#define F8 3, 12, 48, 192, 768, 3072, 12288, -16384
    F8, F8, F8, F8, F8, F8, F8, F8
#undef F8
};
alignas(64) static const int16_t kone[64] = {
#define O8 1, 4, 16, 64, 256, 1024, 4096, 16384
    O8, O8, O8, O8, O8, O8, O8, O8
#undef O8
};
static void decode_tile(const uint8_t *__restrict w) {
  const bfloat16 *kb = (const bfloat16 *)kodd_bf16;
  const aie::vector<int16, 64> field = aie::load_v<64>(kfield), one = aie::load_v<64>(kone), zero = aie::zeros<int16, 64>();
  const aie::vector<int16, 64> v8 = aie::broadcast<int16, 64>((int16)0x4100), v25 = aie::broadcast<int16, 64>((int16)0x41c8),
                               v43 = aie::broadcast<int16, 64>((int16)0x422c); // 8, 25, 43 as bf16 bits
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bm);
  for (int kh = 0; kh < 2; kh++)
    for (int j = 0; j < NR / 8; j++) {
      const uint8_t *band = w + RB * 8 * j;
#define D(r) aie::broadcast<bfloat16, 8>(fp16_to_bf16_shift((uint16_t)(band[RB * (r)] | band[RB * (r) + 1] << 8), 3))
      const aie::vector<bfloat16, 64> dv = BCAST8(D);
#undef D
      for (int g = 4 * kh; g < 4 * kh + 4; g++) {
#define KL(r) aie::broadcast<bfloat16, 8>(kb[band[RB * (r) + 74 + g] & 15])
#define KH(r) aie::broadcast<bfloat16, 8>(kb[band[RB * (r) + 74 + g] >> 4])
#if ABL & 4
        const aie::vector<bfloat16, 64> scl = dv, sch = dv;
#else
        const aie::vector<bfloat16, 64> scl = aie::mul(dv, BCAST8(KL)).to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> sch = aie::mul(dv, BCAST8(KH)).to_vector<bfloat16>();
#endif
#undef KL
#undef KH
        AIE_LOOP_UNROLL_FULL
        for (int l = 0; l < 4; l++) {
#if ABL & 1
#define C(r) aie::broadcast<int16, 8>((int16)iq2s_grid2[band[RB * (r) + 2 + 4 * g + l]])
#else
#define C(r) aie::broadcast<int16, 8>((int16)iq2s_grid2[band[RB * (r) + 2 + 4 * g + l] | ((band[RB * (r) + 66 + g] >> (2 * l)) & 3) << 8])
#endif
          const aie::vector<int16, 64> t = aie::bit_and(BCAST8(C), field);
#undef C
          const aie::vector<int16, 64> vi = aie::select(aie::select(v43, v25, aie::eq(t, one)), v8, aie::eq(t, zero));
#define S(r) ((uint32_t)band[RB * (r) + 34 + 4 * g + l])
#if ABL & 2
          const uint32_t lo = S(0) * 0x01010101u, hi = lo;
#else
          const uint32_t lo = S(0) | S(1) << 8 | S(2) << 16 | S(3) << 24, hi = S(4) | S(5) << 8 | S(6) << 16 | S(7) << 24;
#endif
#undef S
          const aie::vector<bfloat16, 64> sv =
              aie::select(vi, aie::bit_xor(vi, aie::broadcast<int16, 64>((int16)0x8000)), aie::mask<64>::from_uint32(lo, hi)).cast_to<bfloat16>();
          out.push(aie::mul(sv, l < 2 ? scl : sch).to_vector<bfp16ebs8>());
        }
      }
    }
}
#elif QFMT == 4
// IQ2_S, its grid packed, its index and sign path in vectors (as IQ2_XXS's v5/v6): a group's 8 rows' 4 index bytes,
// high bits and 4 sign bytes read as words, then the 32 indices [l][r] (lane 8 l + r) and the 4 octets' 64-bit sign
// masks by shifts and selects at 32 lanes; the octet loop then one load a row and one mask
alignas(64) static const uint16_t kodd_bf16[16] = {0x3f80, 0x4040, 0x40a0, 0x40e0, 0x4110, 0x4130, 0x4150, 0x4170,
                                                   0x4188, 0x4198, 0x41a8, 0x41b8, 0x41c8, 0x41d8, 0x41e8, 0x41f8};
alignas(64) static const int16_t kfield[64] = {
#define F8 3, 12, 48, 192, 768, 3072, 12288, -16384
    F8, F8, F8, F8, F8, F8, F8, F8
#undef F8
};
alignas(64) static const int16_t kone[64] = {
#define O8 1, 4, 16, 64, 256, 1024, 4096, 16384
    O8, O8, O8, O8, O8, O8, O8, O8
#undef O8
};
alignas(64) static const int32_t krow32[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                               2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
static inline uint32_t u32_at2(const uint8_t *p) { const uint16_t *h = (const uint16_t *)p; return (uint32_t)h[0] | (uint32_t)h[1] << 16; }
static void decode_tile(const uint8_t *__restrict w) {
  alignas(64) static int32_t ix[32];
  alignas(64) static uint32_t swm[8];
  alignas(64) static int32_t wq[32];
  const bfloat16 *kb = (const bfloat16 *)kodd_bf16;
  const aie::vector<int16, 64> field = aie::load_v<64>(kfield), one = aie::load_v<64>(kone), zero = aie::zeros<int16, 64>();
  const aie::vector<int16, 64> v8 = aie::broadcast<int16, 64>((int16)0x4100), v25 = aie::broadcast<int16, 64>((int16)0x41c8),
                               v43 = aie::broadcast<int16, 64>((int16)0x422c);
  const aie::vector<int32, 32> rowv = aie::load_v<32>(krow32);
#define m1 aie::eq(rowv, aie::broadcast<int32, 32>(1))
#define m2 aie::eq(rowv, aie::broadcast<int32, 32>(2))
#define m3 aie::eq(rowv, aie::broadcast<int32, 32>(3))
  const aie::vector<int32, 32> b255 = aie::broadcast<int32, 32>(255), b3 = aie::broadcast<int32, 32>(0x300);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bm);
  for (int kh = 0; kh < 2; kh++)
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int j = 0; j < NR / 8; j++) {
      const uint8_t *band = w + RB * 8 * j;
#define D(r) aie::broadcast<bfloat16, 8>(fp16_to_bf16_shift((uint16_t)(band[RB * (r)] | band[RB * (r) + 1] << 8), 3))
      const aie::vector<bfloat16, 64> dv = BCAST8(D);
#undef D
      AIE_LOOP_MIN_ITERATION_COUNT(4)
      for (int g = 4 * kh; g < 4 * kh + 4; g++) {
#define KL(r) aie::broadcast<bfloat16, 8>(kb[band[RB * (r) + 74 + g] & 15])
#define KH(r) aie::broadcast<bfloat16, 8>(kb[band[RB * (r) + 74 + g] >> 4])
        const aie::vector<bfloat16, 64> scl = aie::mul(dv, BCAST8(KL)).to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> sch = aie::mul(dv, BCAST8(KH)).to_vector<bfloat16>();
#undef KL
#undef KH
        AIE_LOOP_UNROLL_FULL
        for (int r = 0; r < 8; r++) {
          wq[r] = (int32)u32_at2(band + RB * r + 2 + 4 * g);
          wq[8 + r] = (int32)u32_at2(band + RB * r + 34 + 4 * g);
          wq[16 + r] = (int32)band[RB * r + 66 + g] << 8;
        }
        const aie::vector<int32, 8> q8 = aie::load_v<8>(wq), s8 = aie::load_v<8>(wq + 8), h8 = aie::load_v<8>(wq + 16);
        // lane 8 l + r: row r's byte l, its two high bits l
        const aie::vector<int32, 32> q = aie::concat(q8, q8, q8, q8), h = aie::concat(h8, h8, h8, h8), sg = aie::concat(s8, s8, s8, s8);
        aie::vector<int32, 32> qb = aie::select(q, aie::downshift(q, 8), m1);
        qb = aie::select(qb, aie::downshift(q, 16), m2);
        qb = aie::select(qb, aie::downshift(q, 24), m3);
        aie::vector<int32, 32> hb = aie::select(h, aie::downshift(h, 2), m1);
        hb = aie::select(hb, aie::downshift(h, 4), m2);
        hb = aie::select(hb, aie::downshift(h, 6), m3);
                aie::store_v(ix, aie::bit_or(aie::bit_and(qb, b255), aie::bit_and(hb, b3)));
        const volatile int32_t *ixv = ix; // read back from memory: Peano's lane-to-scalar copy of a forwarded store fails
        aie::vector<int32, 32> sb = aie::select(sg, aie::downshift(sg, 8), m1);
        sb = aie::select(sb, aie::downshift(sg, 16), m2);
        sb = aie::select(sb, aie::downshift(sg, 24), m3);
        sb = aie::bit_and(sb, b255);
        aie::store_v(swm, aie::filter_even(aie::filter_even(sb.cast_to<int16>(), 1).cast_to<uint8>(), 1).cast_to<uint32>());
        AIE_LOOP_UNROLL_FULL
        for (int l = 0; l < 4; l++) {
#define C(r) aie::broadcast<int16, 8>((int16)iq2s_grid2[ixv[8 * l + (r)]])
          const aie::vector<int16, 64> t = aie::bit_and(BCAST8(C), field);
#undef C
          const aie::vector<int16, 64> vi = aie::select(aie::select(v43, v25, aie::eq(t, one)), v8, aie::eq(t, zero));
          const aie::vector<bfloat16, 64> sv =
              aie::select(vi, aie::bit_xor(vi, aie::broadcast<int16, 64>((int16)0x8000)), aie::mask<64>::from_uint32(swm[2 * l], swm[2 * l + 1])).cast_to<bfloat16>();
          out.push(aie::mul(sv, l < 2 ? scl : sch).to_vector<bfp16ebs8>());
        }
      }
    }
#undef m1
#undef m2
#undef m3
}
#else
// IQ2_S: octet o = 4 g + l of a row: index qs[4 g + l] | ((qh[g] >> 2 l) & 3) << 8, sign byte signs[4 g + l], scale
// d / 8 (2 s + 1), s scales[g]'s low nibble (l < 2) or high
alignas(64) static const uint16_t kodd_bf16[16] = {0x3f80, 0x4040, 0x40a0, 0x40e0, 0x4110, 0x4130, 0x4150, 0x4170,
                                                   0x4188, 0x4198, 0x41a8, 0x41b8, 0x41c8, 0x41d8, 0x41e8, 0x41f8};
static void decode_tile(const uint8_t *__restrict w) {
  const bfloat16 *kb = (const bfloat16 *)kodd_bf16;
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bm);
  for (int kh = 0; kh < 2; kh++)
    for (int j = 0; j < NR / 8; j++) {
      const uint8_t *band = w + RB * 8 * j;
#define D(r) aie::broadcast<bfloat16, 8>(fp16_to_bf16_shift((uint16_t)(band[RB * (r)] | band[RB * (r) + 1] << 8), 3))
      const aie::vector<bfloat16, 64> dv = BCAST8(D);
#undef D
      for (int g = 4 * kh; g < 4 * kh + 4; g++) {
#define KL(r) aie::broadcast<bfloat16, 8>(kb[band[RB * (r) + 74 + g] & 15])
#define KH(r) aie::broadcast<bfloat16, 8>(kb[band[RB * (r) + 74 + g] >> 4])
        const aie::vector<bfloat16, 64> scl = aie::mul(dv, BCAST8(KL)).to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> sch = aie::mul(dv, BCAST8(KH)).to_vector<bfloat16>();
#undef KL
#undef KH
        AIE_LOOP_UNROLL_FULL
        for (int l = 0; l < 4; l++) {
#define ROW(r) aie::load_v<8>((const bfloat16 *)iq2s_grid_bf16[band[RB * (r) + 2 + 4 * g + l] | ((band[RB * (r) + 66 + g] >> (2 * l)) & 3) << 8])
          const aie::vector<int16, 64> vi = BCAST8(ROW).cast_to<int16>();
#undef ROW
#define S(r) ((uint32_t)band[RB * (r) + 34 + 4 * g + l])
          const uint32_t lo = S(0) | S(1) << 8 | S(2) << 16 | S(3) << 24, hi = S(4) | S(5) << 8 | S(6) << 16 | S(7) << 24;
#undef S
          const aie::vector<bfloat16, 64> sv =
              aie::select(vi, aie::bit_xor(vi, aie::broadcast<int16, 64>((int16)0x8000)), aie::mask<64>::from_uint32(lo, hi)).cast_to<bfloat16>();
          out.push(aie::mul(sv, l < 2 ? scl : sch).to_vector<bfp16ebs8>());
        }
      }
    }
}
#endif

extern "C" void fused_step_f32(const uint8_t *__restrict w, uint8_t *__restrict a, float *__restrict c, int32_t kstep,
                               int32_t piece) {
  const int half = piece >> 1, kh = piece & 1;
  float *ch = c + half * (64 * NR);
  if (kstep == 0 && kh == 0)
    for (int i = 0; i < 64 * NR; i += 16) aie::store_v(ch + i, aie::zeros<float, 16>());
#if SPLIT == 1
  if (kstep != 0) goto mm; /* timing only: the decode once */
#endif
#ifdef WHOLEK
  if (piece == 0) decode_tile(w + BLK * kstep); // the rows whole (fused_ds.py KIND=down): K-step s their block s
#else
  if (piece == 0) decode_tile(w + (RB == 2 * BLK ? BLK * (kstep & 1) : 0)); // a pair tile: odd K-steps its second
#endif
#if SPLIT == 1
mm:
#endif
#if SPLIT != 2
  mm_f32acc<64 / 8, 128 / 8, NR / 8>((bfp16ebs8 *)a, (bfp16ebs8 *)(Bm + kh * (NR * 128 * 9 / 8)), ch);
#endif
#ifdef ROWMAJOR
  if (kstep == ROWMAJOR - 1 && piece == 3) // the step's last piece (ROWMAJOR: the K-steps): both halves' sums done,
    for (int h = 0; h < 2; h++) c_rows<NR>(c + h * (64 * NR), (float *)Bm); // and Bm free - to rows
#endif
}
