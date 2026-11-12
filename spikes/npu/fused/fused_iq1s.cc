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
#if IQ1_V == 3
#include "iq1s_grid8.h"
#endif

#ifndef REPS
#define REPS 1
#endif
#ifndef DECODE_ONLY
#define DECODE_ONLY 0
#endif
#ifndef IQ1_V
#define IQ1_V 2 /* 1: an octet's grid values + delta, x the scale; 2: the three values a lane can take, a band-group's
                   vectors (-1 + delta, delta, 1 + delta) x the scale, an octet choosing among them */
#endif

#ifndef NROWS
#define NROWS 64
#endif
#ifndef KHALF
#define KHALF 0 /* 1: Bm as two K halves ([kh][j][i]), each a contiguous NR x 128 operand */
#endif
constexpr int NR = NROWS, KV = 256, BB = 50;

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

#if IQ1_V >= 4
// a stand-in for an NPU layout of the experts it takes: each row's 32 grid indices pre-assembled as uint16 (the GGUF
// splits them over qs's byte and qh's 3 bits) - made once a call here, outside the timed repeats
alignas(64) static uint16_t idx16[NR][32];
static void make_idx(const uint8_t *__restrict w) {
  for (int n = 0; n < NR; n++) {
    const uint8_t *blk = w + BB * n;
    const uint16_t *qh = (const uint16_t *)(blk + 34);
    for (int k = 0; k < 32; k++) idx16[n][k] = blk[2 + k] | ((qh[k >> 2] >> (3 * (k & 3))) & 7) << 8;
  }
}
#endif

#if IQ1_V == 5
// the same indices assembled by vector ops a row (its 32 qs bytes one load; qh's 8 words each over 4 lanes; the
// 3 bits at shift 0, 3, 6, 9 by lane, merged by selects), inside each decode - nothing outside the timed repeats
alignas(64) static const int16_t klane4[32] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                                               0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
alignas(64) static const int16_t krow4[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                              2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
static void make_idx_vec(const uint8_t *__restrict w) {
  const aie::vector<int16, 32> row = aie::load_v<32>(krow4), seven = aie::broadcast<int16, 32>(7);
  const aie::mask<32> r1 = aie::eq(row, aie::broadcast<int16, 32>(1)), r2 = aie::eq(row, aie::broadcast<int16, 32>(2)),
                      r3 = aie::eq(row, aie::broadcast<int16, 32>(3));
  for (int n = 0; n < NR; n++) {
    const uint8_t *blk = w + BB * n;
    const aie::vector<int16, 8> q8 = aie::load_unaligned_v<8>((const int16 *)(blk + 34)); // qh's 8 words
    const aie::vector<int16, 32> qv = aie::concat(q8, q8, q8, q8); // [m][g]: row m to take shift 3m
    aie::vector<int16, 32> h = qv;
    h = aie::select(h, aie::downshift(qv, 3), r1);
    h = aie::select(h, aie::downshift(qv, 6), r2);
    h = aie::select(h, aie::downshift(qv, 9), r3);
    const aie::vector<int16, 32> hb = aie::transpose(aie::bit_and(h, seven), 4, 8); // to [g][m]: octet 4g + m
    const aie::vector<int16, 32> qs = aie::unpack(aie::load_unaligned_v<32>(blk + 2)).cast_to<int16>();
    aie::store_v((int16 *)idx16[n], aie::bit_or(qs, aie::upshift(hb, 8)));
  }
}
#endif

static void decode_tile(const uint8_t *__restrict w) {
#if IQ1_V == 5
  make_idx_vec(w);
#endif
  alignas(64) static bfloat16 d[NR];
  for (int n = 0; n < NR; n++) d[n] = (bfloat16)half_bits((uint16_t)(w[BB * n] | w[BB * n + 1] << 8));
  const bfloat16 *ko = (const bfloat16 *)kodd8_bf16, *kd = (const bfloat16 *)kdelta_bf16;
  const aie::vector<int16, 64> field = aie::load_v<64>(kfield), two = aie::load_v<64>(ktwo);
  const aie::vector<int16, 64> zero = aie::zeros<int16, 64>();
  const aie::vector<bfloat16, 64> one = aie::broadcast<bfloat16, 64>((bfloat16)1.0f), mone = aie::broadcast<bfloat16, 64>((bfloat16)-1.0f);
  const aie::vector<bfloat16, 64> bzero = aie::zeros<bfloat16, 64>();
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bm);
  for (int kh = 0; kh < (KHALF ? 2 : 1); kh++)
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (int j = 0; j < NR / 8; j++) {
    const uint8_t *band = w + BB * 8 * j;
    const aie::vector<bfloat16, 64> dv = aie::concat(
        aie::concat(aie::broadcast<bfloat16, 8>(d[8 * j]), aie::broadcast<bfloat16, 8>(d[8 * j + 1]), aie::broadcast<bfloat16, 8>(d[8 * j + 2]), aie::broadcast<bfloat16, 8>(d[8 * j + 3])),
        aie::concat(aie::broadcast<bfloat16, 8>(d[8 * j + 4]), aie::broadcast<bfloat16, 8>(d[8 * j + 5]), aie::broadcast<bfloat16, 8>(d[8 * j + 6]), aie::broadcast<bfloat16, 8>(d[8 * j + 7])));
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int g = (KHALF ? 4 * kh : 0); g < (KHALF ? 4 * kh + 4 : 8); g++) {
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
#if IQ1_V >= 2
      const aie::vector<bfloat16, 64> vm = aie::mul(aie::add(mone, delta), sc).to_vector<bfloat16>();
      const aie::vector<bfloat16, 64> v0 = aie::mul(delta, sc).to_vector<bfloat16>();
      const aie::vector<bfloat16, 64> vp = aie::mul(aie::add(one, delta), sc).to_vector<bfloat16>();
#endif
      AIE_LOOP_UNROLL_FULL
      for (int l = 0; l < 4; l++) {
#if IQ1_V >= 4
#define C(r) aie::broadcast<int16, 8>((int16)iq1s_grid2[idx16[8 * j + (r)][4 * g + l]])
#elif defined(NO_TABLE) /* timing alone: the index computed, its table load skipped */
#define C(r) aie::broadcast<int16, 8>((int16)(band[BB * (r) + 2 + 4 * g + l] | ((h[r] >> (3 * l)) & 7) << 8))
#elif defined(CONST_ALL)
#define C(r) aie::broadcast<int16, 8>((int16)(0x1111 * (r + 1)))
#elif defined(CONST_CODES) /* timing alone: a row's code a constant (no index, no table load) */
#define C(r) aie::broadcast<int16, 8>((int16)(0x1111 * (r + 1) + l))
#else
#define C(r) aie::broadcast<int16, 8>((int16)iq1s_grid2[band[BB * (r) + 2 + 4 * g + l] | ((h[r] >> (3 * l)) & 7) << 8])
#endif
#if IQ1_V == 3
        const int c = 0;
#else
        const aie::vector<int16, 64> c = aie::concat(aie::concat(C(0), C(1), C(2), C(3)), aie::concat(C(4), C(5), C(6), C(7)));
#endif
#undef C
#if IQ1_V == 3
        // the grid's own int8 rows, 8-byte loads: -1 / 0 / 1 picked by two compares on the row's bytes
#define R(r) aie::load_v<8>((const int8 *)&iq1s_grid8[band[BB * (r) + 2 + 4 * g + l] | ((h[r] >> (3 * l)) & 7) << 8])
        const aie::vector<int16, 64> gi = aie::unpack(aie::concat(aie::concat(R(0), R(1), R(2), R(3)), aie::concat(R(4), R(5), R(6), R(7))));
#undef R
        aie::accum<accfloat, 64> acc;
        acc.from_vector(aie::select(aie::select(v0, vp, aie::eq(gi, aie::broadcast<int16, 64>(1))), vm, aie::eq(gi, aie::broadcast<int16, 64>(-1))));
        out.push(acc.to_vector<bfp16ebs8>());
        (void)c;
#else
        const aie::vector<int16, 64> t = aie::bit_and(c, field);
#if IQ1_V == 2
        aie::accum<accfloat, 64> acc;
        acc.from_vector(aie::select(aie::select(v0, vp, aie::eq(t, two)), vm, aie::eq(t, zero)));
        out.push(acc.to_vector<bfp16ebs8>());
#else
        const aie::vector<bfloat16, 64> gv = aie::select(aie::select(bzero, one, aie::eq(t, two)), mone, aie::eq(t, zero));
        const aie::vector<bfloat16, 64> gd = aie::add(gv, delta);
        out.push(aie::mul(gd, sc).to_vector<bfp16ebs8>());
#endif
#endif
      }
    }
  }
}

#if !defined(MULTI_F32)
extern "C" {
void fused_iq1s(const uint8_t *__restrict w, uint8_t *__restrict a, uint8_t *__restrict c) {
  event0();
#if IQ1_V == 4
  make_idx(w);
#endif
#if IQ1_V == 5
  make_idx(w); /* the checked reference: overwritten by each decode's own */
#endif
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
#endif

#ifdef MULTI_F32
// the 32-core design's step (fused32.py, FMT=iq1s): as fused_iq2xxs.cc's fused_step_f32 - a quarter of a K-step's A
// (piece 2 half + kh), the NR-row W tile decoded at piece 0 (as two K halves), C (2 halves of 64 x NR) kept in f32
template <unsigned rowA, unsigned colA, unsigned colB>
static inline void mm_f32acc(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, float *__restrict pC) {
  using acc_t = aie::accum<accfloat, 64>;
  for (unsigned z = 0; z < rowA; z += 2)
    for (unsigned j = 0; j < colB; j += 2) {
      float *c00 = pC + (z * colB + j) * 64, *c01 = c00 + 64, *c10 = pC + ((z + 1) * colB + j) * 64, *c11 = c10 + 64;
      acc_t a00, a01, a10, a11;
      a00.from_vector(aie::load_v<64>(c00)), a01.from_vector(aie::load_v<64>(c01));
      a10.from_vector(aie::load_v<64>(c10)), a11.from_vector(aie::load_v<64>(c11));
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pAIn(pA);
      pAIn.seek(z * colA);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pBIn(pB);
      pBIn.seek(j * colA);
      AIE_LOOP_UNROLL_FULL
      for (unsigned i = 0; i < colA; ++i) {
        auto A0 = pAIn.pop_seek(colA - 1);
        auto A1 = pAIn.pop_seek(-(int)colA);
        auto B0 = pBIn.pop_seek(colA - 1);
        auto B1 = pBIn.pop_seek(-(int)colA);
        a00 = mac_8x8_8x8T(A0, B0, a00);
        a01 = mac_8x8_8x8T(A0, B1, a01);
        a10 = mac_8x8_8x8T(A1, B0, a10);
        a11 = mac_8x8_8x8T(A1, B1, a11);
      }
      aie::store_v(c00, a00.to_vector<float>()), aie::store_v(c01, a01.to_vector<float>());
      aie::store_v(c10, a10.to_vector<float>()), aie::store_v(c11, a11.to_vector<float>());
    }
}

extern "C" void fused_step_f32(const uint8_t *__restrict w, uint8_t *__restrict a, float *__restrict c, int32_t kstep,
                               int32_t piece) {
  const int half = piece >> 1, kh = piece & 1;
  float *ch = c + half * (64 * NR);
  if (kstep == 0 && kh == 0)
    for (int i = 0; i < 64 * NR; i += 16) aie::store_v(ch + i, aie::zeros<float, 16>());
  if (piece == 0) decode_tile(w);
  mm_f32acc<64 / 8, 128 / 8, NR / 8>((bfp16ebs8 *)a, (bfp16ebs8 *)(Bm + kh * (NR * 128 * 9 / 8)), ch);
}
#endif
