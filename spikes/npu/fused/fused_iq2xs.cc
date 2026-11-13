// 0.7.0: DeepSeek's IQ2_XS experts on the array (fused_ds.py FMT=iq2xs) - a W tile of NR rows decoded straight into
// the native bfp16 matmul's sub-tile order, C kept in f32 (common/mm_f32.h). a 74-byte block: d fp16, qs[32] 16-bit
// (a 9-bit grid index, a 7-bit sign index), scales[8] (two 4-bit a 32: the low nibble octets 0-1, the high 2-3); a
// value d (2s + 1) / 8 grid[j] sign (ggml's dequantize_row_iq2_xs). as fused_iq2xxs.cc's v6: a band's 8 rows at an
// octet one 64-lane vector of grid rows (the grid as bf16, 8 KB), the sign bytes from their 7 bits by parity in
// vectors (a group's 32 at once, ksigns_iq2xs's rule), a group's two scales two vectors.
#define MATMUL_ONLY
#define DIM_M 64
#define DIM_K 256
#define DIM_N 64
#include "linalg/mm_bfp.cc"

#include "aie_kernel_utils.h"
#include "../common/fp16.h"
#include "../common/c_rows.h"
#include "../iq2xs/iq2xs_tables.h"

#ifndef KHALF
#define KHALF 1
#endif
#ifndef NROWS
#define NROWS 24
#endif
#ifndef ROWSTRIDE
#define ROWSTRIDE 148 /* a W tile row's bytes: a K-step pair (two 74-byte blocks), or a row whole (WHOLEK) */
#endif
constexpr int NR = NROWS, KV = 256, BB = ROWSTRIDE;

alignas(64) static uint8_t Bm[NR * KV * 9 / 8]; // the decoded weights, sub-tile order

// the odd factor 2s + 1 (1 .. 31) of a scale, exact in bf16
alignas(64) static const uint16_t kodd_bf16[16] = {0x3f80, 0x4040, 0x40a0, 0x40e0, 0x4110, 0x4130, 0x4150, 0x4170,
                                                   0x4188, 0x4198, 0x41a8, 0x41b8, 0x41c8, 0x41d8, 0x41e8, 0x41f8};
alignas(64) static const int32_t krow32[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                               2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};

static inline uint32_t u32_at(const uint8_t *p) { // 2-byte aligned: two halves
  const uint16_t *h = (const uint16_t *)p;
  return (uint32_t)h[0] | (uint32_t)h[1] << 16;
}

#define BCAST8(f) aie::concat(aie::concat(f(0), f(1), f(2), f(3)), aie::concat(f(4), f(5), f(6), f(7)))

static void decode_tile(const uint8_t *__restrict w) {
  alignas(64) static bfloat16 d8[NR]; // a row's d / 8
  for (int n = 0; n < NR; n++) d8[n] = fp16_to_bf16_shift((uint16_t)(w[BB * n] | w[BB * n + 1] << 8), 3);
  const bfloat16 *kb = (const bfloat16 *)kodd_bf16;
  const aie::vector<int32, 32> rowv = aie::load_v<32>(krow32);
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out((bfp16ebs8 *)Bm);
  for (int kh = 0; kh < (KHALF ? 2 : 1); kh++)
    AIE_LOOP_MIN_ITERATION_COUNT(3)
    for (int j = 0; j < NR / 8; j++) {
      const uint8_t *band = w + BB * 8 * j;
#define D(r) aie::broadcast<bfloat16, 8>(d8[8 * j + (r)])
      const aie::vector<bfloat16, 64> dv = BCAST8(D);
#undef D
      AIE_LOOP_MIN_ITERATION_COUNT(4)
      for (int g = (KHALF ? 4 * kh : 0); g < (KHALF ? 4 * kh + 4 : 8); g++) {
#define KL(r) aie::broadcast<bfloat16, 8>(kb[band[BB * (r) + 66 + g] & 15])
#define KH(r) aie::broadcast<bfloat16, 8>(kb[band[BB * (r) + 66 + g] >> 4])
        const aie::vector<bfloat16, 64> scl = aie::mul(dv, BCAST8(KL)).to_vector<bfloat16>();
        const aie::vector<bfloat16, 64> sch = aie::mul(dv, BCAST8(KH)).to_vector<bfloat16>();
#undef KL
#undef KH
        // a row's 4 qs words of group g as two 32-bit words: octets 0, 1 in lo (bits 0, 16), 2, 3 in hi
        aie::vector<int32, 8> lo8, hi8;
        AIE_LOOP_UNROLL_FULL
        for (int r = 0; r < 8; r++)
          lo8[r] = (int32)u32_at(band + BB * r + 2 + 8 * g), hi8[r] = (int32)u32_at(band + BB * r + 6 + 8 * g);
        // lanes [l][r]: octet l's 7 sign bits (bit 9 of its 16), its sign byte by parity (bit 7 the odd's)
        const aie::vector<int32, 32> x = aie::concat(lo8, lo8, hi8, hi8);
        aie::vector<int32, 32> t = aie::downshift(x, 9);
        t = aie::select(t, aie::downshift(x, 25), aie::eq(rowv, aie::broadcast<int32, 32>(1)));
        t = aie::select(t, aie::downshift(x, 25), aie::eq(rowv, aie::broadcast<int32, 32>(3)));
        t = aie::bit_and(t, aie::broadcast<int32, 32>(127));
        aie::vector<int32, 32> p = aie::bit_xor(t, aie::downshift(t, 4));
        p = aie::bit_xor(p, aie::downshift(p, 2));
        p = aie::bit_xor(p, aie::downshift(p, 1));
        const aie::vector<int32, 32> sb = aie::bit_or(t, aie::upshift(aie::bit_and(p, aie::broadcast<int32, 32>(1)), 7));
        const aie::vector<uint32, 8> sw =
            aie::filter_even(aie::filter_even(sb.cast_to<int16>(), 1).cast_to<uint8>(), 1).cast_to<uint32>();
        const uint16_t *qs = (const uint16_t *)(band + 2 + 8 * g); // row r's octet l index at qs[BB / 2 r + l]
        AIE_LOOP_UNROLL_FULL
        for (int l = 0; l < 4; l++) {
#define ROW(r) aie::load_v<8>((const bfloat16 *)iq2xs_grid_bf16[qs[BB / 2 * (r) + l] & 511])
          const aie::vector<bfloat16, 64> v = BCAST8(ROW);
#undef ROW
          const aie::vector<int16, 64> vi = v.cast_to<int16>();
          const aie::vector<bfloat16, 64> sv =
              aie::select(vi, aie::bit_xor(vi, aie::broadcast<int16, 64>((int16)0x8000)), aie::mask<64>::from_uint32(sw[2 * l], sw[2 * l + 1]))
                  .cast_to<bfloat16>();
          out.push(aie::mul(sv, l < 2 ? scl : sch).to_vector<bfp16ebs8>());
        }
      }
    }
}

#include "../common/mm_f32.h"

// a quarter of a K-step's A: piece p = 2 * half + kh; the W tile decoded at piece 0; C (2 halves of 64 x NR f32)
// zeroed at the first K-step's
extern "C" void fused_step_f32(const uint8_t *__restrict w, uint8_t *__restrict a, float *__restrict c, int32_t kstep,
                               int32_t piece) {
  const int half = piece >> 1, kh = piece & 1;
  float *ch = c + half * (64 * NR);
  if (kstep == 0 && kh == 0)
    for (int i = 0; i < 64 * NR; i += 16) aie::store_v(ch + i, aie::zeros<float, 16>());
#ifdef WHOLEK
  if (piece == 0) decode_tile(w + 74 * kstep); // the rows whole (KIND=down): K-step s their block s
#else
  if (piece == 0) decode_tile(w + 74 * (kstep & 1)); // a pair tile: odd K-steps its second block
#endif
  mm_f32acc<64 / 8, 128 / 8, NR / 8>((bfp16ebs8 *)a, (bfp16ebs8 *)(Bm + kh * (NR * 128 * 9 / 8)), ch);
#ifdef ROWMAJOR
  if (kstep == ROWMAJOR - 1 && piece == 3) // the step's last piece (ROWMAJOR: the K-steps): both halves' sums done,
    for (int h = 0; h < 2; h++) c_rows<NR>(c + h * (64 * NR), (float *)Bm); // and Bm free - to rows
#endif
}
