// 0.7.0 spike: ggml's Q2_K decoded on one AIE2P core - NB blocks of 256 values (84 bytes: scales[16], qs[64], d, dmin
// as fp16) to bf16, ggml's dequantize_row_q2_K: y = d * (sc & 15) * q - dmin * (sc >> 4), q the 2-bit value. REPS:
// the same blocks decoded that many times (the core's rate apart from its DMA). Q2K_VEC=0 the scalar reference.
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../common/fp16.h"

#ifndef NB
#define NB 16
#endif
#ifndef REPS
#define REPS 1
#endif
#ifndef Q2K_VEC
#define Q2K_VEC 1
#endif
#ifndef Q2K_BFP
#define Q2K_BFP 0 /* 1: Q2K_VEC 4's rows written as bfp16ebs8, two a 64-lane accumulator (the native bfp16 matmul's) */
#endif
#ifndef Q2K_NOSCALE
#define Q2K_NOSCALE 0 /* timing alone: the scales' scalar float math replaced by constants (wrong values) */
#endif

static inline float half_to_float(uint16_t h) {
  const uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  uint32_t b;
  if (e == 0) { // zero or subnormal: m * 2^-24
    if (!m) b = s;
    else {
      int k = 0;
      uint32_t mm = m;
      while (!(mm & 0x400)) mm <<= 1, k++;
      b = s | ((uint32_t)(113 - k) << 23) | ((mm & 0x3ff) << 13);
    }
  } else if (e == 31) b = s | 0x7f800000 | (m << 13);
  else b = s | ((e + 112) << 23) | (m << 13);
  float f;
  __builtin_memcpy(&f, &b, 4);
  return f;
}

static inline void block_scalar(const uint8_t *blk, bfloat16 *y) {
  const uint8_t *sc = blk, *q = blk + 16;
  const float d = half_to_float((uint16_t)(blk[80] | blk[81] << 8));
  const float dm = half_to_float((uint16_t)(blk[82] | blk[83] << 8));
  int is = 0;
  for (int n = 0; n < 256; n += 128) {
    for (int shift = 0; shift < 8; shift += 2) {
      for (int h = 0; h < 2; h++) {
        const uint8_t s = sc[is++];
        const float dl = d * (float)(s & 15), ml = dm * (float)(s >> 4);
        for (int l = 0; l < 16; l++) *y++ = (bfloat16)(dl * (float)((q[16 * h + l] >> shift) & 3) - ml);
      }
    }
    q += 32;
  }
}

#if Q2K_VEC == 1
// a half block's 32 bytes hold four 32-value rows of 2-bit values (shift 0, 2, 4, 6); a row's first 16 values take one
// scale byte, its last 16 the next: y = dl * q - ml as one 32-lane multiply-add in f32, rounded to bf16 once
static inline void block_vec(const uint8_t *blk, bfloat16 *y) {
  const float d = half_to_float((uint16_t)(blk[80] | blk[81] << 8));
  const float dm = half_to_float((uint16_t)(blk[82] | blk[83] << 8));
  const uint8_t *sc = blk;
  for (int half = 0; half < 2; half++) {
    const aie::vector<uint8, 32> qb = aie::load_unaligned_v<32>(blk + 16 + 32 * half);
    const aie::vector<int16, 32> q16 = aie::unpack(qb).cast_to<int16>(); // 0 .. 255
    for (int r = 0; r < 4; r++) {
      const int shift = 2 * r;
      const aie::vector<int16, 32> v = aie::bit_and(aie::downshift(q16, shift), aie::broadcast<int16, 32>(3));
      const uint8_t s0 = sc[8 * half + 2 * r], s1 = sc[8 * half + 2 * r + 1];
      const aie::vector<float, 32> dl = aie::concat(aie::broadcast<float, 16>(d * (float)(s0 & 15)),
                                                    aie::broadcast<float, 16>(d * (float)(s1 & 15)));
      const aie::vector<float, 32> ml = aie::concat(aie::broadcast<float, 16>(dm * (float)(s0 >> 4)),
                                                    aie::broadcast<float, 16>(dm * (float)(s1 >> 4)));
      const aie::vector<float, 32> qf = aie::to_float(v, 0);
      const aie::accum<accfloat, 32> acc = aie::mac(aie::from_vector<accfloat>(aie::neg(ml)), qf, dl);
      aie::store_v(y, acc.to_vector<bfloat16>());
      y += 32;
    }
  }
}
#endif

#if Q2K_VEC == 2
// a 16-value sub-block holds four values alone (dl * q - ml, q 0 .. 3): computed once in f32 and rounded once, then
// chosen per lane by the 2-bit code - two selects on its low bit, one on its high: no arithmetic a value
static inline aie::vector<bfloat16, 32> pick4(const float *v0, const float *v1, aie::mask<32> b0, aie::mask<32> b1) {
  // v0 / v1: the four values of the row's first / last 16 lanes
  aie::vector<bfloat16, 32> c[4];
  for (int k = 0; k < 4; k++)
    c[k] = aie::concat(aie::broadcast<bfloat16, 16>((bfloat16)v0[k]), aie::broadcast<bfloat16, 16>((bfloat16)v1[k]));
  const aie::vector<bfloat16, 32> lo = aie::select(c[0], c[1], b0), hi = aie::select(c[2], c[3], b0);
  return aie::select(lo, hi, b1);
}

static inline void block_sel(const uint8_t *blk, bfloat16 *y) {
  const float d = half_to_float((uint16_t)(blk[80] | blk[81] << 8));
  const float dm = half_to_float((uint16_t)(blk[82] | blk[83] << 8));
  const uint8_t *sc = blk;
  for (int half = 0; half < 2; half++) {
    const aie::vector<int16, 32> q16 = aie::unpack(aie::load_unaligned_v<32>(blk + 16 + 32 * half)).cast_to<int16>();
    for (int r = 0; r < 4; r++) {
      const aie::vector<int16, 32> v = aie::downshift(q16, 2 * r);
      const aie::mask<32> b0 = aie::neq(aie::bit_and(v, aie::broadcast<int16, 32>(1)), aie::broadcast<int16, 32>(0));
      const aie::mask<32> b1 = aie::neq(aie::bit_and(v, aie::broadcast<int16, 32>(2)), aie::broadcast<int16, 32>(0));
      float v0[4], v1[4];
      const uint8_t s0 = sc[8 * half + 2 * r], s1 = sc[8 * half + 2 * r + 1];
      const float a0 = d * (float)(s0 & 15), m0 = dm * (float)(s0 >> 4), a1 = d * (float)(s1 & 15), m1 = dm * (float)(s1 >> 4);
      for (int k = 0; k < 4; k++) v0[k] = a0 * (float)k - m0, v1[k] = a1 * (float)k - m1;
      aie::store_v(y, pick4(v0, v1, b0, b1));
      y += 32;
    }
  }
}
#endif

#if Q2K_VEC == 3
// bf16 the whole way: the 2-bit code q to its bf16 bits by integer ops (0, 1, 2, 3: 0x0000 0x3f80 0x4000 0x4040), then
// one native bf16 multiply-add a row against the sub-block's scale and min (each rounded to bf16 once a sub-block)
static inline void block_bf(const uint8_t *blk, bfloat16 *y) {
#if Q2K_NOSCALE
  const float d = 0, dm = 0;
#else
  const float d = half_to_float((uint16_t)(blk[80] | blk[81] << 8));
  const float dm = half_to_float((uint16_t)(blk[82] | blk[83] << 8));
#endif
  const uint8_t *sc = blk;
  const aie::vector<int16, 32> one = aie::broadcast<int16, 32>(1), three = aie::broadcast<int16, 32>(3);
  for (int half = 0; half < 2; half++) {
    const aie::vector<int16, 32> q16 = aie::unpack(aie::load_unaligned_v<32>(blk + 16 + 32 * half)).cast_to<int16>();
    for (int r = 0; r < 4; r++) {
      const aie::vector<int16, 32> q = aie::bit_and(aie::downshift(q16, 2 * r), three);
      const aie::vector<int16, 32> hi = aie::downshift(q, 1), lo = aie::bit_and(q, one);
      // 0x3f80 + (hi << 7) + ((lo & hi) << 6), and 0 where q is 0
      aie::vector<int16, 32> bits = aie::add(aie::add(aie::broadcast<int16, 32>(0x3f80), aie::upshift(hi, 7)),
                                             aie::upshift(aie::bit_and(lo, hi), 6));
      bits = aie::select(bits, aie::zeros<int16, 32>(), aie::eq(q, aie::zeros<int16, 32>()));
      const aie::vector<bfloat16, 32> qf = bits.cast_to<bfloat16>();
      const uint8_t s0 = sc[8 * half + 2 * r], s1 = sc[8 * half + 2 * r + 1];
#if Q2K_NOSCALE
      (void)s0, (void)s1, (void)d, (void)dm;
      const aie::vector<bfloat16, 32> dl = aie::broadcast<bfloat16, 32>((bfloat16)0.5f);
      const aie::vector<bfloat16, 32> nml = aie::broadcast<bfloat16, 32>((bfloat16)-0.25f);
#else
      const aie::vector<bfloat16, 32> dl = aie::concat(aie::broadcast<bfloat16, 16>((bfloat16)(d * (float)(s0 & 15))),
                                                       aie::broadcast<bfloat16, 16>((bfloat16)(d * (float)(s1 & 15))));
      const aie::vector<bfloat16, 32> nml = aie::concat(aie::broadcast<bfloat16, 16>((bfloat16)(-dm * (float)(s0 >> 4))),
                                                        aie::broadcast<bfloat16, 16>((bfloat16)(-dm * (float)(s1 >> 4))));
#endif
      aie::accum<accfloat, 32> acc;
      acc.from_vector(nml);
      acc = aie::mac(acc, qf, dl);
      aie::store_v(y, acc.to_vector<bfloat16>());
      y += 32;
    }
  }
}
#endif

#if Q2K_VEC == 4
// no scalar float (the AIE's scalar unit has none: every float op on it is software): fp16 to f32 by its bits (integer
// ops; a zero or subnormal scale taken as 0 - ggml's Q2_K writes normal ones), a block's 16 scales and mins as two
// 16-lane vector products, then a row's two sub-blocks' as broadcasts of their lanes; the row itself as in Q2K_VEC 3
static inline float half_bits(uint16_t h) { return fp16_to_f32_shift(h, 0); } // subnormals too (common/fp16.h)

static inline void block_v4(const uint8_t *blk, bfloat16 *y) {
  const float d = half_bits((uint16_t)(blk[80] | blk[81] << 8));
  const float dm = half_bits((uint16_t)(blk[82] | blk[83] << 8));
  const aie::vector<int16, 16> s16 = aie::unpack(aie::load_unaligned_v<16>(blk)).cast_to<int16>();
  const aie::vector<float, 16> sl = aie::to_float(aie::bit_and(s16, aie::broadcast<int16, 16>(15)), 0);
  const aie::vector<float, 16> sh = aie::to_float(aie::downshift(s16, 4), 0);
  const aie::vector<bfloat16, 16> dlv = aie::mul(sl, aie::broadcast<float, 16>(d)).to_vector<bfloat16>();
  const aie::vector<bfloat16, 16> mlv = aie::mul(sh, aie::broadcast<float, 16>(-dm)).to_vector<bfloat16>();
  const aie::vector<int16, 32> one = aie::broadcast<int16, 32>(1), three = aie::broadcast<int16, 32>(3);
  for (int half = 0; half < 2; half++) {
    const aie::vector<int16, 32> q16 = aie::unpack(aie::load_unaligned_v<32>(blk + 16 + 32 * half)).cast_to<int16>();
    for (int r = 0; r < 4; r++) {
      const aie::vector<int16, 32> q = aie::bit_and(aie::downshift(q16, 2 * r), three);
      const aie::vector<int16, 32> hi = aie::downshift(q, 1), lo = aie::bit_and(q, one);
      aie::vector<int16, 32> bits = aie::add(aie::add(aie::broadcast<int16, 32>(0x3f80), aie::upshift(hi, 7)),
                                             aie::upshift(aie::bit_and(lo, hi), 6));
      bits = aie::select(bits, aie::zeros<int16, 32>(), aie::eq(q, aie::zeros<int16, 32>()));
      const int j = 8 * half + 2 * r;
      const aie::vector<bfloat16, 32> dl = aie::concat(aie::broadcast<bfloat16, 16>(dlv[j]), aie::broadcast<bfloat16, 16>(dlv[j + 1]));
      const aie::vector<bfloat16, 32> nml = aie::concat(aie::broadcast<bfloat16, 16>(mlv[j]), aie::broadcast<bfloat16, 16>(mlv[j + 1]));
      aie::accum<accfloat, 32> acc;
      acc.from_vector(nml);
      acc = aie::mac(acc, bits.cast_to<bfloat16>(), dl);
      aie::store_v(y, acc.to_vector<bfloat16>());
      y += 32;
    }
  }
}
#endif

#if Q2K_BFP
static inline void block_v4_bfp(const uint8_t *blk, aie::block_vector_output_buffer_stream<bfp16ebs8, 64> &out) {
  const float d = half_bits((uint16_t)(blk[80] | blk[81] << 8));
  const float dm = half_bits((uint16_t)(blk[82] | blk[83] << 8));
  const aie::vector<int16, 16> s16 = aie::unpack(aie::load_unaligned_v<16>(blk)).cast_to<int16>();
  const aie::vector<float, 16> sl = aie::to_float(aie::bit_and(s16, aie::broadcast<int16, 16>(15)), 0);
  const aie::vector<float, 16> sh = aie::to_float(aie::downshift(s16, 4), 0);
  const aie::vector<bfloat16, 16> dlv = aie::mul(sl, aie::broadcast<float, 16>(d)).to_vector<bfloat16>();
  const aie::vector<bfloat16, 16> mlv = aie::mul(sh, aie::broadcast<float, 16>(-dm)).to_vector<bfloat16>();
  const aie::vector<int16, 32> one = aie::broadcast<int16, 32>(1), three = aie::broadcast<int16, 32>(3);
  for (int half = 0; half < 2; half++) {
    const aie::vector<int16, 32> q16 = aie::unpack(aie::load_unaligned_v<32>(blk + 16 + 32 * half)).cast_to<int16>();
    aie::accum<accfloat, 32> rows[4];
    for (int r = 0; r < 4; r++) {
      const aie::vector<int16, 32> q = aie::bit_and(aie::downshift(q16, 2 * r), three);
      const aie::vector<int16, 32> hi = aie::downshift(q, 1), lo = aie::bit_and(q, one);
      aie::vector<int16, 32> bits = aie::add(aie::add(aie::broadcast<int16, 32>(0x3f80), aie::upshift(hi, 7)),
                                             aie::upshift(aie::bit_and(lo, hi), 6));
      bits = aie::select(bits, aie::zeros<int16, 32>(), aie::eq(q, aie::zeros<int16, 32>()));
      const int j = 8 * half + 2 * r;
      const aie::vector<bfloat16, 32> dl = aie::concat(aie::broadcast<bfloat16, 16>(dlv[j]), aie::broadcast<bfloat16, 16>(dlv[j + 1]));
      const aie::vector<bfloat16, 32> nml = aie::concat(aie::broadcast<bfloat16, 16>(mlv[j]), aie::broadcast<bfloat16, 16>(mlv[j + 1]));
      rows[r].from_vector(nml);
      rows[r] = aie::mac(rows[r], bits.cast_to<bfloat16>(), dl);
    }
    out.push(aie::concat(rows[0], rows[1]).to_vector<bfp16ebs8>());
    out.push(aie::concat(rows[2], rows[3]).to_vector<bfp16ebs8>());
  }
}
#endif

extern "C" {
#if Q2K_BFP
void q2k_decode(const uint8_t *__restrict in, uint8_t *__restrict out) {
  event0();
  for (int rep = 0; rep < REPS; rep++) {
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> s((bfp16ebs8 *)out);
    for (int b = 0; b < NB; b++) block_v4_bfp(in + 84 * b, s);
  }
  event1();
}
#else
void q2k_decode(const uint8_t *__restrict in, bfloat16 *__restrict out) {
  event0();
  for (int rep = 0; rep < REPS; rep++)
    for (int b = 0; b < NB; b++) {
#if Q2K_VEC == 4
      block_v4(in + 84 * b, out + 256 * b);
#elif Q2K_VEC == 3
      block_bf(in + 84 * b, out + 256 * b);
#elif Q2K_VEC == 2
      block_sel(in + 84 * b, out + 256 * b);
#elif Q2K_VEC
      block_vec(in + 84 * b, out + 256 * b);
#else
      block_scalar(in + 84 * b, out + 256 * b);
#endif
    }
  event1();
}
#endif
}
