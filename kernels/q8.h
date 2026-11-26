// the lane's experts over int8 inputs (kernels/lane.h's contract; include/osmos/gpu/q8.h): a 32-value block of a
// weight row decoded to int8 codes, two words a chunk of 8, against the input's int8 words by v_dot4 - exact integer
// sums - then each half's lane_q8_part and the block's scale, in the contract's order
#pragma once
#include "lane.h"
#include "port.h"
#include "quant.h"

namespace q8 {

constexpr int kWave = 32;
constexpr int kWavesPerBlock = 8;
constexpr int kWords = LANE_Q8_VALUES / 4;  // a block's int8 x as 32-bit words
// a weight block's blocks of 32 (its values over 32), and a row's bytes
template <quant::Type T> constexpr int kBlocksPer = quant::Block<T>::values / LANE_Q8_VALUES;
template <quant::Type T> constexpr uint64_t row_bytes(uint32_t cols) {
  return (uint64_t) (cols / quant::Block<T>::values) * quant::Block<T>::bytes;
}
// a SIMD's waves the kernels ask for: 16 (gfx1151's 1536 VGPRs a SIMD at <= 96 a lane): past 96 registers fewer
// waves, the memory's latency less hidden. the arithmetic and its order are the source's: no bit moves
constexpr int kFullWaves = 16;

__device__ inline int dot4(int a, int b, int c) { return osmos_sudot4(a, b, c); }

// a block's codes: w its 8 words (two a chunk); IQ1_M's chunk c a sign of its own (its grid + delta as 8 grid +- 1:
// 8 (grid . x) +- the chunk's x summed); a half h's 2s + 1; the 256-value block's scale d
struct Codes {
  int w[kWords];
  int delta[4];  // 0: none (the IQ2s), else +-1 (the IQ1s)
  int s21[2];    // a half's 2s + 1 (IQ3_XXS's 2 (2s + 1): its quarter as the eighth's); Q2_K's 8 sc
  float d;
  int m8[2];     // Q2_K's: a half's 8 m, against dmin
  float dmin;
  // the generic types' (kGeneric): a half's float scale, its codes' offset (the value code - off: off x's sum taken
  // off the dot) and its float minimum (m x's sum added)
  float dh[2];
  int off[2];
  float mh[2];
};

// the types whose block sum is block_sum's generic form: a half's dh (dot - off sx) + mh sx
template <quant::Type T>
constexpr bool kGeneric = T == quant::kQ4_0 || T == quant::kQ5_1 || T == quant::kQ8_0 || T == quant::kQ5_K ||
                          T == quant::kQ6_K || T == quant::kIQ4_NL || T == quant::kIQ4_XS || T == quant::kMXFP4 ||
                          T == quant::kNVFP4 || T == quant::kQ2_0;

// a 16-entry int8 table in four registers (bytes 0 .. 15), and 4 bytes of 4-bit codes (one nibble each, at shift sh)
// through it as an int8 word: v_perm_b32 picks each code's byte from entries 0 .. 7 and from 8 .. 15, its top bit the
// half - registers, not a global load a code (32 a block: the matvec under the bandwidth it could stream)
struct Lut16 {
  uint32_t w[4];
};
constexpr Lut16 lut16(const int8_t (&v)[16]) {
  Lut16 t = {};
  for (int i = 0; i < 16; i++) t.w[i / 4] |= (uint32_t) (uint8_t) v[i] << (8 * (i % 4));
  return t;
}
constexpr int8_t kIq4nlLut[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
constexpr int8_t kMxfp4Lut[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
constexpr Lut16 kIq4nlTab = lut16(kIq4nlLut), kMxfp4Tab = lut16(kMxfp4Lut);
__device__ inline int table4(uint32_t w, int sh, const Lut16& t) {
  const uint32_t n = (w >> sh) & 0x0F0F0F0Fu, sel = n & 0x07070707u;
  const uint32_t lo = __builtin_amdgcn_perm(t.w[1], t.w[0], sel), hi = __builtin_amdgcn_perm(t.w[3], t.w[2], sel);
  const uint32_t m = ((n >> 3) & 0x01010101u) * 0xFFu;
  return (int) ((hi & m) | (lo & ~m));
}
__device__ inline void plain(Codes& c, float d0, float d1, int off, float m0 = 0.0f, float m1 = 0.0f) {
  for (int k = 0; k < 4; k++) c.delta[k] = 0;
  c.dh[0] = d0, c.dh[1] = d1, c.off[0] = c.off[1] = off, c.mh[0] = m0, c.mh[1] = m1;
}

// 8 unsigned grid bytes signed by 8 sign bits, as two words of int8: (v ^ m) + (m & 1) a byte, m 0xFF where negative
__device__ inline void signed8(uint64_t grid, uint32_t signs, int& lo, int& hi) {
  uint32_t m0 = 0, m1 = 0;
  for (int i = 0; i < 4; i++) m0 |= ((signs >> i) & 1u) * (0xFFu << (8 * i)), m1 |= ((signs >> (4 + i)) & 1u) * (0xFFu << (8 * i));
  lo = (int) (((uint32_t) grid ^ m0) + (m0 & 0x01010101u)), hi = (int) (((uint32_t) (grid >> 32) ^ m1) + (m1 & 0x01010101u));
}

template <quant::Type T> __device__ void codes(const uint8_t* b, int j, Codes& c);

template <> __device__ inline void codes<quant::kIQ2_XXS>(const uint8_t* b, int j, Codes& c) {
  const uint8_t* q = b + 2 + 8 * j;
  const uint32_t aux = quant::ld32(q + 4);
  for (int k = 0; k < 4; k++) {
    signed8(kIq2xxsGrid[q[k]], kKsigns[(aux >> (7 * k)) & 0x7F], c.w[2 * k], c.w[2 * k + 1]);
    c.delta[k] = 0;
  }
  c.s21[0] = c.s21[1] = 2 * (int) (aux >> 28) + 1;
  c.d = quant::f16(quant::ld16(b));
}

template <> __device__ inline void codes<quant::kIQ2_XS>(const uint8_t* b, int j, Codes& c) {
  for (int k = 0; k < 4; k++) {
    const uint32_t q = quant::ld16(b + 2 + 2 * (4 * j + k));
    signed8(kIq2xsGrid[q & 511], kKsigns[q >> 9], c.w[2 * k], c.w[2 * k + 1]);
    c.delta[k] = 0;
  }
  const uint32_t s = b[66 + j];
  c.s21[0] = 2 * (int) (s & 0x0F) + 1, c.s21[1] = 2 * (int) (s >> 4) + 1;
  c.d = quant::f16(quant::ld16(b));
}

template <> __device__ inline void codes<quant::kIQ1_M>(const uint8_t* b, int j, Codes& c) {
  const uint64_t sc = *(const uint64_t*) (b + 48);  // (a block is 56 bytes, every one 8-aligned)
  const uint32_t s0 = (uint32_t) sc & 0xFFFF, s1 = (uint32_t) (sc >> 16) & 0xFFFF;
  const uint32_t s2 = (uint32_t) (sc >> 32) & 0xFFFF, s3 = (uint32_t) (sc >> 48);
  for (int k = 0; k < 4; k++) {
    const int kk = 4 * j + k;
    const uint32_t h = (b[32 + kk / 2] >> (4 * (kk % 2))) & 0x0F;
    const uint64_t g = kIq1sGrid[b[kk] | (h & 0x07) << 8];
    c.w[2 * k] = (int) (uint32_t) g, c.w[2 * k + 1] = (int) (uint32_t) (g >> 32);
    c.delta[k] = h & 0x08 ? -1 : 1;
  }
  for (int h = 0; h < 2; h++) {
    const int kk = 4 * j + 2 * h;
    c.s21[h] = 2 * (int) ((sc >> (16 * (kk / 8) + 3 * ((kk / 2) % 4))) & 0x07) + 1;
  }
  c.d = quant::f16((s0 >> 12) | ((s1 >> 8) & 0x00F0) | ((s2 >> 4) & 0x0F00) | (s3 & 0xF000));
}

template <> __device__ inline void codes<quant::kIQ2_S>(const uint8_t* b, int j, Codes& c) {
  for (int k = 0; k < 4; k++) {
    const int kk = 4 * j + k;
    signed8(kIq2sGrid[b[2 + kk] | ((b[66 + j] >> (2 * k)) & 0x03) << 8], b[34 + kk], c.w[2 * k], c.w[2 * k + 1]);
    c.delta[k] = 0;
  }
  const uint32_t s = b[74 + j];
  c.s21[0] = 2 * (int) (s & 0x0F) + 1, c.s21[1] = 2 * (int) (s >> 4) + 1;
  c.d = quant::f16(quant::ld16(b));
}

// IQ3_XXS: a chunk's two grid entries (4 values each) signed, the 32's scale its quarter's - 2 (2s + 1) eighths
template <> __device__ inline void codes<quant::kIQ3_XXS>(const uint8_t* b, int j, Codes& c) {
  const uint8_t* q = b + 2 + 8 * j;
  const uint32_t aux = quant::ld32(b + 2 + 64 + 4 * j);
  for (int k = 0; k < 4; k++) {
    const uint64_t g = (uint64_t) kIq3xxsGrid[q[2 * k]] | (uint64_t) kIq3xxsGrid[q[2 * k + 1]] << 32;
    signed8(g, kKsigns[(aux >> (7 * k)) & 0x7F], c.w[2 * k], c.w[2 * k + 1]);
    c.delta[k] = 0;
  }
  c.s21[0] = c.s21[1] = 2 * (2 * (int) (aux >> 28) + 1);
  c.d = quant::f16(quant::ld16(b));
}

// Q2_K: the 32's 2-bit codes (bytes qs[32 (j / 4) ..] at shift 2 (j % 4)), its halves' sc and m (scale bytes 2j, 2j + 1)
template <> __device__ inline void codes<quant::kQ2_K>(const uint8_t* b, int j, Codes& c) {
  const uint8_t* qs = b + 16 + 32 * (j / 4);
  const int shift = 2 * (j % 4);
  for (int t = 0; t < kWords; t++) c.w[t] = (int) ((quant::ld32(qs + 4 * t) >> shift) & 0x03030303u);
  for (int k = 0; k < 4; k++) c.delta[k] = 0;
  for (int h = 0; h < 2; h++) c.s21[h] = 8 * (b[2 * j + h] & 0x0F), c.m8[h] = 8 * (b[2 * j + h] >> 4);
  c.d = quant::f16(quant::ld16(b + 80)), c.dmin = quant::f16(quant::ld16(b + 82));
}

// IQ1_M's codes with the 32's one scale and one delta sign (qh's u16 j)
template <> __device__ inline void codes<quant::kIQ1_S>(const uint8_t* b, int j, Codes& c) {
  const uint32_t h = quant::ld16(b + 34 + 2 * j);
  for (int k = 0; k < 4; k++) {
    const uint64_t g = kIq1sGrid[b[2 + 4 * j + k] | ((h >> (3 * k)) & 0x07) << 8];
    c.w[2 * k] = (int) (uint32_t) g, c.w[2 * k + 1] = (int) (uint32_t) (g >> 32);
    c.delta[k] = h & 0x8000 ? -1 : 1;
  }
  c.s21[0] = c.s21[1] = 2 * (int) ((h >> 12) & 0x07) + 1;
  c.d = quant::f16(quant::ld16(b));
}

// Q4_K: the 32's 4-bit codes (sub-block j: qs[32 (j / 2) ..]'s low nibbles for an even j, high for an odd), its
// 6-bit scale and minimum (ggml's get_scale_min_k4) - Q2_K's form with one scale for both halves: (d sc) dot - (dmin m)
// x's sum, each as its eighths (8 sc, 8 m)
template <> __device__ inline void codes<quant::kQ4_K>(const uint8_t* b, int j, Codes& c) {
  const uint8_t* qs = b + 16 + 32 * (j / 2);
  const int shift = 4 * (j % 2);
  for (int t = 0; t < kWords; t++) c.w[t] = (int) ((quant::ld32(qs + 4 * t) >> shift) & 0x0F0F0F0Fu);
  for (int k = 0; k < 4; k++) c.delta[k] = 0;
  const uint8_t* sc = b + 4;
  const int s = j < 4 ? sc[j] & 63 : (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
  const int m = j < 4 ? sc[j + 4] & 63 : (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
  c.s21[0] = c.s21[1] = 8 * s, c.m8[0] = c.m8[1] = 8 * m;
  c.d = quant::f16(quant::ld16(b)), c.dmin = quant::f16(quant::ld16(b + 2));
}

// Q3_K: the 32's 3-bit codes q | (hmask bit) << 2, unsigned (0 .. 7: the value's code + 4), each half its signed 6-bit
// scale (sc - 32) - the half's sum (codes . x) - 4 (x's sum), exact in integers (decode<kQ3_K>'s values: a code minus 4,
// or its high bit set none)
template <> __device__ inline void codes<quant::kQ3_K>(const uint8_t* b, int j, Codes& c) {
  const int n = j / 4, jj = j % 4, bit = 4 * n + jj;
  for (int h = 0; h < 2; h++) {
    const uint8_t* qs = b + 32 + 32 * n + 16 * h;
    const uint8_t* hm = b + 16 * h;
    for (int t = 0; t < 4; t++) {
      const uint32_t q2 = (quant::ld32(qs + 4 * t) >> (2 * jj)) & 0x03030303u;
      const uint32_t h1 = (quant::ld32(hm + 4 * t) >> bit) & 0x01010101u;
      c.w[4 * h + t] = (int) (q2 | h1 << 2);
    }
    const int g = 8 * n + 2 * jj + h, grp = g / 4, w = g % 4;
    const uint8_t* sc = b + 96;
    const int low = ((grp & 1 ? sc[4 + w] : sc[w]) >> (grp & 2 ? 4 : 0)) & 0x0F;
    c.s21[h] = 8 * ((low | ((sc[8 + w] >> (2 * grp)) & 0x03) << 4) - 32);
  }
  for (int k = 0; k < 4; k++) c.delta[k] = 0;
  c.d = quant::f16(quant::ld16(b + 108));
}

// the generic types (kGeneric): each a 32-value block j's codes as int8 words and its halves' float scales - the F32
// decoders' values (quant.h's decode<T>) as dh (code - off) + mh

// Q4_0: d (code - 8), codes 0 .. 15 - values 0 .. 15 the low nibbles of qs's 16 bytes, 16 .. 31 the high
template <> __device__ inline void codes<quant::kQ4_0>(const uint8_t* b, int, Codes& c) {
  for (int t = 0; t < 4; t++) {
    const uint32_t w = quant::ld32(b + 2 + 4 * t);
    c.w[t] = (int) (w & 0x0F0F0F0Fu), c.w[4 + t] = (int) ((w >> 4) & 0x0F0F0F0Fu);
  }
  const float d = quant::f16(quant::ld16(b));
  plain(c, d, d, 8);
}
// IQ4_NL: d kvalues[code], Q4_0's layout - the table's int8 values the codes
template <> __device__ inline void codes<quant::kIQ4_NL>(const uint8_t* b, int, Codes& c) {
  for (int t = 0; t < 4; t++) {
    const uint32_t w = quant::ld32(b + 2 + 4 * t);
    c.w[t] = table4(w, 0, kIq4nlTab), c.w[4 + t] = table4(w, 4, kIq4nlTab);
  }
  const float d = quant::f16(quant::ld16(b));
  plain(c, d, d, 0);
}
// MXFP4: 2^(e - 128) times E2M1 doubled (kMxfp4Values: integers), IQ4_NL's layout after its scale byte
template <> __device__ inline void codes<quant::kMXFP4>(const uint8_t* b, int, Codes& c) {
  for (int t = 0; t < 4; t++) {
    const uint32_t w = (uint32_t) b[1 + 4 * t] | (uint32_t) b[2 + 4 * t] << 8 | (uint32_t) b[3 + 4 * t] << 16 |
                       (uint32_t) b[4 + 4 * t] << 24;
    c.w[t] = table4(w, 0, kMxfp4Tab), c.w[4 + t] = table4(w, 4, kMxfp4Tab);
  }
  const float d = ldexpf(1.0f, (int) b[0] - 128);
  plain(c, d, d, 0);
}
// Q8_0: d code, the codes the int8s themselves
template <> __device__ inline void codes<quant::kQ8_0>(const uint8_t* b, int, Codes& c) {
  for (int t = 0; t < kWords; t++) c.w[t] = (int) quant::ld32(b + 2 + 4 * t);
  const float d = quant::f16(quant::ld16(b));
  plain(c, d, d, 0);
}
// Q5_1: d q + m, q the 4 low bits (Q4_0's layout, from byte 8) and value i's fifth bit i of qh (bytes 4 .. 7)
template <> __device__ inline void codes<quant::kQ5_1>(const uint8_t* b, int, Codes& c) {
  const uint32_t qh = quant::ld32(b + 4);
  for (int t = 0; t < 4; t++) {
    const uint32_t w = quant::ld32(b + 8 + 4 * t);
    uint32_t h0 = 0, h1 = 0;
    for (int i = 0; i < 4; i++) h0 |= ((qh >> (4 * t + i)) & 1u) << (8 * i + 4), h1 |= ((qh >> (16 + 4 * t + i)) & 1u) << (8 * i + 4);
    c.w[t] = (int) ((w & 0x0F0F0F0Fu) | h0), c.w[4 + t] = (int) (((w >> 4) & 0x0F0F0F0Fu) | h1);
  }
  const float d = quant::f16(quant::ld16(b)), m = quant::f16(quant::ld16(b + 2));
  plain(c, d, d, 0, m, m);
}
// Q2_0: d (code - 1), 64 values: value i bits 2 (i % 4) of byte 2 + i / 4 - block j its bytes 2 + 8j ..
template <> __device__ inline void codes<quant::kQ2_0>(const uint8_t* b, int j, Codes& c) {
  for (int t = 0; t < kWords; t++) {
    const uint32_t v = b[2 + 8 * j + t];
    c.w[t] = (int) ((v & 3u) | ((v >> 2) & 3u) << 8 | ((v >> 4) & 3u) << 16 | ((v >> 6) & 3u) << 24);
  }
  const float d = quant::f16(quant::ld16(b));
  plain(c, d, d, 1);
}
// NVFP4: 64 values, 4 UE4M3 scales (sub-block s, values 16s ..) then sub-block s's value i < 8 in byte 4 + 8s + i's
// low nibble, i + 8 its high one; E2M1 doubled times half the scale - block j its sub-blocks 2j and 2j + 1, a half each
template <> __device__ inline void codes<quant::kNVFP4>(const uint8_t* b, int j, Codes& c) {
  for (int h = 0; h < 2; h++) {
    const uint8_t* q = b + 4 + 8 * (2 * j + h);
    for (int t = 0; t < 2; t++) {
      const uint32_t w = (uint32_t) q[4 * t] | (uint32_t) q[4 * t + 1] << 8 | (uint32_t) q[4 * t + 2] << 16 |
                         (uint32_t) q[4 * t + 3] << 24;
      c.w[4 * h + t] = table4(w, 0, kMxfp4Tab), c.w[4 * h + 2 + t] = table4(w, 4, kMxfp4Tab);
    }
  }
  plain(c, rnd::ue4m3_value(b[2 * j]) * 0.5f, rnd::ue4m3_value(b[2 * j + 1]) * 0.5f, 0);
}
// Q5_K: (d sc) q - (dmin m), group j's 4 low bits qs[32 (j / 2) ..] (low nibbles even j, high odd) and its fifth
// qh's bit j; scales Q4_K's
template <> __device__ inline void codes<quant::kQ5_K>(const uint8_t* b, int j, Codes& c) {
  const uint8_t *qs = b + 48 + 32 * (j / 2), *qh = b + 16;
  const int shift = 4 * (j % 2);
  for (int t = 0; t < kWords; t++)
    c.w[t] = (int) (((quant::ld32(qs + 4 * t) >> shift) & 0x0F0F0F0Fu) | (((quant::ld32(qh + 4 * t) >> j) & 0x01010101u) << 4));
  const uint8_t* sc = b + 4;
  const int s = j < 4 ? sc[j] & 63 : (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
  const int m = j < 4 ? sc[j + 4] & 63 : (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
  const float d = quant::f16(quant::ld16(b)) * (float) s, dm = quant::f16(quant::ld16(b + 2)) * (float) m;
  plain(c, d, d, 0, -dm, -dm);
}
// Q6_K: (d sc) (q - 32), q 4 bits of ql and 2 of qh; group j (h = j / 4, r = j % 4) its values l: ql[64 h + 32 (r % 2)
// + l] >> 4 (r / 2), qh[32 h + l] >> 2 r; a half's int8 scale sc[2j + half]
template <> __device__ inline void codes<quant::kQ6_K>(const uint8_t* b, int j, Codes& c) {
  const int h = j / 4, r = j % 4;
  const uint8_t *ql = b + 64 * h + 32 * (r % 2), *qh = b + 128 + 32 * h;
  for (int t = 0; t < kWords; t++)
    c.w[t] = (int) (((quant::ld32(ql + 4 * t) >> (4 * (r / 2))) & 0x0F0F0F0Fu) |
                    (((quant::ld32(qh + 4 * t) >> (2 * r)) & 0x03030303u) << 4));
  const int8_t* sc = (const int8_t*) (b + 192);
  const float d = quant::f16(quant::ld16(b + 208));
  plain(c, d * (float) sc[2 * j], d * (float) sc[2 * j + 1], 32);
}
// IQ4_XS: (d (ls - 32)) kvalues[code], group j's 6-bit ls, its codes IQ4_NL's layout in qs[16 j ..] (from byte 8)
template <> __device__ inline void codes<quant::kIQ4_XS>(const uint8_t* b, int j, Codes& c) {
  for (int t = 0; t < 4; t++) {
    const uint32_t w = quant::ld32(b + 8 + 16 * j + 4 * t);
    c.w[t] = table4(w, 0, kIq4nlTab), c.w[4 + t] = table4(w, 4, kIq4nlTab);
  }
  const int ls = ((b[4 + j / 2] >> (4 * (j % 2))) & 0x0F) | ((quant::ld16(b + 2) >> (2 * j)) & 0x03) << 4;
  const float d = quant::f16(quant::ld16(b)) * (float) (ls - 32);
  plain(c, d, d, 0);
}

// a block's exact integers against x: g[h] its half h's codes' dot (16 values), q[k] x's quarter k's sum (8 values).
// every type's block sum is combine<T>'s float arithmetic on them
template <quant::Type T>
constexpr bool kQuarters = kGeneric<T> || T == quant::kQ3_K || T == quant::kQ2_K || T == quant::kQ4_K ||
                           T == quant::kIQ1_M || T == quant::kIQ1_S;

// the block's sum before x's scale, from its integers: each half's, in the contract's order
template <quant::Type T>
__device__ inline float combine(const Codes& c, const int* g, const int* q) {
  if constexpr (kGeneric<T>) {  // each half: dh (its codes' dot - off x's sum) + mh x's sum
    float r = 0.0f;
    for (int h = 0; h < 2; h++) {
      const int sx = q[2 * h] + q[2 * h + 1];
      const float p = c.dh[h] * (float) (g[h] - c.off[h] * sx) + c.mh[h] * (float) sx;
      r = h ? r + p : p;
    }
    return r;
  }
  if constexpr (T == quant::kQ3_K) {  // each half: its codes' dot less 4 x's sum (the codes' + 4), its scale
    float r = 0.0f;
    for (int h = 0; h < 2; h++) {
      const float p = lane_q8_part(c.d, c.s21[h], g[h] - 4 * (q[2 * h] + q[2 * h + 1]));
      r = h ? r + p : p;
    }
    return r;
  }
  if constexpr (T == quant::kQ2_K || T == quant::kQ4_K) {  // each half: its codes' dot and x's sum (a minimum's)
    float r = 0.0f;
    for (int h = 0; h < 2; h++) {
      const int sx = q[2 * h] + q[2 * h + 1];
      r = h ? r + lane_q8_part_min(c.d, c.s21[1], g[1], c.dmin, c.m8[1], sx)
            : lane_q8_part_min(c.d, c.s21[0], g[0], c.dmin, c.m8[0], sx);
    }
    return r;
  }
  int isum[2];
  for (int h = 0; h < 2; h++) {
    if constexpr (T == quant::kIQ1_M || T == quant::kIQ1_S)  // 8 (grid . x) +- each quarter's x sum
      isum[h] = 8 * g[h] + c.delta[2 * h] * q[2 * h] + c.delta[2 * h + 1] * q[2 * h + 1];
    else
      isum[h] = g[h];
  }
  return lane_q8_part(c.d, c.s21[0], isum[0]) + lane_q8_part(c.d, c.s21[1], isum[1]);
}

// the block's sum against x's 8 words, before x's scale: its integers by v_dot4, then combine's
template <quant::Type T>
__device__ inline float block_sum(const Codes& c, const int* x) {
  int g[2], q[4] = {0, 0, 0, 0};
  for (int h = 0; h < 2; h++)
    g[h] = dot4(c.w[4 * h + 3], x[4 * h + 3],
                dot4(c.w[4 * h + 2], x[4 * h + 2], dot4(c.w[4 * h + 1], x[4 * h + 1], dot4(c.w[4 * h], x[4 * h], 0))));
  if constexpr (kQuarters<T>)
    for (int k = 0; k < 4; k++) q[k] = dot4(0x01010101, x[2 * k + 1], dot4(0x01010101, x[2 * k], 0));
  return combine<T>(c, g, q);
}

// the fused gate + up's activation on the two sums of output row r: the lane's (lane_swiglu, its weight scale[win ? win[r]
// : r], NULL 1) or Inkling's (each sum times its row's tensor scale sg[r] / su[r], then gcap tanh(g / gcap) sigmoid(g)
// (ucap tanh(u / ucap)) - ink2's swiglu_of, the weights at its sum). the matvec's and the matmul's the same function:
// the same bits
enum Act { kActLane = 0, kActInk2 = 1 };
struct ActArgs {
  const float* scale;
  const int32_t* win;
  float lim;
  const float *sg, *su;
  float gcap, ucap;
};
template <int ACT>
__device__ inline float activate(float g, float u, const ActArgs& a, uint32_t r) {
  if constexpr (ACT == kActInk2) {
    const float gv = g * a.sg[r], uv = u * a.su[r];
    return a.gcap * tanhf(gv / a.gcap) * (1.0f / (1.0f + expf(-gv))) * (a.ucap * tanhf(uv / a.ucap));
  } else {
    return lane_swiglu(g, u, a.lim, a.scale ? a.scale[a.win ? a.win[r] : r] : 1.0f);
  }
}

// a launch's NT input rows for output rows r0 .. r0 + NT - 1: row r's own (xin NULL), else row xin[r] (a token's row,
// rounded once for all its experts); the rows past the count read as the count's last - summed, never stored - so no
// table entry past it is read
template <int NT>
__device__ inline void in_rows(const int32_t* xin, uint32_t r0, uint32_t cnt, uint64_t* xr) {
  for (int t = 0; t < NT; t++) {
    const uint32_t r = r0 + ((uint32_t) t < cnt ? (uint32_t) t : cnt - 1);
    xr[t] = xin ? (uint64_t) (uint32_t) xin[r] : r;
  }
}

// matrix blockIdx.y: G = 32 / L rows a wave, L = LANE_Q8_LANES(cols) lanes a row; lane l of a row its blocks l,
// l + L, ... in order for each of its NT tokens (input rows by in_rows), then the lanes' tree. rows past cnt[m] summed,
// not stored. SEL which matrices are this launch's (the rest: nothing read or stored): kSelAll every one, kSelExact a
// count of NT (0 not), kSelTwo a count of 2 or more (NT the table's most: the few past one summed at its width)
enum Sel { kSelAll = 0, kSelExact = 1, kSelTwo = 2 };
template <int SEL, int NT>
__device__ inline bool selected(int32_t c) {
  return SEL == kSelAll || (SEL == kSelExact ? c == NT : c >= 2);
}
template <quant::Type T, int NT, int L, int SEL = kSelAll>
__global__ __launch_bounds__(kWave* kWavesPerBlock) __attribute__((amdgpu_waves_per_eu(kFullWaves)))
void group_tok_kernel(const uint8_t* const* __restrict__ w,
                                                                           const int32_t* __restrict__ cnt,
                                                                           const int32_t* __restrict__ xrow,
                                                                           const int32_t* __restrict__ xin,
                                                                           uint32_t rows, uint32_t cols,
                                                                           const int8_t* __restrict__ q,
                                                                           const float* __restrict__ d,
                                                                           float* __restrict__ y, uint32_t ys) {
  constexpr int G = kWave / L;
  const int lane = threadIdx.x % kWave, ll = lane % L;
  const uint32_t m = blockIdx.y, row = (blockIdx.x * kWavesPerBlock + threadIdx.x / kWave) * G + lane / L;
  if constexpr (SEL != kSelAll)
    if (!selected<SEL, NT>(cnt[m])) return;  // the whole block
  const uint32_t nblk = cols / LANE_Q8_VALUES, r0 = (uint32_t) xrow[m];
  const uint8_t* rb = w[m] + (uint64_t) (row < rows ? row : 0) * row_bytes<T>(cols);
  uint64_t xr[NT];
  in_rows<NT>(xin, r0, (uint32_t) cnt[m], xr);
  float acc[NT];
  for (int t = 0; t < NT; t++) acc[t] = 0.0f;
  for (uint32_t s = ll; s < nblk; s += L) {
    Codes c;
    codes<T>(rb + (s / kBlocksPer<T>) * quant::Block<T>::bytes, (int) (s % kBlocksPer<T>), c);
    for (int t = 0; t < NT; t++) {
      const int4* xp = (const int4*) (q + xr[t] * cols + (uint64_t) s * LANE_Q8_VALUES);
      const int4 a = xp[0], b = xp[1];
      const int x[kWords] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      acc[t] = fmaf(d[xr[t] * nblk + s], block_sum<T>(c, x), acc[t]);
    }
  }
  for (int t = 0; t < NT; t++) {
    float v = acc[t];
    for (int o = L / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, kWave);
    if (ll == 0 && row < rows && t < cnt[m]) y[(uint64_t) (r0 + t) * ys + row] = v;
  }
}

// gate and up at once, then the activation: matrix pair blockIdx.y's gate rows (wg) and up rows (wu) of one type,
// each lane's blocks decoded from both and summed against x read once, each sum in group_tok_kernel's order and tree;
// then lane 0 of a row r h = lane_swiglu(gate, up, lim, scale[win ? win[r] : r]) (scale NULL: 1 - a family whose
// weights are applied at the sum) - swiglu_lane_kernel's operation on the sums
// q8_group_tok gives, its two outputs never written. SEL as group_tok_kernel's
template <quant::Type T, int NT, int L, int SEL = kSelAll, int ACT = kActLane>
__global__ __launch_bounds__(kWave* kWavesPerBlock) __attribute__((amdgpu_waves_per_eu(kFullWaves)))
void gate_up_kernel(const uint8_t* const* __restrict__ wg,
                                                                         const uint8_t* const* __restrict__ wu,
                                                                         const int32_t* __restrict__ cnt,
                                                                         const int32_t* __restrict__ xrow,
                                                                         const int32_t* __restrict__ xin,
                                                                         uint32_t rows, uint32_t cols,
                                                                         const int8_t* __restrict__ q,
                                                                         const float* __restrict__ d, ActArgs act,
                                                                         float* __restrict__ h, uint32_t hs) {
  constexpr int G = kWave / L;
  const int lane = threadIdx.x % kWave, ll = lane % L;
  const uint32_t m = blockIdx.y, row = (blockIdx.x * kWavesPerBlock + threadIdx.x / kWave) * G + lane / L;
  if constexpr (SEL != kSelAll)
    if (!selected<SEL, NT>(cnt[m])) return;  // the whole block
  const uint32_t nblk = cols / LANE_Q8_VALUES, r0 = (uint32_t) xrow[m];
  const uint64_t off = (uint64_t) (row < rows ? row : 0) * row_bytes<T>(cols);
  const uint8_t *rg = wg[m] + off, *ru = wu[m] + off;
  uint64_t xr[NT];
  in_rows<NT>(xin, r0, (uint32_t) cnt[m], xr);
  float ag[NT], au[NT];
  for (int t = 0; t < NT; t++) ag[t] = au[t] = 0.0f;
  for (uint32_t s = ll; s < nblk; s += L) {
    Codes cg, cu;
    const uint64_t at = (s / kBlocksPer<T>) * quant::Block<T>::bytes;
    codes<T>(rg + at, (int) (s % kBlocksPer<T>), cg);
    codes<T>(ru + at, (int) (s % kBlocksPer<T>), cu);
    for (int t = 0; t < NT; t++) {
      const int4* xp = (const int4*) (q + xr[t] * cols + (uint64_t) s * LANE_Q8_VALUES);
      const int4 a = xp[0], b = xp[1];
      const int x[kWords] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      const float xd = d[xr[t] * nblk + s];
      ag[t] = fmaf(xd, block_sum<T>(cg, x), ag[t]);
      au[t] = fmaf(xd, block_sum<T>(cu, x), au[t]);
    }
  }
  for (int t = 0; t < NT; t++) {
    float vg = ag[t], vu = au[t];
    for (int o = L / 2; o > 0; o >>= 1) vg += __shfl_xor(vg, o, kWave), vu += __shfl_xor(vu, o, kWave);
    if (ll == 0 && row < rows && t < cnt[m])
      h[(uint64_t) (r0 + t) * hs + row] = activate<ACT>(vg, vu, act, r0 + t);
  }
}

}  // namespace q8
