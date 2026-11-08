#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

#include "lane.h"
#include "quant_grids.h"
#include "osmos/cpu/q8.h"

/* kernels/lane.h's contract on AVX-512 (the GPU's kernels/q8.h): 16 blocks of a row at a time - their codes decoded
   once as 8 vectors of 64 magnitudes and 64 signs, then for each token x negated where a code is, vpdpbusd's exact
   sums, each 16-value half's 4 of them added by an in-lane transpose, and lane.h's float steps a block a float lane.
   built -ffp-contract=off: the one fused product the fmadd the contract writes */

#define QK 256
#define PER (QK / LANE_Q8_VALUES) /* a 256-value block's blocks of 32 */
#define GROUP 16                  /* blocks a step of the float steps: a float vector's lanes */
#define PAIRS (GROUP / 2)         /* 64-byte vectors a group: two blocks each */
#define HALF_GROUP 0x00FF         /* the float lanes of a group's first 8 blocks (a row's last, when it ends there) */
/* a group's float lane e holds its block perm(e) = 8 (e / 8) + 2 (e % 4) + (e / 4) % 2: the transpose's order (a
   128-bit lane of the dots one half-block, a pair's 4 halves a vector's 4 lanes) */
#define SETR(...) _mm512_setr_epi32(__VA_ARGS__) /* (gcc's is a macro: the lists below expanded first) */
#define PERM 0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15
#define PERM_INV 0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15
/* the pairs' 2s + 1 bytes (pair k's at 4 k: block 2 k's two halves, then block 2 k + 1's) to the float lanes: lane
   e's half h is byte 2 perm(e) + h - 16 bytes of first halves, then 16 of second */
#define SCALE_BYTES 0x0C080400, 0x0E0A0602, 0x1C181410, 0x1E1A1612, 0x0D090501, 0x0F0B0703, 0x1D191511, 0x1F1B1713
#define INLINE __attribute__((always_inline)) inline
#define AVX512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx512vbmi,bmi2")))

static uint32_t ld16(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8; }
static uint32_t ld32(const uint8_t *p) { return ld16(p) | ld16(p + 2) << 16; }
static uint64_t ld64(const uint8_t *p) { return (uint64_t)ld32(p) | (uint64_t)ld32(p + 4) << 32; }
static float f16(uint32_t bits)
{
  const uint16_t b = (uint16_t)bits;
  _Float16 h;
  memcpy(&h, &b, sizeof h);
  return (float)h;
}

/* two blocks' codes (64 values): magnitudes as bytes, the negative ones' mask; each half's 2s + 1 (block 0's two,
   block 1's two, a byte each) and the 256-value block's d */
#define NIBBLES 0x0F0F0F0Fu /* 4 fields of 4 bits to a byte each by pdep */
#define TRIPLES 0x07070707u /* of 3 bits */
#define ODD 0x01010101u     /* each byte's + 1 of 2s + 1 (2s + 1 <= 31: no carry) */
typedef struct
{
  __m512i mag;
  __mmask64 neg;
  uint32_t s21; /* a byte each: block 0's halves, then block 1's (Q2_K's 8 sc) */
  float d;
  uint32_t m8; /* Q2_K's: its halves' 8 m, as s21's bytes */
  float dmin;
} Pair;

/* IQ2_XXS's and IQ2_XS's 8 chunks: unsigned grid bytes; a sign byte each, kKsigns' - its 7-bit index and that
   index's parity as bit 7 (an even count of negatives) - computed from the 8 indices (56 bits), not looked up: each
   index to a byte by pdep, each byte's parity folded into its bit 0 (the shifts within a byte), then to bit 7 */
#define KSIGN_BYTES 0x7F7F7F7F7F7F7F7Full
#define KSIGN_LOW 0x0101010101010101ull
#define KSIGN_PARITY 7
AVX512 static INLINE void signed_grid(Pair *p, const uint64_t g[8], uint64_t idx56)
{
  const uint64_t m = _pdep_u64(idx56, KSIGN_BYTES);
  uint64_t x = m ^ (m >> 4);
  x ^= x >> 2;
  x ^= x >> 1;
  p->mag = _mm512_set_epi64((long long)g[7], (long long)g[6], (long long)g[5], (long long)g[4], (long long)g[3],
                            (long long)g[2], (long long)g[1], (long long)g[0]);
  p->neg = _cvtu64_mask64(m | (x & KSIGN_LOW) << KSIGN_PARITY);
}

#define IQ2_XXS_SIGNS 0x0FFFFFFFu /* an aux word's 4 sign indices; its top 4 bits s */
AVX512 static INLINE void pair_iq2_xxs(const uint8_t *b, int j, Pair *p)
{
  const uint8_t *q = b + 2 + 8 * j; /* a block 8 bytes: 4 grid indices, then the aux word */
  const uint32_t a0 = ld32(q + 4), a1 = ld32(q + 12);
  uint64_t g[8];
#pragma GCC unroll 4
  for (int c = 0; c < 4; c++)
    g[c] = kIq2xxsGrid[q[c]], g[4 + c] = kIq2xxsGrid[q[8 + c]];
  signed_grid(p, g, (uint64_t)(a0 & IQ2_XXS_SIGNS) | (uint64_t)(a1 & IQ2_XXS_SIGNS) << 28);
  p->s21 = 2 * ((a0 >> 28) * 0x0101u | (a1 >> 28) * 0x01010000u) + ODD;
  p->d = f16(ld16(b));
}

#define IQ2_XS_SIGNS 0xFE00FE00FE00FE00ull /* 4 words' top 7 bits: their sign indices; the low 9 the grid's */
#define IQ2_XS_GRID 511
AVX512 static INLINE void pair_iq2_xs(const uint8_t *b, int j, Pair *p)
{
  const uint64_t w0 = ld64(b + 2 + 8 * j), w1 = ld64(b + 10 + 8 * j); /* chunks 4 j .. 4 j + 7, a word each */
  uint64_t g[8];
#pragma GCC unroll 4
  for (int k = 0; k < 4; k++)
    g[k] = kIq2xsGrid[(w0 >> (16 * k)) & IQ2_XS_GRID], g[4 + k] = kIq2xsGrid[(w1 >> (16 * k)) & IQ2_XS_GRID];
  signed_grid(p, g, _pext_u64(w0, IQ2_XS_SIGNS) | _pext_u64(w1, IQ2_XS_SIGNS) << 28);
  p->s21 = 2 * _pdep_u32(ld16(b + 66 + j), NIBBLES) + ODD; /* a block's byte of two 4-bit s */
  p->d = f16(ld16(b));
}

/* codes 8 grid +- 1 (the grid's -1 / 0 / 1 bytes; a chunk's delta 1/8 as its +- 1): 8 grid by a shift, the bits
   carried from the byte below cleared. j even: the pair's 8 chunks' high nibbles one word, its 4 halves' 3-bit s at
   8 j + 3 h of the scale words */
#define IQ1_DELTA_BITS 0x88888888u /* each high nibble's bit 3: its chunk's delta negative */
AVX512 static INLINE void pair_iq1_m(const uint8_t *b, int j, Pair *p)
{
  const uint64_t sc = ld64(b + 48), qs = ld64(b + 4 * j);
  const uint32_t qh = ld32(b + 32 + 2 * j);
  uint64_t g[8];
#pragma GCC unroll 8
  for (int k = 0; k < 8; k++)
    g[k] = kIq1sGrid[((qs >> (8 * k)) & 0xFF) | ((qh >> (4 * k)) & 0x07) << 8];
  const __m512i gv = _mm512_set_epi64((long long)g[7], (long long)g[6], (long long)g[5], (long long)g[4],
                                      (long long)g[3], (long long)g[2], (long long)g[1], (long long)g[0]);
  const __m512i g8 = _mm512_and_si512(_mm512_slli_epi64(gv, 3), _mm512_set1_epi8((char)0xF8));
  const __m512i one = _mm512_set1_epi8(1);
  const __m512i w = _mm512_mask_blend_epi64((__mmask8)_pext_u32(qh, IQ1_DELTA_BITS), _mm512_add_epi8(g8, one),
                                            _mm512_sub_epi8(g8, one));
  p->mag = _mm512_abs_epi8(w), p->neg = _mm512_movepi8_mask(w);
  p->s21 = 2 * _pdep_u32((uint32_t)(sc >> (8 * j)), TRIPLES) + ODD;
  const uint32_t s0 = (uint32_t)sc & 0xFFFF, s1 = (uint32_t)(sc >> 16) & 0xFFFF;
  const uint32_t s2 = (uint32_t)(sc >> 32) & 0xFFFF, s3 = (uint32_t)(sc >> 48);
  p->d = f16((s0 >> 12) | ((s1 >> 8) & 0x00F0) | ((s2 >> 4) & 0x0F00) | (s3 & 0xF000));
}

/* IQ2_XS's codes from a grid of 1024 (an index's low 8 bits a byte, its high 2 in qh: 16 bits the pair's 8 chunks)
   and the signs stored: the pair's 8 sign bytes are its negatives' mask, bit 8 i + v chunk i's value v */
#define IQ2_S_TWO_BITS 0x03
AVX512 static INLINE void pair_iq2_s(const uint8_t *b, int j, Pair *p)
{
  const uint64_t qs = ld64(b + 2 + 4 * j);
  const uint32_t qh = ld16(b + 66 + j);
  uint64_t g[8];
#pragma GCC unroll 8
  for (int k = 0; k < 8; k++)
    g[k] = kIq2sGrid[((qs >> (8 * k)) & 0xFF) | ((qh >> (2 * k)) & IQ2_S_TWO_BITS) << 8];
  p->mag = _mm512_set_epi64((long long)g[7], (long long)g[6], (long long)g[5], (long long)g[4], (long long)g[3],
                            (long long)g[2], (long long)g[1], (long long)g[0]);
  p->neg = _cvtu64_mask64(ld64(b + 34 + 4 * j));
  p->s21 = 2 * _pdep_u32(ld16(b + 74 + j), NIBBLES) + ODD; /* a block's byte of two 4-bit s */
  p->d = f16(ld16(b));
}

/* IQ1_M's codes 8 grid +- 1 with a u16 for each block of 32: 3 high index bits a chunk, its one s (bits 12-14) for
   both halves, its one delta sign (bit 15) for its 4 chunks */
#define IQ1_S_SCALE_SHIFT 12
#define IQ1_S_NEG_DELTA 0x8000u
#define IQ1_S_BLOCK_CHUNKS 0x0F /* a block's 4 chunks of the pair's 8, as the blend's qwords */
AVX512 static INLINE void pair_iq1_s(const uint8_t *b, int j, Pair *p)
{
  const uint64_t qs = ld64(b + 2 + 4 * j);
  const uint32_t h0 = ld16(b + 34 + 2 * j), h1 = ld16(b + 36 + 2 * j);
  uint64_t g[8];
#pragma GCC unroll 4
  for (int k = 0; k < 4; k++)
    g[k] = kIq1sGrid[((qs >> (8 * k)) & 0xFF) | ((h0 >> (3 * k)) & 0x07) << 8],
    g[4 + k] = kIq1sGrid[((qs >> (8 * (4 + k))) & 0xFF) | ((h1 >> (3 * k)) & 0x07) << 8];
  const __m512i gv = _mm512_set_epi64((long long)g[7], (long long)g[6], (long long)g[5], (long long)g[4],
                                      (long long)g[3], (long long)g[2], (long long)g[1], (long long)g[0]);
  const __m512i g8 = _mm512_and_si512(_mm512_slli_epi64(gv, 3), _mm512_set1_epi8((char)0xF8));
  const __m512i one = _mm512_set1_epi8(1);
  const __mmask8 neg = (__mmask8)((h0 & IQ1_S_NEG_DELTA ? IQ1_S_BLOCK_CHUNKS : 0) |
                                  (h1 & IQ1_S_NEG_DELTA ? IQ1_S_BLOCK_CHUNKS << 4 : 0));
  const __m512i w = _mm512_mask_blend_epi64(neg, _mm512_add_epi8(g8, one), _mm512_sub_epi8(g8, one));
  p->mag = _mm512_abs_epi8(w), p->neg = _mm512_movepi8_mask(w);
  const uint32_t s0 = (h0 >> IQ1_S_SCALE_SHIFT) & 0x07, s1 = (h1 >> IQ1_S_SCALE_SHIFT) & 0x07;
  p->s21 = 2 * (s0 * 0x0101u | s1 * 0x01010000u) + ODD;
  p->d = f16(ld16(b));
}

/* IQ3_XXS's: a block's 4 chunks each two grid entries (4 unsigned bytes each), signed by kKsigns' 7-bit pattern of
   the block's word (its 4 patterns, then s); both halves 2 (2s + 1) - its quarter as the contract's eighth */
AVX512 static INLINE void pair_iq3_xxs(const uint8_t *b, int j, Pair *p)
{
  uint64_t g[8], neg = 0;
  uint32_t s21 = 0;
  for (int blk = 0; blk < 2; blk++)
  {
    const uint8_t *q = b + 2 + 8 * (j + blk);
    const uint32_t aux = ld32(b + 66 + 4 * (j + blk));
#pragma GCC unroll 4
    for (int k = 0; k < 4; k++)
    {
      g[4 * blk + k] = (uint64_t)kIq3xxsGrid[q[2 * k]] | (uint64_t)kIq3xxsGrid[q[2 * k + 1]] << 32;
      neg |= (uint64_t)kKsigns[(aux >> (7 * k)) & 0x7F] << (8 * (4 * blk + k));
    }
    s21 |= (uint32_t)(2 * (2 * (aux >> 28) + 1)) * 0x0101u << (16 * blk);
  }
  p->mag = _mm512_set_epi64((long long)g[7], (long long)g[6], (long long)g[5], (long long)g[4], (long long)g[3],
                            (long long)g[2], (long long)g[1], (long long)g[0]);
  p->neg = _cvtu64_mask64(neg);
  p->s21 = s21;
  p->d = f16(ld16(b));
}

/* Q2_K's: two blocks' 2-bit codes (the same 32 bytes of qs at shifts 2 (j % 4) and 2 (j % 4) + 2), no signs; each
   half's 8 sc and 8 m from its scale byte (bytes 2j .. 2j + 3: block j's halves, then j + 1's), d and dmin */
#define Q2_K_NIBBLES 0x0F0F0F0Fu
AVX512 static INLINE void pair_q2_k(const uint8_t *b, int j, Pair *p)
{
  const __m256i qs = _mm256_loadu_si256((const __m256i *)(b + 16 + 32 * (j / 4)));
  const __m256i three = _mm256_set1_epi8(3);
  const int sh = 2 * (j % 4);
  const __m256i lo = _mm256_and_si256(_mm256_srl_epi16(qs, _mm_cvtsi32_si128(sh)), three);
  const __m256i hi = _mm256_and_si256(_mm256_srl_epi16(qs, _mm_cvtsi32_si128(sh + 2)), three);
  p->mag = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
  p->neg = 0;
  const uint32_t sc = ld32(b + 2 * j);
  p->s21 = (sc & Q2_K_NIBBLES) << 3, p->m8 = ((sc >> 4) & Q2_K_NIBBLES) << 3;
  p->d = f16(ld16(b + 80)), p->dmin = f16(ld16(b + 82));
}

/* the decoders by type: each row loop its own, the decoder inlined into it; a type with no arm aborts (ROWS_OF names
   only these) */
#define DECODE(type, b, j, p)                                                                                      \
  ((type) == QTYPE_IQ2_XXS  ? pair_iq2_xxs(b, j, p)                                                               \
   : (type) == QTYPE_IQ2_XS ? pair_iq2_xs(b, j, p)                                                                \
   : (type) == QTYPE_IQ1_M  ? pair_iq1_m(b, j, p)                                                                 \
   : (type) == QTYPE_IQ2_S  ? pair_iq2_s(b, j, p)                                                                 \
   : (type) == QTYPE_IQ1_S  ? pair_iq1_s(b, j, p)                                                                 \
   : (type) == QTYPE_IQ3_XXS ? pair_iq3_xxs(b, j, p)                                                              \
   : (type) == QTYPE_Q2_K   ? pair_q2_k(b, j, p)                                                                  \
                            : abort())

/* 4 vectors of 16 int32 to one: lane L's 4 of input i summed into its element i (each 128-bit lane a half-block) */
AVX512 static __m512i hsum4(__m512i a, __m512i b, __m512i c, __m512i d)
{
  const __m512i ab = _mm512_add_epi32(_mm512_unpacklo_epi32(a, b), _mm512_unpackhi_epi32(a, b));
  const __m512i cd = _mm512_add_epi32(_mm512_unpacklo_epi32(c, d), _mm512_unpackhi_epi32(c, d));
  return _mm512_add_epi32(_mm512_unpacklo_epi64(ab, cd), _mm512_unpackhi_epi64(ab, cd));
}

#define SEL_HALF0 0x88 /* _mm512_shuffle_i32x4: 128-bit lanes 0, 2 of a, then of b - the blocks' first halves */
#define SEL_HALF1 0xDD /* lanes 1, 3: their second halves */

/* a group (n 8 or 16 blocks from block s0 of the row) times tokens' x into acc[t] (float lane e: block perm(e)) */
AVX512 static INLINE void group(uint32_t type, const uint8_t *row, uint32_t bytes, uint32_t s0, int n, const int8_t *q,
                         const float *d, uint32_t cols, uint32_t m, __m512 *acc)
{
  const uint32_t nb = cols / LANE_Q8_VALUES;
  __m512i mag[PAIRS];
  __mmask64 neg[PAIRS];
  uint32_t s21[PAIRS] = {0}, m8[PAIRS] = {0};
  float sd[2] = {0}; /* the group's two 256-value blocks' d (float lanes 0 .. 7 the first's blocks, 8 .. 15 the next) */
  float smin[2] = {0}; /* Q2_K's dmin, likewise */
#pragma GCC unroll 8
  for (int k = 0; k < n / 2; k++)
  {
    const uint32_t s = s0 + 2 * (uint32_t)k;
    Pair p;
    DECODE(type, row + (s / PER) * bytes, (int)(s % PER), &p);
    mag[k] = p.mag, neg[k] = p.neg, s21[k] = p.s21, sd[k / (PER / 2)] = p.d;
    m8[k] = p.m8, smin[k / (PER / 2)] = p.dmin;
  }
  /* each float lane's d (2s + 1) / 8 for its block's two halves, in lane.h's order: (d (2s + 1)) / 8 */
  const __m512i by = _mm512_permutexvar_epi8(
      SETR(SCALE_BYTES, 0, 0, 0, 0, 0, 0, 0, 0),
      _mm512_castsi256_si512(_mm256_setr_epi32((int)s21[0], (int)s21[1], (int)s21[2], (int)s21[3], (int)s21[4],
                                               (int)s21[5], (int)s21[6], (int)s21[7])));
  const __m512 eighth = _mm512_set1_ps(LANE_Q8_EIGHTH);
  const __m512 dvv = _mm512_mask_blend_ps((__mmask16)~HALF_GROUP, _mm512_set1_ps(sd[0]), _mm512_set1_ps(sd[1]));
  const __m512 s0f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_castsi512_si128(by)));
  const __m512 s1f = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(by, 1)));
  const __m512 ds0 = _mm512_mul_ps(_mm512_mul_ps(dvv, s0f), eighth);
  const __m512 ds1 = _mm512_mul_ps(_mm512_mul_ps(dvv, s1f), eighth);
  /* Q2_K's minimum: each half's (dmin 8 m) / 8 likewise, against x's sum (the contract's lane_q8_part_min) */
  __m512 dm0 = _mm512_setzero_ps(), dm1 = _mm512_setzero_ps();
  if (type == QTYPE_Q2_K)
  {
    const __m512i bm = _mm512_permutexvar_epi8(
        SETR(SCALE_BYTES, 0, 0, 0, 0, 0, 0, 0, 0),
        _mm512_castsi256_si512(_mm256_setr_epi32((int)m8[0], (int)m8[1], (int)m8[2], (int)m8[3], (int)m8[4],
                                                 (int)m8[5], (int)m8[6], (int)m8[7])));
    const __m512 mvv = _mm512_mask_blend_ps((__mmask16)~HALF_GROUP, _mm512_set1_ps(smin[0]), _mm512_set1_ps(smin[1]));
    dm0 = _mm512_mul_ps(_mm512_mul_ps(mvv, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_castsi512_si128(bm)))), eighth);
    dm1 = _mm512_mul_ps(_mm512_mul_ps(mvv, _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(bm, 1)))),
                        eighth);
  }
  const __m512i perm = SETR(PERM);
  const __mmask16 live = n == GROUP ? (__mmask16)0xFFFF : (__mmask16)HALF_GROUP;
  const __m512i zero = _mm512_setzero_si512();
  for (uint32_t t = 0; t < m; t++)
  {
    const int8_t *x = q + (uint64_t)t * cols + (uint64_t)s0 * LANE_Q8_VALUES;
    __m512i dot[PAIRS];
#pragma GCC unroll 8
    for (int k = 0; k < PAIRS; k++)
    {
      if (k < n / 2)
      {
        const __m512i xv = _mm512_loadu_si512(x + 64 * k);
        dot[k] = _mm512_dpbusd_epi32(zero, mag[k], _mm512_mask_sub_epi8(xv, neg[k], zero, xv));
      }
      else
        dot[k] = zero;
    }
    const __m512i lo = hsum4(dot[0], dot[1], dot[2], dot[3]), hi = hsum4(dot[4], dot[5], dot[6], dot[7]);
    __m512 p0 = _mm512_mul_ps(ds0, _mm512_cvtepi32_ps(_mm512_shuffle_i32x4(lo, hi, SEL_HALF0)));
    __m512 p1 = _mm512_mul_ps(ds1, _mm512_cvtepi32_ps(_mm512_shuffle_i32x4(lo, hi, SEL_HALF1)));
    if (type == QTYPE_Q2_K) /* each half's x sum (codes all 1: vpdpbusd's), its minimum's part subtracted */
    {
      const __m512i ones = _mm512_set1_epi8(1);
      __m512i sx[PAIRS];
#pragma GCC unroll 8
      for (int k = 0; k < PAIRS; k++)
        sx[k] = k < n / 2 ? _mm512_dpbusd_epi32(zero, ones, _mm512_loadu_si512(x + 64 * k)) : zero;
      const __m512i slo = hsum4(sx[0], sx[1], sx[2], sx[3]), shi = hsum4(sx[4], sx[5], sx[6], sx[7]);
      p0 = _mm512_sub_ps(p0, _mm512_mul_ps(dm0, _mm512_cvtepi32_ps(_mm512_shuffle_i32x4(slo, shi, SEL_HALF0))));
      p1 = _mm512_sub_ps(p1, _mm512_mul_ps(dm1, _mm512_cvtepi32_ps(_mm512_shuffle_i32x4(slo, shi, SEL_HALF1))));
    }
    const __m512 xd = _mm512_permutexvar_ps(perm, _mm512_maskz_loadu_ps(live, d + (uint64_t)t * nb + s0));
    acc[t] = _mm512_mask3_fmadd_ps(xd, _mm512_add_ps(p0, p1), acc[t], live);
  }
}

/* the lanes' sums (float lane e of acc[v] lane 16 v + perm(e)) by the GPU's xor tree */
AVX512 static float tree(const __m512 *acc, int lanes)
{
  const __m512i inv = SETR(PERM_INV);
  float s[2 * GROUP];
  for (int v = 0; v < lanes / GROUP; v++)
    _mm512_storeu_ps(s + GROUP * v, _mm512_permutexvar_ps(inv, acc[v]));
  for (int o = lanes / 2; o > 0; o >>= 1) /* lane l's sum and its partner's: for the lanes below o, s[l] + s[l + o] */
    for (int l = 0; l < o; l++)
      s[l] = s[l] + s[l + o];
  return s[0];
}

#define LANE_GROUPS 2 /* LANE_Q8_LANES' widest, in groups */

AVX512 static INLINE void rows(uint32_t type, uint32_t bytes, const uint8_t *w, uint32_t cols, uint32_t r0, uint32_t r1,
                        const int8_t *q, const float *d, uint32_t m, float *y, uint32_t ys)
{
  const uint32_t nb = cols / LANE_Q8_VALUES, lanes = LANE_Q8_LANES(cols), row_bytes = (cols / QK) * bytes;
  for (uint32_t r = r0; r < r1; r++)
  {
    __m512 acc[LANE_GROUPS][CPU_Q8_TOKENS];
    for (int v = 0; v < LANE_GROUPS; v++)
      for (int t = 0; t < CPU_Q8_TOKENS; t++)
        acc[v][t] = _mm512_setzero_ps();
    for (uint32_t s = 0; s < nb; s += GROUP) /* a group's blocks lanes (s % lanes) .. + 15 */
    {
      __m512 *a = acc[(s % lanes) / GROUP];
      if (nb - s >= GROUP) /* the block count a constant: the group's loops unrolled, its codes in registers */
        group(type, w + (uint64_t)r * row_bytes, bytes, s, GROUP, q, d, cols, m, a);
      else
        group(type, w + (uint64_t)r * row_bytes, bytes, s, GROUP / 2, q, d, cols, m, a);
    }
    for (uint32_t t = 0; t < m; t++)
    {
      __m512 a[LANE_GROUPS] = {acc[0][t], acc[1][t]};
      y[(uint64_t)t * ys + r] = tree(a, (int)lanes);
    }
  }
}

#define ROWS_OF(name, type)                                                                                      \
  AVX512 static void name(const uint8_t *w, uint32_t cols, uint32_t r0, uint32_t r1, const int8_t *q, const float *d, \
                          uint32_t m, float *y, uint32_t ys)                                                       \
  {                                                                                                                \
    rows(QTYPE_##type, QTYPE_##type##_BYTES, w, cols, r0, r1, q, d, m, y, ys);                                    \
  }
ROWS_OF(rows_iq2_xxs, IQ2_XXS)
ROWS_OF(rows_iq2_xs, IQ2_XS)
ROWS_OF(rows_iq1_m, IQ1_M)
ROWS_OF(rows_iq2_s, IQ2_S)
ROWS_OF(rows_iq1_s, IQ1_S)
ROWS_OF(rows_iq3_xxs, IQ3_XXS)
ROWS_OF(rows_q2_k, Q2_K)

typedef void (*rows_fn)(const uint8_t *w, uint32_t cols, uint32_t r0, uint32_t r1, const int8_t *q, const float *d,
                        uint32_t m, float *y, uint32_t ys);

bool cpu_q8_ready(void)
{
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
         __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vbmi") &&
         __builtin_cpu_supports("bmi2");
}

bool cpu_q8_rows(uint32_t type, const void *w, uint32_t cols, uint32_t r0, uint32_t r1, const int8_t *q, const float *d,
                 uint32_t n, float *y, uint32_t ys)
{
  rows_fn fn;
  switch (type)
  {
  case QTYPE_IQ2_XXS: fn = rows_iq2_xxs; break;
  case QTYPE_IQ2_XS: fn = rows_iq2_xs; break;
  case QTYPE_IQ1_M: fn = rows_iq1_m; break;
  case QTYPE_IQ2_S: fn = rows_iq2_s; break;
  case QTYPE_IQ1_S: fn = rows_iq1_s; break;
  case QTYPE_IQ3_XXS: fn = rows_iq3_xxs; break;
  case QTYPE_Q2_K: fn = rows_q2_k; break;
  default: return false;
  }
  if (!w || !q || !d || !y || !n || !cols || cols % QK || r1 < r0 || !cpu_q8_ready())
    return false;
  for (uint32_t t0 = 0; t0 < n; t0 += CPU_Q8_TOKENS) /* each token's sums its own: any grouping the same bits */
  {
    const uint32_t m = n - t0 < CPU_Q8_TOKENS ? n - t0 : CPU_Q8_TOKENS, nb = cols / LANE_Q8_VALUES;
    fn((const uint8_t *)w, cols, r0, r1, q + (uint64_t)t0 * cols, d + (uint64_t)t0 * nb, m,
         y + (uint64_t)t0 * ys, ys);
  }
  return true;
}

/* a block a time: its largest magnitude (max is exact in any order), x times 127 / it rounded to even, as rintf */
AVX512 static void round_rows(const float *x, uint32_t rows, uint32_t cols, uint32_t xs, int8_t *q, float *d)
{
  const uint32_t nb = cols / LANE_Q8_VALUES;
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t s = 0; s < nb; s++)
    {
      const float *xb = x + (uint64_t)r * xs + (uint64_t)s * LANE_Q8_VALUES;
      const __m512 a = _mm512_loadu_ps(xb), b = _mm512_loadu_ps(xb + GROUP);
      const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(a), _mm512_abs_ps(b)));
      const __m512 inv = _mm512_set1_ps(lane_q8_inv(amax));
      const int rn = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
      int8_t *qb = q + (uint64_t)r * cols + (uint64_t)s * LANE_Q8_VALUES;
      _mm_storeu_si128((__m128i *)qb, _mm512_cvtepi32_epi8(_mm512_cvt_roundps_epi32(_mm512_mul_ps(a, inv), rn)));
      _mm_storeu_si128((__m128i *)(qb + GROUP),
                       _mm512_cvtepi32_epi8(_mm512_cvt_roundps_epi32(_mm512_mul_ps(b, inv), rn)));
      d[(uint64_t)r * nb + s] = lane_q8_scale(amax);
    }
}

bool cpu_q8_round(const float *x, uint32_t rows, uint32_t cols, uint32_t xs, int8_t *q, float *d)
{
  if (!x || !q || !d || !rows || !cols || cols % LANE_Q8_VALUES || xs < cols || !cpu_q8_ready())
    return false;
  round_rows(x, rows, cols, xs, q, d);
  return true;
}
