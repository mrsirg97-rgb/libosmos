#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "quant_grids.h"
#include "substrate/cpu/expert.h"

/* kernels/quant.h's decoders for the experts' types, line for line, 8 values a vector: each product a float product,
   each rounded (this file is built -ffp-contract=off: none fused), so each value the GPU's bits */

#define QK 256            /* a block's values */
#define IQ2_XXS_BYTES 66  /* ggml's block sizes */
#define IQ2_XS_BYTES 74
#define IQ1_M_BYTES 56
#define IQ1_DELTA 0.125f  /* IQ1's grid offset (gguf-py's IQ1_S.delta) */
#define GGUF_IQ2_XXS 16   /* ggml's type ids */
#define GGUF_IQ2_XS 17
#define GGUF_IQ1_M 29

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

#define CPU_MAX_TOKENS 8 /* a row's tokens at once (the GPU's mv_group_tok's widest entry) */
#define VL 8             /* AVX2's floats: 8 lanes of the 32 at once */
#define QV (CPU_LANES / VL)

/* the decoders' 8 values as one vector: an integer's float exact, then kernels/quant.h's products in its order (a
   negation a sign bit's flip, as -x) */
static inline __m256 signed_vals(__m256 v, uint32_t signs)
{
  const __m256i bit = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
  const __m256i neg = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32((int)signs), bit), bit);
  return _mm256_xor_ps(v, _mm256_castsi256_ps(_mm256_and_si256(neg, _mm256_set1_epi32((int)0x80000000u))));
}

static inline __m256 u8s(uint64_t grid) /* a grid's 8 unsigned bytes as floats */
{
  return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)grid)));
}

/* (d * (0.5 + s) * 0.25) * grid * sign; 8 groups of 32, each two words: 4 grid bytes, then 4 7-bit signs and s */
static inline __m256 v_iq2_xxs(const uint8_t *b, int k)
{
  const uint8_t *q = b + 2 + (k / 4) * 8;
  const uint32_t aux = ld32(q + 4);
  const float db = f16(ld16(b)) * (0.5f + (float)(aux >> 28)) * 0.25f;
  const __m256 x = _mm256_mul_ps(_mm256_set1_ps(db), u8s(kIq2xxsGrid[q[k % 4]]));
  return signed_vals(x, kKsigns[(aux >> (7 * (k % 4))) & 0x7F]);
}

/* as IQ2_XXS, a 16-bit word a chunk: 9 grid bits, 7 sign bits; a 4-bit scale every two chunks */
static inline __m256 v_iq2_xs(const uint8_t *b, int k)
{
  const uint32_t q = ld16(b + 2 + 2 * k);
  const int s = (b[66 + k / 4] >> (4 * ((k / 2) % 2))) & 0x0F;
  const float db = f16(ld16(b)) * (0.5f + (float)s) * 0.25f;
  return signed_vals(_mm256_mul_ps(_mm256_set1_ps(db), u8s(kIq2xsGrid[q & 511])), kKsigns[q >> 9]);
}

/* (d * (2s + 1)) * (grid + delta); d an f16 in the top nibbles of the four scale words, a 3-bit s each 16 values */
static inline __m256 v_iq1_m(const uint8_t *b, int k)
{
  const uint8_t *qs = b, *qh = b + 32;
  const uint64_t sc = ld64(b + 48);
  const uint32_t s0 = (uint32_t)sc & 0xFFFF, s1 = (uint32_t)(sc >> 16) & 0xFFFF;
  const uint32_t s2 = (uint32_t)(sc >> 32) & 0xFFFF, s3 = (uint32_t)(sc >> 48);
  const float d = f16((s0 >> 12) | ((s1 >> 8) & 0x00F0) | ((s2 >> 4) & 0x0F00) | (s3 & 0xF000));
  const int s = (int)((sc >> (16 * (k / 8) + 3 * ((k / 2) % 4))) & 0x07);
  const float dl = d * (float)(2 * s + 1);
  const uint32_t h = (qh[k / 2] >> (4 * (k % 2))) & 0x0F;
  const float delta = h & 0x08 ? -IQ1_DELTA : IQ1_DELTA;
  const uint64_t grid = kIq1sGrid[qs[k] | (h & 0x07) << 8];
  const __m256 g = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_cvtsi64_si128((long long)grid)));
  return _mm256_mul_ps(_mm256_set1_ps(dl), _mm256_add_ps(g, _mm256_set1_ps(delta)));
}

/* r[i] (chunk i's values j) to v[j] (value j of chunks 0 .. 7) */
static inline void transpose8(__m256 r[VL], __m256 v[VL])
{
  __m256 t[VL], u[VL];
  for (int i = 0; i < VL; i += 2)
    t[i] = _mm256_unpacklo_ps(r[i], r[i + 1]), t[i + 1] = _mm256_unpackhi_ps(r[i], r[i + 1]);
  for (int i = 0; i < VL; i += 4)
  {
    u[i] = _mm256_shuffle_ps(t[i], t[i + 2], 0x44), u[i + 1] = _mm256_shuffle_ps(t[i], t[i + 2], 0xEE);
    u[i + 2] = _mm256_shuffle_ps(t[i + 1], t[i + 3], 0x44), u[i + 3] = _mm256_shuffle_ps(t[i + 1], t[i + 3], 0xEE);
  }
  for (int j = 0; j < 4; j++)
    v[j] = _mm256_permute2f128_ps(u[j], u[j + 4], 0x20), v[j + 4] = _mm256_permute2f128_ps(u[j], u[j + 4], 0x31);
}

/* a row: lanes 8 at a time - chunk c's lane c % 32, its 8 values' fmaf in order on that lane's sum, as the scalar
   chain; xt the tokens' x permuted to match (xt[t][c / 8][j][c % 8] = x[t][8 c + j]); then each token's 32 sums by
   the tree */
#define ROW(name, vdec)                                                                                               \
  static void name(const uint8_t *base, uint32_t bytes, uint32_t chunks, const float *xt, uint32_t n, float *out,   \
                   uint32_t ys)                                                                                     \
  {                                                                                                                 \
    const int per = QK / CPU_CHUNK;                                                                                 \
    __m256 acc[CPU_MAX_TOKENS][QV];                                                                                 \
    for (uint32_t t = 0; t < n; t++)                                                                                \
      for (int q = 0; q < QV; q++)                                                                                  \
        acc[t][q] = _mm256_setzero_ps();                                                                            \
    for (uint32_t c0 = 0; c0 < chunks; c0 += CPU_LANES)                                                             \
      for (int q = 0; q < QV; q++)                                                                                  \
      {                                                                                                             \
        __m256 r[VL], v[VL];                                                                                        \
        for (int i = 0; i < VL; i++)                                                                                \
        {                                                                                                           \
          const uint32_t c = c0 + (uint32_t)(q * VL + i);                                                           \
          r[i] = vdec(base + (c / per) * bytes, (int)(c % per));                                                    \
        }                                                                                                           \
        transpose8(r, v);                                                                                           \
        const uint32_t g = (c0 + (uint32_t)q * VL) / VL;                                                            \
        for (uint32_t t = 0; t < n; t++)                                                                            \
        {                                                                                                           \
          const float *xg = xt + ((uint64_t)t * chunks / VL + g) * CPU_CHUNK * VL;                                  \
          __m256 a = acc[t][q];                                                                                     \
          for (int j = 0; j < CPU_CHUNK; j++)                                                                       \
            a = _mm256_fmadd_ps(v[j], _mm256_loadu_ps(xg + j * VL), a);                                             \
          acc[t][q] = a;                                                                                            \
        }                                                                                                           \
      }                                                                                                             \
    for (uint32_t t = 0; t < n; t++)                                                                                \
      out[(uint64_t)t * ys] = tree(acc[t]);                                                                         \
  }

/* the lanes' sums by the GPU's xor tree: each lane its sum and its partner's (a + b = b + a) */
static float tree(const __m256 acc[QV])
{
  float s[CPU_LANES], nx[CPU_LANES];
  for (int q = 0; q < QV; q++)
    _mm256_storeu_ps(s + q * VL, acc[q]);
  for (uint32_t o = CPU_LANES / 2; o > 0; o >>= 1)
  {
    for (uint32_t l = 0; l < CPU_LANES; l++)
      nx[l] = s[l] + s[l ^ o];
    memcpy(s, nx, sizeof nx);
  }
  return s[0];
}

ROW(row_iq2_xxs, v_iq2_xxs)
ROW(row_iq2_xs, v_iq2_xs)
ROW(row_iq1_m, v_iq1_m)

typedef void (*row_fn)(const uint8_t *base, uint32_t bytes, uint32_t chunks, const float *xt, uint32_t n, float *out,
                       uint32_t ys);

static bool kind(uint32_t type, row_fn *fn, uint32_t *bytes)
{
  switch (type)
  {
  case GGUF_IQ2_XXS: *fn = row_iq2_xxs, *bytes = IQ2_XXS_BYTES; return true;
  case GGUF_IQ2_XS: *fn = row_iq2_xs, *bytes = IQ2_XS_BYTES; return true;
  case GGUF_IQ1_M: *fn = row_iq1_m, *bytes = IQ1_M_BYTES; return true;
  default: return false;
  }
}

bool cpu_mv_known(uint32_t type)
{
  row_fn fn;
  uint32_t bytes;
  return kind(type, &fn, &bytes);
}

bool cpu_mv_rows(uint32_t type, const void *w, uint32_t cols, uint32_t r0, uint32_t r1, const float *x, uint32_t xs,
                 uint32_t n, float *y, uint32_t ys)
{
  row_fn fn;
  uint32_t bytes;
  if (!w || !x || !y || !kind(type, &fn, &bytes) || !cols || cols % QK || r1 < r0 || xs < cols)
    return false;
  const uint32_t chunks = cols / CPU_CHUNK; /* a multiple of 32: cols of 256s */
  float *xt = malloc((size_t)CPU_MAX_TOKENS * cols * sizeof *xt);
  if (!xt)
    return false;
  const uint64_t row_bytes = (uint64_t)(cols / QK) * bytes;
  for (uint32_t t0 = 0; t0 < n; t0 += CPU_MAX_TOKENS) /* each token's sums its own: any grouping the same bits */
  {
    const uint32_t m = n - t0 < CPU_MAX_TOKENS ? n - t0 : CPU_MAX_TOKENS;
    for (uint32_t t = 0; t < m; t++)
      for (uint32_t c = 0; c < chunks; c++)
        for (int j = 0; j < CPU_CHUNK; j++)
          xt[((uint64_t)t * chunks / VL + c / VL) * CPU_CHUNK * VL + (uint64_t)j * VL + c % VL] =
              x[(uint64_t)(t0 + t) * xs + (uint64_t)c * CPU_CHUNK + j];
    for (uint32_t r = r0; r < r1; r++)
    {
      const uint8_t *base = (const uint8_t *)w + r * row_bytes;
      float *out = y + (uint64_t)t0 * ys + r;
      fn(base, bytes, chunks, xt, m, out, ys);
    }
  }
  free(xt);
  return true;
}
