#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lane.h"
#include "quant_grids.h"
#include "substrate/cpu/q8.h"

/* the CPU lane's int8 experts against a scalar replay of kernels/lane.h's contract (test_cpu_q8_gpu holds them to the
   GPU's bits on the model's experts): random blocks of each expert type at 16 and 32 lanes and a row whose last
   blocks fill half a vector, rounding's bytes and scales, more tokens than one decode takes, rows split and tokens
   apart the same bits; refusals. this file is built without contraction: the replay's products are each rounded */

#define QK 256
#define ROWS 21
#define TOK 11 /* past CPU_Q8_TOKENS: a second decode's tokens */

static uint32_t rng = 1017;
static uint32_t next(void) { return rng = rng * 1664525u + 1013904223u; }
static float uniform(void) { return (float)(next() >> 8) / (float)(1u << 23) - 1.0f; }
static uint32_t ld16(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8; }
static float f16(uint32_t bits)
{
  const uint16_t b = (uint16_t)bits;
  _Float16 h;
  memcpy(&h, &b, sizeof h);
  return (float)h;
}
static size_t block_bytes(uint32_t type) { return qtype_info(type)->bytes; }

/* random bytes, each block's d a small f16 (the scale words' top nibbles for IQ1_M) */
static void fill(uint32_t type, uint8_t *w, size_t bytes)
{
  const size_t bb = block_bytes(type);
  for (size_t i = 0; i < bytes; i++)
    w[i] = (uint8_t)(next() >> 13);
  for (size_t b = 0; b < bytes; b += bb)
  {
    const uint32_t h = 0x2C00u | (next() >> 22); /* 2^-4 .. 2^-3, a mantissa of its own */
    if (type == QTYPE_IQ1_M)
      for (int k = 0; k < 4; k++)
        w[b + 48 + 2 * k + 1] = (uint8_t)((w[b + 48 + 2 * k + 1] & 0x0F) | ((h >> (4 * k)) & 0x0F) << 4);
    else
      w[b] = (uint8_t)h, w[b + 1] = (uint8_t)(h >> 8);
  }
}

/* block j (32 values) of the 256-value block at b against q: its scale, each half's 2s + 1 and integer dot */
static void halves(uint32_t type, const uint8_t *b, int j, const int8_t *q, float *d, int s21[2], int isum[2])
{
  isum[0] = isum[1] = 0;
  for (int c = 0; c < 4; c++)
  {
    const int k = 4 * j + c, h = c / 2;
    int v[8];
    if (type == QTYPE_IQ1_S) /* gguf-py's IQ1_S: d, qs[32], qh u16[8] (3 index bits an 8, s in 12-14, sign 15) */
    {
      const uint32_t qh = ld16(b + 34 + 2 * (k / 4));
      const uint64_t gr = kIq1sGrid[b[2 + k] | ((qh >> (3 * (k % 4))) & 7) << 8];
      for (int i = 0; i < 8; i++)
        v[i] = 8 * (int)(int8_t)(gr >> (8 * i)) + (qh & 0x8000 ? -1 : 1);
      *d = f16(ld16(b)), s21[h] = 2 * (int)((qh >> 12) & 7) + 1;
    }
    else if (type == QTYPE_IQ1_M)
    {
      uint64_t sc;
      memcpy(&sc, b + 48, 8);
      const uint32_t s0 = (uint32_t)sc & 0xFFFF, s1 = (uint32_t)(sc >> 16) & 0xFFFF;
      const uint32_t s2 = (uint32_t)(sc >> 32) & 0xFFFF, s3 = (uint32_t)(sc >> 48);
      const uint32_t hb = (b[32 + k / 2] >> (4 * (k % 2))) & 0x0F;
      const uint64_t gr = kIq1sGrid[b[k] | (hb & 0x07) << 8];
      for (int i = 0; i < 8; i++)
        v[i] = 8 * (int)(int8_t)(gr >> (8 * i)) + (hb & 0x08 ? -1 : 1);
      *d = f16((s0 >> 12) | ((s1 >> 8) & 0x00F0) | ((s2 >> 4) & 0x0F00) | (s3 & 0xF000));
      s21[h] = 2 * (int)((sc >> (16 * (k / 8) + 3 * ((k / 2) % 4))) & 0x07) + 1;
    }
    else
    {
      uint64_t gr;
      uint8_t sg;
      if (type == QTYPE_IQ2_XXS)
      {
        const uint8_t *qq = b + 2 + 8 * j;
        const uint32_t aux = ld16(qq + 4) | ld16(qq + 6) << 16;
        gr = kIq2xxsGrid[qq[c]], sg = kKsigns[(aux >> (7 * c)) & 0x7F], s21[h] = 2 * (int)(aux >> 28) + 1;
      }
      else if (type == QTYPE_IQ2_XS)
      {
        const uint32_t qq = ld16(b + 2 + 2 * k);
        gr = kIq2xsGrid[qq & 511], sg = kKsigns[qq >> 9];
        s21[h] = 2 * (int)((b[66 + k / 4] >> (4 * ((k / 2) % 2))) & 0x0F) + 1;
      }
      else /* gguf-py's IQ2_S: d, qs[32], signs[32], qh[8] (2 index bits an 8), scales[8] (a nibble a 16) */
      {
        assert(type == QTYPE_IQ2_S);
        gr = kIq2sGrid[b[2 + k] | ((b[66 + k / 4] >> (2 * (k % 4))) & 3) << 8], sg = b[34 + k];
        s21[h] = 2 * (int)((b[74 + k / 4] >> (4 * ((k / 2) % 2))) & 0x0F) + 1;
      }
      for (int i = 0; i < 8; i++)
        v[i] = (sg >> i) & 1 ? -(int)(uint8_t)(gr >> (8 * i)) : (int)(uint8_t)(gr >> (8 * i));
      *d = f16(ld16(b));
    }
    for (int i = 0; i < 8; i++)
      isum[h] += v[i] * q[8 * c + i];
  }
}

static float row_ref(uint32_t type, const uint8_t *row, uint32_t cols, const int8_t *q, const float *qd)
{
  const int L = LANE_Q8_LANES(cols);
  float acc[32] = {0}, nx[32];
  for (uint32_t s = 0; s < cols / LANE_Q8_VALUES; s++)
  {
    float d;
    int s21[2], isum[2];
    halves(type, row + (s / 8) * block_bytes(type), (int)(s % 8), q + s * LANE_Q8_VALUES, &d, s21, isum);
    const float sub = lane_q8_part(d, s21[0], isum[0]) + lane_q8_part(d, s21[1], isum[1]);
    acc[s % L] = fmaf(qd[s], sub, acc[s % L]);
  }
  for (int o = L / 2; o > 0; o >>= 1)
  {
    for (int l = 0; l < L; l++)
      nx[l] = acc[l] + acc[l ^ o];
    memcpy(acc, nx, sizeof nx);
  }
  return acc[0];
}

static void round_ref(const float *x, uint32_t cols, int8_t *q, float *d)
{
  for (uint32_t s = 0; s < cols / LANE_Q8_VALUES; s++)
  {
    float a = 0.0f;
    for (uint32_t i = 0; i < LANE_Q8_VALUES; i++)
      a = fmaxf(a, fabsf(x[s * LANE_Q8_VALUES + i]));
    const float inv = lane_q8_inv(a);
    for (uint32_t i = 0; i < LANE_Q8_VALUES; i++)
      q[s * LANE_Q8_VALUES + i] = (int8_t)lane_q8_round(x[s * LANE_Q8_VALUES + i], inv);
    d[s] = lane_q8_scale(a);
  }
}

static void check(uint32_t type, uint32_t cols)
{
  const uint32_t nb = cols / LANE_Q8_VALUES, xs = cols + 32; /* x's rows apart by more than a row */
  const size_t bytes = (size_t)ROWS * (cols / QK) * block_bytes(type);
  uint8_t *w = malloc(bytes);
  float *x = malloc((size_t)TOK * xs * sizeof *x), *qd = malloc((size_t)TOK * nb * sizeof *qd);
  float *rd = malloc((size_t)TOK * nb * sizeof *rd), *y = malloc((size_t)TOK * ROWS * sizeof *y);
  float *ref = malloc((size_t)TOK * ROWS * sizeof *ref), *part = malloc((size_t)TOK * ROWS * sizeof *part);
  int8_t *q = malloc((size_t)TOK * cols), *rq = malloc((size_t)TOK * cols);
  assert(w && x && qd && rd && y && ref && part && q && rq);
  fill(type, w, bytes);
  for (size_t i = 0; i < (size_t)TOK * xs; i++)
    x[i] = uniform() * 3.0f;
  for (uint32_t i = 0; i < LANE_Q8_VALUES; i++) /* a block of zeros, a block of one value */
    x[i] = 0.0f, x[xs + LANE_Q8_VALUES + i] = i == 5 ? -2.5f : 0.0f;
  assert(cpu_q8_round(x, TOK, cols, xs, q, qd));
  for (uint32_t t = 0; t < TOK; t++)
    round_ref(x + (size_t)t * xs, cols, rq + (size_t)t * cols, rd + (size_t)t * nb);
  assert(memcmp(q, rq, (size_t)TOK * cols) == 0 && memcmp(qd, rd, (size_t)TOK * nb * sizeof *qd) == 0);
  assert(qd[0] == 0.0f && q[cols + LANE_Q8_VALUES + 5] == -127);
  const size_t row_bytes = bytes / ROWS;
  for (uint32_t t = 0; t < TOK; t++)
    for (uint32_t r = 0; r < ROWS; r++)
      ref[t * ROWS + r] = row_ref(type, w + r * row_bytes, cols, q + (size_t)t * cols, qd + (size_t)t * nb);
  assert(cpu_q8_rows(type, w, cols, 0, ROWS, q, qd, TOK, y, ROWS));
  assert(memcmp(y, ref, (size_t)TOK * ROWS * sizeof *y) == 0);
  for (uint32_t t = 0; t < TOK; t++) /* rows in two parts, each token alone */
  {
    assert(cpu_q8_rows(type, w, cols, 0, 8, q + (size_t)t * cols, qd + (size_t)t * nb, 1, part + t * ROWS, ROWS));
    assert(cpu_q8_rows(type, w, cols, 8, ROWS, q + (size_t)t * cols, qd + (size_t)t * nb, 1, part + t * ROWS, ROWS));
  }
  assert(memcmp(part, ref, (size_t)TOK * ROWS * sizeof *part) == 0);
  free(w), free(x), free(qd), free(rd), free(y), free(ref), free(part), free(q), free(rq);
  printf("  type %u: %d x %u (%d lanes), %d tokens: the contract's bits; rounding's bytes; split rows, tokens "
         "apart the same\n", type, ROWS, cols, LANE_Q8_LANES(cols), TOK);
}

int main(void)
{
  printf("[test_cpu_q8] start\n");
  if (!cpu_q8_ready())
  {
    printf("[test_cpu_q8] skipped: no AVX-512 VNNI\n");
    return 0;
  }
  static float x[QK];
  static int8_t q[QK];
  float d[QK / LANE_Q8_VALUES], y[1];
  assert(!cpu_q8_round(x, 1, 48, 48, q, d) && !cpu_q8_round(x, 1, 64, 32, q, d) && !cpu_q8_round(x, 0, 32, 32, q, d));
  assert(!cpu_q8_rows(13, x, QK, 0, 1, q, d, 1, y, 1)); /* Q5_K: not the contract's */
  assert(!cpu_q8_rows(16, x, 128, 0, 1, q, d, 1, y, 1) && !cpu_q8_rows(16, x, QK, 2, 1, q, d, 1, y, 1));
  assert(!cpu_q8_rows(16, x, QK, 0, 1, q, d, 0, y, 1) && !cpu_q8_rows(16, NULL, QK, 0, 1, q, d, 1, y, 1));
  const uint32_t types[] = {QTYPE_IQ2_XXS, QTYPE_IQ2_XS, QTYPE_IQ1_M, QTYPE_IQ1_S, QTYPE_IQ2_S};
  for (size_t i = 0; i < sizeof types / sizeof types[0]; i++)
  {
    check(types[i], 512);  /* 16 lanes, a full vector a step */
    check(types[i], 768);  /* 16 lanes, the last step half a vector (2,304's shape) */
    check(types[i], 5120); /* 32 lanes, two vectors a step */
    check(types[i], 4352); /* 32 lanes, the last step one and a half vectors */
  }
  printf("[test_cpu_q8] ok\n");
  return 0;
}
