#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "substrate/cpu/expert.h"

/* the CPU lane's matvec on its own (test_cpu_expert_gpu holds it to the GPU's bits): synthetic blocks of each expert
   type, the weights read back through unit vectors (a row times e_c is its value: one product, the rest zeros), then
   random x against the exact sum within the order's rounding, rows split and tokens apart the same bits; refusals */

#define ROWS 40
#define COLS 512
#define TOK 3
#define F16_HALF 0x3800 /* 0.5 as f16 bits: every block's d */

static uint32_t rng = 1017;
static uint32_t next(void) { return rng = rng * 1664525u + 1013904223u; }
static float uniform(void) { return (float)(next() >> 8) / (float)(1u << 23) - 1.0f; }

static void fill(uint32_t type, uint8_t *w, size_t bytes, size_t block)
{
  for (size_t i = 0; i < bytes; i++)
    w[i] = (uint8_t)(next() >> 13);
  for (size_t b = 0; b < bytes; b += block)
    if (type == 29) /* IQ1_M: d's nibbles the top of the four scale words */
    {
      uint8_t *sc = w + b + 48;
      for (int k = 0; k < 4; k++)
        sc[2 * k + 1] = (uint8_t)((sc[2 * k + 1] & 0x0F) | (((F16_HALF >> (4 * k)) & 0x0F) << 4));
    }
    else
      w[b] = (uint8_t)F16_HALF, w[b + 1] = (uint8_t)(F16_HALF >> 8);
}

static void check(uint32_t type, size_t block)
{
  const size_t bytes = (size_t)ROWS * COLS / 256 * block;
  uint8_t *w = malloc(bytes);
  float *wv = malloc((size_t)ROWS * COLS * sizeof *wv), *e = calloc(COLS, sizeof *e), col[ROWS];
  float *x = malloc((size_t)TOK * COLS * sizeof *x), y[TOK * ROWS], part[TOK * ROWS];
  assert(w && wv && e && x);
  fill(type, w, bytes, block);
  for (uint32_t c = 0; c < COLS; c++) /* the weights, a column at a time */
  {
    e[c] = 1.0f;
    assert(cpu_mv_rows(type, w, COLS, 0, ROWS, e, COLS, 1, col, ROWS));
    for (uint32_t r = 0; r < ROWS; r++)
      wv[(size_t)r * COLS + c] = col[r];
    e[c] = 0.0f;
  }
  for (size_t i = 0; i < (size_t)TOK * COLS; i++)
    x[i] = uniform();
  assert(cpu_mv_rows(type, w, COLS, 0, ROWS, x, COLS, TOK, y, ROWS));
  const double u = ldexp(1.0, -24), depth = COLS / CPU_LANES + 5 + 2; /* a lane's chain, then the tree */
  for (uint32_t t = 0; t < TOK; t++)
    for (uint32_t r = 0; r < ROWS; r++)
    {
      double s = 0, mag = 0;
      for (uint32_t c = 0; c < COLS; c++)
        s += (double)wv[(size_t)r * COLS + c] * x[(size_t)t * COLS + c],
            mag += fabs((double)wv[(size_t)r * COLS + c] * x[(size_t)t * COLS + c]);
      assert(isfinite(y[t * ROWS + r]) && fabs(y[t * ROWS + r] - s) <= depth * u * mag);
    }
  for (uint32_t t = 0; t < TOK; t++) /* rows in two parts, each token alone */
  {
    assert(cpu_mv_rows(type, w, COLS, 0, 17, x + (size_t)t * COLS, COLS, 1, part + t * ROWS, ROWS));
    assert(cpu_mv_rows(type, w, COLS, 17, ROWS, x + (size_t)t * COLS, COLS, 1, part + t * ROWS, ROWS));
  }
  assert(memcmp(part, y, sizeof y) == 0);
  free(w), free(wv), free(e), free(x);
  printf("  type %u: %d x %d, %d tokens within the order's rounding of the exact sums; split rows, tokens apart the "
         "same bits\n", type, ROWS, COLS, TOK);
}

int main(void)
{
  printf("[test_cpu_expert] start\n");
  float x[COLS] = {0}, y[ROWS];
  assert(!cpu_mv_known(13) && !cpu_mv_rows(13, x, 256, 0, 1, x, 256, 1, y, 1)); /* Q5_K: no CPU decode */
  assert(!cpu_mv_rows(16, x, 255, 0, 1, x, 256, 1, y, 1) && !cpu_mv_rows(16, x, 256, 2, 1, x, 256, 1, y, 1));
  assert(!cpu_mv_rows(16, NULL, 256, 0, 1, x, 256, 1, y, 1) && !cpu_mv_rows(16, x, 512, 0, 1, x, 256, 1, y, 1));
  check(16, 66); /* IQ2_XXS */
  check(17, 74); /* IQ2_XS */
  check(29, 56); /* IQ1_M */
  check(19, 50); /* IQ1_S */
  check(22, 82); /* IQ2_S */
  printf("[test_cpu_expert] ok\n");
  return 0;
}
