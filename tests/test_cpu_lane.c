#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lane.h"
#include "osmos/cpu/lane.h"
#include "osmos/cpu/q8.h"

/* the CPU lane's pool against the same steps done in one thread (cpu_q8_round, cpu_q8_rows, lane_swiglu - their bits
   the GPU's: test_cpu_q8, test_cpu_lane_gpu): random experts of each type, a job of three experts (1, 3 and 2 rows),
   at 1, 3 and 8 workers, two jobs in turn - every output the same bits, the value posted; refusals */

#define QK 256
#define DIM 4352 /* 32 lanes, the last step one and a half vectors */
#define FF 768   /* 16 lanes, the last step half a vector */
#define NE 3
#define ROWS 6
#define LIM 0.6f /* a clamp some activations pass */

static uint32_t rng = 1017;
static uint32_t next(void) { return rng = rng * 1664525u + 1013904223u; }
static float uniform(void) { return (float)(next() >> 8) / (float)(1u << 23) - 1.0f; }
static size_t block_bytes(uint32_t type) { return qtype_info(type)->bytes; }

static void fill(uint32_t type, uint8_t *w, size_t bytes) /* test_cpu_q8's: random bytes, a small d a block */
{
  const size_t bb = block_bytes(type);
  for (size_t i = 0; i < bytes; i++)
    w[i] = (uint8_t)(next() >> 13);
  for (size_t b = 0; b < bytes; b += bb)
  {
    const uint32_t h = 0x2C00u | (next() >> 22);
    if (type == QTYPE_IQ1_M)
      for (int k = 0; k < 4; k++)
        w[b + 48 + 2 * k + 1] = (uint8_t)((w[b + 48 + 2 * k + 1] & 0x0F) | ((h >> (4 * k)) & 0x0F) << 4);
    else if (type == QTYPE_Q2_K) /* d and dmin after the scales and codes */
      w[b + 80] = w[b + 82] = (uint8_t)h, w[b + 81] = w[b + 83] = (uint8_t)(h >> 8);
    else
      w[b] = (uint8_t)h, w[b + 1] = (uint8_t)(h >> 8);
  }
}

/* the job's steps in one thread, each row alone */
static void reference(const CpuLaneJob *j, float *y)
{
  int8_t *q = malloc(DIM), *qh = malloc(FF);
  float d[DIM / LANE_Q8_VALUES], dh[FF / LANE_Q8_VALUES], g[FF], u[FF], h[FF];
  assert(q && qh);
  for (uint32_t k = 0; k < j->n_e; k++)
    for (uint32_t r = j->e[k].row0; r < j->e[k].row0 + j->e[k].cnt; r++)
    {
      assert(cpu_q8_round(j->x + (size_t)r * DIM, 1, DIM, DIM, q, d));
      assert(cpu_q8_rows(j->type_gu, j->e[k].gate, DIM, 0, FF, q, d, 1, g, FF));
      assert(cpu_q8_rows(j->type_gu, j->e[k].up, DIM, 0, FF, q, d, 1, u, FF));
      for (uint32_t i = 0; i < FF; i++)
        h[i] = lane_swiglu(g[i], u[i], j->lim, j->scale[r]);
      assert(cpu_q8_round(h, 1, FF, FF, qh, dh));
      assert(cpu_q8_rows(j->type_down, j->e[k].down, FF, 0, DIM, qh, dh, 1, y + (size_t)r * DIM, DIM));
    }
  free(q), free(qh);
}

static void check(uint32_t type_gu, uint32_t type_down, int threads, const int *cores)
{
  const size_t gu = (size_t)FF * (DIM / QK) * block_bytes(type_gu);
  const size_t dn = (size_t)DIM * (FF / QK) * block_bytes(type_down);
  uint8_t *w = malloc(NE * (2 * gu + dn));
  float *x = malloc((size_t)ROWS * DIM * sizeof *x), *y = malloc((size_t)ROWS * DIM * sizeof *y);
  float *ref = malloc((size_t)ROWS * DIM * sizeof *ref), scale[ROWS];
  assert(w && x && y && ref);
  CpuLaneExpert e[NE];
  const uint32_t cnt[NE] = {1, 3, 2};
  for (uint32_t k = 0, r = 0; k < NE; r += cnt[k], k++)
  {
    uint8_t *base = w + k * (2 * gu + dn);
    fill(type_gu, base, gu), fill(type_gu, base + gu, gu), fill(type_down, base + 2 * gu, dn);
    e[k] = (CpuLaneExpert){base, base + gu, base + 2 * gu, r, cnt[k]};
  }
  CpuLane *c = cpu_lane_create(threads, cores, ROWS, DIM, FF);
  assert(c);
  uint32_t seq = 0;
  for (uint32_t round = 1; round <= 2; round++) /* two jobs in turn, x new each */
  {
    for (size_t i = 0; i < (size_t)ROWS * DIM; i++)
      x[i] = uniform() * 3.0f;
    for (int r = 0; r < ROWS; r++)
      scale[r] = 0.25f + uniform() * uniform();
    memset(y, 0xFF, (size_t)ROWS * DIM * sizeof *y);
    const CpuLaneJob j = {type_gu, type_down, DIM, FF, LIM, x, scale, ROWS, e, NE, y, &seq, round};
    assert(cpu_lane_submit(c, &j) && cpu_lane_wait(c));
    assert(__atomic_load_n(&seq, __ATOMIC_ACQUIRE) == round);
    reference(&j, ref);
    assert(memcmp(y, ref, (size_t)ROWS * DIM * sizeof *y) == 0);
  }
  /* refusals: a type not the contract's, dims not the pool's, an expert's rows past the job's, no outputs */
  CpuLaneJob bad = {13, type_down, DIM, FF, LIM, x, scale, ROWS, e, NE, y, &seq, 9};
  assert(!cpu_lane_submit(c, &bad));
  bad.type_gu = type_gu, bad.dim = DIM + QK;
  assert(!cpu_lane_submit(c, &bad));
  bad.dim = DIM, bad.n_rows = ROWS - 1;
  assert(!cpu_lane_submit(c, &bad));
  bad.n_rows = ROWS, bad.y = NULL;
  assert(!cpu_lane_submit(c, &bad));
  assert(seq == 2 && cpu_lane_wait(c)); /* nothing queued by them */
  cpu_lane_destroy(c);
  free(w), free(x), free(y), free(ref);
  printf("  types %u / %u, %d worker%s%s: two jobs of %d experts (%d rows) the one thread's bits; refusals\n", type_gu,
         type_down, threads, threads == 1 ? "" : "s", cores ? " (pinned)" : "", NE, ROWS);
}

int main(void)
{
  printf("[test_cpu_lane] start\n");
  if (!cpu_q8_ready())
  {
    printf("[test_cpu_lane] skipped: no AVX-512 VNNI\n");
    return 0;
  }
  assert(!cpu_lane_create(0, NULL, ROWS, DIM, FF) && !cpu_lane_create(2, NULL, ROWS, DIM + 1, FF));
  assert(!cpu_lane_create(2, NULL, 0, DIM, FF));
  const int no_core[2] = {0, 1 << 20}; /* a CPU no machine has: refused, never run unpinned */
  assert(!cpu_lane_create(2, no_core, ROWS, DIM, FF));
  cpu_lane_destroy(NULL);
  check(QTYPE_IQ2_XXS, QTYPE_IQ1_M, 1, NULL);
  check(QTYPE_IQ2_XS, QTYPE_IQ2_XXS, 3, NULL);
  check(QTYPE_IQ1_M, QTYPE_IQ2_XS, 8, NULL);
  check(QTYPE_IQ3_XXS, QTYPE_Q2_K, 4, NULL); /* UD-IQ2_XXS's down types */
  const int four_four[8] = {0, 1, 2, 3, 8, 9, 10, 11}; /* 4 + 4: each CCD's first four (Strix Halo's 2 x 8) */
  check(QTYPE_IQ1_M, QTYPE_IQ2_XS, 8, four_four);
  printf("  a core no machine has: refused\n");
  printf("[test_cpu_lane] ok\n");
  return 0;
}
