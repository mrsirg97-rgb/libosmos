/* a Hy4 read's routing (hy4_read_route) against the loop it replaced (an expert at a time over every pick: 0.12.2's
   route): the same tables for random picks - repeated experts in a token, experts no token picked, runs past a
   chunk - at Hy4-flash's 512 experts; a pick out of range refused; and its time at a 4,096-token read (8 picks a
   token) under kBoundMs - 0.12.2's loop took ~10 ms a layer there, the read's host stall between the router and the
   experts */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "osmos/hy4/route.h"

#define E 512
#define LAT 1536
#define FFE 1536
#define TILE 16
static const double kBoundMs = 2.0;

static uint64_t g_rng = 1017;
static uint32_t draw(uint32_t m)
{
  g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(g_rng >> 33) % m;
}

typedef struct Tabs
{
  Hy4ReadRoute r;
  void *mem;
} Tabs;

static Tabs tabs(uint32_t n, uint32_t k)
{
  const uint32_t picks = n * k, ecap = picks + E, scap = picks + E * (TILE - 1);
  Tabs t;
  char *p = t.mem = calloc(1, (size_t)picks * 12 + (size_t)ecap * 24 + (size_t)scap * 8 + (E + 1) * 4);
  assert(p);
  t.r.gidx = (int32_t *)p, p += picks * 4;
  t.r.sidx = (int32_t *)p, p += picks * 4;
  t.r.wsel = (float *)p, p += picks * 4;
  t.r.cnt = (int32_t *)p, p += ecap * 4;
  t.r.expert = (uint32_t *)p, p += ecap * 4;
  t.r.x = (uint32_t *)p, p += ecap * 4;
  t.r.y = (uint32_t *)p, p += ecap * 4;
  t.r.xd = (uint32_t *)p, p += ecap * 4;
  t.r.yd = (uint32_t *)p, p += ecap * 4;
  t.r.rgidx = (int32_t *)p, p += scap * 4;
  t.r.rhidx = (int32_t *)p, p += scap * 4;
  t.r.estart = (uint32_t *)p;
  return t;
}

/* 0.12.2's route (hip_hy4_moe.c), its tables alone */
static void reference(const int32_t *idx, const float *w, uint32_t n, uint32_t k, uint32_t chunk, Hy4ReadRoute *r)
{
  uint32_t q = 0, e_n = 0, s_n = 0;
  for (uint32_t e = 0; e < E; e++)
  {
    const uint32_t first = q;
    for (uint32_t t = 0; t < n; t++)
      for (uint32_t s = 0; s < k; s++)
        if ((uint32_t)idx[t * k + s] == e)
          r->gidx[q] = (int32_t)t, r->sidx[t * k + s] = (int32_t)q, r->wsel[q] = w[t * k + s], q++;
    if (q == first)
      continue;
    for (uint32_t c = first; c < q; c += chunk, e_n++)
    {
      const uint32_t cnt = q - c < chunk ? q - c : chunk, slot = s_n + (c - first);
      r->cnt[e_n] = (int32_t)cnt, r->expert[e_n] = e;
      r->x[e_n] = slot * LAT, r->y[e_n] = c * FFE, r->xd[e_n] = slot * FFE, r->yd[e_n] = c * LAT;
    }
    for (uint32_t j = first; j < q; j++)
      r->rgidx[s_n + (j - first)] = r->gidx[j], r->rhidx[s_n + (j - first)] = (int32_t)j;
    const uint32_t padded = (q - first + TILE - 1) / TILE * TILE;
    for (uint32_t s = s_n + (q - first); s < s_n + padded; s++)
      r->rgidx[s] = r->rhidx[s] = -1;
    s_n += padded;
  }
  r->entries = e_n, r->slots = s_n;
}

static void same(const Hy4ReadRoute *a, const Hy4ReadRoute *b, uint32_t picks)
{
  assert(a->entries == b->entries && a->slots == b->slots);
  assert(!memcmp(a->gidx, b->gidx, picks * 4) && !memcmp(a->sidx, b->sidx, picks * 4) &&
         !memcmp(a->wsel, b->wsel, picks * 4));
  const size_t en = a->entries * 4, sn = a->slots * 4;
  assert(!memcmp(a->cnt, b->cnt, en) && !memcmp(a->expert, b->expert, en) && !memcmp(a->x, b->x, en) &&
         !memcmp(a->y, b->y, en) && !memcmp(a->xd, b->xd, en) && !memcmp(a->yd, b->yd, en));
  assert(!memcmp(a->rgidx, b->rgidx, sn) && !memcmp(a->rhidx, b->rhidx, sn));
}

static double now_ms(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int main(void)
{
  setvbuf(stdout, NULL, _IONBF, 0);
  printf("[test_hy4_route] start\n");
  /* (tokens, picks a token, chunk, experts drawn from): a few experts (long runs past the chunk), all, repeats */
  const uint32_t cases[][4] = {{1, 8, 2048, E}, {7, 8, 2048, 3}, {300, 8, 64, 20}, {2048, 8, 2048, E},
                               {4096, 8, 2048, E}, {513, 16, 100, E}, {64, 1, 2048, 2}};
  for (size_t c = 0; c < sizeof cases / sizeof *cases; c++)
  {
    const uint32_t n = cases[c][0], k = cases[c][1], chunk = cases[c][2], from = cases[c][3], picks = n * k;
    int32_t *idx = malloc(picks * 4);
    float *w = malloc(picks * 4);
    assert(idx && w);
    for (uint32_t i = 0; i < picks; i++)
      idx[i] = (int32_t)(from < E ? draw(from) * (E / from) : draw(E)), w[i] = (float)draw(1000) / 1000.0f;
    Tabs got = tabs(n, k), want = tabs(n, k);
    assert(hy4_read_route(idx, w, n, k, E, chunk, TILE, LAT, FFE, &got.r));
    reference(idx, w, n, k, chunk, &want.r);
    same(&got.r, &want.r, picks);
    printf("  %u tokens x %u picks (experts of %u, chunk %u): %u entries, %u slots - the reference's tables\n", n, k,
           from, chunk, got.r.entries, got.r.slots);
    if (n == 4096)
    {
      double best = 1e9;
      for (int rep = 0; rep < 5; rep++)
      {
        const double t0 = now_ms();
        assert(hy4_read_route(idx, w, n, k, E, chunk, TILE, LAT, FFE, &got.r));
        const double t = now_ms() - t0;
        best = t < best ? t : best;
      }
      printf("  %u tokens: %.3f ms (bound %.1f)\n", n, best, kBoundMs);
      assert(best < kBoundMs);
    }
    idx[picks / 2] = E; /* a pick out of range: refused */
    assert(!hy4_read_route(idx, w, n, k, E, chunk, TILE, LAT, FFE, &got.r));
    idx[picks / 2] = -1;
    assert(!hy4_read_route(idx, w, n, k, E, chunk, TILE, LAT, FFE, &got.r));
    free(got.mem), free(want.mem), free(idx), free(w);
  }
  printf("[test_hy4_route] ok\n");
  return 0;
}
