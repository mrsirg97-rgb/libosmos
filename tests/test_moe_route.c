/* a step's picks grouped by expert (moe_route, moe_entries: every MoE family's host tables) against the plain
   definition - an expert at a time over every pick, the entries sorted stably by count: random picks (repeats in a
   token, experts no token picked, runs past a chunk) at DeepSeek's 256, Hy4-flash's 512 and Inkling's 64 / 128
   experts; a pick out of range refused; a 4,096-token step's sort at least kFaster times under the plain scan's in
   the same run (Hy4's 0.12.2 read: ~10 ms a layer of host time with the GPU idle) */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "osmos/model/moe_route.h"

#define TILE 16
static const double kFaster = 5.0;

static uint64_t g_rng = 1017;
static uint32_t draw(uint32_t m)
{
  g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
  return (uint32_t)(g_rng >> 33) % m;
}

typedef struct Tabs
{
  MoeRoute r;
  MoeEntry *e;
  uint64_t *key;
  void *mem;
} Tabs;

static Tabs tabs(uint32_t picks, uint32_t experts)
{
  const uint64_t slots = moe_route_slots(picks, experts, TILE), ents = picks + experts;
  Tabs t;
  char *p = t.mem = calloc(1, (size_t)picks * 8 + (size_t)experts * 20 + 4 + (size_t)slots * 8 + (size_t)ents * 12);
  t.key = calloc(ents, sizeof *t.key); /* (its own: 8-byte keys aligned, not carved after 12-byte entries) */
  assert(p && t.key);
  t.r.by = (uint32_t *)p, p += picks * 4;
  t.r.row = (uint32_t *)p, p += picks * 4;
  t.r.expert = (uint32_t *)p, p += experts * 4;
  t.r.start = (uint32_t *)p, p += experts * 4;
  t.r.count = (uint32_t *)p, p += experts * 4;
  t.r.slot0 = (uint32_t *)p, p += experts * 4;
  t.r.estart = (uint32_t *)p, p += (experts + 1) * 4;
  t.r.gidx = (int32_t *)p, p += slots * 4;
  t.r.hidx = (int32_t *)p, p += slots * 4;
  t.e = (MoeEntry *)p;
  return t;
}

/* the plain definition: an expert at a time, its picks in pick order; entries sorted stably by count */
static uint32_t reference(const int32_t *idx, uint32_t picks, uint32_t k, uint32_t experts, uint32_t chunk, MoeRoute *r,
                          MoeEntry *e)
{
  uint32_t q = 0, runs = 0, slots = 0;
  for (uint32_t x = 0; x < experts; x++)
  {
    const uint32_t first = q;
    for (uint32_t m = 0; m < picks; m++)
      if ((uint32_t)idx[m] == x)
        r->by[q] = m, r->row[m] = q, q++;
    if (q == first)
      continue;
    r->expert[runs] = x, r->start[runs] = first, r->count[runs] = q - first, r->slot0[runs] = slots;
    for (uint32_t j = first; j < q; j++)
      r->gidx[slots + j - first] = (int32_t)(r->by[j] / k), r->hidx[slots + j - first] = (int32_t)j;
    const uint32_t padded = (q - first + TILE - 1) / TILE * TILE;
    for (uint32_t s = slots + q - first; s < slots + padded; s++)
      r->gidx[s] = r->hidx[s] = -1;
    slots += padded, runs++;
  }
  r->runs = runs, r->slots = slots;
  uint32_t ne = 0;
  for (uint32_t j = 0; j < runs; j++)
    for (uint32_t off = 0; off < r->count[j]; off += chunk)
      e[ne++] = (MoeEntry){j, off, r->count[j] - off < chunk ? r->count[j] - off : chunk};
  for (uint32_t i = 1; i < ne; i++) /* (stable: insertion) */
    for (uint32_t j = i; j > 0 && e[j - 1].cnt > e[j].cnt; j--)
    {
      const MoeEntry t = e[j];
      e[j] = e[j - 1], e[j - 1] = t;
    }
  return ne;
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
  printf("[test_moe_route] start\n");
  /* tokens, picks a token, experts, experts drawn from, chunk */
  const uint32_t cases[][5] = {{1, 8, 512, 512, 2048}, {7, 8, 512, 3, 2048}, {300, 8, 256, 20, 64},
                               {2048, 6, 256, 256, 2048}, {4096, 8, 512, 512, 2048}, {513, 16, 512, 512, 100},
                               {64, 1, 64, 2, 2048}, {4096, 8, 128, 128, 128}};
  for (size_t c = 0; c < sizeof cases / sizeof *cases; c++)
  {
    const uint32_t n = cases[c][0], k = cases[c][1], E = cases[c][2], from = cases[c][3], chunk = cases[c][4];
    const uint32_t picks = n * k;
    int32_t *idx = malloc(picks * 4);
    assert(idx);
    for (uint32_t i = 0; i < picks; i++)
      idx[i] = (int32_t)(from < E ? draw(from) * (E / from) : draw(E));
    Tabs got = tabs(picks, E), want = tabs(picks, E);
    assert(moe_route(idx, picks, k, E, TILE, &got.r));
    uint32_t ne = 0;
    assert(moe_entries(&got.r, chunk, got.e, got.key, &ne));
    const uint32_t ne_ref = reference(idx, picks, k, E, chunk, &want.r, want.e);
    const MoeRoute *a = &got.r, *b = &want.r;
    assert(a->runs == b->runs && a->slots == b->slots && ne == ne_ref);
    assert(a->slots <= moe_route_slots(picks, E, TILE));
    assert(!memcmp(a->by, b->by, picks * 4) && !memcmp(a->row, b->row, picks * 4));
    assert(!memcmp(a->expert, b->expert, a->runs * 4) && !memcmp(a->start, b->start, a->runs * 4) &&
           !memcmp(a->count, b->count, a->runs * 4) && !memcmp(a->slot0, b->slot0, a->runs * 4));
    assert(!memcmp(a->gidx, b->gidx, a->slots * 4) && !memcmp(a->hidx, b->hidx, a->slots * 4));
    assert(!memcmp(got.e, want.e, ne * sizeof(MoeEntry)));
    printf("  %u tokens x %u picks, %u experts (drawn from %u), chunk %u: %u runs, %u slots, %u entries - the plain "
           "definition's\n", n, k, E, from, chunk, a->runs, a->slots, ne);
    if (n == 4096 && E == 512)
    {
      double best = 1e9, ref = 1e9;
      for (int rep = 0; rep < 5; rep++)
      {
        double t0 = now_ms();
        assert(moe_route(idx, picks, k, E, TILE, &got.r));
        uint32_t ne2;
        assert(moe_entries(&got.r, chunk, got.e, got.key, &ne2));
        const double t = now_ms() - t0;
        best = t < best ? t : best;
        t0 = now_ms();
        (void)reference(idx, picks, k, E, chunk, &want.r, want.e);
        const double tr = now_ms() - t0;
        ref = tr < ref ? tr : ref;
      }
      printf("  %u tokens: %.3f ms, the plain scan %.3f (at least %.0fx: %.1fx)\n", n, best, ref, kFaster, ref / best);
      assert(best * kFaster < ref);
    }
    uint32_t nx;
    assert(!moe_entries(&got.r, 0, got.e, got.key, &nx) && !moe_entries(&got.r, 4096, got.e, got.key, &nx));
    idx[picks / 2] = (int32_t)E; /* a pick out of range: refused */
    assert(!moe_route(idx, picks, k, E, TILE, &got.r));
    idx[picks / 2] = -1;
    assert(!moe_route(idx, picks, k, E, TILE, &got.r));
    free(got.mem), free(want.mem), free(got.key), free(want.key), free(idx);
  }
  printf("[test_moe_route] ok\n");
  return 0;
}
