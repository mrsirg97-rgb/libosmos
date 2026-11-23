/* the experts' plan (experts_plan, experts_stream_room: every MoE family's sizing) against the code it replaced -
   Inkling's experts() loop (0.12.3: room and stream settled over passes) and DeepSeek's stream_budget - on random
   units (1-3 parts, uneven sizes) at 4-48 layers of 8-256 experts: budgets from 0 to past every unit, a room given,
   a picks order and the round-robin one. one rule changed on purpose: every unit warm, the stream 3 units (Qwen's),
   not a whole layer (Inkling-2-Mini's whole build held ~0.6 GB of pinned host memory it never read through) */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "osmos/mem/experts.h"

#define GRAIN 4096

static uint64_t g_rng = 1017;
static uint64_t draw(uint64_t m)
{
  g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
  return (g_rng >> 17) % m;
}

/* DeepSeek's stream_budget (run/hip_forward.c, 0.12.3), a warm flag a unit for its rix */
static uint64_t ref_stream(const PoolUnit *u, uint32_t L, uint32_t E, const bool *warm)
{
  uint64_t most = 0, whole = 0;
  for (uint32_t l = 0; l < L; l++)
  {
    uint64_t layer = 0, cold = 0;
    for (uint32_t e = 0; e < E; e++)
    {
      const uint64_t b = pool_unit_bytes(&u[l * E + e], GRAIN);
      layer += b, cold += warm[l * E + e] ? 0 : b;
    }
    most = cold > most ? cold : most, whole = layer > whole ? layer : whole;
  }
  return 2 * most > whole ? 2 * most : whole;
}

/* Inkling's experts() sizing (ink2/hip_forward.c, 0.12.3), with the plan's one change (every unit warm: 3 units) */
static ExpertsPlan ref_plan(const PoolUnit *u, uint32_t L, uint32_t E, const uint32_t *warm, uint64_t room0,
                            uint64_t budget, uint64_t fixed)
{
  const size_t n = (size_t)L * E;
  uint64_t all = 0, largest = 0;
  for (size_t i = 0; i < n; i++)
  {
    const uint64_t b = pool_unit_bytes(&u[i], GRAIN);
    all += b, largest = b > largest ? b : largest;
  }
  uint64_t stream_room = 2 * (uint64_t)E * largest, room = room0;
  size_t n_warm = 0;
  for (int pass = 0; pass < EXPERTS_PASSES; pass++)
  {
    room = room0;
    if (budget)
    {
      const uint64_t rest = fixed + stream_room + 4 * largest + 2 * GRAIN;
      room = budget > rest ? budget - rest : 0;
    }
    if (room > all)
      room = all;
    if (room < (uint64_t)E * largest)
      room = (uint64_t)E * largest;
    n_warm = 0;
    for (uint64_t took = 0; n_warm < n; n_warm++)
    {
      const uint64_t b = pool_unit_bytes(&u[warm[n_warm]], GRAIN);
      if (took + b > room)
        break;
      took += b;
    }
    uint64_t worst = 0, whole = 0;
    for (uint32_t l = 0; l < L; l++)
    {
      uint64_t layer = 0, cold = 0;
      for (size_t r = 0; r < n; r++)
        if (warm[r] / E == l)
        {
          const uint64_t b = pool_unit_bytes(&u[warm[r]], GRAIN);
          layer += b, cold += r < n_warm ? 0 : b;
        }
      worst = cold > worst ? cold : worst, whole = layer > whole ? layer : whole;
    }
    uint64_t want = 2 * worst > whole ? 2 * worst : whole;
    if (n_warm == n)
      want = 3 * largest;
    const bool settled = want >= stream_room;
    stream_room = want < stream_room ? want : stream_room;
    if (settled || !budget)
      break;
  }
  return (ExpertsPlan){room, stream_room, largest, all, n_warm};
}

int main(void)
{
  setvbuf(stdout, NULL, _IONBF, 0);
  printf("[test_experts_plan] start\n");
  int checked = 0;
  for (int c = 0; c < 60; c++)
  {
    const uint32_t L = 4 + (uint32_t)draw(45), E = 8 + (uint32_t)draw(249);
    const size_t n = (size_t)L * E;
    PoolUnit *u = calloc(n, sizeof *u);
    uint32_t *order = malloc(n * sizeof *order);
    bool *warm = calloc(n, 1);
    assert(u && order && warm);
    for (size_t i = 0; i < n; i++)
    {
      u[i].n_parts = 1 + (int)draw(3);
      for (int p = 0; p < u[i].n_parts; p++)
        u[i].part[p] = (PoolRange){3, draw(1u << 30), 1000 + draw(3u << 20)};
    }
    const bool picks = c % 2;  /* a picks order (shuffled), else round-robin over the layers */
    for (size_t i = 0; i < n; i++)
      order[i] = picks ? (uint32_t)i : (uint32_t)((i % L) * E + i / L);
    for (size_t i = n - 1; picks && i > 0; i--)
    {
      const size_t j = (size_t)draw(i + 1);
      const uint32_t t = order[i];
      order[i] = order[j], order[j] = t;
    }
    uint64_t all = 0;
    for (size_t i = 0; i < n; i++)
      all += pool_unit_bytes(&u[i], GRAIN);
    const uint64_t fixed = draw(8ull << 30);
    const uint64_t budgets[] = {0, 1, fixed, fixed + all / 3, fixed + all / 2, fixed + all, 2 * (fixed + all), ~0ull >> 4};
    const uint64_t rooms[] = {0, all / 4, all, 2 * all};
    for (size_t b = 0; b < sizeof budgets / sizeof *budgets; b++)
      for (size_t r = 0; r < (budgets[b] ? 1 : sizeof rooms / sizeof *rooms); r++)
      {
        ExpertsPlan got;
        assert(experts_plan(u, L, E, order, rooms[r], budgets[b], fixed, GRAIN, &got));
        const ExpertsPlan want = ref_plan(u, L, E, order, rooms[r], budgets[b], fixed);
        assert(got.room == want.room && got.stream_room == want.stream_room && got.n_warm == want.n_warm &&
               got.largest == want.largest && got.all == want.all);
        /* the budget a ceiling (0.16.0): the build as made - the fixed part, both slot sets with their 2 units' slack
           and their round-up to a grain - within it, or at the floor */
        assert(!budgets[b] || budgets[b] < fixed + got.room ||
               fixed + got.room + got.stream_room + 4 * got.largest + 2 * GRAIN <= budgets[b] ||
               got.room == (uint64_t)E * got.largest);
        memset(warm, 0, n);
        for (size_t i = 0; i < got.n_warm; i++)
          warm[order[i]] = true;
        if (got.n_warm < n)
          assert(experts_stream_room(u, L, E, warm, GRAIN) == ref_stream(u, L, E, warm));
        checked++;
      }
    uint32_t bad = order[0];
    order[0] = order[1];  /* a unit twice, another never: refused */
    ExpertsPlan p;
    assert(!experts_plan(u, L, E, order, 0, 0, 0, GRAIN, &p));
    order[0] = bad;
    assert(!experts_plan(u, 0, E, order, 0, 0, 0, GRAIN, &p));
    free(u), free(order), free(warm);
  }
  printf("  %d plans: 0.12.3's room, stream and warm set (every unit warm: 3 units); DeepSeek's stream rule\n", checked);
  printf("[test_experts_plan] ok\n");
  return 0;
}
