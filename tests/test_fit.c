#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "osmos/run/fit.h"

/* fit_budget: the built budget and the free memory past the headroom. fit_sized against a modelled engine whose expert
   pool shrinks as experts become resident (forward.c's stream_budget: twice a layer's streamed experts): it ends
   within the slack past the headroom, every build fits, it stops at FIT_BUILDS, a failed build's why passes through */

#define GIB (1ull << 30)

typedef struct Engine
{
  uint64_t memory, other, dense, experts, layers; /* the GPU's, everything not the resident set's, the model's */
  uint64_t overhead_pct; /* what a resident expert's GiB costs past a GiB (its units' padding): the box measured ~6 */
  uint64_t chunk;        /* the memory taken in steps of this (0: none) - the box's room jumped ~0.93 GiB at once */
  int builds, live;
  int fail_at;
  uint64_t wobble; /* the room measured off by this, up then down build by build (0: none) - MemAvailable's: the box
                      read 1.41 then 1.39 GiB for builds 0.32 GiB apart, and a step from that cost went 5 GiB past */
  int64_t least;   /* the least true room any build left (a build past the memory: below 0) */
} Engine;

static uint64_t pool(const Engine *e, uint64_t budget)
{
  const uint64_t resident = budget > e->dense ? budget - e->dense : 0;
  const uint64_t streamed = resident < e->experts ? e->experts - resident : 0;
  return 2 * (streamed / e->layers);
}

static bool build(void *ctx, uint64_t budget, uint64_t *built, uint64_t *room, char err[FIT_ERR])
{
  Engine *e = ctx;
  assert(!e->live); /* the last one unbuilt first */
  if (++e->builds == e->fail_at)
    return snprintf(err, FIT_ERR, "a build refused"), false;
  *built = budget ? budget : e->dense;
  const uint64_t resident = *built > e->dense ? *built - e->dense : 0;
  uint64_t used = e->other + *built + resident * e->overhead_pct / 100;
  if (e->chunk)
    used = (used + e->chunk - 1) / e->chunk * e->chunk;
  const uint64_t p = pool(e, *built);
  const int64_t truth = (int64_t)e->memory - (int64_t)(used + p);
  e->least = truth < e->least ? truth : e->least;
  if (used + p > e->memory) /* the engine refuses what does not fit (fwd_room_left) */
    return snprintf(err, FIT_ERR, "past the memory"), false;
  *room = e->memory - used - p;
  if (e->wobble)
    *room = e->builds % 2 ? *room + e->wobble : (*room > e->wobble ? *room - e->wobble : 0);
  e->live = 1;
  return true;
}

static void unbuild(void *ctx)
{
  ((Engine *)ctx)->live = 0;
}

int main(void)
{
  printf("[test_fit] start\n");
  /* fit_room: the GPU's free memory or the RAM past the RAM kept free (FIT_RAM_FREE, the default; the headroom added
     back), the less */
  const uint64_t K = FIT_RAM_FREE;
  assert(fit_room(50 * GIB, 100 * GIB, K) == 50 * GIB);                  /* the GPU's the bound */
  assert(fit_room(50 * GIB, 20 * GIB, K) == 20 * GIB - K + FIT_HEADROOM); /* the RAM's */
  assert(fit_room(50 * GIB, K, K) == FIT_HEADROOM);                      /* the RAM's floor: just the headroom */
  assert(fit_room(50 * GIB, K - GIB / 2, K) == FIT_HEADROOM - GIB / 2);  /* short: under the headroom */
  assert(fit_room(50 * GIB, K - FIT_HEADROOM, K) == 0 && fit_room(50 * GIB, 0, K) == 0);
  /* --fill: another amount kept free - 0 the RAM's every byte (the headroom still), past what there is none */
  assert(fit_room(50 * GIB, 20 * GIB, 0) == 20 * GIB + FIT_HEADROOM);
  assert(fit_room(50 * GIB, 20 * GIB, 4 * GIB) == 16 * GIB + FIT_HEADROOM);
  assert(fit_room(50 * GIB, 20 * GIB, 1000 * GIB) == 0);
  assert(fit_room(50 * GIB, 20 * GIB, UINT64_MAX) == 0); /* no wrap */
  assert(fit_ram_available() > 0); /* this machine's /proc/meminfo */
  assert(fit_budget(12 * GIB, 95 * GIB, GIB) == 106 * GIB);
  assert(fit_budget(12 * GIB, GIB, GIB) == 12 * GIB);
  assert(fit_budget(12 * GIB, GIB / 2, GIB) == 12 * GIB - GIB / 2); /* short of the headroom: back by the shortfall */
  assert(fit_budget(GIB / 4, 0, GIB) == 0);
  assert(fit_budget(12 * GIB, 95 * GIB, 0) == 107 * GIB);
  /* UD-IQ1_M-like: 113 GiB, 3 of it the rest, 12 dense, 140 of experts over 40 layers */
  /* the last two the box's own shape: the step's buffers and a 1M context's state beside UD-IQ1_S, the experts'
     units 6% past their bytes (osmos-serve's sizing went 0 -> 94.97 -> 101.16 GiB and left 0.61, not 1) */
  const uint64_t shapes[][6] = {{113, 3, 12, 140, 40, 0}, {113, 8, 12, 122, 40, 0}, {96, 3, 12, 140, 40, 0},
                                {64, 2, 12, 140, 40, 0},  {113, 9, 8, 116, 40, 6},  {113, 9, 8, 116, 40, 12}};
  for (size_t k = 0; k < sizeof shapes / sizeof *shapes; k++)
  {
    Engine e = {shapes[k][0] * GIB, shapes[k][1] * GIB, shapes[k][2] * GIB, shapes[k][3] * GIB, shapes[k][4],
                shapes[k][5], 0, 0, 0, 0, 0, INT64_MAX};
    uint64_t budget = 0, built, room;
    char err[FIT_ERR] = "";
    assert(fit_sized(build, unbuild, &e, GIB, &budget, err));
    assert(e.live && e.builds <= FIT_BUILDS + 1); /* (+1: the last that fit, again, after a refused step) */
    room = e.memory - e.other - budget - (budget - e.dense) * e.overhead_pct / 100 - pool(&e, budget);
    printf("  %3llu GiB: budget %.2f GiB in %d builds, %.3f GiB left\n", (unsigned long long)shapes[k][0],
           (double)budget / GIB, e.builds, (double)room / GIB);
    assert(room >= GIB && room <= GIB + FIT_SLACK); /* filled to the headroom, never past it */
    (void)built;
  }
  Engine e = {113 * GIB, 3 * GIB, 12 * GIB, 140 * GIB, 40, 0, 0, 0, 0, 1, 0, INT64_MAX}; /* the dense build refused */
  uint64_t budget = 0;
  char err[FIT_ERR] = "";
  assert(!fit_sized(build, unbuild, &e, GIB, &budget, err) && !strcmp(err, "a build refused") && !e.live);
  Engine f = {113 * GIB, 3 * GIB, 12 * GIB, 140 * GIB, 40, 0, 0, 0, 0, 2, 0, INT64_MAX}; /* a growth step refused: smaller, not fatal */
  budget = 0;
  assert(fit_sized(build, unbuild, &f, GIB, &budget, err) && f.live);
  char refused[FIT_ERR];
  snprintf(refused, sizeof refused, "%s", err);
  const uint64_t smaller = budget;
  /* memory in chunks wider than the window: no budget lands in it - the one kept still leaves the headroom, within a
     chunk of it, whichever side the builds run out on */
  for (uint64_t c = 900; c <= 1000; c += 25)
  {
    Engine q = {113 * GIB, 9 * GIB, 8 * GIB, 116 * GIB, 40, 6, c * GIB / 1000, 0, 0, 0, 0, INT64_MAX};
    budget = 0;
    assert(fit_sized(build, unbuild, &q, GIB, &budget, err) && q.live);
    uint64_t used = q.other + budget + (budget - q.dense) * q.overhead_pct / 100;
    used = (used + q.chunk - 1) / q.chunk * q.chunk;
    const uint64_t left = q.memory - used - pool(&q, budget);
    printf("  chunks of %.3f GiB: budget %.2f GiB in %d builds, %.3f GiB left\n", (double)q.chunk / GIB,
           (double)budget / GIB, q.builds, (double)left / GIB);
    assert(left >= GIB && left <= GIB + q.chunk + FIT_SLACK);
  }
  /* the room's measure wobbling (MemAvailable's, up then down by w): a byte of budget never costs less than a byte, so
     no step goes past the memory nor further under the headroom than the wobble and the slack - the box's OOM
     (2026-11-22, DeepSeek: 98.67 -> 98.99 -> 104.06 GiB, 2.79 GiB of RAM left; the run before, none) */
  for (uint64_t w = 50; w <= 250; w += 50)
    for (size_t k = 0; k < sizeof shapes / sizeof *shapes; k++)
    {
      Engine q = {shapes[k][0] * GIB, shapes[k][1] * GIB, shapes[k][2] * GIB, shapes[k][3] * GIB, shapes[k][4],
                  shapes[k][5], 0, 0, 0, 0, w * GIB / 1000, INT64_MAX};
      budget = 0;
      assert(fit_sized(build, unbuild, &q, GIB, &budget, err) && q.live);
      if (!(q.least >= (int64_t)GIB - 2 * (int64_t)q.wobble - (int64_t)FIT_SLACK))
        fprintf(stderr, "  wobble %.2f GiB, %llu GiB box: a build left %.2f GiB\n", (double)q.wobble / GIB,
               (unsigned long long)shapes[k][0], (double)q.least / GIB);
      assert(q.least >= (int64_t)GIB - 2 * (int64_t)q.wobble - (int64_t)FIT_SLACK);
    }
  printf("  the room's measure wobbling 0.05-0.25 GiB: no build past the memory, none further under the headroom than "
         "the wobble\n");
  printf("  the dense build refused: its why (%s); a growth step refused: built smaller (%.2f GiB)\n[test_fit] ok\n",
         refused, (double)smaller / GIB);
  return 0;
}
