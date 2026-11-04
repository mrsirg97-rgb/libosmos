#include <assert.h>
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
  int builds, live;
  int fail_at;
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
  const uint64_t used = e->other + *built, p = pool(e, *built);
  if (used + p > e->memory) /* the engine refuses what does not fit (fwd_room_left) */
    return snprintf(err, FIT_ERR, "past the memory"), false;
  *room = e->memory - used - p;
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
  assert(fit_budget(12 * GIB, 95 * GIB, GIB) == 106 * GIB);
  assert(fit_budget(12 * GIB, GIB, GIB) == 12 * GIB);
  assert(fit_budget(12 * GIB, GIB / 2, GIB) == 12 * GIB);
  assert(fit_budget(12 * GIB, 95 * GIB, 0) == 107 * GIB);
  /* UD-IQ1_M-like: 113 GiB, 3 of it the rest, 12 dense, 140 of experts over 40 layers */
  const uint64_t shapes[][5] = {{113, 3, 12, 140, 40}, {113, 8, 12, 122, 40}, {96, 3, 12, 140, 40}, {64, 2, 12, 140, 40}};
  for (size_t k = 0; k < sizeof shapes / sizeof *shapes; k++)
  {
    Engine e = {shapes[k][0] * GIB, shapes[k][1] * GIB, shapes[k][2] * GIB, shapes[k][3] * GIB, shapes[k][4], 0, 0, 0};
    uint64_t budget = 0, built, room;
    char err[FIT_ERR] = "";
    assert(fit_sized(build, unbuild, &e, GIB, &budget, err));
    assert(e.live && e.builds <= FIT_BUILDS);
    room = e.memory - e.other - budget - pool(&e, budget);
    printf("  %3llu GiB: budget %.2f GiB in %d builds, %.3f GiB left\n", (unsigned long long)shapes[k][0],
           (double)budget / GIB, e.builds, (double)room / GIB);
    assert(room >= GIB && room <= GIB + FIT_SLACK); /* filled to the headroom, never past it */
    (void)built;
  }
  Engine e = {113 * GIB, 3 * GIB, 12 * GIB, 140 * GIB, 40, 0, 0, 2}; /* the second build refused */
  uint64_t budget = 0;
  char err[FIT_ERR] = "";
  assert(!fit_sized(build, unbuild, &e, GIB, &budget, err) && !strcmp(err, "a build refused") && !e.live);
  printf("  a refused build: its why (%s)\n[test_fit] ok\n", err);
  return 0;
}
