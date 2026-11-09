#include "osmos/run/fit.h"

#include <stdio.h>

uint64_t fit_room(uint64_t gpu_free, uint64_t ram_available, uint64_t ram_kept)
{
  const uint64_t ram = ram_available + FIT_HEADROOM > ram_kept ? ram_available + FIT_HEADROOM - ram_kept : 0;
  return ram < gpu_free ? ram : gpu_free;
}

uint64_t fit_ram_available(void)
{
  FILE *f = fopen("/proc/meminfo", "r");
  unsigned long long kb = 0;
  char line[256];
  while (f && fgets(line, sizeof line, f))
    if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1)
      break;
  if (f)
    fclose(f);
  return (uint64_t)kb << 10;
}

uint64_t fit_budget(uint64_t built, uint64_t free_after, uint64_t headroom)
{
  if (free_after >= headroom)
    return built + (free_after - headroom);
  return headroom - free_after < built ? built - (headroom - free_after) : 0;
}

/* the next budget: the room past (or short of) the target, over what a byte of budget has cost so far - two builds'
   room against their budgets (the pool's shrink and the units' padding both in it); 1 until there are two */
static uint64_t next(uint64_t built, uint64_t room, uint64_t target, double cost)
{
  const double step = ((double)room - (double)target) / cost;
  return step >= 0 ? built + (uint64_t)step : (-step < (double)built ? built - (uint64_t)-step : 0);
}

bool fit_sized(FitBuild build, FitUnbuild unbuild, void *ctx, uint64_t headroom, uint64_t *budget, char err[FIT_ERR])
{
  const uint64_t target = headroom + FIT_SLACK / 2; /* the window's middle: a step's error either side stays in it */
  uint64_t built = 0, room = 0;
  if (!build(ctx, 0, &built, &room, err))
    return false; /* the dense build itself: nothing smaller to try */
  const uint64_t dense = built;          /* the floor: the dense weights alone */
  uint64_t ok = dense, over = UINT64_MAX; /* the largest budget that left the headroom (or the floor), the smallest
                                            that did not or was refused */
  uint64_t last = built, last_room = room;
  double cost = 1.0; /* GPU memory a byte of budget takes: measured from two builds, 1 until then */
  bool live = true;
  for (int i = 1; i < FIT_BUILDS; i++)
  {
    if (live && room >= headroom && (room <= headroom + FIT_SLACK || (over != UINT64_MAX && over - built <= FIT_SLACK)))
      break; /* in the window, or as close as the memory's steps allow */
    if (live && room < headroom && built <= dense)
      break; /* short at the floor: nothing smaller */
    uint64_t want = next(last, last_room, target, cost);
    if (over != UINT64_MAX && (want >= over || want <= ok))
      want = ok + (over - ok) / 2; /* outside the bracket: its middle */
    if (over != UINT64_MAX && over - ok <= FIT_SLACK)
      want = ok; /* the bracket closed: the largest that left the headroom */
    if (live)
      unbuild(ctx);
    live = build(ctx, want, &built, &room, err);
    if (!live)
    {
      over = want < over ? want : over; /* refused: past what fits (it cleaned up after itself) */
      continue;
    }
    if (built != last && room != last_room)
    {
      const double c = ((double)last_room - (double)room) / ((double)built - (double)last);
      cost = c > 0 ? c : cost; /* a nonsensical measure (the room the wrong way) keeps the last */
    }
    last = built, last_room = room;
    if (room >= headroom)
      ok = built > ok ? built : ok;
    else
      over = built < over ? built : over;
  }
  if (!live || (room < headroom && built != ok)) /* never kept short of the headroom when a build that left it is known */
  {
    if (live)
      unbuild(ctx);
    if (!build(ctx, ok, &built, &room, err))
      return false;
  }
  *budget = built;
  return true;
}
