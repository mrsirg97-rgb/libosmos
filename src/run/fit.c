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

bool fit_sized(FitBuild build, FitUnbuild unbuild, void *ctx, uint64_t headroom, uint64_t *budget, char err[OSMOS_ERR])
{
  return fit_sized_hint(build, unbuild, ctx, headroom, 0, NULL, budget, err);
}

/* a hint's budget moved by a room against the hint's: down by all of a room that shrank, up by a grown one over the
   first step's cost - either way short of the memory, not past it (a cost measured between close builds is the
   room's wobble) */
static uint64_t hinted(uint64_t budget, uint64_t then, uint64_t now, uint64_t floor)
{
  const double d = (double)now - (double)then;
  const double moved = d < 0 ? d : d / FIT_FIRST_COST;
  return moved >= 0 ? budget + (uint64_t)moved
                    : (-moved < (double)(budget - floor) ? budget - (uint64_t)-moved : floor);
}

static bool sized(FitBuild build, FitUnbuild unbuild, void *ctx, uint64_t headroom, FitHint *hint, uint64_t *budget,
                  char err[OSMOS_ERR]);

bool fit_sized_hint(FitBuild build, FitUnbuild unbuild, void *ctx, uint64_t headroom, uint64_t start_room,
                    FitHint *hint, uint64_t *budget, char err[OSMOS_ERR])
{
  if (hint && hint->budget && hint->start_room && start_room) /* the hint's build alone first: no dense build */
  {
    /* aimed at the middle of what it keeps (the window: the headroom to a slack past it), from the room the hint's build
       left: the room's wobble between starts either side stays in it */
    const uint64_t aim = headroom + FIT_SLACK / 2;
    uint64_t want = hinted(hint->budget, hint->start_room, start_room, 0);
    if (hint->room && !hint->capped)
      want = hinted(want, aim, hint->room, 0);
    uint64_t built = 0, room = 0;
    char herr[OSMOS_ERR] = "";
    for (int t = 0; t < 2; t++) /* the aimed build, and one step from it when it lands outside */
    {
      if (!build(ctx, want, &built, &room, herr))
        break; /* refused: sized as from cold, the dense build first */
      const bool capped = room >= headroom && built + FIT_SLACK < want;
      if (capped || (room >= headroom && room <= headroom + FIT_SLACK))
      {
        *budget = built;
        /* this start's own: its build (a capped one's: what it was given) at its room, the dense room moved alike
           (what the dense build takes is the build's, not the machine's) */
        const int64_t moved = (int64_t)start_room - (int64_t)hint->start_room;
        hint->dense_room = (int64_t)hint->dense_room + moved > 0 ? (uint64_t)((int64_t)hint->dense_room + moved) : 0;
        *hint = (FitHint){hint->dense_room, capped ? want : built, start_room, room, capped};
        return true;
      }
      unbuild(ctx);
      want = hinted(built, aim, room, 0); /* down by all of a shortfall, up by a surplus over the first cost */
    }
  }
  const bool ok = sized(build, unbuild, ctx, headroom, hint, budget, err);
  if (ok && hint)
    hint->start_room = start_room;
  return ok;
}

static bool sized(FitBuild build, FitUnbuild unbuild, void *ctx, uint64_t headroom, FitHint *hint, uint64_t *budget,
                  char err[OSMOS_ERR])
{
  const uint64_t target = headroom + FIT_SLACK / 2; /* the window's middle: a step's error either side stays in it */
  uint64_t built = 0, room = 0;
  if (!build(ctx, 0, &built, &room, err))
    return false; /* the dense build itself: nothing smaller to try */
  const uint64_t dense = built;          /* the floor: the dense weights alone */
  uint64_t ok = dense, over = UINT64_MAX; /* the largest budget that left the headroom (or the floor), the smallest
                                            that did not or was refused */
  uint64_t last = built, last_room = room;
  uint64_t prev = built, prev_room = room; /* the build before last: two budgets, one room - a flat */
  /* GPU memory a byte of budget takes: measured from two builds, at least 1 - until then FIT_FIRST_COST, the most the
     units' padding has been, so the first step, the longest, lands short of the target, not past it into the RAM
     kept free */
  double cost = FIT_FIRST_COST;
  const bool hinted_first = hint && hint->budget > dense;
  const uint64_t dense_room = room;
  uint64_t asked = 0; /* a capped build's budget (every expert resident in less): the hint's, so the next start's
                         first step is capped again, not taken as all it may hold */
  bool live = true;
  for (int i = 1; i < FIT_BUILDS; i++)
  {
    if (live && room >= headroom && (room <= headroom + FIT_SLACK || (over != UINT64_MAX && over - built <= FIT_SLACK)))
      break; /* in the window, or as close as the memory's steps allow */
    if (live && room < headroom && built <= dense)
      break; /* short at the floor: nothing smaller */
    uint64_t want = next(last, last_room, target, cost);
    if (i == 1 && hinted_first) /* the last start's budget, moved by this machine's room against that start's */
      want = hinted(hint->budget, hint->dense_room, dense_room, dense);
    /* on a flat short of the headroom - two budgets, the room moved less than half as far (a byte of budget costs at
       least a byte off a flat: only the streamed pool's drift moves it there) - the room moves in steps, not with the
       budget, so the shortfall says nothing of how far: stepped below the flat by one step, not crept across it a
       fraction at a time */
    const uint64_t moved = prev_room > last_room ? prev_room - last_room : last_room - prev_room;
    const uint64_t apart = prev > last ? prev - last : last - prev;
    if (live && last_room < headroom && prev_room < headroom && apart && moved < apart / 2)
    {
      const uint64_t low = prev < last ? prev : last;
      want = low > dense + FIT_CHUNK ? low - FIT_CHUNK : dense;
    }
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
      /* a byte of budget takes at least a byte of the room (both its bounds: the GPU's and the RAM's; the units'
         padding past it): a measure under 1 is the room's own wobble (MemAvailable's), and a step over it would go
         past the target by the wobble over the cost */
      const double c = ((double)last_room - (double)room) / ((double)built - (double)last);
      cost = c >= 1.0 ? c : 1.0;
    }
    prev = last, prev_room = last_room;
    last = built, last_room = room;
    if (room >= headroom)
      ok = built > ok ? built : ok;
    else
      over = built < over ? built : over;
    if (room >= headroom && built + FIT_SLACK < want)
    {
      asked = want; /* took less than it was given, the headroom left: every expert resident, nothing more to take
                       (Hy4, Mini: four or five more whole builds at the same size) */
      break;
    }
    if (i == 1 && hinted_first && room >= headroom && room <= headroom + FIT_SLACK)
      break; /* the hint's step a little past the window (the room's wobble between starts): kept, not one more
                whole build for a few experts */
  }
  if (!live || (room < headroom && built != ok)) /* never kept short of the headroom when a build that left it is known */
  {
    if (live)
      unbuild(ctx);
    if (!build(ctx, ok, &built, &room, err))
      return false;
  }
  *budget = built;
  if (hint)
    *hint = (FitHint){dense_room, asked && built == ok ? asked : built, 0, room, asked && built == ok};
  return true;
}
