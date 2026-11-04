#include "osmos/run/fit.h"

uint64_t fit_budget(uint64_t built, uint64_t free_after, uint64_t headroom)
{
  return free_after > headroom ? built + (free_after - headroom) : built;
}

bool fit_sized(FitBuild build, FitUnbuild unbuild, void *ctx, uint64_t headroom, uint64_t *budget, char err[FIT_ERR])
{
  uint64_t want = 0, built = 0, room = 0;
  for (int i = 0; i < FIT_BUILDS; i++)
  {
    if (!build(ctx, want, &built, &room, err))
      return false;
    if (room <= headroom + FIT_SLACK || i == FIT_BUILDS - 1)
      break;
    want = fit_budget(built, room, headroom);
    unbuild(ctx);
  }
  *budget = built;
  return true;
}
