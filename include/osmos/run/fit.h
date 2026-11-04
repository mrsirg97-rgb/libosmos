#ifndef OSMOS_RUN_FIT_H
#define OSMOS_RUN_FIT_H

#include <stdbool.h>
#include <stdint.h>

#define FIT_HEADROOM (1ull << 30) /* GPU memory left free past the experts: the packs' measured margin (DESIGN.md,
                                     "the budgets at 1M": every edge started; 1 GiB under ran) */

#define FIT_SLACK (1ull << 28) /* room past the headroom a sizing stops at: 256 MiB, a few experts' units */
#define FIT_BUILDS 5           /* builds at most: the dense one, then each closer (the pool's room shrinks as experts
                                  become resident, so each step's room is an underestimate the next one corrects) */
#define FIT_ERR 256

/* one whole build at a budget (0: the dense weights alone): what it took (forward_budget) and the room it left
   (forward_room); false with why in err */
typedef bool (*FitBuild)(void *ctx, uint64_t budget, uint64_t *built, uint64_t *room, char err[FIT_ERR]);
typedef void (*FitUnbuild)(void *ctx);

uint64_t fit_budget(uint64_t built, uint64_t free_after, uint64_t headroom);
bool fit_sized(FitBuild build, FitUnbuild unbuild, void *ctx, uint64_t headroom, uint64_t *budget,
               char err[FIT_ERR]);

#endif

/*
fit_budget
  the expert budget that fills this machine: an engine built whole (the forward, its sequences, the drafter) at the
  budget `built` - 0 asks the forward for its dense weights alone (forward_budget then says what that took) - left
  free_after bytes of GPU memory free; all of it but headroom goes to experts. less free than headroom: `built` (no
  experts more).

fit_sized
  the engine built to fill this machine and left built (budget: what it took): built at 0, then again at fit_budget
  of each build's room until that room is within FIT_SLACK past the headroom, FIT_BUILDS builds at most (the last
  kept as it is). every step is short of the fill, never past it: the room each build reports counts the expert
  pool's room at that build's residency, which only shrinks as the budget grows. false with why: a build failed
  (unbuild has run for every build before it; the failed one cleans up after itself).
*/
