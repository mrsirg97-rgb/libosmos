#ifndef OSMOS_RUN_FIT_H
#define OSMOS_RUN_FIT_H

#include <stdbool.h>
#include <stdint.h>

#define FIT_HEADROOM (1ull << 30) /* GPU memory left free past the experts: the packs' measured margin (DESIGN.md,
                                     "the budgets at 1M": every edge started; 1 GiB under ran) */

#define FIT_RAM_FREE (8ull << 30) /* the system's RAM a build leaves free by default (osmos-serve --fill: another): the
                                     operator's rule - on this APU the GPU's memory is the system's, and one box runs
                                     the model beside an editor, a browser, language servers and the agent harness */

#define FIT_SLACK (1ull << 28) /* room past the headroom a sizing stops at: 256 MiB, a few experts' units */
#define FIT_BUILDS 6           /* builds at most past the dense one's first (and one more to keep the largest that
                                  left the headroom): a sizing took 3-6 on this box */
#define FIT_ERR 256

/* one whole build at a budget (0: the dense weights alone): what it took (forward_budget) and the room it left
   (forward_room); false with why in err */
typedef bool (*FitBuild)(void *ctx, uint64_t budget, uint64_t *built, uint64_t *room, char err[FIT_ERR]);
typedef void (*FitUnbuild)(void *ctx);

uint64_t fit_budget(uint64_t built, uint64_t free_after, uint64_t headroom);
uint64_t fit_room(uint64_t gpu_free, uint64_t ram_available, uint64_t ram_kept);
uint64_t fit_ram_available(void);
bool fit_sized(FitBuild build, FitUnbuild unbuild, void *ctx, uint64_t headroom, uint64_t *budget,
               char err[FIT_ERR]);

#endif

/*
fit_budget
  the next budget toward this machine's fill: a build at `built` (0 asks the forward for its dense weights alone;
  forward_budget then says what that took) left free_after bytes of GPU memory free; the budget moves by what is past
  the headroom - up by it, or down by the shortfall when a build left less (never below 0).

fit_room
  the room a build left, against FIT_HEADROOM: the GPU's free memory, or the RAM available past ram_kept (FIT_RAM_FREE
  by default; the headroom added back), whichever is less - so a room of at least FIT_HEADROOM is one that left both
  the GPU's headroom and the system's ram_kept; 0 when the RAM is short of that by the headroom or more (a ram_kept
  past the RAM there is: 0).

fit_ram_available
  /proc/meminfo's MemAvailable in bytes; 0 when it cannot be read (no room: fail closed).

fit_sized
  the engine built to fill this machine and left built (budget: what it took): built at 0 (the dense weights alone,
  the floor), then toward a room of the headroom plus half FIT_SLACK - each step the room past (or short of) it over
  what a byte of budget has cost so far (measured between builds: the expert pool's room shrinks as experts become
  resident, a resident expert's units take more than its bytes - ~6% on UD-IQ1_S at 1M), kept between the largest
  budget that left the headroom and the smallest that did not or was refused. it stops in the window (the headroom to
  FIT_SLACK past it), or where those two budgets close within FIT_SLACK (memory taken in steps wider than the window:
  ~0.93 GiB on this box), or after FIT_BUILDS builds; the build it keeps leaves at least the headroom whenever one has
  (one more build, at the largest that did, if the last fell short), else the floor. false with why: the dense build
  failed, or the largest that fit did not again (unbuild has run for every build before; a failed one cleans up
  after itself).
*/
