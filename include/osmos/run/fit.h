#ifndef OSMOS_RUN_FIT_H
#define OSMOS_RUN_FIT_H

#include "osmos/err.h"
#include <stdbool.h>
#include <stdint.h>

#define FIT_HEADROOM (1ull << 30) /* GPU memory left free past the experts: the packs' measured margin */

#define FIT_RAM_FREE (8ull << 30) /* the system's RAM a build leaves free by default (osmos-serve --fill: another): on
                                     an APU the GPU's memory is the system's, and the box runs other programs */

/* a family's budget asking for as many experts as this machine holds now (experts_fit_now): its own tests' and
   benches'. a build's budget otherwise is the whole build's bytes, 0 the least (a layer's experts), in every family */
#define FIT_BUDGET_NOW UINT64_MAX
#define FIT_SLACK (1ull << 29) /* room past the headroom a sizing stops at: 512 MiB (256 left the room's wobble between
                                  builds to bounce on: Small sized in 6 whole builds) */
#define FIT_CHUNK (1ull << 30) /* the room's steps (memory is taken in steps near this): a build on a flat short of the
                                  headroom stepped below it by one */
#define FIT_FIRST_COST 1.125 /* fit_sized's first step's cost a byte of budget: past the units' padding */
#define FIT_BUILDS 6           /* builds at most past the dense one's first (and one more to keep the largest that
                                  left the headroom) */

/* one whole build at a budget (0: the dense weights alone): what it took (forward_budget) and the room it left
   (forward_room); false with why in err */
typedef bool (*FitBuild)(void *ctx, uint64_t budget, uint64_t *built, uint64_t *room, char err[OSMOS_ERR]);
typedef void (*FitUnbuild)(void *ctx);

uint64_t fit_budget(uint64_t built, uint64_t free_after, uint64_t headroom);
uint64_t fit_room(uint64_t gpu_free, uint64_t ram_available, uint64_t ram_kept);
uint64_t fit_ram_available(void);
bool fit_sized(FitBuild build, FitUnbuild unbuild, void *ctx, uint64_t headroom, uint64_t *budget,
               char err[OSMOS_ERR]);

/* what a sizing learned, for the next start of the same build: the dense build's room and the budget kept (a capped
   build's: what it was given) - budget 0: none known */
typedef struct FitHint
{
  uint64_t dense_room, budget;
  uint64_t start_room; /* the room before any build (osmos_machine_room): 0 unknown */
  uint64_t room;       /* the room the kept build left: 0 unknown */
  bool capped;         /* the kept build took less than its budget (budget: what it was given) */
} FitHint;
bool fit_sized_hint(FitBuild build, FitUnbuild unbuild, void *ctx, uint64_t headroom, uint64_t start_room,
                    FitHint *hint, uint64_t *budget, char err[OSMOS_ERR]);

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
  what a byte of budget has cost so far (measured between builds: a resident expert's units take more than its
  bytes), kept between the largest budget that left the headroom and the smallest that did not or was refused. it
  stops in the window (the headroom to FIT_SLACK past it), or where those two budgets close within FIT_SLACK (memory
  taken in steps wider than the window), or after FIT_BUILDS builds; the build it keeps leaves at least the headroom
  whenever one has
  (one more build, at the largest that did, if the last fell short), else the floor. false with why: the dense build
  failed, or the largest that fit did not again (unbuild has run for every build before; a failed one cleans up
  after itself).

fit_sized_hint
  fit_sized with what an earlier start learned: its first step the hint's budget moved by how far this dense build's
  room is from that one's (down by all of a shrink, up by a growth over FIT_FIRST_COST) - the same build on the same machine lands in the window at once,
  and anything else is a first step like any, checked and stepped from as fit_sized's. *hint is rewritten with what
  this sizing learned (a hint's budget 0: fit_sized's own first step). with start_room (the room before any build)
  and the hint's own, the hint's budget is built first, moved by those two rooms as above and aimed half a slack past
  the headroom by the room the hint's build left, and kept alone when it lands in the window (or capped) - no dense build; else one step from it, and past that unbuilt and sized as above.
*/
