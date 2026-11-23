/* osmos/mem/experts.h - an MoE's experts behind the membrane, sized to a budget: every family's (DeepSeek's,
   Inkling's, Qwen's, Hy4-flash's) one plan and one build */
#pragma once
#include "osmos/err.h"
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "osmos/mem/membrane.h"
#include "osmos/mem/ledger.h"
#include "osmos/mem/pool.h"

typedef struct ExpertsPlan
{
  uint64_t room;        /* the membrane's pool: the warm experts' bytes it holds */
  uint64_t stream_room; /* the stream's: a cold pick's way in */
  uint64_t largest;     /* the largest unit (the slots' slack: 2 of them a side) */
  uint64_t all;         /* every unit's bytes (room == all: every expert resident) */
  size_t n_warm;        /* order's first n_warm the warm set */
} ExpertsPlan;

/*
experts_plan
  the units (layers x experts of them, unit = layer * experts + expert) planned for `order` (every unit, the most
  wanted first): budget 0 - room as given; else the build's whole budget, the rest (fixed's bytes, the stream's and 4
  units' slack) taken first. room at least a layer's experts (a decode's holds and a passing visit's), at most all; the
  warm set order's first that fit; the stream experts_stream_room's of it (every unit warm: 3 units), room and stream
  settled together over at most EXPERTS_PASSES (more room, fewer cold, a smaller stream). sizes pool_unit_bytes at
  grain. false for no units, or an order not of every unit.
experts_stream_room
  the stream's room for a warm set (warm[unit]): twice the most cold bytes of any layer (a prompt step's holds of one,
  its read ahead of the next), and never under a whole layer's (a layer's cold picks held at once).
*/
#define EXPERTS_PASSES 8
bool experts_plan(const PoolUnit *units, uint32_t layers, uint32_t experts, const uint32_t *order, uint64_t room,
                  uint64_t budget, uint64_t fixed, uint64_t grain, ExpertsPlan *p);
uint64_t experts_stream_room(const PoolUnit *units, uint32_t layers, uint32_t experts, const bool *warm,
                             uint64_t grain);

/* the built store: its membrane and the pinned host memory its pool and stream are carved from */
typedef struct Experts
{
  Membrane *mem;
  struct HostSlots *slots, *stream;
} Experts;

/*
experts_create
  the plan built: its pool's and stream's host slots (room + 2 largest, stream_room + 2 largest) and a membrane over
  them, order's first n_warm read in (membrane_warm) - readers, copy, copy_ctx and ahead_limit membrane_create's; the
  slots counted in l (osmos/mem/ledger.h: borrowed - the caller's build's; NULL counts nothing). false with err,
  nothing held, on any refusal.
experts_destroy
  the membrane (its readers stopped), then its memory; a zeroed one nothing.
*/
bool experts_create(OsmosLedger *l, const PoolUnit *units, uint32_t layers, uint32_t experts, const uint32_t *order,
                    const ExpertsPlan *p, uint64_t grain, int readers, membrane_copy_fn copy, void *copy_ctx,
                    uint64_t ahead_limit, bool warm_now, Experts *out, char err[OSMOS_ERR]);
void experts_destroy(Experts *e);

/*
experts_fit_now
  a build's budget sized to this machine as it stands: its fixed part (what it has built, bytes) and what l leaves
  (ledger_left_settled: the room - the GPU's free memory, or the RAM past what l keeps, --fill - less FIT_HEADROOM, read
  once it stands still) - experts_plan's budget for as many experts as fit, every one on a machine with the room; 1
  (the smallest build) when there is none. a NULL l: the machine read as a ledger keeping FIT_RAM_FREE would (0.16.0:
  it read FIT_RAM_FREE whatever --fill said). a forward's own sizing in one build (a test's, a bench's); osmos-serve
  sizes the whole engine over builds (fit_sized)
*/
uint64_t experts_fit_now(OsmosLedger *l, uint64_t fixed);
