/* an MoE's experts behind the membrane, sized to a budget: every family's one plan and one build */
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
  the units (unit = layer * experts + expert) planned for `order` (every unit, the most wanted first): budget 0 takes
  room as given; else budget is the build's whole, fixed's bytes, the stream's and 4 units' slack taken first. room at
  least a layer's experts, at most all; the warm set order's first that fit; the stream experts_stream_room's of it
  (every unit warm: 3 units); room and stream settled together over at most EXPERTS_PASSES. sizes are
  pool_unit_bytes at grain. false for no units, or an order not of every unit.
experts_stream_room
  the stream's room for a warm set (warm[unit]): twice the most cold bytes of any layer (one layer held, the next read
  ahead), and never under a whole layer's.
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
  them, order's first n_warm read in (membrane_warm); readers, copy, copy_ctx and ahead_limit are membrane_create's.
  the slots counted in l (borrowed; NULL counts nothing). false with err, nothing held, on any refusal.
experts_destroy
  the membrane (its readers stopped), then its memory; a zeroed one nothing.
*/
bool experts_create(OsmosLedger *l, const PoolUnit *units, uint32_t layers, uint32_t experts, const uint32_t *order,
                    const ExpertsPlan *p, uint64_t grain, int readers, membrane_copy_fn copy, void *copy_ctx,
                    uint64_t ahead_limit, bool warm_now, Experts *out, char err[OSMOS_ERR]);
void experts_destroy(Experts *e);

/*
experts_fit_now
  experts_plan's budget (bytes) for one build on this machine as it stands: fixed (what it has built) plus what l
  leaves (ledger_left_settled, less FIT_HEADROOM); 1 (the smallest build) when nothing is left. a NULL l: the machine
  read as a ledger keeping FIT_RAM_FREE would. a forward's own sizing (a test's, a bench's); osmos-serve sizes the
  whole engine (fit_sized)
*/
uint64_t experts_fit_now(OsmosLedger *l, uint64_t fixed);
