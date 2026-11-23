/* the experts' store built from a plan (osmos/mem/experts.h): its pool's and stream's pinned host slots, then the
   membrane over them */
#include <stdio.h>
#include <string.h>

#include "osmos/mem/ledger_hip.h"
#include "osmos/mem/experts.h"
#include "osmos/mem/host_slots.h"
#include "osmos/run/fit.h"

#include <hip/hip_runtime_api.h>
#include "osmos/err.h"

bool experts_create(OsmosLedger *l, const PoolUnit *units, uint32_t layers, uint32_t experts, const uint32_t *order,
                    const ExpertsPlan *p, uint64_t grain, int readers, membrane_copy_fn copy, void *copy_ctx,
                    uint64_t ahead_limit, bool warm_now, Experts *out, char err[OSMOS_ERR])
{
  memset(out, 0, sizeof *out);
  char perr[OSMOS_ERR] = "";
  if (!(out->slots = host_slots_create(l, p->room + 2 * p->largest, perr)) ||
      !(out->stream = host_slots_create(l, p->stream_room + 2 * p->largest, perr)))
  {
    osmos_fail(err, "the experts' memory refused: %s", perr);
    return experts_destroy(out), false;
  }
  const MembraneOptions mo = {units, layers, experts, p->room, p->stream_room, host_slots(out->slots),
                              host_slots(out->stream), grain, readers, order, p->n_warm, copy, copy_ctx, ahead_limit};
  if (!(out->mem = membrane_create(&mo, err)) || (warm_now && !membrane_warm(out->mem, err)))
    return experts_destroy(out), false;
  return true;
}

void experts_destroy(Experts *e)
{
  membrane_destroy(e->mem); /* its readers stopped, then its memory */
  host_slots_destroy(e->stream), host_slots_destroy(e->slots);
  memset(e, 0, sizeof *e);
}

uint64_t experts_fit_now(OsmosLedger *l, uint64_t fixed)
{
  char err[OSMOS_ERR] = "";
  OsmosLedger *own = l ? NULL : ledger_create(FIT_RAM_FREE, err); /* (a caller with none: the machine as it stands) */
  if (!l && !own)
    return 1;
  const uint64_t left = ledger_left_settled(l ? l : own);
  ledger_destroy(own);
  return left ? fixed + left : 1;
}
