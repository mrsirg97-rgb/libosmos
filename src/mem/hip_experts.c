/* the experts' store built from a plan (osmos/mem/experts.h): its pool's and stream's pinned host slots, then the
   membrane over them */
#include <stdio.h>
#include <string.h>

#include "osmos/mem/experts.h"
#include "osmos/mem/host_slots.h"

bool experts_create(const PoolUnit *units, uint32_t layers, uint32_t experts, const uint32_t *order,
                    const ExpertsPlan *p, uint64_t grain, int readers, membrane_copy_fn copy, void *copy_ctx,
                    uint64_t ahead_limit, bool warm_now, Experts *out, char err[MEMBRANE_ERR])
{
  memset(out, 0, sizeof *out);
  char perr[POOL_ERR] = "";
  if (!(out->slots = host_slots_create(p->room + 2 * p->largest, perr)) ||
      !(out->stream = host_slots_create(p->stream_room + 2 * p->largest, perr)))
  {
    snprintf(err, MEMBRANE_ERR, "the experts' memory refused: %s", perr);
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
