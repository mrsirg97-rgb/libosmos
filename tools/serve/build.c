#include <stdio.h>
#include <string.h>

#include "osmos/run/fit.h"
#include "serve.h"

/* the engine whole - the model's forward and drafter (osmos_create), every lane's conversation - built at a budget, or sized to this
   machine (osmos/run/fit.h's fit_sized): built with the dense weights alone, then again with the room each build
   left, but the headroom, given to experts - each build holds everything but more experts, so what it leaves is what
   they may take, on any machine, context, lane count or drafter */

void serve_unbuild(Server *sv, Conv *conv, int lanes)
{
  for (int i = 0; i < lanes; i++)
    conv_destroy(&conv[i]);
  osmos_destroy(sv->e);
  sv->e = NULL;
}

static bool build(Server *sv, Conv *conv, int lanes, const OsmosModel *m, const OsmosOptions *o, char err[OSMOS_ERR])
{
  if (!(sv->e = osmos_create(m, o, err)))
    return false;
  for (int i = 0; i < lanes; i++)
    if (!conv_create(&conv[i], sv))
      return snprintf(err, OSMOS_ERR, "%s", conv[i].err), false;
  return true;
}

typedef struct Sizing
{
  Server *sv;
  Conv *conv;
  int lanes;
  const OsmosModel *m;
  OsmosOptions o;
} Sizing;

static bool sized_build(void *ctx, uint64_t budget, uint64_t *built, uint64_t *room, char err[FIT_ERR])
{
  Sizing *z = ctx;
  char ferr[OSMOS_ERR] = "";
  z->o.budget = budget;
  if (!build(z->sv, z->conv, z->lanes, z->m, &z->o, ferr) || !osmos_room(z->sv->e, room))
  {
    serve_unbuild(z->sv, z->conv, z->lanes); /* the failed build's own (fit_sized unbuilt the ones before it) */
    return snprintf(err, FIT_ERR, "%s", ferr[0] ? ferr : "the GPU's free memory could not be read"), false;
  }
  OsmosStats st;
  osmos_stats(z->sv->e, &st);
  *built = st.budget;
  out("NOTE sizing: a build at %.2f GiB took %.2f, left %.2f GiB\n", (double)budget / (1ull << 30),
      (double)*built / (1ull << 30), (double)*room / (1ull << 30));
  return true;
}

static void sized_unbuild(void *ctx)
{
  Sizing *z = ctx;
  serve_unbuild(z->sv, z->conv, z->lanes);
}

bool serve_build(Server *sv, Conv *conv, int lanes, const OsmosModel *m, OsmosOptions o, char err[OSMOS_ERR])
{
  if (o.budget || !osmos_facts(m).streams) /* a budget given: as it is; a family streaming none: once, at 0 */
    return build(sv, conv, lanes, m, &o, err);
  Sizing z = {sv, conv, lanes, m, o};
  uint64_t budget = 0, room = 0;
  char ferr[FIT_ERR] = "";
  if (!fit_sized(sized_build, sized_unbuild, &z, FIT_HEADROOM, &budget, ferr))
    return snprintf(err, OSMOS_ERR, "%s", ferr), false;
  osmos_room(sv->e, &room);
  out("NOTE the budget sized to this GPU: %.2f GiB (%.2f GiB left free; %.2f GiB of RAM kept free, --fill)\n",
      (double)budget / (1ull << 30), (double)room / (1ull << 30), (double)sv->ram_kept / (1ull << 30));
  if (room < FIT_HEADROOM) /* fit_sized's floor kept short: --fill past what this machine can keep free */
    out("NOTE --fill %.2f GiB: more RAM than can be kept free beside the model - the smallest build kept (%.2f GiB: "
        "the dense weights, every expert streamed)\n", (double)sv->ram_kept / (1ull << 30), (double)budget / (1ull << 30));
  return true;
}
