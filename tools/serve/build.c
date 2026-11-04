#include <stdio.h>
#include <string.h>

#include "osmos/run/fit.h"
#include "serve.h"

/* the engine whole - the forward, every lane's conversation, the drafter - built at a budget, or sized to this
   machine (osmos/run/fit.h's fit_sized): built with the dense weights alone, then again with the room each build
   left, but the headroom, given to experts - each build holds everything but more experts, so what it leaves is what
   they may take, on any machine, context, lane count or drafter */

void serve_unbuild(Server *sv, Conv *conv, int lanes)
{
  for (int i = 0; i < lanes; i++)
    conv_destroy(&conv[i]);
  drafter_destroy(sv->d);
  forward_destroy(sv->f);
  sv->d = NULL, sv->f = NULL;
}

static bool build(Server *sv, Conv *conv, int lanes, const Weights *w, const ForwardOptions *fo, const char *dspark,
                  char err[FORWARD_ERR])
{
  if (!(sv->f = forward_create_with(w, fo, err)) || !forward_lane(sv->f, 0, err))
    return false;
  if (dspark && !(sv->d = drafter_create(sv->f, dspark, err)))
    return false;
  for (int i = 0; i < lanes; i++)
    if (!conv_create(&conv[i], sv))
      return snprintf(err, FORWARD_ERR, "%s", conv[i].err), false;
  return true;
}

typedef struct Sizing
{
  Server *sv;
  Conv *conv;
  int lanes;
  const Weights *w;
  ForwardOptions fo;
  const char *dspark;
} Sizing;

static bool sized_build(void *ctx, uint64_t budget, uint64_t *built, uint64_t *room, char err[FIT_ERR])
{
  Sizing *z = ctx;
  char ferr[FORWARD_ERR] = "";
  z->fo.budget = budget;
  if (!build(z->sv, z->conv, z->lanes, z->w, &z->fo, z->dspark, ferr) || !forward_room(z->sv->f, room))
  {
    serve_unbuild(z->sv, z->conv, z->lanes); /* the failed build's own (fit_sized unbuilt the ones before it) */
    return snprintf(err, FIT_ERR, "%s", ferr[0] ? ferr : "the GPU's free memory could not be read"), false;
  }
  *built = forward_budget(z->sv->f);
  return true;
}

static void sized_unbuild(void *ctx)
{
  Sizing *z = ctx;
  serve_unbuild(z->sv, z->conv, z->lanes);
}

bool serve_build(Server *sv, Conv *conv, int lanes, const Weights *w, ForwardOptions fo, const char *dspark,
                 char err[FORWARD_ERR])
{
  if (fo.budget) /* a budget given: as it is */
    return build(sv, conv, lanes, w, &fo, dspark, err);
  Sizing z = {sv, conv, lanes, w, fo, dspark};
  uint64_t budget = 0, room = 0;
  char ferr[FIT_ERR] = "";
  if (!fit_sized(sized_build, sized_unbuild, &z, FIT_HEADROOM, &budget, ferr))
    return snprintf(err, FORWARD_ERR, "%s", ferr), false;
  forward_room(sv->f, &room);
  out("NOTE the budget sized to this GPU: %.2f GiB (%.2f GiB left free)\n", (double)budget / (1ull << 30),
      (double)room / (1ull << 30));
  return true;
}
