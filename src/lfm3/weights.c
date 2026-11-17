/* LFM3's geometry and weights (osmos/lfm3/forward.h): refused until step 1's implementation (DESIGN.md, "vibes-flash:
   LFM3") - its tests fail at run time, the tree builds */
#include <stdio.h>
#include <stdlib.h>

#include "osmos/lfm3/forward.h"

bool lfm3_geometry(const Gguf *model, Lfm3Geometry *g, char err[LFM3_ERR])
{
  (void)model, (void)g;
  snprintf(err, LFM3_ERR, "lfm3: the geometry is not built yet");
  return false;
}

Lfm3Weights *lfm3_weights_create(const Gguf *model, char err[LFM3_ERR])
{
  Lfm3Geometry g;
  if (!lfm3_geometry(model, &g, err))
    return NULL;
  return NULL;
}

void lfm3_weights_destroy(Lfm3Weights *w)
{
  free(w);
}
