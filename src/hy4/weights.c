#include <stdio.h>

#include "osmos/hy4/weights.h"

/* step 1's schema (DESIGN.md, "0.10.0: Hy4-flash (hy4)"): refused until built */
static bool unbuilt(char *err)
{
  snprintf(err, HY4_ERR, "hy4: not built yet");
  return false;
}

bool hy4_geometry(const Gguf *model, Hy4Geometry *g, char err[HY4_ERR])
{
  (void)model, (void)g;
  return unbuilt(err);
}

Hy4Weights *hy4_weights_create(const Gguf *model, char err[HY4_ERR])
{
  (void)model;
  unbuilt(err);
  return NULL;
}

void hy4_weights_destroy(Hy4Weights *w) { (void)w; }

bool hy4_engram_shape(const Hy4Weights *w, EngramShape *s, char err[HY4_ERR])
{
  (void)w, (void)s;
  return unbuilt(err);
}
