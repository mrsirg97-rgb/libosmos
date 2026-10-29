#ifndef WEIGHTS_GEO_H
#define WEIGHTS_GEO_H

#include <stdbool.h>

#include "substrate/model/weights.h"

/* the model's metadata into g, checked (weights.h, weights_create); false with the key and why in err */
bool weights_geometry(const Gguf *model, const Gguf *engram, Geometry *g, char err[WEIGHTS_ERR]);

#endif
