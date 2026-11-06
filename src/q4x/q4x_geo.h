#pragma once

#include <stdbool.h>

#include "osmos/q4x/weights.h"

/* the model's metadata into g, checked (q4x_weights_create); false with the key and why in err */
bool q4x_geometry(const Gguf *model, Q4xGeometry *g, char err[Q4X_ERR]);
