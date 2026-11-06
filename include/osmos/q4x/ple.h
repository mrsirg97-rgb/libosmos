#ifndef OSMOS_Q4X_PLE_H
#define OSMOS_Q4X_PLE_H

#include <stdint.h>

#include "osmos/q4x/weights.h"

void q4x_ple_rows(const Q4xGeometry *g, int32_t token, const int32_t *prev, int n_prev,
                  int32_t rows[Q4X_MAX_PLE_HEADS]);

#endif

/*
q4x_ple_rows
  a token's rows of the n-gram table, a head each (DESIGN.md, "Qwen's forward"): the token and up to ple_ngram - 1
  predecessors (prev[0] the one before it, prev[1] the one before that; n_prev of them - a missing one, or any at or
  before an EOS, read as the EOS; the token's own EOS cuts nothing). for n = 2 .. ple_ngram: mixed = t0 m0 ^ .. ^
  t(n-1) m(n-1) (uint64, wrapping); its ple_heads_per_ngram heads' rows mixed % rows[h] + offset[h].
*/
