#ifndef OSMOS_RUN_AHEAD_H
#define OSMOS_RUN_AHEAD_H

#include <stdbool.h>
#include <stdint.h>

uint32_t ahead_union(const int32_t *guess, uint32_t tokens, uint32_t per_token, uint32_t k, uint32_t n_experts,
                     bool *named);

#endif

/*
ahead_union
  the experts a step's guess names (the next layer's router on this layer's input: each token's per_token experts,
  best first, [token][per_token]): named[e] true for each expert among some token's first k (k at most per_token),
  false for the rest of n_experts; ids out of range skipped. returns how many are named. a prompt's read ahead reads
  these, not every expert of the next layer - over a 2,048-token step every token's 6 name 98.9% of the experts it
  picks that are not in memory, and the experts not named are 41% of a read ahead of all (UD-IQ2_XXS: DESIGN.md,
  "0.5.8").
*/
