#include "osmos/run/ahead.h"

uint32_t ahead_union(const int32_t *guess, uint32_t tokens, uint32_t per_token, uint32_t k, uint32_t n_experts,
                     bool *named)
{
  for (uint32_t e = 0; e < n_experts; e++)
    named[e] = false;
  uint32_t n = 0;
  k = k < per_token ? k : per_token;
  for (uint32_t t = 0; t < tokens; t++)
    for (uint32_t r = 0; r < k; r++)
    {
      const int32_t e = guess[(uint64_t)t * per_token + r];
      if (e >= 0 && (uint32_t)e < n_experts && !named[e])
        named[e] = true, n++;
    }
  return n;
}
