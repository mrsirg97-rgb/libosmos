#include "osmos/q4x/ple.h"

void q4x_ple_rows(const Q4xGeometry *g, int32_t token, const int32_t *prev, int n_prev,
                  int32_t rows[Q4X_MAX_PLE_HEADS])
{
  uint64_t ctx[Q4X_MAX_NGRAM];
  ctx[0] = (uint64_t)(int64_t)token;
  int cut = 0;
  for (int s = 1; s < g->ple_ngram; s++)
  {
    const int32_t t = cut || s > n_prev ? -1 : prev[s - 1];
    cut = cut || t < 0 || t == g->ple_eos;
    ctx[s] = (uint64_t)(int64_t)(cut ? g->ple_eos : t);
  }
  for (int n = 2; n <= g->ple_ngram; n++)
  {
    uint64_t mixed = ctx[0] * (uint64_t)g->ple_mult[0];
    for (int j = 1; j < n; j++)
      mixed ^= ctx[j] * (uint64_t)g->ple_mult[j];
    for (int k = 0; k < g->ple_heads_per_ngram; k++)
    {
      const int h = (n - 2) * g->ple_heads_per_ngram + k;
      rows[h] = (int32_t)(mixed % g->ple_rows[h] + g->ple_offset[h]);
    }
  }
}
