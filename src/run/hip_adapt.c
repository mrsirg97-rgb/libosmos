#include <math.h>

#include "forward_impl.h"

/* the resident set follows the text. each expert's heat is its picks, halved every FWD_HEAT_HALF_LIFE steps (a
   step's own counted in it as they come). when a layer's experts have been queued, the hottest one the layer
   streamed (held in the pool, its bytes in GPU memory) takes the place of the layer's coldest resident one not
   picked in the step, when it is hotter by FWD_SWAP_MARGIN: its bytes copied into that one's memory on the stream,
   behind the layer's kernels (ops_copy: a kernel - hipMemcpyAsync took the pool's memory for the host's and its copy
   engine left the GPU idle ~0.45 ms a swap) - nothing before reads the cold one, everything after takes the new places. at most
   FWD_SWAPS_A_LAYER a layer a step. a layer's experts are one size (the same tensors' slices). which experts are
   resident changes no bit */

static double heat_now(const Forward *f, uint32_t u)
{
  const double past = f->heat[u] * exp2(-(double)(f->steps - f->heat_at[u]) / FWD_HEAT_HALF_LIFE);
  return past + (double)(f->picks[u] - f->seen[u]);
}

bool fwd_swap_layer(Forward *f, int l, char *err)
{
  const uint32_t E = (uint32_t)f->g->n_experts, lo = (uint32_t)l * E;
  for (int s = 0; s < FWD_SWAPS_A_LAYER; s++)
  {
    uint32_t in = UINT32_MAX, out = UINT32_MAX;
    double h_in = 0, h_out = INFINITY;
    for (uint32_t k = 0; k < f->n_held; k++) /* the layer's streamed experts, held */
    {
      const uint32_t u = f->held[k];
      const double h = heat_now(f, u);
      if (u >= lo && u < lo + E && f->rix[u] < 0 && h > h_in)
        in = u, h_in = h;
    }
    for (uint32_t u = lo; in != UINT32_MAX && u < lo + E; u++)
      if (f->rix[u] >= 0 && f->picks[u] == f->seen[u]) /* resident, and not picked this step */
      {
        const double h = heat_now(f, u);
        if (h < h_out)
          out = u, h_out = h;
      }
    if (out == UINT32_MAX || !(h_in > FWD_SWAP_MARGIN * h_out))
      return true;
    const void *src[POOL_PARTS], *dst[POOL_PARTS];
    if (!pool_dev(f->pool, in, src) || !resident_dev(f->res, (size_t)f->rix[out], dst))
      return fwd_fail(err, "a swap's expert not in GPU memory");
    for (int p = 0; p < f->units[in].n_parts; p++)
      if (f->units[in].part[p].bytes != f->units[out].part[p].bytes ||
          !ops_copy((void *)dst[p], src[p], f->units[in].part[p].bytes, NULL))
        return fwd_fail(err, "a swap's copy failed");
    f->rix[in] = f->rix[out], f->rix[out] = -1;
    f->swaps++;
  }
  return true;
}

void fwd_heat(Forward *f)
{
  for (uint32_t u = 0; u < f->n_expert_units; u++) /* the step's picks folded in, at its end */
    if (f->picks[u] != f->seen[u])
      f->heat[u] = heat_now(f, u), f->heat_at[u] = f->steps + 1, f->seen[u] = f->picks[u];
  f->steps++;
}
