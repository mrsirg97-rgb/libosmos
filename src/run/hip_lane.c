#include <stdlib.h>

#include "forward_impl.h"

/* forward_lane's small steps: each (token, routed expert) pair m - the router's order, token-major - through the
   matvec's group (gate and up on the token's FFN input, the lane's activation with the pair's weight, then down),
   each token's pairs then summed in ascending expert id (ref.py's index_add order), the shared expert after */

#define LANE_PAIRS OPS_MAX_EXPERTS /* a small step's pairs: fewer than a layer's experts */

static int cmp_pair(const void *a, const void *b)
{
  const int32_t *x = a, *y = b;
  return (x[0] > y[0]) - (x[0] < y[0]);
}

bool forward_lane(Forward *f, int cpu_threads, char err[FORWARD_ERR])
{
  if (!f || cpu_threads != 0)
    return fwd_fail(err, "no forward, or CPU threads (the GPU's half only, so far)");
  const uint64_t bytes = 3 * LANE_PAIRS * sizeof(void *) + LANE_PAIRS * sizeof(int32_t);
  if (!f->lane_up && (hipHostMalloc((void **)&f->lane_up, bytes, 0) != hipSuccess || !fwd_alloc(&f->lane_dev, bytes)))
    return fwd_fail(err, "memory refused (the lane's tables)");
  f->lane = true, f->lane_threads = cpu_threads;
  return true;
}

bool fwd_lane_experts(Forward *f, int l, const int32_t *ids, uint32_t picks, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t D = g->n_embd, FF = g->expert_ff, E = g->n_experts, K = g->n_used, n = f->n;
  if (picks > LANE_PAIRS)
    return fwd_fail(err, "a small step of more pairs than the lane's tables");
  const void **tab = (const void **)f->lane_up;
  int32_t *order = (int32_t *)(f->lane_up + 3 * LANE_PAIRS * sizeof(void *));
  for (uint32_t m = 0; m < picks; m++)
  {
    const uint32_t u = (uint32_t)l * E + (uint32_t)ids[m];
    const void *ex[POOL_PARTS];
    const bool ok = f->rix[u] >= 0 ? resident_dev(f->res, (size_t)f->rix[u], ex) : fwd_pinned(f, u, ex, err);
    if (!ok)
      return err[0] ? false : fwd_fail(err, "an expert not in GPU memory");
    tab[m] = ex[0], tab[LANE_PAIRS + m] = ex[1], tab[2 * LANE_PAIRS + m] = ex[2];
  }
  for (uint32_t i = 0; i < n; i++) /* a token's pairs in ascending expert id */
  {
    int32_t ord[OPS_MAX_TOPK][2];
    for (uint32_t k = 0; k < K; k++)
      ord[k][0] = ids[i * K + k], ord[k][1] = (int32_t)(i * K + k);
    qsort(ord, K, sizeof *ord, cmp_pair);
    for (uint32_t k = 0; k < K; k++)
      order[i * K + k] = ord[k][1];
  }
  const void *const *dtab = (const void *const *)f->lane_dev;
  const int32_t *dorder = (const int32_t *)((uint8_t *)f->lane_dev + 3 * LANE_PAIRS * sizeof(void *));
  const bool ok =
      fwd_put_async(f->lane_dev, f->lane_up, 3 * LANE_PAIRS * sizeof(void *) + (uint64_t)picks * 4) &&
      mv_group(L->exp_gate->type, dtab, picks, FF, D, b->xf, D, K, b->ga, NULL) &&
      mv_group(L->exp_up->type, dtab + LANE_PAIRS, picks, FF, D, b->xf, D, K, b->upr, NULL) &&
      ops_swiglu_lane(b->ga, b->upr, FF, g->clamp_exp[l], b->rw, picks, b->hh, NULL) &&
      mv_group(L->exp_down->type, dtab + 2 * LANE_PAIRS, picks, D, FF, b->hh, FF, 1, b->ye, NULL) &&
      ops_gather_sum(b->moe, b->ye, dorder, K, n, D, NULL) && ops_add(b->moe, b->sye, (uint64_t)n * D, NULL);
  return ok || fwd_fail(err, "a GPU call failed (the lane's experts)");
}
