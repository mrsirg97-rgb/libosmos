#include <stdlib.h>
#include <string.h>

#include "forward_impl.h"

/* forward_lane's small steps: the step's (token, routed expert) pairs in rows by expert (then token), each expert -
   an entry, or a few of them past MV_MAX_TOKENS tokens - once for its rows through the matvec's group (gate and up on
   the rows' FFN inputs, the lane's activation with each row's weight, then down), each token's rows then summed in
   ascending expert id (ref.py's index_add order), the shared expert after. a row's sums are a token's alone */

#define LANE_PAIRS OPS_MAX_EXPERTS                    /* a small step's pairs: fewer than a layer's experts */
#define LANE_ROWS (LANE_PAIRS + MV_MAX_TOKENS)        /* the rows' buffers: an entry's rows past it read, not used */

/* the tables a step sends, laid out in lane_up (pinned) and lane_dev alike */
typedef struct LaneTabs
{
  const void *w[3][LANE_PAIRS]; /* an entry's gate, up, down */
  int32_t cnt[LANE_PAIRS], xrow[LANE_PAIRS];
  int32_t tok[LANE_PAIRS], pair[LANE_PAIRS]; /* a row's token, and its pair (the router's order) */
  int32_t order[LANE_PAIRS];                 /* each token's rows in ascending expert id */
} LaneTabs;

bool forward_lane(Forward *f, int cpu_threads, char err[FORWARD_ERR])
{
  if (!f || cpu_threads != 0)
    return fwd_fail(err, "no forward, or CPU threads (the GPU's half only, so far)");
  const uint64_t D = f->g->n_embd;
  if (!f->lane_up &&
      (hipHostMalloc((void **)&f->lane_up, sizeof(LaneTabs), 0) != hipSuccess ||
       !fwd_alloc(&f->lane_dev, sizeof(LaneTabs)) || !fwd_alloc(&f->lane_x, LANE_ROWS * D * sizeof(float)) ||
       !fwd_alloc(&f->lane_w, LANE_ROWS * sizeof(float))))
    return fwd_fail(err, "memory refused (the lane's tables)");
  f->lane = true, f->lane_threads = cpu_threads;
  return true;
}

static int cmp_row(const void *a, const void *b)
{
  const int32_t *x = a, *y = b; /* (expert, pair): by expert, then pair */
  return x[0] != y[0] ? (x[0] > y[0]) - (x[0] < y[0]) : (x[1] > y[1]) - (x[1] < y[1]);
}

bool fwd_lane_experts(Forward *f, int l, const int32_t *ids, uint32_t picks, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t D = g->n_embd, FF = g->expert_ff, E = g->n_experts, K = g->n_used, n = f->n;
  if (picks > LANE_PAIRS)
    return fwd_fail(err, "a small step of more pairs than the lane's tables");
  LaneTabs *t = (LaneTabs *)f->lane_up;
  int32_t rows[LANE_PAIRS][2];
  for (uint32_t m = 0; m < picks; m++)
    rows[m][0] = ids[m], rows[m][1] = (int32_t)m;
  qsort(rows, picks, sizeof *rows, cmp_row);
  uint32_t ne = 0, row_of[LANE_PAIRS];
  for (uint32_t r = 0; r < picks; r++)
  {
    const int32_t m = rows[r][1];
    t->tok[r] = m / (int32_t)K, t->pair[r] = m, row_of[m] = r;
    if (r && rows[r][0] == rows[r - 1][0] && (uint32_t)t->cnt[ne - 1] < MV_MAX_TOKENS)
    {
      t->cnt[ne - 1]++;
      continue;
    }
    const uint32_t u = (uint32_t)l * E + (uint32_t)rows[r][0];
    const void *ex[POOL_PARTS];
    if (!(f->rix[u] >= 0 ? resident_dev(f->res, (size_t)f->rix[u], ex) : fwd_pinned(f, u, ex, err)))
      return err[0] ? false : fwd_fail(err, "an expert not in GPU memory");
    t->w[0][ne] = ex[0], t->w[1][ne] = ex[1], t->w[2][ne] = ex[2];
    t->cnt[ne] = 1, t->xrow[ne] = (int32_t)r, ne++;
  }
  for (uint32_t i = 0; i < n; i++) /* a token's rows: its pairs' in ascending expert id (rows ascend by expert) */
  {
    uint32_t k0 = i * K;
    int32_t r[OPS_MAX_TOPK];
    for (uint32_t k = 0; k < K; k++)
      r[k] = (int32_t)row_of[k0 + k];
    for (uint32_t a = 1; a < K; a++) /* (K is a few: insertion) */
      for (uint32_t c = a; c > 0 && r[c] < r[c - 1]; c--)
      {
        const int32_t s = r[c];
        r[c] = r[c - 1], r[c - 1] = s;
      }
    memcpy(&t->order[k0], r, K * sizeof *r);
  }
  /* the entries in ascending count: a launch a count (each its own token width, as mm's column tiles: an expert of
     one token is not run at another's six) */
  for (uint32_t a = 1; a < ne; a++)
    for (uint32_t c = a; c > 0 && t->cnt[c] < t->cnt[c - 1]; c--)
    {
      for (int r = 0; r < 3; r++)
      {
        const void *w = t->w[r][c];
        t->w[r][c] = t->w[r][c - 1], t->w[r][c - 1] = w;
      }
      const int32_t cc = t->cnt[c], xr = t->xrow[c];
      t->cnt[c] = t->cnt[c - 1], t->xrow[c] = t->xrow[c - 1], t->cnt[c - 1] = cc, t->xrow[c - 1] = xr;
    }
  const LaneTabs *d = (const LaneTabs *)f->lane_dev;
  bool ok = fwd_put_async(f->lane_dev, f->lane_up, sizeof(LaneTabs)) &&
            ops_gather_rows(f->lane_x, b->xf, d->tok, picks, D, NULL) &&
            ops_gather_rows(f->lane_w, b->rw, d->pair, picks, 1, NULL);
  for (uint32_t a = 0, z; ok && a < ne; a = z) /* gate and up, a launch a count */
  {
    for (z = a + 1; z < ne && t->cnt[z] == t->cnt[a];)
      z++;
    const uint32_t c = (uint32_t)t->cnt[a];
    ok = mv_group_tok(L->exp_gate->type, d->w[0] + a, d->cnt + a, d->xrow + a, z - a, c, FF, D, f->lane_x, D, b->ga,
                      FF, NULL) &&
         mv_group_tok(L->exp_up->type, d->w[1] + a, d->cnt + a, d->xrow + a, z - a, c, FF, D, f->lane_x, D, b->upr,
                      FF, NULL);
  }
  ok = ok && ops_swiglu_lane(b->ga, b->upr, FF, g->clamp_exp[l], f->lane_w, picks, b->hh, NULL);
  for (uint32_t a = 0, z; ok && a < ne; a = z) /* down */
  {
    for (z = a + 1; z < ne && t->cnt[z] == t->cnt[a];)
      z++;
    ok = mv_group_tok(L->exp_down->type, d->w[2] + a, d->cnt + a, d->xrow + a, z - a, (uint32_t)t->cnt[a], D, FF,
                      b->hh, FF, b->ye, D, NULL);
  }
  ok = ok && ops_gather_sum_add(b->moe, b->ye, d->order, K, n, D, b->sye, NULL);
  return ok || fwd_fail(err, "a GPU call failed (the lane's experts)");
}
