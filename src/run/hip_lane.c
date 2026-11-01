#include <stdlib.h>
#include <string.h>

#include "forward_impl.h"
#include "lane.h"
#include "substrate/gpu/ops.h"
#include "substrate/gpu/q8.h"

/* forward_lane's small steps: the step's (token, routed expert) pairs in rows by expert (then token), each expert -
   an entry, or a few of them past MV_MAX_TOKENS tokens - once for its rows through the matvec's group (gate and up
   on the rows' FFN inputs, the lane's activation with each row's weight - over int8 inputs one launch - then down),
   each token's rows then summed in ascending expert id (ref.py's index_add order), the shared expert after. a row's
   sums are a token's alone */


/* the tables a step sends, laid out in lane_up (pinned) and lane_dev alike */
typedef struct LaneTabs
{
  const void *w[3][LANE_PAIRS]; /* an entry's gate, up, down */
  int32_t cnt[LANE_PAIRS], xrow[LANE_PAIRS];
  int32_t tok[LANE_PAIRS], pair[LANE_PAIRS]; /* a row's token, and its pair (the router's order) */
  int32_t order[LANE_PAIRS];                 /* each token's rows in ascending expert id */
  int32_t cpu_row[LANE_PAIRS];               /* the rows the CPU computes (the lane's CPU half) */
} LaneTabs;

bool forward_lane(Forward *f, int cpu_threads, char err[FORWARD_ERR])
{
  if (!f || cpu_threads < 0 || cpu_threads > CPU_LANE_THREADS_MAX)
    return fwd_fail(err, "no forward, or a count of CPU threads it does not take");
  const uint64_t D = f->g->n_embd;
  if (!f->lane_up &&
      (hipHostMalloc((void **)&f->lane_up, sizeof(LaneTabs), 0) != hipSuccess ||
       !fwd_alloc(&f->lane_dev, sizeof(LaneTabs)) || !fwd_alloc(&f->lane_x, LANE_ROWS * D * sizeof(float)) ||
       !fwd_alloc(&f->lane_w, LANE_ROWS * sizeof(float)) || !fwd_alloc(&f->lane_q, LANE_ROWS * D) ||
       !fwd_alloc(&f->lane_qd, LANE_ROWS * (D / LANE_Q8_VALUES) * sizeof(float))))
    return fwd_fail(err, "memory refused (the lane's tables)");
  if (cpu_threads && !f->cpu && !fwd_cpu_create(f, cpu_threads, err))
    return false;
  f->lane = true, f->lane_threads = cpu_threads;
  return true;
}

static int cmp_row(const void *a, const void *b)
{
  const int32_t *x = a, *y = b; /* (expert, pair): by expert, then pair */
  return x[0] != y[0] ? (x[0] > y[0]) - (x[0] < y[0]) : (x[1] > y[1]) - (x[1] < y[1]);
}

/* an entry's place: the ready ones before the reading ones, each by its count */
static uint64_t entry_key(const LaneTabs *t, const uint32_t *unit_of, uint32_t k)
{
  return (uint64_t)(unit_of[k] != UINT32_MAX) << 32 | (uint32_t)t->cnt[k];
}

/* entries a .. z (in ascending count: a launch a count) through gate and up and the activation (over int8 inputs one
   launch for both, q8_gate_up, when the two are one type; else each, then the activation over every row), the
   activations to int8 (every row: a ready row's again the same bits), then down. their rows into ye */
static bool entries(Forward *f, int l, uint32_t picks, uint32_t a0, uint32_t z0, bool q8)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const LaneTabs *t = (const LaneTabs *)f->lane_up, *d = (const LaneTabs *)f->lane_dev;
  const uint32_t D = g->n_embd, FF = g->expert_ff;
  const bool fused = q8 && L->exp_gate->type == L->exp_up->type;
  if (a0 == z0)
    return true;
  bool ok = true;
  for (uint32_t a = a0, z; ok && a < z0; a = z)
  {
    for (z = a + 1; z < z0 && t->cnt[z] == t->cnt[a];)
      z++;
    const uint32_t c = (uint32_t)t->cnt[a];
    if (fused)
      ok = q8_gate_up(L->exp_gate->type, d->w[0] + a, d->w[1] + a, d->cnt + a, d->xrow + a, z - a, c, FF, D, f->lane_q,
                      f->lane_qd, f->lane_w, g->clamp_exp[l], b->hh, FF, NULL);
    else
      ok = q8 ? q8_group_tok(L->exp_gate->type, d->w[0] + a, d->cnt + a, d->xrow + a, z - a, c, FF, D, f->lane_q,
                             f->lane_qd, b->ga, FF, NULL) &&
                    q8_group_tok(L->exp_up->type, d->w[1] + a, d->cnt + a, d->xrow + a, z - a, c, FF, D, f->lane_q,
                                 f->lane_qd, b->upr, FF, NULL)
              : mv_group_tok(L->exp_gate->type, d->w[0] + a, d->cnt + a, d->xrow + a, z - a, c, FF, D, f->lane_x, D,
                             b->ga, FF, NULL) &&
                    mv_group_tok(L->exp_up->type, d->w[1] + a, d->cnt + a, d->xrow + a, z - a, c, FF, D, f->lane_x, D,
                                 b->upr, FF, NULL);
  }
  ok = ok && (fused || ops_swiglu_lane(b->ga, b->upr, FF, g->clamp_exp[l], f->lane_w, picks, b->hh, NULL));
  ok = ok && (!q8 || q8_round(b->hh, picks, FF, FF, f->lane_q, f->lane_qd, NULL));
  for (uint32_t a = a0, z; ok && a < z0; a = z) /* down */
  {
    for (z = a + 1; z < z0 && t->cnt[z] == t->cnt[a];)
      z++;
    const uint32_t c = (uint32_t)t->cnt[a];
    ok = q8 ? q8_group_tok(L->exp_down->type, d->w[2] + a, d->cnt + a, d->xrow + a, z - a, c, D, FF, f->lane_q,
                           f->lane_qd, b->ye, D, NULL)
            : mv_group_tok(L->exp_down->type, d->w[2] + a, d->cnt + a, d->xrow + a, z - a, c, D, FF, b->hh, FF,
                           b->ye, D, NULL);
  }
  return ok;
}

bool fwd_lane_experts(Forward *f, int l, const int32_t *ids, uint32_t picks, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t D = g->n_embd, E = g->n_experts, K = g->n_used, n = f->n;
  if (picks > LANE_PAIRS)
    return fwd_fail(err, "a small step of more pairs than the lane's tables");
  LaneTabs *t = (LaneTabs *)f->lane_up;
  int32_t rows[LANE_PAIRS][2];
  for (uint32_t m = 0; m < picks; m++)
    rows[m][0] = ids[m], rows[m][1] = (int32_t)m;
  qsort(rows, picks, sizeof *rows, cmp_row);
  /* an expert's rows: the CPU's (fwd_moe marked it this visit) one entry of them all, the GPU's entries of at most
     MV_MAX_TOKENS */
  uint32_t ne = 0, nc = 0, n_cpu = 0, row_of[LANE_PAIRS], unit_of[LANE_PAIRS]; /* unit_of: a reading entry's */
  bool on_cpu = false;
  for (uint32_t r = 0; r < picks; r++)
  {
    const int32_t m = rows[r][1];
    t->tok[r] = m / (int32_t)K, t->pair[r] = m, row_of[m] = r;
    const bool same = r && rows[r][0] == rows[r - 1][0];
    if (same && on_cpu)
    {
      f->cpu_e[nc - 1].cnt++, t->cpu_row[n_cpu++] = (int32_t)r;
      continue;
    }
    if (same && (uint32_t)t->cnt[ne - 1] < MV_MAX_TOKENS)
    {
      t->cnt[ne - 1]++;
      continue;
    }
    const uint32_t u = (uint32_t)l * E + (uint32_t)rows[r][0];
    if ((on_cpu = f->cpu && f->on_cpu[u] == f->visits))
    {
      const void *part[POOL_PARTS];
      fwd_cpu_parts(f, u, part);
      f->cpu_e[nc] = (CpuLaneExpert){part[0], part[1], part[2], r, 1}, f->cpu_unit[nc++] = u;
      t->cpu_row[n_cpu++] = (int32_t)r;
      continue;
    }
    /* in GPU memory now (resident, or read into the pool): its pointers; else still reading - waited for after the
       ready ones are launched */
    const void *ex[POOL_PARTS] = {NULL, NULL, NULL};
    const bool ready = f->rix[u] >= 0 ? resident_dev(f->res, (size_t)f->rix[u], ex) : pool_dev(f->pool, u, ex);
    t->w[0][ne] = ex[0], t->w[1][ne] = ex[1], t->w[2][ne] = ex[2];
    unit_of[ne] = ready ? UINT32_MAX : u;
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
  /* the entries in GPU memory first, then the reading ones; each part in ascending count: a launch a count (each its
     own token width, as mm's column tiles: an expert of one token is not run at another's six) */
  for (uint32_t a = 1; a < ne; a++)
    for (uint32_t c = a; c > 0 && entry_key(t, unit_of, c) < entry_key(t, unit_of, c - 1); c--)
    {
      for (int r = 0; r < 3; r++)
      {
        const void *w = t->w[r][c];
        t->w[r][c] = t->w[r][c - 1], t->w[r][c - 1] = w;
      }
      const int32_t cc = t->cnt[c], xr = t->xrow[c];
      const uint32_t uu = unit_of[c];
      t->cnt[c] = t->cnt[c - 1], t->xrow[c] = t->xrow[c - 1], unit_of[c] = unit_of[c - 1];
      t->cnt[c - 1] = cc, t->xrow[c - 1] = xr, unit_of[c - 1] = uu;
    }
  uint32_t nr = 0; /* the entries ready: 0 .. nr */
  while (nr < ne && unit_of[nr] == UINT32_MAX)
    nr++;
  /* the CPU's experts given first: its threads run while the GPU runs the others */
  if (nc && !fwd_cpu_submit(f, l, picks, nc, t->tok, t->pair, err))
    return false;
  const LaneTabs *d = (const LaneTabs *)f->lane_dev;
  bool ok = fwd_put_async(f->lane_dev, f->lane_up, sizeof(LaneTabs)) &&
            ops_gather_rows(f->lane_x, b->xf, d->tok, picks, D, NULL) &&
            ops_gather_rows(f->lane_w, b->rw, d->pair, picks, 1, NULL);
  /* the experts over int8 inputs (q8.h: the IQ types), else the F32 matvec; either way a token's rows its own */
  const bool q8 = lane_q8_types(L->exp_gate->type, L->exp_up->type, L->exp_down->type);
  ok = ok && (!q8 || q8_round(f->lane_x, picks, D, D, f->lane_q, f->lane_qd, NULL));
  ok = ok && entries(f, l, picks, 0, nr, q8); /* the ready ones, then the reading ones as their reads end */
  for (uint32_t k = nr; ok && k < ne; k++)
  {
    const void *ex[POOL_PARTS];
    if (!fwd_pinned(f, unit_of[k], ex, err)) /* (its err set) */
      return false;
    t->w[0][k] = ex[0], t->w[1][k] = ex[1], t->w[2][k] = ex[2];
  }
  for (int r = 0; ok && r < 3 && nr < ne; r++)
    ok = fwd_put_async((void *)(d->w[r] + nr), t->w[r] + nr, (ne - nr) * sizeof *t->w[r]);
  /* x to int8 again before the reading ones: the ready ones' activations took its buffer (the same bytes again) */
  ok = ok && (!q8 || !nr || nr == ne || q8_round(f->lane_x, picks, D, D, f->lane_q, f->lane_qd, NULL));
  ok = ok && entries(f, l, picks, nr, ne, q8);
  /* the CPU's rows: the stream waits for their post, then takes them into ye beside the GPU's */
  ok = ok && (!nc || (ops_wait(f->cpu_seq_dev, f->cpu_value, NULL) &&
                      ops_copy_rows(b->ye, f->cpu_y_dev, d->cpu_row, n_cpu, D, NULL)));
  ok = ok && ops_gather_sum_add(b->moe, b->ye, d->order, K, n, D, b->sye, NULL);
  return ok || fwd_fail(err, "a GPU call failed (the lane's experts)");
}
