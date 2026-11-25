/* the hy4 latent MoE (hy4_impl.h's hy4_moe): the router, the routed experts in the latent - each expert read once for
   the tokens that picked it, a decode's on the matvec (mv_group_tok), a prompt's on the matrix cores
   (mm_group over each expert's gathered tokens); a decode's tables built on the GPU (hy4_route), a read's on the host - the shared expert, the combine norm. a token's sums its own in
   either: its picks' order and its sum's (slot order) its own */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/hy4_ops.h"
#include "osmos/model/moe_route.h"
#include "osmos/gpu/mm.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/model/qtype.h"
#include "hy4_impl.h"

uint64_t hy4_read_entries(uint64_t n, const Hy4Geometry *g)
{
  const uint64_t picks = n * (uint64_t)g->n_used;
  return picks + (uint64_t)g->n_experts; /* each expert's chunks: its picks / MM_MAX_TOKENS + 1 at most */
}

uint64_t hy4_read_slots(uint64_t n, const Hy4Geometry *g)
{
  return n * (uint64_t)g->n_used + (uint64_t)g->n_experts * (MM_TILE - 1); /* each expert's padded to a tile */
}

/* each run's expert in memory, its parts' addresses into f->h_ex[run]: every expert resident - the address table's
   host copy; else the membrane's (the layer visited - the GPU past the last one's experts: the picks' copy waited for
   it - a decode's picks scored, each run's expert held, then waited for) */
static bool in_memory(Hy4Forward *f, int l, const MoeRoute *rt, uint32_t picks)
{
  const uint32_t E = (uint32_t)f->w->geo.n_experts, ml = (uint32_t)(l - f->w->geo.n_dense);
  if (f->whole)
  {
    for (uint32_t j = 0; j < rt->runs; j++)
      for (int k = 0; k < 3; k++)
        f->h_ex[j][k] = f->h_tab[((uint64_t)ml * 3 + (uint64_t)k) * E + rt->expert[j]];
    return true;
  }
  char merr[OSMOS_ERR] = "";
  Membrane *m = f->ex.mem;
  if (!membrane_visit(m, ml, picks >= E, merr))
    return false;
  for (uint32_t j = 0; picks < E && j < rt->runs; j++)
    for (uint32_t q = 0; q < rt->count[j]; q++)
      membrane_score(m, ml * E + rt->expert[j]);
  for (uint32_t j = 0; j < rt->runs; j++) /* every distinct pick held first: the cold ones' reads under way together */
    if (!membrane_hold(m, ml * E + rt->expert[j], merr))
      return false;
  for (uint32_t j = 0; j < rt->runs; j++)
    if (!membrane_wait(m, ml * E + rt->expert[j], f->h_ex[j], merr))
      return false;
  return true;
}

/* a read's picks grouped by expert on the host (moe_route: ascending; within one, its tokens ascending): each sorted
   row's token and weight, each (token, slot)'s row; mm_group's entries (moe_entries: chunks of MM_MAX_TOKENS, each
   expert's slots a whole number of tiles, the entries ordered by count so a launch is a run of the same column
   tiles). a decode's tables are the GPU's (hy4_route: no host between the router and the experts) */
static bool route(Hy4Forward *f, int l, uint32_t n, uint32_t *entries, uint32_t *slots)
{
  const Hy4Geometry *g = &f->w->geo;
  const uint32_t K = (uint32_t)g->n_used, E = (uint32_t)g->n_experts, nk = n * K;
  const uint32_t lat = (uint32_t)g->latent, ffe = (uint32_t)g->ff_exp;
  if (hipMemcpy(f->h_idx, f->idx, (size_t)nk * 4, hipMemcpyDeviceToHost) != hipSuccess ||
      hipMemcpy(f->h_w, f->wtok, (size_t)nk * F, hipMemcpyDeviceToHost) != hipSuccess)
    return false;
  MoeRoute rt = {f->h_by, (uint32_t *)f->h_sidx, f->h_run, f->h_run + E, f->h_run + 2 * E, f->h_run + 3 * E,
                 f->h_rgidx, f->h_rhidx, f->h_estart, 0, 0};
  uint32_t e_n = 0;
  if (!moe_route(f->h_idx, nk, K, E, MM_TILE, &rt) || !moe_entries(&rt, MM_MAX_TOKENS, f->h_ent, f->rkey, &e_n))
    return false;
  if (!in_memory(f, l, &rt, nk))
    return false;
  for (uint32_t q = 0; q < nk; q++)
    f->h_gidx[q] = (int32_t)(rt.by[q] / K), f->h_wsel[q] = f->h_w[rt.by[q]];
  *entries = e_n, *slots = rt.slots;
  const uint64_t ecap = hy4_read_entries(f->max_tokens, g);
  const void **pw = (const void **)f->h_stage; /* the sorted tables: 3 x ecap pointers, then 5 x ecap counts */
  uint32_t *pu = (uint32_t *)(pw + 3 * ecap);
  for (uint32_t k = 0; k < e_n; k++)
  {
    const MoeEntry *en = &f->h_ent[k];
    const uint32_t slot = rt.slot0[en->run] + en->off, row = rt.start[en->run] + en->off;
    for (int m = 0; m < 3; m++)
      pw[(uint64_t)m * ecap + k] = f->h_ex[en->run][m];
    f->h_rcnt[k] = (int32_t)en->cnt, pu[ecap + k] = slot * lat, pu[2 * ecap + k] = row * ffe;
    pu[3 * ecap + k] = slot * ffe, pu[4 * ecap + k] = row * lat;
  }
  return hipMemcpy(f->sidx, f->h_sidx, (size_t)nk * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->wsel, f->h_wsel, (size_t)nk * F, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->gidx, f->h_gidx, (size_t)nk * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->rw, pw, 3 * ecap * sizeof(void *), hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->rcnt, f->h_rcnt, (size_t)e_n * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->rx, pu + ecap, (size_t)e_n * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->ry, pu + 2 * ecap, (size_t)e_n * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->rxd, pu + 3 * ecap, (size_t)e_n * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->ryd, pu + 4 * ecap, (size_t)e_n * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->rgidx, f->h_rgidx, (size_t)rt.slots * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->rhidx, f->h_rhidx, (size_t)rt.slots * 4, hipMemcpyHostToDevice) == hipSuccess;
}

/* a decode's experts' matvec over the entries (chunk of an expert's tokens an entry at most: the step's tokens, up to
   MV_MAX_TOKENS - one token's decode the single-token kernel, not the 8-token one's idle lanes; a token's sums its own
   whatever the chunk) */
static bool mv_experts(const GgufTensor *t, const void *const *w, const Hy4Forward *f, uint32_t entries, uint32_t chunk,
                       uint32_t rows, uint32_t cols, const float *x, float *y)
{
  return mv_group_tok(t->type, w, f->cnt, f->xrow, entries, chunk, rows, cols, x, cols, y, rows, f->st);
}

/* a read's one role of the experts (the entries in ascending count): a launch a run of the same column tiles */
static bool mm_experts(uint32_t type, const void *const *w, const int32_t *h_cnt, const int32_t *cnt, uint32_t ne,
                       uint32_t rows, uint32_t cols, const void *x16, const uint32_t *xo, float *y, const uint32_t *yo,
                       uint32_t ys, void *stream)
{
  bool ok = true;
  for (uint32_t a = 0, z = 0; ok && a < ne; a = z)
  {
    const uint32_t ct = mm_col_tiles(type, (uint32_t)h_cnt[a]);
    for (z = a + 1; z < ne && mm_col_tiles(type, (uint32_t)h_cnt[z]) == ct;)
      z++;
    ok = mm_group(type, w + a, cnt + a, z - a, (uint32_t)h_cnt[z - 1], rows, cols, x16, xo + a, cols, y, yo + a, ys,
                  stream);
  }
  return ok;
}

/* a decode's tables on the host (some experts not resident: the membrane's addresses), hy4_route's: the picks by
   expert ascending, an expert's tokens ascending - row r's token and weight, pick m's row; the entries each expert's
   rows in chunks (experts ascending), the count 0 past them to n_tok x k. a pick out of range or twice in a token
   refused (hy4_route's fault) */
static bool decode_tables(Hy4Forward *f, int l, uint32_t n, uint32_t chunk)
{
  const Hy4Geometry *g = &f->w->geo;
  const uint32_t K = (uint32_t)g->n_used, E = (uint32_t)g->n_experts, nk = n * K;
  if (hipMemcpy(f->h_idx, f->idx, (size_t)nk * 4, hipMemcpyDeviceToHost) != hipSuccess ||
      hipMemcpy(f->h_w, f->wtok, (size_t)nk * F, hipMemcpyDeviceToHost) != hipSuccess)
    return false;
  for (uint32_t m = 0; m < nk; m++)
    for (uint32_t s = m - m % K; s < m; s++)
      if (f->h_idx[s] == f->h_idx[m])
        return false;
  MoeRoute rt = {f->h_by, (uint32_t *)f->h_sidx, f->h_run, f->h_run + E, f->h_run + 2 * E, f->h_run + 3 * E,
                 f->h_rgidx, f->h_rhidx, f->h_estart, 0, 0};
  if (!moe_route(f->h_idx, nk, K, E, MM_TILE, &rt) || !in_memory(f, l, &rt, nk))
    return false;
  for (uint32_t q = 0; q < nk; q++)
    f->h_gidx[q] = (int32_t)(rt.by[q] / K), f->h_wsel[q] = f->h_w[rt.by[q]];
  int32_t *cnt = f->h_rcnt, *xrow = (int32_t *)f->h_rx;
  const void **wptr = f->h_rw;
  uint32_t ne = 0;
  for (uint32_t j = 0; j < rt.runs; j++)
    for (uint32_t off = 0; off < rt.count[j]; off += chunk, ne++)
    {
      cnt[ne] = (int32_t)(rt.count[j] - off < chunk ? rt.count[j] - off : chunk), xrow[ne] = (int32_t)(rt.start[j] + off);
      for (int k = 0; k < 3; k++)
        wptr[(uint64_t)k * nk + ne] = f->h_ex[j][k];
    }
  for (uint32_t i = ne; i < nk; i++)
    cnt[i] = 0;
  return hipMemcpy(f->gidx, f->h_gidx, (size_t)nk * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->sidx, f->h_sidx, (size_t)nk * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->wsel, f->h_wsel, (size_t)nk * F, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->cnt, cnt, (size_t)nk * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->xrow, xrow, (size_t)nk * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->wptr, wptr, 3 * (size_t)nk * sizeof *wptr, hipMemcpyHostToDevice) == hipSuccess;
}

static bool routed(Hy4Forward *f, int l, uint32_t n, bool read, const float *rin)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  const uint32_t K = (uint32_t)g->n_used, E = (uint32_t)g->n_experts, nk = n * K;
  const uint32_t lat = (uint32_t)g->latent, ffe = (uint32_t)g->ff_exp;
  if (!read)
  {
    const void **wg = (const void **)f->wptr, **wu = wg + nk, **wd = wg + 2 * nk;
    const uint32_t chunk = n < MV_MAX_TOKENS ? n : MV_MAX_TOKENS;
    const void *const *tab = f->exp_tab + (uint64_t)(l - g->n_dense) * 3 * E; /* (every expert resident) */
    double t = f->pf.on > 1 ? hy4_now() : 0;
    return (f->whole ? hy4_route(f->idx, f->wtok, n, K, E, chunk, tab, f->gidx, f->sidx, f->wsel, f->cnt, f->xrow, wg,
                                 f->fault, f->st)
                     : decode_tables(f, l, n, chunk)) &&
           ops_gather_rows(f->xg, rin, f->gidx, nk, lat, f->st) && hy4_part_done(f, 17, &t) &&
           mv_experts(L->exp_gate, wg, f, nk, chunk, ffe, lat, f->xg, f->eg_g) && hy4_part_done(f, 18, &t) &&
           mv_experts(L->exp_up, wu, f, nk, chunk, ffe, lat, f->xg, f->eg_u) && hy4_part_done(f, 19, &t) &&
           ops_swiglu(f->eg_g, f->eg_u, ffe, g->clamp[l], f->wsel, nk, f->eg_h, f->st) &&
           mv_experts(L->exp_down, wd, f, nk, chunk, lat, ffe, f->eg_h, f->eg_y) && hy4_part_done(f, 20, &t);
  }
  uint32_t entries = 0, slots = 0;
  if (!route(f, l, n, &entries, &slots))
    return false;
  const uint64_t ecap = hy4_read_entries(f->max_tokens, g);
  const void **rw = f->rw;
  return mm_x16_rows(rin, lat, f->rgidx, slots, lat, f->xe16, f->overflow, f->st) &&
         mm_experts(L->exp_gate->type, rw, f->h_rcnt, f->rcnt, entries, ffe, lat, f->xe16, f->rx, f->eg_g, f->ry, ffe, f->st) &&
         mm_experts(L->exp_up->type, rw + ecap, f->h_rcnt, f->rcnt, entries, ffe, lat, f->xe16, f->rx, f->eg_u, f->ry,
                    ffe, f->st) &&
         ops_swiglu(f->eg_g, f->eg_u, ffe, g->clamp[l], f->wsel, nk, f->eg_h, f->st) &&
         mm_x16_rows(f->eg_h, ffe, f->rhidx, slots, ffe, f->xe16, f->overflow, f->st) &&
         mm_experts(L->exp_down->type, rw + 2 * ecap, f->h_rcnt, f->rcnt, entries, lat, ffe, f->xe16, f->rxd, f->eg_y,
                    f->ryd, lat, f->st);
}

bool hy4_moe(Hy4Forward *f, uint32_t n, int l, bool read)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, E = (uint32_t)g->n_experts, K = (uint32_t)g->n_used;
  const uint32_t lat = (uint32_t)g->latent, sh = (uint32_t)g->ff_exp * (uint32_t)g->n_shexp;
  /* the matrices on xn: a token's one joint (its outputs in jo_ffn), else each its own */
  const bool joint = n == 1 && !read;
  const Hy4Joint *J = &f->j_ffn[l];
  float *logit = joint ? f->jo_ffn + J->off[0] : f->logit_r, *rin = joint ? f->jo_ffn + J->off[1] : f->rin;
  float *ga = joint ? f->jo_ffn + J->off[2] : f->ga, *up = joint ? f->jo_ffn + J->off[3] : f->up;
  bool on_xn = false;
  double t = f->pf.on > 1 ? hy4_now() : 0;
  const bool on_x = joint ? hy4_joint(f, J, D, f->xn, f->jo_ffn)
                          : hy4_mat(f, L->router, E, D, f->xn, n, logit, read) &&
                                hy4_mat_pair(f, L->routed_down, lat, rin, L->sh_gate, sh, ga, f->xn, D, n, read, &on_xn) &&
                                hy4_mat_pair(f, L->sh_up, sh, up, NULL, 0, NULL, f->xn, D, n, read, &on_xn);
  return on_x && hy4_part_done(f, 15, &t) &&
         ops_router_scored(logit, hy4_fv(f, L->router_bias), E, K, g->expert_scale, OPS_SCORE_SIGMOID, n, f->idx,
                           f->wtok, f->st) && hy4_part_done(f, 16, &t) &&
         routed(f, l, n, read, rin) && (t = f->pf.on > 1 ? hy4_now() : 0, true) &&
         ops_gather_sum(f->rsum, f->eg_y, f->sidx, K, n, lat, f->st) && hy4_part_done(f, 23, &t) &&
         hy4_mat(f, L->routed_up, D, lat, f->rsum, n, f->out, read) && hy4_part_done(f, 21, &t) &&
         ops_swiglu(ga, up, sh, INFINITY, NULL, n, f->act, f->st) &&
         hy4_mat(f, L->sh_down, D, sh, f->act, n, f->sh, read) &&
         ops_add(f->out, f->sh, (uint64_t)n * D, f->st) &&
         ops_rms(f->out, D, D, hy4_fv(f, L->post_ffn_norm), g->rms_eps, n, f->y, D, f->st) &&
         hy4_hc_post(f->h, f->y, f->post, (uint32_t)g->hc, D, n, f->st) && /* the routed and shared sum, its post norm, into the streams */
         hy4_part_done(f, 22, &t);
}
