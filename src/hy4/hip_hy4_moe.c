/* the hy4 latent MoE (hy4_impl.h's hy4_moe): the router, the routed experts in the latent - each expert read once for
   the tokens that picked it, a decode's on the matvec (mv_group_tok), a prompt's on the matrix cores
   (mm_group over each expert's gathered tokens); a decode's tables built on the GPU (route_tok), a read's on the host - the shared expert, the combine norm. a token's sums its own in
   either: its picks' order and its sum's (slot order) its own */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/hy4_ops.h"
#include "osmos/gpu/route.h"
#include "osmos/model/moe_route.h"
#include "osmos/gpu/mm.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q8.h"
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

/* carved device pointers for a prefill MoE layer (one compact async H2D copy in route()) */
typedef struct Hy4ReadTab
{
  const void **wg, **wu, **wd;
  int32_t *rcnt, *rgidx, *rhidx, *sidx;
  uint32_t *rx, *ry, *rxd, *ryd;
  float *wsel;
} Hy4ReadTab;

/* carved device pointers for a streamed decode MoE layer (one compact async H2D copy in decode_tables()) */
typedef struct Hy4DecTab
{
  const void **wg, **wu, **wd;
  int32_t *gidx, *sidx, *cnt, *xrow;
  float *wsel;
  uint32_t ne;
} Hy4DecTab;

/* a read's picks grouped by expert on the host (moe_route: ascending; within one, its tokens ascending): each sorted
   row's weight, each (token, slot)'s row; mm_group's entries (moe_entries: chunks of MM_MAX_TOKENS, each expert's
   slots a whole number of tiles, ordered by count), packed into one compact pinned async H2D transfer */
static bool route(Hy4Forward *f, int l, uint32_t n, uint32_t *entries, uint32_t *slots, Hy4ReadTab *out)
{
  const Hy4Geometry *g = &f->w->geo;
  const uint32_t K = (uint32_t)g->n_used, E = (uint32_t)g->n_experts, nk = n * K;
  const uint32_t lat = (uint32_t)g->latent, ffe = (uint32_t)g->ff_exp;
  if (hipMemcpyAsync(f->pin_idx, f->idx, (size_t)nk * 4, hipMemcpyDeviceToHost, (hipStream_t)f->st) != hipSuccess ||
      hipMemcpyAsync(f->pin_w, f->wtok, (size_t)nk * F, hipMemcpyDeviceToHost, (hipStream_t)f->st) != hipSuccess ||
      hipStreamSynchronize((hipStream_t)f->st) != hipSuccess)
    return false;
  MoeRoute rt = {f->h_by, (uint32_t *)f->h_sidx, f->h_run, f->h_run + E, f->h_run + 2 * E, f->h_run + 3 * E,
                 f->h_rgidx, f->h_rhidx, f->h_estart, 0, 0};
  uint32_t e_n = 0;
  if (!moe_route(f->pin_idx, nk, K, E, MM_TILE, &rt) || !moe_entries(&rt, MM_MAX_TOKENS, f->h_ent, f->rkey, &e_n))
    return false;
  if (!in_memory(f, l, &rt, nk))
    return false;
  *entries = e_n, *slots = rt.slots;
  MoeStage st = {f->moe_build, f->moe_dev, 0, f->moe_cap};
  int32_t *sidx = moe_stage_carve(&st, (uint64_t)nk * 4, (void **)&out->sidx);
  float *wsel = moe_stage_carve(&st, (uint64_t)nk * F, (void **)&out->wsel);
  const void **wg = moe_stage_carve(&st, (uint64_t)e_n * sizeof(void *), (void **)&out->wg);
  const void **wu = moe_stage_carve(&st, (uint64_t)e_n * sizeof(void *), (void **)&out->wu);
  const void **wd = moe_stage_carve(&st, (uint64_t)e_n * sizeof(void *), (void **)&out->wd);
  int32_t *rcnt = moe_stage_carve(&st, (uint64_t)e_n * 4, (void **)&out->rcnt);
  uint32_t *rx = moe_stage_carve(&st, (uint64_t)e_n * 4, (void **)&out->rx);
  uint32_t *ry = moe_stage_carve(&st, (uint64_t)e_n * 4, (void **)&out->ry);
  uint32_t *rxd = moe_stage_carve(&st, (uint64_t)e_n * 4, (void **)&out->rxd);
  uint32_t *ryd = moe_stage_carve(&st, (uint64_t)e_n * 4, (void **)&out->ryd);
  int32_t *rgidx = moe_stage_carve(&st, (uint64_t)rt.slots * 4, (void **)&out->rgidx);
  int32_t *rhidx = moe_stage_carve(&st, (uint64_t)rt.slots * 4, (void **)&out->rhidx);
  if (!sidx || !wsel || !wg || !wu || !wd || !rcnt || !rx || !ry || !rxd || !ryd || !rgidx || !rhidx)
    return false;
  memcpy(sidx, f->h_sidx, (size_t)nk * 4);
  for (uint32_t q = 0; q < nk; q++)
    wsel[q] = f->pin_w[rt.by[q]];
  for (uint32_t k = 0; k < e_n; k++)
  {
    const MoeEntry *en = &f->h_ent[k];
    const uint32_t slot = rt.slot0[en->run] + en->off, row = rt.start[en->run] + en->off;
    wg[k] = f->h_ex[en->run][0], wu[k] = f->h_ex[en->run][1], wd[k] = f->h_ex[en->run][2];
    f->h_rcnt[k] = rcnt[k] = (int32_t)en->cnt;
    rx[k] = slot * lat, ry[k] = row * ffe, rxd[k] = slot * ffe, ryd[k] = row * lat;
  }
  memcpy(rgidx, f->h_rgidx, (size_t)rt.slots * 4);
  memcpy(rhidx, f->h_rhidx, (size_t)rt.slots * 4);
  memcpy(f->moe_pin, f->moe_build, st.used);
  return hipMemcpyAsync(f->moe_dev, f->moe_pin, st.used, hipMemcpyHostToDevice, (hipStream_t)f->st) == hipSuccess;
}

/* a decode's experts' matvec over the entries */
static bool mv_experts(const GgufTensor *t, const void *const *w, const int32_t *cnt, const int32_t *xrow,
                       const Hy4Forward *f, uint32_t entries, uint32_t chunk, uint32_t rows, uint32_t cols,
                       const float *x, float *y)
{
  return mv_group_tok(t->type, w, cnt, xrow, entries, chunk, rows, cols, x, cols, y, rows, f->st);
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

/* a decode's tables on the host (some experts not resident: the membrane's addresses), packed into one compact
   pinned async H2D transfer with entries sorted by descending count (q8_experts' h_cnt) */
static bool decode_tables(Hy4Forward *f, int l, uint32_t n, uint32_t chunk, Hy4DecTab *out)
{
  const Hy4Geometry *g = &f->w->geo;
  const uint32_t K = (uint32_t)g->n_used, E = (uint32_t)g->n_experts, nk = n * K;
  if (hipMemcpyAsync(f->pin_idx, f->idx, (size_t)nk * 4, hipMemcpyDeviceToHost, (hipStream_t)f->st) != hipSuccess ||
      hipMemcpyAsync(f->pin_w, f->wtok, (size_t)nk * F, hipMemcpyDeviceToHost, (hipStream_t)f->st) != hipSuccess ||
      hipStreamSynchronize((hipStream_t)f->st) != hipSuccess)
    return false;
  for (uint32_t m = 0; m < nk; m++)
    for (uint32_t s = m - m % K; s < m; s++)
      if (f->pin_idx[s] == f->pin_idx[m])
        return false;
  MoeRoute rt = {f->h_by, (uint32_t *)f->h_sidx, f->h_run, f->h_run + E, f->h_run + 2 * E, f->h_run + 3 * E,
                 f->h_rgidx, f->h_rhidx, f->h_estart, 0, 0};
  if (!moe_route(f->pin_idx, nk, K, E, MM_TILE, &rt) || !in_memory(f, l, &rt, nk))
    return false;
  uint32_t ne = 0;
  for (uint32_t j = 0; j < rt.runs; j++)
    ne += (rt.count[j] + chunk - 1) / chunk;
  MoeStage st = {f->moe_build, f->moe_dev, 0, f->moe_cap};
  int32_t *gidx = moe_stage_carve(&st, (uint64_t)nk * 4, (void **)&out->gidx);
  int32_t *sidx = moe_stage_carve(&st, (uint64_t)nk * 4, (void **)&out->sidx);
  float *wsel = moe_stage_carve(&st, (uint64_t)nk * F, (void **)&out->wsel);
  int32_t *cnt = moe_stage_carve(&st, (uint64_t)ne * 4, (void **)&out->cnt);
  int32_t *xrow = moe_stage_carve(&st, (uint64_t)ne * 4, (void **)&out->xrow);
  const void **wg = moe_stage_carve(&st, (uint64_t)ne * sizeof(void *), (void **)&out->wg);
  const void **wu = moe_stage_carve(&st, (uint64_t)ne * sizeof(void *), (void **)&out->wu);
  const void **wd = moe_stage_carve(&st, (uint64_t)ne * sizeof(void *), (void **)&out->wd);
  if (!gidx || !sidx || !wsel || !cnt || !xrow || !wg || !wu || !wd)
    return false;
  memcpy(sidx, f->h_sidx, (size_t)nk * 4);
  for (uint32_t q = 0; q < nk; q++)
    gidx[q] = (int32_t)(rt.by[q] / K), wsel[q] = f->pin_w[rt.by[q]];
  uint32_t m = 0;
  for (uint32_t c = chunk; c >= 1; c--)
    for (uint32_t j = 0; j < rt.runs; j++)
      for (uint32_t off = 0; off < rt.count[j]; off += chunk)
      {
        const uint32_t part = rt.count[j] - off < chunk ? rt.count[j] - off : chunk;
        if (part != c)
          continue;
        f->h_rcnt[m] = cnt[m] = (int32_t)part, xrow[m] = (int32_t)(rt.start[j] + off);
        wg[m] = f->h_ex[j][0], wu[m] = f->h_ex[j][1], wd[m] = f->h_ex[j][2], m++;
      }
  out->ne = m;
  memcpy(f->moe_pin, f->moe_build, st.used);
  return hipMemcpyAsync(f->moe_dev, f->moe_pin, st.used, hipMemcpyHostToDevice, (hipStream_t)f->st) == hipSuccess;
}

static bool routed(Hy4Forward *f, int l, uint32_t n, bool read, const float *rin, const int32_t **sidx_out)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  const uint32_t K = (uint32_t)g->n_used, E = (uint32_t)g->n_experts, nk = n * K;
  const uint32_t lat = (uint32_t)g->latent, ffe = (uint32_t)g->ff_exp;
  if (!read)
  {
    const uint32_t chunk = n < MV_MAX_TOKENS ? n : MV_MAX_TOKENS;
    const void *const *tab = f->exp_tab + (uint64_t)(l - g->n_dense) * 3 * E; /* (every expert resident) */
    double t = f->pf.on > 1 ? hy4_now() : 0;
    Hy4DecTab dt = {(const void **)f->wptr, (const void **)f->wptr + nk, (const void **)f->wptr + 2 * nk,
                    f->gidx, f->sidx, f->cnt, f->xrow, f->wsel, nk};
    const bool tables = f->whole ? route_tok(f->idx, f->wtok, n, K, E, chunk, tab, f->gidx, f->sidx, f->wsel, f->cnt,
                                             f->xrow, dt.wg, f->fault, f->st)
                                 : decode_tables(f, l, n, chunk, &dt);
    *sidx_out = dt.sidx;
    if (f->q8 && q8_enabled() && q8_takes(L->exp_gate->type) && q8_takes(L->exp_up->type) && q8_takes(L->exp_down->type))
    { /* each token's row rounded once (rin), its picks' rows by gidx; whole: route_tok by_count, streamed: host h_cnt */
      const Q8Experts e = {.gate_type = L->exp_gate->type, .up_type = L->exp_up->type, .down_type = L->exp_down->type,
                           .wg = dt.wg, .wu = dt.wu, .wd = dt.wd, .cnt = dt.cnt, .xrow = dt.xrow, .xin = dt.gidx,
                           .n = dt.ne, .max_cnt = chunk, .by_count = f->whole, .h_cnt = f->whole ? NULL : f->h_rcnt,
                           .d = lat, .ff = ffe, .x = rin, .x_rows = n, .xq = f->xq, .xqd = f->xqd, .scale = dt.wsel,
                           .lim = g->clamp[l], .ga = f->eg_g, .up = f->eg_u, .h = f->eg_h, .h_rows = nk, .hq = f->hq,
                           .hqd = f->hqd, .y = f->eg_y};
      return tables && hy4_part_done(f, 17, &t) && q8_experts(&e, f->st) && hy4_part_done(f, 20, &t);
    }
    return tables &&
           ops_gather_rows(f->xg, rin, dt.gidx, nk, lat, f->st) && hy4_part_done(f, 17, &t) &&
           mv_experts(L->exp_gate, dt.wg, dt.cnt, dt.xrow, f, dt.ne, chunk, ffe, lat, f->xg, f->eg_g) && hy4_part_done(f, 18, &t) &&
           mv_experts(L->exp_up, dt.wu, dt.cnt, dt.xrow, f, dt.ne, chunk, ffe, lat, f->xg, f->eg_u) && hy4_part_done(f, 19, &t) &&
           ops_swiglu(f->eg_g, f->eg_u, ffe, g->clamp[l], dt.wsel, nk, f->eg_h, f->st) &&
           mv_experts(L->exp_down, dt.wd, dt.cnt, dt.xrow, f, dt.ne, chunk, lat, ffe, f->eg_h, f->eg_y) && hy4_part_done(f, 20, &t);
  }
  uint32_t entries = 0, slots = 0;
  Hy4ReadTab rt = {0};
  double t = f->pf.on > 1 ? hy4_now() : 0;
  if (!route(f, l, n, &entries, &slots, &rt))
    return false;
  *sidx_out = rt.sidx;
  return hy4_part_done(f, 17, &t) && mm_x16_rows(rin, lat, rt.rgidx, slots, lat, f->xe16, f->overflow, f->st) &&
         mm_experts(L->exp_gate->type, rt.wg, f->h_rcnt, rt.rcnt, entries, ffe, lat, f->xe16, rt.rx, f->eg_g, rt.ry, ffe, f->st) &&
         hy4_part_done(f, 18, &t) &&
         mm_experts(L->exp_up->type, rt.wu, f->h_rcnt, rt.rcnt, entries, ffe, lat, f->xe16, rt.rx, f->eg_u, rt.ry,
                    ffe, f->st) && hy4_part_done(f, 19, &t) &&
         ops_swiglu(f->eg_g, f->eg_u, ffe, g->clamp[l], rt.wsel, nk, f->eg_h, f->st) &&
         mm_x16_rows(f->eg_h, ffe, rt.rhidx, slots, ffe, f->xe16, f->overflow, f->st) &&
         mm_experts(L->exp_down->type, rt.wd, f->h_rcnt, rt.rcnt, entries, lat, ffe, f->xe16, rt.rxd, f->eg_y,
                    rt.ryd, lat, f->st) && hy4_part_done(f, 20, &t);
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
                                hy4_mat_pair(f, L->routed_down, lat, rin, NULL, 0, NULL, f->xn, D, n, read, &on_xn) &&
                                hy4_mat_pair(f, L->sh_gate, sh, ga, L->sh_up, sh, up, f->xn, D, n, read, &on_xn);
  const int32_t *sidx = f->sidx;
  return on_x && hy4_part_done(f, 15, &t) &&
         ops_router_scored(logit, hy4_fv(f, L->router_bias), E, K, g->expert_scale, OPS_SCORE_SIGMOID, n, f->idx,
                           f->wtok, f->st) && hy4_part_done(f, 16, &t) &&
         routed(f, l, n, read, rin, &sidx) && (t = f->pf.on > 1 ? hy4_now() : 0, true) &&
         ops_gather_sum(f->rsum, f->eg_y, sidx, K, n, lat, f->st) && hy4_part_done(f, 23, &t) &&
         hy4_mat(f, L->routed_up, D, lat, f->rsum, n, f->out, read) && hy4_part_done(f, 21, &t) &&
         ops_swiglu(ga, up, sh, INFINITY, NULL, n, f->act, f->st) &&
         hy4_mat(f, L->sh_down, D, sh, f->act, n, f->sh, read) &&
         ops_add(f->out, f->sh, (uint64_t)n * D, f->st) &&
         ops_rms(f->out, D, D, hy4_fv(f, L->post_ffn_norm), g->rms_eps, n, f->y, D, f->st) &&
         hy4_hc_post(f->h, f->y, f->post, (uint32_t)g->hc, D, n, f->st) && /* the routed and shared sum, its post norm, into the streams */
         hy4_part_done(f, 22, &t);
}
