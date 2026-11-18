/* the hy4 latent MoE (hy4_impl.h's hy4_moe): the router, the routed experts in the latent - each expert read once for
   the tokens that picked it, a decode's on the matvecs (mv_fp4_tok, mv_group_tok), a prompt's on the matrix cores
   (mm_group over each expert's gathered tokens); a decode's tables built on the GPU (hy4_route), a read's on the host - the shared expert, the combine norm. a token's sums its own in
   either: its picks' order and its sum's (slot order) its own */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/hy4_ops.h"
#include "osmos/gpu/mm.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/model/qtype.h"
#include "hy4_impl.h"

#define FP4_TOK_MAX 4 /* a decode entry's tokens at most: mv_fp4_tok's for NVFP4 experts, mv_group_tok's else */

static uint32_t mv_chunk(const GgufTensor *t) { return t->type == QTYPE_NVFP4 ? FP4_TOK_MAX : MV_MAX_TOKENS; }

uint64_t hy4_read_entries(uint64_t n, const Hy4Geometry *g)
{
  const uint64_t picks = n * (uint64_t)g->n_used;
  return picks + (uint64_t)g->n_experts; /* each expert's chunks: its picks / MM_MAX_TOKENS + 1 at most */
}

uint64_t hy4_read_slots(uint64_t n, const Hy4Geometry *g)
{
  return n * (uint64_t)g->n_used + (uint64_t)g->n_experts * (MM_TILE - 1); /* each expert's padded to a tile */
}

/* a read's picks grouped by expert on the host (ascending; within one, its tokens ascending): each sorted row's token
   and weight, each (token, slot)'s row; mm_group's entries (chunks of MM_MAX_TOKENS, each expert's slots a whole
   number of tiles, the entries ordered by count so a launch is a run of the same column tiles). a decode's tables
   are the GPU's (hy4_route: no host between the router and the experts) */
static bool route(Hy4Forward *f, int l, uint32_t n, uint32_t *entries, uint32_t *slots)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  const uint32_t K = (uint32_t)g->n_used, E = (uint32_t)g->n_experts, nk = n * K;
  const uint32_t lat = (uint32_t)g->latent, ffe = (uint32_t)g->ff_exp;
  if (hipMemcpy(f->h_idx, f->idx, (size_t)nk * 4, hipMemcpyDeviceToHost) != hipSuccess ||
      hipMemcpy(f->h_w, f->wtok, (size_t)nk * F, hipMemcpyDeviceToHost) != hipSuccess)
    return false;
  const uint64_t bytes_g = L->exp_gate->bytes / E, bytes_u = L->exp_up->bytes / E, bytes_d = L->exp_down->bytes / E;
  const uint8_t *bg = hy4_dev(f, L->exp_gate), *bu = hy4_dev(f, L->exp_up), *bd = hy4_dev(f, L->exp_down);
  const uint32_t chunk = MM_MAX_TOKENS;
  uint32_t r = 0, e_n = 0, s_n = 0;
  for (uint32_t e = 0; e < E; e++)
  {
    const uint32_t first = r;
    for (uint32_t t = 0; t < n; t++)
      for (uint32_t s = 0; s < K; s++)
        if ((uint32_t)f->h_idx[t * K + s] == e)
          f->h_gidx[r] = (int32_t)t, f->h_sidx[t * K + s] = (int32_t)r, f->h_wsel[r] = f->h_w[t * K + s], r++;
    if (r == first)
      continue;
    for (uint32_t c = first; c < r; c += chunk, e_n++)
    {
      const uint32_t cnt = r - c < chunk ? r - c : chunk;
      const uint32_t slot = s_n + (c - first);
      f->h_rcnt[e_n] = (int32_t)cnt;
      f->h_rx[e_n] = slot * lat, f->h_ry[e_n] = c * ffe, f->h_rxd[e_n] = slot * ffe, f->h_ryd[e_n] = c * lat;
      f->h_rw[3 * e_n] = bg + e * bytes_g, f->h_rw[3 * e_n + 1] = bu + e * bytes_u, f->h_rw[3 * e_n + 2] = bd + e * bytes_d;
      f->rkey[e_n] = (uint64_t)cnt << 32 | e_n;
    }
    for (uint32_t q = first; q < r; q++)
      f->h_rgidx[s_n + (q - first)] = f->h_gidx[q], f->h_rhidx[s_n + (q - first)] = (int32_t)q;
    const uint32_t padded = (r - first + MM_TILE - 1) / MM_TILE * MM_TILE;
    for (uint32_t s = s_n + (r - first); s < s_n + padded; s++)
      f->h_rgidx[s] = f->h_rhidx[s] = -1;
    s_n += padded;
  }
  *entries = e_n, *slots = s_n;
  if (r != nk)
    return false;
  bool ok = hipMemcpy(f->sidx, f->h_sidx, (size_t)nk * 4, hipMemcpyHostToDevice) == hipSuccess &&
            hipMemcpy(f->wsel, f->h_wsel, (size_t)nk * F, hipMemcpyHostToDevice) == hipSuccess &&
            hipMemcpy(f->gidx, f->h_gidx, (size_t)nk * 4, hipMemcpyHostToDevice) == hipSuccess;
  /* the entries by count (each count's in build order): mm_group's launches runs of the same column tiles */
  for (uint32_t i = 1; i < e_n; i++) /* (insertion sort: a layer's entries are a few hundred) */
    for (uint32_t j = i; j > 0 && f->rkey[j - 1] > f->rkey[j]; j--)
    {
      const uint64_t k = f->rkey[j];
      f->rkey[j] = f->rkey[j - 1], f->rkey[j - 1] = k;
    }
  const uint64_t ecap = hy4_read_entries(f->max_tokens, g);
  const void **pw = (const void **)f->h_stage; /* the sorted tables: 3 x ecap pointers, then 5 x ecap counts */
  uint32_t *pu = (uint32_t *)(pw + 3 * ecap);
  for (uint32_t k = 0; k < e_n; k++)
  {
    const uint32_t i = (uint32_t)f->rkey[k];
    for (int m = 0; m < 3; m++)
      pw[(uint64_t)m * ecap + k] = f->h_rw[3 * i + m];
    pu[k] = (uint32_t)f->h_rcnt[i], pu[ecap + k] = f->h_rx[i], pu[2 * ecap + k] = f->h_ry[i];
    pu[3 * ecap + k] = f->h_rxd[i], pu[4 * ecap + k] = f->h_ryd[i];
  }
  for (uint32_t k = 0; k < e_n; k++)
    f->h_rcnt[k] = (int32_t)pu[k];
  return ok && hipMemcpy(f->rw, pw, 3 * ecap * sizeof(void *), hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->rcnt, f->h_rcnt, (size_t)e_n * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->rx, pu + ecap, (size_t)e_n * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->ry, pu + 2 * ecap, (size_t)e_n * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->rxd, pu + 3 * ecap, (size_t)e_n * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->ryd, pu + 4 * ecap, (size_t)e_n * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->rgidx, f->h_rgidx, (size_t)s_n * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->rhidx, f->h_rhidx, (size_t)s_n * 4, hipMemcpyHostToDevice) == hipSuccess;
}

/* a decode's experts' matvec over the entries: mv_fp4_tok for NVFP4, mv_group_tok else */
static bool mv_experts(const GgufTensor *t, const void *const *w, const Hy4Forward *f, uint32_t entries, uint32_t rows,
                       uint32_t cols, const float *x, float *y)
{
  if (t->type == QTYPE_NVFP4)
    return mv_fp4_tok(w, f->cnt, f->xrow, entries, rows, cols, x, cols, y, rows, f->st);
  return mv_group_tok(t->type, w, f->cnt, f->xrow, entries, MV_MAX_TOKENS, rows, cols, x, cols, y, rows, f->st);
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

static bool routed(Hy4Forward *f, int l, uint32_t n, bool read)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  const uint32_t K = (uint32_t)g->n_used, E = (uint32_t)g->n_experts, nk = n * K;
  const uint32_t lat = (uint32_t)g->latent, ffe = (uint32_t)g->ff_exp;
  if (!read)
  {
    const void *base[3] = {hy4_dev(f, L->exp_gate), hy4_dev(f, L->exp_up), hy4_dev(f, L->exp_down)};
    const uint64_t bytes[3] = {L->exp_gate->bytes / E, L->exp_up->bytes / E, L->exp_down->bytes / E};
    const void **wg = (const void **)f->wptr, **wu = wg + nk, **wd = wg + 2 * nk;
    return hy4_route(f->idx, f->wtok, n, K, E, mv_chunk(L->exp_gate), base, bytes, f->gidx, f->sidx, f->wsel, f->cnt,
                     f->xrow, wg, f->fault, f->st) &&
           ops_gather_rows(f->xg, f->rin, f->gidx, nk, lat, f->st) &&
           mv_experts(L->exp_gate, wg, f, nk, ffe, lat, f->xg, f->eg_g) &&
           mv_experts(L->exp_up, wu, f, nk, ffe, lat, f->xg, f->eg_u) &&
           ops_swiglu(f->eg_g, f->eg_u, ffe, g->clamp[l], f->wsel, nk, f->eg_h, f->st) &&
           mv_experts(L->exp_down, wd, f, nk, lat, ffe, f->eg_h, f->eg_y);
  }
  uint32_t entries = 0, slots = 0;
  if (!route(f, l, n, &entries, &slots))
    return false;
  const uint64_t ecap = hy4_read_entries(f->max_tokens, g);
  const void **rw = f->rw;
  return mm_x16_rows(f->rin, lat, f->rgidx, slots, lat, f->xe16, f->overflow, f->st) &&
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
  return hy4_mat(f, L->router, E, D, f->xn, n, f->logit_r, read) &&
         ops_router_scored(f->logit_r, hy4_fv(f, L->router_bias), E, K, g->expert_scale, OPS_SCORE_SIGMOID, n, f->idx,
                           f->wtok, f->st) &&
         hy4_mat(f, L->routed_down, lat, D, f->xn, n, f->rin, read) && routed(f, l, n, read) &&
         ops_gather_sum(f->rsum, f->eg_y, f->sidx, K, n, lat, f->st) &&
         hy4_mat(f, L->routed_up, D, lat, f->rsum, n, f->out, read) &&
         hy4_mat(f, L->sh_gate, sh, D, f->xn, n, f->ga, read) && hy4_mat(f, L->sh_up, sh, D, f->xn, n, f->up, read) &&
         ops_swiglu(f->ga, f->up, sh, INFINITY, NULL, n, f->act, f->st) &&
         hy4_mat(f, L->sh_down, D, sh, f->act, n, f->sh, read) && ops_add(f->out, f->sh, (uint64_t)n * D, f->st) &&
         ops_rms(f->out, D, D, hy4_fv(f, L->post_ffn_norm), g->rms_eps, n, f->y, D, f->st);
}
