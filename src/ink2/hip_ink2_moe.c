/* Inkling-2-Small's forward, a layer's MoE (osmos/ink2/forward.h): its matrices (mv / mm by rows), the router's picks,
   the experts held from the membrane and run as one grouped matrix */
#define _GNU_SOURCE
#include <fcntl.h>
#include <hip/hip_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "ink2_impl.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/q8.h"
#include "osmos/gpu/route.h"
#include "osmos/model/tensor.h"
#include "osmos/gpu/ink2_ops.h"
#include "osmos/gpu/mm.h"
#include "osmos/gpu/route.h"
#include "osmos/gpu/q8.h"
#include "osmos/mem/experts.h"
#include "osmos/model/moe_route.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/kept.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/gpu/quant.h"
#include "osmos/mem/arena.h"
#include "osmos/err.h"

uint32_t ink2_type(const GgufTensor *t) { return t->type == QUANT_Q5_K ? QUANT_Q5_KR : t->type; }

/* the drafter's y [n][rows] = W x: mv_run, MV_MAX_TOKENS tokens a launch */
bool ink2_mv(const GgufTensor *t, const void *w, const float *x, uint32_t n, float *y)
{
  const uint32_t rows = (uint32_t)t->dims[1], cols = (uint32_t)t->dims[0];
  const uint32_t type = ink2_type(t);
  for (uint32_t at = 0; at < n; at += MV_MAX_TOKENS)
  {
    const uint32_t k = n - at < MV_MAX_TOKENS ? n - at : MV_MAX_TOKENS;
    if (!mv_run(type, w, rows, cols, x + (uint64_t)at * cols, k, y + (uint64_t)at * rows, NULL))
      return false;
  }
  return true;
}

/* the model's y [n][rows] = W x by the matrix's type and shape alone, never the step's: WMMA's sums are not the
   matvec's (DESIGN.md: WMMA's sums probed), so a token's bits are the same in every step only if every
   step takes the same kernel. mm where it has one and the shape tiles (mm_tokens: MM_MAX_TOKENS a launch through
   f->x16), else the matvec (mv_strided: past MV_MAX_TOKENS its wide tiles, each token's sums its own) */
bool ink2_mat(Ink2Forward *f, const GgufTensor *t, const void *w, const float *x, uint32_t n, float *y)
{
  const uint32_t rows = (uint32_t)t->dims[1], cols = (uint32_t)t->dims[0], type = ink2_type(t);
  if (!ink2_mm_takes(f, type, rows, cols))
    return mv_strided(type, w, rows, cols, x, cols, n, y, rows, f->st);
  return mm_tokens(type, w, rows, y, rows, NULL, 0, NULL, 0, cols, x, cols, n, f->x16, f->overflow, false, f->st);
}

/* two matrices on one x (mm_run2: each the bits mm_run gives it alone), or one with w1 NULL; `ready`: f->x16 holds
   this x's f16 already (a matrix before it on the same x) - otherwise each as ink2_mat() */
bool ink2_mat2(Ink2Forward *f, const GgufTensor *t0, const void *w0, float *y0, const GgufTensor *t1, const void *w1,
                 float *y1, const float *x, uint32_t n, bool ready)
{
  const uint32_t cols = (uint32_t)t0->dims[0], type = ink2_type(t0), r0 = (uint32_t)t0->dims[1],
                 r1 = t1 ? (uint32_t)t1->dims[1] : 0;
  if (!ink2_mm_takes(f, type, r0, cols) || (t1 && (ink2_type(t1) != type || (uint32_t)t1->dims[0] != cols ||
                                           !ink2_mm_takes(f, type, r1, cols))))
    return ink2_mat(f, t0, w0, x, n, y0) && (!t1 || ink2_mat(f, t1, w1, x, n, y1));
  if (ready && n <= f->x16_rows) /* (x16 this x's f16, every row: no conversion, any step) */
    return mm_tokens_x16(type, w0, r0, y0, r0, t1 ? w1 : NULL, r1, y1, r1, cols, f->x16, n, f->st);
  return mm_tokens(type, w0, r0, y0, r0, t1 ? w1 : NULL, r1, y1, r1, cols, x, cols, n, f->x16, f->overflow, false,
                   f->st);
}

/* y = x's rows RMS-normed (times w), and for t on mm in one launch (n <= MM_MAX_TOKENS) f->x16 their f16 too -
   ops_rms_x16's, mm_x16's bytes: *ready, mat2's for t on y */
bool ink2_rms_for(Ink2Forward *f, const GgufTensor *t, const float *x, const float *w, uint32_t n, float *y,
                    bool *ready)
{
  const uint32_t C = (uint32_t)f->w->geo.n_embd;
  const float eps = f->w->geo.rms_eps;
  *ready = n <= f->x16_rows && ink2_mm_takes(f, ink2_type(t), (uint32_t)t->dims[1], (uint32_t)t->dims[0]);
  return *ready ? ops_rms_x16(x, C, C, w, eps, n, y, C, f->x16, f->overflow, f->st)
                : ops_rms(x, C, C, w, eps, n, y, C, f->st);
}

/* one role's experts (their entries in ascending count): a launch a run of the same column tiles (mm_col_tiles), so
   a one-tile expert is not run in a hot one's many-tile kernel (hip_ds4_ffn.c's group). each expert's rows its own: the
   launches' order changes no bit */
static bool group(uint32_t type, const void *const *dtab, const int32_t *cnt, const int32_t *dcnt, uint32_t ne,
                  uint32_t rows, uint32_t cols, const void *x16, const uint32_t *dxo, float *y, const uint32_t *dyo,
                  uint32_t ys, void *st)
{
  bool ok = true;
  for (uint32_t a = 0, z = 0; ok && a < ne; a = z)
  {
    const uint32_t ct = mm_col_tiles(type, (uint32_t)cnt[a]);
    for (z = a + 1; z < ne && mm_col_tiles(type, (uint32_t)cnt[z]) == ct;)
      z++;
    ok = mm_group(type, dtab + a, dcnt + a, z - a, (uint32_t)cnt[z - 1], rows, cols, x16, dxo + a, cols, y, dyo + a,
                  ys, st);
  }
  return ok;
}

/* the routed experts of layer l for the run's n tokens: each distinct pick held (its read begun), then read once for
   the tokens that picked it (mm_group at every step: an entry an expert's part of up to MM_MAX_TOKENS of its rows,
   a token's sums its own whatever the others) - gate and up, their scales and the swiglu, down - and each token's
   summed in its slot order (ink2_moe_sum) */
/* moe with every expert in memory: the router's picks never on the host - the tables built on the GPU
   (q4x_expert_tiles: the picks by expert, a part of up to 128 an entry, the entries in runs of their tiles;
   ink2_moe_scales: the scales and weights moe's host loop writes), each run a launch (its entries past the live ones
   none). which picks share a part, and where, changes no bit: the same sums as moe's tables */
static bool moe_whole(Ink2Forward *f, int l, uint32_t n, char *err)
{
  const Ink2Geometry *g = &f->w->geo;
  const Ink2Layer *L = &f->w->layer[l];
  const Ink2Dev *d = &f->L[l];
  const uint32_t C = (uint32_t)g->n_embd, E = (uint32_t)g->n_experts, K = (uint32_t)g->n_used,
                 F = (uint32_t)g->ff_exp, P = n * K;
  Q4xExpertBounds bd;
  q4x_expert_bounds(P, E, &bd);
  Q4xExpertTiles et = {.cnt = f->t_cnt, .gidx = f->t_gidx, .hidx = f->t_hidx, .xg = f->t_xg, .yg = f->t_yg,
                       .xd = f->t_xd, .yd = f->t_yd, .where = f->t_where, .off = f->t_off};
  for (int i = 0; i < 3; i++)
    et.ptr[i] = f->t_ptr + (uint64_t)i * f->t_entries, et.tab[i] = f->exp_tab + ((uint64_t)l * 3 + (uint64_t)i) * E;
  const GgufTensor *t[3] = {L->exp_gate, L->exp_up, L->exp_down};
  const uint32_t rows[3] = {F, F, C}, cols[3] = {C, C, F};
  float *const y[3] = {f->eg, f->eu, f->ed};
  const uint32_t *xo[3] = {f->t_xg, f->t_xg, f->t_xd}, *yo[3] = {f->t_yg, f->t_yg, f->t_yd};
  float *pw = (float *)(f->pick + P);
  const float *scales = f->scale_tab + (uint64_t)l * 3 * E;
  uint32_t *picks = f->picks_d + (uint64_t)l * E;
  bool ok = ink2_mat2(f, L->router, d->router, f->logit, NULL, NULL, NULL, f->xn, n, f->router16);
  /* a decode's or a verify's experts on int8 (q8_experts): route_tok's tables (an expert's picks one entry; its fault -
     a pick out of range or twice, the router's never - the step's overflow flag: refused), the scales by their rows
     (ink2_moe_scales through route_tok's where), each token's row rounded once, read by token; Inkling's activation
     on each row's tensor scales; the weights and down's scale at the sum (m_a). a prompt's on the matrix cores (f16,
     below): the two computed apart */
  if (n <= MV_MAX_TOKENS && !f->reading && !f->no_int8 && q8_enabled() && q8_takes(t[0]->type) && q8_takes(t[1]->type) && q8_takes(t[2]->type) &&
      C % 32 == 0 && F % 32 == 0)
  {
    const uint32_t chunk = n;
    const Q8Experts q8 = {.act = Q8_ACT_INK2, .gate_type = t[0]->type, .up_type = t[1]->type, .down_type = t[2]->type,
                          .wg = f->q_wptr, .wu = f->q_wptr + P, .wd = f->q_wptr + 2 * (uint64_t)P, .cnt = f->q_cnt,
                          .xrow = f->q_xrow, .xin = f->q_gidx, .n = P, .max_cnt = chunk, .d = C, .ff = F, .x = f->xn,
                          .x_rows = n, .xq = f->q_x, .xqd = f->q_xd, .sg = f->t_sg, .su = f->t_su,
                          .gcap = g->gate_softcap, .ucap = g->up_softcap, .h = f->eg, .h_rows = P, .hq = f->q_h,
                          .hqd = f->q_hd, .y = f->ed};
    ok = ok &&
         ops_router_scored(f->logit, d->router_bias, E, K, g->route_scale, OPS_SCORE_SIGMOID, n, f->pick, pw, f->st) &&
         route_tok(f->pick, pw, n, K, E, chunk, et.tab[0], f->q_gidx, f->q_sidx, f->q_wsel, f->q_cnt, f->q_xrow,
                   f->q_wptr, f->overflow, f->st) &&
         ink2_moe_scales(f->pick, pw, f->q_sidx, scales, E, P, f->t_sg, f->t_su, f->t_a, picks, f->st) &&
         q8_experts(&q8, f->st);
    f->m_a = f->t_a, f->m_row = f->q_sidx;
    return ok || osmos_fail(err, "a GPU call failed (the experts on int8, their tables on the GPU)");
  }
  if (ok && P <= INK2_MOE_PREP_PICKS) /* a decode's or a verify's: the three in one launch, the same bits */
    ok = ink2_moe_prep(f->logit, d->router_bias, E, K, g->route_scale, n, f->pick, pw, C, F, &et, scales, f->t_sg,
                       f->t_su, f->t_a, picks, f->st);
  else
    ok = ok &&
         ops_router_scored(f->logit, d->router_bias, E, K, g->route_scale, OPS_SCORE_SIGMOID, n, f->pick, pw, f->st) &&
         q4x_expert_tiles(f->pick, P, K, E, C, F, &et, f->st) &&
         ink2_moe_scales(f->pick, pw, f->t_where, scales, E, P, f->t_sg, f->t_su, f->t_a, picks, f->st);
  ok = ok && mm_x16_rows(f->xn, C, f->t_gidx, bd.slots, C, f->xe16, f->overflow, f->st);
  for (int k = 0; ok && k < 3; k++)
  {
    if (k == 2)
      ok = ink2_swiglu_x16(f->eg, f->eu, f->t_sg, f->t_su, F, g->gate_softcap, g->up_softcap, f->t_hidx, bd.slots,
                           f->xe16, f->overflow, f->st);
    /* run r's experts more than MM_TILE << (r - 1) tokens each (its bound the pairs'): none in a step of fewer
       tokens (a token picks an expert once): not launched empty */
    for (int r = 0; ok && r < Q4X_TILE_RUNS; r++)
      if (bd.count[r] && (r == 0 || n > (uint32_t)MM_TILE << (r - 1)))
      {
        const uint32_t at = bd.first[r];
        ok = mm_group(t[k]->type, et.ptr[k] + at, f->t_cnt + at, bd.count[r], MM_TILE << r, rows[k], cols[k],
                      f->xe16, xo[k] + at, cols[k], y[k], yo[k] + at, rows[k], f->st);
      }
  }
  f->m_a = f->t_a, f->m_row = f->t_where;
  return ok || osmos_fail(err, "a GPU call failed (the experts, their tables on the GPU)");
}

void ink2_forward_reading(Ink2Forward *f, bool on)
{
  if (f)
    f->reading = on;
}

bool ink2_forward_int8(const Ink2Forward *f) { return f && !f->no_int8 && q8_enabled(); }

bool ink2_moe(Ink2Forward *f, int l, uint32_t n, char *err)
{
  if (f->whole)
    return moe_whole(f, l, n, err);
  const Ink2Geometry *g = &f->w->geo;
  const Ink2Layer *L = &f->w->layer[l];
  const Ink2Dev *d = &f->L[l];
  const uint32_t C = (uint32_t)g->n_embd, E = (uint32_t)g->n_experts, K = (uint32_t)g->n_used,
                 F = (uint32_t)g->ff_exp, P = n * K;
  const double r0 = f->pf.on ? ink2_pf_now() : 0;
  /* a prompt's read ahead (its picks a layer's experts at least): the next layer's router on this layer's MoE input,
     the union of its picks read into the stream while this layer's experts and the next attention run. not a
     decode's: there the guess names few of its misses and reads experts a token never picks - the next router's input
     is the attention residual's over the blocks, far from this one's - their reads the misses' bandwidth. which are
     read ahead changes no bit */
  const bool next = l + 1 < g->n_layers && P >= E;
  const Ink2Layer *L1 = next ? &f->w->layer[l + 1] : NULL;
  /* the picks' weights after them (pick + P): the two in one copy to the host */
  float *pw = (float *)(f->pick + P), *hw = (float *)(f->hpick + P);
  if (!ink2_mat2(f, L->router, d->router, f->logit, NULL, NULL, NULL, f->xn, n, f->router16) ||
      !ops_router_scored(f->logit, d->router_bias, E, K, g->route_scale, OPS_SCORE_SIGMOID, n, f->pick, pw, f->st) ||
      (next && (!ink2_mat(f, L1->router, f->L[l + 1].router, f->xn, n, f->logit) ||
                !ops_router_scored(f->logit, f->L[l + 1].router_bias, E, K, g->route_scale, OPS_SCORE_SIGMOID, n,
                                   f->gpick, f->gw, f->st) ||
                hipMemcpyAsync(f->hgpick, f->gpick, P * 4, hipMemcpyDeviceToHost, f->st) != hipSuccess)) ||
      hipMemcpyAsync(f->hpick, f->pick, 2 * P * 4, hipMemcpyDeviceToHost, f->st) != hipSuccess ||
      hipStreamSynchronize(f->st) != hipSuccess)
    return osmos_fail(err, "a GPU call failed (the router)");
  if (f->pf.on)
    f->pf.step[PF_ROUTE] += ink2_pf_now() - r0;
  /* the GPU is past the last layer's experts (the copy waited for it): the visit ends their holds */
  if (!membrane_visit(f->mem, (uint32_t)l, P >= E, err))
    return false;
  for (uint32_t m = 0; P < E && m < P; m++) /* a decode's picks counted: in memory or not (the stats') */
    membrane_score(f->mem, (uint32_t)l * E + (uint32_t)f->hpick[m]);
  /* the tables, one region sent in one copy, each table's place fixed by the step's most pairs: the entries' gate, up
     and down pointers, counts, their slots' and rows' offsets (gate and up's x and y, down's); each slot's token and
     sorted row (-1: a tile's padding); each pair's sorted row and weight, each sorted row's scales */
  const uint64_t ecap = ink2_table_entries((uint64_t)f->max_tokens * K, E), scap = ink2_table_slots((uint64_t)f->max_tokens * K, E);
  uint8_t *h = f->m_build; /* (cached: built here, then one copy into the pinned) */
  const void **tg = (const void **)h, **tu = tg + ecap, **td = tu + ecap;
  int32_t *cnt = (int32_t *)(td + ecap);
  uint32_t *xg = (uint32_t *)(cnt + ecap), *yg = xg + ecap, *xd = yg + ecap, *yd = xd + ecap;
  int32_t *gidx = (int32_t *)(yd + ecap), *hidx = gidx + scap, *row = hidx + scap;
  float *a = (float *)(row + P), *sg = a + P, *su = sg + P;
  /* the pairs by expert, then pair order (moe_route: sorted row r is pair by[r]), each slot's token and row */
  MoeRoute rt = {f->m_by, (uint32_t *)row, f->m_run, f->m_run + E, f->m_run + 2 * E, f->m_run + 3 * E, gidx, hidx,
                 f->m_at, 0, 0};
  if (!moe_route(f->hpick, P, K, E, MM_TILE, &rt))
    return osmos_fail(err, "the router picked an expert the model does not have");
  for (uint32_t m = 0; m < P; m++)
    f->picks[(uint32_t)l * E + (uint32_t)f->hpick[m]]++;
  for (uint32_t j = 0; j < rt.runs; j++) /* every distinct pick held first: the cold ones' reads under way together */
    if (!membrane_hold(f->mem, (uint32_t)l * E + rt.expert[j], err))
      return false;
  /* then the next layer's named: this layer's reads queued first, the stream's room theirs before the guess's */
  for (uint32_t m = 0; next && m < P; m++)
    if (f->hgpick[m] >= 0 && (uint32_t)f->hgpick[m] < E)
    {
      const uint32_t unit = (uint32_t)(l + 1) * E + (uint32_t)f->hgpick[m];
      membrane_ahead(f->mem, unit);
    }
  /* each run's expert in memory (its parts' addresses), its rows' scales; the entries by count (moe_entries: group's
     runs of the same column tiles contiguous, each count's in build order) */
  for (uint32_t j = 0; j < rt.runs; j++)
  {
    const uint32_t e = rt.expert[j], unit = (uint32_t)l * E + e;
    const double wt = f->pf.on ? ink2_pf_now() : 0;
    if (!membrane_wait(f->mem, unit, f->m_ex[j], err))
      return false;
    if (f->pf.on)
      f->pf.step[PF_WAIT] += ink2_pf_now() - wt;
    for (uint32_t q = rt.start[j]; q < rt.start[j] + rt.count[j]; q++)
    {
      const uint32_t m = rt.by[q];
      sg[q] = d->scale[0][e], su[q] = d->scale[1][e], a[m] = hw[m] * d->scale[2][e];
    }
  }
  /* a decode's or a verify's experts on int8 (q8_experts): an entry a run (an expert's picks at most n <=
     MV_MAX_TOKENS: one), its rows from the run's start, each sorted row's token by[q] / K - in the region's places
     (tg, tu, td, cnt; xg the rows, gidx the tokens), one copy; Inkling's activation on each row's scales (sg, su),
     the weights and down's scale at the sum (a). a prompt's on the matrix cores (f16, below): the two apart */
  if (n <= MV_MAX_TOKENS && !f->reading && !f->no_int8 && q8_enabled() && q8_takes(L->exp_gate->type) && q8_takes(L->exp_up->type) &&
      q8_takes(L->exp_down->type) && C % 32 == 0 && F % 32 == 0)
  {
    int32_t *rows = (int32_t *)xg;
    for (uint32_t j = 0; j < rt.runs; j++)
      tg[j] = f->m_ex[j][0], tu[j] = f->m_ex[j][1], td[j] = f->m_ex[j][2], cnt[j] = (int32_t)rt.count[j],
      rows[j] = (int32_t)rt.start[j];
    for (uint32_t q = 0; q < P; q++)
      gidx[q] = (int32_t)(rt.by[q] / K);
    const uint64_t bytes = (uint64_t)((uint8_t *)(su + P) - h);
    uint8_t *dv = f->m_dev;
    const void **dg = (const void **)dv, **du = dg + ecap, **dd = du + ecap;
    const int32_t *dcnt = (const int32_t *)(dd + ecap);
    const int32_t *drows = (const int32_t *)(dcnt + ecap);
    const int32_t *dgidx = (const int32_t *)((const uint32_t *)drows + 4 * ecap), *drow = dgidx + 2 * scap;
    const float *da = (const float *)(drow + P), *dsg = da + P, *dsu = dsg + P;
    const Q8Experts q8 = {.act = Q8_ACT_INK2, .gate_type = L->exp_gate->type, .up_type = L->exp_up->type,
                          .down_type = L->exp_down->type, .wg = dg, .wu = du, .wd = dd, .cnt = dcnt, .xrow = drows,
                          .xin = dgidx, .n = rt.runs, .max_cnt = n, .d = C, .ff = F, .x = f->xn, .x_rows = n,
                          .xq = f->q_x, .xqd = f->q_xd, .sg = dsg, .su = dsu, .gcap = g->gate_softcap,
                          .ucap = g->up_softcap, .h = f->eg, .h_rows = P, .hq = f->q_h, .hqd = f->q_hd, .y = f->ed};
    const double x0 = f->pf.on ? ink2_pf_now() : 0;
    memcpy(f->m_host, h, bytes);
    const bool ok = hipMemcpyAsync(dv, f->m_host, bytes, hipMemcpyHostToDevice, f->st) == hipSuccess &&
                    q8_experts(&q8, f->st);
    f->m_a = da, f->m_row = drow;
    if (!ok)
      return osmos_fail(err, "a GPU call failed (the experts on int8)");
    if (f->pf.on)
      hipDeviceSynchronize(), f->pf.step[PF_XK] += ink2_pf_now() - x0;
    return true;
  }
  uint32_t ne = 0;
  if (!moe_entries(&rt, MM_MAX_TOKENS, f->m_ent, f->m_key, &ne))
    return osmos_fail(err, "the experts' entries refused");
  for (uint32_t k = 0; k < ne; k++)
  {
    const MoeEntry *en = &f->m_ent[k];
    const uint32_t s0 = rt.slot0[en->run] + en->off, q = rt.start[en->run] + en->off;
    tg[k] = f->m_ex[en->run][0], tu[k] = f->m_ex[en->run][1], td[k] = f->m_ex[en->run][2], cnt[k] = (int32_t)en->cnt;
    xg[k] = s0 * C, yg[k] = q * F, xd[k] = s0 * F, yd[k] = q * C;
  }
  const uint32_t slots = rt.slots;
  const uint64_t bytes = (uint64_t)((uint8_t *)(su + P) - h);
  uint8_t *dv = f->m_dev;
  const void **dg = (const void **)dv, **du = dg + ecap, **dd = du + ecap;
  const int32_t *dcnt = (const int32_t *)(dd + ecap);
  const uint32_t *dxg = (const uint32_t *)(dcnt + ecap), *dyg = dxg + ecap, *dxd = dyg + ecap, *dyd = dxd + ecap;
  const int32_t *dgidx = (const int32_t *)(dyd + ecap), *dhidx = dgidx + scap, *drow = dhidx + scap;
  const float *da = (const float *)(drow + P), *dsg = da + P, *dsu = dsg + P;
  const double x0 = f->pf.on ? ink2_pf_now() : 0;
  memcpy(f->m_host, h, bytes);
  const bool ok =
      hipMemcpyAsync(dv, f->m_host, bytes, hipMemcpyHostToDevice, f->st) == hipSuccess &&
      mm_x16_rows(f->xn, C, dgidx, slots, C, f->xe16, f->overflow, f->st) &&
      group(L->exp_gate->type, dg, cnt, dcnt, ne, F, C, f->xe16, dxg, f->eg, dyg, F, f->st) &&
      group(L->exp_up->type, du, cnt, dcnt, ne, F, C, f->xe16, dxg, f->eu, dyg, F, f->st) &&
      ink2_swiglu_x16(f->eg, f->eu, dsg, dsu, F, g->gate_softcap, g->up_softcap, dhidx, slots, f->xe16,
                      f->overflow, f->st) &&
      group(L->exp_down->type, dd, cnt, dcnt, ne, C, F, f->xe16, dxd, f->ed, dyd, C, f->st);
  f->m_a = da, f->m_row = drow;
  if (!ok)
    return osmos_fail(err, "a GPU call failed (the experts)");
  if (f->pf.on)
    hipDeviceSynchronize(), f->pf.step[PF_XK] += ink2_pf_now() - x0;
  return true;
}

/* a conv on tail t's history (its last taps - 1 rows), x's n rows then the tail's: written by the conv as it reads
   them when they fit (ink2_tail_claim), else ink2_tail_put after it - the same rows either way. add (a static
   conv's, NULL none): y = add + the conv, a residual's sum in the conv's launch */
