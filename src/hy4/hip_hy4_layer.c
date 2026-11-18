/* a hy4 layer's blocks (the reference's HYV4FlashDecoderLayer): Engram's residual; iHC around MLA (a sparse layer's
   indexer choosing its keys, a sliding one's window in its ring); iHC around the MLP (dense, or the latent MoE: each
   expert read once for the tokens that picked it). every op a token's own sums: its bits whatever the step */
#include <math.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/hy4_ops.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/model/qtype.h"
#include "hy4_impl.h"

bool hy4_mat(const Hy4Forward *f, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t n,
             float *y)
{
  return mv_strided(t->type, hy4_dev(f, t), rows, cols, x, cols, n, y, rows, NULL);
}

static bool engram(Hy4Forward *f, uint32_t n, int l)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  int e = 0;
  for (int k = 0; k < l; k++)
    e += g->engram[k];
  const uint32_t D = (uint32_t)g->n_embd, hc = (uint32_t)g->hc, dim = (uint32_t)g->engram_dim;
  const uint32_t cols = (uint32_t)((g->engram_ngram - 1) * g->engram_heads);
  const float *rows = f->eg_rows + (uint64_t)e * n * cols * dim;
  return hy4_engram_rows(rows, hy4_fv(f, L->eg_lookup_norm), cols, dim, n, f->emb, NULL) &&
         hy4_mat(f, L->eg_kv, (hc + 1) * D, cols * dim, f->emb, n, f->ekv) &&
         hy4_engram_update(f->h, f->ekv, hy4_fv(f, L->eg_kv_b), hy4_fv(f, L->eg_gate), hy4_fv(f, L->eg_hc_scale),
                           hy4_fv(f, L->eg_pre_gamma), hc, D, n, NULL);
}

/* a sparse layer's run: its token keys at their positions, the groups they complete pooled, each token's groups */
static bool indexer(Hy4Forward *f, Hy4Seq *s, int l, uint32_t off, uint32_t len)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  const uint32_t ID = (uint32_t)g->ix_dim, IH = (uint32_t)g->ix_heads, pool = (uint32_t)g->ix_pool;
  const uint32_t top = (uint32_t)(g->ix_topk / g->ix_pool), P = (uint32_t)s->pos;
  const uint32_t g0 = P / pool, g1 = (P + len) / pool;
  return ops_rms(f->ixk + (uint64_t)off * ID, ID, ID, hy4_fv(f, L->ix_k_norm), g->rms_eps, len,
                 s->keys[l] + (uint64_t)P * ID, ID, NULL) &&
         (g1 == g0 || hy4_ix_pool(s->keys[l], hy4_dev(f, L->ix_pool_value), hy4_dev(f, L->ix_pool_gate),
                                  hy4_fv(f, L->ix_pool_ape), hy4_fv(f, L->ix_pool_norm), ID, pool,
                                  (uint32_t)g->rope_dim, g->rms_eps, f->freqs_sparse, g0, g1 - g0, s->groups[l],
                                  NULL)) &&
         hy4_ix_select(f->ixq + (uint64_t)off * IH * ID, f->ixw + (uint64_t)off * IH, s->groups[l], IH, ID, pool, top,
                       f->pos + off, len, f->max_ctx / pool + 1, f->scores + (uint64_t)off * (f->max_ctx / pool + 1),
                       f->sel + (uint64_t)off * top, f->n_sel + off, NULL);
}

static bool attention(Hy4Forward *f, const Hy4Run *run, uint32_t runs, uint32_t n, int l)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  const bool sw = g->sliding[l];
  const uint32_t D = (uint32_t)g->n_embd, H = (uint32_t)g->n_heads[l], ql = (uint32_t)g->q_lora;
  const uint32_t nope = (uint32_t)g->nope_dim, rd = (uint32_t)g->rope_dim, lat = (uint32_t)g->kv_lora;
  const uint32_t vd = (uint32_t)g->v_dim, qk = nope + rd, row = lat + rd, ID = (uint32_t)g->ix_dim;
  const uint32_t IH = (uint32_t)g->ix_heads, top = (uint32_t)(g->ix_topk / g->ix_pool);
  const float *freqs = sw ? f->freqs_swa : f->freqs_sparse, scale = 1.0f / sqrtf((float)qk);
  bool ok = hy4_mat(f, L->q_a, ql, D, f->xn, n, f->qr) &&
            ops_rms(f->qr, ql, ql, hy4_fv(f, L->q_a_norm), g->rms_eps, n, f->qres, ql, NULL) &&
            hy4_mat(f, L->q_b, H * qk, ql, f->qres, n, f->q) && hy4_mat(f, L->kv_a, row, D, f->xn, n, f->kva) &&
            hy4_mat(f, L->gate, H * vd, D, f->xn, n, f->gt);
  if (ok && !sw)
    ok = hy4_mat(f, L->ix_k, ID, D, f->xn, n, f->ixk) && hy4_mat(f, L->ix_q_b, IH * ID, ql, f->qres, n, f->ixq) &&
         hy4_mat(f, L->ix_proj, IH, D, f->xn, n, f->ixw) &&
         hy4_ix_query(f->ixq, f->ixw, IH, ID, rd, f->freqs_sparse, f->pos, n, NULL);
  for (uint32_t r = 0; ok && r < runs; r++)
  {
    const uint64_t o = run[r].off;
    const uint32_t m = run[r].len;
    Hy4Seq *s = run[r].s;
    ok = hy4_mla_kv(f->kva + o * row, hy4_fv(f, L->kv_a_norm), lat, rd, g->rms_eps, freqs, f->pos + o, m, s->cache[l],
                    sw ? f->ring : 0, NULL) &&
         hy4_mla_q(f->q + o * H * qk, hy4_dev(f, L->k_b), H, nope, lat, rd, freqs, f->pos + o, m, f->qa + o * H * row,
                   f->table, NULL) &&
         (sw || indexer(f, s, l, (uint32_t)o, m)) &&
         hy4_mla_attn(f->qa + o * H * row, s->cache[l], sw ? f->ring : 0, hy4_fv(f, L->sinks), H, lat, rd,
                      sw ? (uint32_t)g->window : 0, sw ? NULL : f->sel + o * top, sw ? NULL : f->n_sel + o,
                      sw ? 0 : top, sw ? 0 : (uint32_t)g->ix_pool, scale, f->pos + o, m, f->olat + o * H * lat, f->attn_part, NULL);
  }
  return ok && hy4_mla_out(f->olat, hy4_dev(f, L->v_b), f->gt, H, lat, vd, n, f->ao, f->table, NULL) &&
         hy4_mat(f, L->out, D, H * vd, f->ao, n, f->out) &&
         ops_rms(f->out, D, D, hy4_fv(f, L->post_attn_norm), g->rms_eps, n, f->y, D, NULL);
}

/* the step's picks grouped by expert (ascending; within one, its tokens ascending): each gathered row's token and
   weight, each (token, slot)'s row, and mv_group_tok's entries (an expert's rows in chunks of MV_MAX_TOKENS) */
/* an entry's tokens at most: mv_fp4_tok's for NVFP4 experts, mv_group_tok's for the rest */
#define FP4_TOK_MAX 4
static uint32_t chunk_of(const GgufTensor *t) { return t->type == QTYPE_NVFP4 ? FP4_TOK_MAX : MV_MAX_TOKENS; }

/* the experts' matvec over the entries: mv_fp4_tok for NVFP4, mv_group_tok else */
static bool experts(const GgufTensor *t, const void *const *w, const Hy4Forward *f, uint32_t entries, uint32_t rows,
                    uint32_t cols, const float *x, float *y)
{
  if (t->type == QTYPE_NVFP4)
    return mv_fp4_tok(w, f->cnt, f->xrow, entries, rows, cols, x, cols, y, rows, NULL);
  return mv_group_tok(t->type, w, f->cnt, f->xrow, entries, MV_MAX_TOKENS, rows, cols, x, cols, y, rows, NULL);
}

static bool route(Hy4Forward *f, int l, uint32_t n, uint32_t *entries)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  const uint32_t K = (uint32_t)g->n_used, E = (uint32_t)g->n_experts, nk = n * K;
  if (hipMemcpy(f->h_idx, f->idx, (size_t)nk * 4, hipMemcpyDeviceToHost) != hipSuccess ||
      hipMemcpy(f->h_w, f->wtok, (size_t)nk * F, hipMemcpyDeviceToHost) != hipSuccess)
    return false;
  const uint64_t bytes_g = L->exp_gate->bytes / E, bytes_u = L->exp_up->bytes / E, bytes_d = L->exp_down->bytes / E;
  uint32_t r = 0, e_n = 0;
  for (uint32_t e = 0; e < E; e++)
  {
    const uint32_t first = r;
    for (uint32_t t = 0; t < n; t++)
      for (uint32_t s = 0; s < K; s++)
        if ((uint32_t)f->h_idx[t * K + s] == e)
          f->h_gidx[r] = (int32_t)t, f->h_sidx[t * K + s] = (int32_t)r, f->h_wsel[r] = f->h_w[t * K + s], r++;
    const uint32_t chunk = chunk_of(L->exp_gate);
    for (uint32_t c = first; c < r; c += chunk, e_n++)
    {
      f->h_cnt[e_n] = (int32_t)(r - c < chunk ? r - c : chunk), f->h_xrow[e_n] = (int32_t)c;
      f->h_wptr[e_n] = (const uint8_t *)hy4_dev(f, L->exp_gate) + e * bytes_g;
      f->h_wptr[nk + e_n] = (const uint8_t *)hy4_dev(f, L->exp_up) + e * bytes_u;
      f->h_wptr[2 * nk + e_n] = (const uint8_t *)hy4_dev(f, L->exp_down) + e * bytes_d;
    }
  }
  *entries = e_n;
  return r == nk && hipMemcpy(f->gidx, f->h_gidx, (size_t)nk * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->sidx, f->h_sidx, (size_t)nk * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->wsel, f->h_wsel, (size_t)nk * F, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->cnt, f->h_cnt, (size_t)e_n * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->xrow, f->h_xrow, (size_t)e_n * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->wptr, f->h_wptr, (size_t)3 * nk * sizeof(void *), hipMemcpyHostToDevice) == hipSuccess;
}

static bool moe(Hy4Forward *f, uint32_t n, int l)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, E = (uint32_t)g->n_experts, K = (uint32_t)g->n_used;
  const uint32_t lat = (uint32_t)g->latent, ffe = (uint32_t)g->ff_exp, sh = ffe * (uint32_t)g->n_shexp, nk = n * K;
  uint32_t entries = 0;
  const void **wg = (const void **)f->wptr, **wu = wg + nk, **wd = wg + 2 * nk;
  return hy4_mat(f, L->router, E, D, f->xn, n, f->logit_r) &&
         ops_router_scored(f->logit_r, hy4_fv(f, L->router_bias), E, K, g->expert_scale, OPS_SCORE_SIGMOID, n, f->idx,
                           f->wtok, NULL) &&
         hy4_mat(f, L->routed_down, lat, D, f->xn, n, f->rin) && route(f, l, n, &entries) &&
         ops_gather_rows(f->xg, f->rin, f->gidx, nk, lat, NULL) &&
         experts(L->exp_gate, wg, f, entries, ffe, lat, f->xg, f->eg_g) &&
         experts(L->exp_up, wu, f, entries, ffe, lat, f->xg, f->eg_u) &&
         ops_swiglu(f->eg_g, f->eg_u, ffe, g->clamp[l], f->wsel, nk, f->eg_h, NULL) &&
         experts(L->exp_down, wd, f, entries, lat, ffe, f->eg_h, f->eg_y) &&
         ops_gather_sum(f->rsum, f->eg_y, f->sidx, K, n, lat, NULL) &&
         hy4_mat(f, L->routed_up, D, lat, f->rsum, n, f->out) && hy4_mat(f, L->sh_gate, sh, D, f->xn, n, f->ga) &&
         hy4_mat(f, L->sh_up, sh, D, f->xn, n, f->up) && ops_swiglu(f->ga, f->up, sh, INFINITY, NULL, n, f->act, NULL) &&
         hy4_mat(f, L->sh_down, D, sh, f->act, n, f->sh) && ops_add(f->out, f->sh, (uint64_t)n * D, NULL) &&
         ops_rms(f->out, D, D, hy4_fv(f, L->post_ffn_norm), g->rms_eps, n, f->y, D, NULL);
}

static bool dense(Hy4Forward *f, uint32_t n, int l)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, ff = (uint32_t)g->n_ff;
  return hy4_mat(f, L->ffn_gate, ff, D, f->xn, n, f->ga) && hy4_mat(f, L->ffn_up, ff, D, f->xn, n, f->up) &&
         ops_swiglu(f->ga, f->up, ff, INFINITY, NULL, n, f->act, NULL) &&
         hy4_mat(f, L->ffn_down, D, ff, f->act, n, f->out) &&
         ops_rms(f->out, D, D, hy4_fv(f, L->post_ffn_norm), g->rms_eps, n, f->y, D, NULL);
}

bool hy4_layer(Hy4Forward *f, const Hy4Run *run, uint32_t runs, uint32_t n, int l)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, hc = (uint32_t)g->hc;
  return (!g->engram[l] || engram(f, n, l)) &&
         hy4_hc_pre(f->h, f->fn_attn[l], hy4_fv(f, L->hc_attn_scale), hy4_fv(f, L->hc_attn_base), 2 * hc, hc, D,
                    g->rms_eps, g->hc_eps, g->hc_magnitude, hy4_fv(f, L->attn_norm), n, f->xn, f->post, f->hcs, NULL) &&
         attention(f, run, runs, n, l) && hy4_hc_post(f->h, f->y, f->post, hc, D, n, NULL) &&
         hy4_hc_pre(f->h, f->fn_ffn[l], hy4_fv(f, L->hc_ffn_scale), hy4_fv(f, L->hc_ffn_base), 2 * hc, hc, D,
                    g->rms_eps, g->hc_eps, g->hc_magnitude, hy4_fv(f, L->ffn_norm), n, f->xn, f->post, f->hcs, NULL) &&
         (l < g->n_dense ? dense(f, n, l) : moe(f, n, l)) && hy4_hc_post(f->h, f->y, f->post, hc, D, n, NULL);
}
