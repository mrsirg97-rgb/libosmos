/* a hy4 layer's blocks (the reference's HYV4FlashDecoderLayer): Engram's residual; iHC around MLA (a sparse layer's
   indexer choosing its keys, a sliding one's window in its ring); iHC around the MLP (dense, or the latent MoE: each
   expert read once for the tokens that picked it). every op a token's own sums: its bits whatever the step */
#include <math.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/hy4_ops.h"
#include "osmos/gpu/mm.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/model/qtype.h"
#include "hy4_impl.h"

/* y = W x: a decode's the matvec (mv_strided: a token's sums its own), a read's the matrix cores where mm takes the
   type and shape (mm_tokens: MM_MAX_TOKENS a launch through f->x16, a token's sums its own), else the matvec */
static bool mat(Hy4Forward *f, uint32_t type, const void *w, uint32_t rows, uint32_t cols, const float *x, uint32_t n,
                float *y, bool read)
{
  if (read && mm_known(type) && mm_fits(type, rows, cols))
    return mm_tokens(type, w, rows, y, rows, NULL, 0, NULL, 0, cols, x, cols, n, f->x16, f->overflow, false, f->st);
  return mv_strided(type, w, rows, cols, x, cols, n, y, rows, f->st);
}

bool hy4_mat(Hy4Forward *f, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t n, float *y,
             bool read)
{
  return mat(f, t->type, hy4_dev(f, t), rows, cols, x, n, y, read);
}

bool hy4_joint(Hy4Forward *f, const Hy4Joint *j, uint32_t cols, const float *x, float *out)
{
  bool ok = true;
  for (uint32_t r = 0; ok && r < j->n_runs; r++)
    ok = mat(f, j->run[r].type, j->run[r].w, j->run[r].rows, cols, x, 1, out + j->run[r].off, false);
  return ok;
}

/* a read's per-head matvecs on the matrix cores: head h's matrix (base + h head_bytes) over x's columns h xh ..
   (rows xs apart: mm_tokens' strided x) into y + h yh (rows ys apart) - mm_tokens a head (mm_group's tiles want rows
   of a whole slab: k_b's 192 columns are not), a token's sums its own */
static bool heads_mm(Hy4Forward *f, uint32_t heads, const void *base, uint64_t head_bytes, uint32_t rows, uint32_t cols,
                     const float *x, uint32_t xs, uint32_t xh, uint32_t n, float *y, uint32_t ys, uint32_t yh)
{
  bool ok = true;
  for (uint32_t h = 0; ok && h < heads; h++)
    ok = mm_tokens(QTYPE_Q8_0, (const uint8_t *)base + h * head_bytes, rows, y + (uint64_t)h * yh, ys, NULL, 0, NULL, 0,
                   cols, x + (uint64_t)h * xh, xs, n, f->x16, f->overflow, false, f->st);
  return ok;
}

/* Engram's residual. its projection on the matvec in a read too: its gate a signed square root of a normed dot - a
   dot near 0 magnifies its input's rounding (the matrix cores' f16: layer 1 6.3e-3 of its largest against the
   reference on osmos's own input, 2.3e-4 in F32 - the other layers' f16 error) */
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
  return hy4_engram_rows(rows, hy4_fv(f, L->eg_lookup_norm), cols, dim, n, f->emb, f->st) &&
         hy4_mat(f, L->eg_kv, (hc + 1) * D, cols * dim, f->emb, n, f->ekv, false) && /* (both maths: see engram()) */
         hy4_engram_update(f->h, f->ekv, hy4_fv(f, L->eg_kv_b), hy4_fv(f, L->eg_gate), hy4_fv(f, L->eg_hc_scale),
                           hy4_fv(f, L->eg_pre_gamma), hc, D, n, f->st);
}

/* a sparse layer's run: its token keys at their positions, the groups they complete pooled, each token's groups -
   every position the GPU's (pos), none the host's */
static bool indexer(Hy4Forward *f, Hy4Seq *s, int l, uint32_t off, uint32_t len, const float *ixk, const float *ixw)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  const uint32_t ID = (uint32_t)g->ix_dim, IH = (uint32_t)g->ix_heads, pool = (uint32_t)g->ix_pool;
  const uint32_t top = (uint32_t)(g->ix_topk / g->ix_pool);
  float *kn = f->ixkn + (uint64_t)off * ID;
  return ops_rms(ixk + (uint64_t)off * ID, ID, ID, hy4_fv(f, L->ix_k_norm), g->rms_eps, len, kn, ID, f->st) &&
         hy4_rows_at(s->keys[l], kn, f->pos + off, len, ID, f->kring, f->st) &&
         hy4_ix_pool_at(s->keys[l], f->kring, hy4_dev(f, L->ix_pool_value), hy4_dev(f, L->ix_pool_gate), hy4_fv(f, L->ix_pool_ape),
                        hy4_fv(f, L->ix_pool_norm), ID, pool, (uint32_t)g->rope_dim, g->rms_eps, f->freqs_sparse,
                        f->pos + off, len, s->groups[l], f->st) &&
         hy4_ix_select(f->ixq + (uint64_t)off * IH * ID, ixw + (uint64_t)off * IH, s->groups[l], IH, ID, pool, top,
                       f->pos + off, len, f->max_ctx / pool + 1, f->scores + (uint64_t)off * (f->max_ctx / pool + 1),
                       f->sel + (uint64_t)off * top, f->n_sel + off, f->st);
}

/* DFly2's snapshot (a draft layer reads this one): the run's last rows' picks and counts, into its sequence's */
static bool dfly_snap(Hy4Forward *f, Hy4Seq *s, int l, uint32_t off, uint32_t len)
{
  const int slot = f->dfly ? f->dfly_slot[l] : -1;
  if (slot < 0)
    return true;
  const Hy4Geometry *g = &f->w->geo;
  const uint32_t top = (uint32_t)(g->ix_topk / g->ix_pool), keep = len < HY4_DFLY_KEEP ? len : HY4_DFLY_KEEP;
  const uint32_t from = off + len - keep;
  return hipMemcpyAsync(s->dsel + (uint64_t)slot * HY4_DFLY_KEEP * top, f->sel + (uint64_t)from * top,
                        (uint64_t)keep * top * 4, hipMemcpyDeviceToDevice, f->st) == hipSuccess &&
         hipMemcpyAsync(s->dn + (uint64_t)slot * HY4_DFLY_KEEP, f->n_sel + from, (uint64_t)keep * 4,
                        hipMemcpyDeviceToDevice, f->st) == hipSuccess;
}

static bool attention(Hy4Forward *f, const Hy4Run *run, uint32_t runs, uint32_t n, int l, bool read)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  const bool sw = g->sliding[l];
  const uint32_t D = (uint32_t)g->n_embd, H = (uint32_t)g->n_heads[l], ql = (uint32_t)g->q_lora;
  const uint32_t nope = (uint32_t)g->nope_dim, rd = (uint32_t)g->rope_dim, lat = (uint32_t)g->kv_lora;
  const uint32_t vd = (uint32_t)g->v_dim, qk = nope + rd, row = lat + rd, ID = (uint32_t)g->ix_dim;
  const uint32_t IH = (uint32_t)g->ix_heads, top = (uint32_t)(g->ix_topk / g->ix_pool);
  const float *freqs = sw ? f->freqs_swa : f->freqs_sparse, scale = 1.0f / sqrtf((float)qk);
  /* the matrices on xn: a token's one joint (its outputs in jo_attn), else each its own */
  const bool joint = n == 1 && !read;
  const Hy4Joint *J = &f->j_attn[l];
  float *qr = joint ? f->jo_attn + J->off[0] : f->qr, *kva = joint ? f->jo_attn + J->off[1] : f->kva;
  float *gt = joint ? f->jo_attn + J->off[2] : f->gt, *ixk = joint ? f->jo_attn + J->off[3] : f->ixk;
  float *ixw = joint ? f->jo_attn + J->off[4] : f->ixw;
  bool ok = joint ? hy4_joint(f, J, D, f->xn, f->jo_attn)
                  : hy4_mat(f, L->q_a, ql, D, f->xn, n, qr, read) && hy4_mat(f, L->kv_a, row, D, f->xn, n, kva, read) &&
                        hy4_mat(f, L->gate, H * vd, D, f->xn, n, gt, read) &&
                        (sw || (hy4_mat(f, L->ix_k, ID, D, f->xn, n, ixk, read) &&
                                hy4_mat(f, L->ix_proj, IH, D, f->xn, n, ixw, read)));
  ok = ok && ops_rms(qr, ql, ql, hy4_fv(f, L->q_a_norm), g->rms_eps, n, f->qres, ql, f->st) &&
       hy4_mat(f, L->q_b, H * qk, ql, f->qres, n, f->q, read);
  if (ok && !sw)
    ok = hy4_mat(f, L->ix_q_b, IH * ID, ql, f->qres, n, f->ixq, read) &&
         hy4_ix_query(f->ixq, ixw, IH, ID, rd, f->freqs_sparse, f->pos, n, f->st);
  for (uint32_t r = 0; ok && r < runs; r++)
  {
    const uint64_t o = run[r].off;
    const uint32_t m = run[r].len;
    Hy4Seq *s = run[r].s;
    ok = hy4_mla_kv(kva + o * row, hy4_fv(f, L->kv_a_norm), lat, rd, g->rms_eps, freqs, f->pos + o, m, s->cache[l],
                    sw ? f->ring : 0, f->kv8, f->st) &&
         (read ? heads_mm(f, H, hy4_dev(f, L->k_b), (uint64_t)lat * nope / 32 * 34, lat, nope, f->q + o * H * qk, H * qk,
                          qk, m, f->qa + o * H * row, H * row, row) &&
                     hy4_mla_qrope(f->q + o * H * qk, H, qk, row, rd, freqs, f->pos + o, m, f->qa + o * H * row, f->st)
                : hy4_mla_q(f->q + o * H * qk, hy4_dev(f, L->k_b), H, nope, lat, rd, freqs, f->pos + o, m,
                            f->qa + o * H * row, f->table, f->st)) &&
         (sw || indexer(f, s, l, (uint32_t)o, m, ixk, ixw)) && dfly_snap(f, s, l, (uint32_t)o, m) &&
         hy4_mla_attn(f->qa + o * H * row, s->cache[l], sw ? f->ring : 0, f->kv8, hy4_fv(f, L->sinks), H, lat, rd,
                      sw ? (uint32_t)g->window : 0, sw ? NULL : f->sel + o * top, sw ? NULL : f->n_sel + o,
                      sw ? 0 : top, sw ? 0 : (uint32_t)g->ix_pool, scale, f->pos + o, m, f->olat + o * H * lat, f->attn_part, f->st);
  }
  ok = ok && (read ? heads_mm(f, H, hy4_dev(f, L->v_b), (uint64_t)vd * lat / 32 * 34, vd, lat, f->olat, H * lat, lat, n,
                              f->ao, H * vd, vd) &&
                         hy4_gate_mul(f->ao, gt, (uint64_t)n * H * vd, f->st)
                    : hy4_mla_out(f->olat, hy4_dev(f, L->v_b), gt, H, lat, vd, n, f->ao, f->table, f->st));
  return ok &&
         hy4_mat(f, L->out, D, H * vd, f->ao, n, f->out, read) &&
         ops_rms(f->out, D, D, hy4_fv(f, L->post_attn_norm), g->rms_eps, n, f->y, D, f->st) &&
         hy4_hc_post(f->h, f->y, f->post, (uint32_t)g->hc, D, n, f->st);
}

/* the dense layer's FFN: gate and up, SwiGLU, down, the post norm into the streams */
static bool dense(Hy4Forward *f, uint32_t n, int l, bool read)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, ff = (uint32_t)g->n_ff;
  const bool joint = n == 1 && !read; /* gate and up on xn: a token's one joint */
  float *ga = joint ? f->jo_ffn + f->j_ffn[l].off[0] : f->ga, *up = joint ? f->jo_ffn + f->j_ffn[l].off[1] : f->up;
  return (joint ? hy4_joint(f, &f->j_ffn[l], D, f->xn, f->jo_ffn)
                : hy4_mat(f, L->ffn_gate, ff, D, f->xn, n, ga, read) && hy4_mat(f, L->ffn_up, ff, D, f->xn, n, up, read)) &&
         ops_swiglu(ga, up, ff, INFINITY, NULL, n, f->act, f->st) &&
         hy4_mat(f, L->ffn_down, D, ff, f->act, n, f->out, read) &&
         ops_rms(f->out, D, D, hy4_fv(f, L->post_ffn_norm), g->rms_eps, n, f->y, D, f->st) &&
         hy4_hc_post(f->h, f->y, f->post, (uint32_t)g->hc, D, n, f->st);
}

bool hy4_layer(Hy4Forward *f, const Hy4Run *run, uint32_t runs, uint32_t n, int l, bool read)
{
  const Hy4Geometry *g = &f->w->geo;
  const Hy4Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, hc = (uint32_t)g->hc;
  return (!g->engram[l] || engram(f, n, l)) &&
         hy4_hc_pre(f->h, f->fn_attn[l], hy4_fv(f, L->hc_attn_scale), hy4_fv(f, L->hc_attn_base), 2 * hc, hc, D,
                    g->rms_eps, g->hc_eps, g->hc_magnitude, hy4_fv(f, L->attn_norm), n, f->xn, f->post, f->hcs, f->st) &&
         attention(f, run, runs, n, l, read) &&
         hy4_hc_pre(f->h, f->fn_ffn[l], hy4_fv(f, L->hc_ffn_scale), hy4_fv(f, L->hc_ffn_base), 2 * hc, hc, D,
                    g->rms_eps, g->hc_eps, g->hc_magnitude, hy4_fv(f, L->ffn_norm), n, f->xn, f->post, f->hcs, f->st) &&
         (l < g->n_dense ? dense(f, n, l, read) : hy4_moe(f, n, l, read)); /* (each ends in the streams' update) */
}
