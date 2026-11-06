#include "osmos/gpu/mm.h"
#include "q4x_impl.h"

/* a layer's blocks, each as its test holds it to llama.cpp's (tests/gpu/test_q4x_*_gpu): the mixer into b.mixed and
   b.inj, a block's output into b.out, the n-gram layer's into b.h itself */

#define TY(t) ((t)->type)

/* mm on n tokens of x: its f16 (b.x16) made here, or `ready` - the previous call's, of this same x (n within one
   chunk) */
static bool mm_on(Q4xForward *f, uint32_t type, const void *w0, uint32_t rows0, float *y0, uint32_t ys0, const void *w1,
                  uint32_t rows1, float *y1, uint32_t ys1, uint32_t cols, const float *x, uint32_t xs, uint32_t n,
                  bool ready)
{
  bool ok = true;
  for (uint32_t k0 = 0; ok && k0 < n; k0 += MM_MAX_TOKENS)
  {
    const uint32_t m = n - k0 < MM_MAX_TOKENS ? n - k0 : MM_MAX_TOKENS;
    ok = ((ready && n <= MM_MAX_TOKENS) || mm_x16(x + (uint64_t)k0 * xs, xs, m, cols, f->b.x16, f->b.overflow, NULL)) &&
         (w1 ? mm_run2(type, w0, rows0, y0 + (uint64_t)k0 * ys0, ys0, w1, rows1, y1 + (uint64_t)k0 * ys1, ys1, cols,
                       f->b.x16, m, NULL)
             : mm_run(type, w0, rows0, cols, f->b.x16, m, y0 + (uint64_t)k0 * ys0, ys0, NULL));
  }
  return ok;
}

static bool mm_takes(const GgufTensor *t, uint32_t rows, uint32_t cols, uint32_t xs, uint32_t ys)
{
  return mm_known(TY(t)) && mm_fits(TY(t), rows, cols) && xs >= cols && ys >= rows;
}

bool q4x_mat(Q4xForward *f, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t xs,
             uint32_t n, float *y, uint32_t ys, bool ready)
{
  if (!mm_takes(t, rows, cols, xs, ys))
    return mv_strided(TY(t), q4x_dev(f, t), rows, cols, x, xs, n, y, ys, NULL);
  return mm_on(f, TY(t), q4x_dev(f, t), rows, y, ys, NULL, 0, NULL, 0, cols, x, xs, n, ready);
}

bool q4x_mat2(Q4xForward *f, const GgufTensor *t0, uint32_t rows0, float *y0, uint32_t ys0, const GgufTensor *t1,
              uint32_t rows1, float *y1, uint32_t ys1, uint32_t cols, const float *x, uint32_t xs, uint32_t n,
              bool ready)
{
  if (!mm_takes(t0, rows0, cols, xs, ys0) || !mm_takes(t1, rows1, cols, xs, ys1) || TY(t0) != TY(t1))
    return q4x_mat(f, t0, rows0, cols, x, xs, n, y0, ys0, ready) &&
           q4x_mat(f, t1, rows1, cols, x, xs, n, y1, ys1, ready || mm_takes(t0, rows0, cols, xs, ys0));
  return mm_on(f, TY(t0), q4x_dev(f, t0), rows0, y0, ys0, q4x_dev(f, t1), rows1, y1, ys1, cols, x, xs, n, ready);
}

bool q4x_overflow_ok(Q4xForward *f, char *err)
{
  if (hipMemcpy(f->b.pin_overflow, f->b.overflow, 4, hipMemcpyDeviceToHost) != hipSuccess)
    return q4x_fail(err, "a GPU copy failed (the f16 overflow flag)");
  if (!*f->b.pin_overflow)
    return true;
  *f->b.pin_overflow = 0;
  hipMemset(f->b.overflow, 0, 4);
  return q4x_fail(err, "an activation past f16's range (mm's): the step refused, its results not to be read");
}

#define mat(f, t, rows, cols, x, xs, n, y, ys) q4x_mat(f, t, rows, cols, x, xs, n, y, ys, false)

static const float *fv(const Q4xForward *f, const GgufTensor *t) { return (const float *)q4x_dev(f, t); }

bool q4x_mix(Q4xForward *f, const GgufTensor *norm, const GgufTensor *down, const GgufTensor *up,
             const GgufTensor *inject, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t hc = (uint32_t)g->hc, D = (uint32_t)g->n_embd, hd = hc * D, r = (uint32_t)g->hc_rank;
  return q4x_hc_norm(b->h, fv(f, norm), n, hc, D, g->rms_eps, b->xn, NULL) && mat(f, down, r, hd, b->xn, hd, n, b->lo, r) &&
         q4x_scale_silu(b->lo, (uint64_t)n * r, 1.0f / (float)hc, NULL) && mat(f, up, hd, r, b->lo, r, n, b->gate, hd) &&
         q4x_hc_gated_mean(b->xn, b->gate, n, hc, D, b->mixed, NULL) &&
         (!inject || mat(f, inject, hc, hd, b->xn, hd, n, b->inj, hc));
}

bool q4x_deltanet(Q4xForward *f, Q4xSeq *s, int l, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xLayer *L = &f->w->layer[l];
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, vh = (uint32_t)g->ssm_v_heads, in = (uint32_t)g->ssm_inner;
  const uint32_t ch = 2u * (uint32_t)g->ssm_groups * (uint32_t)g->ssm_state + in;
  return q4x_mat2(f, L->qkv, ch, b->qkv, ch, L->gate, in, b->z, in, D, b->mixed, D, n, false) &&
         mat(f, L->alpha, vh, D, b->mixed, D, n, b->al, vh) && mat(f, L->beta, vh, D, b->mixed, D, n, b->be, vh) &&
         q4x_ssm_gates(b->al, b->be, fv(f, L->dt), fv(f, L->a), n, vh, b->g, b->b, NULL) &&
         q4x_conv(b->qkv, n, ch, (uint32_t)g->ssm_conv, 1, fv(f, L->conv), s->conv[l], Q4X_SLOTS, s->cur, b->cv,
                  NULL) &&
         q4x_delta(b->cv, b->g, b->b, n, (uint32_t)g->ssm_groups, vh, (uint32_t)g->ssm_state, g->rms_eps, b->qkn,
                   s->ssm[l], Q4X_SLOTS, s->cur, b->ro, NULL) &&
         q4x_gated_norm(b->ro, fv(f, L->ssm_norm), b->z, n, vh, (uint32_t)g->ssm_state, g->rms_eps, b->fo, NULL) &&
         mat(f, L->ssm_out, D, in, b->fo, in, n, b->out, D);
}

bool q4x_attention(Q4xForward *f, const Q4xLayer *L, float *kc, float *vc, int32_t pos, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, heads = (uint32_t)g->n_heads, kvh = (uint32_t)g->n_kv_heads;
  const uint32_t hd = (uint32_t)g->head_dim, q = heads * hd, kv = kvh * hd;
  return mat(f, L->q, 2 * q, D, b->mixed, D, n, b->qf, 2 * q) &&
         q4x_mat2(f, L->k, kv, b->kr, kv, L->v, kv, vc + (uint64_t)pos * kv, kv, D, b->mixed, D, n,
                  mm_takes(L->q, 2 * q, D, D, 2 * q)) &&
         q4x_qk(b->qf, b->kr, n, heads, kvh, hd, (uint32_t)g->rope_dim, fv(f, L->q_norm), fv(f, L->k_norm), g->rms_eps,
                pos, g->rope_base, b->q, b->gt, kc + (uint64_t)pos * kv, NULL) &&
         q4x_attn(b->q, kc, vc, n, heads, kvh, hd, (uint32_t)pos, b->gt, b->fa, b->ao, NULL) &&
         mat(f, L->out, D, q, b->ao, q, n, b->out, D);
}

/* the cache rows alone (the MTP head's seeding): k and v of b.mixed's n tokens at pos .., k normed and roped */
bool q4x_attention_kv(Q4xForward *f, const Q4xLayer *L, float *kc, float *vc, int32_t pos, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, kvh = (uint32_t)g->n_kv_heads, hd = (uint32_t)g->head_dim, kv = kvh * hd;
  return q4x_mat2(f, L->k, kv, b->kr, kv, L->v, kv, vc + (uint64_t)pos * kv, kv, D, b->mixed, D, n, false) &&
         q4x_qk(b->qf, b->kr, n, 0, kvh, hd, (uint32_t)g->rope_dim, fv(f, L->q_norm), fv(f, L->k_norm), g->rms_eps, pos,
                g->rope_base, b->q, b->gt, kc + (uint64_t)pos * kv, NULL);
}

bool q4x_moe(Q4xForward *f, const Q4xLayer *L, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, E = (uint32_t)g->n_experts, k = (uint32_t)g->n_used, ff = (uint32_t)g->ff_exp;
  const uint32_t sff = (uint32_t)g->ff_shexp, e = n * k, ne = q4x_expert_entries(e, E), slots = ne * MM_TILE;
  const GgufTensor *t[Q4X_GROUP_TENSORS] = {L->exp_gate, L->exp_up, L->exp_down};
  Q4xExpertTiles et = {.cnt = b->cnt, .gidx = b->gidx, .hidx = b->hidx, .xg = b->xg, .yg = b->yg, .xd = b->xd,
                       .yd = b->yd, .where = b->where};
  for (int i = 0; i < Q4X_GROUP_TENSORS; i++)
    et.base[i] = q4x_dev(f, t[i]), et.bytes[i] = t[i]->bytes / E, et.ptr[i] = b->ptr + (uint64_t)i * ne;
  /* each expert read once a tile of the step's tokens that picked it (q4x_expert_tiles, mm_group) */
  return mat(f, L->router, E, D, b->mixed, D, n, b->rl, E) && q4x_route(b->rl, n, E, k, b->ids, b->w, b->rows, NULL) &&
         q4x_expert_tiles(b->ids, e, k, E, D, ff, &et, NULL) &&
         mm_x16_rows(b->mixed, D, b->gidx, slots, D, b->xe16, b->overflow, NULL) &&
         mm_group(TY(t[0]), et.ptr[0], b->cnt, ne, MM_TILE, ff, D, b->xe16, b->xg, D, b->ga, b->yg, ff, NULL) &&
         mm_group(TY(t[1]), et.ptr[1], b->cnt, ne, MM_TILE, ff, D, b->xe16, b->xg, D, b->up, b->yg, ff, NULL) &&
         q4x_swiglu(b->ga, b->up, (uint64_t)e * ff, b->ga, NULL) &&
         mm_x16_rows(b->ga, ff, b->hidx, slots, ff, b->xe16, b->overflow, NULL) &&
         mm_group(TY(t[2]), et.ptr[2], b->cnt, ne, MM_TILE, D, ff, b->xe16, b->xd, ff, b->y, b->yd, D, NULL) &&
         q4x_mat2(f, L->sh_gate, sff, b->shg, sff, L->sh_up, sff, b->shu, sff, D, b->mixed, D, n, false) &&
         q4x_swiglu(b->shg, b->shu, (uint64_t)n * sff, b->shg, NULL) && mat(f, L->sh_down, D, sff, b->shg, sff, n, b->sh, D) &&
         mat(f, L->sh_gate_inp, 1, D, b->mixed, D, n, b->sl, 1) &&
         q4x_moe_sum(b->y, b->where, b->w, n, k, D, b->sh, b->sl, b->out, NULL);
}

/* the step's tokens' rows already in b.ple_rows */
bool q4x_ple(Q4xForward *f, Q4xSeq *s, int l, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xLayer *L = &f->w->layer[l];
  const GgufTensor *tab = f->w->ple_table;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, hc = (uint32_t)g->hc, hd = hc * D, heads = (uint32_t)g->ple_heads;
  return q4x_rows(TY(tab), q4x_dev(f, tab), tab->bytes / tab->dims[1], (uint32_t)tab->dims[0], b->ple_rows, n * heads,
                  b->emb, NULL) &&
         q4x_mat2(f, L->ple_key, hd, b->key, hd, L->ple_value, D, b->val, D, D, b->emb, D, n, false) &&
         q4x_hc_norm(b->key, fv(f, L->ple_norm_key), n, hc, D, g->rms_eps, b->kn, NULL) &&
         q4x_hc_norm(b->h, fv(f, L->ple_norm_query), n, hc, D, g->rms_eps, b->qn, NULL) &&
         q4x_ple_gate(b->kn, b->qn, b->val, n, hc, D, b->pg, b->gated, NULL) &&
         q4x_hc_norm(b->gated, fv(f, L->ple_norm_conv), n, hc, D, g->rms_eps, b->nrm, NULL) &&
         q4x_conv(b->nrm, n, hd, (uint32_t)g->ple_conv, (uint32_t)g->ple_ngram, fv(f, L->ple_conv), s->ple_hist,
                  Q4X_SLOTS, s->cur, b->conv, NULL) &&
         q4x_add2(b->h, b->gated, b->conv, (uint64_t)n * hd, NULL);
}
