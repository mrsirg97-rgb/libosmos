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

/* an op's output also as mm's f16 (b.x16), for the matmul after it - a step within one x16's tokens; else NULL (the
   matmul converts) */
static const Q4xX16 *side(Q4xForward *f, uint32_t n, uint32_t cols, Q4xX16 *s)
{
  if (n > MM_MAX_TOKENS)
    return NULL;
  *s = (Q4xX16){f->b.x16, f->b.overflow, cols, 0};
  return s;
}

bool q4x_overflow_read(Q4xForward *f)
{
  return hipMemcpyAsync(f->b.pin_overflow, f->b.overflow, 4, hipMemcpyDeviceToHost, NULL) == hipSuccess;
}

bool q4x_overflow_ok(Q4xForward *f, char *err)
{
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
  Q4xX16 sd;
  /* each op's output its f16 too (side): the matmul after it ready; b.x16 then the mixed input's, for the block's */
  const bool ready = n <= MM_MAX_TOKENS;
  return q4x_hc_norm(b->h, fv(f, norm), n, hc, D, g->rms_eps, b->xn, side(f, n, hd, &sd), NULL) &&
         q4x_mat(f, down, r, hd, b->xn, hd, n, b->lo, r, ready) &&
         q4x_scale_silu(b->lo, (uint64_t)n * r, 1.0f / (float)hc, side(f, n, r, &sd), NULL) &&
         q4x_mat(f, up, hd, r, b->lo, r, n, b->gate, hd, ready) &&
         q4x_hc_gated_mean(b->xn, b->gate, n, hc, D, b->mixed, side(f, n, D, &sd), NULL) &&
         (!inject || mat(f, inject, hc, hd, b->xn, hd, n, b->inj, hc));
}

/* a step's side copy at a run's rows (t0 its first) */
static const Q4xX16 *side_at(const Q4xX16 *all, uint32_t t0, Q4xX16 *s)
{
  if (!all)
    return NULL;
  *s = *all, s->row0 = t0;
  return s;
}

bool q4x_deltanet(Q4xForward *f, const Q4xRun *r, uint32_t runs, int l, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xLayer *L = &f->w->layer[l];
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, vh = (uint32_t)g->ssm_v_heads, in = (uint32_t)g->ssm_inner;
  const uint32_t ch = 2u * (uint32_t)g->ssm_groups * (uint32_t)g->ssm_state + in;
  Q4xX16 sd;
  const bool ready = n <= MM_MAX_TOKENS; /* b.x16 the mixer's mixed (q4x_mix), then the gated norm's */
  bool ok = q4x_mat2(f, L->qkv, ch, b->qkv, ch, L->gate, in, b->z, in, D, b->mixed, D, n, ready) &&
            mat(f, L->alpha, vh, D, b->mixed, D, n, b->al, vh) && mat(f, L->beta, vh, D, b->mixed, D, n, b->be, vh) &&
            q4x_ssm_gates(b->al, b->be, fv(f, L->dt), fv(f, L->a), n, vh, b->g, b->b, NULL);
  for (uint32_t i = 0; ok && i < runs; i++) /* a sequence's conv history and state over its own tokens */
  {
    const Q4xSeq *s = r[i].s;
    const uint64_t t0 = r[i].t0;
    ok = q4x_conv(b->qkv + t0 * ch, r[i].n, ch, (uint32_t)g->ssm_conv, 1, fv(f, L->conv), s->conv[l], Q4X_SLOTS,
                  s->cur, b->cv + t0 * ch, NULL) &&
         q4x_delta(b->cv + t0 * ch, b->g + t0 * vh, b->b + t0 * vh, r[i].n, (uint32_t)g->ssm_groups, vh,
                   (uint32_t)g->ssm_state, g->rms_eps, b->qkn, s->ssm[l], Q4X_SLOTS, s->cur, b->ro + t0 * in, NULL);
  }
  return ok &&
         q4x_gated_norm(b->ro, fv(f, L->ssm_norm), b->z, n, vh, (uint32_t)g->ssm_state, g->rms_eps, b->fo,
                        side(f, n, in, &sd), NULL) &&
         q4x_mat(f, L->ssm_out, D, in, b->fo, in, n, b->out, D, ready);
}

/* QSA's keys: the step's raw indexer keys (b.ikr, from b.mixed), each run's blocks it completes pooled into its own
   (q4x_ix_pool), its ring then its last raw keys */
static bool runs_ix_keys(Q4xForward *f, const Q4xLayer *L, const Q4xCache *c, uint32_t runs, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, ratio = (uint32_t)g->ix_ratio;
  bool ok = mat(f, L->ix_k, Q4X_IX_DIM, D, b->mixed, D, n, b->ikr, Q4X_IX_DIM);
  for (uint32_t i = 0; ok && i < runs; i++)
    ok = q4x_ix_pool(b->ikr + (uint64_t)c[i].t0 * Q4X_IX_DIM, c[i].n, c[i].pos, ratio, c[i].ixr, Q4X_IX_RING(ratio),
                     (uint32_t)g->rope_dim, fv(f, L->ix_k_norm), g->rms_eps, g->rope_base, c[i].ixk, NULL);
  return ok;
}

/* QSA's picks: the step's indexer queries (b.iqr, from b.mixed), each run's normed and roped at its positions
   (b.iq), its tokens' blocks picked from its pooled keys (b.sel, b.npick at its rows) */
static bool runs_ix_pick(Q4xForward *f, const Q4xLayer *L, const Q4xCache *c, uint32_t runs, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, ratio = (uint32_t)g->ix_ratio, blocks = (uint32_t)g->ix_top_k / ratio;
  const uint32_t heads = (uint32_t)g->ix_heads, ixq = heads * Q4X_IX_DIM;
  bool ok = mat(f, L->ix_q, ixq, D, b->mixed, D, n, b->iqr, ixq);
  for (uint32_t i = 0; ok && i < runs; i++)
  {
    const uint64_t t0 = c[i].t0;
    ok = q4x_ix_q(b->iqr + t0 * ixq, c[i].n, heads, (uint32_t)g->rope_dim, fv(f, L->ix_q_norm), g->rms_eps, c[i].pos,
                  g->rope_base, b->iq + t0 * ixq, NULL) &&
         q4x_ix_pick(b->iq + t0 * ixq, c[i].ixk, c[i].n, heads, c[i].pos, ratio, blocks, b->ixs,
                     q4x_ix_scratch(f->max_ctx, ratio), b->sel + t0 * blocks, b->npick + t0, NULL);
  }
  return ok;
}

/* each run's k normed and roped at its own positions (b.kr -> b.kq; with heads, q and its gate too: b.qf -> b.q,
   b.gt), then its k and v (b.vr) into its cache's rows (Q8_0: q4x_kv_put) */
static bool runs_kv(Q4xForward *f, const Q4xLayer *L, const Q4xCache *c, uint32_t runs, uint32_t heads)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t kvh = (uint32_t)g->n_kv_heads, hd = (uint32_t)g->head_dim, q = (uint32_t)g->n_heads * hd;
  const uint32_t kv = kvh * hd;
  const uint64_t row = q4x_kv_row_bytes(kv);
  bool ok = true;
  for (uint32_t i = 0; ok && i < runs; i++)
  {
    const uint64_t t0 = c[i].t0;
    ok = q4x_qk(b->qf + t0 * 2 * q, b->kr + t0 * kv, c[i].n, heads, kvh, hd, (uint32_t)g->rope_dim, fv(f, L->q_norm),
                fv(f, L->k_norm), g->rms_eps, c[i].pos, g->rope_base, b->q + t0 * q, b->gt + t0 * q,
                b->kq + t0 * kv, NULL) &&
         q4x_kv_put(b->kq + t0 * kv, b->vr + t0 * kv, c[i].n, kv, c[i].kc + (uint64_t)c[i].pos * row,
                    c[i].vc + (uint64_t)c[i].pos * row, NULL);
  }
  return ok;
}

bool q4x_attention(Q4xForward *f, const Q4xLayer *L, const Q4xCache *c, uint32_t runs, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, heads = (uint32_t)g->n_heads, kvh = (uint32_t)g->n_kv_heads;
  const uint32_t hd = (uint32_t)g->head_dim, q = heads * hd, kv = kvh * hd;
  Q4xX16 sd, rs;
  const Q4xX16 *all = side(f, n, q, &sd);
  const bool ready = n <= MM_MAX_TOKENS; /* b.x16 the mixer's mixed (q4x_mix), then the attention's output */
  const uint32_t ratio = (uint32_t)g->ix_ratio, blocks = (uint32_t)g->ix_top_k / ratio;
  /* the indexer's before the attention: b.x16 the mixed until the attention's output overwrites it */
  bool ok = q4x_mat(f, L->q, 2 * q, D, b->mixed, D, n, b->qf, 2 * q, ready) &&
            q4x_mat2(f, L->k, kv, b->kr, kv, L->v, kv, b->vr, kv, D, b->mixed, D, n, ready) &&
            runs_kv(f, L, c, runs, heads) && runs_ix_keys(f, L, c, runs, n) && runs_ix_pick(f, L, c, runs, n);
  for (uint32_t i = 0; ok && i < runs; i++) /* each run over its own cache and picks, its output at its rows */
  {
    const uint64_t t0 = c[i].t0;
    const Q4xPick pick = {b->sel + t0 * blocks, b->npick + t0, ratio, blocks};
    ok = q4x_attn(b->q + t0 * q, c[i].kc, c[i].vc, c[i].n, heads, kvh, hd, (uint32_t)c[i].pos, b->gt + t0 * q, &pick,
                  b->fa, b->ao + t0 * q, side_at(all, c[i].t0, &rs), NULL);
  }
  return ok && q4x_mat(f, L->out, D, q, b->ao, q, n, b->out, D, ready);
}

/* the cache rows alone (the MTP head's seeding): k and v of b.mixed's tokens, k normed and roped; the indexer's keys */
bool q4x_attention_kv(Q4xForward *f, const Q4xLayer *L, const Q4xCache *c, uint32_t runs, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, kvh = (uint32_t)g->n_kv_heads, hd = (uint32_t)g->head_dim, kv = kvh * hd;
  return q4x_mat2(f, L->k, kv, b->kr, kv, L->v, kv, b->vr, kv, D, b->mixed, D, n,
                  n <= MM_MAX_TOKENS) && /* b.x16 the mixer's mixed (q4x_mix) */
         runs_kv(f, L, c, runs, 0) && runs_ix_keys(f, L, c, runs, n);
}

/* an expert tensor over the tables' runs, a launch each (mm_group: a part's slab decoded once for its 16 << r tokens) */
static bool experts(const GgufTensor *t, const void **ptr, const Q4xBuf *b, const Q4xExpertBounds *bd, uint32_t rows,
                    uint32_t cols, const uint32_t *xo, float *y, const uint32_t *yo)
{
  bool ok = true;
  for (int r = 0; ok && r < Q4X_TILE_RUNS; r++)
    if (bd->count[r])
    {
      const uint32_t at = bd->first[r];
      ok = mm_group(TY(t), ptr + at, b->cnt + at, bd->count[r], MM_TILE << r, rows, cols, b->xe16, xo + at, cols, y,
                    yo + at, rows, NULL);
    }
  return ok;
}

bool q4x_moe(Q4xForward *f, const Q4xLayer *L, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, E = (uint32_t)g->n_experts, k = (uint32_t)g->n_used, ff = (uint32_t)g->ff_exp;
  const uint32_t sff = (uint32_t)g->ff_shexp, e = n * k;
  Q4xExpertBounds bd;
  q4x_expert_bounds(e, E, &bd);
  const GgufTensor *t[Q4X_GROUP_TENSORS] = {L->exp_gate, L->exp_up, L->exp_down};
  Q4xX16 sd;
  Q4xExpertTiles et = {.cnt = b->cnt, .gidx = b->gidx, .hidx = b->hidx, .xg = b->xg, .yg = b->yg, .xd = b->xd,
                       .yd = b->yd, .where = b->where};
  for (int i = 0; i < Q4X_GROUP_TENSORS; i++)
    et.base[i] = q4x_dev(f, t[i]), et.bytes[i] = t[i]->bytes / E, et.ptr[i] = b->ptr + (uint64_t)i * bd.entries;
  /* each expert read once for up to 128 of the step's tokens that picked it (q4x_expert_tiles, mm_group) */
  return mat(f, L->router, E, D, b->mixed, D, n, b->rl, E) && q4x_route(b->rl, n, E, k, b->ids, b->w, b->rows, NULL) &&
         q4x_expert_tiles(b->ids, e, k, E, D, ff, &et, NULL) &&
         mm_x16_rows(b->mixed, D, b->gidx, bd.slots, D, b->xe16, b->overflow, NULL) &&
         experts(t[0], et.ptr[0], b, &bd, ff, D, b->xg, b->ga, b->yg) &&
         experts(t[1], et.ptr[1], b, &bd, ff, D, b->xg, b->up, b->yg) &&
         q4x_swiglu(b->ga, b->up, (uint64_t)e * ff, b->ga, NULL, NULL) &&
         mm_x16_rows(b->ga, ff, b->hidx, bd.slots, ff, b->xe16, b->overflow, NULL) &&
         experts(t[2], et.ptr[2], b, &bd, D, ff, b->xd, b->y, b->yd) &&
         /* b.x16 still the mixer's mixed: the experts' are their own (b.xe16) */
         q4x_mat2(f, L->sh_gate, sff, b->shg, sff, L->sh_up, sff, b->shu, sff, D, b->mixed, D, n, n <= MM_MAX_TOKENS) &&
         q4x_swiglu(b->shg, b->shu, (uint64_t)n * sff, b->shg, side(f, n, sff, &sd), NULL) &&
         q4x_mat(f, L->sh_down, D, sff, b->shg, sff, n, b->sh, D, n <= MM_MAX_TOKENS) &&
         mat(f, L->sh_gate_inp, 1, D, b->mixed, D, n, b->sl, 1) &&
         q4x_moe_sum(b->y, b->where, b->w, n, k, D, b->sh, b->sl, b->out, NULL);
}

/* the step's tokens' rows already in b.ple_rows */
bool q4x_ple(Q4xForward *f, const Q4xRun *r, uint32_t runs, int l, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xLayer *L = &f->w->layer[l];
  const GgufTensor *tab = f->w->ple_table;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, hc = (uint32_t)g->hc, hd = hc * D, heads = (uint32_t)g->ple_heads;
  bool ok = q4x_rows(TY(tab), q4x_dev(f, tab), tab->bytes / tab->dims[1], (uint32_t)tab->dims[0], b->ple_rows, n * heads,
                  b->emb, NULL) &&
         q4x_mat2(f, L->ple_key, hd, b->key, hd, L->ple_value, D, b->val, D, D, b->emb, D, n, false) &&
         q4x_hc_norm(b->key, fv(f, L->ple_norm_key), n, hc, D, g->rms_eps, b->kn, NULL, NULL) &&
         q4x_hc_norm(b->h, fv(f, L->ple_norm_query), n, hc, D, g->rms_eps, b->qn, NULL, NULL) &&
         q4x_ple_gate(b->kn, b->qn, b->val, n, hc, D, b->pg, b->gated, NULL) &&
         q4x_hc_norm(b->gated, fv(f, L->ple_norm_conv), n, hc, D, g->rms_eps, b->nrm, NULL, NULL);
  for (uint32_t i = 0; ok && i < runs; i++) /* a sequence's conv history over its own tokens */
  {
    const uint64_t t0 = r[i].t0 * (uint64_t)hd;
    ok = q4x_conv(b->nrm + t0, r[i].n, hd, (uint32_t)g->ple_conv, (uint32_t)g->ple_ngram, fv(f, L->ple_conv),
                  r[i].s->ple_hist, Q4X_SLOTS, r[i].s->cur, b->conv + t0, NULL);
  }
  return ok && q4x_add2(b->h, b->gated, b->conv, (uint64_t)n * hd, NULL);
}
