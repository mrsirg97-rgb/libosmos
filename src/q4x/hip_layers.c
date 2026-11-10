#include "osmos/gpu/mm.h"
#include "osmos/model/qtype.h"
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
    ok = ((ready && n <= MM_MAX_TOKENS) || mm_x16(x + (uint64_t)k0 * xs, xs, m, cols, f->b.x16, f->b.overflow, f->st)) &&
         (w1 ? mm_run2(type, w0, rows0, y0 + (uint64_t)k0 * ys0, ys0, w1, rows1, y1 + (uint64_t)k0 * ys1, ys1, cols,
                       f->b.x16, m, f->st)
             : mm_run(type, w0, rows0, cols, f->b.x16, m, y0 + (uint64_t)k0 * ys0, ys0, f->st));
  }
  return ok;
}

static bool mm_takes(const GgufTensor *t, uint32_t rows, uint32_t cols, uint32_t xs, uint32_t ys)
{
  return mm_known(TY(t)) && mm_fits(TY(t), rows, cols) && xs >= cols && ys >= rows;
}

/* a small step's work beside the main stream's on s2 (the shared expert's way): only a matvec's - an mm converts its
   input into b.x16, which the main stream's matmuls read meanwhile. `ts` the work's weights (rows, cols each): every
   one the matvec's, else none goes to s2 */
static bool beside(const GgufTensor *const *ts, const uint32_t *rows, const uint32_t *cols, int nt, uint32_t n)
{
  if (n > Q4X_SIDE_STEP)
    return false;
  for (int i = 0; i < nt; i++)
    if (mm_takes(ts[i], rows[i], cols[i], cols[i], rows[i]))
      return false;
  return true;
}

/* s2 to wait for the main stream's work so far (its inputs written), then take the calls; a test's delay first */
static bool to_side(Q4xForward *f)
{
  if (hipEventRecord(f->e_mix, NULL) != hipSuccess || hipStreamWaitEvent(f->s2, f->e_mix, 0) != hipSuccess)
    return false;
  f->st = f->s2;
  return !f->side_spin || q4x_spin(f->side_spin, f->s2);
}

bool q4x_mat(Q4xForward *f, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t xs,
             uint32_t n, float *y, uint32_t ys, bool ready)
{
  if (!mm_takes(t, rows, cols, xs, ys))
    return mv_strided(TY(t), q4x_dev(f, t), rows, cols, x, xs, n, y, ys, f->st);
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

bool q4x_inj_wait(Q4xForward *f)
{
  if (!f->inj_side)
    return true;
  f->inj_side = false;
  return hipStreamWaitEvent(NULL, f->e_inj, 0) == hipSuccess;
}

/* the inject (b.inj = inject . b.xn): a small step's on s2, from the norm's output, beside the down, silu, up and
   gated mean - the same kernel on the same inputs, the same bits (the shared expert's way, q4x_moe_shared) */
static bool inject_of(Q4xForward *f, const GgufTensor *inject, uint32_t n, uint32_t hc, uint32_t hd)
{
  Q4xBuf *b = &f->b;
  if (!beside(&inject, &hc, &hd, 1, n))
    return mat(f, inject, hc, hd, b->xn, hd, n, b->inj, hc);
  const bool ok = to_side(f) && mat(f, inject, hc, hd, b->xn, hd, n, b->inj, hc);
  f->st = NULL;
  return ok && hipEventRecord(f->e_inj, f->s2) == hipSuccess && (f->inj_side = true);
}

/* the mixer's up and gated mean (b.mixed, its f16 beside): mm_run_gated_mean - the mean the up's epilogue, the up's
   F32 (b.gate, hc n_embd a token) neither written nor read back: at a 2K step 168 MB a mixer fewer (a 4K prompt's
   up and mean 366 -> 281 ms, low-power) - from the silu's f16 in b.x16b (x16b true), else the two calls (the up from
   b.lo). the same bits either way (test_gated_mean_gpu) */
static bool up_mean(Q4xForward *f, const GgufTensor *up, uint32_t n, uint32_t hc, uint32_t D, uint32_t r, bool x16b)
{
  Q4xBuf *b = &f->b;
  const uint32_t hd = hc * D;
  Q4xX16 sd;
  const Q4xX16 *ms = side(f, n, D, &sd);
  if (x16b && mm_takes(up, hd, r, r, hd) &&
      mm_run_gated_mean(TY(up), q4x_dev(f, up), hc, D, r, b->x16b, n, b->xn, b->mixed, ms ? ms->x16 : NULL,
                        ms ? ms->row0 : 0, b->overflow, f->st))
    return true;
  return (x16b && mm_takes(up, hd, r, r, hd) ? mm_run(TY(up), q4x_dev(f, up), hd, r, b->x16b, n, b->gate, hd, f->st)
                                             : q4x_mat(f, up, hd, r, b->lo, r, n, b->gate, hd, false)) &&
         q4x_hc_gated_mean(b->xn, b->gate, n, hc, D, b->mixed, ms, NULL);
}

/* the mixer's down, silu, then up_mean: a small step's down and silu one launch (mm_run_silu: its f16 into b.x16b -
   b.x16 the down's own input meanwhile), the same bits as the two; otherwise the two, the silu's f16 into b.x16b */
static bool down_up(Q4xForward *f, const GgufTensor *down, const GgufTensor *up, uint32_t n, uint32_t hc, uint32_t D,
                    uint32_t r, bool ready)
{
  Q4xBuf *b = &f->b;
  const uint32_t hd = hc * D;
  const float scale = 1.0f / (float)hc;
  if (ready && n <= MM_SILU_MAX_TOKENS && mm_takes(down, r, hd, hd, r) && mm_takes(up, hd, r, r, hd) &&
      mm_run_silu(TY(down), q4x_dev(f, down), r, hd, b->x16, n, b->lo, r, scale, b->x16b, b->overflow, f->st))
    return up_mean(f, up, n, hc, D, r, true);
  const Q4xX16 sx = {b->x16b, b->overflow, r, 0}; /* the silu's f16 for up_mean: b.x16 the mixer's input meanwhile */
  return q4x_mat(f, down, r, hd, b->xn, hd, n, b->lo, r, ready) &&
         q4x_scale_silu(b->lo, (uint64_t)n * r, scale, ready ? &sx : NULL, NULL) && up_mean(f, up, n, hc, D, r, ready);
}

bool q4x_mix(Q4xForward *f, const GgufTensor *norm, const GgufTensor *down, const GgufTensor *up,
             const GgufTensor *inject, uint32_t n, bool combine)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t hc = (uint32_t)g->hc, D = (uint32_t)g->n_embd, hd = hc * D, r = (uint32_t)g->hc_rank;
  Q4xX16 sd;
  /* each op's output its f16 too (side): the matmul after it ready; b.x16 then the mixed input's, for the block's */
  const bool ready = n <= MM_MAX_TOKENS;
  if (!q4x_inj_wait(f)) /* the last inject done: b.inj read here, b.xn written */
    return false;
  /* a prompt's step: the norm and its inject one launch (q4x_hc_norm_inject: xn not read back, mv's bits) where the
     inject is F32 - a 4K prompt's norms and injects 479 -> 361 ms, low-power. a small step keeps the two, the inject
     beside on s2: a block a token there would be the one block (decode 27.4 -> 25.6 tok/s) */
  if (n > Q4X_SIDE_STEP && inject && TY(inject) == QTYPE_F32 && inject->dims[0] == hd && inject->dims[1] == hc &&
      q4x_hc_norm_inject(b->h, b->out, b->inj, n, hc, D, 1.0f / (float)hc, fv(f, norm), g->rms_eps, b->xn,
                         side(f, n, hd, &sd), fv(f, inject), b->inj, combine, NULL))
    return down_up(f, down, up, n, hc, D, r, ready);
  return (combine ? q4x_hc_combine_norm(b->h, b->out, b->inj, n, hc, D, 1.0f / (float)hc, fv(f, norm), g->rms_eps,
                                        b->xn, side(f, n, hd, &sd), NULL)
                  : q4x_hc_norm(b->h, fv(f, norm), n, hc, D, g->rms_eps, b->xn, side(f, n, hd, &sd), NULL)) &&
         (!inject || inject_of(f, inject, n, hc, hd)) && down_up(f, down, up, n, hc, D, r, ready);
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
  /* the gates (alpha, beta: hc rows, latency-bound) read b.mixed alone and only the delta reads them: a small step's
     on s2 beside the qkv / gate matmul and the conv, waited for before the delta - the same kernels, the same bits */
  const GgufTensor *gt[2] = {L->alpha, L->beta};
  const uint32_t gr[2] = {vh, vh}, gc[2] = {D, D};
  const bool side_gates = beside(gt, gr, gc, 2, n);
  bool ok = !side_gates || to_side(f);
  ok = ok && mat(f, L->alpha, vh, D, b->mixed, D, n, b->al, vh) && mat(f, L->beta, vh, D, b->mixed, D, n, b->be, vh) &&
       q4x_ssm_gates(b->al, b->be, fv(f, L->dt), fv(f, L->a), n, vh, b->g, b->b, f->st);
  f->st = NULL;
  ok = ok && (!side_gates || hipEventRecord(f->e_gates, f->s2) == hipSuccess) &&
       q4x_mat2(f, L->qkv, ch, b->qkv, ch, L->gate, in, b->z, in, D, b->mixed, D, n, ready);
  for (uint32_t i = 0; ok && i < runs; i++) /* a sequence's conv history and state over its own tokens */
  {
    const Q4xSeq *s = r[i].s;
    const uint64_t t0 = r[i].t0;
    ok = q4x_conv(b->qkv + t0 * ch, r[i].n, ch, (uint32_t)g->ssm_conv, 1, fv(f, L->conv), s->conv[l], Q4X_SLOTS,
                  s->cur, b->cv + t0 * ch, NULL) &&
         (i || !side_gates || hipStreamWaitEvent(NULL, f->e_gates, 0) == hipSuccess) && /* the gates done */
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
                     &f->rope, fv(f, L->ix_k_norm), g->rms_eps, c[i].ixk, f->st);
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
    ok = q4x_ix_q(b->iqr + t0 * ixq, c[i].n, heads, &f->rope, fv(f, L->ix_q_norm), g->rms_eps, c[i].pos,
                  b->iq + t0 * ixq, f->st) &&
         q4x_ix_pick(b->iq + t0 * ixq, c[i].ixk, c[i].n, heads, c[i].pos, ratio, blocks, b->ixs,
                     q4x_ix_scratch(f->max_ctx, ratio), b->sel + t0 * blocks, b->npick + t0, f->st);
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
    ok = q4x_qk(b->qf + t0 * 2 * q, b->kr + t0 * kv, c[i].n, heads, kvh, hd, &f->rope, fv(f, L->q_norm),
                fv(f, L->k_norm), g->rms_eps, c[i].pos, b->q + t0 * q, b->gt + t0 * q,
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
  /* the indexer (its keys and picks from b.mixed, read by the attention alone): a small step's on s2 beside q, k, v and
     the cache's rows, waited for before the attention - else before the attention, b.x16 the mixed until the
     attention's output overwrites it */
  const uint32_t ixq = (uint32_t)g->ix_heads * Q4X_IX_DIM;
  const GgufTensor *it[2] = {L->ix_k, L->ix_q};
  const uint32_t ir[2] = {Q4X_IX_DIM, ixq}, ic[2] = {D, D};
  const bool side_ix = beside(it, ir, ic, 2, n);
  bool ok = !side_ix || (to_side(f) && runs_ix_keys(f, L, c, runs, n) && runs_ix_pick(f, L, c, runs, n));
  f->st = NULL;
  ok = ok && (!side_ix || hipEventRecord(f->e_ix, f->s2) == hipSuccess) &&
       q4x_mat(f, L->q, 2 * q, D, b->mixed, D, n, b->qf, 2 * q, ready) &&
       q4x_mat2(f, L->k, kv, b->kr, kv, L->v, kv, b->vr, kv, D, b->mixed, D, n, ready) && runs_kv(f, L, c, runs, heads) &&
       (side_ix || (runs_ix_keys(f, L, c, runs, n) && runs_ix_pick(f, L, c, runs, n))) &&
       (!side_ix || hipStreamWaitEvent(NULL, f->e_ix, 0) == hipSuccess); /* the picks done */
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
  const uint32_t e = n * k;
  Q4xExpertBounds bd;
  q4x_expert_bounds(e, E, &bd);
  const GgufTensor *t[Q4X_GROUP_TENSORS] = {L->exp_gate, L->exp_up, L->exp_down};
  Q4xExpertTiles et = {.cnt = b->cnt, .gidx = b->gidx, .hidx = b->hidx, .xg = b->xg, .yg = b->yg, .xd = b->xd,
                       .yd = b->yd, .where = b->where, .off = b->eoff};
  for (int i = 0; i < Q4X_GROUP_TENSORS; i++)
    et.base[i] = q4x_dev(f, t[i]), et.bytes[i] = t[i]->bytes / E, et.ptr[i] = b->ptr + (uint64_t)i * bd.entries;
  /* each expert read once for up to 128 of the step's tokens that picked it (q4x_expert_tiles, mm_group) */
  return mat(f, L->router, E, D, b->mixed, D, n, b->rl, E) && q4x_route(b->rl, n, E, k, b->ids, b->w, b->rows, NULL) &&
         q4x_expert_tiles(b->ids, e, k, E, D, ff, &et, NULL) &&
         mm_x16_rows(b->mixed, D, b->gidx, bd.slots, D, b->xe16, b->overflow, NULL) &&
         experts(t[0], et.ptr[0], b, &bd, ff, D, b->xg, b->ga, b->yg) &&
         experts(t[1], et.ptr[1], b, &bd, ff, D, b->xg, b->up, b->yg) &&
         q4x_swiglu_rows(b->ga, b->up, ff, b->hidx, bd.slots, b->xe16, b->overflow, NULL) &&
         experts(t[2], et.ptr[2], b, &bd, D, ff, b->xd, b->y, b->yd) &&
         (n > Q4X_SIDE_STEP || hipStreamWaitEvent(NULL, f->e_out, 0) == hipSuccess) && /* the shared expert's done */
         q4x_moe_sum(b->y, b->where, b->w, n, k, D, b->sh, b->sl, b->out, NULL);
}

/* the shared expert and its gate's logit (b.sh, b.sl) - after the FFN's mixer, b.x16 its mixed (q4x_mix): a token's
   own, so with the layer's part (a step past MM_MAX_TOKENS: a sub-batch at a time, b.x16 the mixer's still) */
bool q4x_moe_shared(Q4xForward *f, const Q4xLayer *L, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, sff = (uint32_t)g->ff_shexp;
  Q4xX16 sd;
  /* a small step's (decode, a draft's verify): the shared expert on the second stream, beside the routed experts -
     both latency-bound at a few tokens, together nearer the memory's bandwidth (plain decode 23.9 -> 25.8 tok/s,
     drafted 40.1 -> 43.6, low-power). it reads the mixer's outputs alone (b.mixed, b.x16) and writes b.sh, b.sl and
     its own f16 into b.x16, which the routed experts do not touch (theirs b.xe16); q4x_moe_sum waits for it. the same
     kernels on the same inputs: the same bits */
  const bool side_stream = n <= Q4X_SIDE_STEP;
  if (side_stream && (hipEventRecord(f->e_in, NULL) != hipSuccess || hipStreamWaitEvent(f->s2, f->e_in, 0) != hipSuccess))
    return false;
  if (side_stream)
    f->st = f->s2;
  if (side_stream && f->side_spin && !q4x_spin(f->side_spin, f->s2)) /* a test's: the shared expert last */
    return false;
  const bool ok = q4x_mat2(f, L->sh_gate, sff, b->shg, sff, L->sh_up, sff, b->shu, sff, D, b->mixed, D, n,
                           n <= MM_MAX_TOKENS) &&
                  q4x_swiglu(b->shg, b->shu, (uint64_t)n * sff, b->shg, side(f, n, sff, &sd), f->st) &&
                  q4x_mat(f, L->sh_down, D, sff, b->shg, sff, n, b->sh, D, n <= MM_MAX_TOKENS) &&
                  mat(f, L->sh_gate_inp, 1, D, b->mixed, D, n, b->sl, 1);
  f->st = NULL;
  return ok && (!side_stream || hipEventRecord(f->e_out, f->s2) == hipSuccess);
}

/* the step's tokens' rows already in b.ple_rows */
/* stream_ngram: the step's n tokens' rows (b.pin_rows', ple_heads a token) begun on the readers into the stage - once
   the last step's copy of it is done (ple_copied); a step that failed before its n-gram layer: its rows waited for first */
bool q4x_ple_issue(Q4xForward *f, uint32_t n)
{
  const GgufTensor *tab = f->w->ple_table;
  const uint64_t rb = tab->bytes / tab->dims[1];
  const uint32_t rows = n * (uint32_t)f->w->geo.ple_heads;
  if (f->ple_pending)
    (void)rowread_wait(f->ple_read), f->ple_pending = false;
  if (hipEventSynchronize(f->ple_copied) != hipSuccess)
    return false;
  for (uint32_t i = 0; i < rows; i++)
    f->ple_off[i] = tab->offset + (uint64_t)f->b.pin_rows[i] * rb;
  if (!rowread_start(f->ple_read, f->ple_fd, f->ple_off, rows, (uint32_t)rb, f->ple_stage))
    return false;
  f->ple_pending = true, f->ple_n = rows;
  return true;
}

/* the n-gram table's rows of the n tokens: the held table's own, or (stream_ngram) the stage's - read for the step,
   copied in whole at its first part (sub-batch 0: b's views at their start), each part's in its view of ple_sbuf */
static bool ple_table_rows(Q4xForward *f, uint32_t n)
{
  const GgufTensor *tab = f->w->ple_table;
  Q4xBuf *b = &f->b;
  const uint64_t rb = tab->bytes / tab->dims[1];
  const uint32_t heads = (uint32_t)f->w->geo.ple_heads;
  if (!f->stream_ngram)
    return q4x_rows(TY(tab), q4x_dev(f, tab), rb, (uint32_t)tab->dims[0], b->ple_rows, n * heads, b->emb, NULL);
  if (f->ple_pending)
  {
    f->ple_pending = false;
    if (!rowread_wait(f->ple_read) ||
        hipMemcpyAsync(b->ple_sbuf, f->ple_stage, (uint64_t)f->ple_n * rb, hipMemcpyHostToDevice, NULL) != hipSuccess ||
        hipEventRecord(f->ple_copied, NULL) != hipSuccess)
      return false;
  }
  return q4x_rows(TY(tab), b->ple_sbuf, rb, (uint32_t)tab->dims[0], b->ple_iota, n * heads, b->emb, NULL);
}

bool q4x_ple(Q4xForward *f, const Q4xRun *r, uint32_t runs, int l, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xLayer *L = &f->w->layer[l];
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, hc = (uint32_t)g->hc, hd = hc * D;
  bool ok = ple_table_rows(f, n) &&
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
