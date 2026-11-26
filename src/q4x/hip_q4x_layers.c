/* Qwen's layer blocks (q4x_impl.h), each llama.cpp's math: the matrices, the mixer into b.mixed and b.inj, the
   DeltaNet and the attention (its index's keys and picks) */
#include "osmos/mem/membrane.h"
#include "osmos/model/moe_route.h"
#include "osmos/gpu/mm.h"
#include "osmos/model/qtype.h"
#include "q4x_impl.h"
#include "osmos/err.h"


#define TY(t) ((t)->type)
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

/* s2 to wait for the main stream's work so far (its inputs written), then take the calls; side_spin's delay first */
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
  return mm_tokens(TY(t), q4x_dev(f, t), rows, y, ys, NULL, 0, NULL, 0, cols, x, xs, n, f->b.x16, f->b.overflow, ready,
                   f->st);
}

bool q4x_mat2(Q4xForward *f, const GgufTensor *t0, uint32_t rows0, float *y0, uint32_t ys0, const GgufTensor *t1,
              uint32_t rows1, float *y1, uint32_t ys1, uint32_t cols, const float *x, uint32_t xs, uint32_t n,
              bool ready)
{
  if (!mm_takes(t0, rows0, cols, xs, ys0) || !mm_takes(t1, rows1, cols, xs, ys1) || TY(t0) != TY(t1))
    return q4x_mat(f, t0, rows0, cols, x, xs, n, y0, ys0, ready) &&
           q4x_mat(f, t1, rows1, cols, x, xs, n, y1, ys1, ready || mm_takes(t0, rows0, cols, xs, ys0));
  return mm_tokens(TY(t0), q4x_dev(f, t0), rows0, y0, ys0, q4x_dev(f, t1), rows1, y1, ys1, cols, x, xs, n, f->b.x16,
                   f->b.overflow, ready, f->st);
}

/* an op's output also as mm's f16 (b.x16), for the matmul after it - a step within one x16's tokens; else NULL (the
   matmul converts) */
const Q4xX16 *q4x_side(Q4xForward *f, uint32_t n, uint32_t cols, Q4xX16 *s)
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
  char rerr[OSMOS_ERR] = ""; /* (a reset that fails: owed - the next step's start says so) */
  overflow_reset(&f->ovf, rerr);
  return osmos_fail(err, "an activation past f16's range (mm's): the step refused, its results not to be read");
}

#define mat(f, t, rows, cols, x, xs, n, y, ys) q4x_mat(f, t, rows, cols, x, xs, n, y, ys, false)

const float *q4x_fv(const Q4xForward *f, const GgufTensor *t) { return (const float *)q4x_dev(f, t); }

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
   F32 (b.gate, hc n_embd a token) neither written nor read back - from the silu's f16 at x16 (NULL: the two calls,
   the up from b.lo). the same bits either way */
static bool up_mean(Q4xForward *f, const GgufTensor *up, uint32_t n, uint32_t hc, uint32_t D, uint32_t r,
                    const void *x16)
{
  Q4xBuf *b = &f->b;
  const uint32_t hd = hc * D;
  Q4xX16 sd;
  const Q4xX16 *ms = q4x_side(f, n, D, &sd);
  if (x16 && mm_takes(up, hd, r, r, hd) &&
      mm_run_gated_mean(TY(up), q4x_dev(f, up), hc, D, r, x16, n, b->xn, b->mixed, ms ? ms->x16 : NULL,
                        ms ? ms->row0 : 0, b->overflow, f->st))
    return true;
  return (x16 && mm_takes(up, hd, r, r, hd) ? mm_run(TY(up), q4x_dev(f, up), hd, r, x16, n, b->gate, hd, f->st)
                                             : q4x_mat(f, up, hd, r, b->lo, r, n, b->gate, hd, false)) &&
         q4x_hc_gated_mean(b->xn, b->gate, n, hc, D, b->mixed, ms, NULL);
}

/* the mixer's down, silu, then up_mean: a small step's down and silu one launch (mm_run_silu: its f16 into b.x16b -
   b.x16 the down's own input meanwhile), the same bits as the two; otherwise the two, the silu's f16 into b.xe16 -
   the routed experts' f16, theirs only after the mixer in the stream's order (a step past Q4X_SIDE_STEP is one
   stream's), so no memory more: a buffer of its own can tip a big step past the GPU's memory */
static bool down_up(Q4xForward *f, const GgufTensor *down, const GgufTensor *up, uint32_t n, uint32_t hc, uint32_t D,
                    uint32_t r, bool ready)
{
  Q4xBuf *b = &f->b;
  const uint32_t hd = hc * D;
  const float scale = 1.0f / (float)hc;
  if (ready && n <= MM_SILU_MAX_TOKENS && mm_takes(down, r, hd, hd, r) && mm_takes(up, hd, r, r, hd) &&
      mm_run_silu(TY(down), q4x_dev(f, down), r, hd, b->x16, n, b->lo, r, scale, b->x16b, b->overflow, f->st))
    return up_mean(f, up, n, hc, D, r, b->x16b);
  const Q4xX16 sx = {b->xe16, b->overflow, r, 0}; /* the silu's f16 for up_mean: b.x16 the mixer's input meanwhile */
  return q4x_mat(f, down, r, hd, b->xn, hd, n, b->lo, r, ready) &&
         q4x_scale_silu(b->lo, (uint64_t)n * r, scale, ready ? &sx : NULL, NULL) &&
         up_mean(f, up, n, hc, D, r, ready ? b->xe16 : NULL);
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
     inject is F32. a small step keeps the two, the inject beside on s2 (fused, a block a token would be the one
     block: slower) */
  if (n > Q4X_SIDE_STEP && inject && TY(inject) == QTYPE_F32 && inject->dims[0] == hd && inject->dims[1] == hc &&
      q4x_hc_norm_inject(b->h, b->out, b->inj, n, hc, D, 1.0f / (float)hc, q4x_fv(f, norm), g->rms_eps, b->xn,
                         q4x_side(f, n, hd, &sd), q4x_fv(f, inject), b->inj, combine, NULL))
    return down_up(f, down, up, n, hc, D, r, ready);
  return (combine ? q4x_hc_combine_norm(b->h, b->out, b->inj, n, hc, D, 1.0f / (float)hc, q4x_fv(f, norm), g->rms_eps,
                                        b->xn, q4x_side(f, n, hd, &sd), NULL)
                  : q4x_hc_norm(b->h, q4x_fv(f, norm), n, hc, D, g->rms_eps, b->xn, q4x_side(f, n, hd, &sd), NULL)) &&
         (!inject || inject_of(f, inject, n, hc, hd)) && down_up(f, down, up, n, hc, D, r, ready);
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
       q4x_ssm_gates(b->al, b->be, q4x_fv(f, L->dt), q4x_fv(f, L->a), n, vh, b->g, b->b, f->st);
  f->st = NULL;
  ok = ok && (!side_gates || hipEventRecord(f->e_gates, f->s2) == hipSuccess) &&
       q4x_mat2(f, L->qkv, ch, b->qkv, ch, L->gate, in, b->z, in, D, b->mixed, D, n, ready);
  for (uint32_t i = 0; ok && i < runs; i++) /* a sequence's conv history and state over its own tokens */
  {
    const Q4xSeq *s = r[i].s;
    const uint64_t t0 = r[i].t0;
    ok = q4x_conv(b->qkv + t0 * ch, r[i].n, ch, (uint32_t)g->ssm_conv, 1, q4x_fv(f, L->conv), s->conv[l], Q4X_SLOTS,
                  s->cur, b->cv + t0 * ch, NULL) &&
         (i || !side_gates || hipStreamWaitEvent(NULL, f->e_gates, 0) == hipSuccess) && /* the gates done */
         q4x_delta(b->cv + t0 * ch, b->g + t0 * vh, b->b + t0 * vh, r[i].n, (uint32_t)g->ssm_groups, vh,
                   (uint32_t)g->ssm_state, g->rms_eps, b->qkn, s->ssm[l], Q4X_SLOTS, s->cur, b->ro + t0 * in, NULL);
  }
  return ok &&
         q4x_gated_norm(b->ro, q4x_fv(f, L->ssm_norm), b->z, n, vh, (uint32_t)g->ssm_state, g->rms_eps, b->fo,
                        q4x_side(f, n, in, &sd), NULL) &&
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
                     &f->rope, q4x_fv(f, L->ix_k_norm), g->rms_eps, c[i].ixk, f->st);
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
    ok = q4x_ix_q(b->iqr + t0 * ixq, c[i].n, heads, &f->rope, q4x_fv(f, L->ix_q_norm), g->rms_eps, c[i].pos,
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
    ok = q4x_qk(b->qf + t0 * 2 * q, b->kr + t0 * kv, c[i].n, heads, kvh, hd, &f->rope, q4x_fv(f, L->q_norm),
                q4x_fv(f, L->k_norm), g->rms_eps, c[i].pos, b->q + t0 * q, b->gt + t0 * q,
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
  Q4xX16 sd;
  const Q4xX16 *all = q4x_side(f, n, q, &sd);
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
    const AttnStep st = {.q = b->q + t0 * q,
                         .n_tok = c[i].n,
                         .pos0 = (uint32_t)c[i].pos,
                         .keys = {.k = c[i].kc, .v = c[i].vc, .n_keys = (uint32_t)c[i].pos + c[i].n,
                                  .sel = b->sel + t0 * blocks, .npick = b->npick + t0, .stride = blocks},
                         .gate = b->gt + t0 * q,
                         .out = b->ao + t0 * q,
                         .x16 = all ? all->x16 : NULL,
                         .overflow = all ? all->overflow : NULL,
                         .x16_row0 = (uint32_t)t0};
    ok = attn_core_run(f->attn, &st, NULL);
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
