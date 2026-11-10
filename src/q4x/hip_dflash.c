#include <math.h>
#include <string.h>

#include "q4x_impl.h"

/* DFlash2's drafter on the GPU (osmos/q4x/forward.h, q4x_dflash_seed / q4x_dflash_draft): z-lab/dflash's
   DFlash2DraftModel - the context's taps through fc and a norm into each layer's keys and values; a block (the anchor
   and masks, Qwen's embedding) through its layers (a dynamic convolution before and after the attention and the MLP),
   a norm, Qwen's output head, the candidate selector's greedy path */

#define F sizeof(float)

static const float *f32(const Q4xForward *f, const GgufTensor *t) { return (const float *)q4x_dev(f, t); }

static uint32_t kv_of(const Q4xDflashGeometry *g) { return (uint32_t)(g->n_kv_heads * g->head_dim); }

/* a layer's keys (normed a head, rotated) and values of n rows of x at positions pos into d.k, d.v */
static bool keys(Q4xForward *f, const Q4xDflashLayer *L, const float *x, uint32_t n, const int32_t *pos)
{
  const Q4xDflashGeometry *g = &f->df->geo;
  Q4xDfBuf *d = &f->d;
  const uint32_t D = (uint32_t)g->n_embd, kv = kv_of(g), kh = (uint32_t)g->n_kv_heads;
  return q4x_mat2(f, L->k, kv, d->k, kv, L->v, kv, d->v, kv, D, x, D, n, false) &&
         ops_rms(d->k, Q4X_DF_HEAD, Q4X_DF_HEAD, f32(f, L->k_norm), g->eps, n * kh, d->k, Q4X_DF_HEAD, NULL) &&
         q4x_df_rope(d->k, n, kh, kv, pos, g->rope_base, NULL);
}

bool q4x_dflash_seed(Q4xForward *f, Q4xSeq *s, int32_t from, int32_t to, char err[Q4X_FWD_ERR])
{
  if (!f || !s || s->f != f || !f->df)
    return q4x_fail(err, "no forward with a drafter, or a sequence of another forward");
  if (from >= to || !s->step_n || from < s->step_pos0 || to > s->step_pos0 + (int32_t)s->step_n || to > s->pos)
    return q4x_fail(err, "positions not of the sequence's last step, or past its position");
  const int32_t p0 = from;
  const uint32_t n = (uint32_t)(to - from);
  const Q4xDflash *df = f->df;
  const Q4xDflashGeometry *g = &df->geo;
  Q4xDfBuf *d = &f->d;
  const uint32_t D = (uint32_t)g->n_embd, kv = kv_of(g), T = (uint32_t)g->n_taps * D;
  const float *taps = f->b.taps + (uint64_t)(s->step_t0 + (uint32_t)(p0 - s->step_pos0)) * T;
  const int32_t *pos = d->iota + p0;
  bool ok = q4x_mat(f, df->fc, D, T, taps, T, n, d->th, D, false) &&
            ops_rms(d->th, D, D, f32(f, df->hidden_norm), g->eps, n, d->th, D, NULL);
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = keys(f, &df->layer[l], d->th, n, pos) && q4x_df_ring(s->dk[l], f->df_ring, d->k, n, kv, p0, NULL) &&
         q4x_df_ring(s->dv[l], f->df_ring, d->v, n, kv, p0, NULL);
  if (!ok)
    return q4x_fail(err, "a GPU call failed (the drafter's context)");
  s->df_lo = from > s->df_pos ? from : from < s->df_lo ? from : s->df_lo; /* past a gap: the context starts over */
  s->df_pos = to;
  return true;
}

/* x convolved (its dynamic kernel's half `which`, its base kernel's) into y: a block's B tokens */
static bool conv(Q4xForward *f, const GgufTensor *base, const float *x, int which, float *y, uint32_t B)
{
  const Q4xDflashGeometry *g = &f->df->geo;
  const uint32_t D = (uint32_t)g->n_embd, groups = D / (uint32_t)g->conv_group, half = Q4X_DFLASH_CONV * groups;
  return q4x_df_conv(x, B, D, (uint32_t)g->conv_group, f32(f, base) + (uint64_t)which * Q4X_DFLASH_CONV * D,
                     f->d.dyn + (uint64_t)which * half, 2 * half, y, NULL);
}

/* a layer on the block's B tokens of d.h at positions pos0 .., the context's keys from ctx_from */
static bool block_layer(Q4xForward *f, const Q4xSeq *s, int l, uint32_t B, int32_t pos0, int32_t ctx_from)
{
  const Q4xDflashGeometry *g = &f->df->geo;
  const Q4xDflashLayer *L = &f->df->layer[l];
  Q4xDfBuf *d = &f->d;
  const uint32_t D = (uint32_t)g->n_embd, q = (uint32_t)(g->n_heads * g->head_dim), ff = (uint32_t)g->ff;
  const uint32_t dyn = 2 * Q4X_DFLASH_CONV * (D / (uint32_t)g->conv_group), H = (uint32_t)g->n_heads;
  const int32_t *pos = d->iota + pos0;
  return ops_rms(d->h, D, D, f32(f, L->attn_norm), g->eps, B, d->x, D, NULL) &&
         q4x_mat(f, L->attn_conv_proj, dyn, D, d->x, D, B, d->dyn, dyn, false) &&
         conv(f, L->attn_conv, d->x, 0, d->xc, B) &&
         q4x_mat(f, L->q, q, D, d->xc, D, B, d->q, q, false) &&
         ops_rms(d->q, Q4X_DF_HEAD, Q4X_DF_HEAD, f32(f, L->q_norm), g->eps, B * H, d->q, Q4X_DF_HEAD, NULL) &&
         q4x_df_rope(d->q, B, H, q, pos, g->rope_base, NULL) && keys(f, L, d->xc, B, pos) &&
         q4x_df_attn(d->q, B, pos0, d->k, d->v, s->dk[l], s->dv[l], f->df_ring, ctx_from, H, (uint32_t)g->n_kv_heads,
                     (uint32_t)g->window, d->att, NULL) &&
         q4x_mat(f, L->out, D, q, d->att, q, B, d->a, D, false) && conv(f, L->attn_conv, d->a, 1, d->x, B) &&
         ops_add(d->h, d->x, (uint64_t)B * D, NULL) &&
         ops_rms(d->h, D, D, f32(f, L->ffn_norm), g->eps, B, d->x, D, NULL) &&
         q4x_mat(f, L->ffn_conv_proj, dyn, D, d->x, D, B, d->dyn, dyn, false) &&
         conv(f, L->ffn_conv, d->x, 0, d->xc, B) &&
         q4x_mat2(f, L->gate, ff, d->gate, ff, L->up, ff, d->up, ff, D, d->xc, D, B, false) &&
         q4x_swiglu(d->gate, d->up, (uint64_t)B * ff, d->gate, NULL, NULL) &&
         q4x_mat(f, L->down, D, ff, d->gate, ff, B, d->a, D, false) && conv(f, L->ffn_conv, d->a, 1, d->x, B) &&
         ops_add(d->h, d->x, (uint64_t)B * D, NULL);
}

int q4x_dflash_draft(Q4xForward *f, Q4xSeq *s, int32_t y, uint32_t max, float conf_min, float cum_min, int32_t *drafts,
                     float *conf, char err[Q4X_FWD_ERR])
{
  if (!f || !s || s->f != f || !f->df || (max && (!drafts || !conf)))
    return q4x_fail(err, "no forward with a drafter, a sequence of another forward, or no room"), -1;
  const Q4xDflash *df = f->df;
  const Q4xDflashGeometry *g = &df->geo;
  if (y < 0 || y >= g->n_vocab)
    return q4x_fail(err, "a token not of the vocabulary"), -1;
  if (s->df_pos != s->pos)
    return q4x_fail(err, "the drafter's context not seeded to the sequence's position"), -1;
  const int32_t pos0 = s->pos, room = (int32_t)f->max_ctx - 1 - pos0; /* a draft's position below max_ctx */
  uint32_t n = (uint32_t)g->block - 1 < max ? (uint32_t)g->block - 1 : max;
  n = room <= 0 ? 0 : (uint32_t)room < n ? (uint32_t)room : n;
  if (!n)
    return 0;
  Q4xDfBuf *d = &f->d;
  const GgufTensor *e = f->w->embed, *out = f->w->output;
  const uint32_t B = (uint32_t)g->block, D = (uint32_t)g->n_embd, V = (uint32_t)g->n_vocab, R = (uint32_t)g->sel_rank;
  const uint32_t K = (uint32_t)g->sel_top_k;
  const int32_t lo = pos0 - (int32_t)f->df_ring, ctx_from = lo > s->df_lo ? lo : s->df_lo;
  d->pin_tok[0] = y; /* (the pinned row free: the draft before synchronized at its end) */
  bool ok = hipMemcpyAsync(d->tok, d->pin_tok, 4, hipMemcpyHostToDevice, NULL) == hipSuccess &&
            q4x_rows(e->type, q4x_dev(f, e), e->bytes / e->dims[1], D, d->tok, B, d->h, NULL);
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = block_layer(f, s, l, B, pos0, ctx_from);
  /* the block's places after the anchor: the norm, the head, the selector */
  ok = ok && ops_rms(d->h, D, D, f32(f, df->output_norm), g->eps, B, d->hn, D, NULL);
  if (ok)
    q4x_seen(f, "df_out", -1, d->hn, (uint64_t)B * D);
  ok = ok && q4x_mat(f, out, V, D, d->hn + D, D, B - 1, d->logits, V, false) &&
       q4x_df_topk(d->logits, B - 1, V, K, d->val, d->idx, NULL) &&
       q4x_mat(f, df->sel_proj, R, D, d->hn + D, D, B - 1, d->hp, R, false) &&
       q4x_df_select(d->val, d->idx, B - 1, K, d->hp, R, q4x_dev(f, df->sel_pred), q4x_dev(f, df->sel_succ), y,
                     d->path, NULL) &&
       q4x_top(d->logits, V, B - 1, V, d->top_id, d->top_p, NULL) &&
       hipMemcpyAsync(d->pin_path, d->path, (B - 1) * 4, hipMemcpyDeviceToHost, NULL) == hipSuccess &&
       hipMemcpyAsync(d->pin_idx, d->idx, (B - 1) * K * 4, hipMemcpyDeviceToHost, NULL) == hipSuccess &&
       hipMemcpyAsync(d->pin_val, d->val, (B - 1) * K * F, hipMemcpyDeviceToHost, NULL) == hipSuccess &&
       hipMemcpyAsync(d->pin_p, d->top_p, (B - 1) * F, hipMemcpyDeviceToHost, NULL) == hipSuccess &&
       q4x_overflow_read(f) && hipStreamSynchronize(NULL) == hipSuccess;
  if (!ok)
    return q4x_fail(err, "a GPU call failed (a draft)"), -1;
  if (!q4x_overflow_ok(f, err))
    return -1;
  /* a draft's confidence: the head's probability of it at its place - the greatest's (q4x_top) scaled by its logit
     below the greatest's (both the top-k's: the pick is one of its candidates, the greatest its first) */
  float cum = 1;
  uint32_t kept = 0;
  for (; kept < n; kept++)
  {
    const int32_t *ids = d->pin_idx + (uint64_t)kept * K;
    const float *val = d->pin_val + (uint64_t)kept * K;
    uint32_t c = 0;
    while (c < K && ids[c] != d->pin_path[kept])
      c++;
    if (c == K)
      return q4x_fail(err, "the selector's pick not one of its candidates"), -1;
    const float p = d->pin_p[kept] * expf(val[c] - val[0]);
    cum *= p;
    if (p < conf_min || cum < cum_min)
      break;
    drafts[kept] = d->pin_path[kept], conf[kept] = p;
  }
  return (int)kept;
}
