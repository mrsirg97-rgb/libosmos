/* DFlash2 on LFM3 (osmos/lfm3/forward.h: lfm3_dflash_seed, lfm3_dflash_draft): q4x's drafter (src/q4x/hip_dflash.c,
   its kernels q4x_df_*) on the LFM3 forward's taps, embedding and tied head - the context's taps through fc and a
   norm into each layer's keys and values; a block (the anchor and masks) through its layers, a norm, the head, the
   candidate selector's greedy path. every matrix the matvec's (a draft never sets a read's f16 overflow: the verify
   step's check its own). a round's drafts capped where its verify would leave the sequence's reach */
#include <math.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "lfm3_impl.h"

static const void *ddev(const Lfm3Forward *f, const GgufTensor *t) { return f->df_dev[t - f->df->model->tensors]; }
static const float *df32(const Lfm3Forward *f, const GgufTensor *t) { return (const float *)ddev(f, t); }

static bool dmat(Lfm3Forward *f, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t xs,
                 uint32_t n, float *y, uint32_t ys)
{
  return mv_strided(t->type, ddev(f, t), rows, cols, x, xs, n, y, ys, f->st);
}

static uint32_t kv_of(const Q4xDflashGeometry *g) { return (uint32_t)(g->n_kv_heads * g->head_dim); }

/* a layer's keys (normed a head, rotated) and values of n rows of x at positions pos into d.k, d.v */
static bool keys(Lfm3Forward *f, const Q4xDflashLayer *L, const float *x, uint32_t n, const int32_t *pos)
{
  const Q4xDflashGeometry *g = &f->df->geo;
  Lfm3DfBuf *d = &f->d;
  const uint32_t D = (uint32_t)g->n_embd, kv = kv_of(g), kh = (uint32_t)g->n_kv_heads;
  return dmat(f, L->k, kv, D, x, D, n, d->k, kv) && dmat(f, L->v, kv, D, x, D, n, d->v, kv) &&
         ops_rms(d->k, Q4X_DF_HEAD, Q4X_DF_HEAD, df32(f, L->k_norm), g->eps, n * kh, d->k, Q4X_DF_HEAD, f->st) &&
         q4x_df_rope(d->k, n, kh, kv, pos, g->rope_base, f->st);
}

bool lfm3_dflash_seed(Lfm3Forward *f, Lfm3Seq *s, int32_t from, int32_t to, char err[LFM3_ERR])
{
  if (!f || !s || s->f != f || !f->df)
    return lfm3_fail(err, "no forward with a drafter, or a sequence of another forward");
  if (s->broken)
    return lfm3_fail(err, "a sequence after a failed step: load it, or make another");
  if (s->step_n && s->step_id != f->steps)
    return lfm3_fail(err, "the drafter's taps another step's: seed right after the sequence's step");
  if (from >= to || !s->step_n || from < s->step_pos0 || to > s->step_pos0 + (int32_t)s->step_n || to > s->pos)
    return lfm3_fail(err, "positions not of the sequence's last step, or past its position");
  const Q4xDflash *df = f->df;
  const Q4xDflashGeometry *g = &df->geo;
  Lfm3DfBuf *d = &f->d;
  const uint32_t n = (uint32_t)(to - from), D = (uint32_t)g->n_embd, kv = kv_of(g), T = (uint32_t)g->n_taps * D;
  const float *taps = f->taps + (uint64_t)(s->step_t0 + (uint32_t)(from - s->step_pos0)) * T;
  const int32_t *pos = d->iota + from;
  bool ok = dmat(f, df->fc, D, T, taps, T, n, d->th, D) &&
            ops_rms(d->th, D, D, df32(f, df->hidden_norm), g->eps, n, d->th, D, f->st);
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = keys(f, &df->layer[l], d->th, n, pos) && q4x_df_ring(s->dk[l], f->df_ring, d->k, n, kv, from, f->st) &&
         q4x_df_ring(s->dv[l], f->df_ring, d->v, n, kv, from, f->st);
  if (!ok)
    return lfm3_fail(err, "a GPU call failed (the drafter's context)");
  s->df_lo = from > s->df_pos ? from : from < s->df_lo ? from : s->df_lo; /* past a gap: the context starts over */
  s->df_pos = to;
  return true;
}

/* x convolved (its dynamic kernel's half `which`, its base kernel's) into y: a block's B tokens */
static bool conv(Lfm3Forward *f, const GgufTensor *base, const float *x, int which, float *y, uint32_t B)
{
  const Q4xDflashGeometry *g = &f->df->geo;
  const uint32_t D = (uint32_t)g->n_embd, groups = D / (uint32_t)g->conv_group, half = Q4X_DFLASH_CONV * groups;
  return q4x_df_conv(x, B, D, (uint32_t)g->conv_group, df32(f, base) + (uint64_t)which * Q4X_DFLASH_CONV * D,
                     f->d.dyn + (uint64_t)which * half, 2 * half, y, f->st);
}

/* a layer on the block's B tokens of d.h at positions pos0 .., the context's keys from ctx_from */
static bool block_layer(Lfm3Forward *f, const Lfm3Seq *s, int l, uint32_t B, int32_t pos0, int32_t ctx_from)
{
  const Q4xDflashGeometry *g = &f->df->geo;
  const Q4xDflashLayer *L = &f->df->layer[l];
  Lfm3DfBuf *d = &f->d;
  const uint32_t D = (uint32_t)g->n_embd, q = (uint32_t)(g->n_heads * g->head_dim), ff = (uint32_t)g->ff;
  const uint32_t dyn = 2 * Q4X_DFLASH_CONV * (D / (uint32_t)g->conv_group), H = (uint32_t)g->n_heads;
  const int32_t *pos = d->iota + pos0;
  return ops_rms(d->h, D, D, df32(f, L->attn_norm), g->eps, B, d->x, D, f->st) &&
         dmat(f, L->attn_conv_proj, dyn, D, d->x, D, B, d->dyn, dyn) && conv(f, L->attn_conv, d->x, 0, d->xc, B) &&
         dmat(f, L->q, q, D, d->xc, D, B, d->q, q) &&
         ops_rms(d->q, Q4X_DF_HEAD, Q4X_DF_HEAD, df32(f, L->q_norm), g->eps, B * H, d->q, Q4X_DF_HEAD, f->st) &&
         q4x_df_rope(d->q, B, H, q, pos, g->rope_base, f->st) && keys(f, L, d->xc, B, pos) &&
         q4x_df_attn(d->q, B, pos0, d->k, d->v, s->dk[l], s->dv[l], f->df_ring, ctx_from, H, (uint32_t)g->n_kv_heads,
                     (uint32_t)g->window, d->att, f->st) &&
         dmat(f, L->out, D, q, d->att, q, B, d->a, D) && conv(f, L->attn_conv, d->a, 1, d->x, B) &&
         ops_add(d->h, d->x, (uint64_t)B * D, f->st) &&
         ops_rms(d->h, D, D, df32(f, L->ffn_norm), g->eps, B, d->x, D, f->st) &&
         dmat(f, L->ffn_conv_proj, dyn, D, d->x, D, B, d->dyn, dyn) && conv(f, L->ffn_conv, d->x, 0, d->xc, B) &&
         dmat(f, L->gate, ff, D, d->xc, D, B, d->gate, ff) && dmat(f, L->up, ff, D, d->xc, D, B, d->up, ff) &&
         q4x_swiglu(d->gate, d->up, (uint64_t)B * ff, d->gate, NULL, f->st) &&
         dmat(f, L->down, D, ff, d->gate, ff, B, d->a, D) && conv(f, L->ffn_conv, d->a, 1, d->x, B) &&
         ops_add(d->h, d->x, (uint64_t)B * D, f->st);
}

/* a round's drafts at most: its verify (y and the drafts, at pos ..) never closing an LSA block or crossing a YaRN
   stage - a rollback reaches neither (lfm3_seq_reach), so every draft the verify drops stays within it */
static uint32_t round_room(const Lfm3Forward *f, const Lfm3Seq *s)
{
  int64_t room = INT32_MAX;
  if (f->lsa) /* the verify's end below the next close: pos + drafts + 1 < its block's end */
  {
    const int64_t end = ((int64_t)s->pos / LSA_WINDOW + 1) * LSA_WINDOW;
    room = end - s->pos - 2;
  }
  if (!f->yarn_static_ctx && s->stage + 1 < f->stages) /* the verify's end at the next stage's from at most */
  {
    const int64_t r = (int64_t)f->stage[s->stage + 1].from - s->pos - 1;
    room = r < room ? r : room;
  }
  return room > 0 ? (uint32_t)room : 0;
}

int lfm3_dflash_draft(Lfm3Forward *f, Lfm3Seq *s, int32_t y, uint32_t max, float conf_min, float cum_min,
                      int32_t *drafts, float *conf, char err[LFM3_ERR])
{
  if (!f || !s || s->f != f || !f->df || (max && (!drafts || !conf)))
    return lfm3_fail(err, "no forward with a drafter, a sequence of another forward, or no room"), -1;
  if (s->broken)
    return lfm3_fail(err, "a sequence after a failed step: load it, or make another"), -1;
  const Q4xDflash *df = f->df;
  const Q4xDflashGeometry *g = &df->geo;
  if (y < 0 || y >= g->n_vocab)
    return lfm3_fail(err, "a token not of the vocabulary"), -1;
  if (s->df_pos != s->pos)
    return lfm3_fail(err, "the drafter's context not seeded to the sequence's position"), -1;
  const int32_t pos0 = s->pos, room = (int32_t)f->max_ctx - 1 - pos0; /* a draft's position below max_ctx */
  uint32_t n = (uint32_t)g->block - 1 < max ? (uint32_t)g->block - 1 : max;
  n = room <= 0 ? 0 : (uint32_t)room < n ? (uint32_t)room : n;
  const uint32_t reach = round_room(f, s);
  n = reach < n ? reach : n;
  if (!n)
    return 0;
  Lfm3DfBuf *d = &f->d;
  const Lfm3Weights *w = f->w;
  const uint32_t B = (uint32_t)g->block, D = (uint32_t)g->n_embd, V = (uint32_t)g->n_vocab, R = (uint32_t)g->sel_rank;
  const uint32_t K = (uint32_t)g->sel_top_k;
  const int32_t lo = pos0 - (int32_t)g->window, ctx_from = lo > s->df_lo ? lo : s->df_lo; /* (the ring past it) */
  d->pin_tok[0] = y; /* (the pinned row free: the draft before synchronized at its end) */
  bool ok = hipMemcpyAsync(d->tok, d->pin_tok, 4, hipMemcpyHostToDevice, (hipStream_t)f->st) == hipSuccess &&
            ops_embed(w->embd->type, lfm3_dev(f, w->embd), D, d->tok, B, 1, d->h, f->st);
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = block_layer(f, s, l, B, pos0, ctx_from);
  /* the block's places after the anchor: the norm, the head (the step's: LFM3's tied, Q8_0), the selector */
  ok = ok && ops_rms(d->h, D, D, df32(f, df->output_norm), g->eps, B, d->hn, D, f->st) &&
       mv_strided(f->head_type, f->head, V, D, d->hn + D, D, B - 1, d->logits, V, f->st) &&
       q4x_df_topk(d->logits, B - 1, V, K, d->val, d->idx, f->st) &&
       dmat(f, df->sel_proj, R, D, d->hn + D, D, B - 1, d->hp, R) &&
       q4x_df_select(d->val, d->idx, B - 1, K, d->hp, R, ddev(f, df->sel_pred), ddev(f, df->sel_succ), y, d->path,
                     f->st) &&
       q4x_top(d->logits, V, B - 1, V, d->top_id, d->top_p, f->st) &&
       hipMemcpyAsync(d->pin_path, d->path, (B - 1) * 4, hipMemcpyDeviceToHost, (hipStream_t)f->st) == hipSuccess &&
       hipMemcpyAsync(d->pin_idx, d->idx, (B - 1) * K * 4, hipMemcpyDeviceToHost, (hipStream_t)f->st) == hipSuccess &&
       hipMemcpyAsync(d->pin_val, d->val, (B - 1) * K * F, hipMemcpyDeviceToHost, (hipStream_t)f->st) == hipSuccess &&
       hipMemcpyAsync(d->pin_p, d->top_p, (B - 1) * F, hipMemcpyDeviceToHost, (hipStream_t)f->st) == hipSuccess &&
       hipStreamSynchronize((hipStream_t)f->st) == hipSuccess;
  if (!ok)
    return lfm3_fail(err, "a GPU call failed (a draft)"), -1;
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
      return lfm3_fail(err, "the selector's pick not one of its candidates"), -1;
    const float p = d->pin_p[kept] * expf(val[c] - val[0]);
    cum *= p;
    if (p < conf_min || cum < cum_min)
      break;
    drafts[kept] = d->pin_path[kept], conf[kept] = p;
  }
  return (int)kept;
}
