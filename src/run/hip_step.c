#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "forward_impl.h"
#include "osmos/gpu/kvfmt.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"

Seq *seq_create(Forward *f, char err[FORWARD_ERR])
{
  Seq *s = f ? calloc(1, sizeof *s) : NULL;
  if (!s)
    return fwd_fail(err, "no forward, or out of memory"), NULL;
  s->f = f;
  const Geometry *g = f->g;
  const uint64_t F = sizeof(float), hd = (uint64_t)g->head_dim;
  bool ok = true;
  for (int l = 0; ok && l < f->n_layers; l++)
  {
    /* the window's keys, the compressed rows and the indexer's keys as the model rounds them (kvfmt.h) */
    ok = fwd_alloc(&s->ring[l], f->slots * kv_row_bytes(KV_FP8, (uint32_t)hd));
    if (ok && g->kv_source[l])
    {
      const uint64_t rows = f->max_ctx / (uint64_t)g->compress[l] + 1;
      const uint64_t pend = FWD_PEND_ROWS(f->step, g->compress[l]);
      ok = fwd_alloc(&s->ckv[l], rows * kv_row_bytes(KV_FP4_E4M3, (uint32_t)hd)) &&
           (!g->ix_source[l] || fwd_alloc(&s->ik[l], rows * kv_row_bytes(KV_FP4_POW2, (uint32_t)g->ix_head_dim))) &&
           (g->compress[l] < 2 ||
            (fwd_alloc(&s->pend_kvc[l], pend * hd * F) && fwd_alloc(&s->pend_sc[l], pend * hd * F)));
    }
  }
  if (!ok)
  {
    seq_destroy(s);
    return fwd_fail(err, "a GPU allocation refused"), NULL;
  }
  return s;
}

void seq_destroy(Seq *s)
{
  if (!s)
    return;
  for (int l = 0; l < WEIGHTS_MAX_LAYERS; l++)
  {
    hipFree(s->ring[l]);
    hipFree(s->ckv[l]);
    hipFree(s->ik[l]);
    hipFree(s->pend_kvc[l]);
    hipFree(s->pend_sc[l]);
  }
  hipFree(s->dring);
  free(s);
}

static const float *fv(const Forward *f, const GgufTensor *t) { return (const float *)fwd_dev(f, t); }

/* a layer up to its experts' input: Engram, the attention's mixes, the attention, its hc_post, the FFN's mixes - into
   the sub-batch's xf (its rows of the step's), mid, f_post, f_comb */
static bool layer_in(Forward *f, int l, const float *pre, int *nth, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t hc = g->hc, D = g->n_embd, n = f->n;
  const int p = f->at;
  if (f->taps && f->tapped[l] && f->cur + 1 == f->n_sub) /* a drafter's tap, the step's last rows */
  {
    if (!ops_hc_mean(b->h, hc, D, n, f->taps + (uint64_t)f->tap_of[l] * D, (uint32_t)f->n_tap * D, NULL))
      return fwd_fail(err, "a GPU call failed (the drafter's tap)");
    f->tap_n = n;
    memcpy(f->tap_pos, f->pin->pos, n * sizeof *f->tap_pos);
    memcpy(f->tap_bs, f->bs, n * sizeof *f->tap_bs);
  }
  if (g->engram[l] && !fwd_engram(f, l, (*nth)++, err))
    return false;
  fwd_probe(f, "h_in", l, p, b->h, n * hc * D, false);
  /* the mixes and the norm one launch (ops_hc_mix_pre_rms); a probe may stand the oracle's in for x: its x16 the
     attention's own then. the mixes' sums written already when the last layer's end did (layer_out) */
  const bool done = f->hcp_next;
  f->hcp_next = false;
  bool ok = (done || ops_hc_dot(fv(f, L->hc_attn_fn), b->h, hc, D, n, b->hcp, NULL)) &&
            ops_hc_mix_pre_rms(b->hcp, fv(f, L->hc_attn_scale), fv(f, L->hc_attn_base), hc, D, g->rms_eps, g->hc_eps,
                               g->sinkhorn_iters, n, b->a_pre, b->a_post, b->a_comb, b->h, pre, fv(f, L->attn_norm),
                               b->x, f->probe ? NULL : b->x16, f->probe ? NULL : b->overflow, NULL);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the attention's mixes)");
  fwd_probe(f, "a_pre", l, p, b->a_pre, n * hc, false);
  fwd_probe(f, "a_post", l, p, b->a_post, n * hc, false);
  fwd_probe(f, "a_comb", l, p, b->a_comb, n * hc * hc, false);
  fwd_probe(f, "attn_x", l, p, b->x, n * D, false);
  if (!fwd_attention(f, l, err))
    return false;
  /* the FFN's input's x16 beside it for the shared expert (shared(): its rows the step's when one sub-batch is all of
     them; not with a probe, which may stand the oracle's in) */
  f->xf_x16 = f->n_sub == 1 && !f->probe;
  ok = ops_hc_post_dot(b->y, b->h, b->a_post, b->a_comb, hc, D, n, b->mid, fv(f, L->hc_ffn_fn), b->hcp, NULL) &&
       ops_hc_mix_pre_rms(b->hcp, fv(f, L->hc_ffn_scale), fv(f, L->hc_ffn_base), hc, D, g->rms_eps, g->hc_eps,
                          g->sinkhorn_iters, n, b->f_pre, b->f_post, b->f_comb, b->mid, b->a_pre, fv(f, L->ffn_norm),
                          b->xf, f->xf_x16 ? b->x16 : NULL, f->xf_x16 ? b->overflow : NULL, NULL);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the FFN's mixes)");
  fwd_probe(f, "h_mid", l, p, b->mid, n * hc * D, false);
  fwd_probe(f, "f_pre", l, p, b->f_pre, n * hc, false);
  fwd_probe(f, "f_post", l, p, b->f_post, n * hc, false);
  fwd_probe(f, "f_comb", l, p, b->f_comb, n * hc * hc, false);
  fwd_probe(f, "ffn_x", l, p, b->xf, n * D, false);
  return true;
}

/* the layer's end, the experts' output (its rows of the step's moe) back into the streams - and with them the next
   layer's attention mixes' sums (ops_hc_post_dot: its layer_in's ops_hc_dot, one launch fewer) when nothing comes
   between: one sub-batch (b.hcp is the Forward's), no probe (it may stand h in), no Engram next (it changes h), not
   the decoder's compaction (it changes the tokens) */
static bool layer_out(Forward *f, int l, char *err)
{
  Scratch *b = &f->b;
  const uint32_t hc = f->g->hc, D = f->g->n_embd, n = f->n;
  f->hcp_next = f->n_sub == 1 && !f->probe && l + 1 < f->n_layers && !f->g->engram[l + 1] && l != f->dec;
  const bool ok = f->hcp_next ? ops_hc_post_dot(b->moe, b->mid, b->f_post, b->f_comb, hc, D, n, b->h,
                                                fv(f, f->w->layer[l + 1].hc_attn_fn), b->hcp, NULL)
                              : ops_hc_post(b->moe, b->mid, b->f_post, b->f_comb, hc, D, n, b->h, NULL);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the FFN's hc_post)");
  fwd_probe(f, "h_out", l, f->at, b->h, n * hc * D, false);
  return true;
}

/* a layer's routed and shared experts for every sub-batch at once (their xf rows one after another, xrow on): each
   expert read and decoded once for the layer's every token - a token's rows the same bits as alone (mm's) */
static bool layer_experts(Forward *f, int l, uint32_t rows, char *err)
{
  if (!rows)
    return true;
  f->n = rows, f->at = f->sub[0].at, f->b.xf = f->xf0, f->b.moe = f->moe0;
  return fwd_moe(f, l, err);
}

/* sub-batch k's state into the Forward (and its Scratch), to run its next layer; and back */
static void enter(Forward *f, uint32_t k)
{
  const Sub *s = &f->sub[k];
  Scratch *b = &f->b;
  f->cur = k;
  b->h = s->h, b->f_pre = s->f_pre, b->pos = s->pos, b->sel = s->sel, b->n_sel = s->n_sel, b->oa_cnt = s->oa_cnt;
  b->keep = s->keep, b->back = s->back, f->pin = s->pin, f->rows = s->rows, b->cand = s->cand;
  f->n = s->n, f->n_all = s->n_all, f->n_keep = s->n_keep, f->at = s->at;
  const uint64_t B = f->step, hc = f->g->hc, D = f->g->n_embd;
  b->mid = f->mid0 + k * B * hc * D, b->f_post = f->f_post0 + k * B * hc, b->f_comb = f->f_comb0 + k * B * hc * hc;
  b->xf = f->xf0 + (uint64_t)s->xrow * D, b->moe = f->moe0 + (uint64_t)s->xrow * D;
  memcpy(f->bs, s->bs, s->n_all * sizeof *f->bs);
  memcpy(f->row, s->row, s->n_all * sizeof *f->row);
  memcpy(f->want, s->want, s->n_all * sizeof *f->want);
}

static void leave(Forward *f, uint32_t k)
{
  Sub *s = &f->sub[k];
  s->n = f->n, s->n_all = f->n_all, s->n_keep = f->n_keep, s->at = f->at;
  memcpy(s->bs, f->bs, f->n_all * sizeof *f->bs);
  memcpy(s->row, f->row, f->n_all * sizeof *f->row);
  memcpy(s->want, f->want, f->n_all * sizeof *f->want);
}

/* the batch's tokens checked and placed: each at its sequence's next free position (a sequence's tokens in the
   batch's order), its last tokens for Engram's hash updated, its rows hashed - token i to sub-batch i / FWD_MAX_BATCH.
   nothing of a sequence changes before every token has passed */
static bool place(Forward *f, Seq *const *seqs, const int32_t *tokens, uint32_t n, char *err)
{
  const Geometry *g = f->g;
  const int ng = engram_ngram(f->hash);
  for (uint32_t i = 0; i < n; i++)
  {
    if (!seqs[i] || seqs[i]->f != f)
      return fwd_fail(err, "a token of no sequence, or another forward's");
    if (tokens[i] < 0 || tokens[i] >= g->n_vocab)
      return fwd_fail(err, "a token outside the vocabulary");
    seqs[i]->staged = 0;
  }
  for (uint32_t i = 0; i < n; i++)
    if ((uint32_t)(seqs[i]->pos + seqs[i]->staged++) >= f->max_ctx)
      return fwd_fail(err, "a sequence past max_ctx");
  for (uint32_t i = 0; i < n; i++)
    seqs[i]->staged = 0;
  f->n_sub = (n + FWD_MAX_BATCH - 1) / FWD_MAX_BATCH;
  for (uint32_t i = 0; i < n; i++)
  {
    Sub *b = &f->sub[i / FWD_MAX_BATCH];
    const uint32_t j = i % FWD_MAX_BATCH;
    Seq *s = seqs[i];
    b->bs[j] = s, b->pin->tok[j] = tokens[i], b->pin->pos[j] = s->pos + s->staged++;
    b->n = b->n_all = j + 1, b->at = b->pin->pos[0];
    const int cap = ENGRAM_MAX_NGRAM + SEQ_MAX_BACK;
    memmove(s->last + 1, s->last, (size_t)(cap - 1) * sizeof *s->last);
    s->last[0] = tokens[i];
    s->n_last = s->n_last < cap ? s->n_last + 1 : cap;
    if (!engram_rows(f->hash, s->last, s->n_last < ng ? s->n_last : ng,
                     b->rows + (uint64_t)j * WEIGHTS_MAX_LAYERS * g->engram_cols))
      return fwd_fail(err, "a token the Engram hash refused");
  }
  return true;
}

/* a sub-batch's start: its plan (hip_skip.c), its tokens and positions to the GPU, its streams embedded */
static bool begin(Forward *f, char *err)
{
  const Geometry *g = f->g;
  fwd_plan(f);
  for (int k = 0; k < g->o_groups; k++)
    f->pin->oa_cnt[k] = (int32_t)f->n;
  if (!fwd_put_async(f->b.ids, f->pin->tok, f->n * 4) ||
      !fwd_put_async(f->b.oa_cnt, f->pin->oa_cnt, (uint64_t)g->o_groups * 4) ||
      !fwd_put_async(f->b.pos, f->pin->pos, f->n * 4) ||
      !ops_embed(f->w->embed->type, fwd_dev(f, f->w->embed), g->n_embd, f->b.ids, f->n, g->hc, f->b.h, NULL))
    return fwd_fail(err, "a GPU call failed (the embedding)");
  return true;
}

bool forward_batch(Forward *f, Seq *const *seqs, const int32_t *tokens, uint32_t n, char err[FORWARD_ERR])
{
  const Geometry *g = f->g;
  if (!n || n > f->max_tokens)
    return fwd_fail(err, "a batch of no tokens, or more than the forward's max_tokens");
  f->vocab_rows = 0; /* (its head's, when it runs one) */
  f->step_n = n;
  if (!place(f, seqs, tokens, n, err) || !fwd_engram_issue(f, err))
    return false;
  if (hipMemsetAsync(f->b.overflow, 0, 4, NULL) != hipSuccess)
    return fwd_fail(err, "a GPU call failed (the step)");
  f->hcp_next = false;
  bool ok = true;
  for (uint32_t k = 0; ok && k < f->n_sub; k++)
  {
    enter(f, k);
    ok = begin(f, err);
    leave(f, k);
  }
  /* layer by layer, every sub-batch through it before the next: each up to the experts' input, its rows one after
     another; the experts once for them all; then each sub-batch's end of the layer. (a sub-batch's attention reads
     the earlier ones' keys of the layer, which its experts do not touch) */
  for (int l = 0, nth = 0; ok && l < f->n_layers; nth += g->engram[l], l++)
  {
    uint32_t rows = 0;
    for (uint32_t k = 0; ok && k < f->n_sub; k++)
    {
      f->sub[k].xrow = rows;
      enter(f, k);
      int at_nth = nth;
      if (f->n) /* past dec, the kept tokens: maybe none */
        ok = layer_in(f, l, l ? f->b.f_pre : f->b.pre0, &at_nth, err);
      rows += f->n;
      leave(f, k);
    }
    ok = ok && layer_experts(f, l, rows, err);
    for (uint32_t k = 0; ok && k < f->n_sub; k++)
    {
      enter(f, k);
      if (f->n)
        ok = layer_out(f, l, err) && (l != f->dec || fwd_compact(f, err));
      leave(f, k);
    }
  }
  /* the head: the last block's pre mixes the streams, normed, then the vocabulary (the wanted tokens'); the last
     sub-batch's rows the ones left */
  for (uint32_t k = 0; ok && k < f->n_sub; k++)
  {
    enter(f, k);
    ok = fwd_head(f, err);
    leave(f, k);
  }
  if (!ok)
    return false;
  /* the step's one wait: the last layer's experts are read, the pinned buffers free for the next step */
  if (hipMemcpyAsync(&f->pin->overflow, f->b.overflow, 4, hipMemcpyDeviceToHost, NULL) != hipSuccess ||
      !fwd_settle(f))
    return fwd_fail(err, "a GPU call failed (the step)");
  if (f->cpu && !cpu_lane_wait(f->cpu))
    return fwd_fail(err, "the CPU lane's experts failed");
  fwd_release_held(f);
  fwd_heat(f);
  if (f->pin->overflow)
    return fwd_fail(err, "an activation past f16's range: the step's results are not to be trusted");
  for (uint32_t i = 0; i < n; i++)
    seqs[i]->pos++;
  return true;
}

bool seq_prompt(Seq *s, uint32_t end)
{
  if (!s || end > s->f->max_ctx || (int64_t)end < s->prompt_end)
    return false;
  s->prompt_end = (int32_t)end;
  return true;
}

bool fwd_taps_on(Forward *f, const bool *at)
{
  if (f->taps)
    return memcmp(f->tapped, at, (size_t)f->n_layers * sizeof *at) == 0;
  f->n_tap = 0;
  for (int l = 0; l < f->n_layers; l++)
    f->tapped[l] = at[l], f->tap_of[l] = at[l] ? f->n_tap++ : -1;
  return f->n_tap && fwd_alloc(&f->taps, (uint64_t)f->step * f->n_tap * f->g->n_embd * sizeof(float));
}

uint32_t seq_reach(const Seq *s)
{
  if (!s)
    return 0;
  const int64_t past = (int64_t)s->pos - (int64_t)s->prompt_end;
  int64_t r = SEQ_MAX_BACK < s->n_last ? SEQ_MAX_BACK : s->n_last;
  r = past < r ? past : r;
  return r > 0 ? (uint32_t)r : 0;
}

bool seq_rollback(Seq *s, uint32_t n)
{
  if (!s || n > SEQ_MAX_BACK || (int)n > s->n_last || (int64_t)s->pos - n < (int64_t)s->prompt_end)
    return false;
  const int cap = ENGRAM_MAX_NGRAM + SEQ_MAX_BACK;
  memmove(s->last, s->last + n, (size_t)(cap - (int)n) * sizeof *s->last);
  s->n_last -= (int)n, s->pos -= (int32_t)n;
  return true;
}

bool forward_step(Forward *f, Seq *s, int32_t token, char err[FORWARD_ERR])
{
  return forward_batch(f, &s, &token, 1, err);
}
