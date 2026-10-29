#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "forward_impl.h"
#include "substrate/gpu/mv.h"
#include "substrate/gpu/ops.h"

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
    ok = fwd_alloc(&s->ring[l], f->slots * hd * F);
    if (ok && g->kv_source[l])
    {
      const uint64_t rows = f->max_ctx / (uint64_t)g->compress[l] + 1;
      ok = fwd_alloc(&s->ckv[l], rows * hd * F) &&
           (!g->ix_source[l] || fwd_alloc(&s->ik[l], rows * (uint64_t)g->ix_head_dim * F)) &&
           (g->compress[l] < 2 || (fwd_alloc(&s->pend_kvc[l], (uint64_t)FWD_PEND_ROWS(g->compress[l]) * hd * F) &&
                                   fwd_alloc(&s->pend_sc[l], (uint64_t)FWD_PEND_ROWS(g->compress[l]) * hd * F)));
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
  free(s);
}

static const float *fv(const Forward *f, const GgufTensor *t) { return (const float *)fwd_dev(f, t); }

static bool layer(Forward *f, int l, const float *pre, int *nth, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t hc = g->hc, D = g->n_embd, n = f->n;
  const int p = f->at;
  if (g->engram[l] && !fwd_engram(f, l, (*nth)++, err))
    return false;
  fwd_probe(f, "h_in", l, p, b->h, n * hc * D, false);
  bool ok = ops_hc_dot(fv(f, L->hc_attn_fn), b->h, hc, D, n, b->hcp, NULL) &&
            ops_hc_mix(b->hcp, fv(f, L->hc_attn_scale), fv(f, L->hc_attn_base), hc, D,
                       g->rms_eps, g->hc_eps, g->sinkhorn_iters, n, b->a_pre, b->a_post, b->a_comb, NULL) &&
            ops_hc_pre_rms(b->h, pre, hc, D, fv(f, L->attn_norm), g->rms_eps, n, b->x, NULL);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the attention's mixes)");
  fwd_probe(f, "a_pre", l, p, b->a_pre, n * hc, false);
  fwd_probe(f, "a_post", l, p, b->a_post, n * hc, false);
  fwd_probe(f, "a_comb", l, p, b->a_comb, n * hc * hc, false);
  fwd_probe(f, "attn_x", l, p, b->x, n * D, false);
  if (!fwd_attention(f, l, err))
    return false;
  ok = ops_hc_post(b->y, b->h, b->a_post, b->a_comb, hc, D, n, b->mid, NULL) &&
       ops_hc_dot(fv(f, L->hc_ffn_fn), b->mid, hc, D, n, b->hcp, NULL) &&
       ops_hc_mix(b->hcp, fv(f, L->hc_ffn_scale), fv(f, L->hc_ffn_base), hc, D,
                  g->rms_eps, g->hc_eps, g->sinkhorn_iters, n, b->f_pre, b->f_post, b->f_comb, NULL) &&
       ops_hc_pre_rms(b->mid, b->a_pre, hc, D, fv(f, L->ffn_norm), g->rms_eps, n, b->xf, NULL);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the FFN's mixes)");
  fwd_probe(f, "h_mid", l, p, b->mid, n * hc * D, false);
  fwd_probe(f, "f_pre", l, p, b->f_pre, n * hc, false);
  fwd_probe(f, "f_post", l, p, b->f_post, n * hc, false);
  fwd_probe(f, "f_comb", l, p, b->f_comb, n * hc * hc, false);
  fwd_probe(f, "ffn_x", l, p, b->xf, n * D, false);
  if (!fwd_moe(f, l, err))
    return false;
  if (!ops_hc_post(b->moe, b->mid, b->f_post, b->f_comb, hc, D, n, b->h, NULL))
    return fwd_fail(err, "a GPU call failed (the FFN's hc_post)");
  fwd_probe(f, "h_out", l, p, b->h, n * hc * D, false);
  return true;
}

/* the batch's tokens checked and placed: each at its sequence's next free position (a sequence's tokens in the
   batch's order), its last tokens for Engram's hash updated, its rows hashed. nothing of a sequence changes before
   every token has passed */
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
    int32_t at = seqs[i]->pos;
    for (uint32_t j = 0; j < i; j++)
      at += seqs[j] == seqs[i];
    if ((uint32_t)at >= f->max_ctx)
      return fwd_fail(err, "a sequence past max_ctx");
    f->bs[i] = seqs[i], f->pin->tok[i] = tokens[i], f->pin->pos[i] = at;
  }
  for (uint32_t i = 0; i < n; i++)
  {
    Seq *s = seqs[i];
    memmove(s->last + 1, s->last, (size_t)(ng - 1) * sizeof *s->last);
    s->last[0] = tokens[i];
    s->n_last = s->n_last < ng ? s->n_last + 1 : ng;
    if (!engram_rows(f->hash, s->last, s->n_last, f->rows + (uint64_t)i * WEIGHTS_MAX_LAYERS * g->engram_cols))
      return fwd_fail(err, "a token the Engram hash refused");
  }
  f->n = n;
  f->at = f->pin->pos[0];
  return true;
}

bool forward_batch(Forward *f, Seq *const *seqs, const int32_t *tokens, uint32_t n, char err[FORWARD_ERR])
{
  const Geometry *g = f->g;
  if (!n || n > FWD_MAX_BATCH)
    return fwd_fail(err, "a batch of no tokens, or more than FWD_MAX_BATCH");
  if (!place(f, seqs, tokens, n, err))
    return false;
  fwd_plan(f);
  for (int k = 0; k < g->o_groups; k++)
    f->pin->oa_cnt[k] = (int32_t)n;
  if (hipMemsetAsync(f->b.overflow, 0, 4, NULL) != hipSuccess || !fwd_put_async(f->b.ids, f->pin->tok, n * 4) ||
      !fwd_put_async(f->b.oa_cnt, f->pin->oa_cnt, (uint64_t)g->o_groups * 4) ||
      !fwd_put_async(f->b.pos, f->pin->pos, n * 4) ||
      !ops_embed(f->w->embed->type, fwd_dev(f, f->w->embed), g->n_embd, f->b.ids, n, g->hc, f->b.h, NULL))
    return fwd_fail(err, "a GPU call failed (the embedding)");
  int nth = 0;
  for (int l = 0; l < f->n_layers && f->n; l++) /* past dec, the kept tokens: maybe none */
    if (!layer(f, l, l ? f->b.f_pre : f->b.pre0, &nth, err) || (l == f->dec && !fwd_compact(f, err)))
      return false;
  /* the head: the last block's pre mixes the streams, normed, then the vocabulary (the wanted tokens') */
  if (!fwd_head(f, err))
    return false;
  /* the step's one wait: the last layer's experts are read, the pinned buffers free for the next step */
  if (hipMemcpyAsync(&f->pin->overflow, f->b.overflow, 4, hipMemcpyDeviceToHost, NULL) != hipSuccess ||
      hipDeviceSynchronize() != hipSuccess)
    return fwd_fail(err, "a GPU call failed (the step)");
  fwd_release_held(f);
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

bool forward_step(Forward *f, Seq *s, int32_t token, char err[FORWARD_ERR])
{
  return forward_batch(f, &s, &token, 1, err);
}
