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
           (g->compress[l] < 2 || (fwd_alloc(&s->pend_kvc[l], (uint64_t)g->compress[l] * hd * F) &&
                                   fwd_alloc(&s->pend_sc[l], (uint64_t)g->compress[l] * hd * F)));
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

static bool layer(Forward *f, Seq *s, int l, const float *pre, int *nth, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t hc = g->hc, D = g->n_embd, nmix = (2 + hc) * hc;
  const int p = s->pos;
  if (g->engram[l] && !fwd_engram(f, l, (*nth)++, err))
    return false;
  fwd_probe(f, "h_in", l, p, b->h, hc * D, false);
  bool ok = mv_run(L->hc_attn_fn->type, fwd_dev(f, L->hc_attn_fn), nmix, hc * D, b->h, 1, b->mix, NULL) &&
            ops_hc_mix(b->h, b->mix, fv(f, L->hc_attn_scale), fv(f, L->hc_attn_base), hc, D,
                       g->rms_eps, g->hc_eps, g->sinkhorn_iters, 1, b->a_pre, b->a_post, b->a_comb, NULL) &&
            ops_hc_pre_rms(b->h, pre, hc, D, fv(f, L->attn_norm), g->rms_eps, 1, b->x, NULL);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the attention's mixes)");
  fwd_probe(f, "a_pre", l, p, b->a_pre, hc, false);
  fwd_probe(f, "a_post", l, p, b->a_post, hc, false);
  fwd_probe(f, "a_comb", l, p, b->a_comb, hc * hc, false);
  fwd_probe(f, "attn_x", l, p, b->x, D, false);
  if (!fwd_attention(f, s, l, err))
    return false;
  ok = ops_hc_post(b->y, b->h, b->a_post, b->a_comb, hc, D, 1, b->mid, NULL) &&
       mv_run(L->hc_ffn_fn->type, fwd_dev(f, L->hc_ffn_fn), nmix, hc * D, b->mid, 1, b->mix, NULL) &&
       ops_hc_mix(b->mid, b->mix, fv(f, L->hc_ffn_scale), fv(f, L->hc_ffn_base), hc, D,
                  g->rms_eps, g->hc_eps, g->sinkhorn_iters, 1, b->f_pre, b->f_post, b->f_comb, NULL) &&
       ops_hc_pre_rms(b->mid, b->a_pre, hc, D, fv(f, L->ffn_norm), g->rms_eps, 1, b->xf, NULL);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the FFN's mixes)");
  fwd_probe(f, "h_mid", l, p, b->mid, hc * D, false);
  fwd_probe(f, "f_pre", l, p, b->f_pre, hc, false);
  fwd_probe(f, "f_post", l, p, b->f_post, hc, false);
  fwd_probe(f, "f_comb", l, p, b->f_comb, hc * hc, false);
  fwd_probe(f, "ffn_x", l, p, b->xf, D, false);
  if (!fwd_moe(f, l, err))
    return false;
  if (!ops_hc_post(b->moe, b->mid, b->f_post, b->f_comb, hc, D, 1, b->h, NULL))
    return fwd_fail(err, "a GPU call failed (the FFN's hc_post)");
  fwd_probe(f, "h_out", l, p, b->h, hc * D, false);
  return true;
}

bool forward_step(Forward *f, Seq *s, int32_t token, char err[FORWARD_ERR])
{
  const Geometry *g = f->g;
  if (token < 0 || token >= g->n_vocab)
    return fwd_fail(err, "a token outside the vocabulary");
  if ((uint32_t)s->pos >= f->max_ctx)
    return fwd_fail(err, "the sequence is at max_ctx");
  const int ng = engram_ngram(f->hash);
  memmove(s->last + 1, s->last, (size_t)(ng - 1) * sizeof *s->last);
  s->last[0] = token;
  s->n_last = s->n_last < ng ? s->n_last + 1 : ng;
  if (!engram_rows(f->hash, s->last, s->n_last, f->rows))
    return fwd_fail(err, "a token the Engram hash refused");
  if (!fwd_put(f->b.ids, &token, 4) || !fwd_put(f->b.pos, &s->pos, 4) ||
      !ops_embed(f->w->embed->type, fwd_dev(f, f->w->embed), g->n_embd, f->b.ids, 1, g->hc, f->b.h, NULL))
    return fwd_fail(err, "a GPU call failed (the embedding)");
  int nth = 0;
  f->at = s->pos;
  for (int l = 0; l < f->n_layers; l++)
    if (!layer(f, s, l, l ? f->b.f_pre : f->b.pre0, &nth, err))
      return false;
  s->pos++;
  return true;
}
