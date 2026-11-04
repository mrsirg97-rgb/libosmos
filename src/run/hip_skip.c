#include <hip/hip_runtime_api.h>

#include "forward_impl.h"
#include "substrate/gpu/ops.h"

/* the decoder skip. DeepSeek-V4.1's layers past the last KV source (dec: the decoder's first layer) take their
   compressed KV from dec alone and keep nothing of a token but its window; so a prompt token's decoder state past dec
   is read only through the windows - the last token's needs (layers past dec) (window - 1) positions before it
   (each layer a window more). a sequence told its prompt's end (seq_prompt) runs layers past dec only for its tokens
   within that reach of the end; dec itself runs for every token (its compressed rows and window are every position's).
   the tokens kept have every input they read - the windows past dec floored at dec_from, below which nothing kept
   reads - so their bits, and every token's after them, are the bits of the whole run */

void fwd_decoder(Forward *f)
{
  const Geometry *g = f->g;
  f->dec = -1, f->reach = 0;
  int s = -1;
  for (int l = 0; l < f->n_layers; l++)
    if (g->kv_source[l])
      s = l;
  if (s < 0 || s == f->n_layers - 1)
    return;
  for (int l = s + 1; l < f->n_layers; l++) /* past s: no state of a token's own but its window */
    if (g->kv_source[l] || g->engram[l] || (g->compress[l] && g->kv_from[l] != s))
      return;
  f->dec = s;
  f->reach = (uint32_t)(f->n_layers - 1 - s) * (uint32_t)(g->window - 1) + 1;
}

/* the step's plan: each token kept past dec or not, its logits wanted or not; a sequence's tokens ascend in the step,
   so its kept ones, and its wanted ones, are its last */
void fwd_plan(Forward *f)
{
  uint32_t k = 0;
  for (uint32_t i = 0; i < f->n; i++)
  {
    Seq *s = f->bs[i];
    const int64_t p = f->pin->pos[i], end = s->prompt_end;
    const bool keep = f->dec < 0 || !end || p + f->reach >= end;
    f->want[i] = !end || p + 1 >= end;
    if (keep)
      f->row[k++] = (int32_t)i;
    else
      s->dec_from = (int32_t)p + 1;
  }
  f->n_all = f->n;
  f->n_keep = k;
}

/* past dec: the kept tokens only, their streams and the last layer's pre gathered to the step's first rows */
bool fwd_compact(Forward *f, char *err)
{
  const uint32_t nk = f->n_keep, hc = f->g->hc, D = f->g->n_embd;
  if (nk == f->n)
    return true;
  f->n = nk;
  if (!nk) /* the whole step before the reach: done at dec */
    return true;
  Scratch *b = &f->b;
  for (uint32_t k = 0; k < nk; k++)
  {
    const int32_t i = f->row[k]; /* i >= k: in order, nothing read is overwritten first */
    f->pin->keep[k] = i, f->bs[k] = f->bs[i], f->pin->pos[k] = f->pin->pos[i], f->want[k] = f->want[i];
  }
  for (int k = 0; k < f->g->o_groups; k++)
    f->pin->oa_cnt[k] = (int32_t)nk;
  f->at = f->pin->pos[0];
  const bool ok = fwd_put_async(b->keep, f->pin->keep, (uint64_t)nk * 4) &&
                  fwd_put_async(b->pos, f->pin->pos, (uint64_t)nk * 4) &&
                  fwd_put_async(b->oa_cnt, f->pin->oa_cnt, (uint64_t)f->g->o_groups * 4) &&
                  ops_gather_rows(b->mid, b->h, b->keep, nk, hc * D, NULL) &&
                  fwd_copy(b->h, b->mid, (uint64_t)nk * hc * D * sizeof(float)) &&
                  ops_gather_rows(b->a_pre, b->f_pre, b->keep, nk, hc, NULL) &&
                  fwd_copy(b->f_pre, b->a_pre, (uint64_t)nk * hc * sizeof(float)) &&
                  /* the selected rows dec chose, which the layers after reuse (CSA2's Reuse): copies, the bits */
                  ops_gather_rows(b->score, (const float *)b->sel, b->keep, nk, OPS_MAX_SELECT, NULL) &&
                  fwd_copy(b->sel, b->score, (uint64_t)nk * OPS_MAX_SELECT * 4) &&
                  ops_gather_rows((float *)b->rpos, (const float *)b->n_sel, b->keep, nk, 1, NULL) &&
                  fwd_copy(b->n_sel, b->rpos, (uint64_t)nk * 4) &&
                  /* the candidate blocks' bits (built at dec's index layer, read by the index layers after): the
                     kept tokens' rows, the bits copied */
                  (!f->sub[f->cur].cand_on ||
                   (ops_gather_rows(b->score, (const float *)b->cand, b->keep, nk, f->cand_words, NULL) &&
                    fwd_copy(b->cand, b->score, (uint64_t)nk * f->cand_words * 4)));
  return ok || fwd_fail(err, "a GPU call failed (the decoder's compaction)");
}

static const float *fv(const Forward *f, const GgufTensor *t) { return (const float *)fwd_dev(f, t); }

/* the head for the wanted tokens - each run of them whose step rows follow one another one product, into their
   rows - then the streams back at their rows (a token not kept: its row unspecified) */
bool fwd_head(Forward *f, char *err)
{
  const Geometry *g = f->g;
  Scratch *b = &f->b;
  const uint32_t n = f->n, hc = g->hc, D = g->n_embd, V = (uint32_t)g->n_vocab;
  bool ok = true;
  if (!n) /* no token past dec: no head, no stream */
  {
    f->n = f->n_all, f->vocab_rows = 0;
    return true;
  }
  if (f->n_layers == g->n_layers)
  {
    ok = ops_hc_pre_rms(b->h, b->f_pre, hc, D, fv(f, f->w->output_norm), g->rms_eps, n, b->x, NULL);
    for (uint32_t k = 0; ok && k < n;)
    {
      if (!f->want[k])
      {
        k++;
        continue;
      }
      uint32_t e = k + 1;
      while (e < n && f->want[e] && f->row[e] == f->row[e - 1] + 1)
        e++;
      ok = fwd_mat(f, f->w->output, fwd_dev(f, f->w->output), V, D, b->x + (uint64_t)k * D, D, e - k,
                   b->vocab + (uint64_t)f->row[k] * V, V);
      k = e;
    }
    if (!ok)
      return fwd_fail(err, "a GPU call failed (the head)");
    fwd_probe(f, "logits", -1, f->pin->pos[0], b->vocab, (uint64_t)f->n_all * V, false);
    f->vocab_rows = f->n_all;
  }
  if (n == f->n_all)
    return true;
  for (uint32_t i = 0; i < f->n_all; i++)
    f->pin->back[i] = 0;
  for (uint32_t k = 0; k < n; k++)
    f->pin->back[f->row[k]] = (int32_t)k;
  ok = fwd_put_async(b->back, f->pin->back, (uint64_t)f->n_all * 4) &&
       ops_gather_rows(b->mid, b->h, b->back, f->n_all, hc * D, NULL) &&
       fwd_copy(b->h, b->mid, (uint64_t)f->n_all * hc * D * sizeof(float));
  f->n = f->n_all;
  return ok || fwd_fail(err, "a GPU call failed (the stream's rows)");
}
