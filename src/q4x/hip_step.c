#include <string.h>

#include "osmos/q4x/ple.h"
#include "q4x_impl.h"

/* a step: the tokens' embedding into hc streams, their n-gram rows, the layers, the end mixer, the logits */

static bool embed(Q4xForward *f, Q4xSeq *s, const int32_t *tokens, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  const GgufTensor *e = f->w->embed;
  Q4xBuf *b = &f->b;
  int32_t prev[Q4X_MAX_NGRAM];
  for (uint32_t t = 0; t < n; t++) /* a token's predecessors: the step's before it, then the sequence's last */
  {
    int np = 0;
    for (int k = 1; k < g->ple_ngram; k++)
    {
      const int64_t at = (int64_t)t - k;
      if (at >= 0)
        prev[np++] = tokens[at];
      else if (-at - 1 < s->n_last)
        prev[np++] = s->last[-at - 1];
    }
    q4x_ple_rows(g, tokens[t], prev, np, &b->pin_rows[(uint64_t)t * g->ple_heads]);
    b->pin_tok[t] = tokens[t];
  }
  return hipMemcpyAsync(b->tok, b->pin_tok, n * 4, hipMemcpyHostToDevice, NULL) == hipSuccess &&
         hipMemcpyAsync(b->ple_rows, b->pin_rows, (uint64_t)n * g->ple_heads * 4, hipMemcpyHostToDevice, NULL) ==
             hipSuccess &&
         q4x_rows(e->type, q4x_dev(f, e), e->bytes / e->dims[1], (uint32_t)e->dims[0], b->tok, n, b->x, NULL) &&
         q4x_streams(b->x, n, (uint32_t)g->hc, (uint32_t)g->n_embd, b->h, NULL);
}

static bool layer(Q4xForward *f, Q4xSeq *s, int l, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xLayer *L = &f->w->layer[l];
  Q4xBuf *b = &f->b;
  const uint32_t hc = (uint32_t)g->hc, D = (uint32_t)g->n_embd;
  const float inv = 1.0f / (float)hc; /* the inject's scale (llama.cpp's 1 / hc before the sigmoid) */
  return (l != g->ple_layer || q4x_ple(f, s, l, n)) &&
         q4x_mix(f, L->hc_attn_norm, L->hc_attn_down, L->hc_attn_up, L->hc_attn_inject, n) &&
         (g->recurrent[l] ? q4x_deltanet(f, s, l, n) : q4x_attention(f, L, s->kc[l], s->vc[l], s->pos, n)) &&
         q4x_hc_combine(b->h, b->out, b->inj, n, hc, D, inv, NULL) &&
         q4x_mix(f, L->hc_ffn_norm, L->hc_ffn_down, L->hc_ffn_up, L->hc_ffn_inject, n) && q4x_moe(f, L, n) &&
         q4x_hc_combine(b->h, b->out, b->inj, n, hc, D, inv, NULL);
}

bool q4x_batch(Q4xForward *f, Q4xSeq *s, const int32_t *tokens, uint32_t n, uint32_t row0, char err[Q4X_FWD_ERR])
{
  if (!f || !s || s->f != f || !tokens)
    return q4x_fail(err, "no forward, sequence or tokens, or a sequence of another forward");
  if (!n || (uint64_t)row0 + n > f->max_tokens || (uint64_t)s->pos + n > f->max_ctx)
    return q4x_fail(err, "a step of no tokens, its rows past the forward's max_tokens, or past its context");
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint64_t hd = (uint64_t)g->hc * g->n_embd;
  if (!embed(f, s, tokens, n))
    return q4x_fail(err, "a GPU call failed (the embedding)");
  for (int l = 0; l < g->n_layers; l++)
  {
    if (!layer(f, s, l, n))
      return q4x_fail(err, "a GPU call failed (a layer)");
    q4x_seen(f, "l_last", l, b->h, n * hd);
  }
  const Q4xWeights *w = f->w;
  if (f->mtp) /* the head's h: the residual before the end mixer, at the step's rows; its tokens and place */
  {
    if (hipMemcpyAsync(f->m.hid + (uint64_t)row0 * hd, b->h, (uint64_t)n * hd * sizeof(float),
                       hipMemcpyDeviceToDevice, NULL) != hipSuccess)
      return q4x_fail(err, "a GPU call failed (the MTP head's rows)");
    memcpy(s->step_tok, tokens, n * sizeof *tokens);
    s->step_pos0 = s->pos, s->step_row0 = row0, s->step_n = n;
  }
  if (!q4x_mix(f, w->out_hc_norm, w->out_hc_down, w->out_hc_up, NULL, n))
    return q4x_fail(err, "a GPU call failed (the end mixer)");
  q4x_seen(f, "result_norm", -1, b->mixed, (uint64_t)n * g->n_embd);
  if (!mv_strided(w->output->type, q4x_dev(f, w->output), (uint32_t)g->n_vocab, (uint32_t)g->n_embd, b->mixed,
                  (uint32_t)g->n_embd, n, b->logits + (uint64_t)row0 * g->n_vocab, (uint32_t)g->n_vocab, NULL) ||
      hipStreamSynchronize(NULL) != hipSuccess)
    return q4x_fail(err, "a GPU call failed (the logits)");
  const int keep = (int)(sizeof s->last / sizeof *s->last);
  for (uint32_t t = 0; t < n; t++) /* the latest first */
  {
    memmove(s->last + 1, s->last, (size_t)(keep - 1) * sizeof *s->last);
    s->last[0] = tokens[t], s->n_last += s->n_last < keep;
  }
  s->pos += (int32_t)n;
  s->cur = (s->cur + n) % Q4X_SLOTS; /* the rings' latest: the step's last token's (q4x_ops.h) */
  s->back = s->back + n < Q4X_MAX_BACK ? s->back + n : Q4X_MAX_BACK;
  return true;
}

const float *q4x_logits(const Q4xForward *f) { return f ? f->b.logits : NULL; }

bool q4x_greedy(Q4xForward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[Q4X_FWD_ERR])
{
  if (!f || !ids || !rows || row0 + rows > f->max_tokens)
    return q4x_fail(err, "no forward or ids, or rows past a step's");
  const uint32_t V = (uint32_t)f->w->geo.n_vocab;
  int32_t *d = f->b.cnt; /* the MoE's table, free between steps */
  return (ops_argmax(f->b.logits + (uint64_t)row0 * V, V, rows, V, d, NULL) &&
          hipMemcpy(ids, d, rows * 4, hipMemcpyDeviceToHost) == hipSuccess) ||
         q4x_fail(err, "a GPU call failed (the greedy pick)");
}
