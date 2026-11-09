#include <string.h>

#include "osmos/q4x/ple.h"
#include "osmos/gpu/mm.h"
#include "q4x_impl.h"

/* a step: the tokens' embedding into hc streams, their n-gram rows, the layers, the end mixer, the logits - every
   sequence's tokens at once through the weights, each sequence's own through its own (its runs: Q4xRun) */

bool q4x_pin_wait(Q4xForward *f) { return hipEventSynchronize(f->pin_free) == hipSuccess; }

bool q4x_pin_sent(Q4xForward *f) { return hipEventRecord(f->pin_free, NULL) == hipSuccess; }

static bool embed(Q4xForward *f, const Q4xRun *r, uint32_t runs, const int32_t *tokens, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  const GgufTensor *e = f->w->embed;
  Q4xBuf *b = &f->b;
  int32_t prev[Q4X_MAX_NGRAM];
  if (!q4x_pin_wait(f)) /* b.pin_rows and b.pin_tok written below: their last copies done */
    return false;
  for (uint32_t i = 0; i < runs; i++)
    for (uint32_t t = 0; t < r[i].n; t++) /* a token's predecessors: its run's before it, then the sequence's last */
    {
      const Q4xSeq *s = r[i].s;
      const int32_t *tok = tokens + r[i].t0;
      int np = 0;
      for (int k = 1; k < g->ple_ngram; k++)
      {
        const int64_t at = (int64_t)t - k;
        if (at >= 0)
          prev[np++] = tok[at];
        else if (-at - 1 < s->n_last)
          prev[np++] = s->last[-at - 1];
      }
      q4x_ple_rows(g, tok[t], prev, np, &b->pin_rows[(uint64_t)(r[i].t0 + t) * g->ple_heads]);
    }
  memcpy(b->pin_tok, tokens, n * sizeof *tokens);
  return hipMemcpyAsync(b->tok, b->pin_tok, n * 4, hipMemcpyHostToDevice, NULL) == hipSuccess &&
         hipMemcpyAsync(b->ple_rows, b->pin_rows, (uint64_t)n * g->ple_heads * 4, hipMemcpyHostToDevice, NULL) ==
             hipSuccess &&
         q4x_pin_sent(f) &&
         q4x_rows(e->type, q4x_dev(f, e), e->bytes / e->dims[1], (uint32_t)e->dims[0], b->tok, n, b->x, NULL) &&
         q4x_streams(b->x, n, (uint32_t)g->hc, (uint32_t)g->n_embd, b->h, NULL);
}

/* a layer's part before its routed experts, over n tokens (b's rows from 0, the runs' t0 from 0): the n-gram layer,
   the mixer, the DeltaNet or the attention, the FFN's mixer into b.mixed, the shared expert. pending: the layer before's MoE output not yet
   in b.h - combined in the mixer's norm launch, or alone where the residual is read first (the n-gram layer, a probe) */
static bool layer_part(Q4xForward *f, const Q4xRun *r, uint32_t runs, int l, uint32_t n, bool pending)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xLayer *L = &f->w->layer[l];
  Q4xBuf *b = &f->b;
  const uint32_t hc = (uint32_t)g->hc, D = (uint32_t)g->n_embd;
  const float inv = 1.0f / (float)hc; /* the inject's scale (llama.cpp's 1 / hc before the sigmoid) */
  Q4xCache c[Q4X_MAX_RUNS];
  for (uint32_t i = 0; i < runs; i++)
    c[i] = (Q4xCache){r[i].s->kc[l], r[i].s->vc[l], r[i].s->ixk[l], r[i].s->ixr[l], r[i].s->pos, r[i].t0, r[i].n};
  bool ok = true;
  if (pending && (l == g->ple_layer || f->probe))
    ok = q4x_hc_combine(b->h, b->out, b->inj, n, hc, D, inv, NULL), pending = false;
  return ok && (l != g->ple_layer || q4x_ple(f, r, runs, l, n)) &&
         q4x_mix(f, L->hc_attn_norm, L->hc_attn_down, L->hc_attn_up, L->hc_attn_inject, n, pending) &&
         (g->recurrent[l] ? q4x_deltanet(f, r, runs, l, n) : q4x_attention(f, L, c, runs, n)) &&
         q4x_mix(f, L->hc_ffn_norm, L->hc_ffn_down, L->hc_ffn_up, L->hc_ffn_inject, n, true) &&
         q4x_moe_shared(f, L, n);
}

/* tokens k0 .. k0 + m of the step as a step of their own: b's token rows from k0 (f->rows), and each run's part in
   it - its sequence at the part's first token (the rings' cur and the position moved on by the run's tokens before:
   a ring's slot for a run's token t is cur + 1 + t, q4x_ops.h - so a part reads the slot the part before wrote, as a
   step reads the step before's). *saved: b before, restored by sub_end */
static uint32_t sub_begin(Q4xForward *f, const Q4xRun *r, uint32_t runs, uint32_t k0, uint32_t m, Q4xRun *sr,
                          Q4xBuf *saved)
{
  *saved = f->b;
  for (uint32_t i = 0; i < f->n_rows; i++)
    *(uint8_t **)f->rows[i].p += (uint64_t)k0 * f->rows[i].row;
  uint32_t n = 0;
  for (uint32_t i = 0; i < runs; i++)
  {
    const uint32_t a = r[i].t0 > k0 ? r[i].t0 : k0, e = r[i].t0 + r[i].n < k0 + m ? r[i].t0 + r[i].n : k0 + m;
    if (a >= e)
      continue;
    const uint32_t o = a - r[i].t0; /* the run's tokens before the part */
    Q4xSeq *s = &f->sub_seq[n];
    *s = *r[i].s;
    s->pos += (int32_t)o, s->cur = (s->cur + o) % Q4X_SLOTS;
    sr[n++] = (Q4xRun){s, a - k0, e - a};
  }
  return n;
}

static void sub_end(Q4xForward *f, const Q4xBuf *saved) { f->b = *saved; }

/* a layer; *pending: the layer before's MoE output not yet in b.h (in, then this layer's: out). a step past
   MM_MAX_TOKENS: its part a sub-batch of MM_MAX_TOKENS at a time - a 2K step's activations stay in the cache, the
   bigger step's would not - then its experts over the step whole (each expert's weights read once for all its tokens:
   more of them a read). the same bits as steps of MM_MAX_TOKENS (test_q4x_forward_gpu: a step of 3,000) */
static bool layer(Q4xForward *f, const Q4xRun *r, uint32_t runs, int l, uint32_t n, bool *pending)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xLayer *L = &f->w->layer[l];
  Q4xBuf *b = &f->b;
  const uint32_t hc = (uint32_t)g->hc, D = (uint32_t)g->n_embd;
  const float inv = 1.0f / (float)hc;
  bool ok = true;
  if (n <= MM_MAX_TOKENS)
    ok = layer_part(f, r, runs, l, n, *pending);
  else
    for (uint32_t k0 = 0; ok && k0 < n; k0 += MM_MAX_TOKENS)
    {
      const uint32_t m = n - k0 < MM_MAX_TOKENS ? n - k0 : MM_MAX_TOKENS;
      Q4xRun sr[Q4X_MAX_RUNS];
      Q4xBuf saved;
      const uint32_t sn = sub_begin(f, r, runs, k0, m, sr, &saved);
      ok = layer_part(f, sr, sn, l, m, *pending);
      sub_end(f, &saved);
    }
  ok = ok && q4x_moe(f, L, n);
  *pending = true;
  if (ok && f->probe) /* the layer's residual whole for the probe (l_last) */
    ok = q4x_hc_combine(b->h, b->out, b->inj, n, hc, D, inv, NULL), *pending = false;
  return ok;
}

/* the step's runs from seqs and lens (each sequence once, of this forward, within its context): their tokens' count,
   0 when refused */
static uint32_t runs_of(const Q4xForward *f, Q4xSeq *const *seqs, const uint32_t *lens, uint32_t runs, Q4xRun *r)
{
  uint32_t n = 0;
  for (uint32_t i = 0; i < runs; i++)
  {
    if (!seqs[i] || seqs[i]->f != f || !lens[i] || (uint64_t)seqs[i]->pos + lens[i] > f->max_ctx)
      return 0;
    for (uint32_t j = 0; j < i; j++)
      if (seqs[j] == seqs[i])
        return 0;
    r[i] = (Q4xRun){seqs[i], n, lens[i]};
    n += lens[i];
  }
  return n;
}

/* a step's sequences after it: their last tokens, positions and rings' latest; with a head, the step's place */
static void advance(Q4xForward *f, const Q4xRun *r, uint32_t runs, const int32_t *tokens, uint32_t row0)
{
  for (uint32_t i = 0; i < runs; i++)
  {
    Q4xSeq *s = r[i].s;
    const int32_t *tok = tokens + r[i].t0;
    const uint32_t n = r[i].n;
    if (f->mtp)
    {
      memcpy(s->step_tok, tok, n * sizeof *tok);
      s->step_pos0 = s->pos, s->step_row0 = row0 + r[i].t0, s->step_n = n;
    }
    const int keep = (int)(sizeof s->last / sizeof *s->last);
    for (uint32_t t = 0; t < n; t++) /* the latest first */
    {
      memmove(s->last + 1, s->last, (size_t)(keep - 1) * sizeof *s->last);
      s->last[0] = tok[t], s->n_last += s->n_last < keep;
    }
    s->pos += (int32_t)n;
    s->cur = (s->cur + n) % Q4X_SLOTS; /* the rings' latest: the step's last token's (q4x_ops.h) */
    s->back = s->back + n < Q4X_MAX_BACK ? s->back + n : Q4X_MAX_BACK;
  }
}

bool q4x_step(Q4xForward *f, Q4xSeq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
              uint32_t row0, char err[Q4X_FWD_ERR])
{
  Q4xRun r[Q4X_MAX_RUNS];
  if (!f || !seqs || !lens || !tokens || !runs || runs > Q4X_MAX_RUNS)
    return q4x_fail(err, "no forward, sequences or tokens, or runs not 1 .. Q4X_MAX_RUNS");
  const uint32_t n = runs_of(f, seqs, lens, runs, r);
  if (!n || (uint64_t)row0 + n > f->max_tokens)
    return q4x_fail(err, "a run of no tokens, a sequence twice, of another forward or past its context, or the rows "
                         "past the forward's max_tokens");
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint64_t hd = (uint64_t)g->hc * g->n_embd;
  if (!embed(f, r, runs, tokens, n))
    return q4x_fail(err, "a GPU call failed (the embedding)");
  bool pending = false;
  for (int l = 0; l < g->n_layers; l++)
  {
    if (!layer(f, r, runs, l, n, &pending))
      return q4x_fail(err, "a GPU call failed (a layer)");
    q4x_seen(f, "l_last", l, b->h, n * hd);
  }
  const Q4xWeights *w = f->w;
  if (!q4x_mix(f, w->out_hc_norm, w->out_hc_down, w->out_hc_up, NULL, n, pending)) /* the last MoE's into b.h */
    return q4x_fail(err, "a GPU call failed (the end mixer)");
  /* the head's h: the residual the end mixer read, at the step's rows (each sequence's place: advance) */
  if (f->mtp && hipMemcpyAsync(f->m.hid + (uint64_t)row0 * hd, b->h, (uint64_t)n * hd * sizeof(float),
                               hipMemcpyDeviceToDevice, NULL) != hipSuccess)
    return q4x_fail(err, "a GPU call failed (the MTP head's rows)");
  q4x_seen(f, "result_norm", -1, b->mixed, (uint64_t)n * g->n_embd);
  if (!q4x_mat(f, w->output, (uint32_t)g->n_vocab, (uint32_t)g->n_embd, b->mixed, (uint32_t)g->n_embd, n,
               b->logits + (uint64_t)row0 * g->n_vocab, (uint32_t)g->n_vocab, n <= MM_MAX_TOKENS) ||
      !q4x_overflow_read(f) || hipStreamSynchronize(NULL) != hipSuccess)
    return q4x_fail(err, "a GPU call failed (the logits)");
  if (!q4x_overflow_ok(f, err))
    return false;
  advance(f, r, runs, tokens, row0);
  return true;
}

bool q4x_batch(Q4xForward *f, Q4xSeq *s, const int32_t *tokens, uint32_t n, uint32_t row0, char err[Q4X_FWD_ERR])
{
  return q4x_step(f, &s, &n, 1, tokens, row0, err);
}

const float *q4x_logits(const Q4xForward *f) { return f ? f->b.logits : NULL; }

bool q4x_greedy(Q4xForward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[Q4X_FWD_ERR])
{
  if (!f || !ids || !rows || row0 + rows > f->max_tokens)
    return q4x_fail(err, "no forward or ids, or rows past a step's");
  const uint32_t V = (uint32_t)f->w->geo.n_vocab;
  int32_t *d = f->b.cnt; /* the MoE's table, free between steps */
  return (q4x_top(f->b.logits + (uint64_t)row0 * V, V, rows, V, d, NULL, NULL) &&
          hipMemcpy(ids, d, rows * 4, hipMemcpyDeviceToHost) == hipSuccess) ||
         q4x_fail(err, "a GPU call failed (the greedy pick)");
}
