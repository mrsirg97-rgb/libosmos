#include <string.h>

#include "q4x_impl.h"

/* Qwen's MTP head (osmos/q4x/forward.h, q4x_mtp_seed / q4x_mtp_draft): llama.cpp's graph_mtp - h normed a stream, the
   token's embedding normed, both through eh_proj; then one attention layer and the MoE, hyper-connected; then its
   own head mixer and the main model's output */

#define F sizeof(float)

static bool copy(void *to, const void *from, uint64_t bytes)
{
  return hipMemcpyAsync(to, from, bytes, hipMemcpyDeviceToDevice, NULL) == hipSuccess;
}

/* the head's input rows into b.h: tokens (on the device in b.tok) and h (m.hin), n of them */
static bool head_in(Q4xForward *f, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xMtp *mtp = f->mtp;
  const GgufTensor *e = f->w->embed;
  Q4xBuf *b = &f->b;
  Q4xMtpBuf *m = &f->m;
  const uint32_t hc = (uint32_t)g->hc, D = (uint32_t)g->n_embd;
  return q4x_rows(e->type, q4x_dev(f, e), e->bytes / e->dims[1], D, b->tok, n, m->emb, NULL) &&
         q4x_hc_norm(m->emb, (const float *)q4x_dev(f, mtp->enorm), n, 1, D, g->rms_eps, b->x, NULL, NULL) &&
         q4x_hc_norm(m->hin, (const float *)q4x_dev(f, mtp->hnorm), n, hc, D, g->rms_eps, m->hn, NULL, NULL) &&
         q4x_join(b->x, m->hn, n, hc, D, m->join, NULL) &&
         q4x_mat(f, mtp->eh_proj, D, 2 * D, m->join, 2 * D, n * hc, b->h, D, false);
}

static bool tokens_in(Q4xForward *f, const int32_t *tokens, uint32_t n)
{
  memcpy(f->b.pin_tok, tokens, n * sizeof *tokens);
  return hipMemcpyAsync(f->b.tok, f->b.pin_tok, n * 4, hipMemcpyHostToDevice, NULL) == hipSuccess;
}

/* position p - 1's h for the head's first row: the pending one, or zeros */
static bool first_h(Q4xForward *f, Q4xSeq *s, int32_t p, float *to)
{
  const uint64_t hd = (uint64_t)f->w->geo.hc * f->w->geo.n_embd * F;
  return s->pend_pos == p - 1 ? copy(to, s->pend, hd) : hipMemsetAsync(to, 0, hd, NULL) == hipSuccess;
}

bool q4x_mtp_seed(Q4xForward *f, Q4xSeq *s, int32_t from, int32_t to, char err[Q4X_FWD_ERR])
{
  if (!f || !s || s->f != f || !f->mtp)
    return q4x_fail(err, "no forward with an MTP head, or a sequence of another forward");
  if (!s->step_n || from < s->step_pos0 || from >= to || to > s->step_pos0 + (int32_t)s->step_n || to > s->pos)
    return q4x_fail(err, "positions not of the sequence's last step");
  const Q4xGeometry *g = &f->w->geo;
  const Q4xLayer *L = &f->mtp->layer;
  Q4xMtpBuf *m = &f->m;
  const uint32_t n = (uint32_t)(to - from);
  const uint64_t hd = (uint64_t)g->hc * g->n_embd;
  const float *rows = m->hid + (uint64_t)s->step_row0 * hd; /* the step's h, position step_pos0 first */
  const int32_t at = from - s->step_pos0;
  bool ok = tokens_in(f, s->step_tok + at, n);
  if (at > 0)
    ok = ok && copy(m->hin, rows + (uint64_t)(at - 1) * hd, n * hd * F);
  else
    ok = ok && first_h(f, s, from, m->hin) && (n == 1 || copy(m->hin + hd, rows, (n - 1) * hd * F));
  ok = ok && head_in(f, n) && q4x_mix(f, L->hc_attn_norm, L->hc_attn_down, L->hc_attn_up, L->hc_attn_inject, n) &&
       q4x_attention_kv(f, L, s->mkc, s->mvc, from, n) &&
       copy(s->pend, rows + (uint64_t)(to - 1 - s->step_pos0) * hd, hd * F);
  if (!ok) /* its f16 overflow, if any, the next draft's or step's read finds (the flag accumulates) */
    return q4x_fail(err, "a GPU call failed (the MTP head's seeding)");
  s->pend_pos = to - 1;
  return true;
}

/* one draft at position p from b.tok's token and m.hin's h: its greatest token and its probability; m.hin then the
   head's own residual (the next draft's h) */
static bool one(Q4xForward *f, Q4xSeq *s, int32_t p, int32_t *id, float *prob)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xMtp *mtp = f->mtp;
  const Q4xLayer *L = &mtp->layer;
  Q4xBuf *b = &f->b;
  Q4xMtpBuf *m = &f->m;
  const uint32_t hc = (uint32_t)g->hc, D = (uint32_t)g->n_embd, V = (uint32_t)g->n_vocab;
  const uint32_t dv = Q4X_MTP_VOCAB < V ? Q4X_MTP_VOCAB : V; /* the draft's shortlist: the vocabulary's first rows */
  const float inv = 1.0f / (float)hc;
  const bool ok =
      head_in(f, 1) && q4x_mix(f, L->hc_attn_norm, L->hc_attn_down, L->hc_attn_up, L->hc_attn_inject, 1) &&
      q4x_attention(f, L, s->mkc, s->mvc, p, 1) && q4x_hc_combine(b->h, b->out, b->inj, 1, hc, D, inv, NULL) &&
      q4x_mix(f, L->hc_ffn_norm, L->hc_ffn_down, L->hc_ffn_up, L->hc_ffn_inject, 1) && q4x_moe(f, L, 1) &&
      q4x_hc_combine(b->h, b->out, b->inj, 1, hc, D, inv, NULL) && copy(m->hin, b->h, (uint64_t)hc * D * F) &&
      q4x_mix(f, mtp->head_norm, mtp->head_down, mtp->head_up, NULL, 1) &&
      q4x_mat(f, f->w->output, dv, D, b->mixed, D, 1, m->logits, dv, true) && /* b.x16 the head mixer's mixed */
      q4x_top(m->logits, dv, 1, 0, m->top_id, m->top_p, NULL) &&
      hipMemcpyAsync(m->pin_id, m->top_id, 4, hipMemcpyDeviceToHost, NULL) == hipSuccess &&
      hipMemcpyAsync(m->pin_p, m->top_p, 4, hipMemcpyDeviceToHost, NULL) == hipSuccess && q4x_overflow_read(f) &&
      hipStreamSynchronize(NULL) == hipSuccess;
  *id = *m->pin_id, *prob = *m->pin_p;
  return ok && q4x_overflow_ok(f, NULL);
}

int q4x_mtp_draft(Q4xForward *f, Q4xSeq *s, int32_t y, uint32_t max, float conf_min, float cum_min, int32_t *drafts,
                  float *conf, char err[Q4X_FWD_ERR])
{
  if (!f || !s || s->f != f || !f->mtp || !drafts || !conf)
    return q4x_fail(err, "no forward with an MTP head, a sequence of another forward, or no room for drafts"), -1;
  if (!first_h(f, s, s->pos, f->m.hin))
    return q4x_fail(err, "a GPU call failed (the MTP head's h)"), -1;
  float cum = 1;
  int n = 0;
  for (int32_t tok = y; (uint32_t)n < max && (uint32_t)(s->pos + n) < f->max_ctx; tok = drafts[n++])
  {
    int32_t id;
    float p;
    if (!tokens_in(f, &tok, 1) || !one(f, s, s->pos + n, &id, &p))
      return q4x_fail(err, "a GPU call failed (an MTP draft)"), -1;
    cum *= p;
    if (p < conf_min || cum < cum_min)
      break;
    drafts[n] = id, conf[n] = p;
  }
  return n;
}
