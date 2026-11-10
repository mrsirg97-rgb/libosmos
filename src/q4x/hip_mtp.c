#include <string.h>

#include "osmos/gpu/mm.h"
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
  if (!q4x_pin_wait(f)) /* a seed's copy still queued (the GPU busy) reads b.pin_tok yet: not written over */
    return false;
  memcpy(f->b.pin_tok, tokens, n * sizeof *tokens);
  return hipMemcpyAsync(f->b.tok, f->b.pin_tok, n * 4, hipMemcpyHostToDevice, NULL) == hipSuccess && q4x_pin_sent(f);
}

/* position p - 1's h for the head's first row: the pending one, or zeros. none pending is pend_pos -1, which at
   position 0 is p - 1 too: s.pend never written then (its memory not cleared), so zeros (0.5.10 read it) */
static bool first_h(Q4xForward *f, Q4xSeq *s, int32_t p, float *to)
{
  const uint64_t hd = (uint64_t)f->w->geo.hc * f->w->geo.n_embd * F;
  return s->pend_pos >= 0 && s->pend_pos == p - 1 ? copy(to, s->pend, hd)
                                                   : hipMemsetAsync(to, 0, hd, NULL) == hipSuccess;
}

bool q4x_mtp_seed(Q4xForward *f, Q4xSeq *s, int32_t from, int32_t to, char err[Q4X_FWD_ERR])
{
  if (!f || !s || s->f != f || !f->mtp)
    return q4x_fail(err, "no forward with an MTP head, or a sequence of another forward");
  if (s->broken)
    return q4x_fail(err, "a sequence after a failed step (its states part written): load it, or make another");
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
  ok = ok && head_in(f, n) && q4x_mix(f, L->hc_attn_norm, L->hc_attn_down, L->hc_attn_up, L->hc_attn_inject, n, false) &&
       q4x_attention_kv(f, L, &(Q4xCache){s->mkc, s->mvc, s->mixk, s->mixr, from, 0, n}, 1, n) &&
       copy(s->pend, rows + (uint64_t)(to - 1 - s->step_pos0) * hd, hd * F);
  if (!ok) /* its f16 overflow, if any, the next draft's or step's read finds (the flag accumulates) */
    return q4x_fail(err, "a GPU call failed (the MTP head's seeding)");
  s->pend_pos = to - 1;
  return true;
}

/* a draft step for n sequences, row i sequence s[i]'s at position pos[i] from token tok[i] and m.hin's row i: each
   one's greatest token and its probability; m.hin's rows then the head's own residuals (each one's next h) */
static bool step(Q4xForward *f, Q4xSeq *const *s, const int32_t *tok, const int32_t *pos, uint32_t n, int32_t *id,
                 float *prob)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xMtp *mtp = f->mtp;
  const Q4xLayer *L = &mtp->layer;
  Q4xBuf *b = &f->b;
  Q4xMtpBuf *m = &f->m;
  const uint32_t hc = (uint32_t)g->hc, D = (uint32_t)g->n_embd, V = (uint32_t)g->n_vocab;
  const uint32_t dv = Q4X_MTP_VOCAB < V ? Q4X_MTP_VOCAB : V; /* the draft's shortlist: the vocabulary's first rows */
  Q4xCache c[Q4X_MAX_RUNS];
  for (uint32_t i = 0; i < n; i++)
    c[i] = (Q4xCache){s[i]->mkc, s[i]->mvc, s[i]->mixk, s[i]->mixr, pos[i], i, 1};
  const bool ok =
      tokens_in(f, tok, n) && head_in(f, n) &&
      q4x_mix(f, L->hc_attn_norm, L->hc_attn_down, L->hc_attn_up, L->hc_attn_inject, n, false) &&
      q4x_attention(f, L, c, n, n) && /* each block's output into b.h in the next mixer's norm launch */
      q4x_mix(f, L->hc_ffn_norm, L->hc_ffn_down, L->hc_ffn_up, L->hc_ffn_inject, n, true) &&
      q4x_moe_shared(f, L, n) && q4x_moe(f, L, n) &&
      q4x_mix(f, mtp->head_norm, mtp->head_down, mtp->head_up, NULL, n, true) &&
      copy(m->hin, b->h, (uint64_t)n * hc * D * F) && /* the residual the head mixer read: the next draft's h */
      q4x_mat(f, f->w->output, dv, D, b->mixed, D, n, m->logits, dv, n <= MM_MAX_TOKENS) && /* the head mixer's x16 */
      q4x_top(m->logits, dv, n, dv, m->top_id, m->top_p, NULL) &&
      hipMemcpyAsync(m->pin_id, m->top_id, n * 4, hipMemcpyDeviceToHost, NULL) == hipSuccess &&
      hipMemcpyAsync(m->pin_p, m->top_p, n * 4, hipMemcpyDeviceToHost, NULL) == hipSuccess && q4x_overflow_read(f) &&
      hipStreamSynchronize(NULL) == hipSuccess;
  memcpy(id, m->pin_id, n * 4), memcpy(prob, m->pin_p, n * 4);
  return ok && q4x_overflow_ok(f, NULL);
}

bool q4x_mtp_draft_all(Q4xForward *f, Q4xSeq *const *s, const int32_t *y, uint32_t n, uint32_t max, float conf_min,
                       float cum_min, int32_t *drafts, float *conf, uint32_t stride, int *counts,
                       char err[Q4X_FWD_ERR])
{
  if (!f || !f->mtp || !s || !y || !drafts || !conf || !counts || !n || n > Q4X_MAX_RUNS || n > f->max_tokens ||
      stride < max)
    return q4x_fail(err, "no forward with an MTP head, no sequences or room, or not 1 .. Q4X_MAX_RUNS of them");
  for (uint32_t i = 0; i < n; i++)
  {
    if (s[i] && s[i]->broken)
      return q4x_fail(err, "a sequence after a failed step (its states part written): load it, or make another");
    if (!s[i] || s[i]->f != f)
      return q4x_fail(err, "a sequence of another forward");
    for (uint32_t j = 0; j < i; j++)
      if (s[j] == s[i])
        return q4x_fail(err, "a sequence twice");
  }
  const uint64_t hd = (uint64_t)f->w->geo.hc * f->w->geo.n_embd;
  Q4xSeq *as[Q4X_MAX_RUNS];
  uint32_t who[Q4X_MAX_RUNS], na = 0; /* the sequences still drafting, m.hin's rows in their order */
  int32_t tok[Q4X_MAX_RUNS], pos[Q4X_MAX_RUNS], id[Q4X_MAX_RUNS];
  float cum[Q4X_MAX_RUNS], p[Q4X_MAX_RUNS];
  bool ok = true;
  for (uint32_t i = 0; i < n; i++)
  {
    counts[i] = 0;
    if (max && (uint32_t)s[i]->pos < f->max_ctx)
      who[na] = i, as[na] = s[i], tok[na] = y[i], cum[na] = 1, pos[na] = s[i]->pos,
      ok = ok && first_h(f, s[i], s[i]->pos, f->m.hin + na * hd), na++;
  }
  while (ok && na)
  {
    if (!step(f, as, tok, pos, na, id, p))
      return q4x_fail(err, "a GPU call failed (an MTP draft)");
    uint32_t kept = 0;
    for (uint32_t j = 0; ok && j < na; j++)
    {
      const uint32_t i = who[j];
      cum[j] *= p[j];
      if (p[j] < conf_min || cum[j] < cum_min)
        continue;
      drafts[(uint64_t)i * stride + counts[i]] = id[j], conf[(uint64_t)i * stride + counts[i]] = p[j], counts[i]++;
      if ((uint32_t)counts[i] >= max || (uint32_t)(pos[j] + 1) >= f->max_ctx)
        continue;
      if (kept != j) /* its h down to its new row (rows in order: a row only ever moves up) */
        ok = copy(f->m.hin + kept * hd, f->m.hin + j * hd, hd * F);
      who[kept] = i, as[kept] = as[j], tok[kept] = id[j], cum[kept] = cum[j], pos[kept] = pos[j] + 1, kept++;
    }
    na = kept;
  }
  return ok || q4x_fail(err, "a GPU call failed (the MTP head's h)");
}

int q4x_mtp_draft(Q4xForward *f, Q4xSeq *s, int32_t y, uint32_t max, float conf_min, float cum_min, int32_t *drafts,
                  float *conf, char err[Q4X_FWD_ERR])
{
  int n = 0;
  return q4x_mtp_draft_all(f, &s, &y, 1, max, conf_min, cum_min, drafts, conf, max, &n, err) ? n : -1;
}
