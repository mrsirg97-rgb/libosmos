/* the lfm3 forward's step (osmos/lfm3/forward.h): the runs checked; the tokens' embedding; each layer - its operator
   (the short convolution, or attention full or in a window: a run's each, its own state) and its SwiGLU, each normed
   after and added to the residual (Liquid's sandwich norms); then the kept rows' final norm and the tied head. one
   math: every matrix the matvec's over the whole step (mv_strided: a token's sums its own), the attention the
   decode's (q35_attn_decode, Q35_ATTN_DECODE_MAX tokens a call) - a token's bits however its sequence is split into
   steps or shares one */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/attn_shape.h"
#include "osmos/gpu/lfm3_ops.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q35_ops.h"
#include "lfm3_impl.h"

static bool mat(Lfm3Forward *f, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t n, float *y)
{
  return mv_strided(t->type, lfm3_dev(f, t), rows, cols, x, cols, n, y, rows, f->st);
}

static bool conv(Lfm3Forward *f, const Lfm3Run *run, uint32_t runs, int l, uint32_t n)
{
  const Lfm3Geometry *g = &f->w->geo;
  const Lfm3Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd;
  bool ok = mat(f, L->in_proj, 3 * D, D, f->xn, n, f->bcx);
  for (uint32_t r = 0; ok && r < runs; r++)
    ok = lfm3_shortconv(f->bcx + (uint64_t)run[r].off * 3 * D, run[r].len, D, (uint32_t)g->conv_taps,
                        lfm3_fv(f, L->conv), run[r].s->conv[l], LFM3_CONV_SLOTS, run[r].s->cur,
                        f->u + (uint64_t)run[r].off * D, f->st);
  return ok && mat(f, L->out_proj, D, D, f->u, n, f->y);
}

/* a run's K and V rows into its cache: a full layer's at their positions, a sliding layer's ring at pos % ring - a
   step's rows past the ring's end wrapped to its start */
static bool kv_put(Lfm3Forward *f, const Lfm3Run *r, int l, bool sliding, uint32_t kv)
{
  const uint64_t row = q4x_kv_row_bytes(kv);
  const float *k = f->kr + (uint64_t)r->off * kv, *v = f->vf + (uint64_t)r->off * kv;
  uint8_t *kc = r->s->kc[l], *vc = r->s->vc[l];
  const uint32_t p = (uint32_t)r->s->pos, at = sliding ? p % f->ring : p;
  const uint32_t first = sliding && at + r->len > f->ring ? f->ring - at : r->len;
  bool ok = q4x_kv_put(k, v, first, kv, kc + at * row, vc + at * row, f->st);
  if (ok && first < r->len)
    ok = q4x_kv_put(k + (uint64_t)first * kv, v + (uint64_t)first * kv, r->len - first, kv, kc, vc, f->st);
  return ok;
}

static bool attention(Lfm3Forward *f, const Lfm3Run *run, uint32_t runs, int l, uint32_t n)
{
  const Lfm3Geometry *g = &f->w->geo;
  const Lfm3Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, H = (uint32_t)g->n_heads, KV = (uint32_t)g->n_kv_heads;
  const uint32_t hd = (uint32_t)g->head_dim, q = H * hd, kv = KV * hd;
  const bool sliding = g->kind[l] == LFM3_SLIDING;
  const AttnShape shape = {hd, sliding ? (uint32_t)g->window : 0, sliding ? f->ring : 0};
  bool ok = mat(f, L->q, q, D, f->xn, n, f->qf) && mat(f, L->k, kv, D, f->xn, n, f->kf) &&
            mat(f, L->v, kv, D, f->xn, n, f->vf) && mat(f, L->gate, H, D, f->xn, n, f->g);
  for (uint32_t r = 0; ok && r < runs; r++)
  {
    const Lfm3Run *R = &run[r];
    const uint64_t o = R->off;
    ok = q4x_qk(f->qf + o * q, f->kf + o * kv, R->len, H, KV, hd, sliding ? &f->rope_swa : &f->rope_full,
                lfm3_fv(f, L->q_norm), lfm3_fv(f, L->k_norm), g->rms_eps, R->s->pos, f->qr + o * q, NULL,
                f->kr + o * kv, f->st) &&
         kv_put(f, R, l, sliding, kv);
    for (uint32_t t0 = 0; ok && t0 < R->len; t0 += Q35_ATTN_DECODE_MAX)
    {
      const uint32_t m = R->len - t0 < Q35_ATTN_DECODE_MAX ? R->len - t0 : Q35_ATTN_DECODE_MAX;
      ok = q35_attn_decode(f->qr + (o + t0) * q, R->s->kc[l], R->s->vc[l], m, H, KV, hd, &shape,
                           (uint32_t)R->s->pos + t0, f->g + (o + t0) * H, f->attn_scratch, f->att + (o + t0) * q,
                           f->st);
    }
  }
  return ok && mat(f, L->out, D, q, f->att, n, f->y);
}

/* x += rms(y) w: a block's output normed (its post-norm) and added to the residual */
static bool post(Lfm3Forward *f, const GgufTensor *norm, uint32_t n)
{
  const uint32_t D = (uint32_t)f->w->geo.n_embd;
  return ops_rms(f->y, D, D, lfm3_fv(f, norm), f->w->geo.rms_eps, n, f->y, D, f->st) &&
         ops_add(f->h, f->y, (uint64_t)n * D, f->st);
}

static bool layer(Lfm3Forward *f, const Lfm3Run *run, uint32_t runs, int l, uint32_t n)
{
  const Lfm3Geometry *g = &f->w->geo;
  const Lfm3Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, ff = (uint32_t)g->n_ff;
  return ops_rms(f->h, D, D, lfm3_fv(f, L->attn_norm), g->rms_eps, n, f->xn, D, f->st) &&
         (g->kind[l] == LFM3_CONV ? conv(f, run, runs, l, n) : attention(f, run, runs, l, n)) &&
         post(f, L->post_attn_norm, n) &&
         ops_rms(f->h, D, D, lfm3_fv(f, L->ffn_norm), g->rms_eps, n, f->xn, D, f->st) &&
         mat(f, L->ffn_gate, ff, D, f->xn, n, f->ga) && mat(f, L->ffn_up, ff, D, f->xn, n, f->up) &&
         ops_swiglu(f->ga, f->up, ff, INFINITY, NULL, n, f->act, f->st) && mat(f, L->ffn_down, D, ff, f->act, n, f->y) &&
         post(f, L->post_ffn_norm, n);
}

/* the step (lfm3_step_runs); keep_from: rows below it not kept whatever the prompts say (lfm3_step's: its last rows) */
static bool step(Lfm3Forward *f, Lfm3Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                 uint32_t keep_from, char *err)
{
  if (!f || !seqs || !lens || !tokens || !runs || runs > LFM3_MAX_RUNS)
    return lfm3_fail(err, "no forward, sequences or tokens, or past LFM3_MAX_RUNS runs");
  Lfm3Run run[LFM3_MAX_RUNS];
  uint32_t n = 0, kept = 0;
  for (uint32_t r = 0; r < runs; r++)
  {
    Lfm3Seq *s = seqs[r];
    if (!s || s->f != f || !lens[r])
      return lfm3_fail(err, "a run of no sequence, of another forward, or of no tokens");
    if (s->broken)
      return lfm3_fail(err, "a sequence after a failed step: load it, or make another");
    for (uint32_t k = 0; k < r; k++)
      if (seqs[k] == s)
        return lfm3_fail(err, "a sequence twice in one step");
    if ((uint64_t)n + lens[r] > f->max_tokens || (uint64_t)s->pos + lens[r] > f->max_ctx)
      return lfm3_fail(err, "the tokens past the forward's step, or a sequence past its context");
    run[r] = (Lfm3Run){s, n, lens[r]};
    for (uint32_t t = 0; t < lens[r]; t++)
    {
      const bool keep = n + t >= keep_from && (!s->prompt_end || s->pos + t + 1 >= s->prompt_end);
      f->slot[n + t] = keep ? (int32_t)kept : -1;
      kept += keep;
    }
    n += lens[r];
  }
  if (kept > f->logit_rows)
    return lfm3_fail(err, "a step's kept rows past the logits' (Lfm3Options' logit_rows)");
  const Lfm3Geometry *g = &f->w->geo;
  for (uint32_t i = 0; i < n; i++)
    if (tokens[i] < 0 || tokens[i] >= g->n_vocab)
      return lfm3_fail(err, "a token outside the vocabulary");
  const uint32_t D = (uint32_t)g->n_embd, V = (uint32_t)g->n_vocab;
  memcpy(f->pin_tok, tokens, (size_t)n * 4);
  bool ok = hipMemcpyAsync(f->tok, f->pin_tok, (size_t)n * 4, hipMemcpyHostToDevice, f->st) == hipSuccess &&
            ops_embed(f->w->embd->type, lfm3_dev(f, f->w->embd), D, f->tok, n, 1, f->h, f->st);
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    ok = layer(f, run, runs, l, n);
    if (ok && f->tap && (ok = hipStreamSynchronize((hipStream_t)f->st) == hipSuccess))
      f->tap(f->tap_arg, l, "residual", f->h, n, D);
  }
  /* the kept rows' residuals gathered, then the final norm and the head on them alone */
  for (uint32_t row = 0; ok && row < n; row++)
    if (f->slot[row] >= 0)
      ok = hipMemcpyAsync(f->hk + (uint64_t)f->slot[row] * D, f->h + (uint64_t)row * D, (uint64_t)D * F,
                          hipMemcpyDeviceToDevice, f->st) == hipSuccess;
  if (ok && kept)
    ok = ops_rms(f->hk, D, D, lfm3_fv(f, f->w->out_norm), g->rms_eps, kept, f->xf, D, f->st) &&
         mat(f, f->w->embd, V, D, f->xf, kept, f->logits);
  ok = ok && hipStreamSynchronize((hipStream_t)f->st) == hipSuccess;
  f->rows = ok ? n : 0;
  if (!ok)
  {
    for (uint32_t r = 0; r < runs; r++)
      run[r].s->broken = true;
    return lfm3_fail(err, "a GPU call failed in the step (its sequences refused until a load)");
  }
  for (uint32_t r = 0; r < runs; r++)
  {
    Lfm3Seq *s = run[r].s;
    s->pos += (int32_t)run[r].len, s->cur = (s->cur + run[r].len) % LFM3_CONV_SLOTS;
    s->back = s->back + run[r].len < LFM3_MAX_BACK ? s->back + run[r].len : LFM3_MAX_BACK;
  }
  return true;
}

bool lfm3_step_runs(Lfm3Forward *f, Lfm3Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                    char err[LFM3_ERR])
{
  return step(f, seqs, lens, runs, tokens, 0, err);
}

bool lfm3_step(Lfm3Forward *f, Lfm3Seq *s, const int32_t *tokens, uint32_t n, uint32_t rows, char err[LFM3_ERR])
{
  if (!f || !s || !n || !rows || rows > n || rows > f->logit_rows)
    return lfm3_fail(err, "a step of no tokens, or more rows than tokens or the logits'");
  if (!step(f, &s, &n, 1, tokens, n - rows, err))
    return false;
  const uint32_t V = (uint32_t)f->w->geo.n_vocab;
  return hipMemcpy(f->pin_logits, f->logits, (uint64_t)rows * V * F, hipMemcpyDeviceToHost) == hipSuccess ||
         lfm3_fail(err, "a GPU call failed (the logits' copy)");
}

const float *lfm3_logits(const Lfm3Forward *f) { return f ? f->pin_logits : NULL; }

const float *lfm3_row_logits(const Lfm3Forward *f, uint32_t row)
{
  return f && row < f->rows && f->slot[row] >= 0 ? f->logits + (uint64_t)f->slot[row] * f->w->geo.n_vocab : NULL;
}

bool lfm3_greedy(Lfm3Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[LFM3_ERR])
{
  if (!f || !ids || !rows || row0 + rows > f->rows)
    return lfm3_fail(err, "no forward or ids, or rows past the last step's");
  for (uint32_t r = 0; r < rows; r++)
    if (f->slot[row0 + r] < 0 || f->slot[row0 + r] != f->slot[row0] + (int32_t)r)
      return lfm3_fail(err, "a row the last step kept no logits of");
  const uint32_t V = (uint32_t)f->w->geo.n_vocab;
  if (!ops_argmax(f->logits + (uint64_t)f->slot[row0] * V, V, rows, V, f->ids, f->argmax, f->st) ||
      hipMemcpyAsync(ids, f->ids, rows * 4, hipMemcpyDeviceToHost, f->st) != hipSuccess ||
      hipStreamSynchronize((hipStream_t)f->st) != hipSuccess)
    return lfm3_fail(err, "a GPU call failed (the greedy pick)");
  for (uint32_t r = 0; r < rows; r++)
    if (ids[r] < 0)
      return lfm3_fail(err, "a NaN in a row's logits");
  return true;
}
