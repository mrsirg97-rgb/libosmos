/* the lfm3 forward's step (osmos/lfm3/forward.h): the tokens' embedding, each layer - its operator (the short
   convolution, or attention full or in a window) and its SwiGLU, each normed after and added to the residual
   (Liquid's sandwich norms) - then the kept rows' final norm and the tied head. one math: every matrix the matvec's
   (mv_strided: a token's sums its own), the attention the decode's (q35_attn_decode, Q35_ATTN_DECODE_MAX tokens a
   call) - a token's bits however its sequence is split into steps */
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

static bool conv(Lfm3Forward *f, Lfm3Seq *s, int l, uint32_t n)
{
  const Lfm3Geometry *g = &f->w->geo;
  const Lfm3Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd;
  return mat(f, L->in_proj, 3 * D, D, f->xn, n, f->bcx) &&
         lfm3_shortconv(f->bcx, n, D, (uint32_t)g->conv_taps, lfm3_fv(f, L->conv), s->conv[l], LFM3_CONV_SLOTS, 0, f->u,
                        f->st) &&
         mat(f, L->out_proj, D, D, f->u, n, f->y);
}

static bool attention(Lfm3Forward *f, Lfm3Seq *s, int l, uint32_t n)
{
  const Lfm3Geometry *g = &f->w->geo;
  const Lfm3Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, H = (uint32_t)g->n_heads, KV = (uint32_t)g->n_kv_heads;
  const uint32_t hd = (uint32_t)g->head_dim, q = H * hd, kv = KV * hd;
  const bool sliding = g->kind[l] == LFM3_SLIDING;
  const uint64_t row = q4x_kv_row_bytes(kv);
  const AttnShape shape = {hd, sliding ? (uint32_t)g->window : 0, 0};
  bool ok = mat(f, L->q, q, D, f->xn, n, f->qf) && mat(f, L->k, kv, D, f->xn, n, f->kf) &&
            mat(f, L->v, kv, D, f->xn, n, f->vf) && mat(f, L->gate, H, D, f->xn, n, f->g) &&
            q4x_qk(f->qf, f->kf, n, H, KV, hd, sliding ? &f->rope_swa : &f->rope_full, lfm3_fv(f, L->q_norm),
                   lfm3_fv(f, L->k_norm), g->rms_eps, s->pos, f->qr, NULL, f->kr, f->st) &&
            q4x_kv_put(f->kr, f->vf, n, kv, (uint8_t *)s->kc[l] + (uint64_t)s->pos * row,
                       (uint8_t *)s->vc[l] + (uint64_t)s->pos * row, f->st);
  for (uint32_t t0 = 0; ok && t0 < n; t0 += Q35_ATTN_DECODE_MAX)
  {
    const uint32_t m = n - t0 < Q35_ATTN_DECODE_MAX ? n - t0 : Q35_ATTN_DECODE_MAX;
    ok = q35_attn_decode(f->qr + (uint64_t)t0 * q, s->kc[l], s->vc[l], m, H, KV, hd, &shape, (uint32_t)s->pos + t0,
                         f->g + (uint64_t)t0 * H, f->attn_scratch, f->att + (uint64_t)t0 * q, f->st);
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

static bool layer(Lfm3Forward *f, Lfm3Seq *s, int l, uint32_t n)
{
  const Lfm3Geometry *g = &f->w->geo;
  const Lfm3Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, ff = (uint32_t)g->n_ff;
  return ops_rms(f->h, D, D, lfm3_fv(f, L->attn_norm), g->rms_eps, n, f->xn, D, f->st) &&
         (g->kind[l] == LFM3_CONV ? conv(f, s, l, n) : attention(f, s, l, n)) && post(f, L->post_attn_norm, n) &&
         ops_rms(f->h, D, D, lfm3_fv(f, L->ffn_norm), g->rms_eps, n, f->xn, D, f->st) &&
         mat(f, L->ffn_gate, ff, D, f->xn, n, f->ga) && mat(f, L->ffn_up, ff, D, f->xn, n, f->up) &&
         ops_swiglu(f->ga, f->up, ff, INFINITY, NULL, n, f->act, f->st) && mat(f, L->ffn_down, D, ff, f->act, n, f->y) &&
         post(f, L->post_ffn_norm, n);
}

bool lfm3_step(Lfm3Forward *f, Lfm3Seq *s, const int32_t *tokens, uint32_t n, uint32_t rows, char err[LFM3_ERR])
{
  if (!f || !s || s->f != f || !tokens || !n || n > f->max_tokens || !rows || rows > n || rows > f->logit_rows)
    return lfm3_fail(err, "no forward, sequence or tokens; past the step; or more rows than tokens or the logits'");
  const Lfm3Geometry *g = &f->w->geo;
  if ((uint64_t)s->pos + n > f->max_ctx)
    return lfm3_fail(err, "the tokens past the sequence's context");
  for (uint32_t i = 0; i < n; i++)
    if (tokens[i] < 0 || tokens[i] >= g->n_vocab)
      return lfm3_fail(err, "a token outside the vocabulary");
  const uint32_t D = (uint32_t)g->n_embd, V = (uint32_t)g->n_vocab;
  memcpy(f->pin_tok, tokens, (size_t)n * 4);
  bool ok = hipMemcpyAsync(f->tok, f->pin_tok, (size_t)n * 4, hipMemcpyHostToDevice, f->st) == hipSuccess &&
            ops_embed(f->w->embd->type, lfm3_dev(f, f->w->embd), D, f->tok, n, 1, f->h, f->st);
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    ok = layer(f, s, l, n);
    if (ok && f->tap && (ok = hipStreamSynchronize((hipStream_t)f->st) == hipSuccess))
      f->tap(f->tap_arg, l, "residual", f->h, n, D);
  }
  ok = ok && ops_rms(f->h + (uint64_t)(n - rows) * D, D, D, lfm3_fv(f, f->w->out_norm), g->rms_eps, rows, f->xf, D, f->st) &&
       mat(f, f->w->embd, V, D, f->xf, rows, f->logits) &&
       hipMemcpyAsync(f->pin_logits, f->logits, (uint64_t)rows * V * F, hipMemcpyDeviceToHost, f->st) == hipSuccess &&
       hipStreamSynchronize((hipStream_t)f->st) == hipSuccess;
  if (!ok)
    return lfm3_fail(err, "a GPU call failed in the step (the sequence's state now unknown: make another)");
  s->pos += (int32_t)n;
  return true;
}

const float *lfm3_logits(const Lfm3Forward *f) { return f ? f->pin_logits : NULL; }
