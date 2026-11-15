/* Inkling-2's drafter on the GPU (osmos/ink2/forward.h: ink2_seq_seed / ink2_seq_draft): DSpark with the RNN Markov
   head, TokenSpeed's inkling2_mtp.py's math (tools/convert/ink2_mtp_oracle.py holds it in PyTorch). the target's taps
   - h at the entry of its target layers and its final normed hidden - seed each drafter layer's context: its own
   mixture of them normed, its ctx keys (normed, RoPE'd) and values into the sequence's rings. a round's block: the
   anchor's normed embedding then mask rows at the sequence's position on, through the layers (attention over the
   context and the block, not causal), the target's head on rows 1 .. block - 1, then the Markov steps - each its
   GRU-like state over the step's previous pick, the folded head's correction, the softcap, the greatest - on the
   device, one after another */
#include <hip/hip_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ink2_impl.h"
#include "osmos/gpu/ink2_ops.h"
#include "osmos/gpu/mm.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q4x_ops.h"

#define T_F32 0u
#define T_Q8_0 8u
#define Q8_0_BYTES(n) ((uint64_t)(n) / 32 * 34)

static bool fail(char *err, const char *why) { return ink2_fail(err, why); }

static uint32_t kvw(const Ink2Mtp *m) { return (uint32_t)m->n_kv_heads * INK2_MTP_HEAD; }
static uint32_t qw(const Ink2Mtp *m) { return (uint32_t)m->n_heads * INK2_MTP_HEAD; }

/* a local layer's ring: its window and a block back of the anchor, a round's block ahead (rounded up) */
static uint32_t ring_rows(const Ink2Forward *f, int l)
{
  const Ink2Mtp *m = f->mtp;
  return m->local[l] ? (uint32_t)(m->window + 2 * m->block) : f->max_ctx;
}

/* the block's y0 = W0 x (and y1 = W1 x: W1 NULL, or of W0's type and cols) on n rows: mm where it tiles (the weights
   read once, a slab decoded once for the tile), else the matvec. *held: d->x16 holds this x (an mm before it on the
   same x), set by an mm */
static bool dmat(Ink2Forward *f, const GgufTensor *t0, const void *w0, float *y0, const GgufTensor *t1, const void *w1,
                 float *y1, const float *x, uint32_t n, bool *held)
{
  Ink2MtpDev *d = &f->md;
  const uint32_t type = ink2_type(t0), cols = (uint32_t)t0->dims[0], r0 = (uint32_t)t0->dims[1],
                 r1 = t1 ? (uint32_t)t1->dims[1] : 0;
  const bool mm = mm_known(type) && mm_fits(type, r0, cols) &&
                  (!t1 || (ink2_type(t1) == type && (uint32_t)t1->dims[0] == cols && mm_fits(type, r1, cols)));
  if (!mm)
    return ink2_mv(t0, w0, x, n, y0) && (!t1 || ink2_mv(t1, w1, x, n, y1));
  *held = mm_tokens(type, w0, r0, y0, r0, t1 ? w1 : NULL, r1, y1, r1, cols, x, cols, n, d->x16, d->overflow, *held,
                    NULL);
  return *held;
}

bool ink2_mtp_up(Ink2Forward *f, char *err)
{
  const Ink2Mtp *m = f->mtp;
  const Gguf *g = m->model;
  Ink2MtpDev *d = &f->md;
  const uint32_t C = (uint32_t)m->n_embd, R = (uint32_t)m->rank, T = f->max_tokens, B = (uint32_t)m->block,
                 VR = (uint32_t)f->w->output->dims[1], KV = kvw(m), Q = qw(m), F = (uint32_t)m->ff,
                 wide = F > Q ? (F > C ? F : C) : (Q > C ? Q : C); /* the block's widest input */
  bool ok = true;
  for (int l = 0; ok && l < m->n_layers; l++)
  {
    const Ink2MtpLayer *L = &m->layer[l];
    ok = ink2_up_raw_of(f, g, L->q, &d->wq[l]) && ink2_up_raw_of(f, g, L->k, &d->wk[l]) &&
         ink2_up_raw_of(f, g, L->v, &d->wv[l]) && ink2_up_raw_of(f, g, L->k_ctx, &d->wkc[l]) &&
         ink2_up_raw_of(f, g, L->v_ctx, &d->wvc[l]) && ink2_up_raw_of(f, g, L->o, &d->wo[l]) &&
         ink2_up_raw_of(f, g, L->gate, &d->wgate[l]) && ink2_up_raw_of(f, g, L->up, &d->wup[l]) &&
         ink2_up_raw_of(f, g, L->down, &d->wdown[l]) && ink2_up_f32_of(f, g, L->in_norm, &d->in_norm[l]) &&
         ink2_up_f32_of(f, g, L->post_norm, &d->post_norm[l]) && ink2_up_f32_of(f, g, L->dflare_norm, &d->dflare[l]) &&
         ink2_up_f32_of(f, g, L->q_norm, &d->q_norm[l]) && ink2_up_f32_of(f, g, L->k_norm, &d->k_norm[l]) &&
         ink2_up_f32_of(f, g, L->k_norm_ctx, &d->k_norm_ctx[l]);
  }
  ok = ok && ink2_up_f32_of(f, g, m->mask, &d->mask) && ink2_up_f32_of(f, g, m->norm, &d->norm) &&
       ink2_up_f32_of(f, g, m->m_dflare, &d->m_dflare) && ink2_up_f32_of(f, g, m->m_rnn_b, &d->rnn_b) &&
       ink2_up_raw_of(f, g, m->m_w1, &d->w1);
  if (!ok)
    return fail(err, "inkling2_mtp: a drafter tensor not read or uploaded");
  /* the mixtures: softmax over the taps, a stage a row */
  uint64_t n = 0;
  float *fw = ink2_host_f32(g, m->fuse, &n), *rnn = NULL, *hw = NULL, *hb = NULL;
  ok = fw && n == (uint64_t)(m->n_layers + 1) * m->n_taps;
  for (int st = 0; ok && st <= m->n_layers; st++)
  {
    float *row = fw + (size_t)st * m->n_taps, mx = row[0], sum = 0;
    for (int k = 1; k < m->n_taps; k++)
      mx = row[k] > mx ? row[k] : mx;
    for (int k = 0; k < m->n_taps; k++)
      sum += row[k] = expf(row[k] - mx);
    for (int k = 0; k < m->n_taps; k++)
      row[k] /= sum;
  }
  ok = ok && ink2_up_host(f, fw, n, &d->alpha);
  /* rnn_proj's columns apart: [state | prev] a step's own, [hidden | features] the step's invariant */
  ok = ok && (rnn = ink2_host_f32(g, m->m_rnn_w, &n)) != NULL;
  if (ok)
  {
    const uint32_t cols = 2 * R + 2 * C, rows = 3 * R;
    float *sp = malloc((size_t)rows * 2 * R * 4), *hf = malloc((size_t)rows * 2 * C * 4);
    ok = sp && hf;
    for (uint32_t r = 0; ok && r < rows; r++)
      memcpy(sp + (size_t)r * 2 * R, rnn + (size_t)r * cols, 2 * R * 4),
          memcpy(hf + (size_t)r * 2 * C, rnn + (size_t)r * cols + 2 * R, 2 * C * 4);
    ok = ok && ink2_up_host(f, sp, (uint64_t)rows * 2 * R, &d->w_sp) &&
         ink2_up_host(f, hf, (uint64_t)rows * 2 * C, &d->w_hf);
    free(sp), free(hf);
  }
  /* the fold: the target's head through hidden_out - its rank columns as tokens (the head's own quantized values,
     the base logits' head), transposed to [rows][rank] Q8_0 for the matvec (a step reads it whole: 55 MB, BF16's
     103); and the bias's */
  ok = ok && (hw = ink2_host_f32(g, m->m_hidden_w, &n)) != NULL && (hb = ink2_host_f32(g, m->m_hidden_b, &n)) != NULL;
  float *cols = NULL, *dcols = NULL, *dt = NULL, *db = NULL;
  if (ok)
  {
    cols = malloc((size_t)R * C * 4); /* hidden_out's weight [C][R] as R vectors of C */
    ok = cols != NULL;
    for (uint32_t r = 0; ok && r < R; r++)
      for (uint32_t c = 0; c < C; c++)
        cols[(size_t)r * C + c] = hw[(size_t)c * R + r];
    ok = ok && hipMalloc((void **)&dcols, (size_t)R * C * 4) == hipSuccess &&
         hipMemcpy(dcols, cols, (size_t)R * C * 4, hipMemcpyHostToDevice) == hipSuccess &&
         hipMalloc((void **)&dt, (size_t)R * VR * 4) == hipSuccess && hipMalloc((void **)&db, (size_t)C * 4) == hipSuccess &&
         hipMemcpy(db, hb, (size_t)C * 4, hipMemcpyHostToDevice) == hipSuccess &&
         ink2_mv(f->w->output, f->output, dcols, R, dt) && ink2_dalloc(f, &d->fold, Q8_0_BYTES((uint64_t)VR * R)) &&
         ink2_transpose_q8_0(dt, R, VR, d->fold, NULL) && ink2_dalloc(f, (void **)&d->fold_b, (uint64_t)VR * 4) &&
         ink2_mv(f->w->output, f->output, db, 1, d->fold_b) && hipDeviceSynchronize() == hipSuccess;
  }
  free(fw), free(rnn), free(hw), free(hb), free(cols);
  if (dcols)
    hipFree(dcols);
  if (dt)
    hipFree(dt);
  if (db)
    hipFree(db);
  if (!ok)
    return fail(err, "inkling2_mtp: the drafter's mixtures, Markov projections or fold not made");
  struct
  {
    void **p;
    uint64_t bytes;
  } b[] = {
      {(void **)&d->taps, (uint64_t)T * m->n_taps * C * 4}, {(void **)&d->mix, (uint64_t)T * C * 4},
      {(void **)&d->kctx, (uint64_t)T * KV * 4}, {(void **)&d->vctx, (uint64_t)T * KV * 4},
      {(void **)&d->x, (uint64_t)B * C * 4}, {(void **)&d->res, (uint64_t)B * C * 4}, {(void **)&d->nrm, (uint64_t)B * C * 4},
      {(void **)&d->q, (uint64_t)B * Q * 4}, {(void **)&d->kb, (uint64_t)B * KV * 4}, {(void **)&d->vb, (uint64_t)B * KV * 4},
      {(void **)&d->att, (uint64_t)B * Q * 4}, {(void **)&d->ob, (uint64_t)B * C * 4}, {(void **)&d->g, (uint64_t)B * F * 4},
      {(void **)&d->u, (uint64_t)B * F * 4}, {(void **)&d->hf, (uint64_t)B * 2 * C * 4},
      {(void **)&d->inv, (uint64_t)B * 3 * R * 4}, {(void **)&d->vec, 2 * R * 4}, {(void **)&d->p, 3 * R * 4},
      {(void **)&d->state, R * 4}, {(void **)&d->latent, R * 4}, {(void **)&d->corr, (uint64_t)VR * 4},
      {(void **)&d->base, (uint64_t)B * VR * 4}, {(void **)&d->lg, ink2_mtp_pick_scratch()}, {(void **)&d->probs, B * 4},
      {(void **)&d->scr, ink2_mtp_attn_scratch(B, (uint32_t)m->n_heads, f->max_ctx)}, {(void **)&d->f_rnn, C * 4},
      {(void **)&d->pos, B * 4}, {(void **)&d->prev, 4}, {(void **)&d->ids, B * 4},
      {&d->x16, (uint64_t)MM_TILE * wide * 2}, {(void **)&d->overflow, 4},
  };
  for (size_t i = 0; i < sizeof b / sizeof *b; i++)
    if (!ink2_dalloc(f, b[i].p, b[i].bytes))
      return fail(err, "inkling2_mtp: GPU memory refused (the drafter's buffers)");
  return hipMemset(d->overflow, 0, 4) == hipSuccess || fail(err, "inkling2_mtp: a GPU call failed (its overflow flag)");
}

void ink2_mtp_down(Ink2Forward *f)
{
  Ink2MtpDev *d = &f->md;
  void *p[] = {d->alpha, d->mask, d->norm, d->m_dflare, d->w_sp, d->w_hf, d->rnn_b, d->fold_b, d->w1, d->fold,
               d->taps, d->mix, d->kctx, d->vctx, d->x, d->res, d->nrm, d->q, d->kb, d->vb, d->att, d->ob, d->g, d->u,
               d->hf, d->inv, d->vec, d->p, d->state, d->latent, d->corr, d->base, d->lg, d->probs, d->scr, d->f_rnn,
               d->pos, d->prev, d->ids, d->x16, d->overflow};
  for (size_t i = 0; i < sizeof p / sizeof *p; i++)
    if (p[i])
      hipFree(p[i]);
  for (int l = 0; l < INK2_MTP_MAX_LAYERS; l++)
  {
    void *q[] = {d->wq[l], d->wk[l], d->wv[l], d->wkc[l], d->wvc[l], d->wo[l], d->wgate[l], d->wup[l], d->wdown[l],
                 d->in_norm[l], d->post_norm[l], d->dflare[l], d->q_norm[l], d->k_norm[l], d->k_norm_ctx[l]};
    for (size_t i = 0; i < sizeof q / sizeof *q; i++)
      if (q[i])
        hipFree(q[i]);
  }
  memset(d, 0, sizeof *d);
}

bool ink2_mtp_seq_make(Ink2Seq *s)
{
  const Ink2Forward *f = s->f;
  const uint32_t KV = kvw(f->mtp);
  for (int l = 0; l < f->mtp->n_layers; l++)
  {
    s->mrows[l] = ring_rows(f, l);
    const uint64_t bytes = (uint64_t)s->mrows[l] * KV * 2;
    if (hipMalloc((void **)&s->mk[l], bytes) != hipSuccess || hipMalloc((void **)&s->mv[l], bytes) != hipSuccess)
      return false;
  }
  s->mfrom = s->mend = 0;
  return hipMalloc((void **)&s->mlast, (uint64_t)f->mtp->n_taps * f->mtp->n_embd * 4) == hipSuccess;
}

void ink2_mtp_seq_free(Ink2Seq *s)
{
  for (int l = 0; l < INK2_MTP_MAX_LAYERS; l++)
  {
    if (s->mk[l])
      hipFree(s->mk[l]);
    if (s->mv[l])
      hipFree(s->mv[l]);
  }
  if (s->mlast)
    hipFree(s->mlast);
}

bool ink2_mtp_tap(Ink2Forward *f, int slot, const float *x, uint32_t n, uint32_t row0)
{
  const uint32_t C = (uint32_t)f->mtp->n_embd, K = (uint32_t)f->mtp->n_taps;
  return hipMemcpy2DAsync(f->md.taps + ((uint64_t)row0 * K + (uint32_t)slot) * C, (uint64_t)K * C * 4, x,
                          (uint64_t)C * 4, (uint64_t)C * 4, n, hipMemcpyDeviceToDevice, NULL) == hipSuccess;
}

static bool positions(Ink2Forward *f, int32_t p0, uint32_t n)
{
  int32_t pos[INK2_MTP_BLOCK];
  for (uint32_t i = 0; i < n; i++)
    pos[i] = p0 + (int32_t)i;
  return hipMemcpy(f->md.pos, pos, n * 4, hipMemcpyHostToDevice) == hipSuccess;
}

bool ink2_seq_seed(Ink2Forward *f, Ink2Seq *s, int32_t from, int32_t to, char err[INK2_FWD_ERR])
{
  if (!f || !s || s->f != f || !f->mtp)
    return fail(err, "inkling2: no forward with a drafter, or a sequence of another forward");
  if (s->failed)
    return fail(err, "inkling2: a sequence after a failed step: load it, or make another");
  if (from >= to)
    return true;
  if (from < s->tap_pos0 || to > s->pos || (uint32_t)(to - s->tap_pos0) > f->max_tokens)
    return fail(err, "inkling2: positions not of the sequence's last step, or past its position");
  const Ink2Mtp *m = f->mtp;
  Ink2MtpDev *d = &f->md;
  const uint32_t C = (uint32_t)m->n_embd, KV = kvw(m), K = (uint32_t)m->n_taps, n = (uint32_t)(to - from);
  const float *taps = d->taps + ((uint64_t)s->tap_row0 + (uint32_t)(from - s->tap_pos0)) * K * C;
  /* the context: the positions seeded since the last gap (a seed past it starts it over) */
  if (from > s->mend || from < s->mfrom)
    s->mfrom = from;
  bool ok = true;
  for (int32_t at = 0; ok && at < (int32_t)n; at += INK2_MTP_BLOCK) /* the rope's positions a block at a time */
  {
    const uint32_t k = n - (uint32_t)at < INK2_MTP_BLOCK ? n - (uint32_t)at : INK2_MTP_BLOCK;
    ok = positions(f, from + at, k);
    for (int l = 0; ok && l < m->n_layers; l++)
    {
      const Ink2MtpLayer *L = &m->layer[l];
      bool held = false;
      ok = ink2_mtp_mix(taps + (uint64_t)at * K * C, K, d->alpha + (size_t)l * K, k, C, d->mix, NULL) &&
           ops_rms(d->mix, C, C, d->dflare[l], m->eps, k, d->mix, C, NULL) &&
           dmat(f, L->k_ctx, d->wkc[l], d->kctx, L->v_ctx, d->wvc[l], d->vctx, d->mix, k, &held) &&
           ops_rms(d->kctx, INK2_MTP_HEAD, INK2_MTP_HEAD, d->k_norm_ctx[l], m->eps, k * (uint32_t)m->n_kv_heads,
                   d->kctx, INK2_MTP_HEAD, NULL) &&
           q4x_df_rope(d->kctx, k, (uint32_t)m->n_kv_heads, KV, d->pos, m->rope_base, NULL) &&
           ink2_mtp_kv_put(s->mk[l], s->mrows[l], d->kctx, k, KV, from + at, NULL) &&
           ink2_mtp_kv_put(s->mv[l], s->mrows[l], d->vctx, k, KV, from + at, NULL);
    }
  }
  int32_t overflow = 0;
  ok = ok && hipMemcpyAsync(s->mlast, taps + (uint64_t)(n - 1) * K * C, (uint64_t)K * C * 4, hipMemcpyDeviceToDevice,
                            NULL) == hipSuccess &&
       hipMemcpy(&overflow, d->overflow, 4, hipMemcpyDeviceToHost) == hipSuccess;
  if (!ok)
    return fail(err, "inkling2: a GPU call failed (the drafter's context)");
  if (overflow)
    return (void)hipMemset(d->overflow, 0, 4), fail(err, "inkling2: the drafter's context past f16's range");
  s->mend = to;
  return true;
}

/* a drafter layer over the block (B rows at positions pos .. ; d->x its input on the first, d->ob the last's MLP) */
static bool layer(Ink2Forward *f, const Ink2Seq *s, int l, uint32_t B, int32_t pos)
{
  const Ink2Mtp *m = f->mtp;
  const Ink2MtpLayer *L = &m->layer[l];
  Ink2MtpDev *d = &f->md;
  const uint32_t C = (uint32_t)m->n_embd, KV = kvw(m), Q = qw(m), H = (uint32_t)m->n_heads,
                 KH = (uint32_t)m->n_kv_heads, F = (uint32_t)m->ff;
  bool ok = true;
  if (l == 0) /* residual the input, normed */
    ok = hipMemcpyAsync(d->res, d->x, (uint64_t)B * C * 4, hipMemcpyDeviceToDevice, NULL) == hipSuccess;
  else /* residual += the last MLP's */
    ok = ink2_axpy(d->res, d->ob, (uint64_t)B * C, 1.0f, NULL);
  int32_t from = s->mfrom;
  if ((int64_t)pos - from > (int64_t)s->mrows[l]) /* the ring's newest */
    from = pos - (int32_t)s->mrows[l];
  const uint32_t window = m->local[l] ? (uint32_t)(m->window + m->block - 1) : 0;
  bool held = false, att = false, post = false, act = false;
  return ok && ops_rms(d->res, C, C, d->in_norm[l], m->eps, B, d->nrm, C, NULL) &&
         dmat(f, L->q, d->wq[l], d->q, NULL, NULL, NULL, d->nrm, B, &held) &&
         dmat(f, L->k, d->wk[l], d->kb, L->v, d->wv[l], d->vb, d->nrm, B, &held) &&
         ops_rms(d->q, INK2_MTP_HEAD, INK2_MTP_HEAD, d->q_norm[l], m->eps, B * H, d->q, INK2_MTP_HEAD, NULL) &&
         ops_rms(d->kb, INK2_MTP_HEAD, INK2_MTP_HEAD, d->k_norm[l], m->eps, B * KH, d->kb, INK2_MTP_HEAD, NULL) &&
         q4x_df_rope(d->q, B, H, Q, d->pos, m->rope_base, NULL) &&
         q4x_df_rope(d->kb, B, KH, KV, d->pos, m->rope_base, NULL) &&
         ink2_mtp_attn(d->q, B, pos, d->kb, d->vb, s->mk[l], s->mv[l], s->mrows[l], from, H, KH, window, d->scr, d->att,
                       NULL) &&
         dmat(f, L->o, d->wo[l], d->ob, NULL, NULL, NULL, d->att, B, &att) &&
         ink2_axpy(d->res, d->ob, (uint64_t)B * C, 1.0f, NULL) &&
         ops_rms(d->res, C, C, d->post_norm[l], m->eps, B, d->nrm, C, NULL) &&
         dmat(f, L->gate, d->wgate[l], d->g, L->up, d->wup[l], d->u, d->nrm, B, &post) &&
         q4x_swiglu(d->g, d->u, (uint64_t)B * F, d->g, NULL, NULL) &&
         dmat(f, L->down, d->wdown[l], d->ob, NULL, NULL, NULL, d->g, B, &act);
}

int ink2_seq_draft(Ink2Forward *f, Ink2Seq *s, int32_t y, uint32_t max, float conf_min, float cum_min, int32_t *drafts,
                   float *conf, char err[INK2_FWD_ERR])
{
  if (!f || !s || s->f != f || !f->mtp || (max && (!drafts || !conf)))
    return fail(err, "inkling2: no forward with a drafter, a sequence of another forward, or no room"), -1;
  if (s->failed)
    return fail(err, "inkling2: a sequence after a failed step: load it, or make another"), -1;
  if (y < 0 || y >= f->w->geo.n_vocab)
    return fail(err, "inkling2: a token not of the vocabulary"), -1;
  if (s->mend != s->pos || s->mfrom >= s->mend)
    return fail(err, "inkling2: the drafter's context not seeded to the sequence's position"), -1;
  const Ink2Mtp *m = f->mtp;
  Ink2MtpDev *d = &f->md;
  const uint32_t C = (uint32_t)m->n_embd, B = (uint32_t)m->block, R = (uint32_t)m->rank, K = (uint32_t)m->n_taps,
                 VR = (uint32_t)f->w->output->dims[1], V = (uint32_t)f->w->geo.n_vocab;
  const uint32_t steps = max < B - 1 ? max : B - 1;
  const int32_t pos = s->pos;
  bool held = false;
  /* the block: the anchor's normed embedding (the target's), then mask rows */
  bool ok = positions(f, pos, B) && hipMemcpy(d->prev, &y, 4, hipMemcpyHostToDevice) == hipSuccess &&
            ops_embed(f->w->embed->type, f->embed, C, d->prev, 1, 1, d->x, NULL) &&
            ops_rms(d->x, C, C, f->embed_norm, f->w->geo.rms_eps, 1, d->x, C, NULL);
  for (uint32_t r = 1; ok && r < B; r++)
    ok = hipMemcpyAsync(d->x + (uint64_t)r * C, d->mask, (uint64_t)C * 4, hipMemcpyDeviceToDevice, NULL) == hipSuccess;
  for (int l = 0; ok && l < m->n_layers; l++)
    ok = layer(f, s, l, B, pos);
  /* the final norm, the head on rows 1 .. (uncapped), the features: the last context position's taps mixed */
  ok = ok && ink2_axpy(d->res, d->ob, (uint64_t)B * C, 1.0f, NULL) &&
       ops_rms(d->res, C, C, d->norm, m->eps, B, d->nrm, C, NULL) &&
       dmat(f, f->w->output, f->output, d->base, NULL, NULL, NULL, d->nrm + C, B - 1, &held) &&
       ink2_mtp_mix(s->mlast, K, d->alpha + (size_t)m->n_layers * K, 1, C, d->f_rnn, NULL) &&
       ops_rms(d->f_rnn, C, C, d->m_dflare, m->eps, 1, d->f_rnn, C, NULL);
  /* the steps' invariant: rnn_proj's [hidden | features] columns on each row's (bias added in the step) */
  for (uint32_t r = 0; ok && r < B - 1; r++)
    ok = hipMemcpyAsync(d->hf + (uint64_t)r * 2 * C, d->nrm + (uint64_t)(r + 1) * C, (uint64_t)C * 4,
                        hipMemcpyDeviceToDevice, NULL) == hipSuccess &&
         hipMemcpyAsync(d->hf + (uint64_t)r * 2 * C + C, d->f_rnn, (uint64_t)C * 4, hipMemcpyDeviceToDevice, NULL) ==
             hipSuccess;
  ok = ok && mv_run(T_F32, d->w_hf, 3 * R, 2 * C, d->hf, B - 1 > MV_MAX_TOKENS ? MV_MAX_TOKENS : B - 1, d->inv, NULL) &&
       hipMemsetAsync(d->state, 0, R * 4, NULL) == hipSuccess;
  for (uint32_t st = 0; ok && st < steps; st++)
  {
    const int32_t *prev = st ? d->ids + st - 1 : d->prev;
    ok = ink2_mtp_markov_in(d->state, d->w1, prev, R, d->vec, NULL) &&
         mv_run(T_F32, d->w_sp, 3 * R, 2 * R, d->vec, 1, d->p, NULL) &&
         ink2_mtp_markov_gru(d->p, d->inv + (uint64_t)st * 3 * R, d->rnn_b, R, d->state, d->latent, NULL) &&
         mv_run(T_Q8_0, d->fold, VR, R, d->latent, 1, d->corr, NULL) &&
         ink2_mtp_markov_pick(d->base + (uint64_t)st * VR, d->corr, d->fold_b, f->w->geo.logit_softcap, V, d->lg,
                              d->ids + st, d->probs + st, NULL);
  }
  int32_t ids[INK2_MTP_BLOCK], overflow = 0;
  float p[INK2_MTP_BLOCK];
  ok = ok && hipMemcpy(ids, d->ids, steps * 4, hipMemcpyDeviceToHost) == hipSuccess &&
       hipMemcpy(p, d->probs, steps * 4, hipMemcpyDeviceToHost) == hipSuccess &&
       hipMemcpy(&overflow, d->overflow, 4, hipMemcpyDeviceToHost) == hipSuccess;
  if (!ok)
    return fail(err, "inkling2: a GPU call failed (a draft)"), -1;
  if (overflow)
    return (void)hipMemset(d->overflow, 0, 4), fail(err, "inkling2: a draft's activations past f16's range"), -1;
  int k = 0;
  float cum = 1.0f;
  for (; k < (int)steps; k++)
  {
    cum *= p[k];
    if (p[k] < conf_min || cum < cum_min)
      break;
    drafts[k] = ids[k], conf[k] = p[k];
  }
  return k;
}
