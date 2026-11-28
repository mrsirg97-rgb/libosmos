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
#include "osmos/gpu/mm.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/kept.h"
#include "osmos/gpu/ops.h"
#include "osmos/model/qtype.h"
#include "osmos/gpu/q35_ops.h"
#include "lfm3_impl.h"
#include "osmos/err.h"

/* a read's matrix on mm: its table's types (mm_known), and BF16 where the model's matrices are (f->bf16_mm: mm takes
   BF16 by name - its f16 the value's where f16 holds it, a read's sums not the matvec's bits; osmos1.7b reads apart) */
static bool mm_takes(const Lfm3Forward *f, const GgufTensor *t, uint32_t rows, uint32_t cols)
{
  return (mm_known(t->type) || (t->type == QTYPE_BF16 && f->bf16_mm)) && mm_fits(t->type, rows, cols);
}

/* y = W x: a decode's the matvec (mv_strided: a token's sums its own), a read's the matrix cores where mm takes the
   type and shape (mm_tokens, f16 x: a token's sums its own), else the matvec */
static bool mat(Lfm3Forward *f, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t n, float *y,
                bool read)
{
  if (read && mm_takes(f, t, rows, cols))
    return mm_tokens(t->type, lfm3_dev(f, t), rows, y, rows, NULL, 0, NULL, 0, cols, x, cols, n, f->x16, f->overflow,
                     false, f->st);
  return mv_strided(t->type, lfm3_dev(f, t), rows, cols, x, cols, n, y, rows, f->st);
}

/* x's f16 once for the matrices on it (mm_x16_whole): true when x16 holds all n rows (a read on the matrix cores) -
   then mat16's matrices read it, not each converting x again; false (not a read, or refused): each converts its own */
static bool x16_once(Lfm3Forward *f, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t n,
                     bool read)
{
  return read && mm_takes(f, t, rows, cols) && n <= f->max_tokens &&
         mm_x16_whole(x, cols, n, cols, f->x16, f->overflow, f->st);
}

/* mat on x, its f16 already whole in x16 (have16: x16_once's) where the matrix is mm's - the same bits */
static bool mat16(Lfm3Forward *f, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t n,
                  float *y, bool read, bool have16)
{
  if (have16 && read && mm_takes(f, t, rows, cols))
    return mm_tokens_x16(t->type, lfm3_dev(f, t), rows, y, rows, NULL, 0, NULL, 0, cols, f->x16, n, f->st);
  return mat(f, t, rows, cols, x, n, y, read);
}

/* the FFN's gate and up on xn, its f16 once */
static bool ffn_in(Lfm3Forward *f, const Lfm3Layer *L, uint32_t ff, uint32_t D, uint32_t n, float *ga, float *up,
                   bool read)
{
  const bool xn16 = x16_once(f, L->ffn_gate, ff, D, f->xn, n, read);
  return mat16(f, L->ffn_gate, ff, D, f->xn, n, ga, read, xn16) && mat16(f, L->ffn_up, ff, D, f->xn, n, up, read, xn16);
}

static bool conv(Lfm3Forward *f, const Lfm3Run *run, uint32_t runs, int l, uint32_t n, bool read)
{
  const Lfm3Geometry *g = &f->w->geo;
  const Lfm3Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd;
  bool ok = mat(f, L->in_proj, 3 * D, D, f->xn, n, f->bcx, read);
  for (uint32_t r = 0; ok && r < runs; r++)
    ok = lfm3_shortconv(f->bcx + (uint64_t)run[r].off * 3 * D, run[r].len, D, (uint32_t)g->conv_taps,
                        lfm3_fv(f, L->conv), run[r].s->conv[l], LFM3_CONV_SLOTS, run[r].s->cur,
                        f->u + (uint64_t)run[r].off * D, f->st);
  return ok && mat(f, L->out_proj, D, D, f->u, n, f->y, read);
}

/* a run's K and V rows into its cache: a full layer's at their positions, a sliding layer's ring at pos % ring - a
   step's rows past the ring's end wrapped to its start */
static bool kv_put(Lfm3Forward *f, const Lfm3Run *r, const float *vf, int l, bool sliding, uint32_t kv)
{
  const uint64_t row = q4x_kv_row_bytes(kv);
  const float *k = f->kr + (uint64_t)r->off * kv, *v = vf + (uint64_t)r->off * kv;
  uint8_t *kc = r->s->kc[l], *vc = r->s->vc[l];
  const uint32_t p = (uint32_t)r->s->pos, at = sliding ? p % f->ring : p;
  const uint32_t first = sliding && at + r->len > f->ring ? f->ring - at : r->len;
  bool ok = q4x_kv_put(k, v, first, kv, kc + at * row, vc + at * row, f->st);
  if (ok && first < r->len)
    ok = q4x_kv_put(k + (uint64_t)first * kv, v + (uint64_t)first * kv, r->len - first, kv, kc, vc, f->st);
  return ok;
}

/* a full layer's blocks closed up to `want` (each its 512 keys indexed): its index vectors to the host, Ward there
   (src/lsa's lsa_block), its buckets back - every full layer's at once (the count the sequence's) */
static bool close_blocks(Lfm3Forward *f, Lfm3Seq *s, int l, uint32_t want, uint32_t from, bool defer)
{
  /* defer: the step reads none of them (every read below sparse_from) - the closer's (osmos/gpu/lsa_close.h), the
     step's GPU never waiting on the host; a batch it refused closed here as before */
  /* blocks from .. want: up to close_max a batch - their keys' index to the host in one copy and one wait, each
     bucketed (lsa_block, in block order: the same buckets), then their centroids and members back in one copy each.
     a block a wait was 30 (a 3K prompt) to 48 (a 4,096-token step) drains of the GPU's queue a step */
  for (uint32_t b0 = from; b0 < want; b0 += f->close_max)
  {
    const uint32_t k = want - b0 < f->close_max ? want - b0 : f->close_max;
    if (defer && lsa_closer_queue(f->closer, s->ik[l] + (uint64_t)b0 * LSA_WINDOW * LSA_DIM, b0, k,
                                  s->cent[l] + (uint64_t)b0 * LSA_BUCKETS * LSA_DIM,
                                  s->mem[l] + (uint64_t)b0 * LSA_BUCKETS * LSA_CAP, &s->broken, f->st))
      continue;
    if (hipMemcpyAsync(f->h_ik, s->ik[l] + (uint64_t)b0 * LSA_WINDOW * LSA_DIM, (uint64_t)k * LSA_WINDOW * LSA_DIM * F,
                       hipMemcpyDeviceToHost, f->st) != hipSuccess ||
        hipStreamSynchronize((hipStream_t)f->st) != hipSuccess)
      return false;
    for (uint32_t i = 0; i < k; i++)
      if (!lsa_block(f->h_ik + (uint64_t)i * LSA_WINDOW * LSA_DIM, b0 + i, f->h_cent + (uint64_t)i * LSA_BUCKETS * LSA_DIM,
                     f->h_mem + (uint64_t)i * LSA_BUCKETS * LSA_CAP))
        return false;
    if (hipMemcpy(s->cent[l] + (uint64_t)b0 * LSA_BUCKETS * LSA_DIM, f->h_cent, (uint64_t)k * LSA_BUCKETS * LSA_DIM * F,
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(s->mem[l] + (uint64_t)b0 * LSA_BUCKETS * LSA_CAP, f->h_mem, (uint64_t)k * LSA_BUCKETS * LSA_CAP * 4,
                  hipMemcpyHostToDevice) != hipSuccess)
      return false;
  }
  return true;
}

/* LSA on a full layer's run: its keys indexed (the zero tail's dims, post-norm; the rotation leaves them), the blocks
   they close bucketed; a decode's tokens each its read (the blocks closed before its window) and its attention over
   the read alone; a prompt's tokens from sparse_from the same reads in a batch (F32 scores) and their attention on
   the matrix cores over them, those before dense (*dense: the run's first tokens left to the dense attention) */
static bool lsa_attention(Lfm3Forward *f, const Lfm3Run *R, int l, const float *gt, bool read, uint32_t *dense)
{
  const Lfm3Geometry *g = &f->w->geo;
  const uint32_t H = (uint32_t)g->n_heads, KV = (uint32_t)g->n_kv_heads, hd = (uint32_t)g->head_dim;
  const uint32_t q = H * hd, kv = KV * hd, P = (uint32_t)R->s->pos, n = R->len;
  const LsaDims fd = lfm3_free_dims(g);
  Lfm3Seq *s = R->s;
  *dense = n;
  if (!lsa_index_gpu(f->kr + (uint64_t)R->off * kv, n, kv, KV, 1, hd, &fd, f->wk[l], true,
                     s->ik[l] + (uint64_t)P * LSA_DIM, f->st))
    return false;
  if (read)
  {
    const uint32_t d = P >= f->sparse_from ? 0 : f->sparse_from - P < n ? f->sparse_from - P : n, m = n - d;
    if (!close_blocks(f, s, l, (P + n) / LSA_WINDOW, s->closed, !m)) /* (none of them read: the closer's) */
      return false;
    *dense = d;
    if (!m)
      return true;
    const uint64_t at = (uint64_t)R->off + d;
    const AttnStep st = {.q = f->qr + at * q,
                         .n_tok = m,
                         .pos0 = P + d,
                         .keys = {.k = s->kc[l], .v = s->vc[l], .n_keys = P + n, .lists = f->lists, .counts = f->counts,
                                  .stride = LSA_READ_MAX},
                         .gate = gt + at * H,
                         .out = f->att + at * q,
                         .read = true};
    return lsa_index_gpu(f->qr + at * q, m, q, KV, H / KV, hd, &fd, f->wq[l], false, f->iqb, f->st) &&
           lsa_read_batch_gpu(f->iqb, m, P + d, s->cent[l], s->mem[l], (P + n) / LSA_WINDOW, f->lsa_sc, f->sub_n,
                              f->max_buckets, f->lists, f->counts, f->st) &&
           attn_core_run(f->attn[0], &st, f->st);
  }
  if (P + n <= f->sparse_from) /* a decode below sparse_from: the dense attention's, as a prompt's rows there (its
                                  blocks still closed: the reads past it) - LSA's read every step cost 6% of decode */
    return close_blocks(f, s, l, (P + n) / LSA_WINDOW, s->closed, true);
  uint32_t closed = s->closed;
  for (uint32_t t = 0; t < n; t++)
  {
    const uint64_t at = (uint64_t)R->off + t;
    const uint32_t want = (P + t) / LSA_WINDOW;
    if (!close_blocks(f, s, l, want, closed, false) ||
        !lsa_index_gpu(f->qr + at * q, 1, q, KV, H / KV, hd, &fd, f->wq[l], false, f->iq, f->st) ||
        !lsa_read_gpu(f->iq, P + t, s->cent[l], s->mem[l], want, f->lsa_s, f->list, f->list_n, f->st) ||
        !attn_core_run(f->attn[0],
                       &(AttnStep){.q = f->qr + at * q,
                                   .n_tok = 1,
                                   .pos0 = P + t,
                                   .keys = {.k = s->kc[l], .v = s->vc[l], .n_keys = P + t + 1, .lists = f->list,
                                            .counts = (const int32_t *)f->list_n, .stride = LSA_READ_MAX},
                                   .gate = gt + at * H,
                                   .out = f->att + at * q},
                       f->st))
      return false;
    closed = want > closed ? want : closed;
  }
  *dense = 0;
  return close_blocks(f, s, l, (P + n) / LSA_WINDOW, closed, true);
}

static bool attention(Lfm3Forward *f, const Lfm3Run *run, uint32_t runs, int l, uint32_t n, bool read)
{
  const Lfm3Geometry *g = &f->w->geo;
  const Lfm3Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, H = (uint32_t)g->n_heads, KV = (uint32_t)g->n_kv_heads;
  const uint32_t hd = (uint32_t)g->head_dim, q = H * hd, kv = KV * hd;
  const bool sliding = g->kind[l] == LFM3_SLIDING;
  /* q, k, v and the gate on xn: a token's one joint launch (their outputs in jo, one after another), else each */
  const bool joint = n == 1 && !read && f->j_attn[l];
  float *qf = joint ? f->jo : f->qf, *kf = joint ? f->jo + q : f->kf, *vf = joint ? f->jo + q + kv : f->vf;
  float *gt = joint ? f->jo + q + 2 * kv : f->g;
  const bool xn16 = !joint && x16_once(f, L->q, q, D, f->xn, n, read); /* (q, k, v and the gate on one f16) */
  bool ok = joint ? mv_strided(L->q->type, f->j_attn[l], q + 2 * kv + H, D, f->xn, D, 1, f->jo, q + 2 * kv + H, f->st)
                  : mat16(f, L->q, q, D, f->xn, n, qf, read, xn16) && mat16(f, L->k, kv, D, f->xn, n, kf, read, xn16) &&
                        mat16(f, L->v, kv, D, f->xn, n, vf, read, xn16) && mat16(f, L->gate, H, D, f->xn, n, gt, read, xn16);
  for (uint32_t r = 0; ok && r < runs; r++)
  {
    const Lfm3Run *R = &run[r];
    const uint64_t o = R->off;
    ok = q4x_qk(qf + o * q, kf + o * kv, R->len, H, KV, hd, sliding ? &f->rope_swa : &f->rope_full[R->s->stage],
                lfm3_fv(f, L->q_norm), lfm3_fv(f, L->k_norm), g->rms_eps, R->s->pos, f->qr + o * q, NULL,
                f->kr + o * kv, f->st) &&
         kv_put(f, R, vf, l, sliding, kv);
    uint32_t dense = R->len; /* the run's first tokens the dense attention's (LSA's reads the rest) */
    if (ok && f->lsa && g->kind[l] == LFM3_FULL)
      ok = lsa_attention(f, R, l, gt, read, &dense);
    if (!dense)
      continue;
    /* the layer kind's core: a prompt's (a read) the matrix cores' over its dense rows, a decode's flash decoding */
    const AttnStep st = {.q = f->qr + o * q,
                         .n_tok = read ? dense : R->len,
                         .pos0 = (uint32_t)R->s->pos,
                         .keys = {.k = R->s->kc[l], .v = R->s->vc[l], .n_keys = sliding ? f->ring : f->max_ctx},
                         .gate = gt + o * H,
                         .out = f->att + o * q,
                         .read = read};
    ok = ok && attn_core_run(f->attn[sliding], &st, f->st);
  }
  return ok && mat(f, L->out, D, q, f->att, n, f->y, read);
}

/* a layer: its operator and its SwiGLU, each on xn (the norm before it, already made) and each one launch after into
   the residual with the next norm (ops_rms_add_rms: the post-norm, the add, the next pre-norm - the FFN's, then the
   next layer's attention's; the last layer's none) */
static bool layer(Lfm3Forward *f, const Lfm3Run *run, uint32_t runs, int l, uint32_t n, bool read)
{
  const Lfm3Geometry *g = &f->w->geo;
  const Lfm3Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, ff = (uint32_t)g->n_ff;
  const float *next = l + 1 < g->n_layers ? lfm3_fv(f, f->w->layer[l + 1].attn_norm) : NULL;
  const bool joint = n == 1 && !read && f->j_ffn[l];
  float *ga = joint ? f->jo : f->ga, *up = joint ? f->jo + ff : f->up;
  return (g->kind[l] == LFM3_CONV ? conv(f, run, runs, l, n, read) : attention(f, run, runs, l, n, read)) &&
         ops_rms_add_rms(f->h, f->y, lfm3_fv(f, L->post_attn_norm), lfm3_fv(f, L->ffn_norm), g->rms_eps, D, n, f->xn,
                         f->st) &&
         (joint ? mv_strided(L->ffn_gate->type, f->j_ffn[l], 2 * ff, D, f->xn, D, 1, f->jo, 2 * ff, f->st)
                : ffn_in(f, L, ff, D, n, ga, up, read)) &&
         ops_swiglu(ga, up, ff, INFINITY, NULL, n, f->act, f->st) && mat(f, L->ffn_down, D, ff, f->act, n, f->y, read) &&
         ops_rms_add_rms(f->h, f->y, lfm3_fv(f, L->post_ffn_norm), next, g->rms_eps, D, n, f->xn, f->st);
}

/* the step (lfm3_step_runs); keep_from: rows below it not kept whatever the prompts say (lfm3_step's: its last rows) */
static bool step(Lfm3Forward *f, Lfm3Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                 bool read, uint32_t keep_from, bool honor_prompt, char *err)
{
  if (!f || !seqs || !lens || !tokens || !runs || runs > LFM3_MAX_RUNS)
    return osmos_fail(err, "no forward, sequences or tokens, or past LFM3_MAX_RUNS runs");
  if (!overflow_settle(&f->ovf, err)) /* (a reset the last step owes: refused with its cause while it fails) */
    return false;
  (void)lsa_closer_join(f->closer); /* every bucket an earlier step queued in place (a failed one: its sequence
                                       broken - refused below) */
  f->rows = 0; /* no row kept until the step runs: a refused one leaves none, not the step before's through a
                  half-rewritten slot table */
  Lfm3Run run[LFM3_MAX_RUNS];
  uint32_t n = 0, kept = 0;
  for (uint32_t r = 0; r < runs; r++)
  {
    Lfm3Seq *s = seqs[r];
    if (!s || s->f != f || !lens[r])
      return osmos_fail(err, "a run of no sequence, of another forward, or of no tokens");
    if (s->broken)
      return osmos_fail(err, "a sequence after a failed step: load it, or make another");
    for (uint32_t k = 0; k < r; k++)
      if (seqs[k] == s)
        return osmos_fail(err, "a sequence twice in one step");
    if ((uint64_t)n + lens[r] > f->max_tokens || (uint64_t)s->pos + lens[r] > f->max_ctx)
      return osmos_fail(err, "the tokens past the forward's step, or a sequence past its context");
    run[r] = (Lfm3Run){s, n, lens[r]};
    for (uint32_t t = 0; t < lens[r]; t++)
    {
      const bool keep = n + t >= keep_from && (!honor_prompt || !s->prompt_end || s->pos + t + 1 >= s->prompt_end);
      f->slot[n + t] = keep ? (int32_t)kept : -1;
      kept += keep;
    }
    n += lens[r];
  }
  if (kept > f->logit_rows)
    return osmos_fail(err, "a step's kept rows past the logits' (Lfm3Options' logit_rows)");
  const Lfm3Geometry *g = &f->w->geo;
  uint32_t stage[LFM3_MAX_RUNS]; /* each run's stage after the step: the least whose span holds its positions */
  for (uint32_t r = 0; r < runs; r++)
  {
    const uint64_t end = (uint64_t)run[r].s->pos + run[r].len;
    stage[r] = run[r].s->stage;
    while (stage[r] + 1 < f->stages && end > f->stage[stage[r] + 1].from)
      stage[r]++;
  }
  for (uint32_t i = 0; i < n; i++)
    if (tokens[i] < 0 || tokens[i] >= g->n_vocab)
      return osmos_fail(err, "a token outside the vocabulary");
  const uint32_t D = (uint32_t)g->n_embd, V = (uint32_t)g->n_vocab;
  bool ok = true;
  for (uint32_t r = 0; r < runs; r++) /* a crossing: the full layers' cached keys turned to the new stage first */
  {
    Lfm3Seq *s = run[r].s;
    for (int l = 0; ok && stage[r] != s->stage && l < g->n_layers; l++)
      if (g->kind[l] == LFM3_FULL)
        ok = lfm3_kv_turn(s->kc[l], (uint32_t)s->pos, (uint32_t)g->n_kv_heads, (uint32_t)g->head_dim,
                          &f->rope_full[s->stage], &f->rope_full[stage[r]], f->st);
    if (stage[r] != s->stage)
      s->stage = stage[r], s->turned = (uint32_t)s->pos, s->back = 0;
  }
  memcpy(f->pin_tok, tokens, (size_t)n * 4);
  ok = ok && hipMemcpyAsync(f->tok, f->pin_tok, (size_t)n * 4, hipMemcpyHostToDevice, f->st) == hipSuccess &&
            ops_embed(f->w->embd->type, lfm3_dev(f, f->w->embd), D, f->tok, n, 1, f->h, f->st) &&
            ops_rms(f->h, D, D, lfm3_fv(f, f->w->layer[0].attn_norm), g->rms_eps, n, f->xn, D, f->st);
  const uint64_t T = f->df ? (uint64_t)f->df->geo.n_taps * D : 0; /* a token's taps (the drafter's) */
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    ok = layer(f, run, runs, l, n, read);
    if (ok && f->df && f->tap_of[l] >= 0) /* the drafter's feature: the layer's output, every row of the step */
      ok = hipMemcpy2DAsync(f->taps + (uint64_t)f->tap_of[l] * D, T * F, f->h, (uint64_t)D * F, (uint64_t)D * F, n,
                            hipMemcpyDeviceToDevice, (hipStream_t)f->st) == hipSuccess;
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
         mv_strided(f->head_type, f->head, V, D, f->xf, D, kept, f->logits, V, f->st);
  /* f->overflow is only written by mm_x16 during reads: skip its D2H on decode steps */
  ok = ok && (!read || overflow_read(&f->ovf, f->st)) && hipStreamSynchronize((hipStream_t)f->st) == hipSuccess;
  const int32_t over = ok && read ? *f->pin_overflow : 0;
  f->rows = ok && !over ? n : 0;
  if (!ok || over)
  {
    for (uint32_t r = 0; r < runs; r++)
      run[r].s->broken = true;
    if (read)
      { char rerr_[OSMOS_ERR] = ""; overflow_reset(&f->ovf, rerr_); } /* (failing: owed, the next step says so) */
    return osmos_fail(err, "%s", over ? "an activation past f16's range in a read (the step refused)"
                                       : "a GPU call failed in the step (its sequences refused until a load)");
  }
  f->steps++;
  for (uint32_t r = 0; r < runs; r++)
  {
    Lfm3Seq *s = run[r].s;
    s->step_pos0 = s->pos, s->step_t0 = run[r].off, s->step_n = run[r].len, s->step_id = f->steps; /* its taps' rows */
    s->pos += (int32_t)run[r].len, s->cur = (s->cur + run[r].len) % LFM3_CONV_SLOTS;
    if (f->lsa)
      s->closed = (uint32_t)s->pos / LSA_WINDOW; /* every full layer's blocks closed to here (lsa_attention's) */
    s->back = s->back + run[r].len < LFM3_MAX_BACK ? s->back + run[r].len : LFM3_MAX_BACK;
  }
  return true;
}

bool lfm3_step_runs(Lfm3Forward *f, Lfm3Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                    char err[OSMOS_ERR])
{
  return step(f, seqs, lens, runs, tokens, false, 0, true, err);
}

bool lfm3_step_read(Lfm3Forward *f, Lfm3Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                    char err[OSMOS_ERR])
{
  return step(f, seqs, lens, runs, tokens, true, 0, true, err);
}

bool lfm3_step(Lfm3Forward *f, Lfm3Seq *s, const int32_t *tokens, uint32_t n, uint32_t rows, char err[OSMOS_ERR])
{
  if (!f || !s || !n || !rows || rows > n || rows > f->logit_rows)
    return osmos_fail(err, "a step of no tokens, or more rows than tokens or the logits'");
  if (!step(f, &s, &n, 1, tokens, n > LFM3_MV_TOKENS, n - rows, false, err))
    return false;
  const uint32_t V = (uint32_t)f->w->geo.n_vocab;
  return hipMemcpy(f->pin_logits, f->logits, (uint64_t)rows * V * F, hipMemcpyDeviceToHost) == hipSuccess ||
         osmos_fail(err, "a GPU call failed (the logits' copy)");
}

const float *lfm3_logits(const Lfm3Forward *f) { return f ? f->pin_logits : NULL; }

void lfm3_kept(const Lfm3Forward *f, KeptView *k)
{
  *k = (KeptView){0};
  if (f)
    *k = (KeptView){f->slot, f->rows, f->logits, (uint32_t)f->w->geo.n_vocab, f->ids, f->argmax, f->st, NULL, NULL, NULL};
}

const float *lfm3_row_logits(const Lfm3Forward *f, uint32_t row)
{
  KeptView k;
  lfm3_kept(f, &k);
  return kept_row(&k, row);
}

bool lfm3_greedy(Lfm3Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR])
{
  KeptView k;
  lfm3_kept(f, &k);
  const char *why = kept_greedy(&k, row0, rows, ids);
  return !why || osmos_fail(err, "%s", why);
}
