/* Qwen's forward made (osmos/q4x/forward.h): the build in order (hip_q4x_resident.c's parts), destroyed in reverse,
   its reads; a sequence's buffers made and freed */
#define _GNU_SOURCE /* O_DIRECT */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "osmos/gpu/upload.h"
#include "osmos/gpu/mm.h"
#include "osmos/mem/arena.h"
#include "osmos/mem/host_slots.h"
#include "osmos/mem/experts.h"
#include "osmos/mem/membrane.h"
#include "osmos/run/fit.h"
#include "q4x_impl.h"
#include "osmos/err.h"

#define F sizeof(float)
#define Q8_0 8 /* ggml's: the selector kernel's codebooks */

void q4x_forward_destroy(Q4xForward *f)
{
  if (!f)
    return;
  (void)hipDeviceSynchronize(); /* a failed step's kernels done: the experts' host memory is unmapped below */
  q4x_npu_destroy(f);            /* its imports of the buffers and weights first: freed below */
  membrane_destroy(f->mem); /* its readers stopped, then its memory */
  host_slots_destroy(f->exp_slots), host_slots_destroy(f->exp_stream);
  for (int i = 0; f->exp_fds && i < f->w->model->n_files; i++)
    if (f->exp_fds[i] >= 0)
      close(f->exp_fds[i]);
  free(f->exp_fds);
  free(f->h_ex), free(f->h_by), free(f->h_row), free(f->h_run), free(f->h_est), free(f->h_ids), free(f->h_gidx);
  free(f->h_hidx), free(f->h_cnt), free(f->h_xg), free(f->h_yg), free(f->h_xd), free(f->h_yd), free(f->h_ptr);
  q4x_free_table(f, f->w->model, f->dev);
  if (f->w->overlay)
    q4x_free_table(f, f->w->overlay, f->ov_dev);
  if (f->mtp)
    q4x_free_table(f, f->mtp->model, f->mtp_dev);
  if (f->df)
    q4x_free_table(f, f->df->model, f->df_dev);
  df_buffers_free(&f->d);
  free(f->b.slot);
  if (f->s2)
    (void)hipStreamDestroy(f->s2);
  if (f->e_in)
    (void)hipEventDestroy(f->e_in);
  if (f->e_out)
    (void)hipEventDestroy(f->e_out);
  if (f->pin_free)
    (void)hipEventDestroy(f->pin_free);
  if (f->e_mix)
    (void)hipEventDestroy(f->e_mix);
  if (f->e_inj)
    (void)hipEventDestroy(f->e_inj);
  if (f->e_gates)
    (void)hipEventDestroy(f->e_gates);
  if (f->e_ix)
    (void)hipEventDestroy(f->e_ix);
  if (f->ple_read)
    rowread_destroy(f->ple_read); /* (a pending step's rows waited for first) */
  if (f->ple_fd >= 0)
    close(f->ple_fd);
  free(f->ple_off);
  if (f->ple_copied)
    (void)hipEventDestroy(f->ple_copied);
  attn_core_destroy(f->attn);
  gpu_bufs_free(&f->bufs); /* every GPU and pinned buffer the forward made (the step's, the MTP head's, the experts'
                              table, the n-gram stage): from the record - the readers of the stage stopped above */
  free(f->sub_seq);
  free(f);
}

Q4xForward *q4x_forward_create(const Q4xWeights *w, const Q4xMtp *mtp, uint32_t max_ctx, uint32_t max_tokens,
                               char err[OSMOS_ERR])
{
  const Q4xOptions o = {max_ctx, max_tokens, false, NULL, 0, NULL, FIT_BUDGET_NOW, NULL, NULL, 0}; /* (fit now) */
  return q4x_forward_create_with(w, mtp, &o, err);
}

Q4xForward *q4x_forward_create_with(const Q4xWeights *w, const Q4xMtp *mtp, const Q4xOptions *o, char err[OSMOS_ERR])
{
  const uint32_t max_ctx = o ? o->max_ctx : 0, max_tokens = o ? o->max_tokens : 0;
  if (!w || !max_ctx || max_ctx > OPS_Q4X_MAX_KEYS || !max_tokens || max_tokens > max_ctx)
    return osmos_fail(err, "no weights, or a context past the dense attention's reach, or no tokens a step"), NULL;
  Q4xForward *f = calloc(1, sizeof *f);
  if (!f)
    return osmos_fail(err, "out of memory"), NULL;
  f->w = w, f->mtp = mtp, f->max_ctx = max_ctx, f->max_tokens = max_tokens, f->stream_ngram = o->stream_ngram;
  f->ledger = o->ledger;
  const Q4xDflash *df = o->dflash;
  /* a drafter's: its geometry these weights' (q4x_dflash_create's checks), its codebooks the selector's Q8_0 */
  const bool df_ok = !df || (df->geo.n_embd == w->geo.n_embd && df->geo.n_vocab == w->geo.n_vocab &&
                             df->sel_pred->type == Q8_0 && df->sel_succ->type == Q8_0 &&
                             (!o->n_taps || (o->tap_layers && o->n_taps == (uint32_t)df->geo.n_taps)));
  const int32_t *tl = df ? df->geo.taps : o->tap_layers; /* a drafter's taps: its own */
  const uint32_t nt = df ? (uint32_t)df->geo.n_taps : o->n_taps;
  bool taps_ok = df_ok && nt <= Q4X_MAX_TAPS && (!nt || tl);
  for (uint32_t i = 0; taps_ok && i < nt; i++)
    taps_ok = tl[i] >= 0 && tl[i] < w->geo.n_layers && (!i || tl[i] > tl[i - 1]) &&
              (!df || !o->n_taps || o->tap_layers[i] == tl[i]),
    f->tap_layer[i] = tl[i];
  f->n_taps = taps_ok ? nt : 0;
  f->df = taps_ok ? df : NULL;
  /* the window and the positions a rollback leaves seeded past the sequence (Q4X_SLOTS - 1 at most): theirs never a
     window's slot (a ring of the window alone: a rollback, then a shorter seed, reads dropped positions' keys as the
     window's oldest) */
  f->df_ring = df && (uint32_t)df->geo.window + Q4X_SLOTS < max_ctx ? (uint32_t)df->geo.window + Q4X_SLOTS : max_ctx;
  f->ple_fd = -1;
  if (hipStreamCreateWithFlags(&f->s2, hipStreamNonBlocking) != hipSuccess ||
      hipEventCreateWithFlags(&f->e_in, hipEventDisableTiming) != hipSuccess ||
      hipEventCreateWithFlags(&f->e_out, hipEventDisableTiming) != hipSuccess ||
      hipEventCreateWithFlags(&f->pin_free, hipEventDisableTiming) != hipSuccess ||
      hipEventCreateWithFlags(&f->e_mix, hipEventDisableTiming) != hipSuccess ||
      hipEventCreateWithFlags(&f->e_inj, hipEventDisableTiming) != hipSuccess ||
      hipEventCreateWithFlags(&f->e_gates, hipEventDisableTiming) != hipSuccess ||
      hipEventCreateWithFlags(&f->e_ix, hipEventDisableTiming) != hipSuccess)
    return osmos_fail(err, "a stream or event refused"), q4x_forward_destroy(f), NULL;
  if (!df_ok)
    return osmos_fail(err, "a drafter not of these weights, its codebooks not Q8_0, or its taps not tap_layers"),
           q4x_forward_destroy(f), NULL;
  if (!taps_ok)
    return osmos_fail(err, "tap layers past the model's, out of order, or more than Q4X_MAX_TAPS"), q4x_forward_destroy(f), NULL;
  if (!(f->sub_seq = calloc(Q4X_MAX_RUNS, sizeof *f->sub_seq)))
    return osmos_fail(err, "out of memory"), q4x_forward_destroy(f), NULL;
  if (!q4x_rope((uint32_t)w->geo.rope_dim, (uint32_t)w->geo.rope_dim / 2, w->geo.rope_base, w->geo.native_ctx, max_ctx, Q4X_THETA_GGML, &f->rope))
    return osmos_fail(err, "no rotation for this context (q4x_rope: past 64 times the native)"), q4x_forward_destroy(f),
           NULL;
  const GgufTensor *skip = f->stream_ngram ? w->ple_table : NULL;
  /* the weights' bytes - every file's tensors (but an overlay's stand-ins' and a streamed table) - against the
     memory left, before any is read: a forward that cannot fit takes nothing */
  f->exp_units = q4x_weights_experts(w, NULL, NULL, 0) > 0;
  uint64_t weights = 0, experts_bytes = 0; /* (the experts the membrane's, sized after: by the budget, or by the room
                                               the build leaves - not counted here, or a machine short of the whole
                                               model is refused rather than streamed) */
  for (int l = 0; f->exp_units && l < w->geo.n_layers; l++)
    experts_bytes += w->layer[l].exp_gate->bytes + w->layer[l].exp_up->bytes + w->layer[l].exp_down->bytes;
  for (size_t i = 0; i < w->model->n_tensors; i++)
    if (&w->model->tensors[i] != skip && !(w->overlay && gguf_tensor(w->overlay, w->model->tensors[i].name)))
      weights += w->model->tensors[i].bytes;
  const Gguf *more[] = {w->overlay, mtp ? mtp->model : NULL, f->df ? f->df->model : NULL};
  for (size_t k = 0; k < sizeof more / sizeof *more; k++)
    for (size_t i = 0; more[k] && i < more[k]->n_tensors; i++)
      weights += more[k]->tensors[i].bytes;
  if (!q4x_room_left_past(f, f->exp_units ? weights - experts_bytes : weights, "the forward", err))
    return q4x_forward_destroy(f), NULL;
  if (!q4x_upload(f->ledger, f, w->model, w->overlay, skip, &f->dev, err) ||
      (w->overlay && !q4x_upload(f->ledger, NULL, w->overlay, NULL, NULL, &f->ov_dev, err)) || !q4x_buffers(f, err) ||
      (f->stream_ngram && !q4x_ple_stream(f, err)) ||
      (mtp && (!q4x_upload(f->ledger, NULL, mtp->model, NULL, NULL, &f->mtp_dev, err) || !q4x_mtp_buffers(f, err))) ||
      (f->df && (!q4x_upload(f->ledger, NULL, f->df->model, NULL, NULL, &f->df_dev, err) || !q4x_df_buffers(f, err))))
    return q4x_forward_destroy(f), NULL;
  /* the fixed part, counted (experts_plan's): the tables read, the records' buffers - not the GPU's free memory
     before and after, which anything else moves */
  f->fixed_bytes = gpu_table_bytes(w->model, f->dev, NULL) + gpu_table_bytes(w->overlay, f->ov_dev, NULL) +
                   (mtp ? gpu_table_bytes(mtp->model, f->mtp_dev, NULL) : 0) +
                   (f->df ? gpu_table_bytes(f->df->model, f->df_dev, NULL) + f->d.bufs.bytes : 0) + f->bufs.bytes;
  const uint64_t budget = o->budget == FIT_BUDGET_NOW ? experts_fit_now(f->ledger, f->fixed_bytes) : o->budget;
  if (!q4x_experts(f, budget, err) ||
      !q4x_room_left(f, "the forward", err) || (o->npu && !q4x_npu_make(f, o->npu, o->npu_share, err))) /* (whole) */
    return q4x_forward_destroy(f), NULL;
  return f;
}

uint64_t q4x_forward_bytes(const Q4xForward *f) { return f ? f->fixed_bytes + f->pinned_bytes : 0; }

bool q4x_forward_whole(const Q4xForward *f) { return f && f->mem && membrane_whole(f->mem); }

void q4x_forward_side_spin(Q4xForward *f, uint32_t us)
{
  if (f)
    f->side_spin = us;
}

void q4x_forward_probe(Q4xForward *f, q4x_probe_fn fn, void *arg)
{
  if (f)
    f->probe = fn, f->probe_arg = arg;
}

Q4xSeqBytes q4x_seq_bytes(const Q4xGeometry *g)
{
  const uint64_t ch = 2ull * g->ssm_groups * g->ssm_state + (uint64_t)g->ssm_inner;
  return (Q4xSeqBytes){(uint64_t)g->ssm_v_heads * g->ssm_state * g->ssm_state * F, (g->ssm_conv - 1) * ch * F,
                       q4x_kv_row_bytes((uint32_t)(g->n_kv_heads * g->head_dim)),
                       (uint64_t)(g->ple_conv - 1) * g->ple_ngram * g->hc * g->n_embd * F, Q4X_IX_DIM * F,
                       (uint64_t)Q4X_IX_RING(g->ix_ratio) * Q4X_IX_DIM * F};
}

void q4x_seq_destroy(Q4xSeq *s)
{
  if (!s)
    return;
  gpu_bufs_free(&s->bufs); /* every GPU buffer it made: from its record */
  free(s->step_tok);
  free(s);
}

Q4xSeq *q4x_seq_create(Q4xForward *f, char err[OSMOS_ERR])
{
  if (!f)
    return osmos_fail(err, "no forward"), NULL;
  const Q4xGeometry *g = &f->w->geo;
  Q4xSeq *s = calloc(1, sizeof *s);
  if (!s)
    return osmos_fail(err, "out of memory"), NULL;
  s->f = f;
  const Q4xSeqBytes sb = q4x_seq_bytes(g);
  const uint64_t pools = f->max_ctx / (uint64_t)g->ix_ratio + 1; /* an attention's pooled keys: its blocks */
  bool ok = q4x_seq_alloc(s, &s->ple_hist, Q4X_SLOTS * sb.ple);
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = g->recurrent[l] ? q4x_seq_alloc(s, &s->ssm[l], Q4X_SLOTS * sb.ssm) && q4x_seq_alloc(s, &s->conv[l], Q4X_SLOTS * sb.conv)
                         : q4x_seq_alloc(s, &s->kc[l], f->max_ctx * sb.kv_row) && q4x_seq_alloc(s, &s->vc[l], f->max_ctx * sb.kv_row) &&
                               q4x_seq_alloc(s, &s->ixk[l], pools * sb.ix_block) && q4x_seq_alloc(s, &s->ixr[l], sb.ix_ring);
  if (ok && f->mtp) /* the head's cache and indexer, its pending h, its last step's tokens */
    ok = q4x_seq_alloc(s, &s->mkc, f->max_ctx * sb.kv_row) && q4x_seq_alloc(s, &s->mvc, f->max_ctx * sb.kv_row) &&
         q4x_seq_alloc(s, &s->mixk, pools * sb.ix_block) && q4x_seq_alloc(s, &s->mixr, sb.ix_ring) &&
         q4x_seq_alloc(s, &s->pend, (uint64_t)g->hc * g->n_embd * F) && (s->step_tok = calloc(f->max_tokens, 4));
  const uint64_t ring = f->df ? (uint64_t)f->df_ring * f->df->geo.n_kv_heads * f->df->geo.head_dim * F : 0;
  for (int l = 0; ok && f->df && l < f->df->geo.n_layers; l++) /* the drafter's rings */
    ok = q4x_seq_alloc(s, &s->dk[l], ring) && q4x_seq_alloc(s, &s->dv[l], ring);
  if (!ok || !q4x_seq_reset(s))
    return q4x_seq_destroy(s), osmos_fail(err, "GPU memory refused (a sequence's state)"), NULL;
  if (!q4x_room_left(s->f, "the sequence", err))
    return q4x_seq_destroy(s), NULL;
  return s;
}

int32_t q4x_seq_position(const Q4xSeq *s) { return s ? s->pos : -1; }
