/* the lfm3 forward's making (osmos/lfm3/forward.h): every tensor to the GPU, the ropes, a step's buffers; a sequence's
   state (its convolutions' rings, its attention layers' caches) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/q35_ops.h"
#include "osmos/gpu/upload.h"
#include "lfm3_impl.h"

#define READ_PIECE (64ull << 20)

bool lfm3_fail(char *err, const char *why)
{
  if (err)
    snprintf(err, LFM3_ERR, "%s", why);
  return false;
}

bool lfm3_alloc(void *p, uint64_t bytes) { return hipMalloc((void **)p, bytes ? bytes : 1) == hipSuccess; }

static bool upload(Lfm3Forward *f, char *err)
{
  const Gguf *g = f->w->model;
  void *pin = NULL;
  if (!(f->dev = calloc(g->n_tensors, sizeof *f->dev)) || hipHostMalloc(&pin, READ_PIECE, 0) != hipSuccess)
    return lfm3_fail(err, "out of memory (the weights' table, or a pinned piece)");
  bool ok = true;
  for (size_t i = 0; ok && i < g->n_tensors; i++)
  {
    const char *why = gpu_read_tensor(g, &g->tensors[i], pin, READ_PIECE, &f->dev[i]);
    if ((ok = !why) == false && err)
      snprintf(err, LFM3_ERR, "%s: not read into GPU memory (%s)", g->tensors[i].name, why);
  }
  hipHostFree(pin);
  return ok;
}

void lfm3_forward_destroy(Lfm3Forward *f)
{
  if (!f)
    return;
  if (f->st)
    hipStreamSynchronize((hipStream_t)f->st), hipStreamDestroy((hipStream_t)f->st);
  for (size_t i = 0; f->dev && i < f->w->model->n_tensors; i++)
    hipFree(f->dev[i]);
  free(f->dev);
  void *b[] = {f->tok, f->h, f->xn, f->y, f->bcx, f->u, f->qf, f->kf, f->vf, f->qr, f->kr, f->g, f->att, f->ga,
               f->up, f->act, f->attn_scratch, f->xf, f->logits};
  for (size_t i = 0; i < sizeof b / sizeof *b; i++)
    hipFree(b[i]);
  hipHostFree(f->pin_tok), hipHostFree(f->pin_logits);
  free(f);
}

Lfm3Forward *lfm3_forward_create(const Lfm3Weights *w, const Lfm3Options *o, char err[LFM3_ERR])
{
  if (!w || !o || !o->max_tokens || !o->max_ctx || o->max_tokens > o->max_ctx)
    return lfm3_fail(err, "no weights or options, or a step past the context"), NULL;
  Lfm3Forward *f = calloc(1, sizeof *f);
  if (!f)
    return lfm3_fail(err, "out of memory"), NULL;
  const Lfm3Geometry *g = &w->geo;
  f->w = w, f->max_tokens = o->max_tokens, f->max_ctx = o->max_ctx;
  f->logit_rows = o->logit_rows ? o->logit_rows : o->max_tokens;
  const uint64_t n = o->max_tokens, lr = f->logit_rows, D = (uint64_t)g->n_embd, ff = (uint64_t)g->n_ff;
  const uint64_t q = (uint64_t)g->n_heads * g->head_dim, kv = (uint64_t)g->n_kv_heads * g->head_dim;
  /* the full layers' rotation the zero tail's (YaRN past the native length), the sliding ones' every pair, never
     scaled (the window's keys are near) - transformers' angle step, Liquid's */
  bool ok = q4x_rope((uint32_t)g->head_dim, (uint32_t)g->rope_pairs, g->rope_base, g->native_ctx, f->max_ctx,
                     Q4X_THETA_HF, &f->rope_full) &&
            q4x_rope((uint32_t)g->head_dim, (uint32_t)g->head_dim / 2, g->rope_base_swa, g->native_ctx, g->native_ctx,
                     Q4X_THETA_HF, &f->rope_swa);
  ok = ok || lfm3_fail(err, "the ropes refused (their shape, or a context past YaRN's reach)");
  ok = ok && (hipStreamCreate((hipStream_t *)&f->st) == hipSuccess || lfm3_fail(err, "the step's stream not made"));
  ok = ok && upload(f, err);
  const struct
  {
    void *p;
    uint64_t bytes;
  } want[] = {
      {&f->tok, n * 4},          {&f->h, n * D * F},       {&f->xn, n * D * F},    {&f->y, n * D * F},
      {&f->bcx, n * 3 * D * F},  {&f->u, n * D * F},       {&f->qf, n * q * F},    {&f->kf, n * kv * F},
      {&f->vf, n * kv * F},      {&f->qr, n * q * F},      {&f->kr, n * kv * F},   {&f->g, n * g->n_heads * F},
      {&f->att, n * q * F},      {&f->ga, n * ff * F},     {&f->up, n * ff * F},   {&f->act, n * ff * F},
      {&f->attn_scratch, q35_attn_decode_scratch((uint32_t)g->n_heads, Q35_ATTN_DECODE_MAX) * F},
      {&f->xf, lr * D * F},      {&f->logits, lr * g->n_vocab * F},
  };
  for (size_t i = 0; ok && i < sizeof want / sizeof *want; i++)
    ok = lfm3_alloc(want[i].p, want[i].bytes) || lfm3_fail(err, "GPU memory refused (a step's buffers)");
  ok = ok && hipHostMalloc((void **)&f->pin_tok, n * 4, 0) == hipSuccess &&
       hipHostMalloc((void **)&f->pin_logits, lr * g->n_vocab * F, 0) == hipSuccess &&
       hipDeviceSynchronize() == hipSuccess;
  if (!ok)
    return lfm3_forward_destroy(f), (err && !*err ? lfm3_fail(err, "the forward not made") : false), NULL;
  return f;
}

void lfm3_seq_destroy(Lfm3Seq *s)
{
  if (!s)
    return;
  for (int l = 0; l < LFM3_MAX_LAYERS; l++)
    hipFree(s->conv[l]), hipFree(s->kc[l]), hipFree(s->vc[l]);
  free(s);
}

Lfm3Seq *lfm3_seq_create(Lfm3Forward *f, char err[LFM3_ERR])
{
  if (!f)
    return lfm3_fail(err, "no forward"), NULL;
  Lfm3Seq *s = calloc(1, sizeof *s);
  if (!s)
    return lfm3_fail(err, "out of memory"), NULL;
  s->f = f;
  const Lfm3Geometry *g = &f->w->geo;
  const uint64_t D = (uint64_t)g->n_embd, row = q4x_kv_row_bytes((uint32_t)(g->n_kv_heads * g->head_dim));
  const uint64_t conv = (uint64_t)LFM3_CONV_SLOTS * (uint64_t)(g->conv_taps - 1) * D * F;
  bool ok = true;
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = g->kind[l] == LFM3_CONV ? lfm3_alloc(&s->conv[l], conv) && hipMemset(s->conv[l], 0, conv) == hipSuccess
                                 : lfm3_alloc(&s->kc[l], f->max_ctx * row) && lfm3_alloc(&s->vc[l], f->max_ctx * row);
  if (!ok)
    return lfm3_seq_destroy(s), lfm3_fail(err, "GPU memory refused (a sequence's state)"), NULL;
  return s;
}

int32_t lfm3_seq_position(const Lfm3Seq *s) { return s ? s->pos : -1; }

void lfm3_forward_tap(Lfm3Forward *f, lfm3_tap_fn fn, void *arg)
{
  if (f)
    f->tap = fn, f->tap_arg = arg;
}
