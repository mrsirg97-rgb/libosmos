/* the lfm3 forward's making (osmos/lfm3/forward.h): every tensor to the GPU, the ropes, a step's buffers; a sequence's
   state (its convolutions' rings, its attention layers' caches) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/mm.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q35_ops.h"
#include "osmos/gpu/quant.h"
#include "osmos/model/qtype.h"
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

/* tensors' rows concatenated (one type) into one buffer, their device pointers then into it; NULL if their types
   differ or one is missing */
static const void *join(Lfm3Forward *f, const GgufTensor *const *t, int k, bool *ok)
{
  const Gguf *g = f->w->model;
  uint64_t bytes = 0;
  for (int i = 0; i < k; i++)
  {
    if (!t[i] || t[i]->type != t[0]->type)
      return NULL;
    bytes += t[i]->bytes;
  }
  uint8_t *buf = NULL;
  if (!lfm3_alloc(&buf, bytes))
    return *ok = false, NULL;
  f->joints[f->n_joints++] = buf;
  uint64_t at = 0;
  for (int i = 0; i < k && *ok; i++)
  {
    const size_t idx = (size_t)(t[i] - g->tensors);
    *ok = hipMemcpy(buf + at, f->dev[idx], t[i]->bytes, hipMemcpyDeviceToDevice) == hipSuccess;
    hipFree(f->dev[idx]);
    f->dev[idx] = buf + at, f->in_joint[idx] = true, at += t[i]->bytes;
  }
  return buf;
}

static bool joints(Lfm3Forward *f, char *err)
{
  const Lfm3Geometry *g = &f->w->geo;
  bool ok = (f->in_joint = calloc(f->w->model->n_tensors, sizeof *f->in_joint)) != NULL;
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    const Lfm3Layer *L = &f->w->layer[l];
    const GgufTensor *ffn[2] = {L->ffn_gate, L->ffn_up}, *att[4] = {L->q, L->k, L->v, L->gate};
    f->j_ffn[l] = join(f, ffn, 2, &ok);
    if (ok && g->kind[l] != LFM3_CONV)
      f->j_attn[l] = join(f, att, 4, &ok);
  }
  return ok || lfm3_fail(err, "the joints not made (GPU memory)");
}

/* the tied head's Q8_0 copy (q35's head_q8: ggml's quantizer through q4x_kv_put, a chunk of rows at a time) - the
   embedding's table kept the file's: the head alone reads the copy */
static bool head_q8(Lfm3Forward *f, char *err)
{
  const GgufTensor *t = f->w->embd;
  const uint32_t V = (uint32_t)f->w->geo.n_vocab, D = (uint32_t)f->w->geo.n_embd, chunk = 4096;
  const uint64_t rb = q4x_kv_row_bytes(D);
  void *spare = NULL;
  float *tmp = NULL;
  bool ok = lfm3_alloc(&f->head_q8, (uint64_t)V * rb) && lfm3_alloc(&spare, (uint64_t)chunk * rb) &&
            lfm3_alloc(&tmp, (uint64_t)chunk * D * F);
  for (uint32_t r = 0; ok && r < V; r += chunk)
  {
    const uint32_t m = V - r < chunk ? V - r : chunk;
    ok = quant_dequant(t->type, (const uint8_t *)lfm3_dev(f, t) + (uint64_t)r * (t->bytes / V), (uint64_t)m * D, tmp,
                       NULL) &&
         q4x_kv_put(tmp, tmp, m, D, (uint8_t *)f->head_q8 + (uint64_t)r * rb, spare, NULL);
  }
  ok = ok && hipDeviceSynchronize() == hipSuccess;
  hipFree(spare), hipFree(tmp);
  if (!ok)
    return lfm3_fail(err, "the head's Q8_0 not made (GPU memory, or a launch)");
  f->head = f->head_q8, f->head_type = QTYPE_Q8_0;
  return true;
}

void lfm3_forward_destroy(Lfm3Forward *f)
{
  if (!f)
    return;
  if (f->st)
    hipStreamSynchronize((hipStream_t)f->st), hipStreamDestroy((hipStream_t)f->st);
  for (size_t i = 0; f->dev && i < f->w->model->n_tensors; i++)
    if (!f->in_joint || !f->in_joint[i])
      hipFree(f->dev[i]);
  free(f->dev);
  for (uint32_t i = 0; i < f->n_joints; i++)
    hipFree(f->joints[i]);
  free(f->in_joint);
  void *b[] = {f->tok, f->h, f->xn, f->y, f->bcx, f->u, f->qf, f->kf, f->vf, f->qr, f->kr, f->g, f->att, f->ga,
               f->up, f->act, f->attn_scratch, f->xf, f->logits, f->hk, f->ids, f->argmax, f->jo, f->x16, f->overflow};
  for (size_t i = 0; i < sizeof b / sizeof *b; i++)
    hipFree(b[i]);
  hipHostFree(f->pin_tok), hipHostFree(f->pin_logits);
  hipFree(f->head_q8);
  free(f->slot);
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
  f->ring = (uint32_t)g->window + o->max_tokens + LFM3_MAX_BACK;
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
  ok = ok && joints(f, err);
  if (ok)
    f->head = lfm3_dev(f, w->embd), f->head_type = w->embd->type;
  ok = ok && (o->head == LFM3_HEAD_FILE || o->head == LFM3_HEAD_Q8_0 || lfm3_fail(err, "a head this build does not make")) &&
       (o->head == LFM3_HEAD_FILE || head_q8(f, err));
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
      {&f->xf, lr * D * F},      {&f->logits, lr * g->n_vocab * F}, {&f->hk, lr * D * F},
      {&f->ids, lr * 4},         {&f->argmax, ops_argmax_scratch((uint32_t)lr)},
      {&f->x16, (uint64_t)MM_MAX_TOKENS * (ff > 3 * D ? ff : 3 * D) * 2}, {&f->overflow, 4},
      {&f->jo, (2 * ff > q + 2 * kv + (uint64_t)g->n_heads ? 2 * ff : q + 2 * kv + (uint64_t)g->n_heads) * F},
  };
  for (size_t i = 0; ok && i < sizeof want / sizeof *want; i++)
    ok = lfm3_alloc(want[i].p, want[i].bytes) || lfm3_fail(err, "GPU memory refused (a step's buffers)");
  ok = ok && hipHostMalloc((void **)&f->pin_tok, n * 4, 0) == hipSuccess &&
       hipHostMalloc((void **)&f->pin_logits, lr * g->n_vocab * F, 0) == hipSuccess && (f->slot = malloc(n * 4)) &&
       hipMemset(f->overflow, 0, 4) == hipSuccess && hipDeviceSynchronize() == hipSuccess;
  if (!ok)
    return lfm3_forward_destroy(f), (err && !*err ? lfm3_fail(err, "the forward not made") : false), NULL;
  return f;
}

void lfm3_forward_tap(Lfm3Forward *f, lfm3_tap_fn fn, void *arg)
{
  if (f)
    f->tap = fn, f->tap_arg = arg;
}
