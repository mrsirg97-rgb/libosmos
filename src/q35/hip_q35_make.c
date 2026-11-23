/* the q35 forward made (osmos/q35/forward.h): the model's tensors read into GPU memory, its LM head as Q8_0, LSA's maps,
   a step's buffers - and unmade */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/mm.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q35_ops.h"
#include "osmos/gpu/quant.h"
#include "osmos/gpu/bufs.h"
#include "osmos/gpu/upload.h"
#include "osmos/model/qtype.h"
#include "q35_impl.h"
#include "osmos/err.h"

#define READ_PIECE (64ull << 20)
#define alloc q35_alloc

void q35_forward_tap(Q35Forward *f, q35_tap_fn fn, void *arg) { f->tap = fn, f->tap_arg = arg; }

static bool past_layers(const GgufTensor *t, void *arg) /* the MTP head's layers: not this forward's */
{
  return strncmp(t->name, "blk.", 4) == 0 && atoi(t->name + 4) >= ((const Q35Forward *)arg)->w->geo.n_layers;
}

static bool upload(Q35Forward *f, char *err)
{
  return gpu_upload_table(f->w->model, READ_PIECE, past_layers, f, &f->dev, err);
}

void q35_forward_destroy(Q35Forward *f)
{
  if (!f)
    return;
  gpu_free_table(f->w->model, f->dev, NULL);
  gpu_bufs_free(&f->bufs); /* the step's buffers: the table's */
  hipHostFree(f->pin_logits), hipHostFree(f->pin_tok);
  free(f->slot);
  for (int l = 0; l < Q35_MAX_LAYERS; l++)
    hipFree(f->wq[l]), hipFree(f->wk[l]);
  lsa_scratch_destroy(f->lsa_s);
  hipFree(f->iq), hipFree(f->list), hipFree(f->list_n);
  free(f->h_ik), free(f->h_cent), free(f->h_mem);
  free(f);
}

bool q35_forward_lsa(Q35Forward *f, const char *maps, float yarn, char err[OSMOS_ERR])
{
  const Q35Geometry *g = &f->w->geo;
  if (f->lsa)
    return osmos_fail(err, "the read already on");
  if (!maps) /* YaRN alone: the dense model under the maps' rotation (the read's comparison) */
    return q4x_rope((uint32_t)g->rope_dim, (uint32_t)g->rope_dim / 2, g->rope_base, g->native_ctx, (uint32_t)(yarn * (float)g->native_ctx), Q4X_THETA_GGML, &f->rope) ||
           osmos_fail(err, "the YaRN rotation refused");
  if ((uint32_t)g->n_kv_heads * (uint32_t)(g->head_dim - g->rope_dim) != LSA_IN)
    return osmos_fail(err, "the model's free key dims are not the maps' input");
  FILE *fp = fopen(maps, "rb");
  uint32_t hdr[4] = {0};
  if (!fp || fread(hdr, 4, 4, fp) != 4 || memcmp(hdr, "LSAM", 4) || hdr[2] != LSA_IN || hdr[3] != LSA_DIM)
  {
    if (fp)
      fclose(fp);
    return osmos_fail(err, "the maps file not read (tools/convert/lsa_maps.py's, LSA_IN x LSA_DIM)");
  }
  const size_t m = (size_t)LSA_IN * LSA_DIM;
  float *buf = malloc(2 * m * F);
  bool ok = buf != NULL;
  uint32_t seen = 0;
  for (uint32_t i = 0; ok && i < hdr[1]; i++)
  {
    uint32_t L = 0;
    ok = fread(&L, 4, 1, fp) == 1 && fread(buf, F, 2 * m, fp) == 2 * m && L < (uint32_t)g->n_layers &&
         !g->recurrent[L] && !f->wq[L] && alloc(&f->wq[L], m * F) && alloc(&f->wk[L], m * F) &&
         hipMemcpy(f->wq[L], buf, m * F, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->wk[L], buf + m, m * F, hipMemcpyHostToDevice) == hipSuccess;
    seen += ok;
  }
  fclose(fp);
  free(buf);
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = g->recurrent[l] || f->wq[l] || (osmos_fail(err, "an attention layer without its maps"), false);
  if (!ok)
    return err && !*err ? osmos_fail(err, "the maps not loaded") : false;
  /* the maps' rotation: YaRN at yarn x native (their training's), whatever this forward's context */
  if (!q4x_rope((uint32_t)g->rope_dim, (uint32_t)g->rope_dim / 2, g->rope_base, g->native_ctx, (uint32_t)(yarn * (float)g->native_ctx), Q4X_THETA_GGML, &f->rope))
    return osmos_fail(err, "the maps' YaRN rotation refused");
  ok = (f->lsa_s = lsa_scratch_create(f->max_ctx / LSA_WINDOW * LSA_BUCKETS + 1)) && alloc(&f->iq, LSA_DIM * F) &&
       alloc(&f->list, LSA_READ_MAX * 4) && alloc(&f->list_n, 4) && (f->h_ik = malloc(LSA_WINDOW * LSA_DIM * F)) &&
       (f->h_cent = malloc(LSA_BUCKETS * LSA_DIM * F)) && (f->h_mem = malloc(LSA_BUCKETS * LSA_CAP * 4));
  if (!ok)
    return osmos_fail(err, "out of memory (the read's scratch)");
  f->lsa = seen > 0;
  return true;
}

/* the LM head (BF16 in the file: 2.54 GB a decode token reads) as Q8_0 on the GPU (ggml's quantizer, q4x_kv_put's:
   a row's blocks of 32, d = amax / 127): 1.35 GB. Q35_HEAD=bf16 keeps the file's */
static bool head_q8(Q35Forward *f, char *err)
{
  const GgufTensor *t = f->w->output;
  const char *keep = getenv("Q35_HEAD");
  if (t->type != QTYPE_BF16 || (keep && strcmp(keep, "bf16") == 0))
    return true;
  const uint32_t V = (uint32_t)f->w->geo.n_vocab, D = (uint32_t)f->w->geo.n_embd;
  void **slot = &f->dev[t - f->w->model->tensors];
  void *q = NULL;
  if (!gpu_q8_rows(QTYPE_BF16, *slot, (uint64_t)D * 2, V, D, &q))
    return osmos_fail(err, "the LM head's Q8_0 not made (GPU memory, or a launch)");
  hipFree(*slot);
  *slot = q, f->head_type = QTYPE_Q8_0;
  return true;
}

Q35Forward *q35_forward_create(const Q35Weights *w, uint32_t max_tokens, uint32_t max_ctx, char err[OSMOS_ERR])
{
  const Q35Options o = {max_tokens, max_ctx, 0};
  return q35_forward_create_with(w, &o, err);
}

Q35Forward *q35_forward_create_with(const Q35Weights *w, const Q35Options *o, char err[OSMOS_ERR])
{
  if (!w || !o || !o->max_tokens || o->max_tokens > o->max_ctx || o->max_ctx > OPS_Q4X_MAX_KEYS)
    return osmos_fail(err, "no weights, no tokens, or a context past the attention's keys"), NULL;
  Q35Forward *f = calloc(1, sizeof *f);
  if (!f)
    return osmos_fail(err, "out of memory"), NULL;
  const uint32_t max_tokens = o->max_tokens, max_ctx = o->max_ctx;
  f->w = w, f->max_tokens = max_tokens, f->max_ctx = max_ctx;
  f->logit_rows = o->logit_rows && o->logit_rows < max_tokens ? o->logit_rows : max_tokens;
  const uint64_t lr = f->logit_rows;
  const Q35Geometry *g = &w->geo;
  const uint64_t n = max_tokens, D = (uint64_t)g->n_embd, ff = (uint64_t)g->n_ff, vh = (uint64_t)g->ssm_v_heads;
  const uint64_t in = (uint64_t)g->ssm_inner, ch = 2ull * g->ssm_groups * g->ssm_state + in;
  const uint64_t q = (uint64_t)g->n_heads * g->head_dim, kv = (uint64_t)g->n_kv_heads * g->head_dim;
  const uint64_t wide = ff > ch ? ff : ch; /* mm's widest input: the SwiGLU's down */
  const GpuWant want[] = {
      {&f->h, n * D * F},       {&f->xn, n * D * F},  {&f->out, n * D * F},     {&f->qkv, n * ch * F},
      {&f->z, n * in * F},      {&f->al, n * vh * F}, {&f->be, n * vh * F},     {&f->g, n * vh * F},
      {&f->b, n * vh * F},      {&f->cv, n * ch * F}, {&f->qkn, n * 2 * g->ssm_groups * g->ssm_state * F},
      {&f->ro, n * in * F},     {&f->fo, n * in * F}, {&f->qf, n * 2 * q * F},  {&f->kr, n * kv * F},
      {&f->vr, n * kv * F},     {&f->q, n * q * F},   {&f->gt, n * q * F},      {&f->kq, n * kv * F},
      {&f->ao, n * q * F},      {&f->fa, q4x_attn_scratch((uint32_t)g->n_heads, max_ctx) * F},
      {&f->da, q35_attn_decode_scratch((uint32_t)g->n_heads, Q35_ATTN_DECODE_MAX) * F},
      {&f->ga, n * ff * F},     {&f->up, n * ff * F}, {&f->y, n * ff * F},      {&f->logits, lr * g->n_vocab * F},
      {&f->hx, lr * D * F},     {&f->ids, lr * 4},    {&f->argmax, ops_argmax_scratch((uint32_t)lr)},
      {&f->tok, n * 4},         {&f->overflow, 4},    {&f->x16, (uint64_t)MM_MAX_TOKENS * wide * 2},
  };
  bool ok = q4x_rope((uint32_t)g->rope_dim, (uint32_t)g->rope_dim / 2, g->rope_base, g->native_ctx, max_ctx, Q4X_THETA_GGML, &f->rope) && upload(f, err);
  ok = ok && (gpu_bufs_alloc(&f->bufs, want, sizeof want / sizeof *want) ||
              osmos_fail(err, "GPU memory refused (a step's buffers)"));
  f->head_type = w->output->type;
  ok = ok && head_q8(f, err) && (f->ovf = (OverflowFlag){f->overflow, 4, false}, overflow_reset(&f->ovf, err)) &&
       hipHostMalloc((void **)&f->pin_logits, lr * g->n_vocab * F, 0) == hipSuccess &&
       hipHostMalloc((void **)&f->pin_tok, n * 4, 0) == hipSuccess && (f->slot = malloc(n * 4));
  if (!ok)
    return q35_forward_destroy(f), (err && !*err ? osmos_fail(err, "the forward not made") : false), NULL;
  return f;
}
