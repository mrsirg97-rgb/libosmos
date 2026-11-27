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

void q35_forward_tap(Q35Forward *f, q35_tap_fn fn, void *arg) { f->tap = fn, f->tap_arg = arg; }

static bool past_layers(const GgufTensor *t, void *arg) /* the MTP head's layers: not this forward's */
{
  return strncmp(t->name, "blk.", 4) == 0 && atoi(t->name + 4) >= ((const Q35Forward *)arg)->w->geo.n_layers;
}

static bool upload(Q35Forward *f, char *err)
{
  return gpu_upload_table(f->ledger, f->w->model, READ_PIECE, past_layers, f, &f->dev, err);
}

void q35_forward_destroy(Q35Forward *f)
{
  if (!f)
    return;
  lsa_closer_destroy(f->closer);
  gpu_free_table(f->ledger, f->w->model, f->dev, NULL);
  attn_core_destroy(f->attn);
  gpu_bufs_free(&f->bufs); /* every GPU and pinned buffer the forward made - the step's, LSA's maps and read, the
                              pins: from the record */
  free(f->slot);
  lsa_scratch_destroy(f->lsa_s);
  free(f->h_ik), free(f->h_cent), free(f->h_mem);
  free(f);
}

static bool q35_yarn_rope(const Q35Geometry *g, float yarn, Q4xRope *out, char *err)
{
  if (!isfinite(yarn) || yarn <= 0.0f)
    return osmos_fail(err, "the YaRN scale must be finite and positive");
  const double ctx = (double)yarn * (double)g->native_ctx;
  if (!(ctx >= 1.0 && ctx <= (double)UINT32_MAX))
    return osmos_fail(err, "the YaRN context is out of range");
  return q4x_rope((uint32_t)g->rope_dim, (uint32_t)g->rope_dim / 2, g->rope_base, g->native_ctx, (uint32_t)ctx,
                  Q4X_THETA_GGML, out) ||
         osmos_fail(err, "the YaRN rotation refused");
}

static void q35_lsa_rollback(Q35Forward *f, uint32_t bufs_was, uint64_t bytes_was)
{
  lsa_closer_destroy(f->closer), f->closer = NULL;
  lsa_scratch_destroy(f->lsa_s), f->lsa_s = NULL;
  free(f->h_ik), free(f->h_cent), free(f->h_mem);
  f->h_ik = f->h_cent = NULL, f->h_mem = NULL;
  for (uint32_t i = f->bufs.n; i-- > bufs_was;)
  {
    void **p = f->bufs.slot[i].p;
    if (*p)
      f->bufs.slot[i].host ? ledger_host_free(f->bufs.ledger, *p) : ledger_free(f->bufs.ledger, *p);
    *p = NULL;
  }
  f->bufs.n = bufs_was, f->bufs.bytes = bytes_was;
  f->close_max = 0;
}

bool q35_forward_lsa(Q35Forward *f, const char *maps, float yarn, char err[OSMOS_ERR])
{
  const Q35Geometry *g = &f->w->geo;
  if (f->lsa)
    return osmos_fail(err, "the read already on");
  Q4xRope new_rope;
  if (!q35_yarn_rope(g, yarn, &new_rope, err))
    return false;
  if (!maps) /* YaRN alone: the dense model under the maps' rotation (the read's comparison) */
    return f->rope = new_rope, true;
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
  const uint32_t bufs_was = f->bufs.n;
  const uint64_t bytes_was = f->bufs.bytes;
  const size_t m = (size_t)LSA_IN * LSA_DIM;
  float *buf = malloc(2 * m * F);
  bool ok = buf != NULL;
  uint32_t seen = 0;
  for (uint32_t i = 0; ok && i < hdr[1]; i++)
  {
    uint32_t L = 0;
    ok = fread(&L, 4, 1, fp) == 1 && fread(buf, F, 2 * m, fp) == 2 * m && L < (uint32_t)g->n_layers &&
         !g->recurrent[L] && !f->wq[L] && q35_alloc(&f->bufs, &f->wq[L], m * F) && q35_alloc(&f->bufs, &f->wk[L], m * F) &&
         hipMemcpy(f->wq[L], buf, m * F, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->wk[L], buf + m, m * F, hipMemcpyHostToDevice) == hipSuccess;
    seen += ok;
  }
  fclose(fp);
  free(buf);
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = g->recurrent[l] || f->wq[l] || (osmos_fail(err, "an attention layer without its maps"), false);
  if (!ok)
    return q35_lsa_rollback(f, bufs_was, bytes_was), err && !*err ? osmos_fail(err, "the maps not loaded") : false;
  uint32_t attn_layers = 0;
  for (int l = 0; l < g->n_layers; l++)
    attn_layers += !g->recurrent[l];
  f->close_max = f->max_tokens / LSA_WINDOW + 2;
  ok = (f->lsa_s = lsa_scratch_create(f->ledger, f->max_ctx / LSA_WINDOW * LSA_BUCKETS + 1)) &&
       (f->closer = lsa_closer_create(f->ledger, attn_layers ? attn_layers : 1, f->close_max)) &&
       q35_alloc(&f->bufs, &f->iq, LSA_DIM * F) && q35_alloc(&f->bufs, &f->list, LSA_READ_MAX * 4) &&
       q35_alloc(&f->bufs, &f->list_n, 4) &&
       (f->h_ik = malloc((uint64_t)f->close_max * LSA_WINDOW * LSA_DIM * F)) &&
       (f->h_cent = malloc((uint64_t)f->close_max * LSA_BUCKETS * LSA_DIM * F)) &&
       (f->h_mem = malloc((uint64_t)f->close_max * LSA_BUCKETS * LSA_CAP * 4));
  if (!ok)
    return q35_lsa_rollback(f, bufs_was, bytes_was), osmos_fail(err, "out of memory (the read's scratch)");
  f->rope = new_rope;
  f->lsa = seen > 0;
  return true;
}

/* the LM head (BF16 in the file, a decode token's largest read) as Q8_0 on the GPU (ggml's quantizer, q4x_kv_put's:
   a row's blocks of 32, d = amax / 127). Q35_HEAD=bf16 keeps the file's */
static bool head_q8(Q35Forward *f, char *err)
{
  const GgufTensor *t = f->w->output;
  if (t->type != QTYPE_BF16)
    return true;
  const uint32_t V = (uint32_t)f->w->geo.n_vocab, D = (uint32_t)f->w->geo.n_embd;
  void **slot = &f->dev[t - f->w->model->tensors];
  void *q = NULL;
  if (!gpu_q8_rows(f->ledger, QTYPE_BF16, *slot, (uint64_t)D * 2, V, D, &q))
    return osmos_fail(err, "the LM head's Q8_0 not made (GPU memory, or a launch)");
  ledger_free(f->ledger, *slot); /* (the table's BF16 head, its slot the Q8_0 copy's: the table frees it) */
  *slot = q, f->head_type = QTYPE_Q8_0;
  return true;
}

Q35Forward *q35_forward_create(const Q35Weights *w, uint32_t max_tokens, uint32_t max_ctx, char err[OSMOS_ERR])
{
  const Q35Options o = {max_tokens, max_ctx, 0, NULL};
  return q35_forward_create_with(w, &o, err);
}

Q35Forward *q35_forward_create_with(const Q35Weights *w, const Q35Options *o, char err[OSMOS_ERR])
{
  if (!w || !o || !o->max_tokens || o->max_tokens > o->max_ctx || o->max_ctx > OPS_Q4X_MAX_KEYS)
    return osmos_fail(err, "no weights, no tokens, or a context past the attention's keys"), NULL;
  Q35Forward *f = calloc(1, sizeof *f);
  if (!f)
    return osmos_fail(err, "out of memory"), NULL;
  f->ledger = o->ledger, f->bufs.ledger = o->ledger;
  const uint32_t max_tokens = o->max_tokens, max_ctx = o->max_ctx;
  f->w = w, f->max_tokens = max_tokens, f->max_ctx = max_ctx;
  f->logit_rows = o->logit_rows && o->logit_rows < max_tokens ? o->logit_rows : max_tokens;
  const uint64_t lr = f->logit_rows;
  const Q35Geometry *g = &w->geo;
  const uint64_t n = max_tokens, D = (uint64_t)g->n_embd, ff = (uint64_t)g->n_ff, vh = (uint64_t)g->ssm_v_heads;
  const uint64_t in = (uint64_t)g->ssm_inner, ch = 2ull * g->ssm_groups * g->ssm_state + in;
  const uint64_t q = (uint64_t)g->n_heads * g->head_dim, kv = (uint64_t)g->n_kv_heads * g->head_dim;
  uint64_t wide = ff > ch ? ff : ch; /* mm's widest input: the SwiGLU's down, attention out, or hidden */
  if (q > wide)
    wide = q;
  if (D > wide)
    wide = D;
  const AttnCoreDesc ad = {.heads = (uint32_t)g->n_heads,
                           .kv_heads = (uint32_t)g->n_kv_heads,
                           .dk = (uint32_t)g->head_dim,
                           .dv = (uint32_t)g->head_dim,
                           .k_rows = ATTN_ROWS_Q8_0,
                           .v_rows = ATTN_ROWS_Q8_0,
                           .scaling = 1.0f / sqrtf((float)g->head_dim),
                           .term = ATTN_TERM_NONE,
                           .gate = ATTN_GATE_VALUE,
                           .block = ATTN_BLOCK_HEADS,
                           .unit = 128,
                           .apart = true,
                           .max_tokens = max_tokens,
                           .max_keys = max_ctx};
  const uint64_t scratch = attn_core_scratch(&ad);
  const GpuWant want[] = {
      {&f->h, n * D * F},       {&f->xn, n * D * F},  {&f->out, n * D * F},     {&f->qkv, n * ch * F},
      {&f->z, n * in * F},      {&f->al, n * vh * F}, {&f->be, n * vh * F},     {&f->g, n * vh * F},
      {&f->b, n * vh * F},      {&f->cv, n * ch * F}, {&f->qkn, n * 2 * g->ssm_groups * g->ssm_state * F},
      {&f->ro, n * in * F},     {&f->fo, n * in * F}, {&f->qf, n * 2 * q * F},  {&f->kr, n * kv * F},
      {&f->vr, n * kv * F},     {&f->q, n * q * F},   {&f->gt, n * q * F},      {&f->kq, n * kv * F},
      {&f->ao, n * q * F},      {&f->fa, scratch},
      {&f->ga, n * ff * F},     {&f->up, n * ff * F}, {&f->y, n * ff * F},      {&f->logits, lr * g->n_vocab * F},
      {&f->hx, lr * D * F},     {&f->ids, lr * 4},    {&f->argmax, ops_argmax_scratch((uint32_t)lr)},
      {&f->tok, n * 4},         {&f->overflow, 4},    {&f->x16, (uint64_t)MM_MAX_TOKENS * wide * 2},
  };
  bool ok = q4x_rope((uint32_t)g->rope_dim, (uint32_t)g->rope_dim / 2, g->rope_base, g->native_ctx, max_ctx, Q4X_THETA_GGML, &f->rope) && upload(f, err);
  ok = ok && (gpu_bufs_alloc(&f->bufs, f->ledger, want, sizeof want / sizeof *want) ||
              osmos_fail(err, "GPU memory refused (a step's buffers)"));
  ok = ok && (f->attn = attn_core_create(&ad, f->fa, scratch, err)) != NULL;
  f->head_type = w->output->type;
  ok = ok && head_q8(f, err) && q35_pinned(&f->bufs, &f->pin_overflow, 4) && !(*f->pin_overflow = 0) &&
       (f->ovf = (OverflowFlag){f->overflow, 4, false, f->pin_overflow}, overflow_reset(&f->ovf, err)) &&
       q35_pinned(&f->bufs, &f->pin_logits, lr * g->n_vocab * F) && q35_pinned(&f->bufs, &f->pin_tok, n * 4) &&
       (f->slot = malloc(n * 4));
  if (!ok)
    return q35_forward_destroy(f), (err && !*err ? osmos_fail(err, "the forward not made") : false), NULL;
  return f;
}
