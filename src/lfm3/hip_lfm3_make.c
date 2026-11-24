/* the lfm3 forward's making (osmos/lfm3/forward.h): every tensor to the GPU, the ropes, a step's buffers; a sequence's
   state (its convolutions' rings, its attention layers' caches) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/bufs.h"
#include "osmos/gpu/mm.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q35_ops.h"
#include "osmos/gpu/quant.h"
#include "osmos/model/qtype.h"
#include "osmos/gpu/upload.h"
#include "osmos/model/fnv.h"
#include "osmos/text/json.h"
#include "lfm3_impl.h"
#include "osmos/err.h"

#define READ_PIECE (64ull << 20)

bool lfm3_alloc(GpuBufs *b, void *p, uint64_t bytes) { return gpu_bufs_one(b, b->ledger, p, bytes); }

void lfm3_free(Lfm3Forward *f, void *p)
{
  if (p)
    ledger_free(f->ledger, p);
}

/* every tensor of g into GPU memory, its device pointer at its index in *table */
static bool upload_file(Lfm3Forward *f, const Gguf *g, void ***table, char *err)
{
  return gpu_upload_table(f->ledger, g, READ_PIECE, NULL, NULL, table, err);
}

static bool upload(Lfm3Forward *f, char *err) { return upload_file(f, f->w->model, &f->dev, err); }

/* the drafter's buffers (the DFlash2 core's), and its taps' rows: a step's max_tokens */
static bool lfm3_df_buffers(Lfm3Forward *f, char *err)
{
  const uint64_t T = (uint64_t)f->df->geo.n_taps * (uint64_t)f->df->geo.n_embd;
  return (lfm3_alloc(&f->bufs, &f->taps, (uint64_t)f->max_tokens * T * F) && df_buffers(&f->d, f->ledger, &f->df->geo, f->max_tokens, f->max_ctx)) ||
         osmos_fail(err, "GPU memory refused (the drafter's buffers)");
}

/* the drafter: its geometry these weights' (its taps their layers'), its tensors uploaded, its buffers */
static bool df_make(Lfm3Forward *f, const Q4xDflash *df, char *err)
{
  const Lfm3Geometry *g = &f->w->geo;
  for (int l = 0; l < LFM3_MAX_LAYERS; l++)
    f->tap_of[l] = -1;
  if (!df)
    return true;
  const Q4xDflashGeometry *dg = &df->geo;
  if (dg->n_embd != g->n_embd || dg->n_vocab != g->n_vocab || dg->n_taps < 1)
    return osmos_fail(err, "a drafter not of these weights (its width, its vocabulary, no taps)");
  for (int i = 0; i < dg->n_taps; i++)
  {
    if (dg->taps[i] < 0 || dg->taps[i] >= g->n_layers)
      return osmos_fail(err, "a drafter's tap past the model's layers");
    f->tap_of[dg->taps[i]] = i;
  }
  f->df = df;
  f->df_ring = (uint32_t)dg->window + LFM3_CONV_SLOTS < f->max_ctx ? (uint32_t)dg->window + LFM3_CONV_SLOTS : f->max_ctx;
  return upload_file(f, df->model, &f->df_dev, err) && lfm3_df_buffers(f, err);
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
  if (!lfm3_alloc(&f->bufs, &f->joints[f->n_joints], bytes)) /* (its table's slot recorded: freed from it) */
    return *ok = false, NULL;
  uint8_t *buf = f->joints[f->n_joints++];
  uint64_t at = 0;
  for (int i = 0; i < k && *ok; i++)
  {
    const size_t idx = (size_t)(t[i] - g->tensors);
    *ok = hipMemcpy(buf + at, f->dev[idx], t[i]->bytes, hipMemcpyDeviceToDevice) == hipSuccess;
    lfm3_free(f, f->dev[idx]);
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
  return ok || osmos_fail(err, "the joints not made (GPU memory)");
}

/* the tied head's Q8_0 copy (q35's head_q8: ggml's quantizer through q4x_kv_put, a chunk of rows at a time) - the
   embedding's table kept the file's: the head alone reads the copy */
static bool head_q8(Lfm3Forward *f, char *err)
{
  const GgufTensor *t = f->w->embd;
  const uint32_t V = (uint32_t)f->w->geo.n_vocab, D = (uint32_t)f->w->geo.n_embd;
  if (!gpu_q8_rows(f->ledger, t->type, lfm3_dev(f, t), t->bytes / V, V, D, &f->head_q8))
    return osmos_fail(err, "the head's Q8_0 not made (GPU memory, or a launch)");
  f->head = f->head_q8, f->head_type = QTYPE_Q8_0;
  return true;
}

/* the maps (a safetensors file: its JSON header, then the tensors' bytes): every full layer's layers.<l>.W_Q_idx and
   W_K_idx, F32 [lsa_in x LSA_DIM], to the GPU - a layer missing, another shape or dtype refused */
static bool lsa_load(Lfm3Forward *f, const char *path, char *err)
{
  const Lfm3Geometry *g = &f->w->geo;
  const LsaDims fd = lfm3_free_dims(g);
  f->lsa_in = (uint32_t)g->n_kv_heads * ((fd.hi0 - fd.lo0) + (fd.hi1 - fd.lo1));
  if (f->lsa_in > LSA_IN_MAX)
    return osmos_fail(err, "LSA: the index's input past LSA_IN_MAX");
  FILE *fp = fopen(path, "rb");
  uint64_t hn = 0;
  if (!fp || fread(&hn, 8, 1, fp) != 1 || hn > (64u << 20))
  {
    if (fp)
      fclose(fp);
    return osmos_fail(err, "LSA: the maps file not read (a safetensors file)");
  }
  char *head = malloc(hn + 1);
  char jerr[OSMOS_ERR] = "";
  Json *j = head && fread(head, 1, hn, fp) == hn ? json_parse(head, hn, jerr) : NULL;
  const uint64_t m = (uint64_t)f->lsa_in * LSA_DIM;
  float *buf = malloc(m * F);
  bool ok = j && buf;
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    if (g->kind[l] != LFM3_FULL)
      continue;
    for (int w = 0; ok && w < 2; w++)
    {
      char name[64];
      snprintf(name, sizeof name, "layers.%d.W_%c_idx", l, w ? 'K' : 'Q');
      const Json *t = json_get(j, name), *dt = json_get(t, "dtype"), *sh = json_get(t, "shape"),
                 *off = json_get(t, "data_offsets");
      ok = t && dt && dt->type == JSON_STRING && !strcmp(dt->s, "F32") && sh && sh->type == JSON_ARRAY && sh->n == 2 &&
           sh->item[0]->type == JSON_INT && strtoull(sh->item[0]->s, NULL, 10) == f->lsa_in &&
           sh->item[1]->type == JSON_INT && strtoull(sh->item[1]->s, NULL, 10) == LSA_DIM && off &&
           off->type == JSON_ARRAY && off->n == 2 && off->item[0]->type == JSON_INT;
      if (!ok)
      {
        snprintf(err, OSMOS_ERR, "LSA: %s missing, or not F32 [%u x %u]", name, f->lsa_in, LSA_DIM);
        break;
      }
      float **dst = w ? &f->wk[l] : &f->wq[l];
      ok = fseeko(fp, (off_t)(8 + hn + strtoull(off->item[0]->s, NULL, 10)), SEEK_SET) == 0 &&
           fread(buf, F, m, fp) == m && lfm3_alloc(&f->bufs, dst, m * F) &&
           hipMemcpy(*dst, buf, m * F, hipMemcpyHostToDevice) == hipSuccess;
      if (!ok)
        osmos_fail(err, "LSA: a map not read (its bytes, or GPU memory)");
    }
  }
  fclose(fp), free(head), free(buf), json_free(j);
  if (!ok)
    return err && !*err ? osmos_fail(err, "LSA: the maps not read") : false;
  ok = (f->lsa_s = lsa_scratch_create(f->ledger, f->max_ctx / LSA_WINDOW * LSA_BUCKETS + 1)) && lfm3_alloc(&f->bufs, &f->iq, LSA_DIM * F) &&
       lfm3_alloc(&f->bufs, &f->list, LSA_READ_MAX * 4) && lfm3_alloc(&f->bufs, &f->list_n, 4) &&
       (f->h_ik = malloc(LSA_WINDOW * LSA_DIM * F)) && (f->h_cent = malloc(LSA_BUCKETS * LSA_DIM * F)) &&
       (f->h_mem = malloc(LSA_BUCKETS * LSA_CAP * 4));
  /* the scores' scratch: a sub-batch's tokens x every closed bucket, at most a step's prompt rows' F32 (its q, k, v,
     gate and attention: max_tokens x (2 q + 2 kv + heads)) - one more of a step's own size */
  const uint64_t rows = (uint64_t)f->max_tokens * (2 * g->n_heads * g->head_dim + 2 * g->n_kv_heads * g->head_dim);
  f->max_buckets = f->max_ctx / LSA_WINDOW * LSA_BUCKETS + 1;
  f->sub_n = rows / f->max_buckets < f->max_tokens ? (uint32_t)(rows / f->max_buckets) : f->max_tokens;
  f->sub_n = f->sub_n ? f->sub_n : 1;
  ok = ok && lfm3_alloc(&f->bufs, &f->iqb, (uint64_t)f->max_tokens * LSA_DIM * F) &&
       lfm3_alloc(&f->bufs, &f->lists, (uint64_t)f->max_tokens * LSA_READ_MAX * 4) &&
       lfm3_alloc(&f->bufs, &f->counts, (uint64_t)f->max_tokens * 4) &&
       lfm3_alloc(&f->bufs, &f->lsa_sc, lsa_batch_scratch(f->sub_n, f->max_buckets) * F);
  return (f->lsa = ok) || osmos_fail(err, "LSA: out of memory (the read's scratch)");
}

void lfm3_forward_destroy(Lfm3Forward *f)
{
  if (!f)
    return;
  lsa_scratch_destroy(f->lsa_s);
  free(f->h_ik), free(f->h_cent), free(f->h_mem);
  if (f->st)
    hipStreamSynchronize((hipStream_t)f->st), hipStreamDestroy((hipStream_t)f->st);
  gpu_free_table(f->ledger, f->w->model, f->dev, f->in_joint); /* (a joint's members its buffer's) */
  free(f->in_joint);
  lfm3_free(f, f->head_q8); /* (gpu_q8_rows': not the record's) */
  gpu_bufs_free(&f->bufs); /* every GPU and pinned buffer lfm3_alloc made - the step's, LSA's, the joints', the taps':
                              from the record, before the joints' table it NULLs */
  free(f->slot);
  if (f->df)
    gpu_free_table(f->ledger, f->df->model, f->df_dev, NULL);
  df_buffers_free(&f->d);
  free(f);
}

/* osmos1.7b's: lfm3_rope_nll's crossovers (DESIGN.md, "osmos1.7b: rope") */
const Lfm3Stage lfm3_stages_default[] = {{0, 1}};
const uint32_t lfm3_stages_default_n = sizeof lfm3_stages_default / sizeof lfm3_stages_default[0];

/* (osmos/lfm3/forward.h) "from:factor,from:factor,..." - each a decimal of 32 bits, LFM3_MAX_STAGES at most, nothing
   else: a malformed string refused whole */
bool lfm3_stages_parse(const char *c, Lfm3Stage t[LFM3_MAX_STAGES], uint32_t *n)
{
  uint32_t m = 0;
  for (;;)
  {
    unsigned long long v[2];
    for (int i = 0; i < 2; i++) /* from, then factor: a ':' between them */
    {
      if (*c < '0' || *c > '9')
        return false;
      char *e;
      v[i] = strtoull(c, &e, 10);
      if (v[i] > UINT32_MAX || (i == 0 && *e != ':'))
        return false;
      c = i == 0 ? e + 1 : e;
    }
    if (m == LFM3_MAX_STAGES)
      return false;
    t[m++] = (Lfm3Stage){(uint32_t)v[0], (uint32_t)v[1]};
    if (!*c)
      return *n = m, true;
    if (*c++ != ',')
      return false;
  }
}

static bool stages_ok(const Lfm3Stage *t, uint32_t n)
{
  if (!t || !n || n > LFM3_MAX_STAGES || t[0].from)
    return false;
  for (uint32_t s = 0; s < n; s++)
    if (t[s].factor < 1 || t[s].factor > 64 || (s && t[s].from <= t[s - 1].from))
      return false;
  return true;
}

Lfm3Forward *lfm3_forward_create(const Lfm3Weights *w, const Lfm3Options *o, char err[OSMOS_ERR])
{
  if (!w || !o || !o->max_tokens || !o->max_ctx || o->max_tokens > o->max_ctx || o->max_ctx > LFM3_MAX_CTX)
    return osmos_fail(err, "no weights or options, a step past the context, or a context past LFM3_MAX_CTX"), NULL;
  Lfm3Forward *f = calloc(1, sizeof *f);
  if (!f)
    return osmos_fail(err, "out of memory"), NULL;
  const Lfm3Geometry *g = &w->geo;
  f->w = w, f->max_tokens = o->max_tokens, f->max_ctx = o->max_ctx, f->ledger = o->ledger, f->bufs.ledger = o->ledger;
  f->logit_rows = o->logit_rows ? o->logit_rows : o->max_tokens;
  f->ring = (uint32_t)g->window + o->max_tokens + LFM3_MAX_BACK;
  const uint64_t n = o->max_tokens, lr = f->logit_rows, D = (uint64_t)g->n_embd, ff = (uint64_t)g->n_ff;
  const uint64_t q = (uint64_t)g->n_heads * g->head_dim, kv = (uint64_t)g->n_kv_heads * g->head_dim;
  /* the full layers' rotation the zero tail's (YaRN past the native length), the sliding ones' every pair, never
     scaled (the window's keys are near) - transformers' angle step, Liquid's */
  f->yarn_from = o->yarn_from ? o->yarn_from : g->native_ctx;
  f->sparse_from = o->lsa_sparse_from ? o->lsa_sparse_from : LFM3_LSA_SPARSE_FROM;
  const Lfm3Stage *table = o->stages ? o->stages : lfm3_stages_default;
  const uint32_t nt = o->stages ? o->n_stages : lfm3_stages_default_n;
  bool ok = (o->yarn_static_ctx || stages_ok(table, nt) ||
            osmos_fail(err, "the YaRN stages: none, past LFM3_MAX_STAGES, not from 0, out of order, or a factor not 1-64"));
  f->yarn_static_ctx = o->yarn_static_ctx;
  if (ok && o->yarn_static_ctx) /* one stage: every position YaRN at yarn_static_ctx over yarn_from */
    f->stages = 1, f->stage[0] = (Lfm3Stage){0, 0},
    ok = q4x_rope((uint32_t)g->head_dim, (uint32_t)g->rope_pairs, g->rope_base, f->yarn_from, o->yarn_static_ctx,
                  Q4X_THETA_HF, &f->rope_full[0]);
  for (uint32_t s = 0; ok && !o->yarn_static_ctx && s < nt; s++)
    f->stage[s] = table[s], f->stages = s + 1,
    ok = q4x_rope((uint32_t)g->head_dim, (uint32_t)g->rope_pairs, g->rope_base, f->yarn_from,
                  f->yarn_from * table[s].factor, Q4X_THETA_HF, &f->rope_full[s]);
  ok = ok && q4x_rope((uint32_t)g->head_dim, (uint32_t)g->head_dim / 2, g->rope_base_swa, g->native_ctx, g->native_ctx,
                     Q4X_THETA_HF, &f->rope_swa);
  if (!ok && !err[0]) /* a refusal before keeps its own message */
    osmos_fail(err, "the ropes refused (their shape, or a context past YaRN's reach)");
  ok = ok && (hipStreamCreate((hipStream_t *)&f->st) == hipSuccess || osmos_fail(err, "the step's stream not made"));
  ok = ok && upload(f, err) && df_make(f, o->dflash, err);
  ok = ok && joints(f, err) && (!o->lsa || lsa_load(f, o->lsa, err));
  if (ok && o->lsa)
    f->lsa_id = fnv_file(o->lsa, FNV_SEED);
  if (ok)
    f->head = lfm3_dev(f, w->embd), f->head_type = w->embd->type;
  ok = ok && (o->head == LFM3_HEAD_FILE || o->head == LFM3_HEAD_Q8_0 || osmos_fail(err, "a head this build does not make")) &&
       (o->head == LFM3_HEAD_FILE || head_q8(f, err));
  const GpuWant want[] = {
      {&f->tok, n * 4},          {&f->h, n * D * F},       {&f->xn, n * D * F},    {&f->y, n * D * F},
      {&f->bcx, n * 3 * D * F},  {&f->u, n * D * F},       {&f->qf, n * q * F},    {&f->kf, n * kv * F},
      {&f->vf, n * kv * F},      {&f->qr, n * q * F},      {&f->kr, n * kv * F},   {&f->g, n * g->n_heads * F},
      {&f->att, n * q * F},      {&f->ga, n * ff * F},     {&f->up, n * ff * F},   {&f->act, n * ff * F},
      {&f->attn_scratch, q35_attn_decode_scratch((uint32_t)g->n_heads, Q35_ATTN_DECODE_MAX) * F},
      {&f->xf, lr * D * F},      {&f->logits, lr * g->n_vocab * F}, {&f->hk, lr * D * F},
      {&f->ids, lr * 4},         {&f->argmax, ops_argmax_scratch((uint32_t)lr)},
      {&f->fa_scratch, q4x_attn_scratch_dim((uint32_t)g->n_heads, (uint32_t)g->n_kv_heads, (uint32_t)g->head_dim, f->max_ctx) * F},
      {&f->x16, (uint64_t)MM_MAX_TOKENS * (ff > 3 * D ? ff : 3 * D) * 2}, {&f->overflow, 4},
      {&f->jo, (2 * ff > q + 2 * kv + (uint64_t)g->n_heads ? 2 * ff : q + 2 * kv + (uint64_t)g->n_heads) * F},
  };
  ok = ok && (gpu_bufs_alloc(&f->bufs, f->ledger, want, sizeof want / sizeof *want) ||
              osmos_fail(err, "GPU memory refused (a step's buffers)"));
  ok = ok && gpu_bufs_host(&f->bufs, f->ledger, &f->pin_tok, n * 4) &&
       gpu_bufs_host(&f->bufs, f->ledger, &f->pin_logits, lr * g->n_vocab * F) &&
       (f->slot = malloc(n * 4)) &&
       (f->ovf = (OverflowFlag){f->overflow, 4, false}, overflow_reset(&f->ovf, err)) && hipDeviceSynchronize() == hipSuccess;
  if (!ok)
    return lfm3_forward_destroy(f), (err && !*err ? osmos_fail(err, "the forward not made") : false), NULL;
  return f;
}

void lfm3_forward_tap(Lfm3Forward *f, lfm3_tap_fn fn, void *arg)
{
  if (f)
    f->tap = fn, f->tap_arg = arg;
}
