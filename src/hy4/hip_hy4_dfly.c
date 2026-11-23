/* DFly2, Hy4-flash's drafter (osmos/hy4/forward.h: hy4_dfly_load, hy4_seq_draft; DESIGN.md, "0.11.0: DFly2"): the
   release's five draft layers read from its own safetensors (BF16 matrices, the small tensors as F32, kv_b split into
   the target's absorbed k_b / v_b as Q8_0), the draft block run as vLLM's HYV4DFlyQueryModel runs it - the anchor's
   embedding fused with the target's head input at the last accepted token, each layer reading its target sparse
   layer's picks and cache and the block's own rows, the final norm, then the hidden correction's chain through the
   target's head */
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/hy4_ops.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/gpu/upload.h"
#include "osmos/model/qtype.h"
#include "osmos/model/safetensors.h"
#include "hy4_impl.h"
#include "osmos/err.h"

/* the release's config.json: dfly_target_kv_layers (= dfly_target_index_layer), dfly_mask_token_id */
static const int TARGETS[HY4_DFLY_LAYERS] = {25, 29, 33, 37, 41};
#define MASK_TOKEN 120009
#define PREFIX "model.dfly_layers."
#define ROWS (HY4_DFLY_DRAFTS + 1)
#define READ_PIECE (64u << 20)

typedef struct Layer
{
  const void *q_a, *q_b, *kv_a, *gate, *o, *ffn_gate, *ffn_up, *ffn_down; /* BF16 on the GPU (the file's tensors) */
  void *k_b, *v_b;                                                       /* Q8_0, built from kv_b */
  float *in_norm, *q_a_norm, *kv_a_norm, *post_proj_norm, *ffn_norm, *sinks; /* F32 on the GPU */
} Layer;

struct Hy4Dfly
{
  Gguf *st;
  bool q8;           /* the matrices Q8_0 (ggml's rounding of the file's BF16), else as the file holds them */
  uint32_t wtype;    /* the layers' and the correction's matrices' type: QTYPE_Q8_0 or QTYPE_BF16 */
  uint32_t ehtype;   /* the fusion's: QTYPE_Q8_0 or QTYPE_F32 */
  void *head8;       /* q8: the target's head as Q8_0 for the drafts' picks (NULL: the target's own) */
  void *head4;       /* the drafts' head as Q4_0 (half head8's reads; NULL: head8's or the target's) */
  void **dev; /* the file's tensors on the GPU, by index (the matrices alone) */
  Layer layer[HY4_DFLY_LAYERS];
  float *eh;  /* the fusion's eh_proj, its rows times its gate (F32: gate * (W x) = (gate W) x) */
  float *enorm, *hnorm, *final_norm, *hidden_norm, *embed_norm;
  const void *corr_gate, *corr_up, *corr_down;
  uint32_t ff, corr_ff, cap;
  /* a draft's buffers (ROWS rows) */
  int32_t *ids, *pos, *sel, *n_sel, *pick;
  float *prob, *h, *xn, *cat, *y, *qr, *qres, *q, *kva, *gt, *qa, *olat, *ao, *out, *ga, *up, *act, *z, *e, *c;
  float *logits;
  int32_t *pin_ids, *pin_pos, *pin_pick;
  float *pin_prob;
};


/* a tensor of the file by its name under the prefix, of these dims (ggml's order), BF16 */
static const GgufTensor *need(const Gguf *g, const char *name, uint64_t d0, uint64_t d1, char *err)
{
  char full[160];
  snprintf(full, sizeof full, PREFIX "%s", name);
  const GgufTensor *t = gguf_tensor(g, full);
  if (!t || t->type != QTYPE_BF16 || t->dims[0] != d0 || (d1 ? t->n_dims != 2 || t->dims[1] != d1 : t->n_dims != 1))
  {
    char why[OSMOS_ERR];
    snprintf(why, sizeof why, "DFly2: %s missing, not BF16 or of another shape", full);
    return osmos_fail(err, "%s", why), NULL;
  }
  return t;
}

/* a tensor's BF16 values as F32 on the host (read from its file) */
float *hy4_host_f32(const Gguf *g, const GgufTensor *t)
{
  const uint64_t n = t->bytes / 2;
  uint16_t *b = malloc(t->bytes);
  float *x = malloc(n * sizeof *x);
  const int fd = open(g->paths[t->file], O_RDONLY);
  bool ok = b && x && fd >= 0;
  for (uint64_t at = 0; ok && at < t->bytes;)
  {
    const ssize_t r = pread(fd, (uint8_t *)b + at, t->bytes - at, (off_t)(t->offset + at));
    ok = r > 0, at += ok ? (uint64_t)r : 0;
  }
  if (fd >= 0)
    close(fd);
  for (uint64_t i = 0; ok && i < n; i++)
  {
    const uint32_t u = (uint32_t)b[i] << 16;
    memcpy(&x[i], &u, 4);
  }
  free(b);
  if (!ok)
    free(x), x = NULL;
  return x;
}

static bool up_f32(const float *x, uint64_t n, float **dev)
{
  return x && hy4_alloc(dev, n * F) && hipMemcpy(*dev, x, n * F, hipMemcpyHostToDevice) == hipSuccess;
}

/* a small BF16 tensor to the GPU as F32 */
static bool small(const Gguf *g, const GgufTensor *t, float **dev)
{
  float *x = t ? hy4_host_f32(g, t) : NULL;
  const bool ok = up_f32(x, t ? t->bytes / 2 : 0, dev);
  free(x);
  return ok;
}

/* ggml's quantize_row_q8_0_ref: a block's d = amax / 127 (an f16), each code round(x / d) */
static void q8_row(const float *x, uint32_t n, uint8_t *out)
{
  for (uint32_t b = 0; b < n / 32; b++)
  {
    float amax = 0;
    for (int i = 0; i < 32; i++)
      amax = fmaxf(amax, fabsf(x[b * 32 + i]));
    const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
    const _Float16 h = (_Float16)d;
    memcpy(out + b * 34, &h, 2);
    for (int i = 0; i < 32; i++)
      out[b * 34 + 2 + i] = (uint8_t)(int8_t)roundf(x[b * 32 + i] * id);
  }
}

/* kv_b [heads x (nope + v_dim)] x latent split as the target's GGUF holds them: k_b a head's nope rows transposed
   ([latent][nope]), v_b its v rows ([v_dim][latent]) - both Q8_0 (llama.cpp's converter's split) */
static bool split_kvb(const Gguf *g, const GgufTensor *t, uint32_t H, uint32_t nope, uint32_t vd, uint32_t lat,
                      void **k_b, void **v_b)
{
  float *kv = hy4_host_f32(g, t);
  const uint64_t kb_row = nope / 32 * 34, vb_row = lat / 32 * 34;
  const uint64_t kb = (uint64_t)H * lat * kb_row, vb = (uint64_t)H * vd * vb_row;
  uint8_t *hk = malloc(kb), *hv = malloc(vb);
  float *tmp = malloc((uint64_t)nope * sizeof *tmp);
  bool ok = kv && hk && hv && tmp;
  for (uint32_t h = 0; ok && h < H; h++)
  {
    const float *base = kv + (uint64_t)h * (nope + vd) * lat;
    for (uint32_t j = 0; j < lat; j++)
    {
      for (uint32_t i = 0; i < nope; i++)
        tmp[i] = base[(uint64_t)i * lat + j];
      q8_row(tmp, nope, hk + ((uint64_t)h * lat + j) * kb_row);
    }
    for (uint32_t r = 0; r < vd; r++)
      q8_row(base + (uint64_t)(nope + r) * lat, lat, hv + ((uint64_t)h * vd + r) * vb_row);
  }
  ok = ok && hy4_alloc(k_b, kb) && hy4_alloc(v_b, vb) && hipMemcpy(*k_b, hk, kb, hipMemcpyHostToDevice) == hipSuccess &&
       hipMemcpy(*v_b, hv, vb, hipMemcpyHostToDevice) == hipSuccess;
  free(kv), free(hk), free(hv), free(tmp);
  return ok;
}

/* an F32 matrix [rows][cols] as Q8_0 on the GPU */
bool hy4_q8_up(const float *x, uint64_t rows, uint32_t cols, void **dev)
{
  const uint64_t rb = cols / 32 * 34;
  uint8_t *q = x && cols % 32 == 0 ? malloc(rows * rb) : NULL;
  for (uint64_t r = 0; q && r < rows; r++)
    q8_row(x + r * cols, cols, q + r * rb);
  const bool ok = q && hy4_alloc(dev, rows * rb) && hipMemcpy(*dev, q, rows * rb, hipMemcpyHostToDevice) == hipSuccess;
  free(q);
  return ok;
}

/* an F32 matrix [rows][cols] as Q4_0 on the GPU (ggml's quantize_row_q4_0_ref: d = the block's signed largest / -8,
   a code min(15, x / d + 8.5), codes j and j + 16 a byte) - the drafts' head: its rounding moves their acceptance,
   never the output */
static bool q4_up(const float *x, uint64_t rows, uint32_t cols, void **dev)
{
  const uint64_t rb = cols / 32 * 18;
  uint8_t *q = x && cols % 32 == 0 ? malloc(rows * rb) : NULL;
  for (uint64_t r = 0; q && r < rows; r++)
    for (uint32_t b = 0; b < cols / 32; b++)
    {
      const float *v = x + r * cols + b * 32;
      float amax = 0.0f, max = 0.0f;
      for (int i = 0; i < 32; i++)
        if (fabsf(v[i]) > amax)
          amax = fabsf(v[i]), max = v[i];
      const float dd = max / -8.0f, id = dd ? 1.0f / dd : 0.0f;
      uint8_t *o = q + r * rb + b * 18;
      const _Float16 h = (_Float16)dd;
      memcpy(o, &h, 2);
      for (int i = 0; i < 16; i++)
      {
        const int a = (int)(v[i] * id + 8.5f), c = (int)(v[i + 16] * id + 8.5f);
        o[2 + i] = (uint8_t)((a > 15 ? 15 : a) | (c > 15 ? 15 : c) << 4);
      }
    }
  const bool ok = q && hy4_alloc(dev, rows * rb) && hipMemcpy(*dev, q, rows * rb, hipMemcpyHostToDevice) == hipSuccess;
  free(q);
  return ok;
}

/* a matrix of the file to the GPU: as it is (BF16), or q8 its Q8_0 */
static bool matrix(struct Hy4Dfly *d, const GgufTensor *t, void *pin, const void **dev)
{
  void **slot = &d->dev[t - d->st->tensors];
  if (!*slot)
  {
    if (d->q8)
    {
      float *x = hy4_host_f32(d->st, t);
      const bool ok = hy4_q8_up(x, t->dims[1], (uint32_t)t->dims[0], slot);
      free(x);
      if (!ok)
        return false;
    }
    else if (gpu_read_tensor(d->st, t, pin, READ_PIECE, slot))
      return false;
  }
  *dev = *slot;
  return true;
}

void hy4_dfly_free(Hy4Forward *f)
{
  struct Hy4Dfly *d = f ? f->dfly : NULL;
  if (!d)
    return;
  for (size_t i = 0; d->dev && i < d->st->n_tensors; i++)
    hipFree(d->dev[i]);
  free(d->dev);
  for (int i = 0; i < HY4_DFLY_LAYERS; i++)
  {
    Layer *L = &d->layer[i];
    hipFree(L->k_b), hipFree(L->v_b), hipFree(L->in_norm), hipFree(L->q_a_norm), hipFree(L->kv_a_norm);
    hipFree(L->post_proj_norm), hipFree(L->ffn_norm), hipFree(L->sinks);
  }
  void *b[] = {d->eh,   d->enorm, d->hnorm, d->final_norm, d->hidden_norm, d->embed_norm, d->ids, d->pos, d->sel,
               d->n_sel, d->pick, d->prob, d->h, d->xn, d->cat, d->y, d->qr, d->qres, d->q, d->kva, d->gt, d->qa,
               d->olat, d->ao, d->out, d->ga, d->up, d->act, d->z, d->e, d->c, d->logits};
  for (size_t i = 0; i < sizeof b / sizeof *b; i++)
    hipFree(b[i]);
  hipHostFree(d->pin_ids), hipHostFree(d->pin_pos), hipHostFree(d->pin_pick), hipHostFree(d->pin_prob);
  hipFree(d->head8), hipFree(d->head4);
  gguf_destroy(d->st);
  free(d);
  f->dfly = NULL;
  for (int l = 0; l < HY4_MAX_LAYERS; l++)
    f->dfly_slot[l] = -1;
}

bool hy4_dfly_load(Hy4Forward *f, const char *path, bool q8, char err[OSMOS_ERR])
{
  if (!f || !path)
    return osmos_fail(err, "no forward or drafter path");
  if (f->dfly)
    return osmos_fail(err, "DFly2: already loaded");
  const Hy4Geometry *g = &f->w->geo;
  const uint32_t D = (uint32_t)g->n_embd, ql = (uint32_t)g->q_lora, lat = (uint32_t)g->kv_lora;
  const uint32_t nope = (uint32_t)g->nope_dim, rd = (uint32_t)g->rope_dim, vd = (uint32_t)g->v_dim;
  for (int i = 0; i < HY4_DFLY_LAYERS; i++)
    if (TARGETS[i] >= g->n_layers || g->sliding[TARGETS[i]])
      return osmos_fail(err, "DFly2: a target layer not one of this model's sparse layers");
  if (MASK_TOKEN >= g->n_vocab)
    return osmos_fail(err, "DFly2: its mask token past this model's vocabulary");
  if (f->max_tokens < ROWS) /* a draft's block in the step's buffers (its MLA table max_tokens x heads - 0.11.2) */
    return osmos_fail(err, "DFly2: a forward's step smaller than a draft block");
  const uint32_t H = (uint32_t)g->n_heads[TARGETS[0]];
  struct Hy4Dfly *d = calloc(1, sizeof *d);
  if (!d)
    return osmos_fail(err, "out of memory");
  f->dfly = d;
  d->q8 = q8, d->wtype = q8 ? QTYPE_Q8_0 : QTYPE_BF16, d->ehtype = q8 ? QTYPE_Q8_0 : QTYPE_F32;
  if (!(d->st = st_create(path, err)))
    return hy4_dfly_free(f), false;
  void *pin = NULL;
  if (!(d->dev = calloc(d->st->n_tensors, sizeof *d->dev)) || hipHostMalloc(&pin, READ_PIECE, 0) != hipSuccess)
    return hy4_dfly_free(f), osmos_fail(err, "out of memory (DFly2's table or a pinned piece)");
  bool ok = true;
  { /* the FFN's width from the file (9,216 in the release) */
    const GgufTensor *gp = gguf_tensor(d->st, PREFIX "0.mlp.gate_proj.weight");
    d->ff = gp && gp->n_dims == 2 ? (uint32_t)gp->dims[1] : 0;
    ok = d->ff || osmos_fail(err, "DFly2: no layer 0 FFN gate (not the drafter's file)");
  }
  for (int i = 0; ok && i < HY4_DFLY_LAYERS; i++)
  {
    Layer *L = &d->layer[i];
    char n[96];
#define T(name, d0, d1) (snprintf(n, sizeof n, "%d." name, i), need(d->st, n, (d0), (d1), err))
    const GgufTensor *q_a = T("self_attn.q_a_proj.weight", D, ql), *q_b = T("self_attn.q_b_proj.weight", ql, H * (nope + rd));
    const GgufTensor *kv_a = T("self_attn.kv_a_proj_with_mqa.weight", D, lat + rd);
    const GgufTensor *kv_b = T("self_attn.kv_b_proj.weight", lat, H * (nope + vd));
    const GgufTensor *gate = T("self_attn.linear_gate.weight", D, H * vd), *o = T("self_attn.o_proj.weight", H * vd, D);
    const GgufTensor *fg = T("mlp.gate_proj.weight", D, d->ff), *fu = T("mlp.up_proj.weight", D, d->ff);
    const GgufTensor *fd = T("mlp.down_proj.weight", d->ff, D);
    const GgufTensor *inn = T("input_layernorm.weight", D, 0), *qan = T("self_attn.q_a_layernorm.weight", ql, 0);
    const GgufTensor *kvn = T("self_attn.kv_a_layernorm.weight", lat, 0);
    const GgufTensor *ppn = T("self_attn.post_proj_norm.weight", D, 0), *fn = T("post_attention_layernorm.weight", D, 0);
    const GgufTensor *sk = T("self_attn.learnable_sink_param", H, 0);
#undef T
    ok = q_a && q_b && kv_a && kv_b && gate && o && fg && fu && fd && inn && qan && kvn && ppn && fn && sk;
    ok = ok && ((matrix(d, q_a, pin, &L->q_a) && matrix(d, q_b, pin, &L->q_b) && matrix(d, kv_a, pin, &L->kv_a) &&
                 matrix(d, gate, pin, &L->gate) && matrix(d, o, pin, &L->o) && matrix(d, fg, pin, &L->ffn_gate) &&
                 matrix(d, fu, pin, &L->ffn_up) && matrix(d, fd, pin, &L->ffn_down) && small(d->st, inn, &L->in_norm) &&
                 small(d->st, qan, &L->q_a_norm) && small(d->st, kvn, &L->kv_a_norm) &&
                 small(d->st, ppn, &L->post_proj_norm) && small(d->st, fn, &L->ffn_norm) && small(d->st, sk, &L->sinks) &&
                 split_kvb(d->st, kv_b, H, nope, vd, lat, &L->k_b, &L->v_b)) ||
                osmos_fail(err, "DFly2: a layer's tensors not read into GPU memory"));
  }
  const GgufTensor *eh = ok ? need(d->st, "mtp_fusion.eh_proj.weight", 2 * D, D, err) : NULL;
  const GgufTensor *eg = ok ? need(d->st, "mtp_fusion.gate", D, 0, err) : NULL;
  const GgufTensor *en = ok ? need(d->st, "mtp_fusion.enorm.weight", D, 0, err) : NULL;
  const GgufTensor *hn = ok ? need(d->st, "mtp_fusion.hnorm.weight", D, 0, err) : NULL;
  const GgufTensor *fnm = ok ? need(d->st, "final_norm.weight", D, 0, err) : NULL;
  const GgufTensor *chn = ok ? need(d->st, "hidden_correction.hidden_norm.weight", D, 0, err) : NULL;
  const GgufTensor *cen = ok ? need(d->st, "hidden_correction.embed_norm.weight", D, 0, err) : NULL;
  const GgufTensor *cgp = gguf_tensor(d->st, PREFIX "hidden_correction.gate_proj.weight");
  d->corr_ff = cgp && cgp->n_dims == 2 ? (uint32_t)cgp->dims[1] : 0;
  const GgufTensor *cg = ok && d->corr_ff ? need(d->st, "hidden_correction.gate_proj.weight", 2 * D, d->corr_ff, err) : NULL;
  const GgufTensor *cu = cg ? need(d->st, "hidden_correction.up_proj.weight", 2 * D, d->corr_ff, err) : NULL;
  const GgufTensor *cd = cu ? need(d->st, "hidden_correction.down_proj.weight", d->corr_ff, D, err) : NULL;
  ok = ok && eh && eg && en && hn && fnm && chn && cen && cg && cu && cd;
  if (ok)
  { /* the fusion's matrix with its gate folded in: row r times gate[r] */
    float *w = hy4_host_f32(d->st, eh), *gv = hy4_host_f32(d->st, eg);
    for (uint64_t r = 0; w && gv && r < D; r++)
      for (uint64_t c = 0; c < 2ull * D; c++)
        w[r * 2 * D + c] *= gv[r];
    ok = (w && gv && (q8 ? hy4_q8_up(w, D, 2 * D, (void **)&d->eh) : up_f32(w, (uint64_t)D * 2 * D, &d->eh))) ||
         osmos_fail(err, "DFly2: the fusion not made");
    free(w), free(gv);
  }
  ok = ok && ((small(d->st, en, &d->enorm) && small(d->st, hn, &d->hnorm) && small(d->st, fnm, &d->final_norm) &&
               small(d->st, chn, &d->hidden_norm) && small(d->st, cen, &d->embed_norm) &&
               matrix(d, cg, pin, &d->corr_gate) && matrix(d, cu, pin, &d->corr_up) &&
               matrix(d, cd, pin, &d->corr_down)) ||
              osmos_fail(err, "DFly2: its shared tensors not read into GPU memory"));
  hipHostFree(pin);
  const GgufTensor *out = f->w->output;
  if (ok && q8 && !f->head8 && (out->type == QTYPE_BF16 || out->type == QTYPE_F32))
  { /* the drafts' head: the target's own as Q8_0 (a draft's precision moves its acceptance, never the output) */
    float *x = out->type == QTYPE_BF16 ? hy4_host_f32(f->w->model, out) : NULL;
    ok = (x && hy4_q8_up(x, out->dims[1], (uint32_t)out->dims[0], &d->head8)) ||
         osmos_fail(err, "DFly2: the head's Q8_0 copy not made (BF16 heads alone)");
    free(x);
  }
  if (ok && q8 && !getenv("HY4_DFLY_HEAD8") && (out->type == QTYPE_BF16 || out->type == QTYPE_F32))
  { /* the engine's (q8) drafts' head Q4_0 (a round reads it a draft: 5 x 0.39 GB at Q8_0). Q3_K at 1K, 128 tokens, A/B in one
       session (2026-11-19): code 24.1-24.6 -> 25.6-26.2 tok/s (the same 87 accepted, 106 drafts against 112), prose
       27.5-28.1 -> 28.3-29.0 (the same 96); +209 MB. HY4_DFLY_HEAD8=1: head8's / the target's */
    float *x = out->type == QTYPE_BF16 ? hy4_host_f32(f->w->model, out) : NULL;
    ok = (x && q4_up(x, out->dims[1], (uint32_t)out->dims[0], &d->head4)) || osmos_fail(err, "DFly2: the head's Q4_0 copy");
    free(x);
  }
  /* the buffers: ROWS rows each */
  const uint64_t R = ROWS, row = lat + rd, qk = nope + rd, V = (uint64_t)g->n_vocab, top = g->ix_topk / g->ix_pool;
  d->cap = (uint32_t)(top + (ROWS + g->ix_pool - 1) / g->ix_pool + 1);
  const uint64_t ffm = d->ff > d->corr_ff ? d->ff : d->corr_ff;
  ok = ok && hy4_alloc(&d->ids, R * 4) && hy4_alloc(&d->pos, R * 4) && hy4_alloc(&d->sel, R * d->cap * 4) &&
       hy4_alloc(&d->n_sel, R * 4) && hy4_alloc(&d->pick, R * 4) && hy4_alloc(&d->prob, R * F) &&
       hy4_alloc(&d->h, R * D * F) && hy4_alloc(&d->xn, R * D * F) && hy4_alloc(&d->cat, 2ull * D * F) &&
       hy4_alloc(&d->y, R * D * F) && hy4_alloc(&d->qr, R * ql * F) && hy4_alloc(&d->qres, R * ql * F) &&
       hy4_alloc(&d->q, R * H * qk * F) && hy4_alloc(&d->kva, R * row * F) && hy4_alloc(&d->gt, R * H * vd * F) &&
       hy4_alloc(&d->qa, R * H * row * F) && hy4_alloc(&d->olat, R * H * lat * F) && hy4_alloc(&d->ao, R * H * vd * F) &&
       hy4_alloc(&d->out, R * D * F) && hy4_alloc(&d->ga, R * ffm * F) && hy4_alloc(&d->up, R * ffm * F) &&
       hy4_alloc(&d->act, R * ffm * F) && hy4_alloc(&d->z, R * D * F) && hy4_alloc(&d->e, D * F) &&
       hy4_alloc(&d->c, D * F) && hy4_alloc(&d->logits, V * F) &&
       hipHostMalloc((void **)&d->pin_ids, R * 4, 0) == hipSuccess &&
       hipHostMalloc((void **)&d->pin_pos, R * 4, 0) == hipSuccess &&
       hipHostMalloc((void **)&d->pin_pick, R * 4, 0) == hipSuccess &&
       hipHostMalloc((void **)&d->pin_prob, R * F, 0) == hipSuccess;
  if (!ok)
  {
    if (!err[0])
      osmos_fail(err, "GPU memory refused (DFly2's buffers)");
    return hy4_dfly_free(f), false;
  }
  for (int i = 0; i < HY4_DFLY_LAYERS; i++)
    f->dfly_slot[TARGETS[i]] = i;
  return true;
}

/* y = W x for n rows, W the file's BF16 [rows x cols] or an F32 one */
static bool mat(Hy4Forward *f, uint32_t type, const void *w, uint32_t rows, uint32_t cols, const float *x, uint32_t n,
                float *y)
{
  return mv_strided(type, w, rows, cols, x, cols, n, y, rows, f->st);
}

int hy4_seq_draft(Hy4Forward *f, Hy4Seq *s, int32_t y, uint32_t most, float conf_min, float cum_min, int32_t *drafts,
                  float *conf, char err[OSMOS_ERR])
{
  struct Hy4Dfly *d = f ? f->dfly : NULL;
  if (!d || !s || s->f != f || s->broken || !drafts || !conf)
    return 0;
  const Hy4Geometry *g = &f->w->geo;
  const int32_t anchor = s->pos, i = anchor - 1 - s->dpos0;
  if (anchor < 1 || y < 0 || y >= g->n_vocab || !s->dkeep || i < 0 || i >= (int32_t)s->dkeep || !s->dhid_ok[i])
    return 0;
  if (most > HY4_DFLY_DRAFTS)
    most = HY4_DFLY_DRAFTS;
  if ((uint64_t)anchor + most + 1 > f->max_ctx)
    most = anchor + 1 < (int32_t)f->max_ctx ? f->max_ctx - (uint32_t)anchor - 1 : 0;
  if (!most)
    return 0;
  const uint32_t R = most + 1, D = (uint32_t)g->n_embd, ql = (uint32_t)g->q_lora, lat = (uint32_t)g->kv_lora;
  const uint32_t nope = (uint32_t)g->nope_dim, rd = (uint32_t)g->rope_dim, vd = (uint32_t)g->v_dim, qk = nope + rd;
  const uint32_t row = lat + rd, H = (uint32_t)g->n_heads[TARGETS[0]], V = (uint32_t)g->n_vocab;
  const uint32_t top = (uint32_t)(g->ix_topk / g->ix_pool), pool = (uint32_t)g->ix_pool;
  const float eps = g->rms_eps, scale = 1.0f / sqrtf((float)qk);
  const GgufTensor *emb = f->w->embed;
  const uint64_t erow = emb->bytes / emb->dims[1];
  for (uint32_t j = 0; j < R; j++)
    d->pin_ids[j] = j ? MASK_TOKEN : y, d->pin_pos[j] = anchor + (int32_t)j;
  bool ok = hipMemcpyAsync(d->ids, d->pin_ids, R * 4, hipMemcpyHostToDevice, f->st) == hipSuccess &&
            hipMemcpyAsync(d->pos, d->pin_pos, R * 4, hipMemcpyHostToDevice, f->st) == hipSuccess &&
            q4x_rows(emb->type, hy4_dev(f, emb), erow, D, d->ids, R, d->h, f->st);
  /* the fusion on the anchor's row: h0 += (gate W) [enorm(e0); hnorm(target hidden)] */
  ok = ok && ops_rms(d->h, D, D, d->enorm, eps, 1, d->cat, 2 * D, f->st) &&
       ops_rms(s->dhid + (uint64_t)i * D, D, D, d->hnorm, eps, 1, d->cat + D, 2 * D, f->st) &&
       mat(f, d->ehtype, d->eh, D, 2 * D, d->cat, 1, d->y) && ops_add(d->h, d->y, D, f->st);
  for (int li = 0; ok && li < HY4_DFLY_LAYERS; li++)
  {
    const Layer *L = &d->layer[li];
    const int tl = TARGETS[li];
    ok = ops_rms(d->h, D, D, L->in_norm, eps, R, d->xn, D, f->st) &&
         mat(f, d->wtype, L->q_a, ql, D, d->xn, R, d->qr) && mat(f, d->wtype, L->kv_a, row, D, d->xn, R, d->kva) &&
         mat(f, d->wtype, L->gate, H * vd, D, d->xn, R, d->gt) &&
         ops_rms(d->qr, ql, ql, L->q_a_norm, eps, R, d->qres, ql, f->st) &&
         mat(f, d->wtype, L->q_b, H * qk, ql, d->qres, R, d->q) &&
         hy4_mla_kv(d->kva, L->kv_a_norm, lat, rd, eps, f->freqs_sparse, d->pos, R, s->cache[tl], 0, f->kv8, f->st) &&
         hy4_mla_q(d->q, L->k_b, H, nope, lat, rd, f->freqs_sparse, d->pos, R, d->qa, f->table, f->st) &&
         hy4_dfly_sel(s->dsel + ((uint64_t)li * HY4_DFLY_KEEP + (uint32_t)i) * top,
                      s->dn + (uint64_t)li * HY4_DFLY_KEEP + (uint32_t)i, anchor, pool, top, R, d->cap, d->sel, d->n_sel,
                      f->st) &&
         hy4_mla_attn(d->qa, s->cache[tl], 0, f->kv8, L->sinks, H, lat, rd, 0, d->sel, d->n_sel, d->cap, pool, scale, d->pos, R,
                      d->olat, f->attn_part, false, f->st) &&
         hy4_mla_out(d->olat, L->v_b, d->gt, H, lat, vd, R, d->ao, f->table, f->st) &&
         mat(f, d->wtype, L->o, D, H * vd, d->ao, R, d->out) &&
         ops_rms_add_rms(d->h, d->out, L->post_proj_norm, L->ffn_norm, eps, D, R, d->xn, f->st) &&
         mat(f, d->wtype, L->ffn_gate, d->ff, D, d->xn, R, d->ga) &&
         mat(f, d->wtype, L->ffn_up, d->ff, D, d->xn, R, d->up) &&
         ops_swiglu(d->ga, d->up, d->ff, INFINITY, NULL, R, d->act, f->st) &&
         mat(f, d->wtype, L->ffn_down, D, d->ff, d->act, R, d->y) && ops_add(d->h, d->y, (uint64_t)R * D, f->st);
  }
  ok = ok && ops_rms(d->h, D, D, d->final_norm, eps, R, d->z, D, f->st);
  /* the chain: draft j from row j's hidden corrected by the token before it (the anchor's, then each draft's) */
  for (uint32_t j = 1; ok && j < R; j++)
  {
    const int32_t *prev = j == 1 ? d->ids : d->pick + (j - 2);
    ok = ops_rms(d->z + (uint64_t)j * D, D, D, d->hidden_norm, eps, 1, d->cat, 2 * D, f->st) &&
         q4x_rows(emb->type, hy4_dev(f, emb), erow, D, prev, 1, d->e, f->st) &&
         ops_rms(d->e, D, D, d->embed_norm, eps, 1, d->cat + D, 2 * D, f->st) &&
         mat(f, d->wtype, d->corr_gate, d->corr_ff, 2 * D, d->cat, 1, d->ga) &&
         mat(f, d->wtype, d->corr_up, d->corr_ff, 2 * D, d->cat, 1, d->up) &&
         ops_swiglu(d->ga, d->up, d->corr_ff, INFINITY, NULL, 1, d->act, f->st) &&
         mat(f, d->wtype, d->corr_down, D, d->corr_ff, d->act, 1, d->c) &&
         ops_add(d->c, d->z + (uint64_t)j * D, D, f->st) &&
         (d->head4 ? mat(f, QTYPE_Q4_0, d->head4, V, D, d->c, 1, d->logits)
          : d->head8 || f->head8 ? mat(f, QTYPE_Q8_0, d->head8 ? d->head8 : f->head8, V, D, d->c, 1, d->logits)
                   : hy4_mat(f, f->w->output, V, D, d->c, 1, d->logits, false)) &&
         hy4_dfly_pick(d->logits, V, d->pick + (j - 1), d->prob + (j - 1), f->st);
  }
  ok = ok && hipMemcpyAsync(d->pin_pick, d->pick, most * 4, hipMemcpyDeviceToHost, f->st) == hipSuccess &&
       hipMemcpyAsync(d->pin_prob, d->prob, most * F, hipMemcpyDeviceToHost, f->st) == hipSuccess &&
       hipStreamSynchronize(f->st) == hipSuccess;
  if (!ok)
    return osmos_fail(err, "DFly2: a GPU call failed (a draft)"), -1;
  int k = 0;
  float cum = 1.0f;
  for (; k < (int)most; k++)
  {
    cum *= d->pin_prob[k];
    if (d->pin_prob[k] < conf_min || cum < cum_min)
      break;
    drafts[k] = d->pin_pick[k], conf[k] = d->pin_prob[k];
  }
  return k;
}
