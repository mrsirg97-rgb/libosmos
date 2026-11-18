/* the hy4 forward made (osmos/hy4/forward.h): the model's tensors read into GPU memory but the Engram table (its rows
   read a step at a time) and the drafter's layers, iHC's mixes as F32, the ropes' tables, a step's buffers - and
   unmade */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/hy4_ops.h"
#include "osmos/gpu/mm.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/quant.h"
#include "osmos/gpu/upload.h"
#include "osmos/model/fnv.h"
#include "osmos/model/qtype.h"
#include "hy4_impl.h"

#define READ_PIECE (64ull << 20)

bool hy4_fail(char *err, const char *why)
{
  if (err)
    snprintf(err, HY4_ERR, "%s", why);
  return false;
}

bool hy4_alloc(void *p, uint64_t bytes) { return hipMalloc((void **)p, bytes ? bytes : 1) == hipSuccess; }

void hy4_forward_tap(Hy4Forward *f, hy4_tap_fn fn, void *arg) { f->tap = fn, f->tap_arg = arg; }

static bool resident(const Hy4Weights *w, const GgufTensor *t)
{
  if (t == w->engram_table)
    return false;
  return !(strncmp(t->name, "blk.", 4) == 0 && atoi(t->name + 4) >= w->geo.n_layers); /* the drafter's: not ours */
}

static bool upload(Hy4Forward *f, char *err)
{
  const Gguf *g = f->w->model;
  void *pin = NULL;
  if (!(f->dev = calloc(g->n_tensors, sizeof *f->dev)) || hipHostMalloc(&pin, READ_PIECE, 0) != hipSuccess)
    return hy4_fail(err, "out of memory (the weights' table, or a pinned piece)");
  bool ok = true;
  for (size_t i = 0; ok && i < g->n_tensors; i++)
  {
    const GgufTensor *t = &g->tensors[i];
    if (!resident(f->w, t))
      continue;
    const char *why = gpu_read_tensor(g, t, pin, READ_PIECE, &f->dev[i]);
    ok = !why;
    if (!ok && err)
      snprintf(err, HY4_ERR, "%s: not read into GPU memory (%s)", t->name, why);
  }
  hipHostFree(pin);
  return ok;
}

/* a matrix the kernels read as F32 (iHC's mixes: Q8_0 in the file) */
static bool as_f32(Hy4Forward *f, const GgufTensor *t, float **out)
{
  const uint64_t n = t->dims[0] * t->dims[1];
  return hy4_alloc(out, n * F) && quant_dequant(t->type, hy4_dev(f, t), n, *out, NULL);
}

/* a rope's frequencies: 1 / base^(2i / dim), as the reference computes them (F32) */
static bool freqs(float base, uint32_t dim, float **out)
{
  float h[256];
  for (uint32_t i = 0; i < dim / 2; i++)
    h[i] = 1.0f / powf(base, (float)(2 * i) / (float)dim);
  return dim / 2 <= 256 && hy4_alloc(out, dim / 2 * F) && hipMemcpy(*out, h, dim / 2 * F, hipMemcpyHostToDevice) == hipSuccess;
}

static bool engram_reader(Hy4Forward *f, const int *fds, char *err)
{
  const Hy4Weights *w = f->w;
  const Hy4Geometry *g = &w->geo;
  EngramShape shape;
  char eerr[ENGRAM_ERR] = "";
  if (!hy4_engram_shape(w, &shape, err))
    return false;
  if (!(f->eg = engram_create_shape(&shape, eerr)))
    return hy4_fail(err, eerr);
  const GgufKv *offs = gguf_kv(w->model, "hy_v4_flash.ple.head_offsets");
  const int cols = (g->engram_ngram - 1) * g->engram_heads;
  for (int e = 0; e < g->n_engram; e++)
  {
    int64_t v = 0;
    if (!offs || !gguf_int(offs, (size_t)e * cols, &v) || v < 0)
      return hy4_fail(err, "hy_v4_flash.ple.head_offsets: an Engram layer's first row");
    f->eg_first[e] = (uint64_t)v;
  }
  const GgufTensor *t = w->engram_table;
  f->eg_fd = fds[t->file];
  f->eg_row_bytes = t->bytes / t->dims[1];
  const uint64_t rows = (uint64_t)f->max_tokens * g->n_engram * cols;
  return ((f->eg_read = rowread_create(HY4_ROW_THREADS)) && (f->eg_off = malloc(rows * 8)) &&
          hipHostMalloc((void **)&f->eg_stage, rows * f->eg_row_bytes, 0) == hipSuccess &&
          hy4_alloc(&f->eg_raw, rows * f->eg_row_bytes) && hy4_alloc(&f->eg_rows, rows * g->engram_dim * F)) ||
         hy4_fail(err, "out of memory (the Engram rows' buffers)");
}

void hy4_forward_destroy(Hy4Forward *f)
{
  if (!f)
    return;
  if (f->st)
    hipStreamSynchronize((hipStream_t)f->st), hipStreamDestroy((hipStream_t)f->st);
  for (size_t i = 0; f->dev && i < f->w->model->n_tensors; i++)
    hipFree(f->dev[i]);
  free(f->dev);
  for (int l = 0; l < HY4_MAX_LAYERS; l++)
    hipFree(f->fn_attn[l]), hipFree(f->fn_ffn[l]);
  void *b[] = {f->fn_out, f->freqs_swa, f->freqs_sparse, f->eg_raw, f->eg_rows, f->h, f->xn, f->y, f->post, f->hcs, f->table, f->attn_part, f->wtok, f->qr,
               f->qres, f->q, f->kva, f->gt, f->qa, f->olat, f->ao, f->out, f->ixk, f->ixkn, f->ixq, f->ixw, f->scores, f->sel,
               f->n_sel, f->x16, f->xe16, f->overflow, f->rx, f->ry, f->rxd, f->ryd, f->rcnt, f->rw, f->rgidx, f->rhidx,
               f->emb, f->ekv, f->ga, f->up, f->act, f->logit_r, f->rin, f->xg, f->eg_g, f->eg_u, f->eg_h,
               f->eg_y, f->wsel, f->rsum, f->sh, f->idx, f->gidx, f->sidx, f->cnt, f->xrow, f->wptr, f->tok, f->pos,
               f->hx, f->logits, f->ids, f->argmax};
  for (size_t i = 0; i < sizeof b / sizeof *b; i++)
    hipFree(b[i]);
  hipHostFree(f->eg_stage), hipHostFree(f->pin_tok), hipHostFree(f->pin_pos), hipHostFree(f->pin_logits), hipHostFree(f->pin_flags);
  rowread_destroy(f->eg_read);
  engram_destroy(f->eg);
  free(f->eg_off), free(f->slot), free(f->h_idx), free(f->h_gidx), free(f->h_sidx);
  free(f->h_w), free(f->h_wsel);
  free(f->h_rx), free(f->h_ry), free(f->h_rxd), free(f->h_ryd), free(f->h_rcnt), free(f->h_rw), free(f->rkey);
  free(f->h_rgidx), free(f->h_rhidx), free(f->h_stage);
  free(f);
}

Hy4Forward *hy4_forward_create(const Hy4Weights *w, const int *fds, const Hy4Options *o, char err[HY4_ERR])
{
  if (!w || !fds || !o || !o->max_tokens || o->max_tokens > o->max_ctx)
    return hy4_fail(err, "no weights, files or options, or a step past the context"), NULL;
  if (w->geo.engram_ngram - 1 > HY4_STATE_HISTORY)
    return hy4_fail(err, "an n-gram longer than a saved state's history keeps"), NULL;
  Hy4Forward *f = calloc(1, sizeof *f);
  if (!f)
    return hy4_fail(err, "out of memory"), NULL;
  const Hy4Geometry *g = &w->geo;
  f->w = w, f->max_tokens = o->max_tokens, f->max_ctx = o->max_ctx;
  f->logit_rows = o->logit_rows && o->logit_rows < o->max_tokens ? o->logit_rows : o->max_tokens;
  f->ring = (uint32_t)g->window + o->max_tokens + HY4_MAX_BACK;
  f->eg_fd = -1;
  int hmax = 0;
  for (int l = 0; l < g->n_layers; l++)
    hmax = g->n_heads[l] > hmax ? g->n_heads[l] : hmax;
  const uint64_t n = o->max_tokens, lr = f->logit_rows, D = (uint64_t)g->n_embd, hc = (uint64_t)g->hc;
  const uint64_t H = (uint64_t)hmax, qk = (uint64_t)(g->nope_dim + g->rope_dim), row = (uint64_t)(g->kv_lora + g->rope_dim);
  const uint64_t K = (uint64_t)g->n_used, L = (uint64_t)g->latent, ffe = (uint64_t)g->ff_exp;
  const uint64_t eg_in = (uint64_t)(g->engram_ngram - 1) * g->engram_heads * g->engram_dim;
  const uint64_t entries = n * K; /* mv_group_tok's entries at most: a token's picks each its own at worst */
  /* (xg and eg_h a chunk's rows past the last: mv_group_tok reads an entry's MV_MAX_TOKENS rows, uses its count) */
  const struct
  {
    void *p;
    uint64_t bytes;
  } want[] = {
      {&f->h, n * hc * D * F}, {&f->xn, n * D * F}, {&f->y, n * D * F}, {&f->post, n * hc * F}, {&f->hcs, hy4_hc_scratch((uint32_t)n) * F},
      {&f->table, hy4_mla_table_bytes((uint32_t)n, (uint32_t)H)}, {&f->attn_part, hy4_mla_attn_scratch((uint32_t)H) * F},
      {&f->qr, n * g->q_lora * F}, {&f->qres, n * g->q_lora * F}, {&f->q, n * H * qk * F}, {&f->kva, n * row * F},
      {&f->gt, n * H * g->v_dim * F}, {&f->qa, n * H * row * F}, {&f->olat, n * H * g->kv_lora * F},
      {&f->ao, n * H * g->v_dim * F}, {&f->out, n * D * F}, {&f->ixk, n * g->ix_dim * F}, {&f->ixkn, n * g->ix_dim * F},
      {&f->ixq, n * g->ix_heads * g->ix_dim * F}, {&f->ixw, n * g->ix_heads * F},
      {&f->scores, n * (o->max_ctx / g->ix_pool + 1) * F}, {&f->sel, n * (g->ix_topk / g->ix_pool) * 4},
      {&f->n_sel, n * 4}, {&f->emb, n * eg_in * F}, {&f->ekv, n * (hc + 1) * D * F}, {&f->ga, n * g->n_ff * F},
      {&f->up, n * g->n_ff * F}, {&f->act, n * g->n_ff * F}, {&f->logit_r, n * g->n_experts * F}, {&f->wtok, n * K * F},
      {&f->rin, n * L * F}, {&f->xg, (n * K + MV_MAX_TOKENS) * L * F}, {&f->eg_g, n * K * ffe * F},
      {&f->eg_u, n * K * ffe * F}, {&f->eg_h, (n * K + MV_MAX_TOKENS) * ffe * F}, {&f->eg_y, n * K * L * F}, {&f->wsel, n * K * F}, {&f->rsum, n * L * F},
      {&f->sh, n * D * F}, {&f->idx, n * K * 4}, {&f->gidx, n * K * 4}, {&f->sidx, n * K * 4},
      {&f->cnt, entries * 4}, {&f->xrow, entries * 4}, {&f->wptr, 3 * entries * sizeof(void *)},
      {&f->tok, n * 4}, {&f->pos, n * 4}, {&f->hx, lr * hc * D * F}, {&f->logits, lr * g->n_vocab * F},
      {&f->ids, lr * 4}, {&f->argmax, ops_argmax_scratch((uint32_t)lr)},
  };
  /* the read math's: the widest matrix input an mm reads (o_proj's heads x v_dim, the dense down's n_ff, Engram's) */
  uint64_t wide = (uint64_t)g->n_ff > H * g->v_dim ? (uint64_t)g->n_ff : H * g->v_dim;
  wide = wide > H * qk ? wide : H * qk, wide = wide > H * g->kv_lora ? wide : H * g->kv_lora;
  const uint64_t ecap = hy4_read_entries(n, g), scap = hy4_read_slots(n, g);
  const uint64_t xe = L > ffe ? L : ffe;
  const struct
  {
    void *p;
    uint64_t bytes;
  } read_want[] = {
      {&f->x16, (uint64_t)MM_MAX_TOKENS * wide * 2}, {&f->xe16, scap * xe * 2}, {&f->overflow, 8},
      {&f->rx, ecap * 4}, {&f->ry, ecap * 4}, {&f->rxd, ecap * 4}, {&f->ryd, ecap * 4}, {&f->rcnt, ecap * 4},
      {&f->rw, 3 * ecap * sizeof(void *)}, {&f->rgidx, scap * 4}, {&f->rhidx, scap * 4},

  };
  bool ok = (uint64_t)hmax <= HY4_MAX_HEADS || hy4_fail(err, "more heads than this build's tables");
  ok = ok && (hipStreamCreate((hipStream_t *)&f->st) == hipSuccess || hy4_fail(err, "the step's stream not made"));
  ok = ok && upload(f, err);
  for (size_t i = 0; ok && i < sizeof read_want / sizeof *read_want; i++)
    ok = hy4_alloc(read_want[i].p, read_want[i].bytes) || hy4_fail(err, "GPU memory refused (the read's buffers)");
  ok = ok && hipMemset(f->overflow, 0, 8) == hipSuccess && (f->fault = f->overflow + 1) &&
       hipHostMalloc((void **)&f->pin_flags, 8, 0) == hipSuccess && (f->h_rx = malloc(ecap * 4)) && (f->h_ry = malloc(ecap * 4)) &&
       (f->h_rxd = malloc(ecap * 4)) && (f->h_ryd = malloc(ecap * 4)) && (f->h_rcnt = malloc(ecap * 4)) &&
       (f->h_rw = malloc(3 * ecap * sizeof(void *))) && (f->rkey = malloc(ecap * 8)) &&
       (f->h_rgidx = malloc(scap * 4)) && (f->h_rhidx = malloc(scap * 4)) &&
       (f->h_stage = malloc(3 * ecap * sizeof(void *) + 5 * ecap * 4));
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = (as_f32(f, w->layer[l].hc_attn_fn, &f->fn_attn[l]) && as_f32(f, w->layer[l].hc_ffn_fn, &f->fn_ffn[l])) ||
         hy4_fail(err, "iHC's mixes not made F32");
  ok = ok && (as_f32(f, w->out_hc_fn, &f->fn_out) || hy4_fail(err, "the head's iHC not made F32")) &&
       ((freqs(g->rope_base_swa, (uint32_t)g->rope_dim, &f->freqs_swa) &&
         freqs(g->rope_base, (uint32_t)g->rope_dim, &f->freqs_sparse)) ||
        hy4_fail(err, "the ropes' tables not made")) &&
       engram_reader(f, fds, err);
  for (size_t i = 0; ok && i < sizeof want / sizeof *want; i++)
    ok = hy4_alloc(want[i].p, want[i].bytes) || hy4_fail(err, "GPU memory refused (a step's buffers)");
  ok = ok && hipHostMalloc((void **)&f->pin_tok, n * 4, 0) == hipSuccess &&
       hipHostMalloc((void **)&f->pin_pos, n * 4, 0) == hipSuccess &&
       hipHostMalloc((void **)&f->pin_logits, lr * g->n_vocab * F, 0) == hipSuccess && (f->slot = malloc(n * 4)) &&
       (f->h_idx = malloc(n * K * 4)) && (f->h_gidx = malloc(n * K * 4)) && (f->h_sidx = malloc(n * K * 4)) &&
       (f->h_w = malloc(n * K * F)) &&
       (f->h_wsel = malloc(n * K * F)) && hipDeviceSynchronize() == hipSuccess;
  if (!ok)
    return hy4_forward_destroy(f), (err && !*err ? hy4_fail(err, "the forward not made") : false), NULL;
  return f;
}

uint64_t hy4_identity(const Hy4Forward *f)
{
  if (!f)
    return 0;
  uint64_t h = fnv(&f->w->geo, sizeof f->w->geo, FNV_SEED);
  h = fnv_u64(f->ring, h); /* a sliding layer's ring: its rows' places another forward's */
  for (int i = 0; i < f->w->model->n_files; i++)
    h = fnv_file(f->w->model->paths[i], h);
  return h;
}
