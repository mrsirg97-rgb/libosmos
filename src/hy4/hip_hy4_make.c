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
#include "osmos/gpu/bufs.h"
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

static bool not_resident(const GgufTensor *t, void *arg) { return !resident(((const Hy4Forward *)arg)->w, t); }

static bool upload(Hy4Forward *f, char *err)
{
  return gpu_upload_table(f->w->model, READ_PIECE, not_resident, f, &f->dev, err, err ? HY4_ERR : 0);
}

/* a joint of members (NULL ones skipped; rows each, cols alike): grouped by type in the members' order, each type's
   two or more concatenated into one buffer (the tensors' device pointers then into it) */
static bool joint_make(Hy4Forward *f, const GgufTensor *const *m, const uint32_t *rows, int k, Hy4Joint *j)
{
  const Gguf *g = f->w->model;
  memset(j, 0, sizeof *j);
  bool done[HY4_JOINT_MAX] = {false};
  for (int a = 0; a < k; a++)
  {
    if (!m[a] || done[a])
      continue;
    uint32_t type = m[a]->type, sum = 0;
    uint64_t bytes = 0;
    int count = 0;
    for (int b = a; b < k; b++)
      if (m[b] && !done[b] && m[b]->type == type)
        j->off[b] = j->total + sum, sum += rows[b], bytes += m[b]->bytes, count++;
    const void *w = hy4_dev(f, m[a]);
    if (count > 1)
    {
      uint8_t *buf = NULL;
      if (hipMalloc((void **)&buf, bytes) != hipSuccess)
        return false;
      f->joints[f->n_joints++] = buf, w = buf;
      uint64_t at = 0;
      for (int b = a; b < k; b++)
        if (m[b] && !done[b] && m[b]->type == type)
        {
          const size_t i = (size_t)(m[b] - g->tensors);
          if (hipMemcpy(buf + at, f->dev[i], m[b]->bytes, hipMemcpyDeviceToDevice) != hipSuccess)
            return false;
          hipFree(f->dev[i]);
          f->dev[i] = buf + at, f->in_joint[i] = true, at += m[b]->bytes;
        }
    }
    for (int b = a; b < k; b++)
      if (m[b] && m[b]->type == type)
        done[b] = true;
    j->run[j->n_runs++] = (typeof(j->run[0])){type, sum, j->total, w};
    j->total += sum;
  }
  return true;
}

/* every layer's joints: the attention input's (q_a, kv_a, gate, the indexer's k and weights) and the FFN input's
   (the router, the routed latent's down and the shared expert's gate and up; the dense layer's gate and up) */
static bool joints_make(Hy4Forward *f, uint32_t *most)
{
  const Hy4Geometry *g = &f->w->geo;
  const uint32_t ql = (uint32_t)g->q_lora, row = (uint32_t)(g->kv_lora + g->rope_dim), vd = (uint32_t)g->v_dim;
  const uint32_t ID = (uint32_t)g->ix_dim, IH = (uint32_t)g->ix_heads, E = (uint32_t)g->n_experts;
  const uint32_t lat = (uint32_t)g->latent, sh = (uint32_t)(g->ff_exp * g->n_shexp), ff = (uint32_t)g->n_ff;
  if (!(f->joints = calloc(2 * (size_t)g->n_layers * HY4_JOINT_MAX, sizeof *f->joints)))
    return false;
  for (int l = 0; l < g->n_layers; l++)
  {
    const Hy4Layer *L = &f->w->layer[l];
    const bool sw = g->sliding[l];
    const GgufTensor *a[5] = {L->q_a, L->kv_a, L->gate, sw ? NULL : L->ix_k, sw ? NULL : L->ix_proj};
    const uint32_t ar[5] = {ql, row, (uint32_t)g->n_heads[l] * vd, ID, IH};
    const GgufTensor *d[4] = {L->router, L->routed_down, L->sh_gate, L->sh_up};
    const uint32_t dr[4] = {E, lat, sh, sh};
    const GgufTensor *dd[2] = {L->ffn_gate, L->ffn_up};
    const uint32_t ddr[2] = {ff, ff};
    if (!joint_make(f, a, ar, 5, &f->j_attn[l]) ||
        !(l < g->n_dense ? joint_make(f, dd, ddr, 2, &f->j_ffn[l]) : joint_make(f, d, dr, 4, &f->j_ffn[l])))
      return false;
    *most = f->j_attn[l].total > *most ? f->j_attn[l].total : *most;
    *most = f->j_ffn[l].total > *most ? f->j_ffn[l].total : *most;
  }
  return true;
}

/* a matrix the kernels read as F32 (iHC's mixes: Q4_K_M's F32, copied; SPARK-4Bit's were Q8_0, decoded) */
static bool as_f32(Hy4Forward *f, const GgufTensor *t, float **out)
{
  const uint64_t n = t->dims[0] * t->dims[1];
  if (!hy4_alloc(out, n * F))
    return false;
  if (t->type == QTYPE_F32)
    return hipMemcpy(*out, hy4_dev(f, t), n * F, hipMemcpyDeviceToDevice) == hipSuccess;
  return quant_dequant(t->type, hy4_dev(f, t), n, *out, NULL);
}

/* a rope's frequencies: 1 / base^(2i / dim), as the reference computes them (F32) */
static bool freqs(float base, uint32_t dim, float **out)
{
  float h[256];
  if (dim / 2 > 256) /* (before the writes: a file's rope_dim past 512 wrote past h - 0.11.2) */
    return false;
  for (uint32_t i = 0; i < dim / 2; i++)
    h[i] = 1.0f / powf(base, (float)(2 * i) / (float)dim);
  return hy4_alloc(out, dim / 2 * F) && hipMemcpy(*out, h, dim / 2 * F, hipMemcpyHostToDevice) == hipSuccess;
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
  hy4_dfly_free(f);
  hipFree(f->head8);
  hipFree(f->kb_tab), hipFree(f->vb_tab), hipFree(f->kb_xo), hipFree(f->kb_yo), hipFree(f->vb_xo), hipFree(f->vb_yo);
  hipFree(f->hcnt);
  gpu_free_table(f->w->model, f->dev, f->in_joint); /* (a joint's members its buffer's) */
  for (uint32_t i = 0; i < f->n_joints; i++)
    hipFree(f->joints[i]);
  free(f->joints), free(f->in_joint);
  for (int l = 0; l < HY4_MAX_LAYERS; l++)
    hipFree(f->fn_attn[l]), hipFree(f->fn_ffn[l]);
  gpu_bufs_free(&f->bufs); /* the step's buffers: the table's */
  void *b[] = {f->fn_out, f->freqs_swa, f->freqs_sparse, f->eg_raw, f->eg_rows, f->jo_attn, f->jo_ffn, /* (allocated  */
               f->x16,    f->xe16,      f->overflow,     f->rx,     f->ry,      f->rxd,     f->ryd,    /* outside the */
               f->rcnt,   f->rw,        f->rgidx,        f->rhidx};                                    /* table)      */
  for (size_t i = 0; i < sizeof b / sizeof *b; i++)
    hipFree(b[i]);
  hipHostFree(f->eg_stage), hipHostFree(f->pin_tok), hipHostFree(f->pin_pos), hipHostFree(f->pin_logits), hipHostFree(f->pin_flags);
  rowread_destroy(f->eg_read);
  engram_destroy(f->eg);
  free(f->eg_off), free(f->slot), free(f->h_idx), free(f->h_gidx), free(f->h_sidx);
  free(f->h_w), free(f->h_wsel);
  free(f->h_rx), free(f->h_ry), free(f->h_rxd), free(f->h_ryd), free(f->h_rcnt), free(f->h_rw), free(f->rkey);
  free(f->h_rgidx), free(f->h_rhidx), free(f->h_stage), free(f->h_estart), free(f->h_rexp);
  free(f);
}

/* a read's per-head absorption (k_b: q_nope to the latent; v_b: the latent to the head's values) as one mm_group a
   layer over its heads: each layer's heads' matrices, and where head h reads (its columns of the tiled x16: the q /
   latent rows converted once) and writes - fixed for the forward's life */
static bool head_tables(Hy4Forward *f)
{
  const Hy4Geometry *g = &f->w->geo;
  const uint32_t L = (uint32_t)g->n_layers, M = HY4_MAX_HEADS, nope = (uint32_t)g->nope_dim, rd = (uint32_t)g->rope_dim;
  const uint32_t qk = nope + rd, lat = (uint32_t)g->kv_lora, vd = (uint32_t)g->v_dim, row = lat + rd;
  const void **kb = calloc((size_t)L * M, sizeof *kb), **vb = calloc((size_t)L * M, sizeof *vb);
  uint32_t xo[2][HY4_MAX_HEADS], yo[2][HY4_MAX_HEADS];
  for (uint32_t h = 0; h < M; h++)
    xo[0][h] = h * qk * MM_TILE, yo[0][h] = h * row, xo[1][h] = h * lat * MM_TILE, yo[1][h] = h * vd;
  bool ok = kb && vb;
  for (uint32_t l = 0; ok && l < L; l++)
  {
    const Hy4Layer *Ly = &f->w->layer[l];
    for (uint32_t h = 0; h < (uint32_t)g->n_heads[l]; h++)
      kb[(size_t)l * M + h] = (const uint8_t *)hy4_dev(f, Ly->k_b) + (uint64_t)h * lat * nope / 32 * 34,
      vb[(size_t)l * M + h] = (const uint8_t *)hy4_dev(f, Ly->v_b) + (uint64_t)h * vd * lat / 32 * 34;
  }
  const uint64_t tb = (uint64_t)L * M * sizeof *kb;
  ok = ok && hipMalloc((void **)&f->kb_tab, tb) == hipSuccess && hipMalloc((void **)&f->vb_tab, tb) == hipSuccess &&
       hipMalloc((void **)&f->kb_xo, M * 4) == hipSuccess && hipMalloc((void **)&f->kb_yo, M * 4) == hipSuccess &&
       hipMalloc((void **)&f->vb_xo, M * 4) == hipSuccess && hipMalloc((void **)&f->vb_yo, M * 4) == hipSuccess &&
       hipMalloc((void **)&f->hcnt, M * 4) == hipSuccess &&
       hipMemcpy(f->kb_tab, kb, tb, hipMemcpyHostToDevice) == hipSuccess &&
       hipMemcpy(f->vb_tab, vb, tb, hipMemcpyHostToDevice) == hipSuccess &&
       hipMemcpy(f->kb_xo, xo[0], M * 4, hipMemcpyHostToDevice) == hipSuccess &&
       hipMemcpy(f->kb_yo, yo[0], M * 4, hipMemcpyHostToDevice) == hipSuccess &&
       hipMemcpy(f->vb_xo, xo[1], M * 4, hipMemcpyHostToDevice) == hipSuccess &&
       hipMemcpy(f->vb_yo, yo[1], M * 4, hipMemcpyHostToDevice) == hipSuccess;
  free(kb), free(vb);
  return ok;
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
  f->w = w, f->max_tokens = o->max_tokens, f->max_ctx = o->max_ctx, f->kv8 = !o->kv_f32;
  f->logit_rows = o->logit_rows && o->logit_rows < o->max_tokens ? o->logit_rows : o->max_tokens;
  f->ring = (uint32_t)g->window + o->max_tokens + HY4_MAX_BACK;
  f->kring = (o->max_tokens + (uint32_t)g->ix_pool + HY4_MAX_BACK + (uint32_t)g->ix_pool - 1) / (uint32_t)g->ix_pool *
             (uint32_t)g->ix_pool;
  f->eg_fd = -1;
  for (int l = 0; l < HY4_MAX_LAYERS; l++)
    f->dfly_slot[l] = -1;
  int hmax = 0;
  for (int l = 0; l < g->n_layers; l++)
    hmax = g->n_heads[l] > hmax ? g->n_heads[l] : hmax;
  const uint64_t n = o->max_tokens, lr = f->logit_rows, D = (uint64_t)g->n_embd, hc = (uint64_t)g->hc;
  const uint64_t H = (uint64_t)hmax, qk = (uint64_t)(g->nope_dim + g->rope_dim), row = (uint64_t)(g->kv_lora + g->rope_dim);
  const uint64_t K = (uint64_t)g->n_used, L = (uint64_t)g->latent, ffe = (uint64_t)g->ff_exp;
  const uint64_t eg_in = (uint64_t)(g->engram_ngram - 1) * g->engram_heads * g->engram_dim;
  const uint64_t entries = n * K; /* mv_group_tok's entries at most: a token's picks each its own at worst */
  /* (xg and eg_h a chunk's rows past the last: mv_group_tok reads an entry's MV_MAX_TOKENS rows, uses its count) */
  const GpuWant want[] = {
      {&f->h, n * hc * D * F}, {&f->xn, n * D * F}, {&f->y, n * D * F}, {&f->post, n * hc * F}, {&f->hcs, hy4_hc_scratch((uint32_t)n) * F},
      {&f->table, hy4_mla_table_bytes((uint32_t)n, (uint32_t)H)}, {&f->attn_part, hy4_mla_attn_scratch((uint32_t)H) * F},
      {&f->qr, n * g->q_lora * F}, {&f->qres, n * g->q_lora * F}, {&f->q, n * H * qk * F}, {&f->kva, n * row * F},
      {&f->gt, n * H * g->v_dim * F}, {&f->qa, n * H * row * F}, {&f->olat, n * H * g->kv_lora * F},
      {&f->ao, n * H * g->v_dim * F}, {&f->out, n * D * F}, {&f->ixk, n * g->ix_dim * F}, {&f->ixkn, n * g->ix_dim * F},
      {&f->ixq, n * g->ix_heads * g->ix_dim * F}, {&f->ixw, n * g->ix_heads * F},
      {&f->scores, (n < HY4_SCORE_ROWS ? n : HY4_SCORE_ROWS) * (o->max_ctx / g->ix_pool + 1) * F}, {&f->sel, n * (g->ix_topk / g->ix_pool) * 4},
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
  if (ok)
    ok = head_tables(f) || hy4_fail(err, "a GPU allocation refused (the heads' tables)");
  if (ok && o->head_q8 && f->w->output->type == QTYPE_BF16)
  { /* the head's Q8_0 copy (Hy4Options' head_q8): its BF16 freed */
    const GgufTensor *out = f->w->output;
    float *x = hy4_host_f32(f->w->model, out);
    ok = (x && hy4_q8_up(x, out->dims[1], (uint32_t)out->dims[0], &f->head8)) ||
         hy4_fail(err, "the head's Q8_0 copy not made");
    free(x);
    if (ok)
    {
      const size_t i = (size_t)(out - f->w->model->tensors);
      hipFree(f->dev[i]), f->dev[i] = NULL;
    }
  }
  uint32_t joint_most = 0;
  ok = ok && (((f->in_joint = calloc(w->model->n_tensors, sizeof *f->in_joint)) && joints_make(f, &joint_most) &&
               hy4_alloc(&f->jo_attn, (uint64_t)joint_most * F) && hy4_alloc(&f->jo_ffn, (uint64_t)joint_most * F)) ||
              hy4_fail(err, "the joint matrices not made (GPU memory)"));
  for (size_t i = 0; ok && i < sizeof read_want / sizeof *read_want; i++)
    ok = hy4_alloc(read_want[i].p, read_want[i].bytes) || hy4_fail(err, "GPU memory refused (the read's buffers)");
  ok = ok && hipMemset(f->overflow, 0, 8) == hipSuccess && (f->fault = f->overflow + 1) &&
       hipHostMalloc((void **)&f->pin_flags, 8, 0) == hipSuccess && (f->h_rx = malloc(ecap * 4)) && (f->h_ry = malloc(ecap * 4)) &&
       (f->h_rxd = malloc(ecap * 4)) && (f->h_ryd = malloc(ecap * 4)) && (f->h_rcnt = malloc(ecap * 4)) &&
       (f->h_rw = malloc(3 * ecap * sizeof(void *))) && (f->rkey = malloc(ecap * 8)) &&
       (f->h_rgidx = malloc(scap * 4)) && (f->h_rhidx = malloc(scap * 4)) &&
       (f->h_stage = malloc(3 * ecap * sizeof(void *) + 5 * ecap * 4)) &&
       (f->h_estart = malloc(((size_t)g->n_experts + 1) * 4)) && (f->h_rexp = malloc(ecap * 4));
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = (as_f32(f, w->layer[l].hc_attn_fn, &f->fn_attn[l]) && as_f32(f, w->layer[l].hc_ffn_fn, &f->fn_ffn[l])) ||
         hy4_fail(err, "iHC's mixes not made F32");
  ok = ok && (as_f32(f, w->out_hc_fn, &f->fn_out) || hy4_fail(err, "the head's iHC not made F32")) &&
       ((freqs(g->rope_base_swa, (uint32_t)g->rope_dim, &f->freqs_swa) &&
         freqs(g->rope_base, (uint32_t)g->rope_dim, &f->freqs_sparse)) ||
        hy4_fail(err, "the ropes' tables not made")) &&
       engram_reader(f, fds, err);
  ok = ok && (gpu_bufs_alloc(&f->bufs, want, sizeof want / sizeof *want) ||
              hy4_fail(err, "GPU memory refused (a step's buffers)"));
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
  h = fnv_u64(f->kv8, h);  /* the cache rows' format */
  for (int i = 0; i < f->w->model->n_files; i++)
    h = fnv_file(f->w->model->paths[i], h);
  return h;
}
