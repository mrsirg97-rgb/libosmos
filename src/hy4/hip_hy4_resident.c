/* Hy4's forward, its weights made resident (hy4_impl.h): the dense tensors on the device, the routed experts the
   membrane's pool, the joined matrices, the rope tables, Engram's reader */
#define _GNU_SOURCE
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime_api.h>
#include "osmos/model/tensor.h"
#include "osmos/gpu/hy4_ops.h"
#include "osmos/gpu/route.h"
#include "osmos/gpu/mm.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/quant.h"
#include "osmos/gpu/bufs.h"
#include "osmos/gpu/upload.h"
#include "osmos/mem/arena.h"
#include "osmos/mem/host_slots.h"
#include "osmos/model/expert_units.h"
#include "osmos/model/fnv.h"
#include "osmos/model/qtype.h"
#include "hy4_impl.h"
#include "osmos/err.h"
#include "osmos/err_hip.h"


#define READ_PIECE (64ull << 20)

bool hy4_alloc(GpuBufs *b, void *p, uint64_t bytes) { return gpu_bufs_one(b, b->ledger, p, bytes); }
bool hy4_pinned(GpuBufs *b, void *p, uint64_t bytes) { return gpu_bufs_host(b, b->ledger, p, bytes); }

void hy4_forward_tap(Hy4Forward *f, hy4_tap_fn fn, void *arg) { f->tap = fn, f->tap_arg = arg; }
static bool resident(const Hy4Weights *w, const GgufTensor *t)
{
  if (t == w->engram_table)
    return false;
  return !(strncmp(t->name, "blk.", 4) == 0 && atoi(t->name + 4) >= w->geo.n_layers); /* the drafter's: not ours */
}

/* a MoE layer's routed experts: the membrane's (hy4_experts()), not the upload's */
static bool routed_expert(const Hy4Weights *w, const GgufTensor *t)
{
  for (int l = w->geo.n_dense; l < w->geo.n_layers; l++)
    if (t == w->layer[l].exp_gate || t == w->layer[l].exp_up || t == w->layer[l].exp_down)
      return true;
  return false;
}

static bool not_resident(const GgufTensor *t, void *arg)
{
  const Hy4Forward *f = (const Hy4Forward *)arg;
  const Hy4Weights *w = f->w;
  return !resident(w, t) || routed_expert(w, t) || (f->skip_output && t == w->output);
}

/* a promotion's copy, the stream's slot to the pool's (both pinned host memory the GPU reads in place) */
static bool copy_host(void *dst, const void *src, uint64_t bytes, void *ctx)
{
  (void)ctx;
  memcpy(dst, src, bytes);
  return true;
}

/* the routed experts behind the membrane: the MoE layers' units (expert_units), the warm set an expert of every layer
   at a time, sized to the budget (experts_plan: 0 every one), read in (experts_create); every one resident: their
   addresses' device table for the decode's GPU tables (route_tok) */
bool hy4_experts(Hy4Forward *f, uint64_t budget, char *err)
{
  const Hy4Weights *w = f->w;
  const Gguf *m = w->model;
  const uint32_t E = (uint32_t)w->geo.n_experts, D0 = (uint32_t)w->geo.n_dense, ML = (uint32_t)w->geo.n_layers - D0;
  if (!ML)
    return true;
  ExpertTensors t[HY4_MAX_LAYERS];
  for (uint32_t l = 0; l < ML; l++)
  {
    const Hy4Layer *L = &w->layer[D0 + l];
    t[l] = (ExpertTensors){L->exp_gate, L->exp_up, L->exp_down, false};
  }
  const size_t n = (size_t)ML * E;
  PoolUnit *units = calloc(n, sizeof *units);
  uint32_t *order = malloc(n * sizeof *order);
  bool ok = units && order && (f->xfds = malloc((size_t)m->n_files * sizeof *f->xfds));
  for (int i = 0; ok && i < m->n_files; i++)
    f->xfds[i] = -1;
  for (int i = 0; ok && i < m->n_files; i++)
    ok = (f->xfds[i] = open(m->paths[i], O_RDONLY | O_DIRECT)) >= 0;
  ok = (ok && expert_units(t, ML, E, f->xfds, units, n) == n) ||
       osmos_fail(err, "out of memory, a model file not opened O_DIRECT, or the experts not n_experts equal slices");
  for (size_t u = 0; ok && u < n; u++) /* expert 0 of every layer, then 1, ... */
    order[u] = (uint32_t)((u % ML) * E + u / ML);
  ExpertsPlan plan;
  char merr[OSMOS_ERR] = "";
  ok = ok && (experts_plan(units, ML, E, order, 0, budget, f->fixed_bytes, ARENA_GRAIN, &plan) ||
              osmos_fail(err, "the experts' plan refused"));
  ok = ok && (experts_create(f->ledger, units, ML, E, order, &plan, ARENA_GRAIN, HY4_EXPERT_READERS, copy_host, NULL, 0, true,
                             &f->ex, merr) ||
              osmos_fail(err, "%s", merr[0] ? merr : "the experts' membrane refused"));
  if (ok)
    f->pinned_bytes = host_slots_bytes(f->ex.slots) + host_slots_bytes(f->ex.stream);
  if (ok && (f->whole = membrane_whole(f->ex.mem)))
  {
    const void **tab = malloc(n * 3 * sizeof *tab);
    ok = tab != NULL;
    for (size_t u = 0; ok && u < n; u++)
    {
      const void *dev[POOL_PARTS];
      ok = membrane_ready(f->ex.mem, (uint32_t)u, dev);
      for (int k = 0; ok && k < 3; k++)
        tab[((u / E) * 3 + (size_t)k) * E + u % E] = dev[k];
    }
    ok = (ok && hy4_alloc(&f->bufs, &f->exp_tab, n * 3 * sizeof *tab) &&
          hipMemcpy(f->exp_tab, tab, n * 3 * sizeof *tab, hipMemcpyHostToDevice) == hipSuccess) ||
         osmos_fail(err, "the experts' address table not made");
    f->h_tab = tab;
  }
  ok = ok && ((f->h_ex = malloc(E * sizeof *f->h_ex)) || osmos_fail(err, "out of memory"));
  free(units), free(order);
  return ok;
}

bool hy4_upload(Hy4Forward *f, char *err)
{
  return gpu_upload_table(f->ledger, f->w->model, READ_PIECE, not_resident, f, &f->dev, err);
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
      if (!hy4_alloc(&f->bufs, &f->joints[f->n_joints], bytes)) /* (its table's slot recorded: freed from it) */
        return false;
      uint8_t *buf = f->joints[f->n_joints++];
      w = buf;
      uint64_t at = 0;
      for (int b = a; b < k; b++)
        if (m[b] && !done[b] && m[b]->type == type)
        {
          const size_t i = (size_t)(m[b] - g->tensors);
          if (hipMemcpy(buf + at, f->dev[i], m[b]->bytes, hipMemcpyDeviceToDevice) != hipSuccess)
            return false;
          ledger_free(f->ledger, f->dev[i]);
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
bool hy4_joints_make(Hy4Forward *f, uint32_t *most)
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

/* a matrix the kernels read as F32 (iHC's mixes: Q4_K_M's F32, copied; SPARK-4Bit's Q8_0, decoded); the source tensor
   slot is freed once converted so it does not stay resident alongside the F32 copy */
bool hy4_as_f32(Hy4Forward *f, const GgufTensor *t, float **out)
{
  const uint64_t n = t->dims[0] * t->dims[1];
  if (!hy4_alloc(&f->bufs, out, n * F))
    return false;
  const bool ok = t->type == QTYPE_F32
                      ? hipMemcpy(*out, hy4_dev(f, t), n * F, hipMemcpyDeviceToDevice) == hipSuccess
                      : quant_dequant(t->type, hy4_dev(f, t), n, *out, NULL) && hipDeviceSynchronize() == hipSuccess;
  if (ok)
  {
    const size_t idx = (size_t)(t - f->w->model->tensors);
    if (f->dev[idx] && (!f->in_joint || !f->in_joint[idx]))
      ledger_free(f->ledger, f->dev[idx]), f->dev[idx] = NULL;
  }
  return ok;
}

/* a rope's frequencies: 1 / base^(2i / dim), as the reference computes them (F32) */
bool hy4_freqs(Hy4Forward *f, float base, uint32_t dim, float **out)
{
  float h[256];
  if (dim / 2 > 256) /* (before the writes: a file's rope_dim past 512 writes past h) */
    return false;
  for (uint32_t i = 0; i < dim / 2; i++)
    h[i] = 1.0f / powf(base, (float)(2 * i) / (float)dim);
  return hy4_alloc(&f->bufs, out, dim / 2 * F) && hipMemcpy(*out, h, dim / 2 * F, hipMemcpyHostToDevice) == hipSuccess;
}

bool hy4_engram_reader(Hy4Forward *f, const int *fds, char *err)
{
  const Hy4Weights *w = f->w;
  const Hy4Geometry *g = &w->geo;
  EngramShape shape;
  char eerr[OSMOS_ERR] = "";
  if (!hy4_engram_shape(w, &shape, err))
    return false;
  if (!(f->eg = engram_create_shape(&shape, eerr)))
    return osmos_fail(err, "%s", eerr);
  const GgufKv *offs = gguf_kv(w->model, "hy_v4_flash.ple.head_offsets");
  const int cols = (g->engram_ngram - 1) * g->engram_heads;
  for (int e = 0; e < g->n_engram; e++)
  {
    int64_t v = 0;
    if (!offs || !gguf_int(offs, (size_t)e * cols, &v) || v < 0)
      return osmos_fail(err, "hy_v4_flash.ple.head_offsets: an Engram layer's first row");
    f->eg_first[e] = (uint64_t)v;
  }
  const GgufTensor *t = w->engram_table;
  f->eg_fd = fds[t->file];
  f->eg_row_bytes = t->bytes / t->dims[1];
  const uint64_t one = (uint64_t)f->max_tokens * cols;
  const uint64_t rows = one * g->n_engram;
  const bool ok = (f->eg_read = rowread_create(HY4_ROW_THREADS)) && (f->eg_off = malloc(rows * 8)) &&
                  hy4_pinned(&f->bufs, &f->eg_stage, rows * f->eg_row_bytes) &&
                  hy4_alloc(&f->bufs, &f->eg_raw, one * f->eg_row_bytes) &&
                  hy4_alloc(&f->bufs, &f->eg_rows, one * g->engram_dim * F);
  f->eg_rs = (RowStream){f->eg_read, f->eg_fd, f->eg_off, f->eg_stage, f->eg_row_bytes, 0, 0, false};
  return ok || osmos_fail(err, "out of memory (the Engram rows' buffers)");
}
