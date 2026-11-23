/* Inkling-2-Small's forward, its weights made resident (ink2_impl.h): the dense weights uploaded (F32 / F16 / BF16
   on the host, raw bytes as stored), the convs' and loops' tables, the routed experts the membrane's pool in their
   order, the whole build's tables */
#define _GNU_SOURCE
#include <fcntl.h>
#include <hip/hip_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "ink2_impl.h"
#include "osmos/model/tensor.h"
#include "osmos/gpu/ink2_ops.h"
#include "osmos/gpu/mm.h"
#include "osmos/mem/experts.h"
#include "osmos/model/moe_route.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/kept.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/gpu/quant.h"
#include "osmos/mem/arena.h"
#include "osmos/err.h"

/* Inkling-2-Small's forward, its making (osmos/ink2/forward.h): the dense weights resident, the routed experts
   the membrane's (their pool and order), the tables and buffers a step reads, freed in reverse; the profile's timer */


#define T_F32 0u
#define T_F16 1u
#define T_BF16 30u
bool ink2_dalloc(Ink2Forward *f, void **p, uint64_t bytes)
{
  if (!gpu_bufs_one(&f->bufs, f->ledger, p, bytes)) /* (its pad osmos/gpu/bufs.h's, every family's) */
    return false;
  f->dense_bytes += bytes;
  return true;
}

bool ink2_up_host(Ink2Forward *f, const float *v, uint64_t n, float **out)
{
  return ink2_dalloc(f, (void **)out, n * 4) && hipMemcpy(*out, v, n * 4, hipMemcpyHostToDevice) == hipSuccess;
}

/* a vector's F32 on the device, from g's file; NULL tensor: nothing (true) */
bool ink2_up_f32_of(Ink2Forward *f, const Gguf *g, const GgufTensor *t, float **out)
{
  *out = NULL;
  if (!t)
    return true;
  uint64_t n = 0;
  float *v = tensor_f32(g, t, &n);
  const bool ok = v && ink2_up_host(f, v, n, out);
  free(v);
  return ok;
}

/* a matrix's bytes as stored, from g's file - a Q5_K one repacked (quant_repack_q5k: 16-row tiles, the matvec's
   loads one run a wave; the same values, so the same bits): every resident Q5_K the forward's Q5_KR (mat's
   type), one whose rows are not whole tiles refused */
bool ink2_up_raw_of(Ink2Forward *f, const Gguf *g, const GgufTensor *t, void **out)
{
  *out = NULL;
  if (!t)
    return true;
  const bool q5 = t->type == QUANT_Q5_K;
  if (q5 && (t->n_dims != 2 || t->dims[1] % 16))
    return false;
  void *b = malloc(t->bytes), *stage = NULL;
  const uint64_t rows = t->n_dims == 2 ? t->dims[1] : 1, packed = q5 ? t->bytes / 176 * 180 : t->bytes;
  bool ok = b && tensor_read(g, t, b) && ink2_dalloc(f, out, packed);
  if (ok && q5)
    ok = ledger_malloc(f->ledger, &stage, t->bytes) == hipSuccess && hipMemcpy(stage, b, t->bytes, hipMemcpyHostToDevice) == hipSuccess &&
         quant_repack_q5k(stage, (uint32_t)rows, (uint32_t)t->dims[0], *out, NULL) && hipDeviceSynchronize() == hipSuccess;
  else if (ok)
    ok = hipMemcpy(*out, b, t->bytes, hipMemcpyHostToDevice) == hipSuccess;
  if (stage)
    ledger_free(f->ledger, stage);
  free(b);
  return ok;
}

static bool up_f32(Ink2Forward *f, const GgufTensor *t, float **out) { return ink2_up_f32_of(f, f->w->model, t, out); }
static bool up_raw(Ink2Forward *f, const GgufTensor *t, void **out) { return ink2_up_raw_of(f, f->w->model, t, out); }

/* a full layer's static conv as the kernel's [taps][ch], tap 0 first (llama.cpp's [ch][taps], tap 0 last) */
static bool up_conv_static(Ink2Forward *f, const GgufTensor *t, float **out)
{
  uint64_t n = 0;
  float *v = tensor_f32(f->w->model, t, &n);
  if (!v)
    return false;
  const uint32_t taps = (uint32_t)f->w->geo.conv;
  if (f->w->geo.keys != INK2_KEYS_OURS)
  {
    float *tw = malloc(n * 4);
    if (!tw)
      return free(v), false;
    const uint64_t ch = n / taps;
    for (uint64_t c = 0; c < ch; c++)
      for (uint32_t j = 0; j < taps; j++)
        tw[(uint64_t)j * ch + c] = v[c * taps + (taps - 1 - j)];
    free(v), v = tw;
  }
  const bool ok = ink2_up_host(f, v, n, out);
  free(v);
  return ok;
}

/* the loop's: each core layer's controls, and its four message queries from the two tables (core layer i's q the
   first table's column 2i, ffn its 2i + 1; k the second's 2i, v its 2i + 1) as [q, k, v, ffn][C] */
static bool up_loop(Ink2Forward *f)
{
  const Ink2Weights *w = f->w;
  const Ink2Geometry *g = &w->geo;
  if (!g->loop.passes)
    return true;
  const uint64_t C = (uint64_t)g->n_embd, core = (uint64_t)(g->loop.end - g->loop.start);
  uint64_t nq = 0, nk = 0;
  float *q = tensor_f32(w->model, w->loop_query, &nq), *kv = tensor_f32(w->model, w->loop_kv_query, &nk),
        *four = malloc(INK2_CTRLS * C * 4);
  bool ok = q && kv && four && nq == 2 * core * C && nk == 2 * core * C;
  for (int l = g->loop.start; ok && l < g->loop.end; l++)
  {
    const uint64_t i = (uint64_t)(l - g->loop.start);
    const float *col[INK2_CTRLS] = {q + 2 * i * C, kv + 2 * i * C, kv + (2 * i + 1) * C, q + (2 * i + 1) * C};
    for (int c = 0; c < INK2_CTRLS; c++)
      memcpy(four + c * C, col[c], C * 4);
    Ink2Dev *d = &f->L[l];
    const Ink2Layer *L = &w->layer[l];
    ok = ink2_up_host(f, four, INK2_CTRLS * C, &d->msg_q);
    for (int c = 0; ok && c < INK2_CTRLS; c++)
      ok = up_raw(f, L->ctrl_down[c], &d->ctrl_down[c]) && up_raw(f, L->ctrl_gain[c], &d->ctrl_gain[c]) &&
           up_raw(f, L->ctrl_forget[c], &d->ctrl_forget[c]);
  }
  free(q), free(kv), free(four);
  return ok;
}

bool ink2_upload(Ink2Forward *f)
{
  const Ink2Weights *w = f->w;
  const uint32_t E = (uint32_t)w->geo.n_experts;
  bool ok = up_raw(f, w->embed, &f->embed) && up_raw(f, w->output, &f->output) &&
            up_f32(f, w->embed_norm, &f->embed_norm) && up_f32(f, w->output_res_query, &f->out_res_q) &&
            up_f32(f, w->output_norm, &f->out_norm);
  for (int l = 0; ok && l < w->geo.n_layers; l++)
  {
    const Ink2Layer *L = &w->layer[l];
    Ink2Dev *d = &f->L[l];
    ok = up_f32(f, L->attn_res_query, &d->attn_res_q) && up_f32(f, L->ffn_res_query, &d->ffn_res_q) &&
         up_f32(f, L->attn_norm, &d->attn_norm) && up_f32(f, L->q_norm, &d->q_norm) &&
         up_f32(f, L->k_norm, &d->k_norm) && up_f32(f, L->v_norm, &d->v_norm) &&
         up_f32(f, L->ffn_norm, &d->ffn_norm) && up_raw(f, L->q, &d->q) && up_raw(f, L->k, &d->k) &&
         up_raw(f, L->v, &d->v) && up_raw(f, L->rel, &d->rel) && up_raw(f, L->out, &d->out) &&
         up_raw(f, L->gate, &d->gate) && up_raw(f, L->sink, &d->sink) && up_raw(f, L->router, &d->router) &&
         up_raw(f, L->k_conv_proj, &d->k_conv_proj) && up_raw(f, L->v_conv_proj, &d->v_conv_proj) &&
         up_f32(f, L->q_clip, &d->q_clip) && up_f32(f, L->rel_clip, &d->rel_clip) &&
         up_f32(f, L->rel_proj, &d->rel_proj) && up_f32(f, L->router_bias, &d->router_bias) &&
         up_f32(f, L->out_conv, &d->out_conv) && up_f32(f, L->ffn_out_conv, &d->ffn_out_conv);
    if (ok && w->geo.sliding[l])
      ok = up_f32(f, L->k_conv, &d->k_conv) && up_f32(f, L->v_conv, &d->v_conv);
    else if (ok)
      ok = up_conv_static(f, L->k_conv_static, &d->k_conv) && up_conv_static(f, L->v_conv_static, &d->v_conv);
    const GgufTensor *sc[3] = {L->exp_gate_scale, L->exp_up_scale, L->exp_down_scale};
    for (int k = 0; ok && k < 3; k++)
    {
      uint64_t n = E;
      d->scale[k] = sc[k] ? tensor_f32(w->model, sc[k], &n) : malloc(E * sizeof(float));
      ok = d->scale[k] && n == E;
      for (uint32_t e = 0; ok && !sc[k] && e < E; e++)
        d->scale[k][e] = 1.0f;
    }
  }
  return ok && up_loop(f);
}

static bool copy_host(void *dst, const void *src, uint64_t bytes, void *ctx)
{
  (void)ctx;
  memcpy(dst, src, bytes); /* both pools pinned host memory: a promotion is the CPU's copy */
  return true;
}

/* the experts as the membrane's units: room's worth warmed, an expert of every layer at a time (expert 0 of each
   layer, then 1, ...); room 0 (or a budget short of the rest) the least: a layer's */
static const uint64_t *g_rank; /* by_picks's counts (qsort takes no context) */
static int by_picks(const void *a, const void *b)
{
  const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  const uint64_t px = g_rank[x], py = g_rank[y];
  return px != py ? (px < py) - (px > py) : (x > y) - (x < y);
}

bool ink2_experts(Ink2Forward *f, uint64_t room, uint64_t budget, const uint64_t *picks, char *err)
{
  const Ink2Weights *w = f->w;
  const Gguf *m = w->model;
  const uint32_t E = (uint32_t)w->geo.n_experts, L = (uint32_t)w->geo.n_layers;
  const size_t n = ink2_weights_experts(w, NULL, NULL, 0);
  if (!n)
    return osmos_fail(err, "the experts' tensors not n_experts equal slices");
  PoolUnit *units = calloc(n, sizeof *units);
  uint32_t *warm = malloc(n * sizeof *warm);
  bool ok = units && warm && (f->fds = malloc((size_t)m->n_files * sizeof *f->fds));
  for (int i = 0; ok && i < m->n_files; i++)
    f->fds[i] = -1;
  for (int i = 0; ok && i < m->n_files; i++)
    ok = (f->fds[i] = open(m->paths[i], O_RDONLY | O_DIRECT)) >= 0;
  if (!ok)
    return free(units), free(warm), osmos_fail(err, "out of memory, or a model file not opened O_DIRECT");
  ok = ink2_weights_experts(w, f->fds, units, n) == n;
  /* the warm set, the most wanted first: by picks counted on calibration text (most first, then the lower unit), else
     an expert of every layer at a time (expert 0 of each layer, then 1, ...) */
  for (size_t u = 0; u < n; u++)
    warm[u] = picks ? (uint32_t)u : (uint32_t)((u % L) * E + u / L);
  if (picks)
  {
    g_rank = picks;
    qsort(warm, n, sizeof *warm, by_picks);
  }
  /* the room, stream and warm set for the budget (experts_plan: the build's whole budget, the rest taken first - its
     fixed bytes, the stream, the slack), then the slots and membrane over them (experts_create) */
  ExpertsPlan plan;
  Experts ex;
  char merr[OSMOS_ERR] = "";
  if (ok && !experts_plan(units, L, E, warm, room, budget, f->fixed_bytes, ARENA_GRAIN, &plan))
    ok = osmos_fail(err, "the experts' plan refused");
  if (ok && !experts_create(f->ledger, units, L, E, warm, &plan, ARENA_GRAIN, INK2_EXPERT_READERS, copy_host, NULL, 0, true, &ex,
                            merr))
    ok = osmos_fail(err, "%s", merr[0] ? merr : "the experts' membrane refused");
  if (ok)
  {
    f->mem = ex.mem, f->slots = ex.slots, f->stream = ex.stream;
    f->pinned_bytes = host_slots_bytes(f->slots) + host_slots_bytes(f->stream);
  }
  free(units), free(warm);
  return ok;
}

/* moe's tables for P pairs at most: an expert's parts of at most MM_MAX_TOKENS rows - P + E entries at most, each 3
   pointers, a count and 4 offsets; its slots its rows padded to whole tiles - P + E (MM_TILE - 1) at most, each 2
   int32; each pair's row and 3 floats */

/* every expert in memory (membrane_whole): their addresses held for the membrane's life - a table of them written
   once, the layers' scales beside it, and the GPU's tables for the step's most picks (moe_whole) */
bool ink2_whole_tables(Ink2Forward *f, char *err)
{
  const Ink2Geometry *g = &f->w->geo;
  if (!(f->whole = membrane_whole(f->mem)))
    return true;
  const uint64_t L = (uint64_t)g->n_layers, E = (uint64_t)g->n_experts, P = (uint64_t)f->max_tokens * g->n_used;
  Q4xExpertBounds bd;
  q4x_expert_bounds((uint32_t)P, (uint32_t)E, &bd);
  const void **tab = malloc(L * 3 * E * sizeof *tab);
  float *sc = malloc(L * 3 * E * sizeof *sc);
  bool ok = tab && sc;
  for (uint64_t u = 0; ok && u < L * E; u++)
  {
    const void *dev[POOL_PARTS];
    ok = membrane_ready(f->mem, (uint32_t)u, dev);
    for (int k = 0; ok && k < 3; k++)
      tab[((u / E) * 3 + (uint64_t)k) * E + u % E] = dev[k], sc[((u / E) * 3 + (uint64_t)k) * E + u % E] =
                                                                 f->L[u / E].scale[k][u % E];
  }
  f->t_entries = bd.entries;
  struct
  {
    void **p;
    uint64_t bytes;
  } b[] = {
      {(void **)&f->exp_tab, L * 3 * E * sizeof *tab}, {(void **)&f->scale_tab, L * 3 * E * 4},
      {(void **)&f->picks_d, L * E * 4}, {(void **)&f->t_ptr, 3ull * bd.entries * sizeof(void *)},
      {(void **)&f->t_cnt, bd.entries * 4}, {(void **)&f->t_xg, bd.entries * 4}, {(void **)&f->t_yg, bd.entries * 4},
      {(void **)&f->t_xd, bd.entries * 4}, {(void **)&f->t_yd, bd.entries * 4}, {(void **)&f->t_gidx, bd.slots * 4},
      {(void **)&f->t_hidx, bd.slots * 4}, {(void **)&f->t_where, P * 4}, {(void **)&f->t_off, 3 * E * 4},
      {(void **)&f->t_sg, P * 4}, {(void **)&f->t_su, P * 4}, {(void **)&f->t_a, P * 4},
  };
  for (size_t i = 0; ok && i < sizeof b / sizeof *b; i++)
    ok = ink2_dalloc(f, b[i].p, b[i].bytes);
  ok = ok && hipMemcpy(f->exp_tab, tab, L * 3 * E * sizeof *tab, hipMemcpyHostToDevice) == hipSuccess &&
       hipMemcpy(f->scale_tab, sc, L * 3 * E * 4, hipMemcpyHostToDevice) == hipSuccess &&
       hipMemset(f->picks_d, 0, L * E * 4) == hipSuccess && (f->picks_all = calloc(L * E, sizeof *f->picks_all));
  free(tab), free(sc);
  return ok || osmos_fail(err, "GPU memory refused (the experts' tables on the GPU)");
}
