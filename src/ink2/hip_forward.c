/* Inkling-2-Small's forward with a cache (osmos/ink2/forward.h): tools/bench/ink2_run's math a step at a time - the
   dense weights resident, the routed experts the membrane's, a sequence's rows its tails (hip_seq.c) */
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

/* Ink2Options' profile: a step's sections timed, each after a synchronize (the bits the same), totals at destroy */
enum
{
  PF_IN,     /* the layer's inputs: attention residuals, norms, q / k / v / rel / gate / sink, convs, clips */
  PF_ATTN,   /* a sliding layer's attention (its window's keys) */
  PF_FULL,   /* a full layer's attention (every key so far) */
  PF_MID,    /* the output's matrix and conv, the residual, the MoE's input; the MoE's conv and residual */
  PF_MOE,    /* the router, the experts (their waits apart: PF_WAIT) */
  PF_WAIT,   /* the experts' holds and waits (the membrane's reads), inside PF_MOE */
  PF_ROUTE,  /* the router's matrix and picks, their copy to the host, inside PF_MOE */
  PF_XK,     /* the experts' tables sent and their kernels, inside PF_MOE */
  PF_HEAD,   /* the embedding, the head */
  PF_N
};
static const char *const kPfName[PF_N] = {"inputs",         "attn sliding",  "attn full",     "mid",           "moe",
                                          "  of it waits", "  of it router", "  of it kernels", "head"};
_Static_assert(PF_N == INK2_PF_N, "the profile's sections: Ink2Forward's pf");
static double pf_now(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static void pf(Ink2Forward *f, int b)
{
  if (!f->pf.on)
    return;
  hipDeviceSynchronize();
  const double t = pf_now();
  if (b >= 0)
    f->pf.step[b] += t - f->pf.at;
  f->pf.at = t;
}

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

static bool upload(Ink2Forward *f)
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

static bool experts(Ink2Forward *f, uint64_t room, uint64_t budget, const uint64_t *picks, char *err)
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
static uint64_t table_entries(uint64_t P, uint64_t E) { return P + E; }

/* every expert in memory (membrane_whole): their addresses held for the membrane's life - a table of them written
   once, the layers' scales beside it, and the GPU's tables for the step's most picks (moe_whole) */
static bool whole_tables(Ink2Forward *f, char *err)
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
static uint64_t table_slots(uint64_t P, uint64_t E) { return P + E * (MM_TILE - 1); }
static uint64_t table_bytes(uint64_t P, uint64_t E)
{
  return table_entries(P, E) * (3 * sizeof(void *) + 5 * 4) + table_slots(P, E) * 2 * 4 + P * 4 * 4;
}

/* mm's: its table's types, and BF16 where the model's dense weights are BF16 (f->bf16_mm: Mini's, as released) - mm's
   by name, not its table's. a model whose BF16 is a side tensor (Inkling-2-Small's sliding layers' sink projections)
   keeps its matvec, its bits 0.11's; the other families' BF16 matrices their matvec's as before. by the model, the
   type and the shape alone: a decode's and a prompt's the same */
static bool mm_takes(const Ink2Forward *f, uint32_t type, uint32_t rows, uint32_t cols)
{
  return (mm_known(type) || (type == QTYPE_BF16 && f->bf16_mm)) && mm_fits(type, rows, cols);
}

static bool buffers(Ink2Forward *f)
{
  const Ink2Geometry *g = &f->w->geo;
  const uint64_t T = f->max_tokens, C = (uint64_t)g->n_embd, H = (uint64_t)g->n_heads,
                 KV = (uint64_t)g->n_kv_heads, D = (uint64_t)g->head_dim, R = (uint64_t)g->d_rel,
                 taps = (uint64_t)g->conv, E = (uint64_t)g->n_experts, K = (uint64_t)g->n_used,
                 F = (uint64_t)g->ff_exp, VR = f->w->output->dims[1],
                 CH0 = H * D > C ? H * D : C, CH = g->loop.passes && 2 * C > CH0 ? 2 * C : CH0,
                 X16 = ((T < MM_MAX_TOKENS ? T : MM_MAX_TOKENS) + MM_TILE - 1) / MM_TILE * MM_TILE;
  const uint32_t head = ink2_type(f->w->output);
  f->head_rows = mm_takes(f, head, (uint32_t)VR, (uint32_t)C) ? MM_TILE * mm_col_tiles(head, MM_MAX_TOKENS) : MV_MAX_TOKENS;
  /* the experts' down rows and a decode's split attention share one buffer: the attention's partials are dead before
     the layer's MoE writes the down rows, and those are summed before the next layer's attention */
  const uint64_t fa_bytes = ink2_fa_scratch((uint32_t)H, f->max_ctx), down_bytes = T * K * C * 4,
                 ed_bytes = fa_bytes > down_bytes ? fa_bytes : down_bytes;
  struct
  {
    void **p;
    uint64_t bytes;
  } b[] = {
      {(void **)&f->ids_d, T * 4}, {(void **)&f->h, T * C * 4}, {(void **)&f->blocks, T * f->maxb * C * 4},
      {(void **)&f->a3, 3 * T * C * 4}, {(void **)&f->xn, T * C * 4}, {(void **)&f->kn, T * C * 4},
      {(void **)&f->vn, T * C * 4}, {(void **)&f->q, T * H * D * 4}, {(void **)&f->k, T * KV * D * 4},
      {(void **)&f->v, T * KV * D * 4}, {(void **)&f->kc, T * KV * D * 4}, {(void **)&f->vc, T * KV * D * 4},
      {(void **)&f->rel, T * H * R * 4}, {(void **)&f->gate, T * H * D * 4}, {(void **)&f->sink, T * H * 4},
      {(void **)&f->att, T * H * D * 4}, {(void **)&f->dyn, 2 * T * KV * taps * 4}, {(void **)&f->o, T * C * 4},
      {(void **)&f->r, T * C * 4}, {(void **)&f->moe, T * C * 4}, {(void **)&f->logit, T * E * 4},
      {(void **)&f->tmp, T * C * 4}, {(void **)&f->pick, 2 * T * K * 4},
      {(void **)&f->gpick, T * K * 4}, {(void **)&f->gw, T * K * 4},
      {(void **)&f->eg, T * K * F * 4}, {(void **)&f->eu, T * K * F * 4}, {(void **)&f->ed, ed_bytes}, {&f->m_dev, table_bytes(T * K, E)}, {&f->x16, X16 * CH * 2},
      {&f->xe16, table_slots(T * K, E) * C * 2}, {(void **)&f->overflow, 4},
      {(void **)&f->head, f->head_rows * VR * 4}, {(void **)&f->logits, (uint64_t)f->logit_rows * f->vocab * 4},
      {(void **)&f->gid, T * 4}, {&f->amax, ops_argmax_scratch((uint32_t)T)},
      {(void **)&f->res_sc, INK2_RES_SCRATCH * 4}, {&f->bx16, INK2_BX16 * MM_BATCH_TOKENS * CH * 2},
  };
  for (size_t i = 0; i < sizeof b / sizeof *b; i++)
    if (!ink2_dalloc(f, b[i].p, b[i].bytes))
      return false;
  if (g->loop.passes) /* the loop's */
  {
    const uint64_t core = (uint64_t)(g->loop.end - g->loop.start), rank = (uint64_t)g->loop.gate_rank;
    f->maxbank = core / (uint64_t)g->res_block > 1 ? (uint32_t)(core / (uint64_t)g->res_block - 1) : 1;
    struct
    {
      float **p;
      uint64_t bytes;
    } lb[] = {
        {&f->pre_h, T * C * 4}, {&f->bank, T * f->maxbank * C * 4}, {&f->bank_out, T * C * 4},
        {&f->msg, INK2_CTRLS * T * C * 4}, {&f->ctl, 3 * T * C * 4}, {&f->ccat, 2 * T * C * 4},
        {&f->cgate, T * rank * 4}, {&f->cgain, T * C * 4}, {&f->cforget, T * C * 4},
        {&f->ccat3, 3 * MM_BATCH_TOKENS * 2 * C * 4}, {&f->cgate3, 3 * MM_BATCH_TOKENS * rank * 4},
        {&f->cgain3, 3 * MM_BATCH_TOKENS * C * 4}, {&f->cforget3, 3 * MM_BATCH_TOKENS * C * 4},
    };
    for (size_t i = 0; i < sizeof lb / sizeof *lb; i++)
      if (!ink2_dalloc(f, (void **)lb[i].p, lb[i].bytes))
        return false;
  }
  f->fa_part = (float *)f->ed; /* (the down rows' buffer: ed_bytes) */
  f->ovf = (OverflowFlag){f->overflow, 4, false};
  if (!overflow_reset(&f->ovf, NULL))
    return false;
  /* the router's picks and weights and the experts' tables through pinned host memory: copies queued on the stream,
     one synchronize a layer for the picks, none for the tables (the next layer's picks' wait is past them) */
  if (!gpu_bufs_host(&f->bufs, f->ledger, &f->hpick, 2 * T * K * 4) ||
      !gpu_bufs_host(&f->bufs, f->ledger, &f->hgpick, T * K * 4) ||
      !gpu_bufs_host(&f->bufs, f->ledger, &f->m_host, table_bytes(T * K, E)))
    return false;
  f->slot = malloc(T * sizeof *f->slot);
  f->m_at = malloc((E + 1) * sizeof *f->m_at), f->m_by = malloc(T * K * sizeof *f->m_by);
  f->m_key = malloc(table_entries(T * K, E) * sizeof *f->m_key);
  f->m_ent = malloc(table_entries(T * K, E) * sizeof *f->m_ent), f->m_run = malloc(4 * E * sizeof *f->m_run);
  f->m_ex = malloc(E * sizeof *f->m_ex);
  f->m_build = malloc(table_bytes(T * K, E)); /* the tables built in cached memory, then one copy into the pinned */
  return f->hpick && f->slot && f->hgpick && f->m_at && f->m_by && f->m_host && f->m_key && f->m_build &&
         f->m_ent && f->m_run && f->m_ex;
}

Ink2Forward *ink2_forward_create(const Ink2Weights *w, const Ink2Options *o, char err[OSMOS_ERR])
{
  if (!w || !o || !o->max_ctx || !o->max_tokens)
    return osmos_fail(err, "no weights, or max_ctx / max_tokens 0"), NULL;
  Ink2Forward *f = calloc(1, sizeof *f);
  if (!f)
    return osmos_fail(err, "out of memory"), NULL;
  f->ledger = o->ledger;
  f->pf.on = o->profile;
  f->bf16_mm = w->layer[0].q && ink2_type(w->layer[0].q) == QTYPE_BF16; /* (mm_takes': BF16 the model's dense weights) */
  f->w = w, f->v8 = o->values_q8, f->max_ctx = o->max_ctx, f->max_tokens = o->max_tokens, f->vocab = (uint32_t)w->geo.n_vocab;
  f->logit_rows = o->logit_rows && o->logit_rows < o->max_tokens ? o->logit_rows : o->max_tokens;
  f->maxb = (uint32_t)((w->geo.n_layers + w->geo.res_block - 1) / w->geo.res_block);
  const uint32_t C = (uint32_t)w->geo.n_embd, F = (uint32_t)w->geo.ff_exp;
  for (int l = 0; l < w->geo.n_layers; l++) /* the experts on mm at every step (moe) */
  {
    const Ink2Layer *L = &w->layer[l];
    if (!mm_takes(f, L->exp_gate->type, F, C) || !mm_takes(f, L->exp_up->type, F, C) || !mm_takes(f, L->exp_down->type, C, F))
      return osmos_fail(err, "an expert tensor of a type or shape mm has no kernel for"),
             ink2_forward_destroy(f), NULL;
  }
  if (!upload(f))
    return osmos_fail(err, "a dense tensor not read or uploaded (GPU memory, or its file)"),
           ink2_forward_destroy(f), NULL;
  if (!buffers(f))
    return osmos_fail(err, "GPU memory refused (the step's buffers)"), ink2_forward_destroy(f), NULL;
  f->mtp = o->mtp;
  if (f->mtp && !ink2_mtp_up(f, err))
    return ink2_forward_destroy(f), NULL;
  if (!(f->picks = calloc((size_t)w->geo.n_layers * w->geo.n_experts, sizeof *f->picks)))
    return osmos_fail(err, "out of memory"), ink2_forward_destroy(f), NULL;
  f->fixed_bytes = f->bufs.bytes; /* the fixed part, counted: its record (the dense weights, the buffers, the drafter -
                                     0.16.0: the GPU's free memory read before and after, anything else's moving it) */
  if (!experts(f, o->room, o->budget, o->picks, err) || !whole_tables(f, err))
    return ink2_forward_destroy(f), NULL;
  return f;
}

void ink2_forward_destroy(Ink2Forward *f)
{
  if (!f)
    return;
  for (int k = 0; f->pf.on && k < 2; k++)
  {
    double all = 0;
    for (int b = 0; b < PF_N; b++)
      all += b == PF_WAIT || b == PF_ROUTE || b == PF_XK ? 0 : f->pf.total[k][b];
    if (!f->pf.steps[k])
      continue;
    fprintf(stderr, "ink2 profile (%s steps): %llu steps, %llu tokens, %.3f s\n", k ? "prompt" : "decode",
            (unsigned long long)f->pf.steps[k], (unsigned long long)f->pf.tokens[k], all);
    for (int b = 0; b < PF_N; b++)
      fprintf(stderr, "  %-13s %8.2f ms a step  %5.1f%%\n", kPfName[b], 1e3 * f->pf.total[k][b] / (double)f->pf.steps[k],
              100 * f->pf.total[k][b] / all);
  }
  if (f->pf.on && (f->pf.steps[0] || f->pf.steps[1]))
  {
    MembraneStats ms;
    membrane_stats(f->mem, &ms);
    fprintf(stderr, "  membrane (all steps): %llu picks not in memory, %llu of them guessed; %llu guesses read; waits "
                    "%llu unguessed %.1f s, %llu guessed %.1f s\n",
            (unsigned long long)ms.streamed, (unsigned long long)ms.guessed, (unsigned long long)ms.guess_read,
            (unsigned long long)ms.waits[0], ms.wait_ns[0] * 1e-9, (unsigned long long)ms.waits[1],
            ms.wait_ns[1] * 1e-9);
  }
  hipDeviceSynchronize();
  membrane_destroy(f->mem);
  host_slots_destroy(f->slots), host_slots_destroy(f->stream);
  for (int i = 0; f->fds && i < f->w->model->n_files; i++)
    if (f->fds[i] >= 0)
      close(f->fds[i]);
  free(f->fds);
  for (int l = 0; l < INK2_MAX_LAYERS; l++)
    for (int k = 0; k < 3; k++)
      free(f->L[l].scale[k]);
  gpu_bufs_free(&f->bufs); /* every GPU and pinned buffer, the forward's and its drafter's: from the record */
  free(f->picks), free(f->picks_all), free(f->slot), free(f->m_at), free(f->m_by);
  free(f->m_key), free(f->m_build), free(f->m_ent), free(f->m_run), free(f->m_ex);
  free(f);
}

uint64_t ink2_forward_bytes(const Ink2Forward *f) { return f ? f->fixed_bytes + f->pinned_bytes : 0; }

bool ink2_forward_whole(const Ink2Forward *f) { return f && f->whole; }

/* the host's counts, plus (a whole membrane's) the GPU's: copied at the call */
const uint64_t *ink2_forward_picks(const Ink2Forward *f)
{
  if (!f || !f->whole)
    return f ? f->picks : NULL;
  const size_t n = (size_t)f->w->geo.n_layers * f->w->geo.n_experts;
  uint32_t *d = malloc(n * 4);
  if (!d || hipMemcpy(d, f->picks_d, n * 4, hipMemcpyDeviceToHost) != hipSuccess)
    return free(d), NULL;
  for (size_t u = 0; u < n; u++)
    f->picks_all[u] = f->picks[u] + d[u];
  free(d);
  return f->picks_all;
}

uint32_t ink2_type(const GgufTensor *t) { return t->type == QUANT_Q5_K ? QUANT_Q5_KR : t->type; }

/* the drafter's y [n][rows] = W x: mv_run, MV_MAX_TOKENS tokens a launch */
bool ink2_mv(const GgufTensor *t, const void *w, const float *x, uint32_t n, float *y)
{
  const uint32_t rows = (uint32_t)t->dims[1], cols = (uint32_t)t->dims[0];
  const uint32_t type = ink2_type(t);
  for (uint32_t at = 0; at < n; at += MV_MAX_TOKENS)
  {
    const uint32_t k = n - at < MV_MAX_TOKENS ? n - at : MV_MAX_TOKENS;
    if (!mv_run(type, w, rows, cols, x + (uint64_t)at * cols, k, y + (uint64_t)at * rows, NULL))
      return false;
  }
  return true;
}

/* the model's y [n][rows] = W x by the matrix's type and shape alone, never the step's: WMMA's sums are not the
   matvec's (DESIGN.md, "0.5.5: ... WMMA's sums probed"), so a token's bits are the same in every step only if every
   step takes the same kernel. mm where it has one and the shape tiles (mm_tokens: MM_MAX_TOKENS a launch through
   f->x16), else the matvec (mv_strided: past MV_MAX_TOKENS its wide tiles, each token's sums its own) */
static bool mat(Ink2Forward *f, const GgufTensor *t, const void *w, const float *x, uint32_t n, float *y)
{
  const uint32_t rows = (uint32_t)t->dims[1], cols = (uint32_t)t->dims[0], type = ink2_type(t);
  if (!mm_takes(f, type, rows, cols))
    return mv_strided(type, w, rows, cols, x, cols, n, y, rows, NULL);
  return mm_tokens(type, w, rows, y, rows, NULL, 0, NULL, 0, cols, x, cols, n, f->x16, f->overflow, false, NULL);
}

/* two matrices on one x (mm_run2: each the bits mm_run gives it alone), or one with w1 NULL; `ready`: f->x16 holds
   this x's f16 already (a matrix before it on the same x) - otherwise each as mat() */
static bool mat2(Ink2Forward *f, const GgufTensor *t0, const void *w0, float *y0, const GgufTensor *t1, const void *w1,
                 float *y1, const float *x, uint32_t n, bool ready)
{
  const uint32_t cols = (uint32_t)t0->dims[0], type = ink2_type(t0), r0 = (uint32_t)t0->dims[1],
                 r1 = t1 ? (uint32_t)t1->dims[1] : 0;
  if (!mm_takes(f, type, r0, cols) || (t1 && (ink2_type(t1) != type || (uint32_t)t1->dims[0] != cols ||
                                           !mm_takes(f, type, r1, cols))))
    return mat(f, t0, w0, x, n, y0) && (!t1 || mat(f, t1, w1, x, n, y1));
  return mm_tokens(type, w0, r0, y0, r0, t1 ? w1 : NULL, r1, y1, r1, cols, x, cols, n, f->x16, f->overflow, ready,
                   NULL);
}

/* y = x's rows RMS-normed (times w), and for t on mm in one launch (n <= MM_MAX_TOKENS) f->x16 their f16 too -
   ops_rms_x16's, mm_x16's bytes: *ready, mat2's for t on y */
static bool rms_for(Ink2Forward *f, const GgufTensor *t, const float *x, const float *w, uint32_t n, float *y,
                    bool *ready)
{
  const uint32_t C = (uint32_t)f->w->geo.n_embd;
  const float eps = f->w->geo.rms_eps;
  *ready = n <= MM_MAX_TOKENS && mm_takes(f, ink2_type(t), (uint32_t)t->dims[1], (uint32_t)t->dims[0]);
  return *ready ? ops_rms_x16(x, C, C, w, eps, n, y, C, f->x16, f->overflow, NULL)
                : ops_rms(x, C, C, w, eps, n, y, C, NULL);
}

/* one role's experts (their entries in ascending count): a launch a run of the same column tiles (mm_col_tiles), so
   a one-tile expert is not run in a hot one's many-tile kernel (hip_ffn.c's group). each expert's rows its own: the
   launches' order changes no bit */
static bool group(uint32_t type, const void *const *dtab, const int32_t *cnt, const int32_t *dcnt, uint32_t ne,
                  uint32_t rows, uint32_t cols, const void *x16, const uint32_t *dxo, float *y, const uint32_t *dyo,
                  uint32_t ys)
{
  bool ok = true;
  for (uint32_t a = 0, z = 0; ok && a < ne; a = z)
  {
    const uint32_t ct = mm_col_tiles(type, (uint32_t)cnt[a]);
    for (z = a + 1; z < ne && mm_col_tiles(type, (uint32_t)cnt[z]) == ct;)
      z++;
    ok = mm_group(type, dtab + a, dcnt + a, z - a, (uint32_t)cnt[z - 1], rows, cols, x16, dxo + a, cols, y, dyo + a,
                  ys, NULL);
  }
  return ok;
}

/* the routed experts of layer l for the run's n tokens: each distinct pick held (its read begun), then read once for
   the tokens that picked it (mm_group at every step: an entry an expert's part of up to MM_MAX_TOKENS of its rows,
   a token's sums its own whatever the others) - gate and up, their scales and the swiglu, down - and each token's
   summed in its slot order (ink2_moe_sum) */
/* moe with every expert in memory: the router's picks never on the host - the tables built on the GPU
   (q4x_expert_tiles: the picks by expert, a part of up to 128 an entry, the entries in runs of their tiles;
   ink2_moe_scales: the scales and weights moe's host loop writes), each run a launch (its entries past the live ones
   none). which picks share a part, and where, changes no bit: the same sums as moe's tables */
static bool moe_whole(Ink2Forward *f, int l, uint32_t n, char *err)
{
  const Ink2Geometry *g = &f->w->geo;
  const Ink2Layer *L = &f->w->layer[l];
  const Ink2Dev *d = &f->L[l];
  const uint32_t C = (uint32_t)g->n_embd, E = (uint32_t)g->n_experts, K = (uint32_t)g->n_used,
                 F = (uint32_t)g->ff_exp, P = n * K;
  Q4xExpertBounds bd;
  q4x_expert_bounds(P, E, &bd);
  Q4xExpertTiles et = {.cnt = f->t_cnt, .gidx = f->t_gidx, .hidx = f->t_hidx, .xg = f->t_xg, .yg = f->t_yg,
                       .xd = f->t_xd, .yd = f->t_yd, .where = f->t_where, .off = f->t_off};
  for (int i = 0; i < 3; i++)
    et.ptr[i] = f->t_ptr + (uint64_t)i * f->t_entries, et.tab[i] = f->exp_tab + ((uint64_t)l * 3 + (uint64_t)i) * E;
  const GgufTensor *t[3] = {L->exp_gate, L->exp_up, L->exp_down};
  const uint32_t rows[3] = {F, F, C}, cols[3] = {C, C, F};
  float *const y[3] = {f->eg, f->eu, f->ed};
  const uint32_t *xo[3] = {f->t_xg, f->t_xg, f->t_xd}, *yo[3] = {f->t_yg, f->t_yg, f->t_yd};
  float *pw = (float *)(f->pick + P);
  bool ok = mat2(f, L->router, d->router, f->logit, NULL, NULL, NULL, f->xn, n, f->router16) &&
            ops_router_scored(f->logit, d->router_bias, E, K, g->route_scale, OPS_SCORE_SIGMOID, n, f->pick, pw,
                              NULL) &&
            q4x_expert_tiles(f->pick, P, K, E, C, F, &et, NULL) &&
            ink2_moe_scales(f->pick, pw, f->t_where, f->scale_tab + (uint64_t)l * 3 * E, E, P, f->t_sg, f->t_su,
                            f->t_a, f->picks_d + (uint64_t)l * E, NULL) &&
            mm_x16_rows(f->xn, C, f->t_gidx, bd.slots, C, f->xe16, f->overflow, NULL);
  for (int k = 0; ok && k < 3; k++)
  {
    if (k == 2)
      ok = ink2_swiglu_x16(f->eg, f->eu, f->t_sg, f->t_su, F, g->gate_softcap, g->up_softcap, f->t_hidx, bd.slots,
                           f->xe16, f->overflow, NULL);
    /* run r's experts more than MM_TILE << (r - 1) tokens each (its bound the pairs'): none in a step of fewer
       tokens (a token picks an expert once) - a decode's verify launched them empty, 3 a pass */
    for (int r = 0; ok && r < Q4X_TILE_RUNS; r++)
      if (bd.count[r] && (r == 0 || n > (uint32_t)MM_TILE << (r - 1)))
      {
        const uint32_t at = bd.first[r];
        ok = mm_group(t[k]->type, et.ptr[k] + at, f->t_cnt + at, bd.count[r], MM_TILE << r, rows[k], cols[k],
                      f->xe16, xo[k] + at, cols[k], y[k], yo[k] + at, rows[k], NULL);
      }
  }
  f->m_a = f->t_a, f->m_row = f->t_where;
  return ok || osmos_fail(err, "a GPU call failed (the experts, their tables on the GPU)");
}

static bool moe(Ink2Forward *f, int l, uint32_t n, char *err)
{
  if (f->whole)
    return moe_whole(f, l, n, err);
  const Ink2Geometry *g = &f->w->geo;
  const Ink2Layer *L = &f->w->layer[l];
  const Ink2Dev *d = &f->L[l];
  const uint32_t C = (uint32_t)g->n_embd, E = (uint32_t)g->n_experts, K = (uint32_t)g->n_used,
                 F = (uint32_t)g->ff_exp, P = n * K;
  const double r0 = f->pf.on ? pf_now() : 0;
  /* a prompt's read ahead (its picks a layer's experts at least): the next layer's router on this layer's MoE input,
     the union of its picks read into the stream while this layer's experts and the next attention run (27.0 against
     31.5 s of prompt steps, 900 tokens). not a decode's: measured, the guess named 26% of its misses and read ~80
     experts a token never picked - the next router's input is the attention residual's over the blocks, far from
     this one's - their reads the misses' bandwidth (4.25 against 4.6 tok/s). which are read ahead changes no bit */
  const bool next = l + 1 < g->n_layers && P >= E;
  const Ink2Layer *L1 = next ? &f->w->layer[l + 1] : NULL;
  /* the picks' weights after them (pick + P): the two in one copy to the host */
  float *pw = (float *)(f->pick + P), *hw = (float *)(f->hpick + P);
  if (!mat2(f, L->router, d->router, f->logit, NULL, NULL, NULL, f->xn, n, f->router16) ||
      !ops_router_scored(f->logit, d->router_bias, E, K, g->route_scale, OPS_SCORE_SIGMOID, n, f->pick, pw, NULL) ||
      (next && (!mat(f, L1->router, f->L[l + 1].router, f->xn, n, f->logit) ||
                !ops_router_scored(f->logit, f->L[l + 1].router_bias, E, K, g->route_scale, OPS_SCORE_SIGMOID, n,
                                   f->gpick, f->gw, NULL) ||
                hipMemcpyAsync(f->hgpick, f->gpick, P * 4, hipMemcpyDeviceToHost, NULL) != hipSuccess)) ||
      hipMemcpyAsync(f->hpick, f->pick, 2 * P * 4, hipMemcpyDeviceToHost, NULL) != hipSuccess ||
      hipStreamSynchronize(NULL) != hipSuccess)
    return osmos_fail(err, "a GPU call failed (the router)");
  if (f->pf.on)
    f->pf.step[PF_ROUTE] += pf_now() - r0;
  /* the GPU is past the last layer's experts (the copy waited for it): the visit ends their holds */
  if (!membrane_visit(f->mem, (uint32_t)l, P >= E, err))
    return false;
  for (uint32_t m = 0; P < E && m < P; m++) /* a decode's picks counted: in memory or not (the stats') */
    membrane_score(f->mem, (uint32_t)l * E + (uint32_t)f->hpick[m]);
  /* the tables, one region sent in one copy, each table's place fixed by the step's most pairs: the entries' gate, up
     and down pointers, counts, their slots' and rows' offsets (gate and up's x and y, down's); each slot's token and
     sorted row (-1: a tile's padding); each pair's sorted row and weight, each sorted row's scales */
  const uint64_t ecap = table_entries((uint64_t)f->max_tokens * K, E), scap = table_slots((uint64_t)f->max_tokens * K, E);
  uint8_t *h = f->m_build; /* (cached: built here, then one copy into the pinned) */
  const void **tg = (const void **)h, **tu = tg + ecap, **td = tu + ecap;
  int32_t *cnt = (int32_t *)(td + ecap);
  uint32_t *xg = (uint32_t *)(cnt + ecap), *yg = xg + ecap, *xd = yg + ecap, *yd = xd + ecap;
  int32_t *gidx = (int32_t *)(yd + ecap), *hidx = gidx + scap, *row = hidx + scap;
  float *a = (float *)(row + P), *sg = a + P, *su = sg + P;
  /* the pairs by expert, then pair order (moe_route: sorted row r is pair by[r]), each slot's token and row */
  MoeRoute rt = {f->m_by, (uint32_t *)row, f->m_run, f->m_run + E, f->m_run + 2 * E, f->m_run + 3 * E, gidx, hidx,
                 f->m_at, 0, 0};
  if (!moe_route(f->hpick, P, K, E, MM_TILE, &rt))
    return osmos_fail(err, "the router picked an expert the model does not have");
  for (uint32_t m = 0; m < P; m++)
    f->picks[(uint32_t)l * E + (uint32_t)f->hpick[m]]++;
  for (uint32_t j = 0; j < rt.runs; j++) /* every distinct pick held first: the cold ones' reads under way together */
    if (!membrane_hold(f->mem, (uint32_t)l * E + rt.expert[j], err))
      return false;
  /* then the next layer's named: this layer's reads queued first, the stream's room theirs before the guess's */
  for (uint32_t m = 0; next && m < P; m++)
    if (f->hgpick[m] >= 0 && (uint32_t)f->hgpick[m] < E)
    {
      const uint32_t unit = (uint32_t)(l + 1) * E + (uint32_t)f->hgpick[m];
      membrane_ahead(f->mem, unit);
    }
  /* each run's expert in memory (its parts' addresses), its rows' scales; the entries by count (moe_entries: group's
     runs of the same column tiles contiguous, each count's in build order) */
  for (uint32_t j = 0; j < rt.runs; j++)
  {
    const uint32_t e = rt.expert[j], unit = (uint32_t)l * E + e;
    const double wt = f->pf.on ? pf_now() : 0;
    if (!membrane_wait(f->mem, unit, f->m_ex[j], err))
      return false;
    if (f->pf.on)
      f->pf.step[PF_WAIT] += pf_now() - wt;
    for (uint32_t q = rt.start[j]; q < rt.start[j] + rt.count[j]; q++)
    {
      const uint32_t m = rt.by[q];
      sg[q] = d->scale[0][e], su[q] = d->scale[1][e], a[m] = hw[m] * d->scale[2][e];
    }
  }
  uint32_t ne = 0;
  if (!moe_entries(&rt, MM_MAX_TOKENS, f->m_ent, f->m_key, &ne))
    return osmos_fail(err, "the experts' entries refused");
  for (uint32_t k = 0; k < ne; k++)
  {
    const MoeEntry *en = &f->m_ent[k];
    const uint32_t s0 = rt.slot0[en->run] + en->off, q = rt.start[en->run] + en->off;
    tg[k] = f->m_ex[en->run][0], tu[k] = f->m_ex[en->run][1], td[k] = f->m_ex[en->run][2], cnt[k] = (int32_t)en->cnt;
    xg[k] = s0 * C, yg[k] = q * F, xd[k] = s0 * F, yd[k] = q * C;
  }
  const uint32_t slots = rt.slots;
  const uint64_t bytes = (uint64_t)((uint8_t *)(su + P) - h);
  uint8_t *dv = f->m_dev;
  const void **dg = (const void **)dv, **du = dg + ecap, **dd = du + ecap;
  const int32_t *dcnt = (const int32_t *)(dd + ecap);
  const uint32_t *dxg = (const uint32_t *)(dcnt + ecap), *dyg = dxg + ecap, *dxd = dyg + ecap, *dyd = dxd + ecap;
  const int32_t *dgidx = (const int32_t *)(dyd + ecap), *dhidx = dgidx + scap, *drow = dhidx + scap;
  const float *da = (const float *)(drow + P), *dsg = da + P, *dsu = dsg + P;
  const double x0 = f->pf.on ? pf_now() : 0;
  memcpy(f->m_host, h, bytes);
  const bool ok =
      hipMemcpyAsync(dv, f->m_host, bytes, hipMemcpyHostToDevice, NULL) == hipSuccess &&
      mm_x16_rows(f->xn, C, dgidx, slots, C, f->xe16, f->overflow, NULL) &&
      group(L->exp_gate->type, dg, cnt, dcnt, ne, F, C, f->xe16, dxg, f->eg, dyg, F) &&
      group(L->exp_up->type, du, cnt, dcnt, ne, F, C, f->xe16, dxg, f->eu, dyg, F) &&
      ink2_swiglu_x16(f->eg, f->eu, dsg, dsu, F, g->gate_softcap, g->up_softcap, dhidx, slots, f->xe16,
                      f->overflow, NULL) &&
      group(L->exp_down->type, dd, cnt, dcnt, ne, C, F, f->xe16, dxd, f->ed, dyd, C);
  f->m_a = da, f->m_row = drow;
  if (!ok)
    return osmos_fail(err, "a GPU call failed (the experts)");
  if (f->pf.on)
    hipDeviceSynchronize(), f->pf.step[PF_XK] += pf_now() - x0;
  return true;
}

/* a conv on tail t's history (its last taps - 1 rows), x's n rows then the tail's: written by the conv as it reads
   them when they fit (ink2_tail_claim), else ink2_tail_put after it - the same rows either way. add (a static
   conv's, NULL none): y = add + the conv, a residual's sum in the conv's launch */
static bool conv_put(Ink2Tail *t, const float *x, uint32_t n, uint32_t ch, uint32_t taps, const float *w,
                     uint32_t heads, const float *dyn, float res, float scale, const float *add, float *y)
{
  const float *hist = ink2_tail_last(t, taps - 1);
  float *keep = ink2_tail_claim(t, n);
  const bool ok = dyn ? !add && ink2_conv_dyn_keep(x, n, ch, taps, heads, w, dyn, hist, res, scale, y, keep, NULL)
                      : ink2_conv_keep(x, n, ch, taps, w, hist, res, scale, add, y, keep, NULL);
  return ok && (keep || ink2_tail_put(t, x, n));
}

/* a run: n tokens of s at its next positions through the model, their logits at rows row0 .. (those before the
   prompt's end - 1 not) */
/* f->bx16's input i: MM_BATCH_TOKENS rows of the widest input (buffers' CH) */
static void *bx16_at(const Ink2Forward *f, int i)
{
  const Ink2Geometry *g = &f->w->geo;
  const uint64_t C = (uint64_t)g->n_embd, HD = (uint64_t)g->n_heads * (uint64_t)g->head_dim,
                 CH0 = HD > C ? HD : C, CH = g->loop.passes && 2 * C > CH0 ? 2 * C : CH0;
  return (uint8_t *)f->bx16 + (uint64_t)i * MM_BATCH_TOKENS * CH * 2;
}

/* matrices of one type and cols on mm (mm_takes): an mm_batch's */
static bool batchable(const Ink2Forward *f, const GgufTensor *const *t, int count, uint32_t cols)
{
  const uint32_t type = ink2_type(t[0]);
  for (int i = 0; i < count; i++)
    if (t[i] && (ink2_type(t[i]) != type || (uint32_t)t[i]->dims[0] != cols ||
                 !mm_takes(f, type, (uint32_t)t[i]->dims[1], cols)))
      return false;
  return true;
}

/* a decode's (n <= MM_BATCH_TOKENS) norms in one launch (q's into xn and f->x16; apart: k's into kn and bx16 0,
   v's into vn and bx16 1 - each rms_for's bits) and its projections on them in one mm_batch: q, gate and rel on q's
   norm, k and v on theirs (else q's), a full layer's conv projections on k's and v's - each matrix's bits mat2's
   (mm_run's tiles; the norms' f16 mm_x16's bytes). false: not taken (a matrix not the batch's type on mm), nothing
   launched; *ok the launches' */
static bool proj_batch(Ink2Forward *f, const Ink2Layer *L, const Ink2Dev *d, bool full, bool apart, const float *aq,
                       const float *ak, const float *av, const float *q_norm, const float *k_norm,
                       const float *v_norm, uint32_t n, float *dyn_k, float *dyn_v, bool *ok)
{
  const uint32_t C = (uint32_t)f->w->geo.n_embd;
  const GgufTensor *t[7] = {L->q, L->gate, L->rel, L->k, L->v, full ? L->k_conv_proj : NULL,
                            full ? L->v_conv_proj : NULL};
  if (n > MM_BATCH_TOKENS || !batchable(f, t, 7, C))
    return false;
  void *x16k = apart ? bx16_at(f, 0) : f->x16, *x16v = apart ? bx16_at(f, 1) : f->x16;
  const OpsRmsPart norms[3] = {{aq, q_norm, f->xn, f->x16, C, C, C, 0},
                               {ak, k_norm, f->kn, x16k, C, C, C, 0},
                               {av, v_norm, f->vn, x16v, C, C, C, 0}};
  *ok = ops_rms_x16_many(norms, apart ? 3 : 1, C, f->w->geo.rms_eps, n, f->overflow, NULL);
  const void *w[7] = {d->q, d->gate, d->rel, d->k, d->v, d->k_conv_proj, d->v_conv_proj};
  const void *x[7] = {f->x16, f->x16, f->x16, x16k, x16v, x16k, x16v};
  float *y[7] = {f->q, f->gate, f->rel, f->k, f->v, dyn_k, dyn_v};
  MmPart p[7];
  uint32_t count = 0;
  for (int i = 0; i < 7; i++)
    if (t[i])
    {
      const uint32_t rows = (uint32_t)t[i]->dims[1];
      p[count++] = (MmPart){w[i], rows, x[i], y[i], rows};
    }
  *ok = *ok && mm_batch(ink2_type(L->q), C, p, count, n, NULL);
  return true;
}

/* a loop's control of input x by message m (upstream's control): gate = down(concat(rms(x), rms(m))), out = x -
   tanh(forget(gate)) x + tanh(gain(gate)) m; out may be x, NULL: the gain and forget alone (the caller applies them:
   ops_rms_x16_ctrl) */
static bool control(Ink2Forward *f, const Ink2Layer *L, const Ink2Dev *d, int c, const float *x, const float *m,
                    uint32_t n, float *out)
{
  const uint32_t C = (uint32_t)f->w->geo.n_embd;
  const float eps = f->w->geo.rms_eps;
  /* the two norms' f16 side by side in x16 where down is mm's: its input's, not converted again (ops_rms's bits) */
  const bool r16 = n <= MM_MAX_TOKENS && mm_takes(f, ink2_type(L->ctrl_down[c]), (uint32_t)L->ctrl_down[c]->dims[1],
                                                  (uint32_t)L->ctrl_down[c]->dims[0]);
  const OpsRmsPart norms[2] = {{x, NULL, f->ccat, f->x16, C, 2 * C, 2 * C, 0},
                               {m, NULL, f->ccat + C, f->x16, C, 2 * C, 2 * C, C}};
  const bool normed = r16 ? ops_rms_x16_many(norms, 2, C, eps, n, f->overflow, NULL)
                          : ops_rms(x, C, C, NULL, eps, n, f->ccat, 2 * C, NULL) &&
                                ops_rms(m, C, C, NULL, eps, n, f->ccat + C, 2 * C, NULL);
  if (!normed)
    return false;
  /* the down's gate and its f16 in one launch (mm_run_x16: mm_run then mm_x16's bytes), the gain and forget on that
     f16 (bx16 5's: the controls' gates', free by the FFN's); else the down, then the pair converting its gate */
  const uint32_t G = (uint32_t)f->w->geo.loop.gate_rank;
  const GgufTensor *gf[2] = {L->ctrl_gain[c], L->ctrl_forget[c]};
  void *g16 = bx16_at(f, 5);
  if (r16 && batchable(f, gf, 2, G) &&
      mm_run_x16(ink2_type(L->ctrl_down[c]), d->ctrl_down[c], G, 2 * C, f->x16, n, f->cgate, G, g16, f->overflow, NULL))
    return mm_tokens(ink2_type(gf[0]), d->ctrl_gain[c], C, f->cgain, C, d->ctrl_forget[c], C, f->cforget, C, G,
                     f->cgate, G, n, g16, f->overflow, true, NULL) &&
           (!out || ink2_loop_ctrl(x, m, f->cgain, f->cforget, (uint64_t)n * C, out, NULL));
  return mat2(f, L->ctrl_down[c], d->ctrl_down[c], f->cgate, NULL, NULL, NULL, f->ccat, n, r16) &&
         mat2(f, L->ctrl_gain[c], d->ctrl_gain[c], f->cgain, L->ctrl_forget[c], d->ctrl_forget[c], f->cforget,
              f->cgate, n, false) &&
         (!out || ink2_loop_ctrl(x, m, f->cgain, f->cforget, (uint64_t)n * C, out, NULL));
}

/* a decode's three controls (q, k, v: control's for each) in batches: their norms each control's (x16s bx16 2 .. 4),
   the three downs one mm_batch, their gates' f16 (bx16 5 .. 7), the gains and forgets one mm_batch, the three
   ink2_loop_ctrl - each value control's bits. false: not taken (n past MM_BATCH_TOKENS, a matrix not on mm) */
static bool controls3(Ink2Forward *f, const Ink2Layer *L, const Ink2Dev *d, const float *const *x, const float *m,
                      uint32_t n, float *out, bool *ok)
{
  const uint32_t C = (uint32_t)f->w->geo.n_embd, G = (uint32_t)f->w->geo.loop.gate_rank;
  const float eps = f->w->geo.rms_eps;
  const uint64_t nC = (uint64_t)n * C;
  const GgufTensor *down[3] = {L->ctrl_down[INK2_CTRL_Q], L->ctrl_down[INK2_CTRL_K], L->ctrl_down[INK2_CTRL_V]};
  const GgufTensor *gf[6] = {L->ctrl_gain[INK2_CTRL_Q],   L->ctrl_forget[INK2_CTRL_Q], L->ctrl_gain[INK2_CTRL_K],
                             L->ctrl_forget[INK2_CTRL_K], L->ctrl_gain[INK2_CTRL_V],   L->ctrl_forget[INK2_CTRL_V]};
  if (n > MM_BATCH_TOKENS || !batchable(f, down, 3, 2 * C) || !batchable(f, gf, 6, G))
    return false;
  MmPart pd[3], pg[6];
  OpsRmsPart norms[6];
  const uint64_t n16 = (n + MM_TILE - 1) / MM_TILE * MM_TILE;
  for (int c = 0; c < 3; c++)
  {
    const int k = INK2_CTRL_Q + c;
    /* the gates side by side in a row (gate c at c G, rows 3 G apart): one mm_x16_heads their three f16s, gate c's at
       c n16 G halves of bx16 5's (each mm_x16's bits) */
    float *cat = f->ccat3 + (uint64_t)c * MM_BATCH_TOKENS * 2 * C, *gate = f->cgate3 + (uint64_t)c * G;
    void *x16c = bx16_at(f, 2 + c), *x16g = (uint8_t *)bx16_at(f, 5) + (uint64_t)c * n16 * G * 2;
    norms[2 * c] = (OpsRmsPart){x[c], NULL, cat, x16c, C, 2 * C, 2 * C, 0};
    norms[2 * c + 1] = (OpsRmsPart){m + c * nC, NULL, cat + C, x16c, C, 2 * C, 2 * C, C};
    pd[c] = (MmPart){d->ctrl_down[k], G, x16c, gate, 3 * G};
    float *gain = f->cgain3 + (uint64_t)c * MM_BATCH_TOKENS * C, *forget = f->cforget3 + (uint64_t)c * MM_BATCH_TOKENS * C;
    pg[2 * c] = (MmPart){d->ctrl_gain[k], C, x16g, gain, C};
    pg[2 * c + 1] = (MmPart){d->ctrl_forget[k], C, x16g, forget, C};
  }
  *ok = ops_rms_x16_many(norms, 6, C, eps, n, f->overflow, NULL) && mm_batch(ink2_type(down[0]), 2 * C, pd, 3, n, NULL);
  *ok = *ok && mm_x16_heads(f->cgate3, 3 * G, n, 3, G, G, bx16_at(f, 5), f->overflow, NULL);
  *ok = *ok && mm_batch(ink2_type(gf[0]), G, pg, 6, n, NULL);
  Ink2LoopCtrlPart lc[3];
  for (int c = 0; c < 3; c++)
    lc[c] = (Ink2LoopCtrlPart){x[c], m + c * nC, f->cgain3 + (uint64_t)c * MM_BATCH_TOKENS * C,
                               f->cforget3 + (uint64_t)c * MM_BATCH_TOKENS * C, out + c * nC};
  *ok = *ok && ink2_loop_ctrl_many(lc, 3, nC, NULL);
  return true;
}

static bool run(Ink2Forward *f, Ink2Seq *s, const int32_t *tok, uint32_t n, uint32_t row0, char *err)
{
  const Ink2Weights *W = f->w;
  const Ink2Geometry *g = &W->geo;
  const uint32_t C = (uint32_t)g->n_embd, H = (uint32_t)g->n_heads, KV = (uint32_t)g->n_kv_heads,
                 D = (uint32_t)g->head_dim, R = (uint32_t)g->d_rel, taps = (uint32_t)g->conv, maxb = f->maxb,
                 pos = (uint32_t)s->pos;
  const bool bare = g->keys == INK2_KEYS_UNSLOTH_SMALL; /* its convs' residual and scale not applied */
  const float res_scale = 1.0f / sqrtf((float)C), conv_res = bare ? 0.0f : 1.0f,
              conv_scale = bare ? 1.0f : 1.0f / sqrtf((float)taps + 1),
              dyn_scale = bare ? 1.0f : 1.0f / sqrtf(2.0f * taps + 1);
#define GPU(x)                                                                                                         \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!(x))                                                                                                          \
      return osmos_fail(err, "a GPU call failed: " #x);                                                       \
  } while (0)
  pf(f, -1);
  GPU(hipMemcpy(f->ids_d, tok, n * 4, hipMemcpyHostToDevice) == hipSuccess);
  GPU(ops_embed(W->embed->type, f->embed, C, f->ids_d, n, 1, f->h, NULL));
  GPU(ops_rms(f->h, C, C, f->embed_norm, g->rms_eps, n, f->h, C, NULL));
  for (int c = 0; c < INK2_CONVS; c++) /* every tail's room first: the step's reads and writes in one place */
    for (int sl = 0; sl < g->n_slots; sl++)
      GPU(ink2_tail_room(&s->conv[sl][c], n));
  uint32_t nb = 0;
  pf(f, PF_HEAD);
  if (f->mtp) /* the drafter's taps: this run's rows of the step's */
    s->tap_row0 = row0, s->tap_pos0 = (int32_t)pos, s->tap_step = f->steps;
  /* the slots in their order: the prelude, the core `passes` times (each from the prelude's output and blocks; a pass
     after the first its own slots and its layers' messages from the pass before's bank), the tail */
  const int core = g->loop.passes ? g->loop.end - g->loop.start : 0, passes = core ? g->loop.passes : 1,
            cs = core ? g->loop.start : g->n_layers;
  const uint64_t nC = (uint64_t)n * C;
  uint32_t pre_nb = 0, nbank = 0;
  for (int step = 0; step < g->n_slots; step++)
  {
    const int in = step - cs, pass = core && in >= 0 && in < passes * core ? in / core : -1;
    const int l = pass >= 0 ? cs + in % core : step < cs ? step : step - (passes - 1) * core;
    const int slot = pass > 0 ? g->n_layers + (pass - 1) * core + in % core : l;
    const bool looped = pass > 0;
    if (core && step == cs) /* the prelude's output and blocks: each pass's start */
    {
      GPU(hipMemcpyAsync(f->pre_h, f->h, nC * 4, hipMemcpyDeviceToDevice, NULL) == hipSuccess);
      pre_nb = nb;
    }
    if (looped && l == cs) /* the pass before's bank (its blocks after its first) and output; then the prelude's */
    {
      nbank = nb - pre_nb - 1;
      if (nbank > f->maxbank)
        return osmos_fail(err, "a pass's blocks past the loop's bank");
      if (nbank)
        GPU(hipMemcpy2DAsync(f->bank, (uint64_t)f->maxbank * C * 4, f->blocks + (uint64_t)(pre_nb + 1) * C,
                             (uint64_t)maxb * C * 4, (uint64_t)nbank * C * 4, n, hipMemcpyDeviceToDevice,
                             NULL) == hipSuccess);
      GPU(hipMemcpyAsync(f->bank_out, f->h, nC * 4, hipMemcpyDeviceToDevice, NULL) == hipSuccess &&
          hipMemcpyAsync(f->h, f->pre_h, nC * 4, hipMemcpyDeviceToDevice, NULL) == hipSuccess);
      nb = pre_nb;
    }
    const Ink2Layer *L = &W->layer[l];
    const Ink2Dev *d = &f->L[l];
    const bool full = !g->sliding[l], boundary = l % g->res_block == 0;
    if (looped) /* the layer's messages: q, k, v, ffn */
      GPU(ink2_attn_res(d->msg_q, INK2_CTRLS, f->bank, nbank, f->maxbank, f->bank_out, n, C, g->rms_eps, res_scale,
                        f->msg, f->res_sc, NULL));
    for (int k = 0; f->mtp && k + 1 < f->mtp->n_taps; k++) /* h entering a target layer: the attention residual's
                                                              running partial */
      if (f->mtp->targets[k] == step) /* (a step of the slots' order: unlooped, the layer) */
        GPU(ink2_mtp_tap(f, k, f->h, n, row0));
    const float *aq = f->h, *ak = f->h, *av = f->h;
    if (d->attn_res_q)
    {
      GPU(ink2_attn_res(d->attn_res_q, full ? 3 : 1, f->blocks, nb, maxb, f->h, n, C, g->rms_eps, res_scale, f->a3,
                        f->res_sc, NULL));
      aq = f->a3, ak = full ? f->a3 + (uint64_t)n * C : f->a3, av = full ? f->a3 + (uint64_t)2 * n * C : f->a3;
    }
    if (looped) /* each input by its message: a sliding layer's q, k and v then apart */
    {
      const float *const xs[3] = {aq, ak, av};
      bool cok = false;
      if (controls3(f, L, d, xs, f->msg, n, f->ctl, &cok))
        GPU(cok);
      else
        GPU(control(f, L, d, INK2_CTRL_Q, aq, f->msg, n, f->ctl) &&
            control(f, L, d, INK2_CTRL_K, ak, f->msg + nC, n, f->ctl + nC) &&
            control(f, L, d, INK2_CTRL_V, av, f->msg + 2 * nC, n, f->ctl + 2 * nC));
      aq = f->ctl, ak = f->ctl + nC, av = f->ctl + 2 * nC;
    }
    const bool apart = full || looped; /* q, k and v each their own norm's */
    const float *k_norm = full ? d->k_norm : d->attn_norm, *v_norm = full ? d->v_norm : d->attn_norm;
    bool ready, pok = false;
    const float *q_norm = full ? d->q_norm : d->attn_norm;
    const float *xk = apart ? f->kn : f->xn, *xv = apart ? f->vn : f->xn;
    float *dyn_k = full ? f->dyn : NULL, *dyn_v = full ? f->dyn + (uint64_t)n * KV * taps : NULL;
    if (proj_batch(f, L, d, full, apart, aq, ak, av, q_norm, k_norm, v_norm, n, dyn_k, dyn_v, &pok))
      GPU(pok && mat(f, L->sink, d->sink, f->xn, n, f->sink));
    else
    {
      GPU(rms_for(f, L->q, aq, q_norm, n, f->xn, &ready));
      /* the projections on xn in pairs through its one f16 (the norm's; a sliding layer's k and v too: xn theirs),
         then a full layer's k on kn and v on vn, each its norm's f16 - each matrix's bits its own launch's */
      GPU(mat2(f, L->q, d->q, f->q, L->gate, d->gate, f->gate, f->xn, n, ready) &&
          mat2(f, L->rel, d->rel, f->rel, NULL, NULL, NULL, f->xn, n, true));
      if (apart)
      {
        GPU(rms_for(f, L->k, ak, k_norm, n, f->kn, &ready) &&
            mat2(f, L->k, d->k, f->k, NULL, NULL, NULL, xk, n, ready));
        GPU(rms_for(f, L->v, av, v_norm, n, f->vn, &ready) &&
            mat2(f, L->v, d->v, f->v, NULL, NULL, NULL, xv, n, ready));
      }
      else
        GPU(mat2(f, L->k, d->k, f->k, L->v, d->v, f->v, f->xn, n, true));
      GPU(mat(f, L->sink, d->sink, f->xn, n, f->sink));
      if (full)
      {
        GPU(mat(f, L->k_conv_proj, d->k_conv_proj, xk, n, dyn_k));
        GPU(mat(f, L->v_conv_proj, d->v_conv_proj, xv, n, dyn_v));
      }
    }
    /* the keys: a full layer's every row to its position, a sliding layer's window (its last window - 1 and these) */
    /* a decode's split attention: its fold writes the out matrix's f16 too (an mm one's) */
    const bool att16 = n <= INK2_FA_TILE && L->out->dims[0] == (uint64_t)H * D &&
                       mm_takes(f, ink2_type(L->out), (uint32_t)L->out->dims[1], H * D);
    const Ink2Attn at = {H,
                         KV,
                         D,
                         R,
                         full ? (uint32_t)g->rel_extent : (uint32_t)g->window,
                         full ? 0u : (uint32_t)g->window,
                         1.0f / sqrtf((float)D),
                         att16 ? f->x16 : NULL,
                         att16 ? f->overflow : NULL};
    Ink2Tail *tk = &s->kw[slot], *tv = &s->vw[slot];
    uint32_t n_keys = pos + n;
    if (full)
      tk->held = tv->held = pos; /* the rows to its position, then these */
    else
    {
      GPU(ink2_tail_room(tk, n) && ink2_tail_room(tv, n));
      /* the window's keys from a position a multiple of INK2_FA_TILE: a query's tiles the same in every step */
      const uint32_t want = tk->held < (uint32_t)g->window - 1 ? tk->held : (uint32_t)g->window - 1;
      n_keys = want + (pos - want) % INK2_FA_TILE + n;
    }
    /* the convs, clips, v's norm and the caches' rows in one launch (ink2_attn_prep); a conv tail past its room
       takes its rows after (ink2_tail_put) */
    Ink2Tail *ck = &s->conv[slot][INK2_CONV_K], *cv = &s->conv[slot][INK2_CONV_V];
    const float *hk = ink2_tail_last(ck, taps - 1), *hv = ink2_tail_last(cv, taps - 1);
    float *keep_k = ink2_tail_claim(ck, n), *keep_v = ink2_tail_claim(cv, n);
    void *store_k = ink2_tail_claim(tk, n), *store_v = ink2_tail_claim(tv, n);
    if (!store_k || !store_v)
      return osmos_fail(err, "a layer's keys past its cache's room");
    const Ink2Prep prep = {n, H, KV, D, R, taps, f->q, f->rel, d->q_clip, d->rel_clip, g->q_clip, g->rel_clip,
                           g->k_clip, g->rms_eps, conv_res, full ? dyn_scale : conv_scale,
                           (Ink2Tau){pos, full ? g->log_alpha : 0.0f, (float)g->log_floor}, full, f->v8,
                           {f->k, d->k_conv, dyn_k, hk, f->kc, keep_k, store_k},
                           {f->v, d->v_conv, dyn_v, hv, f->vc, keep_v, store_v}};
    GPU(ink2_attn_prep(&prep, NULL));
    GPU((keep_k || ink2_tail_put(ck, f->k, n)) && (keep_v || ink2_tail_put(cv, f->v, n)));
    const void *keys = full ? tk->d : ink2_tail_last(tk, n_keys), *vals = full ? tv->d : ink2_tail_last(tv, n_keys);
    /* key 0's position: a full layer's 0, a window's its first's (a tile's start); a decode's units split */
    const uint32_t pos0 = full ? 0 : pos + n - n_keys;
    pf(f, PF_IN);
    /* a full layer's cache: keys Q8 records, values Q5 (ink2_fa85) or Q8 (ink2_fa8: Ink2Options' values_q8); a
       window's f16 */
    const bool attended =
        full && f->v8 ? ink2_fa8(&at, f->q, f->rel, f->sink, f->gate, n, keys, vals, n_keys, pos0, d->rel_proj, f->att,
                                 f->fa_part, NULL)
        : full ? ink2_fa85(&at, f->q, f->rel, f->sink, f->gate, n, keys, vals, n_keys, pos0, d->rel_proj, f->att,
                         f->fa_part, NULL)
             : ink2_fa16(&at, f->q, f->rel, f->sink, f->gate, n, keys, vals, n_keys, pos0, d->rel_proj, f->att,
                         f->fa_part, NULL);
    GPU(attended);
    pf(f, full ? PF_FULL : PF_ATTN);
    GPU(mat2(f, L->out, d->out, f->o, NULL, NULL, NULL, f->att, n, att16));
    Ink2Tail *co = &s->conv[slot][INK2_CONV_OUT], *cf = &s->conv[slot][INK2_CONV_FFN];
    /* r: the output's conv, plus h within a block (a block's first layer's h a block state of its own) */
    if (boundary)
    {
      if (nb >= maxb)
        return osmos_fail(err, "more blocks than the geometry's");
      GPU(hipMemcpy2DAsync(f->blocks + (uint64_t)nb * C, (uint64_t)maxb * C * 4, f->h, (uint64_t)C * 4,
                           (uint64_t)C * 4, n, hipMemcpyDeviceToDevice, NULL) == hipSuccess);
      nb++;
    }
    GPU(conv_put(co, f->o, n, C, taps, d->out_conv, 0, NULL, conv_res, conv_scale, boundary ? NULL : f->h, f->r));
    GPU(ink2_attn_res(d->ffn_res_q, 1, f->blocks, nb, maxb, f->r, n, C, g->rms_eps, res_scale, f->tmp, f->res_sc,
                      NULL));
    /* a looped layer's FFN control, its last step (x - tanh(forget) x + tanh(gain) m) in the router's norm's launch
       where the router is mm's (ops_rms_x16_ctrl: ink2_loop_ctrl then rms_for's bits) */
    const bool ctl16 = looped && n <= MM_MAX_TOKENS &&
                       mm_takes(f, ink2_type(L->router), (uint32_t)L->router->dims[1], (uint32_t)L->router->dims[0]);
    if (looped)
      GPU(control(f, L, d, INK2_CTRL_FFN, f->tmp, f->msg + 3 * nC, n, ctl16 ? NULL : f->tmp));
    if (ctl16)
    {
      const OpsCtrl ctl = {f->tmp, f->msg + 3 * nC, f->cgain, f->cforget, f->tmp};
      GPU(ops_rms_x16_ctrl(&ctl, C, d->ffn_norm, g->rms_eps, n, f->xn, C, f->x16, f->overflow, NULL));
      f->router16 = true;
    }
    else
      GPU(rms_for(f, L->router, f->tmp, d->ffn_norm, n, f->xn, &f->router16)); /* (the router's x16: moe's first) */
    pf(f, PF_MID);
    if (!moe(f, l, n, err))
      return false;
    pf(f, PF_MOE);
    /* the layer's output, r + the MoE's conv: the next layer's h - the experts' sums in the conv's launch, or
       (its tail past its room) summed first, the tail's rows */
    {
      const uint32_t K = (uint32_t)g->n_used;
      const float *hist = ink2_tail_last(cf, taps - 1);
      float *keep = ink2_tail_claim(cf, n);
      if (keep)
        GPU(ink2_moe_conv(f->ed, f->m_a, f->m_row, K, n, C, taps, d->ffn_out_conv, hist, conv_res, conv_scale, f->r,
                          f->h, keep, NULL));
      else
        GPU(ink2_moe_sum(f->moe, f->ed, f->m_a, f->m_row, K, n, C, NULL) &&
            conv_put(cf, f->moe, n, C, taps, d->ffn_out_conv, 0, NULL, conv_res, conv_scale, f->r, f->h));
    }
    if (f->tap)
      f->tap(f->tap_arg, slot, "l_out", f->h, n, C);
  }
  pf(f, PF_MID);
  /* the head: the rows from the prompt's end - 1 on */
  GPU(ink2_attn_res(f->out_res_q, 1, f->blocks, nb, maxb, f->h, n, C, g->rms_eps, res_scale, f->tmp, f->res_sc,
                    NULL));
  GPU(ops_rms(f->tmp, C, C, f->out_norm, g->rms_eps, n, f->xn, C, NULL));
  if (f->mtp) /* the final normed hidden: the last tap */
    GPU(ink2_mtp_tap(f, f->mtp->n_taps - 1, f->xn, n, row0));
  const uint32_t VR = (uint32_t)W->output->dims[1], V = f->vocab;
  uint32_t first = 0;
  if (s->prompt_end && s->prompt_end - 1 > pos)
    first = s->prompt_end - 1 - pos < n ? s->prompt_end - 1 - pos : n;
  if (f->n_slots + (n - first) > f->logit_rows)
    return osmos_fail(err, "a step's kept rows past the logits' (Ink2Options' logit_rows)");
  const uint32_t slot0 = f->n_slots; /* this run's kept rows the next slots, in order */
  for (uint32_t t = 0; t < n; t++)
    f->slot[row0 + t] = t >= first ? (int32_t)(slot0 + t - first) : -1;
  f->n_slots += n - first;
  for (uint32_t at = first; at < n; at += f->head_rows)
  {
    const uint32_t k = n - at < f->head_rows ? n - at : f->head_rows;
    GPU(mat(f, W->output, f->output, f->xn + (uint64_t)at * C, k, f->head));
    GPU(hipMemcpy2DAsync(f->logits + (uint64_t)(slot0 + at - first) * V, (uint64_t)V * 4, f->head, (uint64_t)VR * 4,
                         (uint64_t)V * 4, k, hipMemcpyDeviceToDevice, NULL) == hipSuccess);
  }
  if (first < n)
    GPU(ink2_softcap(f->logits + (uint64_t)slot0 * V, (uint64_t)(n - first) * V, g->logit_softcap, NULL));
#undef GPU
  pf(f, PF_HEAD);
  const int kind = n > MV_MAX_TOKENS;
  for (int b = 0; f->pf.on && b < PF_N; b++)
    f->pf.total[kind][b] += f->pf.step[b], f->pf.step[b] = 0;
  if (f->pf.on)
    f->pf.tokens[kind] += n, f->pf.steps[kind]++;
  s->pos += (int32_t)n;
  s->reach = s->reach + n < INK2_MAX_BACK ? s->reach + n : INK2_MAX_BACK;
  return true;
}

void ink2_forward_tap(Ink2Forward *f, ink2_tap_fn fn, void *arg)
{
  if (f)
    f->tap = fn, f->tap_arg = arg;
}

bool ink2_step(Ink2Forward *f, Ink2Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
               char err[OSMOS_ERR])
{
  if (!f || !runs || runs > INK2_MAX_RUNS)
    return osmos_fail(err, "a step's runs 1 .. INK2_MAX_RUNS");
  if (!overflow_settle(&f->ovf, err)) /* (a reset the last step owes: refused with its cause while it fails) */
    return false;
  uint64_t total = 0, kept = 0;
  for (uint32_t i = 0; i < runs; i++)
  {
    if (!seqs[i] || seqs[i]->f != f || seqs[i]->failed || !lens[i])
      return osmos_fail(err, "a run of no tokens, or a sequence of another forward or failed");
    for (uint32_t j = 0; j < i; j++)
      if (seqs[j] == seqs[i])
        return osmos_fail(err, "a sequence twice in a step");
    if ((uint64_t)seqs[i]->pos + lens[i] > f->max_ctx)
      return osmos_fail(err, "a sequence past max_ctx");
    total += lens[i];
    /* the rows its logits are kept of: from its prompt's end - 1 on (run()'s) - counted before any work, so a step
       past the logits' rows is refused with its sequences as they were (it was found after the layers ran: failed) */
    const uint32_t pos = (uint32_t)seqs[i]->pos, end = seqs[i]->prompt_end; /* (a position: never negative) */
    const uint32_t first = end && end - 1 > pos ? (end - 1 - pos < lens[i] ? end - 1 - pos : lens[i]) : 0;
    kept += lens[i] - first;
  }
  if (total > f->max_tokens)
    return osmos_fail(err, "a step's tokens past max_tokens");
  if (kept > f->logit_rows)
    return osmos_fail(err, "a step's kept rows past the logits' (Ink2Options' logit_rows)");
  f->n = 0;
  for (uint32_t r = 0; r < f->max_tokens; r++) /* (no row kept until the step keeps it) */
    f->slot[r] = -1;
  f->n_slots = 0;
  f->steps++;
  bool ok = true;
  for (uint32_t i = 0, at = 0; ok && i < runs; at += lens[i++])
    ok = run(f, seqs[i], tokens + at, lens[i], at, err);
  ok = ok && (hipDeviceSynchronize() == hipSuccess || osmos_fail(err, "a GPU call failed"));
  hipDeviceSynchronize();
  membrane_end(f->mem);
  int32_t overflow = 0;
  if (ok && hipMemcpy(&overflow, f->overflow, 4, hipMemcpyDeviceToHost) != hipSuccess)
    ok = osmos_fail(err, "a GPU call failed (the step's overflow flag)");
  if (ok && overflow)
    ok = osmos_fail(err, "an activation past f16's range: the step's results are not to be trusted");
  if (!ok) /* what the failed step may have set: not the next step's to answer for */
    { char rerr_[OSMOS_ERR] = ""; overflow_reset(&f->ovf, rerr_); } /* (failing: owed, the next step says so) */
  if (!ok)
  {
    for (uint32_t i = 0; i < runs; i++)
      ink2_seq_fail(seqs[i]);
    return false;
  }
  f->n = (uint32_t)total;
  return true;
}

const float *ink2_logits(const Ink2Forward *f) { return f ? f->logits : NULL; }

void ink2_kept_view(const Ink2Forward *f, KeptView *k)
{
  *k = (KeptView){0};
  if (f)
    *k = (KeptView){f->slot, f->n, f->logits, f->vocab, f->gid, f->amax, NULL, NULL, NULL, NULL};
}

const float *ink2_row_logits(const Ink2Forward *f, uint32_t row)
{
  KeptView k;
  ink2_kept_view(f, &k);
  return kept_row(&k, row);
}

bool ink2_kept(const Ink2Forward *f, uint32_t row) { return ink2_row_logits(f, row) != NULL; }

bool ink2_greedy(Ink2Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR])
{
  KeptView k;
  ink2_kept_view(f, &k);
  const char *why = kept_greedy(&k, row0, rows, ids);
  return why ? osmos_fail(err, "%s", why) : true;
}
