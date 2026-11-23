/* Inkling-2-Small's forward, its making (osmos/ink2/forward.h): the dense weights resident, the routed experts
   the membrane's (their pool and order), the tables and buffers a step reads, freed in reverse; the profile's timer */
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

static const char *const kPfName[PF_N] = {"inputs",         "attn sliding",  "attn full",     "mid",           "moe",
                                          "  of it waits", "  of it router", "  of it kernels", "head"};
_Static_assert(PF_N == INK2_PF_N, "the profile's sections: Ink2Forward's pf");

double ink2_pf_now(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
void ink2_pf(Ink2Forward *f, int b)
{
  if (!f->pf.on)
    return;
  hipDeviceSynchronize();
  const double t = ink2_pf_now();
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

/* mm's: its table's types, and BF16 where the model's dense weights are BF16 (f->bf16_mm: Mini's, as released) - mm's
   by name, not its table's. a model whose BF16 is a side tensor (Inkling-2-Small's sliding layers' sink projections)
   keeps its matvec, its bits 0.11's; the other families' BF16 matrices their matvec's as before. by the model, the
   type and the shape alone: a decode's and a prompt's the same */
bool ink2_mm_takes(const Ink2Forward *f, uint32_t type, uint32_t rows, uint32_t cols)
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
  f->head_rows = ink2_mm_takes(f, head, (uint32_t)VR, (uint32_t)C) ? MM_TILE * mm_col_tiles(head, MM_MAX_TOKENS) : MV_MAX_TOKENS;
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
      {(void **)&f->eg, T * K * F * 4}, {(void **)&f->eu, T * K * F * 4}, {(void **)&f->ed, ed_bytes}, {&f->m_dev, ink2_table_bytes(T * K, E)}, {&f->x16, X16 * CH * 2},
      {&f->xe16, ink2_table_slots(T * K, E) * C * 2}, {(void **)&f->overflow, 4},
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
      !gpu_bufs_host(&f->bufs, f->ledger, &f->m_host, ink2_table_bytes(T * K, E)))
    return false;
  f->slot = malloc(T * sizeof *f->slot);
  f->m_at = malloc((E + 1) * sizeof *f->m_at), f->m_by = malloc(T * K * sizeof *f->m_by);
  f->m_key = malloc(ink2_table_entries(T * K, E) * sizeof *f->m_key);
  f->m_ent = malloc(ink2_table_entries(T * K, E) * sizeof *f->m_ent), f->m_run = malloc(4 * E * sizeof *f->m_run);
  f->m_ex = malloc(E * sizeof *f->m_ex);
  f->m_build = malloc(ink2_table_bytes(T * K, E)); /* the tables built in cached memory, then one copy into the pinned */
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
    if (!ink2_mm_takes(f, L->exp_gate->type, F, C) || !ink2_mm_takes(f, L->exp_up->type, F, C) || !ink2_mm_takes(f, L->exp_down->type, C, F))
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
