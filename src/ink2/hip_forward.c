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
#include "osmos/gpu/ink2_ops.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/mem/arena.h"

/* INK2_PROFILE=1: a step's sections timed, each after a synchronize (the bits the same), totals at destroy */
enum
{
  PF_IN,     /* the layer's inputs: attention residuals, norms, q / k / v / rel / gate / sink, convs, clips */
  PF_ATTN,   /* the attention */
  PF_MID,    /* the output's matrix and conv, the residual, the MoE's input; the MoE's conv and residual */
  PF_MOE,    /* the router, the experts (their waits apart: PF_WAIT) */
  PF_WAIT,   /* the experts' holds and waits (the membrane's reads), inside PF_MOE */
  PF_ROUTE,  /* the router's matrix and picks, their copy to the host, inside PF_MOE */
  PF_XK,     /* the experts' tables sent and their kernels, inside PF_MOE */
  PF_HEAD,   /* the embedding, the head */
  PF_N
};
static const char *const kPfName[PF_N] = {"inputs",         "attention",     "mid",           "moe",
                                          "  of it waits", "  of it router", "  of it kernels", "head"};
static double g_pf[PF_N], g_step[PF_N], g_pf_at; /* decode steps' totals (MV_MAX_TOKENS at most); this step's */
static uint64_t g_pf_tokens, g_pf_steps;
static bool pf_on(void)
{
  static int on = -1;
  if (on < 0)
    on = getenv("INK2_PROFILE") && atoi(getenv("INK2_PROFILE"));
  return on;
}
static double pf_now(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static void pf(int b)
{
  if (!pf_on())
    return;
  hipDeviceSynchronize();
  const double t = pf_now();
  if (b >= 0)
    g_step[b] += t - g_pf_at;
  g_pf_at = t;
}

#define T_F32 0u
#define T_F16 1u
#define T_BF16 30u

bool ink2_fail(char *err, const char *why)
{
  if (err)
    snprintf(err, INK2_FWD_ERR, "%s", why);
  return false;
}

bool ink2_dalloc(Ink2Forward *f, void **p, uint64_t bytes)
{
  if (hipMalloc(p, bytes + 16) != hipSuccess)
    return *p = NULL, false;
  f->dense_bytes += bytes;
  return true;
}

/* a tensor's bytes from its file */
bool ink2_read_tensor(const Gguf *g, const GgufTensor *t, void *out)
{
  const int fd = open(g->paths[t->file], O_RDONLY);
  if (fd < 0)
    return false;
  uint64_t at = 0;
  while (at < t->bytes)
  {
    const ssize_t n = pread(fd, (uint8_t *)out + at, t->bytes - at, (off_t)(t->offset + at));
    if (n <= 0)
      break;
    at += (uint64_t)n;
  }
  close(fd);
  return at == t->bytes;
}

static float half_f(uint16_t h)
{
  const uint32_t s = (uint32_t)(h >> 15) << 31, e = h >> 10 & 31, m = h & 1023;
  uint32_t u;
  if (!e)
  {
    float v = ldexpf((float)m, -24);
    return s ? -v : v;
  }
  u = e == 31 ? s | 0x7f800000u | m << 13 : s | (e + 112) << 23 | m << 13;
  float v;
  memcpy(&v, &u, 4);
  return v;
}

/* a small tensor's values as F32 on the host (F32, F16 or BF16 in the file); NULL where it is another type */
float *ink2_host_f32(const Gguf *g, const GgufTensor *t, uint64_t *n)
{
  const uint32_t es = t->type == T_F32 ? 4 : t->type == T_F16 || t->type == T_BF16 ? 2 : 0;
  if (!es)
    return NULL;
  *n = t->bytes / es;
  uint8_t *b = malloc(t->bytes);
  float *v = malloc(*n * 4 + 4);
  if (!b || !v || !ink2_read_tensor(g, t, b))
    return free(b), free(v), NULL;
  for (uint64_t i = 0; i < *n; i++)
  {
    if (es == 4)
      memcpy(&v[i], b + 4 * i, 4);
    else
    {
      const uint16_t h = (uint16_t)(b[2 * i] | b[2 * i + 1] << 8);
      uint32_t u = (uint32_t)h << 16;
      if (t->type == T_BF16)
        memcpy(&v[i], &u, 4);
      else
        v[i] = half_f(h);
    }
  }
  free(b);
  return v;
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
  float *v = ink2_host_f32(g, t, &n);
  const bool ok = v && ink2_up_host(f, v, n, out);
  free(v);
  return ok;
}

/* a matrix's bytes as stored, from g's file */
bool ink2_up_raw_of(Ink2Forward *f, const Gguf *g, const GgufTensor *t, void **out)
{
  *out = NULL;
  if (!t)
    return true;
  void *b = malloc(t->bytes);
  const bool ok = b && ink2_read_tensor(g, t, b) && ink2_dalloc(f, out, t->bytes) &&
                  hipMemcpy(*out, b, t->bytes, hipMemcpyHostToDevice) == hipSuccess;
  free(b);
  return ok;
}

static bool up_f32(Ink2Forward *f, const GgufTensor *t, float **out) { return ink2_up_f32_of(f, f->w->model, t, out); }
static bool up_raw(Ink2Forward *f, const GgufTensor *t, void **out) { return ink2_up_raw_of(f, f->w->model, t, out); }

/* a full layer's static conv as the kernel's [taps][ch], tap 0 first (llama.cpp's [ch][taps], tap 0 last) */
static bool up_conv_static(Ink2Forward *f, const GgufTensor *t, float **out)
{
  uint64_t n = 0;
  float *v = ink2_host_f32(f->w->model, t, &n);
  if (!v)
    return false;
  const uint32_t taps = (uint32_t)f->w->geo.conv;
  if (f->w->geo.llama_names)
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
      d->scale[k] = sc[k] ? ink2_host_f32(w->model, sc[k], &n) : malloc(E * sizeof(float));
      ok = d->scale[k] && n == E;
      for (uint32_t e = 0; ok && !sc[k] && e < E; e++)
        d->scale[k][e] = 1.0f;
    }
  }
  return ok;
}

static bool copy_host(void *dst, const void *src, uint64_t bytes, void *ctx)
{
  (void)ctx;
  memcpy(dst, src, bytes); /* both pools pinned host memory: a promotion is the CPU's copy */
  return true;
}

/* the experts as the membrane's units: room's worth warmed, an expert of every layer at a time (expert 0 of each
   layer, then 1, ...); room 0 every one */
static const uint64_t *g_rank; /* by_picks's counts (qsort takes no context) */
static int by_picks(const void *a, const void *b)
{
  const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  const uint64_t px = g_rank[x], py = g_rank[y];
  return px != py ? (px < py) - (px > py) : (x > y) - (x < y);
}

static bool experts(Ink2Forward *f, uint64_t room, const uint64_t *picks, char *err)
{
  const Ink2Weights *w = f->w;
  const Gguf *m = w->model;
  const uint32_t E = (uint32_t)w->geo.n_experts, L = (uint32_t)w->geo.n_layers;
  const size_t n = ink2_weights_experts(w, NULL, NULL, 0);
  if (!n)
    return ink2_fail(err, "inkling2: the experts' tensors not n_experts equal slices");
  PoolUnit *units = calloc(n, sizeof *units);
  uint32_t *warm = malloc(n * sizeof *warm);
  bool ok = units && warm && (f->fds = malloc((size_t)m->n_files * sizeof *f->fds));
  for (int i = 0; ok && i < m->n_files; i++)
    f->fds[i] = -1;
  for (int i = 0; ok && i < m->n_files; i++)
    ok = (f->fds[i] = open(m->paths[i], O_RDONLY | O_DIRECT)) >= 0;
  if (!ok)
    return free(units), free(warm), ink2_fail(err, "out of memory, or a model file not opened O_DIRECT");
  uint64_t all = 0, largest = 0;
  ok = ink2_weights_experts(w, f->fds, units, n) == n;
  for (size_t u = 0; ok && u < n; u++)
  {
    const uint64_t b = pool_unit_bytes(&units[u], ARENA_GRAIN);
    all += b, largest = b > largest ? b : largest;
  }
  if (!room || room > all)
    room = all;
  if (room < (uint64_t)E * largest) /* a decode's holds and a passing visit's in the stream: a layer's at most */
    room = (uint64_t)E * largest;
  /* the warm set, the most wanted first: by picks counted on calibration text (most first, then the lower unit), else
     an expert of every layer at a time (expert 0 of each layer, then 1, ...) - as many as the room holds */
  for (size_t u = 0; u < n; u++)
    warm[u] = picks ? (uint32_t)u : (uint32_t)((u % L) * E + u / L);
  if (picks)
  {
    g_rank = picks;
    qsort(warm, n, sizeof *warm, by_picks);
  }
  size_t n_warm = 0;
  for (uint64_t took = 0; ok && n_warm < n; n_warm++)
  {
    const uint64_t b = pool_unit_bytes(&units[warm[n_warm]], ARENA_GRAIN);
    if (took + b > room)
      break;
    took += b;
  }
  char perr[POOL_ERR] = "", merr[MEMBRANE_ERR] = "";
  const uint64_t stream_room = (uint64_t)E * largest;
  if (ok && !((f->slots = host_slots_create(room + 2 * largest, perr)) &&
              (f->stream = host_slots_create(stream_room + 2 * largest, perr))))
    ok = ink2_fail(err, perr[0] ? perr : "inkling2: the experts' memory refused");
  if (ok)
  {
    f->pinned_bytes = host_slots_bytes(f->slots) + host_slots_bytes(f->stream);
    const MembraneOptions mo = {units, L, E, room, stream_room, host_slots(f->slots), host_slots(f->stream),
                                ARENA_GRAIN, INK2_EXPERT_READERS, warm, n_warm, copy_host, NULL, 0};
    ok = (f->mem = membrane_create(&mo, merr)) && membrane_warm(f->mem, merr);
    if (!ok)
      ink2_fail(err, merr[0] ? merr : "inkling2: the experts' membrane refused");
  }
  free(units), free(warm);
  return ok;
}

/* moe's tables for P pairs at most: 3 pointer tables and 2 int32 tables of P + E entries, then 2 int32 and 3 float
   tables of P */
static uint64_t table_bytes(uint64_t P, uint64_t E) { return 3 * (P + E) * sizeof(void *) + 2 * (P + E) * 4 + 5 * P * 4; }

static bool buffers(Ink2Forward *f)
{
  const Ink2Geometry *g = &f->w->geo;
  const uint64_t T = f->max_tokens, C = (uint64_t)g->n_embd, H = (uint64_t)g->n_heads,
                 KV = (uint64_t)g->n_kv_heads, D = (uint64_t)g->head_dim, R = (uint64_t)g->d_rel,
                 taps = (uint64_t)g->conv, E = (uint64_t)g->n_experts, K = (uint64_t)g->n_used,
                 F = (uint64_t)g->ff_exp, VR = f->w->output->dims[1];
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
      {(void **)&f->att, T * H * D * 4}, {(void **)&f->dyn, T * KV * taps * 4}, {(void **)&f->o, T * C * 4},
      {(void **)&f->r, T * C * 4}, {(void **)&f->moe, T * C * 4}, {(void **)&f->logit, T * E * 4},
      {(void **)&f->tmp, T * C * 4}, {(void **)&f->pick, T * K * 4}, {(void **)&f->pw, T * K * 4},
      {(void **)&f->gpick, T * K * 4}, {(void **)&f->gw, T * K * 4},
      {(void **)&f->eg, T * K * F * 4}, {(void **)&f->eu, T * K * F * 4}, {(void **)&f->eh, T * K * F * 4},
      {(void **)&f->ed, T * K * C * 4}, {(void **)&f->xs, T * K * C * 4}, {&f->m_dev, table_bytes(T * K, E)},
      {(void **)&f->head, MV_MAX_TOKENS * VR * 4}, {(void **)&f->logits, T * f->vocab * 4},
      {(void **)&f->gid, T * 4}, {&f->amax, ops_argmax_scratch((uint32_t)T)},
      {(void **)&f->fa_part, ink2_fa_scratch((uint32_t)H, f->max_ctx)},
  };
  for (size_t i = 0; i < sizeof b / sizeof *b; i++)
    if (!ink2_dalloc(f, b[i].p, b[i].bytes))
      return false;
  f->hpick = malloc(T * K * 4), f->hw = malloc(T * K * 4), f->kept = calloc(T, 1), f->hgpick = malloc(T * K * 4);
  f->m_at = malloc((E + 1) * sizeof *f->m_at), f->m_by = malloc(T * K * sizeof *f->m_by);
  f->m_host = malloc(table_bytes(T * K, E));
  return f->hpick && f->hw && f->kept && f->hgpick && f->m_at && f->m_by && f->m_host;
}

Ink2Forward *ink2_forward_create(const Ink2Weights *w, const Ink2Options *o, char err[INK2_FWD_ERR])
{
  if (!w || !o || !o->max_ctx || !o->max_tokens)
    return ink2_fail(err, "inkling2: no weights, or max_ctx / max_tokens 0"), NULL;
  Ink2Forward *f = calloc(1, sizeof *f);
  if (!f)
    return ink2_fail(err, "out of memory"), NULL;
  f->w = w, f->max_ctx = o->max_ctx, f->max_tokens = o->max_tokens, f->vocab = (uint32_t)w->geo.n_vocab;
  f->maxb = (uint32_t)((w->geo.n_layers + w->geo.res_block - 1) / w->geo.res_block);
  if (!upload(f))
    return ink2_fail(err, "inkling2: a dense tensor not read or uploaded (GPU memory, or its file)"),
           ink2_forward_destroy(f), NULL;
  if (!buffers(f))
    return ink2_fail(err, "inkling2: GPU memory refused (the step's buffers)"), ink2_forward_destroy(f), NULL;
  f->mtp = o->mtp;
  if (f->mtp && !ink2_mtp_up(f, err))
    return ink2_forward_destroy(f), NULL;
  if (!(f->picks = calloc((size_t)w->geo.n_layers * w->geo.n_experts, sizeof *f->picks)))
    return ink2_fail(err, "out of memory"), ink2_forward_destroy(f), NULL;
  if (!experts(f, o->room, o->picks, err))
    return ink2_forward_destroy(f), NULL;
  return f;
}

void ink2_forward_destroy(Ink2Forward *f)
{
  if (!f)
    return;
  if (pf_on() && g_pf_steps)
  {
    double all = 0;
    for (int b = 0; b < PF_N; b++)
      all += b == PF_WAIT || b == PF_ROUTE || b == PF_XK ? 0 : g_pf[b];
    fprintf(stderr, "ink2 profile (decode steps): %llu steps, %llu tokens, %.3f s\n", (unsigned long long)g_pf_steps,
            (unsigned long long)g_pf_tokens, all);
    for (int b = 0; b < PF_N; b++)
      fprintf(stderr, "  %-13s %8.2f ms a step  %5.1f%%\n", kPfName[b], 1e3 * g_pf[b] / (double)g_pf_steps,
              100 * g_pf[b] / all);
    MembraneStats ms;
    membrane_stats(f->mem, &ms);
    fprintf(stderr, "  membrane (all steps): %llu picks not in memory, %llu of them guessed; %llu guesses read; waits "
                    "%llu unguessed %.1f s, %llu guessed %.1f s\n",
            (unsigned long long)ms.streamed, (unsigned long long)ms.guessed, (unsigned long long)ms.guess_read,
            (unsigned long long)ms.waits[0], ms.wait_ns[0] * 1e-9, (unsigned long long)ms.waits[1],
            ms.wait_ns[1] * 1e-9);
  }
  hipDeviceSynchronize();
  if (f->mtp)
    ink2_mtp_down(f);
  membrane_destroy(f->mem);
  host_slots_destroy(f->slots), host_slots_destroy(f->stream);
  for (int i = 0; f->fds && i < f->w->model->n_files; i++)
    if (f->fds[i] >= 0)
      close(f->fds[i]);
  free(f->fds);
  void *dev[] = {f->embed, f->output, f->embed_norm, f->out_res_q, f->out_norm, f->ids_d, f->h, f->blocks, f->a3,
                 f->xn, f->kn, f->vn, f->q, f->k, f->v, f->kc, f->vc, f->rel, f->gate, f->sink, f->att, f->dyn,
                 f->o, f->r, f->moe, f->logit, f->tmp, f->pick, f->pw, f->eg, f->eu, f->eh, f->ed, f->head,
                 f->logits, f->gid, f->amax, f->xs, f->m_dev, f->fa_part, f->gpick, f->gw};
  for (size_t i = 0; i < sizeof dev / sizeof *dev; i++)
    if (dev[i])
      hipFree(dev[i]);
  for (int l = 0; l < INK2_MAX_LAYERS; l++)
  {
    Ink2Dev *d = &f->L[l];
    void *p[] = {d->attn_res_q, d->ffn_res_q, d->attn_norm, d->q_norm, d->k_norm, d->v_norm, d->ffn_norm, d->q,
                 d->k, d->v, d->rel, d->out, d->gate, d->sink, d->router, d->k_conv_proj, d->v_conv_proj,
                 d->q_clip, d->rel_clip, d->rel_proj, d->router_bias, d->k_conv, d->v_conv, d->out_conv,
                 d->ffn_out_conv};
    for (size_t i = 0; i < sizeof p / sizeof *p; i++)
      if (p[i])
        hipFree(p[i]);
    for (int k = 0; k < 3; k++)
      free(d->scale[k]);
  }
  free(f->picks), free(f->hpick), free(f->hw), free(f->kept), free(f->hgpick), free(f->m_at), free(f->m_by), free(f->m_host);
  free(f);
}

uint64_t ink2_forward_bytes(const Ink2Forward *f) { return f ? f->dense_bytes + f->pinned_bytes : 0; }

const uint64_t *ink2_forward_picks(const Ink2Forward *f) { return f ? f->picks : NULL; }

/* y [n][rows] = W x, MV_MAX_TOKENS tokens a launch (a token's bits its own) */
bool ink2_mat(const GgufTensor *t, const void *w, const float *x, uint32_t n, float *y)
{
  const uint32_t rows = (uint32_t)t->dims[1], cols = (uint32_t)t->dims[0];
  for (uint32_t at = 0; at < n; at += MV_MAX_TOKENS)
  {
    const uint32_t k = n - at < MV_MAX_TOKENS ? n - at : MV_MAX_TOKENS;
    if (!mv_run(t->type, w, rows, cols, x + (uint64_t)at * cols, k, y + (uint64_t)at * rows, NULL))
      return false;
  }
  return true;
}

/* the routed experts of layer l for the run's n tokens: each distinct pick held (its read begun), then read once for
   the tokens that picked it (mv_group_tok: an entry an expert and up to MV_MAX_TOKENS of its tokens, a token's sums
   its own alone) - gate and up, their scales and the swiglu, down - and each token's summed in its slot order: the
   bits of ink2_run's one pick at a time */
static bool moe(Ink2Forward *f, int l, uint32_t n, char *err)
{
  const Ink2Geometry *g = &f->w->geo;
  const Ink2Layer *L = &f->w->layer[l];
  const Ink2Dev *d = &f->L[l];
  const uint32_t C = (uint32_t)g->n_embd, E = (uint32_t)g->n_experts, K = (uint32_t)g->n_used,
                 F = (uint32_t)g->ff_exp, P = n * K;
  const double r0 = pf_on() ? pf_now() : 0;
  /* a prompt's read ahead (its picks a layer's experts at least): the next layer's router on this layer's MoE input,
     the union of its picks read into the stream while this layer's experts and the next attention run (27.0 against
     31.5 s of prompt steps, 900 tokens). not a decode's: measured, the guess named 26% of its misses and read ~80
     experts a token never picked - the next router's input is the attention residual's over the blocks, far from
     this one's - their reads the misses' bandwidth (4.25 against 4.6 tok/s). which are read ahead changes no bit */
  const bool next = l + 1 < g->n_layers && P >= E;
  const Ink2Layer *L1 = next ? &f->w->layer[l + 1] : NULL;
  if (!ink2_mat(L->router, d->router, f->xn, n, f->logit) ||
      !ops_router_scored(f->logit, d->router_bias, E, K, g->route_scale, OPS_SCORE_SIGMOID, n, f->pick, f->pw,
                         NULL) ||
      (next && (!ink2_mat(L1->router, f->L[l + 1].router, f->xn, n, f->logit) ||
                !ops_router_scored(f->logit, f->L[l + 1].router_bias, E, K, g->route_scale, OPS_SCORE_SIGMOID, n,
                                   f->gpick, f->gw, NULL) ||
                hipMemcpy(f->hgpick, f->gpick, P * 4, hipMemcpyDeviceToHost) != hipSuccess)) ||
      hipMemcpy(f->hpick, f->pick, P * 4, hipMemcpyDeviceToHost) != hipSuccess ||
      hipMemcpy(f->hw, f->pw, P * 4, hipMemcpyDeviceToHost) != hipSuccess)
    return ink2_fail(err, "inkling2: a GPU call failed (the router)");
  if (pf_on())
    g_step[PF_ROUTE] += pf_now() - r0;
  /* the GPU is past the last layer's experts (the copy waited for it): the visit ends their holds */
  if (!membrane_visit(f->mem, (uint32_t)l, P >= E, err))
    return false;
  for (uint32_t m = 0; P < E && m < P; m++) /* a decode's picks counted: in memory or not (the stats') */
    membrane_score(f->mem, (uint32_t)l * E + (uint32_t)f->hpick[m]);
  for (uint32_t m = 0; next && m < P; m++)
    if (f->hgpick[m] >= 0 && (uint32_t)f->hgpick[m] < E)
    {
      const uint32_t unit = (uint32_t)(l + 1) * E + (uint32_t)f->hgpick[m];
      membrane_ahead(f->mem, unit);
    }
  /* the pairs by expert, then pair order (a counting sort): sorted row r is pair by[r] */
  uint32_t *at = f->m_at, *by = f->m_by;
  memset(at, 0, (E + 1) * sizeof *at);
  for (uint32_t m = 0; m < P; m++)
  {
    if (f->hpick[m] < 0 || (uint32_t)f->hpick[m] >= E)
      return ink2_fail(err, "inkling2: the router picked an expert the model does not have");
    at[f->hpick[m] + 1]++;
    f->picks[(uint32_t)l * E + (uint32_t)f->hpick[m]]++;
  }
  for (uint32_t e = 0; e < E; e++)
    at[e + 1] += at[e];
  for (uint32_t m = 0; m < P; m++)
    by[at[f->hpick[m]]++] = m;
  /* the tables, one region sent in one copy: the entries' gate, up and down pointers, counts and first rows; the
     sorted rows' tokens, scales; each pair's sorted row and weight */
  const uint32_t cap = P + E;
  uint8_t *h = f->m_host;
  const void **tg = (const void **)h, **tu = tg + cap, **td = tu + cap;
  int32_t *cnt = (int32_t *)(td + cap), *xrow = cnt + cap, *gidx = xrow + cap, *row = gidx + P;
  float *a = (float *)(row + P), *sg = a + P, *su = sg + P;
  for (uint32_t r = 0; r < P; r++) /* every distinct pick held first: the cold ones' reads under way together */
    if ((r == 0 || f->hpick[by[r]] != f->hpick[by[r - 1]]) &&
        !membrane_hold(f->mem, (uint32_t)l * E + (uint32_t)f->hpick[by[r]], err))
      return false;
  uint32_t ne = 0, maxc = 1;
  for (uint32_t r = 0; r < P;)
  {
    const uint32_t e = (uint32_t)f->hpick[by[r]], unit = (uint32_t)l * E + e;
    uint32_t end = r;
    while (end < P && (uint32_t)f->hpick[by[end]] == e)
      end++;
    const void *ex[POOL_PARTS];
    const double wt = pf_on() ? pf_now() : 0;
    if (!membrane_wait(f->mem, unit, ex, err))
      return false;
    if (pf_on())
      g_step[PF_WAIT] += pf_now() - wt;
    for (uint32_t q = r; q < end; q += MV_MAX_TOKENS, ne++)
    {
      const uint32_t c = end - q < MV_MAX_TOKENS ? end - q : MV_MAX_TOKENS;
      tg[ne] = ex[0], tu[ne] = ex[1], td[ne] = ex[2], cnt[ne] = (int32_t)c, xrow[ne] = (int32_t)q;
      maxc = c > maxc ? c : maxc;
    }
    for (uint32_t q = r; q < end; q++)
    {
      const uint32_t m = by[q];
      gidx[q] = (int32_t)(m / K), row[m] = (int32_t)q, sg[q] = d->scale[0][e], su[q] = d->scale[1][e];
      a[m] = f->hw[m] * d->scale[2][e];
    }
    r = end;
  }
  /* the pointer tables at cap apart: every table's place fixed by the step's most pairs */
  const uint64_t bytes = (uint64_t)((uint8_t *)(su + P) - h);
  uint8_t *dv = f->m_dev;
  const void **dg = (const void **)dv, **du = dg + cap, **dd = du + cap;
  const int32_t *dcnt = (const int32_t *)(dd + cap), *dxrow = dcnt + cap, *dgidx = dxrow + cap, *drow = dgidx + P;
  const float *da = (const float *)(drow + P), *dsg = da + P, *dsu = dsg + P;
  const double x0 = pf_on() ? pf_now() : 0;
  if (hipMemcpy(dv, h, bytes, hipMemcpyHostToDevice) != hipSuccess ||
      !ops_gather_rows(f->xs, f->xn, dgidx, P, C, NULL) ||
      !mv_group_tok(L->exp_gate->type, dg, dcnt, dxrow, ne, maxc, F, C, f->xs, C, f->eg, F, NULL) ||
      !mv_group_tok(L->exp_up->type, du, dcnt, dxrow, ne, maxc, F, C, f->xs, C, f->eu, F, NULL) ||
      !ink2_swiglu_rows(f->eg, f->eu, dsg, dsu, F, g->gate_softcap, g->up_softcap, P, f->eh, NULL) ||
      !mv_group_tok(L->exp_down->type, dd, dcnt, dxrow, ne, maxc, C, F, f->eh, F, f->ed, C, NULL) ||
      !ink2_moe_sum(f->moe, f->ed, da, drow, K, n, C, NULL))
    return ink2_fail(err, "inkling2: a GPU call failed (the experts)");
  if (pf_on())
    hipDeviceSynchronize(), g_step[PF_XK] += pf_now() - x0;
  return true;
}

/* a run: n tokens of s at its next positions through the model, their logits at rows row0 .. (those before the
   prompt's end - 1 not) */
static bool run(Ink2Forward *f, Ink2Seq *s, const int32_t *tok, uint32_t n, uint32_t row0, char *err)
{
  const Ink2Weights *W = f->w;
  const Ink2Geometry *g = &W->geo;
  const uint32_t C = (uint32_t)g->n_embd, H = (uint32_t)g->n_heads, KV = (uint32_t)g->n_kv_heads,
                 D = (uint32_t)g->head_dim, R = (uint32_t)g->d_rel, taps = (uint32_t)g->conv, maxb = f->maxb,
                 kd = KV * D, pos = (uint32_t)s->pos;
  const bool llama = g->llama_names;
  const float res_scale = 1.0f / sqrtf((float)C), conv_res = llama ? 0.0f : 1.0f,
              conv_scale = llama ? 1.0f : 1.0f / sqrtf((float)taps + 1),
              dyn_scale = llama ? 1.0f : 1.0f / sqrtf(2.0f * taps + 1);
#define GPU(x)                                                                                                         \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!(x))                                                                                                          \
      return ink2_fail(err, "inkling2: a GPU call failed: " #x);                                                       \
  } while (0)
  pf(-1);
  GPU(hipMemcpy(f->ids_d, tok, n * 4, hipMemcpyHostToDevice) == hipSuccess);
  GPU(ops_embed(W->embed->type, f->embed, C, f->ids_d, n, 1, f->h, NULL));
  GPU(ops_rms(f->h, C, C, f->embed_norm, g->rms_eps, n, f->h, C, NULL));
  for (int c = 0; c < INK2_CONVS; c++) /* every tail's room first: the step's reads and writes in one place */
    for (int l = 0; l < g->n_layers; l++)
      GPU(ink2_tail_room(&s->conv[l][c], n));
  uint32_t nb = 0;
  pf(PF_HEAD);
  if (f->mtp) /* the drafter's taps: this run's rows of the step's */
    s->tap_row0 = row0, s->tap_pos0 = (int32_t)pos;
  for (int l = 0; l < g->n_layers; l++)
  {
    const Ink2Layer *L = &W->layer[l];
    const Ink2Dev *d = &f->L[l];
    const bool full = !g->sliding[l], boundary = l % g->res_block == 0;
    for (int k = 0; f->mtp && k + 1 < f->mtp->n_taps; k++) /* h entering a target layer: the attention residual's
                                                              running partial */
      if (f->mtp->targets[k] == l)
        GPU(ink2_mtp_tap(f, k, f->h, n, row0));
    const float *aq = f->h, *ak = f->h, *av = f->h;
    if (d->attn_res_q)
    {
      GPU(ink2_attn_res(d->attn_res_q, full ? 3 : 1, f->blocks, nb, maxb, f->h, n, C, g->rms_eps, res_scale, f->a3,
                        NULL));
      aq = f->a3, ak = full ? f->a3 + (uint64_t)n * C : f->a3, av = full ? f->a3 + (uint64_t)2 * n * C : f->a3;
    }
    GPU(ops_rms(aq, C, C, full ? d->q_norm : d->attn_norm, g->rms_eps, n, f->xn, C, NULL));
    if (full)
    {
      GPU(ops_rms(ak, C, C, d->k_norm, g->rms_eps, n, f->kn, C, NULL));
      GPU(ops_rms(av, C, C, d->v_norm, g->rms_eps, n, f->vn, C, NULL));
    }
    const float *xk = full ? f->kn : f->xn, *xv = full ? f->vn : f->xn;
    GPU(ink2_mat(L->q, d->q, f->xn, n, f->q) && ink2_mat(L->k, d->k, xk, n, f->k) && ink2_mat(L->v, d->v, xv, n, f->v));
    GPU(ink2_mat(L->rel, d->rel, f->xn, n, f->rel) && ink2_mat(L->gate, d->gate, f->xn, n, f->gate) &&
        ink2_mat(L->sink, d->sink, f->xn, n, f->sink));
    Ink2Tail *ck = &s->conv[l][INK2_CONV_K], *cv = &s->conv[l][INK2_CONV_V];
    if (full)
    {
      GPU(ink2_mat(L->k_conv_proj, d->k_conv_proj, xk, n, f->dyn));
      GPU(ink2_conv_dyn(f->k, n, kd, taps, KV, d->k_conv, f->dyn, ink2_tail_last(ck, taps - 1), conv_res, dyn_scale,
                        f->kc, NULL));
      GPU(ink2_mat(L->v_conv_proj, d->v_conv_proj, xv, n, f->dyn));
      GPU(ink2_conv_dyn(f->v, n, kd, taps, KV, d->v_conv, f->dyn, ink2_tail_last(cv, taps - 1), conv_res, dyn_scale,
                        f->vc, NULL));
    }
    else
    {
      GPU(ink2_conv(f->k, n, kd, taps, d->k_conv, ink2_tail_last(ck, taps - 1), conv_res, conv_scale, f->kc, NULL));
      GPU(ink2_conv(f->v, n, kd, taps, d->v_conv, ink2_tail_last(cv, taps - 1), conv_res, conv_scale, f->vc, NULL));
    }
    GPU(ink2_tail_put(ck, f->k, n) && ink2_tail_put(cv, f->v, n));
    const Ink2Tau none = {0, 0, 0}, tau = {pos, full ? g->log_alpha : 0.0f, (float)g->log_floor};
    GPU(ink2_clip_rows(f->q, n, H, D, d->q_clip, g->q_clip, tau, NULL));
    GPU(ink2_clip_rows(f->rel, n, H, R, d->rel_clip, g->rel_clip, tau, NULL));
    GPU(ink2_clip_rows(f->kc, n, KV, D, NULL, g->k_clip, none, NULL));
    GPU(ops_rms(f->vc, D, D, NULL, g->rms_eps, n * KV, f->vc, D, NULL));
    /* the keys: a full layer's every row to its position, a sliding layer's window (its last window - 1 and these) */
    const Ink2Attn at = {H, KV, D, R, full ? (uint32_t)g->rel_extent : (uint32_t)g->window,
                         full ? 0u : (uint32_t)g->window, 1.0f / sqrtf((float)D)};
    Ink2Tail *tk = &s->kw[l], *tv = &s->vw[l];
    const void *keys, *vals;
    uint32_t n_keys;
    if (full)
    {
      tk->held = tv->held = pos; /* the rows to its position, then these */
      GPU(ink2_tail_put(tk, f->kc, n) && ink2_tail_put(tv, f->vc, n));
      keys = tk->d, vals = tv->d, n_keys = pos + n;
    }
    else
    {
      GPU(ink2_tail_room(tk, n) && ink2_tail_room(tv, n));
      /* the window's keys from a position a multiple of INK2_FA_TILE: a query's tiles the same in every step */
      const uint32_t want = tk->held < (uint32_t)g->window - 1 ? tk->held : (uint32_t)g->window - 1;
      const uint32_t back = want + (pos - want) % INK2_FA_TILE;
      GPU(ink2_tail_put(tk, f->kc, n) && ink2_tail_put(tv, f->vc, n));
      keys = ink2_tail_last(tk, back + n), vals = ink2_tail_last(tv, back + n), n_keys = back + n;
    }
    /* key 0's position: a full layer's 0, a window's its first's (a tile's start); a decode's units split */
    const uint32_t pos0 = full ? 0 : pos + n - n_keys;
    pf(PF_IN);
    GPU(ink2_fa16(&at, f->q, f->rel, f->sink, f->gate, n, keys, vals, n_keys, pos0, d->rel_proj, f->att, f->fa_part,
                  NULL));
    pf(PF_ATTN);
    GPU(ink2_mat(L->out, d->out, f->att, n, f->o));
    Ink2Tail *co = &s->conv[l][INK2_CONV_OUT], *cf = &s->conv[l][INK2_CONV_FFN];
    GPU(ink2_conv(f->o, n, C, taps, d->out_conv, ink2_tail_last(co, taps - 1), conv_res, conv_scale, f->r, NULL));
    GPU(ink2_tail_put(co, f->o, n));
    if (boundary)
    {
      if (nb >= maxb)
        return ink2_fail(err, "inkling2: more blocks than the geometry's");
      GPU(hipMemcpy2DAsync(f->blocks + (uint64_t)nb * C, (uint64_t)maxb * C * 4, f->h, (uint64_t)C * 4,
                           (uint64_t)C * 4, n, hipMemcpyDeviceToDevice, NULL) == hipSuccess);
      nb++;
    }
    else
      GPU(ink2_axpy(f->r, f->h, (uint64_t)n * C, 1.0f, NULL));
    GPU(ink2_attn_res(d->ffn_res_q, 1, f->blocks, nb, maxb, f->r, n, C, g->rms_eps, res_scale, f->tmp, NULL));
    GPU(ops_rms(f->tmp, C, C, d->ffn_norm, g->rms_eps, n, f->xn, C, NULL));
    pf(PF_MID);
    if (!moe(f, l, n, err))
      return false;
    pf(PF_MOE);
    GPU(ink2_conv(f->moe, n, C, taps, d->ffn_out_conv, ink2_tail_last(cf, taps - 1), conv_res, conv_scale, f->tmp,
                  NULL));
    GPU(ink2_tail_put(cf, f->moe, n));
    GPU(ink2_axpy(f->r, f->tmp, (uint64_t)n * C, 1.0f, NULL));
    GPU(hipMemcpyAsync(f->h, f->r, (uint64_t)n * C * 4, hipMemcpyDeviceToDevice, NULL) == hipSuccess);
  }
  pf(PF_MID);
  /* the head: the rows from the prompt's end - 1 on */
  GPU(ink2_attn_res(f->out_res_q, 1, f->blocks, nb, maxb, f->h, n, C, g->rms_eps, res_scale, f->tmp, NULL));
  GPU(ops_rms(f->tmp, C, C, f->out_norm, g->rms_eps, n, f->xn, C, NULL));
  if (f->mtp) /* the final normed hidden: the last tap */
    GPU(ink2_mtp_tap(f, f->mtp->n_taps - 1, f->xn, n, row0));
  const uint32_t VR = (uint32_t)W->output->dims[1], V = f->vocab;
  uint32_t first = 0;
  if (s->prompt_end && s->prompt_end - 1 > pos)
    first = s->prompt_end - 1 - pos < n ? s->prompt_end - 1 - pos : n;
  for (uint32_t t = 0; t < n; t++)
    f->kept[row0 + t] = t >= first;
  for (uint32_t at = first; at < n; at += MV_MAX_TOKENS)
  {
    const uint32_t k = n - at < MV_MAX_TOKENS ? n - at : MV_MAX_TOKENS;
    GPU(mv_run(W->output->type, f->output, VR, C, f->xn + (uint64_t)at * C, k, f->head, NULL));
    GPU(hipMemcpy2DAsync(f->logits + (uint64_t)(row0 + at) * V, (uint64_t)V * 4, f->head, (uint64_t)VR * 4,
                         (uint64_t)V * 4, k, hipMemcpyDeviceToDevice, NULL) == hipSuccess);
  }
  if (first < n)
    GPU(ink2_softcap(f->logits + (uint64_t)(row0 + first) * V, (uint64_t)(n - first) * V, g->logit_softcap, NULL));
#undef GPU
  pf(PF_HEAD);
  for (int b = 0; pf_on() && b < PF_N; b++) /* a decode's step kept, a prompt's not */
    g_pf[b] += n <= MV_MAX_TOKENS ? g_step[b] : 0, g_step[b] = 0;
  if (pf_on() && n <= MV_MAX_TOKENS)
    g_pf_tokens += n, g_pf_steps++;
  s->pos += (int32_t)n;
  s->reach = s->reach + n < INK2_MAX_BACK ? s->reach + n : INK2_MAX_BACK;
  return true;
}

bool ink2_step(Ink2Forward *f, Ink2Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
               char err[INK2_FWD_ERR])
{
  if (!f || !runs || runs > INK2_MAX_RUNS)
    return ink2_fail(err, "inkling2: a step's runs 1 .. INK2_MAX_RUNS");
  uint64_t total = 0;
  for (uint32_t i = 0; i < runs; i++)
  {
    if (!seqs[i] || seqs[i]->f != f || seqs[i]->failed || !lens[i])
      return ink2_fail(err, "inkling2: a run of no tokens, or a sequence of another forward or failed");
    for (uint32_t j = 0; j < i; j++)
      if (seqs[j] == seqs[i])
        return ink2_fail(err, "inkling2: a sequence twice in a step");
    if ((uint64_t)seqs[i]->pos + lens[i] > f->max_ctx)
      return ink2_fail(err, "inkling2: a sequence past max_ctx");
    total += lens[i];
  }
  if (total > f->max_tokens)
    return ink2_fail(err, "inkling2: a step's tokens past max_tokens");
  f->n = 0;
  memset(f->kept, 0, f->max_tokens);
  bool ok = true;
  for (uint32_t i = 0, at = 0; ok && i < runs; at += lens[i++])
    ok = run(f, seqs[i], tokens + at, lens[i], at, err);
  ok = ok && (hipDeviceSynchronize() == hipSuccess || ink2_fail(err, "inkling2: a GPU call failed"));
  hipDeviceSynchronize();
  membrane_end(f->mem);
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

bool ink2_kept(const Ink2Forward *f, uint32_t row) { return f && row < f->n && f->kept[row]; }

bool ink2_greedy(Ink2Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[INK2_FWD_ERR])
{
  for (uint32_t r = 0; r < rows; r++)
    if (!ink2_kept(f, row0 + r))
      return ink2_fail(err, "inkling2: a row the last step kept no logits of");
  if (!rows || !ops_argmax(f->logits + (uint64_t)row0 * f->vocab, f->vocab, rows, f->vocab, f->gid, f->amax, NULL) ||
      hipMemcpy(ids, f->gid, rows * 4, hipMemcpyDeviceToHost) != hipSuccess)
    return ink2_fail(err, "inkling2: a GPU call failed (greedy)");
  return true;
}
