#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>

#include "forward_impl.h"
#include "osmos/mem/hip_slots.h"
#include "osmos/gpu/kvfmt.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/model/qtype.h"
#include "osmos/model/rope.h"

#define FWD_REPACK_STAGE (64u << 20) /* the Q5_K repack's chunk of a matrix, copied out (hip_forward.c) */
#define FWD_FETCH_THREADS 12 /* DESIGN.md, the fetcher: 4 at the SSD's ceiling; 12 a decode's miss sooner (UD, step 5) */
#define FWD_ROW_THREADS 64  /* DESIGN.md, Engram's reads: 64 threads 832,000 random rows/s, 128 no more */
#define TENSORS_A_LAYER (sizeof(LayerWeights) / sizeof(const GgufTensor *))
_Static_assert(sizeof(LayerWeights) % sizeof(const GgufTensor *) == 0, "LayerWeights is its tensors and only them");
_Static_assert(sizeof(Scratch) % sizeof(void *) == 0, "Scratch is its device buffers and only them");

bool fwd_fail(char *err, const char *why)
{
  snprintf(err, FORWARD_ERR, "forward: %s", why);
  return false;
}

bool fwd_put(void *dev, const void *host, uint64_t bytes)
{
  return hipMemcpy(dev, host, bytes, hipMemcpyHostToDevice) == hipSuccess;
}

bool fwd_put_async(void *dev, const void *pinned, uint64_t bytes)
{
  return hipMemcpyAsync(dev, pinned, bytes, hipMemcpyHostToDevice, NULL) == hipSuccess;
}

bool fwd_copy(void *dst, const void *src, uint64_t bytes)
{
  return hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, NULL) == hipSuccess;
}


void fwd_probe(const Forward *f, const char *step, int layer, int index, void *dev, uint64_t n, bool ints)
{
  if (f->probe && hipDeviceSynchronize() == hipSuccess)
    f->probe(step, layer, index, dev, n, ints, f->probe_arg);
}

static uint64_t unit_bytes(const PoolUnit *u)
{
  uint64_t b = 0;
  for (int k = 0; k < u->n_parts; k++)
    b += (u->part[k].bytes + POOL_ALIGN - 1) / POOL_ALIGN * POOL_ALIGN;
  return b;
}

/* the resident set: every tensor the layers use but the routed experts, then experts while the budget lasts - expert
   0 of each layer, then 1, ..: an equal share of each layer, so every layer's streaming is alike. f->rix: an
   expert unit's place in the set, -1 for one the pool streams. the units past n_res are the experts chosen */
typedef struct Layout
{
  uint64_t total, used, cap; /* the arenas' bytes; the last one's fill and size */
} Layout;

/* the GPU memory with a unit of `b` bytes added, laid out as resident_create lays it (a unit never spans arenas) */
static uint64_t with(Layout a, uint64_t b)
{
  return a.used + b <= a.cap ? a.total : a.total + (b > RESIDENT_ARENA ? b : RESIDENT_ARENA);
}

static void add(Layout *a, uint64_t b)
{
  if (a->used + b > a->cap)
    a->cap = b > RESIDENT_ARENA ? b : RESIDENT_ARENA, a->total += a->cap, a->used = 0;
  a->used += b;
}

static const uint64_t *g_rank_picks; /* rank_cmp's: a layer's picks (qsort takes no context) */
static int rank_cmp(const void *a, const void *b)
{
  const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  const uint64_t px = g_rank_picks[x], py = g_rank_picks[y];
  return px != py ? (px < py) - (px > py) : (x > y) - (x < y); /* more picks first, then the lower id */
}

/* the resident experts: a rank at a time, each layer's expert of that rank while the budget holds it - by the
   picks counted on calibration text (forward_create_ranked's, most first), else by id. so each layer's share is
   about equal and its most-picked first */
static size_t choose(Forward *f, const PoolUnit *ex, PoolUnit *res, size_t n_res, uint64_t budget,
                     const uint64_t *picks, char *err)
{
  const uint32_t E = (uint32_t)f->g->n_experts;
  Layout a = {0, 0, 0};
  for (size_t i = 0; i < n_res; i++)
    add(&a, unit_bytes(&res[i]));
  if (a.total > budget)
    return fwd_fail(err, "the layers' weights are more than the budget"), 0;
  uint32_t *order = malloc((size_t)f->n_layers * E * sizeof *order); /* [layer][rank]: the expert */
  if (!order)
    return fwd_fail(err, "out of memory"), 0;
  for (int l = 0; l < f->n_layers; l++)
  {
    uint32_t *o = order + (size_t)l * E;
    for (uint32_t e = 0; e < E; e++)
      o[e] = e;
    if (picks)
      g_rank_picks = picks + (size_t)l * E, qsort(o, E, sizeof *o, rank_cmp);
  }
  size_t n = n_res;
  for (uint32_t i = 0; i < f->n_expert_units; i++)
    f->rix[i] = -1;
  for (uint32_t r = 0; r < E; r++)
    for (int l = 0; l < f->n_layers; l++)
    {
      const uint32_t i = (uint32_t)l * E + order[(size_t)l * E + r];
      if (with(a, unit_bytes(&ex[i])) <= budget)
        f->rix[i] = (int32_t)n, res[n++] = ex[i], add(&a, unit_bytes(&ex[i]));
    }
  free(order);
  return n;
}

/* the pool's room: two layers' streamed experts (the one the GPU is in, the next one's reads) */
static uint64_t stream_budget(const Forward *f, const PoolUnit *ex)
{
  const uint32_t E = (uint32_t)f->g->n_experts;
  uint64_t most = 0, unit = 0;
  for (int l = 0; l < f->n_layers; l++)
  {
    uint64_t layer = 0;
    for (uint32_t e = 0; e < E; e++)
    {
      const uint64_t b = unit_bytes(&ex[l * E + e]);
      unit = b > unit ? b : unit;
      layer += f->rix[l * E + e] < 0 ? b : 0;
    }
    most = layer > most ? layer : most;
  }
  return 2 * most > unit ? 2 * most : unit;
}

/* a resident Q5_K matrix the matmuls read (all but the embedding's rows, which ops_embed reads a token at a time):
   repacked at load (quant_repack_q5k), its unit the file's bytes and room after them for the larger layout */
static bool repacks(const Weights *w, const GgufTensor *t, uint32_t *rows, uint32_t *cols)
{
  if (t->type != QUANT_Q5_K || t == w->embed || t->n_dims < 2)
    return false;
  uint64_t r = 1;
  for (uint32_t d = 1; d < t->n_dims; d++)
    r *= t->dims[d];
  *rows = (uint32_t)r, *cols = (uint32_t)t->dims[0];
  return r % MM_TILE == 0 && t->dims[0] % QUANT_Q5_K_VALUES == 0 && r <= UINT32_MAX;
}

/* each repacked matrix in place: its tiles from the last back, FWD_REPACK_STAGE bytes of them at a time copied out
   and repacked into their places - a tile's repacked bytes start at or past its file bytes' start (180 >= 176 a
   block), so a chunk's writes never reach the tiles before it, which are still to be read */
static bool repack_all(Forward *f, const GgufTensor *const *t_res, size_t n_t)
{
  void *stage = NULL;
  bool ok = hipMalloc(&stage, FWD_REPACK_STAGE) == hipSuccess;
  for (size_t i = 0; ok && i < n_t; i++)
  {
    uint32_t rows, cols;
    if (!repacks(f->w, t_res[i], &rows, &cols))
      continue;
    uint8_t *dev = (uint8_t *)fwd_dev(f, t_res[i]);
    const uint64_t in = (uint64_t)MM_TILE * (cols / QUANT_Q5_K_VALUES) * QUANT_Q5_K_BYTES;
    const uint64_t out = QUANT_Q5_KR_BYTES(MM_TILE, cols), tiles = rows / MM_TILE;
    const uint64_t per = FWD_REPACK_STAGE / in ? FWD_REPACK_STAGE / in : 1;
    ok = per * in <= FWD_REPACK_STAGE;
    for (uint64_t hi = tiles, lo; ok && hi > 0; hi = lo)
    {
      lo = hi > per ? hi - per : 0;
      ok = hipMemcpy(stage, dev + lo * in, (hi - lo) * in, hipMemcpyDeviceToDevice) == hipSuccess &&
           quant_repack_q5k(stage, (uint32_t)((hi - lo) * MM_TILE), cols, dev + lo * out, NULL) &&
           hipDeviceSynchronize() == hipSuccess;
    }
    f->ttype[weights_tensor_index(f->w, t_res[i])] = QUANT_Q5_KR;
  }
  hipFree(stage);
  return ok;
}

/* the units: each layer's experts (the pool's to stream, but the resident ones), then every other tensor the layers
   use - all of them resident (resident.h). Engram's tables are read a row at a time and are not units */
static bool units(Forward *f, uint64_t budget, const uint64_t *picks, char *err)
{
  const Weights *w = f->w;
  const uint32_t E = (uint32_t)f->g->n_experts;
  const size_t n_res_max = 3 + (size_t)f->n_layers * TENSORS_A_LAYER;
  f->n_expert_units = (uint32_t)f->n_layers * E;
  PoolUnit *all = calloc((size_t)w->geo.n_layers * E, sizeof *all);
  PoolUnit *res = calloc(n_res_max + f->n_expert_units, sizeof *res);
  const GgufTensor **t_res = calloc(n_res_max, sizeof *t_res);
  f->tdev = calloc(weights_tensors(w), sizeof *f->tdev);
  f->ttype = calloc(weights_tensors(w), sizeof *f->ttype);
  f->rix = calloc(f->n_expert_units, sizeof *f->rix);
  f->is_held = calloc(f->n_expert_units, sizeof *f->is_held);
  f->held_layer = -1;
  bool ok = all && res && t_res && f->tdev && f->ttype && f->rix && f->is_held &&
            weights_experts(w, f->fds, all, (size_t)w->geo.n_layers * E) == (size_t)w->geo.n_layers * E;
  if (!ok)
  {
    free(all), free(res), free(t_res);
    return fwd_fail(err, "out of memory, or the experts' units");
  }
  size_t n_t = 0;
  t_res[n_t++] = w->embed, t_res[n_t++] = w->output, t_res[n_t++] = w->output_norm;
  for (int l = 0; l < f->n_layers; l++)
  {
    const LayerWeights *L = &w->layer[l];
    const GgufTensor *const *t = (const GgufTensor *const *)L;
    for (size_t i = 0; i < TENSORS_A_LAYER; i++)
      if (t[i] && t[i] != L->eg_embd && t[i] != L->exp_gate && t[i] != L->exp_up && t[i] != L->exp_down &&
          t[i] != L->eg_q && t[i] != L->eg_k && t[i] != L->eg_gate) /* the gate: its own buffer (gates) */
        t_res[n_t++] = t[i];
  }
  for (size_t i = 0; i < n_t; i++)
  {
    uint32_t rows, cols;
    res[i] = (PoolUnit){.part = {{f->fds[t_res[i]->file], t_res[i]->offset, t_res[i]->bytes}}, .n_parts = 1};
    if (repacks(w, t_res[i], &rows, &cols)) /* the room the repacked layout needs past the file's bytes */
      res[i].part[1] = (PoolRange){-1, 0, QUANT_Q5_KR_BYTES(rows, cols) - t_res[i]->bytes}, res[i].n_parts = 2;
  }
  const size_t n_res = choose(f, all, res, n_t, budget, picks, err);
  char perr[POOL_ERR] = "";
  if (!n_res)
    ok = false;
  else if (!(f->res = resident_create(res, n_res, hip_slots(), FWD_FETCH_THREADS, perr)))
    ok = fwd_fail(err, perr);
  else if (!(f->fq = fetchq_create(FWD_FETCH_THREADS, fwd_on_read, &f->in)))
    ok = fwd_fail(err, "the fetcher's workers could not start");
  else if (!(f->pool_room = stream_budget(f, all)) ||
           !(f->pool = pool_create(all, f->n_expert_units, f->pool_room, hip_slots(), fetchq_fetcher(f->fq),
                                   NULL, NULL, perr)))
    ok = fwd_fail(err, perr);
  for (size_t i = 0; ok && i < n_t; i++)
  {
    const void *dev[POOL_PARTS];
    ok = resident_dev(f->res, i, dev);
    f->tdev[weights_tensor_index(w, t_res[i])] = dev[0];
  }
  ok = ok && (repack_all(f, t_res, n_t) || fwd_fail(err, "the Q5_K matrices' repack failed"));
  free(res), free(t_res);
  f->units = all; /* the swaps' reads */
  return ok;
}

/* each layer's wo_a, a matrix a group (the GGUF's [in, out, groups]: group k's rows k * rows ..), and where each
   group reads and writes: mm_group's tables, fixed for the forward's life */
static bool oa_tables(Forward *f, char *err)
{
  const Geometry *g = f->g;
  const uint32_t G = (uint32_t)g->o_groups, oin = (uint32_t)(g->n_heads * g->head_dim / G);
  if (G > FWD_MAX_GROUPS)
    return fwd_fail(err, "more wo_a groups than this build runs");
  const void **tab = calloc((size_t)f->n_layers * G, sizeof *tab);
  uint32_t xo[FWD_MAX_GROUPS], yo[FWD_MAX_GROUPS];
  for (uint32_t k = 0; k < G; k++)
    xo[k] = k * oin * MM_TILE, yo[k] = k * (uint32_t)g->o_lora; /* group k's first column, in the tiled x16 */
  bool ok = tab && fwd_alloc(&f->oa_tab, (uint64_t)f->n_layers * G * sizeof *tab) &&
            fwd_alloc(&f->oa_xoff, G * 4) && fwd_alloc(&f->oa_yoff, G * 4) && fwd_put(f->oa_xoff, xo, G * 4) &&
            fwd_put(f->oa_yoff, yo, G * 4);
  for (int l = 0; ok && l < f->n_layers; l++)
    for (uint32_t k = 0; k < G; k++)
      tab[(size_t)l * G + k] = fwd_dev(f, f->w->layer[l].out_a) + k * (fwd_bytes(f, f->w->layer[l].out_a) / G);
  ok = ok && fwd_put(f->oa_tab, tab, (uint64_t)f->n_layers * G * sizeof *tab);
  free(tab);
  return ok || fwd_fail(err, "a GPU allocation refused (the wo_a tables)");
}

bool fwd_alloc(void *p, uint64_t bytes)
{
  void **out = p;
  return hipMalloc(out, bytes ? bytes : 4) == hipSuccess && hipMemset(*out, 0, bytes ? bytes : 4) == hipSuccess;
}

static bool subs(Forward *f);

static size_t engram_layers(const Weights *w, int n_layers)
{
  size_t n = 0;
  for (int l = 0; l < n_layers; l++)
    n += w->geo.engram[l] != 0;
  return n;
}

/* the file the Engram's rows are read from (its own descriptor: rowread's): ds1017's second file, or the model's own
   shard that holds per_layer_token_embd (every layer's slice in it) */
static int engram_fd(const Forward *f, const Weights *w)
{
  if (!w->geo.engram_own)
    return open(w->engram->paths[0], O_RDONLY);
  for (int l = 0; l < w->geo.n_layers; l++)
    if (w->geo.engram[l])
      return dup(f->fds[w->layer[l].eg_embd->file]);
  return -1;
}

/* a tensor's bytes from the model's files into host memory */
static bool read_all(const Forward *f, const GgufTensor *t, void *out)
{
  for (uint64_t got = 0; got < t->bytes;)
  {
    const ssize_t n = pread(f->fds[t->file], (char *)out + got, t->bytes - got, (off_t)(t->offset + got));
    if (n <= 0)
      return false;
    got += (uint64_t)n;
  }
  return true;
}

/* each Engram layer's gate as ops_engram_gate reads it: F32 [hc][dim], q k. a llama.cpp GGUF's folded (engram_gate:
   read as it is); ds1017's q and k (BF16) multiplied here - a product of two BF16 values exact in F32 (8-bit
   significands, 16 bits of product): the kernel's own product before, the same bits */
static bool gates(Forward *f, const Weights *w, char *err)
{
  const uint64_t n = (uint64_t)w->geo.hc * (uint64_t)w->geo.n_embd;
  float *g = malloc(n * sizeof *g);
  uint16_t *q = malloc(n * sizeof *q), *k = malloc(n * sizeof *k);
  bool ok = g && q && k;
  for (int l = 0; ok && l < w->geo.n_layers; l++)
  {
    if (!w->geo.engram[l])
      continue;
    const LayerWeights *L = &w->layer[l];
    if (w->geo.engram_own)
      ok = read_all(f, L->eg_gate, g);
    else if ((ok = read_all(f, L->eg_q, q) && read_all(f, L->eg_k, k)))
      for (uint64_t i = 0; i < n; i++)
      {
        const uint32_t a = (uint32_t)q[i] << 16, b = (uint32_t)k[i] << 16;
        float x, y;
        memcpy(&x, &a, 4), memcpy(&y, &b, 4);
        g[i] = x * y;
      }
    ok = ok && fwd_alloc(&f->eg_gate[l], n * sizeof *g) && fwd_put(f->eg_gate[l], g, n * sizeof *g);
  }
  free(g), free(q), free(k);
  return ok || fwd_fail(err, "the Engram's gates could not be read");
}

/* an F32 router (llama.cpp's GGUFs) rounded to BF16 (nearest, ties to even) for the guess alone: the guess picks the
   experts read ahead, which change no bit - half its bytes a layer (unsloth's UD: 7.9 -> 3.9 MB). the routing itself
   reads the F32 */
static bool guess_routers(Forward *f, const Weights *w, char *err)
{
  bool ok = true;
  for (int l = 0; ok && !f->no_guess && l < f->n_layers; l++) /* no guess: no copy */
  {
    const GgufTensor *t = w->layer[l].router;
    if (t->type != QTYPE_F32)
      continue;
    const uint64_t n = t->bytes / sizeof(float);
    float *x = malloc(t->bytes);
    uint16_t *h = malloc(n * sizeof *h);
    ok = x && h && read_all(f, t, x);
    for (uint64_t i = 0; ok && i < n; i++)
    {
      uint32_t u;
      memcpy(&u, &x[i], 4);
      h[i] = (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
    }
    ok = ok && fwd_alloc(&f->guess_router[l], n * sizeof *h) && fwd_put(f->guess_router[l], h, n * sizeof *h);
    free(x), free(h);
  }
  return ok || fwd_fail(err, "a router's copy for the guess could not be made");
}

static bool scratch(Forward *f)
{
  const Geometry *g = f->g;
  Scratch *b = &f->b;
  const uint64_t D = g->n_embd, hc = g->hc, F = sizeof(float), HQ = (uint64_t)g->n_heads * g->head_dim;
  const uint64_t row = gguf_bytes(f->w->layer[0].eg_embd ? f->w->layer[0].eg_embd->type : 0, g->engram_head_dim);
  const uint64_t cols = g->engram_cols, K = (uint64_t)g->n_used, FF = (uint64_t)g->expert_ff, hd = g->head_dim;
  const uint64_t ih = g->ix_head_dim, nh = g->ix_heads, egv = cols * (uint64_t)g->engram_head_dim;
  const uint64_t B = f->step, Fs = OPS_MAX_SELECT, NS = f->n_subs;
  const uint64_t S = f->max_tokens, P = S * K; /* the experts': a layer's sub-batches at once (forward's max_tokens) */
  uint64_t widest = (uint64_t)g->o_groups * g->o_lora > D ? (uint64_t)g->o_groups * g->o_lora : D;
  widest = HQ > widest ? HQ : widest; /* the attention's output: wo_a's groups read it whole */
  const struct
  {
    void *field;
    uint64_t bytes;
  } want[] = {
      {&b->h, B * hc * D * F},    {&b->hcp, B * ((2 + hc) * hc + 1) * OPS_HC_PARTS * F}, {&b->a_pre, B * hc * F},
      {&b->a_post, B * hc * F},   {&b->a_comb, B * hc * hc * F},    {&b->f_pre, B * hc * F},
      {&b->f_post, NS * B * hc * F}, {&b->f_comb, NS * B * hc * hc * F}, {&b->pre0, B * hc * F},
      {&b->x, B * D * F},         {&b->qa, B * (uint64_t)g->q_lora * F}, {&b->q, B * HQ * F},
      {&b->kv, B * hd * F},       {&b->o, B * HQ * F},  {&b->oa, B * (uint64_t)g->o_groups * g->o_lora * F},
      {&b->y, B * D * F},         {&b->mid, NS * B * hc * D * F},   {&b->xf, S * D * F},
      {&b->logits, S * (uint64_t)g->n_experts * F}, {&b->vocab, B * (uint64_t)g->n_vocab * F},
      {&b->rw, S * K * F},        {&b->ga, P * FF * F},             {&b->upr, P * FF * F},
      {&b->hh, P * FF * F},       {&b->ye, P * D * F},              {&b->moe, S * D * F},
      {&b->sga, S * FF * F},      {&b->sup, S * FF * F},            {&b->shh, S * FF * F},
      {&b->sye, S * D * F},       {&b->up, FWD_UP_BYTES(P)}, {&b->lat, B * hd * F},
      {&b->kvc, B * hd * F},      {&b->scg, B * hd * F},            {&b->ikr, B * ih * F},
      {&b->qix, B * nh * ih * F}, {&b->wix, B * nh * F},
      {&b->kv_p, B * kv_row_bytes(KV_FP8, (uint32_t)hd)}, {&b->lat_p, B * kv_row_bytes(KV_FP4_E4M3, (uint32_t)hd)},
      {&b->ikr_p, B * kv_row_bytes(KV_FP4_POW2, (uint32_t)ih)}, {&b->row_f, hd * F},
      {&b->ckv_f, (uint64_t)f->ckv_f_rows * hd * F}, {&b->ring_f, (uint64_t)f->slots * hd * F},
      {&b->score, B * (f->score_stride > Fs ? f->score_stride : Fs) * F}, /* the indexer's; hip_skip.c's sel */
      {&b->att, OPS_ATTN_SCRATCH(B, g->n_heads, (uint64_t)g->window + g->ix_topk) * F},
      {&b->erow, B * cols * row}, {&b->erow_f, B * egv * F},        {&b->ekv, B * (hc + 1) * D * F},
      {&b->ids, B * 4},           {&b->pos, B * 4},                 {&b->rpos, B * 4},
      {&b->ridx, S * K * 4},      {&b->sel, B * Fs * 4},            {&b->n_sel, B * 4},
      {&b->cand, B * f->cand_words * 4},
      {&b->rwp, P * F},                 {&b->x16, B * (widest > egv ? widest : egv) * 2},
      {&b->xe16, FWD_MAX_SLOTS(P, g->n_experts) * (D > FF ? D : FF) * 2},
      {&b->overflow, 4},          {&b->oa_cnt, FWD_MAX_GROUPS * 4},
      {&b->keep, B * 4},          {&b->back, B * 4},
      {&b->gw, B * FWD_GUESS_K * F},
  };
  _Static_assert(sizeof want / sizeof *want == sizeof(Scratch) / sizeof(void *), "every buffer of Scratch, once");
  for (size_t i = 0; i < sizeof want / sizeof *want; i++)
    if (!fwd_alloc(want[i].field, want[i].bytes))
      return false;
  float pre0[FWD_MAX_BATCH * OPS_MAX_HC] = {0}; /* the first block's pre, a token: stream 0 */
  for (uint64_t i = 0; i < B; i++)
    pre0[i * hc] = 1.0f;
  int n_eg = 0;
  for (int l = 0; l < g->n_layers; l++)
    n_eg += g->engram[l];
  f->xf0 = b->xf, f->moe0 = b->moe, f->mid0 = b->mid, f->f_post0 = b->f_post, f->f_comb0 = b->f_comb;
  return fwd_put(b->pre0, pre0, B * hc * F) &&
         hipHostMalloc((void **)&f->r_ids, P * 4, hipHostMallocMapped | hipHostMallocCoherent) == hipSuccess &&
         hipHostGetDevicePointer((void **)&f->r_ids_dev, f->r_ids, 0) == hipSuccess &&
         hipHostMalloc((void **)&f->r_guess, B * FWD_GUESS_K * 4, hipHostMallocMapped | hipHostMallocCoherent) ==
             hipSuccess &&
         hipHostGetDevicePointer((void **)&f->r_guess_dev, f->r_guess, 0) == hipSuccess &&
         (f->guessed = calloc(f->n_expert_units, sizeof *f->guessed)) &&
         hipHostMalloc((void **)&f->r_up, FWD_UP_BYTES(P), 0) == hipSuccess &&
         (f->m_by = malloc(P * 4)) && (f->m_row = malloc(P * 4)) && (f->m_base = malloc(P * 4)) &&
         (f->m_slot0 = malloc(P * 4)) &&
         hipHostMalloc((void **)&f->stage, NS * B * (uint64_t)(f->n_eg = n_eg ? n_eg : 1) * cols * row, 0) ==
             hipSuccess &&
         hipHostMalloc((void **)&f->pin, sizeof *f->pin, 0) == hipSuccess && subs(f) &&
         hipEventCreateWithFlags(&f->routed, hipEventDisableTiming) == hipSuccess;
}

/* the sub-batches' own state (forward_impl.h's Sub): the first the Forward's own buffers, the others their own */
static bool subs(Forward *f)
{
  const Geometry *g = f->g;
  Scratch *b = &f->b;
  const uint64_t B = f->step, hc = g->hc, F = sizeof(float);
  Sub *s0 = &f->sub[0];
  s0->h = b->h, s0->f_pre = b->f_pre, s0->pos = b->pos, s0->sel = b->sel, s0->n_sel = b->n_sel, s0->cand = b->cand;
  s0->oa_cnt = b->oa_cnt, s0->keep = b->keep, s0->back = b->back, s0->pin = f->pin, s0->rows = f->rows;
  bool ok = true;
  for (uint32_t k = 1; ok && k < f->n_subs; k++)
  {
    Sub *s = &f->sub[k];
    ok = fwd_alloc(&s->h, B * hc * g->n_embd * F) && fwd_alloc(&s->f_pre, B * hc * F) && fwd_alloc(&s->pos, B * 4) &&
         fwd_alloc(&s->sel, B * OPS_MAX_SELECT * 4) && fwd_alloc(&s->n_sel, B * 4) &&
         fwd_alloc(&s->cand, B * f->cand_words * 4) &&
         fwd_alloc(&s->oa_cnt, FWD_MAX_GROUPS * 4) && fwd_alloc(&s->keep, B * 4) && fwd_alloc(&s->back, B * 4) &&
         hipHostMalloc((void **)&s->pin, sizeof *s->pin, 0) == hipSuccess &&
         (s->rows = calloc(B * WEIGHTS_MAX_LAYERS * g->engram_cols, sizeof *s->rows)) != NULL;
  }
  return ok;
}

static bool freqs(Forward *f)
{
  const Geometry *g = f->g;
  float a[ROPE_MAX_DIM / 2], c[ROPE_MAX_DIM / 2];
  const Rope win = {g->rope_dim, g->rope_base, 0, g->rope_factor, g->yarn_beta_fast, g->yarn_beta_slow};
  const Rope comp = {g->rope_dim, g->compress_rope_base, g->rope_original, g->rope_factor, g->yarn_beta_fast,
                     g->yarn_beta_slow};
  const uint64_t bytes = (uint64_t)g->rope_dim / 2 * sizeof(float);
  return rope_freqs(&win, a) && rope_freqs(&comp, c) && fwd_alloc(&f->freqs_win, bytes) &&
         fwd_alloc(&f->freqs_comp, bytes) && fwd_put(f->freqs_win, a, bytes) && fwd_put(f->freqs_comp, c, bytes);
}

Forward *forward_create(const Weights *w, int n_layers, uint32_t max_ctx, uint64_t budget, char err[FORWARD_ERR])
{
  return forward_create_ranked(w, n_layers, max_ctx, budget, NULL, err);
}

Forward *forward_create_ranked(const Weights *w, int n_layers, uint32_t max_ctx, uint64_t budget,
                               const uint64_t *picks, char err[FORWARD_ERR])
{
  const ForwardOptions o = {n_layers, max_ctx, budget, picks, FORWARD_MAX_TOKENS, false};
  return forward_create_with(w, &o, err);
}

Forward *forward_create_with(const Weights *w, const ForwardOptions *o, char err[FORWARD_ERR])
{
  const int n_layers = o ? o->n_layers : 0;
  const uint32_t max_ctx = o ? o->max_ctx : 0, max_tokens = o && o->max_tokens ? o->max_tokens : FORWARD_MAX_TOKENS;
  const uint64_t budget = o ? o->budget : 0, *picks = o ? o->picks : NULL;
  if (!w || n_layers < 1 || n_layers > w->geo.n_layers || !max_ctx || max_tokens > FORWARD_MAX_TOKENS)
    return fwd_fail(err, "no weights, or layers, a context or a step it cannot run"), NULL;
  if (w->n_files > FWD_MAX_FILES)
    return fwd_fail(err, "more files than this build maps"), NULL;
  Forward *f = calloc(1, sizeof *f);
  if (!f)
    return fwd_fail(err, "out of memory"), NULL;
  f->w = w, f->g = &w->geo, f->n_layers = n_layers, f->max_ctx = max_ctx, f->no_guess = o->no_guess;
  if (max_tokens < FWD_MAX_BATCH) /* a small step's server: every buffer of a step at its size (DESIGN.md) */
    f->n_subs = 1, f->max_tokens = max_tokens, f->step = (max_tokens + MM_TILE - 1) / MM_TILE * MM_TILE;
  else
    f->n_subs = (max_tokens + FWD_MAX_BATCH - 1) / FWD_MAX_BATCH, f->max_tokens = f->n_subs * FWD_MAX_BATCH,
    f->step = FWD_MAX_BATCH;
  f->slots = w->geo.window + f->step;
  fwd_decoder(f);
  f->score_stride = 1; /* no compressed layer: never read */
  for (int l = 0; l < n_layers; l++)
    if (w->geo.compress[l] && max_ctx / (uint32_t)w->geo.compress[l] > f->score_stride)
      f->score_stride = max_ctx / (uint32_t)w->geo.compress[l];
  f->ckv_f_rows = f->score_stride < FWD_CKV_F32_ROWS ? f->score_stride : FWD_CKV_F32_ROWS;
  if (w->geo.cand_source >= 0 && w->geo.cand_source < n_layers && w->geo.compress[w->geo.cand_source] &&
      (max_ctx / (uint32_t)w->geo.compress[w->geo.cand_source] + w->geo.cand_block - 1) / w->geo.cand_block >
          (uint32_t)w->geo.cand_blocks) /* a context this long passes the candidate blocks: their bits a token */
    f->cand_words = ((f->score_stride + w->geo.cand_block - 1) / w->geo.cand_block + 31) / 32;
  f->eg_fd = -1;
  for (int i = 0; i < FWD_MAX_FILES; i++)
    f->fds[i] = -1;
  pthread_mutex_init(&f->in.mu, NULL);
  pthread_cond_init(&f->in.cv, NULL);
  bool ok = true;
  for (int i = 0; ok && i < w->n_files; i++) /* the model's files, then an overlay's (weights.h) */
    ok = (f->fds[i] = open(w->paths[i], O_RDONLY)) >= 0;
  char eerr[ENGRAM_ERR] = "";
  if (!ok)
    fwd_fail(err, "a model file could not be opened");
  else if ((f->eg_fd = engram_fd(f, w)) < 0)
    ok = fwd_fail(err, "the Engram's file could not be opened");
  else if (!(f->hash = engram_create(w, f->fds, eerr)))
    ok = fwd_fail(err, eerr);
  else if (!gates(f, w, err) || !guess_routers(f, w, err))
    ok = false;
  else if (!(f->rows = calloc((size_t)FWD_MAX_BATCH * WEIGHTS_MAX_LAYERS * f->g->engram_cols, sizeof *f->rows)) ||
           !(f->eg_off = calloc((size_t)FWD_MAX_SUBS * FWD_MAX_BATCH * engram_layers(w, n_layers) *
                                    f->g->engram_cols + 1,
                                sizeof *f->eg_off)))
    ok = fwd_fail(err, "out of memory");
  else if (!(f->eg_read = rowread_create(FWD_ROW_THREADS)))
    ok = fwd_fail(err, "Engram's readers could not start");
  const size_t nu = (size_t)n_layers * w->geo.n_experts;
  ok = ok && units(f, budget, picks, err) && oa_tables(f, err) &&
       (((f->picks = calloc(nu, sizeof *f->picks)) && (f->seen = calloc(nu, sizeof *f->seen)) &&
         (f->heat = calloc(nu, sizeof *f->heat)) && (f->heat_at = calloc(nu, sizeof *f->heat_at))) ||
        fwd_fail(err, "out of memory"));
  if (ok && !(freqs(f) && scratch(f)))
    ok = fwd_fail(err, "a GPU allocation refused");
  ok = ok && fwd_room_left(f, err);
  if (!ok)
  {
    forward_destroy(f);
    return NULL;
  }
  return f;
}

/* the pool takes its room as experts stream, after create: a budget that leaves the GPU less than that refused now,
   not at a request's first miss (the server READY, then every request failing) */
bool fwd_room_left(const Forward *f, char *err)
{
  size_t free_b = 0, total = 0;
  if (hipMemGetInfo(&free_b, &total) != hipSuccess)
    return fwd_fail(err, "the GPU's free memory could not be read");
  if ((uint64_t)free_b >= f->pool_room)
    return true;
  char why[160];
  snprintf(why, sizeof why, "the GPU's free memory (%.2f GiB) under the expert pool's room (%.2f GiB): a smaller budget",
           (double)free_b / (1ull << 30), (double)f->pool_room / (1ull << 30));
  return fwd_fail(err, why);
}

void forward_destroy(Forward *f)
{
  if (!f)
    return;
  (void)hipDeviceSynchronize(); /* a failed step's kernels done: what is freed or unlocked, the GPU is past */
  fetchq_destroy(f->fq);        /* its workers stopped before the pool they feed goes */
  pool_destroy(f->pool);
  resident_destroy(f->res);
  free(f->rix);
  free(f->is_held);
  rowread_destroy(f->eg_read);
  free(f->eg_off);
  free(f->tdev);
  free(f->ttype);
  hipFree(f->oa_tab);
  hipFree(f->oa_xoff);
  hipFree(f->oa_yoff);
  free(f->in.r);
  pthread_cond_destroy(&f->in.cv);
  pthread_mutex_destroy(&f->in.mu);
  if (f->xf0) /* the views back at the whole buffers: Scratch frees them */
    f->b.xf = f->xf0, f->b.moe = f->moe0, f->b.mid = f->mid0, f->b.f_post = f->f_post0, f->b.f_comb = f->f_comb0;
  hipHostFree(f->r_ids);
  hipHostFree(f->r_guess);
  free(f->guessed);
  hipHostFree(f->r_up);
  free(f->m_by), free(f->m_row), free(f->m_base), free(f->m_slot0);
  free(f->picks), free(f->seen), free(f->heat), free(f->heat_at), free(f->units);
  fwd_cpu_destroy(f);
  hipHostFree(f->lane_up);
  hipFree(f->lane_x), hipFree(f->lane_w), hipFree(f->lane_q), hipFree(f->lane_qd),
      hipFree(f->lane_xq), hipFree(f->lane_xqd);
  hipFree(f->taps);
  if (f->sub[0].h) /* the first sub-batch's buffers back where they came from: Scratch frees them */
  {
    const Sub *s = &f->sub[0];
    f->b.h = s->h, f->b.f_pre = s->f_pre, f->b.pos = s->pos, f->b.sel = s->sel, f->b.n_sel = s->n_sel;
    f->b.cand = s->cand;
    f->b.oa_cnt = s->oa_cnt, f->b.keep = s->keep, f->b.back = s->back, f->pin = s->pin, f->rows = s->rows;
  }
  for (int k = 1; k < FWD_MAX_SUBS; k++)
  {
    Sub *s = &f->sub[k];
    hipFree(s->h), hipFree(s->f_pre), hipFree(s->pos), hipFree(s->sel), hipFree(s->n_sel), hipFree(s->oa_cnt);
    hipFree(s->keep), hipFree(s->back), hipHostFree(s->pin), free(s->rows), hipFree(s->cand);
  }
  void **dev = (void **)&f->b;
  for (size_t i = 0; i < sizeof f->b / sizeof(void *); i++)
    hipFree(dev[i]);
  hipFree(f->freqs_win);
  hipFree(f->freqs_comp);
  hipHostFree(f->stage);
  hipHostFree(f->pin);
  if (f->routed)
    hipEventDestroy(f->routed);
  for (int i = 0; i < FWD_MAX_FILES; i++)
    if (f->fds[i] >= 0)
      close(f->fds[i]);
  if (f->eg_fd >= 0)
    close(f->eg_fd);
  for (int l = 0; l < WEIGHTS_MAX_LAYERS; l++)
    hipFree(f->eg_gate[l]), hipFree(f->guess_router[l]);
  engram_destroy(f->hash);
  free(f->rows);
  free(f);
}

void forward_probe(Forward *f, forward_probe_fn fn, void *arg)
{
  f->probe = fn;
  f->probe_arg = arg;
}

const float *forward_stream(const Forward *f) { return f->b.h; }
const uint64_t *forward_picks(const Forward *f) { return f->picks; }
uint64_t forward_swaps(const Forward *f) { return f->swaps; }
void forward_guesses(const Forward *f, uint64_t *streamed, uint64_t *guessed, uint64_t *read)
{
  *streamed = f->g_need, *guessed = f->g_hit;
  if (read)
    *read = f->g_read;
}
void forward_waits(const Forward *f, uint64_t n[2], uint64_t ns[2])
{
  n[0] = f->w_n[0], n[1] = f->w_n[1], ns[0] = f->w_ns[0], ns[1] = f->w_ns[1];
}
const float *forward_logits(const Forward *f) { return f->b.vocab; }

bool forward_greedy(Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[FORWARD_ERR])
{
  if (!f || !ids || !rows || row0 + rows < row0 || row0 + rows > f->vocab_rows)
    return fwd_fail(err, "rows the last step's head did not compute");
  const uint32_t V = (uint32_t)f->g->n_vocab;
  /* b->ids free between steps (the next one's tokens written at its start) */
  if (!ops_argmax(f->b.vocab + (uint64_t)row0 * V, V, rows, V, f->b.ids, NULL) ||
      hipMemcpyAsync(f->pin->greedy, f->b.ids, rows * sizeof(int32_t), hipMemcpyDeviceToHost, NULL) != hipSuccess ||
      hipStreamSynchronize(NULL) != hipSuccess)
    return fwd_fail(err, "a GPU call failed (the greedy picks)");
  for (uint32_t r = 0; r < rows; r++)
    if ((ids[r] = f->pin->greedy[r]) < 0)
      return fwd_fail(err, "a NaN in a row's logits");
  return true;
}
