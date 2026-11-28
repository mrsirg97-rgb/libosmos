/* DeepSeek's forward, its weights made resident (ds4_impl.h): the host-to-device copies, the dense part (Q5_K repacked),
   the experts' pool in their order, the attention's tables, Engram's rows, the gates and the guess's routers */
#define _GNU_SOURCE /* O_DIRECT */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <hip/hip_runtime_api.h>
#include "osmos/model/tensor.h"
#include "osmos/mem/ledger_hip.h"
#include "osmos/run/fit.h"
#include "ds4_impl.h"
#include "osmos/mem/experts.h"
#include "osmos/mem/arena.h"
#include "osmos/mem/hip_slots.h"
#include "osmos/gpu/kvfmt.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/model/qtype.h"
#include "osmos/model/rope.h"
#include "osmos/err.h"
#include "osmos/model/dspark.h"

#define FWD_REPACK_STAGE (64u << 20) /* the Q5_K repack's chunk of a matrix, copied out (hip_ds4_make.c) */
/* the fetcher's readers (DESIGN.md): more than the SSD's ceiling needs, a decode's miss sooner */
#define FWD_FETCH_THREADS 12
#define TENSORS_A_LAYER (sizeof(LayerWeights) / sizeof(const GgufTensor *))
_Static_assert(sizeof(LayerWeights) % sizeof(const GgufTensor *) == 0, "LayerWeights is its tensors and only them");
_Static_assert(sizeof(Scratch) % sizeof(void *) == 0, "Scratch is its device buffers and only them");

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

/* the pool's units read O_DIRECT: each part's file its direct twin (the resident set keeps the buffered ones) */
static bool direct_units(Forward *f, PoolUnit *all)
{
  for (uint32_t u = 0; u < f->n_expert_units; u++)
    for (int k = 0; k < all[u].n_parts; k++)
    {
      int i = 0;
      while (i < FWD_MAX_FILES && f->fds[i] != all[u].part[k].fd)
        i++;
      if (i == FWD_MAX_FILES || f->dfds[i] < 0)
        return false;
      all[u].part[k].fd = f->dfds[i];
    }
  return true;
}

static int rank_cmp(const void *a, const void *b, void *ctx) /* (qsort_r's: a layer's picks its context) */
{
  const uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
  const uint64_t *picks = ctx, px = picks[x], py = picks[y];
  return px != py ? (px < py) - (px > py) : (x > y) - (x < y); /* more picks first, then the lower id */
}

/* the resident experts: a rank at a time, each layer's expert of that rank while the budget holds it - by the
   picks counted on calibration text (forward_create_ranked's, most first), else by id. so each layer's share is
   about equal and its most-picked first */
/* every expert unit, the most wanted first: rank by rank, a layer's at each (its most-picked first with picks: about
   an equal share of each layer's), the shared plan's order (experts_plan) - NULL, out of memory */
static uint32_t *expert_order(const Forward *f, const uint64_t *picks)
{
  const uint32_t E = (uint32_t)f->g->n_experts, L = (uint32_t)f->n_layers;
  uint32_t *rank = malloc((size_t)L * E * sizeof *rank), *order = malloc((size_t)L * E * sizeof *order);
  if (!rank || !order)
    return free(rank), free(order), NULL;
  for (uint32_t l = 0; l < L; l++)
  {
    uint32_t *o = rank + (size_t)l * E;
    for (uint32_t e = 0; e < E; e++)
      o[e] = e;
    if (picks)
      qsort_r(o, E, sizeof *o, rank_cmp, (void *)(picks + (size_t)l * E));
  }
  size_t n = 0;
  for (uint32_t r = 0; r < E; r++)
    for (uint32_t l = 0; l < L; l++)
      order[n++] = l * E + rank[(size_t)l * E + r];
  free(rank);
  return order;
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
  bool any = false;
  for (size_t i = 0; i < n_t && !any; i++)
  {
    uint32_t rows, cols;
    any = repacks(f->w, t_res[i], &rows, &cols);
  }
  if (!any)
    return true;
  void *stage = NULL;
  bool ok = ledger_malloc(f->ledger, &stage, FWD_REPACK_STAGE) == hipSuccess;
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
  if (stage)
    ledger_free(f->ledger, stage);
  return ok;
}

/* a promotion's copy (membrane.h): on the forward's stream, after the kernels that read the stream's copy before */
static bool copy_on_stream(void *dst, const void *src, uint64_t bytes, void *ctx)
{
  (void)ctx;
  return ops_copy(dst, src, bytes, NULL);
}

/* the units: each layer's experts (the pool's, every one: the budget's read in first), then every other tensor the
   layers use - all of them resident (resident.h). Engram's tables are read a row at a time and are not units */
/* the dense weights resident (every tensor the layers use but the routed experts and the Engram's own), and every
   expert's unit - the experts' plan after the rest of the build (experts): a budget's one meaning, the whole build's */
bool fwd_dense(Forward *f, char *err)
{
  const Weights *w = f->w;
  const uint32_t E = (uint32_t)f->g->n_experts;
  const size_t n_res_max = 3 + (size_t)f->n_layers * TENSORS_A_LAYER;
  f->n_expert_units = (uint32_t)f->n_layers * E;
  PoolUnit *all = calloc((size_t)w->geo.n_layers * E, sizeof *all);
  PoolUnit *res = calloc(n_res_max, sizeof *res);
  const GgufTensor **t_res = calloc(n_res_max, sizeof *t_res);
  f->tdev = calloc(weights_tensors(w), sizeof *f->tdev);
  f->ttype = calloc(weights_tensors(w), sizeof *f->ttype);
  bool ok = all && res && t_res && f->tdev && f->ttype &&
            weights_experts(w, f->fds, all, (size_t)w->geo.n_layers * E) == (size_t)w->geo.n_layers * E;
  if (!ok)
  {
    free(all), free(res), free(t_res);
    return osmos_fail(err, "out of memory, or the experts' units");
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
  char perr[OSMOS_ERR] = "";
  if (!(f->gpu_slots = hip_slots_create(f->ledger, perr)) ||
      !(f->res = resident_create(res, n_t, hip_slots(f->gpu_slots), FWD_FETCH_THREADS, perr)))
    ok = osmos_fail(err, "%s", perr);
  for (size_t i = 0; ok && i < n_t; i++)
  {
    const void *dev[POOL_PARTS];
    ok = resident_dev(f->res, i, dev);
    f->tdev[weights_tensor_index(w, t_res[i])] = dev[0];
  }
  ok = ok && (repack_all(f, t_res, n_t) || osmos_fail(err, "the Q5_K matrices' repack failed"));
  free(res), free(t_res);
  f->units = all; /* the CPU lane's parts, a promotion's copy, the experts' plan */
  return ok;
}

/* what the forward has built so far (its records' bytes, the resident set's, the drafter's): the plan's fixed part,
   counted - not a reading of the GPU's free memory (others' work moves that) */
static uint64_t built(const Forward *f) { return f->bufs.bytes + resident_bytes(f->res) + fwd_drafter_bytes(f->drafter); }

/* the experts through the membrane (osmos/mem/experts.h): the shared plan, the whole build's budget - the experts'
   room what the rest leaves (the build so far, the stream, the slack; budget 0 the least: a layer's) - their warm
   start read at the first step (membrane_warm), not here: fit.h's builds */
bool fwd_experts(Forward *f, uint64_t budget, const uint64_t *picks, char *err)
{
  const uint32_t E = (uint32_t)f->g->n_experts;
  uint32_t *order = expert_order(f, picks);
  ExpertsPlan plan = {0};
  const uint64_t fixed = built(f);
  bool ok = order || osmos_fail(err, "out of memory (the experts' order)");
  ok = ok && (experts_plan(f->units, (uint32_t)f->n_layers, E, order, 0, budget, fixed, ARENA_GRAIN, &plan) ||
              osmos_fail(err, "the experts' plan (their units, or their order)"));
  ok = ok && (direct_units(f, f->units) || osmos_fail(err, "out of memory (the pools' units)"));
  if (ok)
  {
    Experts ex;
    f->pool_room = plan.room;
    ok = experts_create(f->ledger, f->units, (uint32_t)f->n_layers, E, order, &plan, ARENA_GRAIN, FWD_FETCH_THREADS,
                        copy_on_stream, NULL, f->ahead_limit, false, &ex, err);
    f->mem = ex.mem, f->hs = ex.slots, f->hs_stream = ex.stream;
  }
  free(order);
  if (ok) /* (forward_budget: the whole build's bytes, as made - the fixed part and the experts' host slots) */
    f->budget = fixed + host_slots_bytes(f->hs) + host_slots_bytes(f->hs_stream);
  return ok;
}

/* each layer's wo_a, a matrix a group (the GGUF's [in, out, groups]: group k's rows k * rows ..), and where each
   group reads and writes: mm_group's tables, fixed for the forward's life */
bool fwd_oa_tables(Forward *f, char *err)
{
  const Geometry *g = f->g;
  const uint32_t G = (uint32_t)g->o_groups, oin = (uint32_t)(g->n_heads * g->head_dim / G);
  if (G > FWD_MAX_GROUPS)
    return osmos_fail(err, "more wo_a groups than this build runs");
  const void **tab = calloc((size_t)f->n_layers * G, sizeof *tab);
  uint32_t xo[FWD_MAX_GROUPS], yo[FWD_MAX_GROUPS];
  for (uint32_t k = 0; k < G; k++)
    xo[k] = k * oin * MM_TILE, yo[k] = k * (uint32_t)g->o_lora; /* group k's first column, in the tiled x16 */
  bool ok = tab && fwd_alloc(&f->bufs, &f->oa_tab, (uint64_t)f->n_layers * G * sizeof *tab) &&
            fwd_alloc(&f->bufs, &f->oa_xoff, G * 4) && fwd_alloc(&f->bufs, &f->oa_yoff, G * 4) && fwd_put(f->oa_xoff, xo, G * 4) &&
            fwd_put(f->oa_yoff, yo, G * 4);
  for (int l = 0; ok && l < f->n_layers; l++)
    for (uint32_t k = 0; k < G; k++)
      tab[(size_t)l * G + k] = fwd_dev(f, f->w->layer[l].out_a) + k * (fwd_bytes(f, f->w->layer[l].out_a) / G);
  ok = ok && fwd_put(f->oa_tab, tab, (uint64_t)f->n_layers * G * sizeof *tab);
  free(tab);
  return ok || osmos_fail(err, "a GPU allocation refused (the wo_a tables)");
}

bool fwd_alloc(GpuBufs *b, void *p, uint64_t bytes)
{
  void **out = p;
  (void)out;
  return gpu_bufs_one(b, b->ledger, p, bytes); /* (zeroed and padded: osmos/gpu/bufs.h's policy, every family's) */
}

bool fwd_pinned(GpuBufs *b, void *p, uint64_t bytes) { return gpu_bufs_host(b, b->ledger, p, bytes); }
bool fwd_mapped(GpuBufs *b, void *p, uint64_t bytes) { return gpu_bufs_mapped(b, b->ledger, p, bytes); }

size_t fwd_engram_layers(const Weights *w, int n_layers)
{
  size_t n = 0;
  for (int l = 0; l < n_layers; l++)
    n += w->geo.engram[l] != 0;
  return n;
}

/* the file the Engram's rows are read from (its own descriptor: rowread's): ds1017's second file, or the model's own
   shard that holds per_layer_token_embd (every layer's slice in it) */
int fwd_engram_fd(const Forward *f, const Weights *w)
{
  if (!w->geo.engram_own)
    return open(w->engram->paths[0], O_RDONLY);
  for (int l = 0; l < w->geo.n_layers; l++)
    if (w->geo.engram[l])
      return dup(f->fds[w->layer[l].eg_embd->file]);
  return -1;
}


/* each Engram layer's gate as ops_engram_gate reads it: F32 [hc][dim], q k. a llama.cpp GGUF's folded (engram_gate:
   read as it is); ds1017's q and k (BF16) multiplied here - a product of two BF16 values exact in F32 (8-bit
   significands, 16 bits of product): the kernel's own product before, the same bits */
bool fwd_gates(Forward *f, const Weights *w, char *err)
{
  const uint64_t n = (uint64_t)w->geo.hc * (uint64_t)w->geo.n_embd;
  float *g = malloc(n * sizeof *g);
  uint16_t *q = malloc(n * sizeof *q), *k = malloc(n * sizeof *k);
  bool ok = (g && q && k) || osmos_fail(err, "out of memory (the Engram's gates)");
  for (int l = 0; ok && l < w->geo.n_layers; l++)
  {
    if (!w->geo.engram[l])
      continue;
    const LayerWeights *L = &w->layer[l];
    if (w->geo.engram_own)
      ok = tensor_read_fd(f->fds[(L->eg_gate)->file], L->eg_gate, g);
    else if ((ok = tensor_read_fd(f->fds[(L->eg_q)->file], L->eg_q, q) && tensor_read_fd(f->fds[(L->eg_k)->file], L->eg_k, k)))
      for (uint64_t i = 0; i < n; i++)
      {
        const uint32_t a = (uint32_t)q[i] << 16, b = (uint32_t)k[i] << 16;
        float x, y;
        memcpy(&x, &a, 4), memcpy(&y, &b, 4);
        g[i] = x * y;
      }
    /* each cause its own (one message for all three read "could not be read" for a refused allocation) */
    ok = (ok || osmos_fail(err, "the Engram's gates of layer %d could not be read", l)) &&
         (fwd_alloc(&f->bufs, &f->eg_gate[l], n * sizeof *g) || osmos_fail(err, "GPU memory refused (the Engram's gates)")) &&
         (fwd_put(f->eg_gate[l], g, n * sizeof *g) || osmos_fail(err, "a GPU copy failed (the Engram's gates)"));
  }
  free(g), free(q), free(k);
  return ok;
}

/* an F32 router (llama.cpp's GGUFs) rounded to BF16 (nearest, ties to even) for the guess alone: the guess picks the
   experts read ahead, which change no bit - half its bytes a layer. the routing itself reads the F32 */
bool fwd_guess_routers(Forward *f, const Weights *w, char *err)
{
  bool ok = true;
  for (int l = 1; ok && !f->no_guess && l < f->n_layers; l++) /* no guess: no copy; layer 0 is never guessed */
  {
    const GgufTensor *t = w->layer[l].router;
    if (t->type != QTYPE_F32)
      continue;
    const uint64_t n = t->bytes / sizeof(float);
    float *x = malloc(t->bytes);
    uint16_t *h = malloc(n * sizeof *h);
    ok = x && h && tensor_read_fd(f->fds[(t)->file], t, x);
    for (uint64_t i = 0; ok && i < n; i++)
    {
      uint32_t u;
      memcpy(&u, &x[i], 4);
      h[i] = (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
    }
    ok = ok && fwd_alloc(&f->bufs, &f->guess_router[l], n * sizeof *h) && fwd_put(f->guess_router[l], h, n * sizeof *h);
    free(x), free(h);
  }
  return ok || osmos_fail(err, "a router's copy for the guess could not be made");
}
