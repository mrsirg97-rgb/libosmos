#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>

#include "forward_impl.h"
#include "substrate/mem/hip_pinner.h"
#include "substrate/gpu/mv.h"
#include "substrate/gpu/ops.h"
#include "substrate/model/rope.h"

#define FWD_FETCH_THREADS 4 /* DESIGN.md, the fetcher: 4 workers read at the SSD's ceiling (6.85 GB/s) */
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

void fwd_probe(const Forward *f, const char *step, int layer, int index, void *dev, uint64_t n, bool ints)
{
  if (f->probe && hipDeviceSynchronize() == hipSuccess)
    f->probe(step, layer, index, dev, n, ints, f->probe_arg);
}

/* a fetchq worker's read is done: the unit goes to the inbox, for the forward to hand the pool */
static void on_read(uint32_t unit, bool ok, void *arg)
{
  (void)ok; /* a failed read is the pin's own read to do: the pool reads through the registration */
  Inbox *in = arg;
  pthread_mutex_lock(&in->mu);
  if (in->n == in->cap)
  {
    const size_t cap = in->cap ? in->cap * 2 : 64;
    uint32_t *grown = realloc(in->unit, cap * sizeof *grown);
    if (grown)
      in->unit = grown, in->cap = cap;
  }
  if (in->n < in->cap)
    in->unit[in->n++] = unit;
  else
    in->broken = true;
  pthread_cond_signal(&in->cv);
  pthread_mutex_unlock(&in->mu);
}

/* the unit held and pinned: its reads handed to the pool as they finish, until it is (or is refused) */
bool fwd_pinned(Forward *f, uint32_t unit, const void *dev[POOL_PARTS], char *err)
{
  while (pool_state(f->pool, unit) == POOL_READING)
  {
    pthread_mutex_lock(&f->in.mu);
    while (!f->in.n && !f->in.broken)
      pthread_cond_wait(&f->in.cv, &f->in.mu);
    const bool broken = f->in.broken;
    uint32_t got[64];
    size_t n = 0;
    while (f->in.n && n < sizeof got / sizeof *got)
      got[n++] = f->in.unit[--f->in.n];
    pthread_mutex_unlock(&f->in.mu);
    if (broken)
      return fwd_fail(err, "a finished read the inbox could not hold");
    for (size_t i = 0; i < n; i++)
      pool_ready(f->pool, got[i]);
  }
  return pool_dev(f->pool, unit, dev) || fwd_fail(err, "a unit the pool could not pin (its budget)");
}

/* the units: each layer's experts (the pool's to cycle), then every other tensor the layers use (pinned and held
   for the forward's life). Engram's tables are read a row at a time and are not units */
static bool units(Forward *f, uint64_t budget, char *err)
{
  const Weights *w = f->w;
  const uint32_t E = (uint32_t)f->g->n_experts;
  const size_t n_res_max = 3 + (size_t)f->n_layers * TENSORS_A_LAYER;
  f->n_expert_units = (uint32_t)f->n_layers * E;
  PoolUnit *u = calloc(f->n_expert_units + n_res_max, sizeof *u);
  const GgufTensor **res = calloc(n_res_max, sizeof *res);
  f->tdev = calloc(w->model->n_tensors, sizeof *f->tdev);
  bool ok = u && res && f->tdev;
  if (!ok)
  {
    free(u), free(res);
    return fwd_fail(err, "out of memory");
  }
  PoolUnit *all = calloc((size_t)w->geo.n_layers * E, sizeof *all);
  ok = all && weights_experts(w, f->fds, all, (size_t)w->geo.n_layers * E) == (size_t)w->geo.n_layers * E;
  if (ok)
    memcpy(u, all, f->n_expert_units * sizeof *u);
  free(all);
  size_t n_res = 0;
  res[n_res++] = w->embed, res[n_res++] = w->output, res[n_res++] = w->output_norm;
  for (int l = 0; l < f->n_layers; l++)
  {
    const LayerWeights *L = &w->layer[l];
    const GgufTensor *const *t = (const GgufTensor *const *)L;
    for (size_t i = 0; i < TENSORS_A_LAYER; i++)
      if (t[i] && t[i] != L->eg_embd && t[i] != L->exp_gate && t[i] != L->exp_up && t[i] != L->exp_down)
        res[n_res++] = t[i];
  }
  for (size_t i = 0; i < n_res; i++)
  {
    const PoolRange r = {f->fds[res[i]->file], res[i]->offset, res[i]->bytes};
    u[f->n_expert_units + i] = (PoolUnit){.part = {r}, .n_parts = 1};
  }
  char perr[POOL_ERR] = "";
  if (ok && !(f->fq = fetchq_create(FWD_FETCH_THREADS, on_read, &f->in)))
    ok = fwd_fail(err, "the fetcher's workers could not start");
  else if (ok && !(f->pool = pool_create(u, f->n_expert_units + n_res, budget, hip_pinner(), fetchq_fetcher(f->fq),
                                          NULL, NULL, perr)))
    ok = fwd_fail(err, perr);
  for (size_t i = 0; ok && i < n_res; i++) /* every resident unit held: all their reads at once */
    ok = pool_pin(f->pool, f->n_expert_units + (uint32_t)i, (const void *[POOL_PARTS]){0}) != POOL_FULL ||
         fwd_fail(err, "the layers' weights are more than the budget");
  for (size_t i = 0; ok && i < n_res; i++)
  {
    const void *dev[POOL_PARTS];
    ok = fwd_pinned(f, f->n_expert_units + (uint32_t)i, dev, err);
    if (ok)
      f->tdev[res[i] - w->model->tensors] = dev[0];
  }
  free(u), free(res);
  return ok;
}

/* each layer's wo_a, a matrix a group (the GGUF's [in, out, groups]: group k's rows k * rows ..), mv_group's table */
static bool oa_tables(Forward *f, char *err)
{
  const uint32_t G = (uint32_t)f->g->o_groups;
  const void **tab = calloc((size_t)f->n_layers * G, sizeof *tab);
  bool ok = tab && fwd_alloc(&f->oa_tab, (uint64_t)f->n_layers * G * sizeof *tab);
  for (int l = 0; ok && l < f->n_layers; l++)
    for (uint32_t k = 0; k < G; k++)
      tab[(size_t)l * G + k] = fwd_dev(f, f->w->layer[l].out_a) + k * (f->w->layer[l].out_a->bytes / G);
  ok = ok && fwd_put(f->oa_tab, tab, (uint64_t)f->n_layers * G * sizeof *tab);
  free(tab);
  return ok || fwd_fail(err, "a GPU allocation refused (the wo_a tables)");
}

bool fwd_alloc(void *p, uint64_t bytes)
{
  void **out = p;
  return hipMalloc(out, bytes ? bytes : 4) == hipSuccess && hipMemset(*out, 0, bytes ? bytes : 4) == hipSuccess;
}

static bool scratch(Forward *f)
{
  const Geometry *g = f->g;
  Scratch *b = &f->b;
  const uint64_t D = g->n_embd, hc = g->hc, F = sizeof(float), HQ = (uint64_t)g->n_heads * g->head_dim;
  const uint64_t row = gguf_bytes(f->w->layer[0].eg_embd ? f->w->layer[0].eg_embd->type : 0, g->engram_head_dim);
  const uint64_t cols = g->engram_cols, K = (uint64_t)g->n_used, FF = (uint64_t)g->expert_ff, hd = g->head_dim;
  const uint64_t ih = g->ix_head_dim, nh = g->ix_heads, egv = cols * (uint64_t)g->engram_head_dim;
  const struct
  {
    void *field;
    uint64_t bytes;
  } want[] = {
      {&b->h, hc * D * F},        {&b->mix, (2 + hc) * hc * F},  {&b->a_pre, hc * F},
      {&b->a_post, hc * F},       {&b->a_comb, hc * hc * F},     {&b->f_pre, hc * F},
      {&b->f_post, hc * F},       {&b->f_comb, hc * hc * F},     {&b->pre0, hc * F},
      {&b->x, D * F},             {&b->qa, (uint64_t)g->q_lora * F}, {&b->q, HQ * F},
      {&b->kv, hd * F},           {&b->o, HQ * F},               {&b->oa, (uint64_t)g->o_groups * g->o_lora * F},
      {&b->y, D * F},             {&b->mid, hc * D * F},         {&b->xf, D * F},
      {&b->logits, (uint64_t)g->n_experts * F}, {&b->vocab, (uint64_t)g->n_vocab * F}, {&b->rw, K * F},
      {&b->ga, K * FF * F},       {&b->up, K * FF * F},          {&b->hh, K * FF * F},
      {&b->ye, K * D * F},        {&b->moe, D * F},              {&b->sga, FF * F},
      {&b->sup, FF * F},          {&b->shh, FF * F},             {&b->sye, D * F},
      {&b->wtab, 3 * K * sizeof(void *)}, {&b->lat, hd * F},     {&b->ikr, ih * F},
      {&b->qix, nh * ih * F},     {&b->wix, nh * F},             {&b->score, (uint64_t)f->max_ctx * F},
      {&b->erow, cols * row},     {&b->erow_f, egv * F},         {&b->ekv, (hc + 1) * D * F},
      {&b->ids, 4},               {&b->pos, 4},                  {&b->rpos, 4},
      {&b->ridx, K * 4},          {&b->sel, OPS_MAX_SELECT * 4}, {&b->n_sel, 4},
  };
  _Static_assert(sizeof want / sizeof *want == sizeof(Scratch) / sizeof(void *), "every buffer of Scratch, once");
  for (size_t i = 0; i < sizeof want / sizeof *want; i++)
    if (!fwd_alloc(want[i].field, want[i].bytes))
      return false;
  const float one = 1.0f;
  return fwd_put(b->pre0, &one, F) && hipHostMalloc((void **)&f->stage, cols * row, 0) == hipSuccess;
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
  if (!w || n_layers < 1 || n_layers > w->geo.n_layers || !max_ctx)
    return fwd_fail(err, "no weights, or layers or a context it cannot run"), NULL;
  if (w->model->n_files > FWD_MAX_FILES)
    return fwd_fail(err, "more files than this build maps"), NULL;
  Forward *f = calloc(1, sizeof *f);
  if (!f)
    return fwd_fail(err, "out of memory"), NULL;
  f->w = w, f->g = &w->geo, f->n_layers = n_layers, f->max_ctx = max_ctx, f->slots = w->geo.window + MV_MAX_TOKENS;
  f->eg_fd = -1;
  for (int i = 0; i < FWD_MAX_FILES; i++)
    f->fds[i] = -1;
  pthread_mutex_init(&f->in.mu, NULL);
  pthread_cond_init(&f->in.cv, NULL);
  bool ok = true;
  for (int i = 0; ok && i < w->model->n_files; i++)
    ok = (f->fds[i] = open(w->model->paths[i], O_RDONLY)) >= 0;
  char eerr[ENGRAM_ERR] = "";
  if (!ok)
    fwd_fail(err, "a model file could not be opened");
  else if ((f->eg_fd = open(w->engram->paths[0], O_RDONLY)) < 0)
    ok = fwd_fail(err, "the Engram file could not be opened");
  else if (!(f->hash = engram_create(w, f->fds, eerr)))
    ok = fwd_fail(err, eerr);
  else if (!(f->rows = calloc((size_t)WEIGHTS_MAX_LAYERS * f->g->engram_cols, sizeof *f->rows)))
    ok = fwd_fail(err, "out of memory");
  ok = ok && units(f, budget, err) && oa_tables(f, err);
  if (ok && !(freqs(f) && scratch(f)))
    ok = fwd_fail(err, "a GPU allocation refused");
  if (!ok)
  {
    forward_destroy(f);
    return NULL;
  }
  return f;
}

void forward_destroy(Forward *f)
{
  if (!f)
    return;
  fetchq_destroy(f->fq); /* its workers stopped before the pool they feed goes */
  pool_destroy(f->pool);
  free(f->tdev);
  hipFree(f->oa_tab);
  free(f->in.unit);
  pthread_cond_destroy(&f->in.cv);
  pthread_mutex_destroy(&f->in.mu);
  void **dev = (void **)&f->b;
  for (size_t i = 0; i < sizeof f->b / sizeof(void *); i++)
    hipFree(dev[i]);
  hipFree(f->freqs_win);
  hipFree(f->freqs_comp);
  hipHostFree(f->stage);
  for (int i = 0; i < FWD_MAX_FILES; i++)
    if (f->fds[i] >= 0)
      close(f->fds[i]);
  if (f->eg_fd >= 0)
    close(f->eg_fd);
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
const float *forward_logits(const Forward *f) { return f->b.vocab; }
