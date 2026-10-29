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
  if (ok && !(f->fq = fetchq_create(FWD_FETCH_THREADS, fwd_on_read, &f->in)))
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
    xo[k] = k * oin, yo[k] = k * (uint32_t)g->o_lora;
  bool ok = tab && fwd_alloc(&f->oa_tab, (uint64_t)f->n_layers * G * sizeof *tab) &&
            fwd_alloc(&f->oa_xoff, G * 4) && fwd_alloc(&f->oa_yoff, G * 4) && fwd_put(f->oa_xoff, xo, G * 4) &&
            fwd_put(f->oa_yoff, yo, G * 4);
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
  const uint64_t B = FWD_MAX_BATCH, P = FWD_MAX_BATCH * K, Fs = OPS_MAX_SELECT;
  uint64_t widest = (uint64_t)g->o_groups * g->o_lora > D ? (uint64_t)g->o_groups * g->o_lora : D;
  widest = HQ > widest ? HQ : widest; /* the attention's output: wo_a's groups read it whole */
  const struct
  {
    void *field;
    uint64_t bytes;
  } want[] = {
      {&b->h, B * hc * D * F},    {&b->hcp, B * ((2 + hc) * hc + 1) * OPS_HC_PARTS * F}, {&b->a_pre, B * hc * F},
      {&b->a_post, B * hc * F},   {&b->a_comb, B * hc * hc * F},    {&b->f_pre, B * hc * F},
      {&b->f_post, B * hc * F},   {&b->f_comb, B * hc * hc * F},    {&b->pre0, B * hc * F},
      {&b->x, B * D * F},         {&b->qa, B * (uint64_t)g->q_lora * F}, {&b->q, B * HQ * F},
      {&b->kv, B * hd * F},       {&b->o, B * HQ * F},  {&b->oa, B * (uint64_t)g->o_groups * g->o_lora * F},
      {&b->y, B * D * F},         {&b->mid, B * hc * D * F},        {&b->xf, B * D * F},
      {&b->logits, B * (uint64_t)g->n_experts * F}, {&b->vocab, B * (uint64_t)g->n_vocab * F},
      {&b->rw, B * K * F},        {&b->ga, P * FF * F},             {&b->upr, P * FF * F},
      {&b->hh, P * FF * F},       {&b->ye, P * D * F},              {&b->moe, B * D * F},
      {&b->sga, B * FF * F},      {&b->sup, B * FF * F},            {&b->shh, B * FF * F},
      {&b->sye, B * D * F},       {&b->up, FWD_UP_BYTES}, {&b->lat, B * hd * F},
      {&b->kvc, B * hd * F},      {&b->scg, B * hd * F},            {&b->ikr, ih * F},
      {&b->qix, B * nh * ih * F}, {&b->wix, B * nh * F},            {&b->score, (uint64_t)f->max_ctx * F},
      {&b->att, OPS_ATTN_SCRATCH(B, g->n_heads, (uint64_t)g->window + g->ix_topk) * F},
      {&b->erow, B * cols * row}, {&b->erow_f, B * egv * F},        {&b->ekv, B * (hc + 1) * D * F},
      {&b->ids, B * 4},           {&b->pos, B * 4},                 {&b->rpos, B * 4},
      {&b->ridx, B * K * 4},      {&b->sel, B * Fs * 4},            {&b->n_sel, B * 4},
      {&b->rwp, P * F},                 {&b->x16, B * (widest > egv ? widest : egv) * 2},
      {&b->xe16, FWD_MAX_SLOTS(P, g->n_experts) * (D > FF ? D : FF) * 2},
      {&b->overflow, 4},          {&b->oa_cnt, FWD_MAX_GROUPS * 4},
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
  return fwd_put(b->pre0, pre0, B * hc * F) &&
         hipHostMalloc((void **)&f->stage, B * (uint64_t)(n_eg ? n_eg : 1) * cols * row, 0) == hipSuccess &&
         hipHostMalloc((void **)&f->pin, sizeof *f->pin, 0) == hipSuccess &&
         hipEventCreateWithFlags(&f->routed, hipEventDisableTiming) == hipSuccess;
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
  f->w = w, f->g = &w->geo, f->n_layers = n_layers, f->max_ctx = max_ctx, f->slots = w->geo.window + FWD_MAX_BATCH;
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
  else if (!(f->rows = calloc((size_t)FWD_MAX_BATCH * WEIGHTS_MAX_LAYERS * f->g->engram_cols, sizeof *f->rows)))
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
  hipFree(f->oa_xoff);
  hipFree(f->oa_yoff);
  free(f->in.unit);
  pthread_cond_destroy(&f->in.cv);
  pthread_mutex_destroy(&f->in.mu);
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
