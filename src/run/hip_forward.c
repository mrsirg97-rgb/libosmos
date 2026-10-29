#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>

#include "forward_impl.h"
#include "substrate/gpu/mv.h"
#include "substrate/gpu/ops.h"
#include "substrate/model/rope.h"

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

static int by_at(const void *a, const void *b)
{
  const Span *x = a, *y = b;
  return x->at < y->at ? -1 : x->at > y->at;
}

static bool add_span(Forward *f, const GgufTensor *t, uint64_t pg, char *err)
{
  if (f->n_span == FWD_MAX_SPANS)
    return fwd_fail(err, "more tensors than this build registers");
  uint8_t *a = f->base[t->file] + t->offset / pg * pg;
  uint8_t *b = f->base[t->file] + (t->offset + t->bytes + pg - 1) / pg * pg;
  f->span[f->n_span++] = (Span){a, (uint64_t)(b - a)};
  return true;
}

/* every tensor the layers use (Engram's tables aside: read a row at a time), page-aligned, merged where pages touch */
static bool spans(Forward *f, uint64_t max_resident, char *err)
{
  const uint64_t pg = (uint64_t)sysconf(_SC_PAGESIZE);
  const GgufTensor *top[] = {f->w->embed, f->w->output, f->w->output_norm};
  for (size_t i = 0; i < sizeof top / sizeof *top; i++)
    if (!add_span(f, top[i], pg, err))
      return false;
  for (int l = 0; l < f->n_layers; l++)
  {
    const GgufTensor *const *t = (const GgufTensor *const *)&f->w->layer[l];
    for (size_t i = 0; i < TENSORS_A_LAYER; i++)
      if (t[i] && t[i] != f->w->layer[l].eg_embd && !add_span(f, t[i], pg, err))
        return false;
  }
  qsort(f->span, (size_t)f->n_span, sizeof *f->span, by_at);
  int n = 0;
  uint64_t total = 0;
  for (int i = 0; i < f->n_span; i++)
  {
    Span *last = n ? &f->span[n - 1] : NULL;
    if (last && (uint8_t *)f->span[i].at <= (uint8_t *)last->at + last->bytes)
    {
      uint8_t *end = (uint8_t *)f->span[i].at + f->span[i].bytes;
      if (end > (uint8_t *)last->at + last->bytes)
        last->bytes = (uint64_t)(end - (uint8_t *)last->at);
    }
    else
      f->span[n++] = f->span[i];
  }
  f->n_span = n;
  for (int i = 0; i < n; i++)
    total += f->span[i].bytes;
  if (total > max_resident)
    return fwd_fail(err, "the layers' weights are more than max_resident");
  for (int i = 0; i < n; i++)
  {
    void *dev = NULL;
    if (hipHostRegister(f->span[i].at, f->span[i].bytes, hipHostRegisterMapped | hipHostRegisterReadOnly) !=
        hipSuccess)
      return fwd_fail(err, "a registration refused");
    f->n_reg = i + 1;
    if (hipHostGetDevicePointer(&dev, f->span[i].at, 0) != hipSuccess || dev != f->span[i].at)
      return fwd_fail(err, "a registration the GPU reads elsewhere than the host");
  }
  return true;
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
  const uint64_t row = g->engram_head_dim / 32 * 18, cols = g->engram_cols;
  const float one = 1.0f;
  return fwd_alloc(&b->h, hc * D * F) && fwd_alloc(&b->mix, (2 + hc) * hc * F) && fwd_alloc(&b->a_pre, hc * F) &&
         fwd_alloc(&b->a_post, hc * F) && fwd_alloc(&b->a_comb, hc * hc * F) && fwd_alloc(&b->f_pre, hc * F) &&
         fwd_alloc(&b->f_post, hc * F) && fwd_alloc(&b->f_comb, hc * hc * F) && fwd_alloc(&b->pre0, hc * F) &&
         fwd_put(b->pre0, &one, F) && fwd_alloc(&b->x, D * F) && fwd_alloc(&b->qa, (uint64_t)g->q_lora * F) &&
         fwd_alloc(&b->q, HQ * F) && fwd_alloc(&b->kv, (uint64_t)g->head_dim * F) && fwd_alloc(&b->o, HQ * F) &&
         fwd_alloc(&b->oa, (uint64_t)g->o_groups * g->o_lora * F) && fwd_alloc(&b->y, D * F) &&
         fwd_alloc(&b->mid, hc * D * F) && fwd_alloc(&b->xf, D * F) && fwd_alloc(&b->logits, (uint64_t)g->n_experts * F) &&
         fwd_alloc(&b->rw, (uint64_t)g->n_used * F) && fwd_alloc(&b->ga, (uint64_t)g->expert_ff * F) &&
         fwd_alloc(&b->up, (uint64_t)g->expert_ff * F) && fwd_alloc(&b->hh, (uint64_t)g->expert_ff * F) &&
         fwd_alloc(&b->ye, D * F) && fwd_alloc(&b->moe, D * F) && fwd_alloc(&b->lat, (uint64_t)g->head_dim * F) &&
         fwd_alloc(&b->ikr, (uint64_t)g->ix_head_dim * F) &&
         fwd_alloc(&b->qix, (uint64_t)g->ix_heads * g->ix_head_dim * F) && fwd_alloc(&b->wix, (uint64_t)g->ix_heads * F) &&
         fwd_alloc(&b->score, (uint64_t)f->max_ctx * F) && fwd_alloc(&b->erow, cols * row) &&
         fwd_alloc(&b->erow_f, cols * g->engram_head_dim * F) && fwd_alloc(&b->ekv, (hc + 1) * D * F) &&
         fwd_alloc(&b->ids, 4) && fwd_alloc(&b->pos, 4) && fwd_alloc(&b->rpos, 4) && fwd_alloc(&b->ridx, (uint64_t)g->n_used * 4) &&
         fwd_alloc(&b->sel, OPS_MAX_SELECT * 4) && fwd_alloc(&b->n_sel, 4) &&
         hipHostMalloc((void **)&f->stage, cols * row, 0) == hipSuccess;
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

Forward *forward_create(const Weights *w, int n_layers, uint32_t max_ctx, uint64_t max_resident, char err[FORWARD_ERR])
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
  bool ok = true;
  for (int i = 0; ok && i < w->model->n_files; i++)
  {
    ok = (f->fds[i] = open(w->model->paths[i], O_RDONLY)) >= 0;
    f->size[i] = w->model->sizes[i];
    ok = ok && (f->base[i] = mmap(NULL, f->size[i], PROT_READ, MAP_SHARED, f->fds[i], 0)) != MAP_FAILED;
    if (!ok)
      f->base[i] = NULL;
  }
  char eerr[ENGRAM_ERR] = "";
  if (!ok)
    fwd_fail(err, "a model file could not be opened or mapped");
  else if ((f->eg_fd = open(w->engram->paths[0], O_RDONLY)) < 0)
    ok = fwd_fail(err, "the Engram file could not be opened");
  else if (!(f->hash = engram_create(w, f->fds, eerr)))
    ok = fwd_fail(err, eerr);
  else if (!(f->rows = calloc((size_t)WEIGHTS_MAX_LAYERS * f->g->engram_cols, sizeof *f->rows)))
    ok = fwd_fail(err, "out of memory");
  ok = ok && spans(f, max_resident, err);
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
  for (int i = 0; i < f->n_reg; i++)
    hipHostUnregister(f->span[i].at);
  void **dev = (void **)&f->b;
  for (size_t i = 0; i < sizeof f->b / sizeof(void *); i++)
    hipFree(dev[i]);
  hipFree(f->freqs_win);
  hipFree(f->freqs_comp);
  hipHostFree(f->stage);
  for (int i = 0; i < FWD_MAX_FILES; i++)
  {
    if (f->base[i])
      munmap(f->base[i], f->size[i]);
    if (f->fds[i] >= 0)
      close(f->fds[i]);
  }
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
