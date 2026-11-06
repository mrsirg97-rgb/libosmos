#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "q4x_impl.h"

/* the forward built: every tensor's bytes on the device (read through a pinned piece), a step's buffers; a sequence's
   state */

#define READ_PIECE (256ull << 20) /* a read's bytes: under a read's ~2 GB, the staging buffer's size */
#define F sizeof(float)

bool q4x_fail(char *err, const char *why)
{
  if (err)
    snprintf(err, Q4X_FWD_ERR, "%s", why);
  return false;
}

const void *q4x_dev(const Q4xForward *f, const GgufTensor *t) { return f->dev[t - f->w->model->tensors]; }

void q4x_seen(const Q4xForward *f, const char *step, int layer, float *dev, uint64_t n)
{
  if (f->probe && hipDeviceSynchronize() == hipSuccess)
    f->probe(step, layer, dev, n, f->probe_arg);
}

static bool alloc(void *p, uint64_t bytes) { return hipMalloc((void **)p, bytes ? bytes : 1) == hipSuccess; }

static bool upload(Q4xForward *f, char *err)
{
  const Gguf *g = f->w->model;
  void *pin = NULL;
  if (!(f->dev = calloc(g->n_tensors, sizeof *f->dev)) || hipHostMalloc(&pin, READ_PIECE, 0) != hipSuccess)
    return q4x_fail(err, "out of memory (the weights' table, or a pinned piece)");
  bool ok = true;
  for (size_t i = 0; ok && i < g->n_tensors; i++)
  {
    const GgufTensor *t = &g->tensors[i];
    const int fd = open(g->paths[t->file], O_RDONLY);
    ok = fd >= 0 && alloc(&f->dev[i], t->bytes);
    for (uint64_t at = 0; ok && at < t->bytes; at += READ_PIECE)
    {
      const size_t n = t->bytes - at < READ_PIECE ? (size_t)(t->bytes - at) : READ_PIECE;
      ok = pread(fd, pin, n, (off_t)(t->offset + at)) == (ssize_t)n &&
           hipMemcpy((uint8_t *)f->dev[i] + at, pin, n, hipMemcpyHostToDevice) == hipSuccess;
    }
    if (fd >= 0)
      close(fd);
    if (!ok)
      snprintf(err, Q4X_FWD_ERR, "%s: not read into GPU memory", t->name);
  }
  hipHostFree(pin);
  return ok;
}

static bool buffers(Q4xForward *f, char *err)
{
  const Q4xGeometry *g = &f->w->geo;
  const uint64_t n = f->max_tokens, D = (uint64_t)g->n_embd, hd = (uint64_t)g->hc * D, k = (uint64_t)g->n_used;
  const uint64_t ch = 2ull * g->ssm_groups * g->ssm_state + (uint64_t)g->ssm_inner, in = (uint64_t)g->ssm_inner;
  const uint64_t q = (uint64_t)g->n_heads * g->head_dim, kv = (uint64_t)g->n_kv_heads * g->head_dim;
  const uint64_t vh = (uint64_t)g->ssm_v_heads, ff = (uint64_t)g->ff_exp, sff = (uint64_t)g->ff_shexp;
  Q4xBuf *b = &f->b;
  const struct
  {
    void *p;
    uint64_t bytes;
  } want[] = {
      {&b->h, n * hd * F},          {&b->xn, n * hd * F},         {&b->lo, n * g->hc_rank * F}, {&b->gate, n * hd * F},
      {&b->mixed, n * D * F},       {&b->inj, n * g->hc * F},     {&b->out, n * D * F},         {&b->x, n * D * F},
      {&b->qkv, n * ch * F},        {&b->z, n * in * F},          {&b->al, n * vh * F},         {&b->be, n * vh * F},
      {&b->g, n * vh * F},          {&b->b, n * vh * F},          {&b->cv, n * ch * F},         {&b->ro, n * in * F},
      {&b->fo, n * in * F},         {&b->qf, n * 2 * q * F},      {&b->kr, n * kv * F},         {&b->q, n * q * F},
      {&b->gt, n * q * F},          {&b->ao, n * q * F},          {&b->rl, n * g->n_experts * F}, {&b->w, n * k * F},
      {&b->xe, n * k * D * F},      {&b->ga, n * k * ff * F},     {&b->up, n * k * ff * F},     {&b->y, n * k * D * F},
      {&b->shg, n * sff * F},       {&b->shu, n * sff * F},       {&b->sh, n * D * F},          {&b->sl, n * F},
      {&b->ids, n * k * 4},         {&b->rows, n * k * 4},        {&b->cnt, n * k * 4},         {&b->row, n * k * 4},
      {&b->ptr, n * k * sizeof(void *)}, {&b->emb, n * D * F},    {&b->key, n * hd * F},        {&b->val, n * D * F},
      {&b->kn, n * hd * F},         {&b->qn, n * hd * F},         {&b->pg, n * g->hc * F},      {&b->gated, n * hd * F},
      {&b->nrm, n * hd * F},        {&b->conv, n * hd * F},       {&b->ple_rows, n * g->ple_heads * 4},
      {&b->tok, n * 4},             {&b->logits, n * (uint64_t)g->n_vocab * F},
  };
  for (size_t i = 0; i < sizeof want / sizeof *want; i++)
    if (!alloc(want[i].p, want[i].bytes))
      return q4x_fail(err, "GPU memory refused (a step's buffers)");
  return (hipHostMalloc((void **)&b->pin_rows, n * g->ple_heads * 4, 0) == hipSuccess &&
          hipHostMalloc((void **)&b->pin_tok, n * 4, 0) == hipSuccess) ||
         q4x_fail(err, "pinned memory refused (a step's tokens)");
}

void q4x_forward_destroy(Q4xForward *f)
{
  if (!f)
    return;
  for (size_t i = 0; f->dev && i < f->w->model->n_tensors; i++)
    hipFree(f->dev[i]);
  free(f->dev);
  void **bufs = (void **)&f->b;
  for (size_t i = 0; i < offsetof(Q4xBuf, pin_rows) / sizeof(void *); i++)
    hipFree(bufs[i]);
  hipHostFree(f->b.pin_rows), hipHostFree(f->b.pin_tok), hipFree(f->b.logits);
  free(f);
}

Q4xForward *q4x_forward_create(const Q4xWeights *w, uint32_t max_ctx, uint32_t max_tokens, char err[Q4X_FWD_ERR])
{
  if (!w || !max_ctx || max_ctx > OPS_Q4X_MAX_KEYS || !max_tokens || max_tokens > max_ctx)
    return q4x_fail(err, "no weights, or a context past the dense attention's reach, or no tokens a step"), NULL;
  Q4xForward *f = calloc(1, sizeof *f);
  if (!f)
    return q4x_fail(err, "out of memory"), NULL;
  f->w = w, f->max_ctx = max_ctx, f->max_tokens = max_tokens;
  if (!upload(f, err) || !buffers(f, err))
    return q4x_forward_destroy(f), NULL;
  return f;
}

void q4x_forward_probe(Q4xForward *f, q4x_probe_fn fn, void *arg)
{
  if (f)
    f->probe = fn, f->probe_arg = arg;
}

Q4xSeqBytes q4x_seq_bytes(const Q4xGeometry *g)
{
  const uint64_t ch = 2ull * g->ssm_groups * g->ssm_state + (uint64_t)g->ssm_inner;
  return (Q4xSeqBytes){(uint64_t)g->ssm_v_heads * g->ssm_state * g->ssm_state * F, (g->ssm_conv - 1) * ch * F,
                       (uint64_t)g->n_kv_heads * g->head_dim * F,
                       (uint64_t)(g->ple_conv - 1) * g->ple_ngram * g->hc * g->n_embd * F};
}

void q4x_seq_destroy(Q4xSeq *s)
{
  if (!s)
    return;
  for (int l = 0; l < Q4X_MAX_LAYERS; l++)
    hipFree(s->ssm[l]), hipFree(s->conv[l]), hipFree(s->kc[l]), hipFree(s->vc[l]);
  hipFree(s->ple_hist);
  free(s);
}

Q4xSeq *q4x_seq_create(Q4xForward *f, char err[Q4X_FWD_ERR])
{
  if (!f)
    return q4x_fail(err, "no forward"), NULL;
  const Q4xGeometry *g = &f->w->geo;
  Q4xSeq *s = calloc(1, sizeof *s);
  if (!s)
    return q4x_fail(err, "out of memory"), NULL;
  s->f = f;
  const Q4xSeqBytes sb = q4x_seq_bytes(g);
  bool ok = alloc(&s->ple_hist, sb.ple);
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = g->recurrent[l] ? alloc(&s->ssm[l], sb.ssm) && alloc(&s->conv[l], sb.conv)
                         : alloc(&s->kc[l], f->max_ctx * sb.kv_row) && alloc(&s->vc[l], f->max_ctx * sb.kv_row);
  if (!ok || !q4x_seq_reset(s))
    return q4x_seq_destroy(s), q4x_fail(err, "GPU memory refused (a sequence's state)"), NULL;
  return s;
}

int32_t q4x_seq_position(const Q4xSeq *s) { return s ? s->pos : -1; }
