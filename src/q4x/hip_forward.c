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

/* a tensor's device bytes: the model's, or the MTP head's (its own file's table) */
const void *q4x_dev(const Q4xForward *f, const GgufTensor *t)
{
  const Gguf *m = f->w->model;
  if (t >= m->tensors && t < m->tensors + m->n_tensors)
    return f->dev[t - m->tensors];
  return f->mtp_dev[t - f->mtp->model->tensors];
}

void q4x_seen(const Q4xForward *f, const char *step, int layer, float *dev, uint64_t n)
{
  if (f->probe && hipDeviceSynchronize() == hipSuccess)
    f->probe(step, layer, dev, n, f->probe_arg);
}

static bool alloc(void *p, uint64_t bytes) { return hipMalloc((void **)p, bytes ? bytes : 1) == hipSuccess; }

/* every tensor of g read into GPU memory through a pinned piece: *dev its table */
static bool upload(const Gguf *g, void ***dev, char *err)
{
  void *pin = NULL;
  if (!(*dev = calloc(g->n_tensors, sizeof **dev)) || hipHostMalloc(&pin, READ_PIECE, 0) != hipSuccess)
    return q4x_fail(err, "out of memory (the weights' table, or a pinned piece)");
  bool ok = true;
  for (size_t i = 0; ok && i < g->n_tensors; i++)
  {
    const GgufTensor *t = &g->tensors[i];
    const int fd = open(g->paths[t->file], O_RDONLY);
    ok = fd >= 0 && alloc(&(*dev)[i], t->bytes);
    for (uint64_t at = 0; ok && at < t->bytes; at += READ_PIECE)
    {
      const size_t n = t->bytes - at < READ_PIECE ? (size_t)(t->bytes - at) : READ_PIECE;
      ok = pread(fd, pin, n, (off_t)(t->offset + at)) == (ssize_t)n &&
           hipMemcpy((uint8_t *)(*dev)[i] + at, pin, n, hipMemcpyHostToDevice) == hipSuccess;
    }
    if (fd >= 0)
      close(fd);
    if (!ok)
      snprintf(err, Q4X_FWD_ERR, "%s: not read into GPU memory", t->name);
  }
  hipHostFree(pin);
  return ok;
}

static void free_table(const Gguf *g, void **dev)
{
  for (size_t i = 0; dev && i < g->n_tensors; i++)
    hipFree(dev[i]);
  free(dev);
}

/* the MTP head's buffers */
static bool mtp_buffers(Q4xForward *f, char *err)
{
  const Q4xGeometry *g = &f->w->geo;
  const uint64_t n = f->max_tokens, D = (uint64_t)g->n_embd, hd = (uint64_t)g->hc * D;
  Q4xMtpBuf *m = &f->m;
  const bool ok = alloc(&m->hid, n * hd * F) && alloc(&m->hin, n * hd * F) && alloc(&m->emb, n * D * F) &&
                  alloc(&m->hn, n * hd * F) && alloc(&m->join, 2 * n * hd * F) &&
                  alloc(&m->logits, (uint64_t)g->n_vocab * F) && alloc(&m->top_id, 4) && alloc(&m->top_p, 4) &&
                  hipHostMalloc((void **)&m->pin_id, 4, 0) == hipSuccess &&
                  hipHostMalloc((void **)&m->pin_p, 4, 0) == hipSuccess;
  return ok || q4x_fail(err, "GPU memory refused (the MTP head's buffers)");
}

static bool buffers(Q4xForward *f, char *err)
{
  const Q4xGeometry *g = &f->w->geo;
  const uint64_t n = f->max_tokens, D = (uint64_t)g->n_embd, hd = (uint64_t)g->hc * D, k = (uint64_t)g->n_used;
  const uint64_t ch = 2ull * g->ssm_groups * g->ssm_state + (uint64_t)g->ssm_inner, in = (uint64_t)g->ssm_inner;
  const uint64_t q = (uint64_t)g->n_heads * g->head_dim, kv = (uint64_t)g->n_kv_heads * g->head_dim;
  const uint64_t vh = (uint64_t)g->ssm_v_heads, ff = (uint64_t)g->ff_exp, sff = (uint64_t)g->ff_shexp;
  /* the experts' rows: a slot each, and the rows a group reads past its last (mv_group_tok's); their entries' bound */
  const uint64_t slots = n * k + MV_MAX_TOKENS, entries = 2 * n * k;
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
      {&b->xe, slots * D * F},      {&b->ga, slots * ff * F},     {&b->up, slots * ff * F},     {&b->y, slots * D * F},
      {&b->shg, n * sff * F},       {&b->shu, n * sff * F},       {&b->sh, n * D * F},          {&b->sl, n * F},
      {&b->ids, n * k * 4},         {&b->rows, n * k * 4},        {&b->cnt, entries * 4},       {&b->row, entries * 4},
      {&b->etok, n * k * 4},        {&b->where, n * k * 4},
      {&b->ptr, Q4X_GROUP_TENSORS * entries * sizeof(void *)},  {&b->emb, n * D * F},    {&b->key, n * hd * F},        {&b->val, n * D * F},
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
  free_table(f->w->model, f->dev);
  if (f->mtp)
    free_table(f->mtp->model, f->mtp_dev);
  Q4xMtpBuf *m = &f->m;
  hipFree(m->hid), hipFree(m->hin), hipFree(m->emb), hipFree(m->hn), hipFree(m->join), hipFree(m->logits);
  hipFree(m->top_id), hipFree(m->top_p), hipHostFree(m->pin_id), hipHostFree(m->pin_p);
  void **bufs = (void **)&f->b;
  for (size_t i = 0; i < offsetof(Q4xBuf, pin_rows) / sizeof(void *); i++)
    hipFree(bufs[i]);
  hipHostFree(f->b.pin_rows), hipHostFree(f->b.pin_tok), hipFree(f->b.logits);
  free(f);
}

Q4xForward *q4x_forward_create(const Q4xWeights *w, const Q4xMtp *mtp, uint32_t max_ctx, uint32_t max_tokens,
                               char err[Q4X_FWD_ERR])
{
  if (!w || !max_ctx || max_ctx > OPS_Q4X_MAX_KEYS || !max_tokens || max_tokens > max_ctx)
    return q4x_fail(err, "no weights, or a context past the dense attention's reach, or no tokens a step"), NULL;
  Q4xForward *f = calloc(1, sizeof *f);
  if (!f)
    return q4x_fail(err, "out of memory"), NULL;
  f->w = w, f->mtp = mtp, f->max_ctx = max_ctx, f->max_tokens = max_tokens;
  if (!upload(w->model, &f->dev, err) || !buffers(f, err) ||
      (mtp && (!upload(mtp->model, &f->mtp_dev, err) || !mtp_buffers(f, err))))
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
  hipFree(s->ple_hist), hipFree(s->mkc), hipFree(s->mvc), hipFree(s->pend);
  free(s->step_tok);
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
  bool ok = alloc(&s->ple_hist, Q4X_SLOTS * sb.ple);
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = g->recurrent[l] ? alloc(&s->ssm[l], Q4X_SLOTS * sb.ssm) && alloc(&s->conv[l], Q4X_SLOTS * sb.conv)
                         : alloc(&s->kc[l], f->max_ctx * sb.kv_row) && alloc(&s->vc[l], f->max_ctx * sb.kv_row);
  if (ok && f->mtp) /* the head's cache, its pending h, its last step's tokens */
    ok = alloc(&s->mkc, f->max_ctx * sb.kv_row) && alloc(&s->mvc, f->max_ctx * sb.kv_row) &&
         alloc(&s->pend, (uint64_t)g->hc * g->n_embd * F) && (s->step_tok = calloc(f->max_tokens, 4));
  if (!ok || !q4x_seq_reset(s))
    return q4x_seq_destroy(s), q4x_fail(err, "GPU memory refused (a sequence's state)"), NULL;
  return s;
}

int32_t q4x_seq_position(const Q4xSeq *s) { return s ? s->pos : -1; }
