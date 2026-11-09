#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "osmos/gpu/mm.h"
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

/* a tensor's device bytes: the model's, the overlay's, or the MTP head's (each file's own table) */
const void *q4x_dev(const Q4xForward *f, const GgufTensor *t)
{
  const Gguf *m = f->w->model, *o = f->w->overlay;
  if (t >= m->tensors && t < m->tensors + m->n_tensors)
    return f->dev[t - m->tensors];
  if (o && t >= o->tensors && t < o->tensors + o->n_tensors)
    return f->ov_dev[t - o->tensors];
  return f->mtp_dev[t - f->mtp->model->tensors];
}

void q4x_seen(const Q4xForward *f, const char *step, int layer, float *dev, uint64_t n)
{
  if (f->probe && hipDeviceSynchronize() == hipSuccess)
    f->probe(step, layer, dev, n, f->probe_arg);
}

static bool alloc(void *p, uint64_t bytes) { return hipMalloc((void **)p, bytes ? bytes : 1) == hipSuccess; }

/* every tensor of g read into GPU memory through a pinned piece: *dev its table - but those `shadow` has a tensor of
   the name of (an overlay's stand-ins: their bytes never read) */
static bool upload(const Gguf *g, const Gguf *shadow, void ***dev, char *err)
{
  void *pin = NULL;
  if (!(*dev = calloc(g->n_tensors, sizeof **dev)) || hipHostMalloc(&pin, READ_PIECE, 0) != hipSuccess)
    return q4x_fail(err, "out of memory (the weights' table, or a pinned piece)");
  bool ok = true;
  for (size_t i = 0; ok && i < g->n_tensors; i++)
  {
    const GgufTensor *t = &g->tensors[i];
    if (shadow && gguf_tensor(shadow, t->name))
      continue;
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
  /* a draft step's rows: a sequence each (q4x_mtp_draft_all), over the shortlist */
  const uint64_t rows = n < Q4X_MAX_RUNS ? n : Q4X_MAX_RUNS, dv = Q4X_MTP_VOCAB < g->n_vocab ? Q4X_MTP_VOCAB : g->n_vocab;
  Q4xMtpBuf *m = &f->m;
  const bool ok = alloc(&m->hid, n * hd * F) && alloc(&m->hin, n * hd * F) && alloc(&m->emb, n * D * F) &&
                  alloc(&m->hn, n * hd * F) && alloc(&m->join, 2 * n * hd * F) &&
                  alloc(&m->logits, (uint64_t)rows * dv * F) && alloc(&m->top_id, rows * 4) &&
                  alloc(&m->top_p, rows * 4) && hipHostMalloc((void **)&m->pin_id, rows * 4, 0) == hipSuccess &&
                  hipHostMalloc((void **)&m->pin_p, rows * 4, 0) == hipSuccess;
  return ok || q4x_fail(err, "GPU memory refused (the MTP head's buffers)");
}

static bool buffers(Q4xForward *f, char *err)
{
  const Q4xGeometry *g = &f->w->geo;
  const uint64_t n = f->max_tokens, D = (uint64_t)g->n_embd, hd = (uint64_t)g->hc * D, k = (uint64_t)g->n_used;
  const uint64_t ch = 2ull * g->ssm_groups * g->ssm_state + (uint64_t)g->ssm_inner, in = (uint64_t)g->ssm_inner;
  const uint64_t q = (uint64_t)g->n_heads * g->head_dim, kv = (uint64_t)g->n_kv_heads * g->head_dim;
  const uint64_t vh = (uint64_t)g->ssm_v_heads, ff = (uint64_t)g->ff_exp, sff = (uint64_t)g->ff_shexp;
  /* QSA's indexer: its heads' q, a token's blocks at most */
  const uint32_t ratio = (uint32_t)g->ix_ratio;
  const uint64_t ixq = (uint64_t)g->ix_heads * Q4X_IX_DIM, blocks = ratio ? (uint64_t)g->ix_top_k / ratio : 1;
  /* the experts: mm_group's entries and slots (q4x_expert_bounds'); mm's f16 activations, a chunk
     of MM_MAX_TOKENS rows of the widest input (hc n_embd) */
  Q4xExpertBounds bd;
  q4x_expert_bounds((uint32_t)(n * k), (uint32_t)g->n_experts, &bd);
  const uint64_t entries = bd.entries, slots = bd.slots + 8 * MM_TILE; /* past the last: a part's tiles read whole */
  const uint64_t x16 = (uint64_t)MM_MAX_TOKENS * hd * 2, xe16 = slots * (D > ff ? D : ff) * 2;
  Q4xBuf *b = &f->b;
  const struct
  {
    void *p;
    uint64_t bytes;
    int per_token; /* n rows, a token's each (a sub-batch's view moves it) */
  } want[] = {
      {&b->h, n * hd * F, 1},          {&b->xn, n * hd * F, 1},         {&b->lo, n * g->hc_rank * F, 1}, {&b->gate, n * hd * F, 1},
      {&b->mixed, n * D * F, 1},       {&b->inj, n * g->hc * F, 1},     {&b->out, n * D * F, 1},         {&b->x, n * D * F, 1},
      {&b->qkv, n * ch * F, 1},        {&b->z, n * in * F, 1},          {&b->al, n * vh * F, 1},         {&b->be, n * vh * F, 1},
      {&b->g, n * vh * F, 1},          {&b->b, n * vh * F, 1},          {&b->cv, n * ch * F, 1},         {&b->qkn, n * 2 * g->ssm_groups * g->ssm_state * F, 1},         {&b->ro, n * in * F, 1},
      {&b->fo, n * in * F, 1},         {&b->qf, n * 2 * q * F, 1},      {&b->kr, n * kv * F, 1},         {&b->kq, n * kv * F, 1},         {&b->vr, n * kv * F, 1},         {&b->q, n * q * F, 1},
      {&b->iqr, n * ixq * F, 1},       {&b->iq, n * ixq * F, 1},        {&b->ikr, n * Q4X_IX_DIM * F, 1},
      {&b->ixs, q4x_ix_scratch(f->max_ctx, ratio) * F, 0},           {&b->sel, n * blocks * 4, 1},    {&b->npick, n * 4, 1},
      {&b->gt, n * q * F, 1},          {&b->ao, n * q * F, 1},          {&b->fa, q4x_attn_scratch((uint32_t)g->n_heads, f->max_ctx) * F, 0},          {&b->rl, n * g->n_experts * F, 1}, {&b->w, n * k * F, 1},
      {&b->ga, n * k * ff * F, 1},     {&b->up, n * k * ff * F, 1},     {&b->y, n * k * D * F, 1},
      {&b->shg, n * sff * F, 1},       {&b->shu, n * sff * F, 1},       {&b->sh, n * D * F, 1},          {&b->sl, n * F, 1},
      {&b->ids, n * k * 4, 1},         {&b->rows, n * k * 4, 1},        {&b->cnt, entries * 4, 0},       {&b->gidx, slots * 4, 0},
      {&b->hidx, slots * 4, 0},        {&b->eoff, 3ull * g->n_experts * 4, 0},        {&b->where, n * k * 4, 1},       {&b->xg, entries * 4, 0},        {&b->yg, entries * 4, 0},
      {&b->xd, entries * 4, 0},        {&b->yd, entries * 4, 0},        {&b->x16, x16, 0},               {&b->xe16, xe16, 0},
      {&b->overflow, 4, 0},
      {&b->ptr, Q4X_GROUP_TENSORS * entries * sizeof(void *), 0},  {&b->emb, n * D * F, 1},    {&b->key, n * hd * F, 1},        {&b->val, n * D * F, 1},
      {&b->kn, n * hd * F, 1},         {&b->qn, n * hd * F, 1},         {&b->pg, n * g->hc * F, 1},      {&b->gated, n * hd * F, 1},
      {&b->nrm, n * hd * F, 1},        {&b->conv, n * hd * F, 1},       {&b->ple_rows, n * g->ple_heads * 4, 1},
      {&b->tok, n * 4, 1},             {&b->top_part, q4x_argmax_scratch((uint32_t)n), 0},
      {&b->logits, n * (uint64_t)g->n_vocab * F, 1},
  };
  for (size_t i = 0; i < sizeof want / sizeof *want; i++)
    if (!alloc(want[i].p, want[i].bytes))
      return q4x_fail(err, "GPU memory refused (a step's buffers)");
  _Static_assert(sizeof want / sizeof *want <= Q4X_ROW_BUFS, "Q4X_ROW_BUFS: the table's");
  f->n_rows = 0;
  for (size_t i = 0; i < sizeof want / sizeof *want; i++)
    if (want[i].per_token)
      f->rows[f->n_rows].p = want[i].p, f->rows[f->n_rows++].row = want[i].bytes / n;
  if (hipMemset(b->overflow, 0, 4) != hipSuccess)
    return q4x_fail(err, "a GPU call failed (the f16 overflow flag)");
  return (hipHostMalloc((void **)&b->pin_rows, n * g->ple_heads * 4, 0) == hipSuccess &&
          hipHostMalloc((void **)&b->pin_tok, n * 4, 0) == hipSuccess &&
          hipHostMalloc((void **)&b->pin_overflow, 4, 0) == hipSuccess && !(*b->pin_overflow = 0)) ||
         q4x_fail(err, "pinned memory refused (a step's tokens)");
}

void q4x_forward_destroy(Q4xForward *f)
{
  if (!f)
    return;
  free_table(f->w->model, f->dev);
  if (f->w->overlay)
    free_table(f->w->overlay, f->ov_dev);
  if (f->mtp)
    free_table(f->mtp->model, f->mtp_dev);
  Q4xMtpBuf *m = &f->m;
  hipFree(m->hid), hipFree(m->hin), hipFree(m->emb), hipFree(m->hn), hipFree(m->join), hipFree(m->logits);
  hipFree(m->top_id), hipFree(m->top_p), hipHostFree(m->pin_id), hipHostFree(m->pin_p);
  void **bufs = (void **)&f->b;
  for (size_t i = 0; i < offsetof(Q4xBuf, pin_rows) / sizeof(void *); i++)
    hipFree(bufs[i]);
  hipHostFree(f->b.pin_rows), hipHostFree(f->b.pin_tok), hipHostFree(f->b.pin_overflow), hipFree(f->b.logits);
  if (f->s2)
    (void)hipStreamDestroy(f->s2);
  if (f->e_in)
    (void)hipEventDestroy(f->e_in);
  if (f->e_out)
    (void)hipEventDestroy(f->e_out);
  if (f->pin_free)
    (void)hipEventDestroy(f->pin_free);
  if (f->e_mix)
    (void)hipEventDestroy(f->e_mix);
  if (f->e_inj)
    (void)hipEventDestroy(f->e_inj);
  if (f->e_gates)
    (void)hipEventDestroy(f->e_gates);
  if (f->e_ix)
    (void)hipEventDestroy(f->e_ix);
  free(f->sub_seq);
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
  if (hipStreamCreateWithFlags(&f->s2, hipStreamNonBlocking) != hipSuccess ||
      hipEventCreateWithFlags(&f->e_in, hipEventDisableTiming) != hipSuccess ||
      hipEventCreateWithFlags(&f->e_out, hipEventDisableTiming) != hipSuccess ||
      hipEventCreateWithFlags(&f->pin_free, hipEventDisableTiming) != hipSuccess ||
      hipEventCreateWithFlags(&f->e_mix, hipEventDisableTiming) != hipSuccess ||
      hipEventCreateWithFlags(&f->e_inj, hipEventDisableTiming) != hipSuccess ||
      hipEventCreateWithFlags(&f->e_gates, hipEventDisableTiming) != hipSuccess ||
      hipEventCreateWithFlags(&f->e_ix, hipEventDisableTiming) != hipSuccess)
    return q4x_fail(err, "a stream or event refused"), free(f), NULL;
  if (!(f->sub_seq = calloc(Q4X_MAX_RUNS, sizeof *f->sub_seq)))
    return q4x_fail(err, "out of memory"), free(f), NULL;
  if (!q4x_rope((uint32_t)w->geo.rope_dim, w->geo.rope_base, w->geo.native_ctx, max_ctx, &f->rope))
    return q4x_fail(err, "no rotation for this context (q4x_rope: past 64 times the native)"), free(f->sub_seq),
           free(f), NULL;
  if (!upload(w->model, w->overlay, &f->dev, err) || (w->overlay && !upload(w->overlay, NULL, &f->ov_dev, err)) ||
      !buffers(f, err) || (mtp && (!upload(mtp->model, NULL, &f->mtp_dev, err) || !mtp_buffers(f, err))))
    return q4x_forward_destroy(f), NULL;
  return f;
}

void q4x_forward_side_spin(Q4xForward *f, uint32_t us)
{
  if (f)
    f->side_spin = us;
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
                       q4x_kv_row_bytes((uint32_t)(g->n_kv_heads * g->head_dim)),
                       (uint64_t)(g->ple_conv - 1) * g->ple_ngram * g->hc * g->n_embd * F, Q4X_IX_DIM * F,
                       (uint64_t)Q4X_IX_RING(g->ix_ratio) * Q4X_IX_DIM * F};
}

void q4x_seq_destroy(Q4xSeq *s)
{
  if (!s)
    return;
  for (int l = 0; l < Q4X_MAX_LAYERS; l++)
    hipFree(s->ssm[l]), hipFree(s->conv[l]), hipFree(s->kc[l]), hipFree(s->vc[l]), hipFree(s->ixk[l]),
        hipFree(s->ixr[l]);
  hipFree(s->ple_hist), hipFree(s->mkc), hipFree(s->mvc), hipFree(s->mixk), hipFree(s->mixr), hipFree(s->pend);
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
  const uint64_t pools = f->max_ctx / (uint64_t)g->ix_ratio + 1; /* an attention's pooled keys: its blocks */
  bool ok = alloc(&s->ple_hist, Q4X_SLOTS * sb.ple);
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = g->recurrent[l] ? alloc(&s->ssm[l], Q4X_SLOTS * sb.ssm) && alloc(&s->conv[l], Q4X_SLOTS * sb.conv)
                         : alloc(&s->kc[l], f->max_ctx * sb.kv_row) && alloc(&s->vc[l], f->max_ctx * sb.kv_row) &&
                               alloc(&s->ixk[l], pools * sb.ix_block) && alloc(&s->ixr[l], sb.ix_ring);
  if (ok && f->mtp) /* the head's cache and indexer, its pending h, its last step's tokens */
    ok = alloc(&s->mkc, f->max_ctx * sb.kv_row) && alloc(&s->mvc, f->max_ctx * sb.kv_row) &&
         alloc(&s->mixk, pools * sb.ix_block) && alloc(&s->mixr, sb.ix_ring) &&
         alloc(&s->pend, (uint64_t)g->hc * g->n_embd * F) && (s->step_tok = calloc(f->max_tokens, 4));
  if (!ok || !q4x_seq_reset(s))
    return q4x_seq_destroy(s), q4x_fail(err, "GPU memory refused (a sequence's state)"), NULL;
  return s;
}

int32_t q4x_seq_position(const Q4xSeq *s) { return s ? s->pos : -1; }
