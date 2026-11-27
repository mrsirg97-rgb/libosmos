/* Qwen's forward, its weights made resident (q4x_impl.h): every tensor's bytes on the device (read through a
   pinned piece) or the membrane's, the room left as it goes, the experts' pool, the step's and drafters' buffers */
#define _GNU_SOURCE /* O_DIRECT */
#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "osmos/gpu/upload.h"
#include "osmos/gpu/mm.h"
#include "osmos/mem/arena.h"
#include "osmos/mem/host_slots.h"
#include "osmos/mem/experts.h"
#include "osmos/mem/membrane.h"
#include "osmos/run/fit.h"
#include "osmos/gpu/mv.h"
#include "q4x_impl.h"
#include "osmos/err.h"


#define READ_PIECE (256ull << 20) /* a read's bytes: under a read's ~2 GB, the staging buffer's size */
#define F sizeof(float)

/* a tensor's device bytes: the model's, the overlay's, the MTP head's or the drafter's (each file's own table) */
const void *q4x_dev(const Q4xForward *f, const GgufTensor *t)
{
  const Gguf *m = f->w->model, *o = f->w->overlay;
  if (t >= m->tensors && t < m->tensors + m->n_tensors)
    return f->dev[t - m->tensors];
  if (o && t >= o->tensors && t < o->tensors + o->n_tensors)
    return f->ov_dev[t - o->tensors];
  const Gguf *d = f->df ? f->df->model : NULL;
  if (d && t >= d->tensors && t < d->tensors + d->n_tensors)
    return f->df_dev[t - d->tensors];
  return f->mtp_dev[t - f->mtp->model->tensors];
}

void q4x_seen(const Q4xForward *f, const char *step, int layer, float *dev, uint64_t n)
{
  if (f->probe && hipDeviceSynchronize() == hipSuccess)
    f->probe(step, layer, dev, n, f->probe_arg);
}

/* a GPU buffer of the forward's (its ledger's, recorded: freed at its destroy), and a pinned host one alike */
static bool alloc(Q4xForward *f, void *p, uint64_t bytes) { return gpu_bufs_one(&f->bufs, f->ledger, p, bytes); }
static bool pinned(Q4xForward *f, void *p, uint64_t bytes) { return gpu_bufs_host(&f->bufs, f->ledger, p, bytes); }
/* a sequence's (its own record, its forward's ledger's) */
bool q4x_seq_alloc(Q4xSeq *s, void *p, uint64_t bytes) { return gpu_bufs_one(&s->bufs, s->f->ledger, p, bytes); }

/* the memory a build can still take (fit.h's fit_room): the GPU's free, or the system's RAM available past what a
   build leaves the system (FIT_RAM_FREE, the headroom added back) where that is less - the experts lie in host memory
   the GPU reads in place (the membrane's), which the GPU's own count does not hold, and on this machine the GPU's
   memory is the system's. so FIT_HEADROOM of it left is the GPU's headroom and the system's RAM both */
static bool mem_left(uint64_t *left)
{
  size_t free_b = 0, total = 0;
  if (hipMemGetInfo(&free_b, &total) != hipSuccess)
    return false;
  *left = fit_room((uint64_t)free_b, fit_ram_available(), FIT_RAM_FREE);
  return true;
}

/* the memory left after a build (the forward's, a sequence's) at least FIT_HEADROOM: the runtime loads a kernel's
   code at its first launch and takes the memory then (with none left, a segfault in hipLaunchKernel) - and a system
   with no RAM left kills the process. `more`: bytes about to be taken (0: the build done) - refused before they are,
   so a build that cannot fit takes none of it. false with why */
bool q4x_room_left_past(Q4xForward *f, uint64_t more, const char *after, char *err)
{
  uint64_t left = 0;
  if (f->ledger) /* the engine's: its room, --fill's RAM kept */
    left = ledger_left_settled(f->ledger) + FIT_HEADROOM;
  else if (!mem_left(&left))
    return osmos_fail(err, "a GPU call failed (the memory left)");
  if (left >= more && left - more >= FIT_HEADROOM)
    return true;
  if (err)
    snprintf(err, OSMOS_ERR,
             "GPU memory: %.2f GiB left after %s, under the %.0f GiB its kernels need loaded as they first run (the "
             "GPU's memory, or the system's RAM past what --fill keeps): stream the n-gram table, or a smaller max_tokens "
             "or max_context",
             left >= more ? (double)(left - more) / (1ull << 30) : -(double)(more - left) / (1ull << 30), after,
             (double)FIT_HEADROOM / (1ull << 30));
  return false;
}

bool q4x_room_left(Q4xForward *f, const char *after, char *err) { return q4x_room_left_past(f, 0, after, err); }

/* every tensor of g read into GPU memory through a pinned piece: *dev its table - but those `shadow` has a tensor of
   the name of (an overlay's stand-ins: their bytes never read), and `skip` (a table read from its file: stream_ngram) */
/* a tensor the experts' membrane reads (the forward's mem): a layer's exp_gate, exp_up or exp_down - not uploaded */
static bool on_membrane(const Q4xForward *f, const GgufTensor *t)
{
  for (int l = 0; f && f->exp_units && l < f->w->geo.n_layers; l++)
    if (t == f->w->layer[l].exp_gate || t == f->w->layer[l].exp_up || t == f->w->layer[l].exp_down)
      return true;
  return false;
}

/* a promotion's copy, the stream's slot to the pool's (both pinned host memory the GPU reads in place) */
static bool host_copy(void *dst, const void *src, uint64_t bytes, void *ctx)
{
  (void)ctx;
  memcpy(dst, src, bytes);
  return true;
}

/* the experts as a membrane's units, sized to the budget (experts_plan: 0 every one, all read at once into memory the
   GPU reads in place; else what the rest of the build leaves, the warm set an expert of every layer at a time), their
   files O_DIRECT; every one resident: the device table of their addresses (q4x_expert_tiles'). nothing where the
   experts are not units (an overlay's expert tensor: uploaded with the rest) */
bool q4x_experts(Q4xForward *f, uint64_t budget, char *err)
{
  const Q4xWeights *w = f->w;
  const Gguf *m = w->model;
  const uint32_t E = (uint32_t)w->geo.n_experts, L = (uint32_t)w->geo.n_layers, K = (uint32_t)w->geo.n_used;
  const size_t n = q4x_weights_experts(w, NULL, NULL, 0);
  if (!n)
    return true;
  PoolUnit *units = calloc(n, sizeof *units);
  uint32_t *warm = malloc(n * sizeof *warm);
  bool ok = units && warm && (f->exp_fds = malloc((size_t)m->n_files * sizeof *f->exp_fds));
  for (int i = 0; ok && i < m->n_files; i++)
    f->exp_fds[i] = -1;
  for (int i = 0; ok && i < m->n_files; i++)
    ok = (f->exp_fds[i] = open(m->paths[i], O_RDONLY | O_DIRECT)) >= 0;
  if (!ok)
    osmos_fail(err, "out of memory, or a model file not opened O_DIRECT (the experts' reads)");
  ok = ok && q4x_weights_experts(w, f->exp_fds, units, n) == n;
  for (size_t u = 0; ok && u < n; u++) /* expert 0 of every layer, then 1, ... */
    warm[u] = (uint32_t)((u % L) * E + u / L);
  ExpertsPlan plan;
  Experts ex;
  char merr[OSMOS_ERR] = "";
  if (ok && !experts_plan(units, L, E, warm, 0, budget, f->fixed_bytes, ARENA_GRAIN, &plan))
    ok = osmos_fail(err, "the experts' plan refused");
  if (ok && !experts_create(f->ledger, units, L, E, warm, &plan, ARENA_GRAIN, Q4X_EXPERT_READERS, host_copy, NULL, 0, true, &ex,
                            merr))
    ok = osmos_fail(err, "%s", merr[0] ? merr : "the experts' membrane refused");
  if (ok)
    f->mem = ex.mem, f->exp_slots = ex.slots, f->exp_stream = ex.stream,
    f->pinned_bytes = host_slots_bytes(ex.slots) + host_slots_bytes(ex.stream);
  if (ok && membrane_whole(f->mem))
  {
    const void **tab = malloc(n * Q4X_GROUP_TENSORS * sizeof *tab);
    ok = tab != NULL;
    for (size_t u = 0; ok && u < n; u++) /* [layer][tensor][expert]: a layer's tensor's E addresses side by side */
    {
      const void *dev[POOL_PARTS];
      ok = membrane_ready(f->mem, (uint32_t)u, dev) || osmos_fail(err, "an expert not in memory");
      for (int k = 0; ok && k < Q4X_GROUP_TENSORS; k++)
        tab[((u / E) * Q4X_GROUP_TENSORS + (size_t)k) * E + u % E] = dev[k];
    }
    ok = ok && ((alloc(f, &f->exp_tab, n * Q4X_GROUP_TENSORS * sizeof *tab) &&
                 hipMemcpy(f->exp_tab, tab, n * Q4X_GROUP_TENSORS * sizeof *tab, hipMemcpyHostToDevice) == hipSuccess) ||
                osmos_fail(err, "GPU memory refused (the experts' table)"));
    free(tab);
  }
  else if (ok) /* streamed: a step's tables the host's (q4x_moe) */
  {
    const uint64_t P = (uint64_t)f->max_tokens * K;
    Q4xExpertBounds bd;
    q4x_expert_bounds((uint32_t)P, E, &bd);
    ok = (f->h_ex = malloc(E * sizeof *f->h_ex)) && (f->h_by = malloc(P * 4)) && (f->h_row = malloc(P * 4)) &&
         (f->h_run = malloc(4 * (size_t)E * 4)) && (f->h_est = malloc(((size_t)E + 1) * 4)) &&
         (f->h_ids = malloc(P * 4)) && (f->h_gidx = malloc(bd.slots * 4)) && (f->h_hidx = malloc(bd.slots * 4)) &&
         (f->h_cnt = malloc(bd.entries * 4)) && (f->h_xg = malloc(bd.entries * 4)) && (f->h_yg = malloc(bd.entries * 4)) &&
         (f->h_xd = malloc(bd.entries * 4)) && (f->h_yd = malloc(bd.entries * 4)) &&
         (f->h_ptr = malloc(Q4X_GROUP_TENSORS * (size_t)bd.entries * sizeof *f->h_ptr));
    if (!ok)
      osmos_fail(err, "out of memory (the experts' host tables)");
  }
  free(units), free(warm);
  return ok;
}

/* a tensor not read: one the overlay stands in for, the one skipped, or one on the experts' membrane */
typedef struct UploadSkip
{
  const Q4xForward *f;
  const Gguf *shadow;
  const GgufTensor *skip;
} UploadSkip;

static bool not_uploaded(const GgufTensor *t, void *arg)
{
  const UploadSkip *u = arg;
  return (u->shadow && gguf_tensor(u->shadow, t->name)) || t == u->skip || on_membrane(u->f, t);
}

/* f: whose experts the membrane reads (NULL: a file of none - the overlay's, the heads'); l: the forward's ledger */
bool q4x_upload(OsmosLedger *l, const Q4xForward *f, const Gguf *g, const Gguf *shadow, const GgufTensor *skip,
                   void ***dev, char *err)
{
  UploadSkip u = {f, shadow, skip};
  return gpu_upload_table(l, g, READ_PIECE, not_uploaded, &u, dev, err);
}

void q4x_free_table(Q4xForward *f, const Gguf *g, void **dev) { gpu_free_table(f->ledger, g, dev, NULL); }

bool q4x_mtp_buffers(Q4xForward *f, char *err)
{
  const Q4xGeometry *g = &f->w->geo;
  const uint64_t n = f->max_tokens, D = (uint64_t)g->n_embd, hd = (uint64_t)g->hc * D;
  /* a draft step's rows: a sequence each (q4x_mtp_draft_all), over the shortlist */
  const uint64_t rows = n < Q4X_MAX_RUNS ? n : Q4X_MAX_RUNS, dv = Q4X_MTP_VOCAB < g->n_vocab ? Q4X_MTP_VOCAB : g->n_vocab;
  Q4xMtpBuf *m = &f->m;
  const bool ok = alloc(f, &m->hid, n * hd * F) && alloc(f, &m->hin, n * hd * F) && alloc(f, &m->emb, n * D * F) &&
                  alloc(f, &m->hn, n * hd * F) && alloc(f, &m->join, 2 * n * hd * F) &&
                  alloc(f, &m->logits, (uint64_t)rows * dv * F) && alloc(f, &m->top_id, rows * 4) &&
                  alloc(f, &m->top_p, rows * 4) && pinned(f, &m->pin_id, rows * 4) && pinned(f, &m->pin_p, rows * 4);
  return ok || osmos_fail(err, "GPU memory refused (the MTP head's buffers)");
}


/* the drafter's buffers (the DFlash2 core's): a feed's max_tokens context rows, a block's */
bool q4x_df_buffers(Q4xForward *f, char *err)
{
  return df_buffers(&f->d, f->ledger, &f->df->geo, f->max_tokens, f->max_ctx, f->df_ring) ||
         osmos_fail(err, "GPU memory refused (the drafter's buffers)");
}

/* a row of the n-gram table: its bytes in the file (the held table's row stride) */
static uint64_t ple_row_bytes(const Q4xForward *f) { return f->w->ple_table->bytes / f->w->ple_table->dims[1]; }

/* stream_ngram's: the table's file, its readers, the pinned stage of a step's rows and their offsets */
bool q4x_ple_stream(Q4xForward *f, char *err)
{
  const GgufTensor *t = f->w->ple_table;
  const uint64_t rows = (uint64_t)f->max_tokens * f->w->geo.ple_heads;
  f->ple_fd = open(f->w->model->paths[t->file], O_RDONLY);
  if (f->ple_fd < 0 || !(f->ple_read = rowread_create(Q4X_ROW_THREADS)) || !(f->ple_off = malloc(rows * 8)) ||
      !pinned(f, &f->ple_stage, rows * ple_row_bytes(f)) ||
      hipEventCreateWithFlags(&f->ple_copied, hipEventDisableTiming) != hipSuccess)
    return osmos_fail(err, "the n-gram table's file, its readers or their stage refused (stream_ngram)");
  return true;
}

/* the attention's core: Qwen's heads block (a token's query heads, its QSA picks, chunks of 128 merged max-first:
   q4x_attn's bits) */
static AttnCoreDesc attn_desc(const Q4xForward *f)
{
  const Q4xGeometry *g = &f->w->geo;
  return (AttnCoreDesc){.heads = (uint32_t)g->n_heads,
                        .kv_heads = (uint32_t)g->n_kv_heads,
                        .dk = (uint32_t)g->head_dim,
                        .dv = (uint32_t)g->head_dim,
                        .k_rows = ATTN_ROWS_Q8_0,
                        .v_rows = ATTN_ROWS_Q8_0,
                        .scaling = 1.0f / sqrtf((float)g->head_dim),
                        .term = ATTN_TERM_NONE,
                        .gate = ATTN_GATE_VALUE,
                        .block = ATTN_BLOCK_HEADS,
                        .ratio = (uint32_t)g->ix_ratio,
                        .unit = 128,
                        .max_tokens = f->max_tokens,
                        .max_keys = f->max_ctx};
}

static uint64_t attn_scratch_bytes(const Q4xForward *f)
{
  const AttnCoreDesc d = attn_desc(f);
  return attn_core_scratch(&d);
}

bool q4x_buffers(Q4xForward *f, char *err)
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
  const uint64_t dk = (uint64_t)MV_MAX_TOKENS * k; /* a decode's picks at most (the int8 experts' tables) */
  const uint64_t wide = (uint64_t)f->n_taps * D > hd ? (uint64_t)f->n_taps * D : hd; /* the drafter's fc's columns */
  const uint64_t x16 = (uint64_t)MM_MAX_TOKENS * wide * 2, xe16 = slots * (D > ff ? D : ff) * 2;
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
      {&b->gt, n * q * F, 1},          {&b->ao, n * q * F, 1},          {&b->fa, attn_scratch_bytes(f), 0},          {&b->rl, n * g->n_experts * F, 1}, {&b->w, n * k * F, 1},
      {&b->ga, n * k * ff * F, 1},     {&b->up, n * k * ff * F, 1},     {&b->y, n * k * D * F, 1},
      {&b->shg, n * sff * F, 1},       {&b->shu, n * sff * F, 1},       {&b->sh, n * D * F, 1},          {&b->sl, n * F, 1},
      {&b->ids, n * k * 4, 1},         {&b->rows, n * k * 4, 1},        {&b->cnt, entries * 4, 0},       {&b->gidx, slots * 4, 0},
      {&b->hidx, slots * 4, 0},        {&b->eoff, 3ull * g->n_experts * 4, 0},        {&b->where, n * k * 4, 1},       {&b->xg, entries * 4, 0},        {&b->yg, entries * 4, 0},
      {&b->xd, entries * 4, 0},        {&b->yd, entries * 4, 0},        {&b->x16, x16, 0},               {&b->xe16, xe16, 0},
      {&b->overflow, 4, 0},
      {&b->q_gidx, dk * 4, 0},         {&b->q_sidx, dk * 4, 0},         {&b->q_cnt, dk * 4, 0},          {&b->q_xrow, dk * 4, 0},
      {&b->q_wsel, dk * F, 0},         {&b->q_wptr, 3 * dk * sizeof(void *), 0},
      {&b->q_x, MV_MAX_TOKENS * D, 0}, {&b->q_xd, MV_MAX_TOKENS * (D / 32) * F, 0},
      {&b->q_h, dk * ff, 0},           {&b->q_hd, dk * (ff / 32) * F, 0},
      {&b->x16b, (uint64_t)MM_SILU_MAX_TOKENS * g->hc_rank * 2, 0},
      {&b->ptr, Q4X_GROUP_TENSORS * entries * sizeof(void *), 0},  {&b->emb, n * D * F, 1},    {&b->key, n * hd * F, 1},        {&b->val, n * D * F, 1},
      {&b->kn, n * hd * F, 1},         {&b->qn, n * hd * F, 1},         {&b->pg, n * g->hc * F, 1},      {&b->gated, n * hd * F, 1},
      {&b->nrm, n * hd * F, 1},        {&b->conv, n * hd * F, 1},       {&b->ple_rows, n * g->ple_heads * 4, 1},
      {&b->ple_sbuf, n * g->ple_heads * ple_row_bytes(f), 1},          {&b->ple_iota, n * g->ple_heads * 4, 0},
      {&b->taps, n * f->n_taps * D * F, 0},
      {&b->tok, n * 4, 1},             {&b->top_part, ops_argmax_scratch((uint32_t)n), 0},
      {&b->logits, n * (uint64_t)g->n_vocab * F, 1},
  };
  /* b.xe16 also holds a mixer's up's f16 input (hip_q4x_layers.c down_up: a step's tokens x hc_rank, whole tiles) */
  if (xe16 < (n + MM_TILE - 1) / MM_TILE * MM_TILE * (uint64_t)g->hc_rank * 2)
    return osmos_fail(err, "the experts' f16 buffer smaller than a mixer's up's input (hc_rank past the experts' dims)");
  for (size_t i = 0; i < sizeof want / sizeof *want; i++)
    if (!alloc(f, want[i].p, want[i].bytes))
      return osmos_fail(err, "GPU memory refused (a step's buffers)");
  _Static_assert(sizeof want / sizeof *want <= Q4X_ROW_BUFS, "Q4X_ROW_BUFS: the table's");
  f->n_rows = 0;
  for (size_t i = 0; i < sizeof want / sizeof *want; i++)
    if (want[i].per_token)
      f->rows[f->n_rows].p = want[i].p, f->rows[f->n_rows++].row = want[i].bytes / n;
  f->ovf = (OverflowFlag){b->overflow, 4, false, NULL};
  if (!overflow_reset(&f->ovf, err))
    return false;
  int32_t *iota = malloc(n * g->ple_heads * 4);
  for (uint64_t i = 0; iota && i < n * g->ple_heads; i++)
    iota[i] = (int32_t)i;
  const bool put = iota && hipMemcpy(b->ple_iota, iota, n * g->ple_heads * 4, hipMemcpyHostToDevice) == hipSuccess;
  free(iota);
  if (!put)
    return osmos_fail(err, "a GPU call failed (the n-gram rows' places)");
  /* the attention's core: Qwen's heads block (a token's query heads, its QSA picks, chunks of 128 merged max-first:
     q4x_attn's bits), over b.fa */
  const AttnCoreDesc ad = attn_desc(f);
  if (!(f->attn = attn_core_create(&ad, b->fa, attn_core_scratch(&ad), err)))
    return false;
  return (pinned(f, &b->pin_rows, n * g->ple_heads * 4) && pinned(f, &b->pin_tok, n * 4) &&
          pinned(f, &b->pin_overflow, 4) && !(*b->pin_overflow = 0) && (f->ovf.pin = b->pin_overflow) &&
          (b->slot = malloc(n * sizeof *b->slot)) && memset(b->slot, 0xff, n * sizeof *b->slot)) ||
         osmos_fail(err, "pinned memory refused (a step's tokens)");
}
