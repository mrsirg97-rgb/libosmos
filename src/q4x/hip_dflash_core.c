/* DFlash2's drafter for any target (dflash_core.h): z-lab/dflash's DFlash2DraftModel - the context's taps through fc
   and a norm into each layer's keys and values; a block (the anchor and masks, the target's embedding) through its
   layers (a dynamic convolution before and after the attention and the MLP), a norm, the target's output head, the
   candidate selector's greedy path. the target's calls through its DfHost (q4x's, LFM3's: each its own matrices'
   path and stream) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/ops.h"
#include "osmos/gpu/q4x_ops.h"
#include "dflash_core.h"
#include "osmos/err.h"

#define F sizeof(float)

static const float *f32(const DfHost *h, const GgufTensor *t) { return (const float *)h->dev(h->ctx, t); }

static uint32_t kv_of(const Q4xDflashGeometry *g) { return (uint32_t)(g->n_kv_heads * g->head_dim); }

static bool mat2(const DfHost *h, const GgufTensor *t0, uint32_t rows0, float *y0, uint32_t ys0, const GgufTensor *t1,
                 uint32_t rows1, float *y1, uint32_t ys1, uint32_t cols, const float *x, uint32_t xs, uint32_t n)
{
  return h->mat2 ? h->mat2(h->ctx, t0, rows0, y0, ys0, t1, rows1, y1, ys1, cols, x, xs, n)
                 : h->mat(h->ctx, t0, rows0, cols, x, xs, n, y0, ys0) && h->mat(h->ctx, t1, rows1, cols, x, xs, n, y1, ys1);
}

bool df_buffers(DfBuf *d, OsmosLedger *l, const Q4xDflashGeometry *g, uint32_t max_tokens, uint32_t max_ctx,
                uint32_t ring)
{
  const uint64_t n = max_tokens, B = (uint64_t)g->block, D = (uint64_t)g->n_embd, V = (uint64_t)g->n_vocab;
  const uint64_t kv = (uint64_t)g->n_kv_heads * g->head_dim, q = (uint64_t)g->n_heads * g->head_dim;
  const uint64_t rows = n > B ? n : B, dyn = 2ull * Q4X_DFLASH_CONV * (D / (uint64_t)g->conv_group);
  const uint64_t ff = (uint64_t)g->ff, k = (uint64_t)g->sel_top_k, iota = (uint64_t)max_ctx + B;
  const GpuWant want[] = {
      {&d->th, n * D * F},     {&d->k, rows * kv * F},  {&d->v, rows * kv * F},    {&d->h, B * D * F},
      {&d->x, B * D * F},      {&d->xc, B * D * F},     {&d->dyn, B * dyn * F},    {&d->q, B * q * F},
      {&d->att, B * q * F},    {&d->a, B * D * F},      {&d->gate, B * ff * F},    {&d->up, B * ff * F},
      {&d->hn, B * D * F},     {&d->logits, B * V * F}, {&d->hp, B * (uint64_t)g->sel_rank * F},
      {&d->val, B * k * F},    {&d->idx, B * k * 4},    {&d->top_p, B * F},        {&d->path, B * 4},
      {&d->top_id, B * 4},     {&d->tok, B * 4},        {&d->iota, iota * 4},
  };
  bool ok = gpu_bufs_alloc(&d->bufs, l, want, sizeof want / sizeof *want) && gpu_bufs_host(&d->bufs, l, &d->pin_tok, B * 4) &&
            gpu_bufs_host(&d->bufs, l, &d->pin_path, B * 4) && gpu_bufs_host(&d->bufs, l, &d->pin_idx, B * k * 4) &&
            gpu_bufs_host(&d->bufs, l, &d->pin_val, B * k * F) && gpu_bufs_host(&d->bufs, l, &d->pin_p, B * F);
  int32_t *h = ok ? malloc(iota * 4) : NULL;
  for (uint64_t i = 0; h && i < iota; i++)
    h[i] = (int32_t)i;
  for (uint64_t i = 1; h && i < B; i++) /* the block's tokens after its anchor: the mask */
    d->pin_tok[i] = g->mask_token;
  ok = h && hipMemcpy(d->iota, h, iota * 4, hipMemcpyHostToDevice) == hipSuccess &&
       hipMemcpy(d->tok, d->pin_tok, B * 4, hipMemcpyHostToDevice) == hipSuccess;
  free(h);
  const AttnCoreDesc ad = {.heads = (uint32_t)g->n_heads, .kv_heads = (uint32_t)g->n_kv_heads, .dk = (uint32_t)g->head_dim,
                           .dv = (uint32_t)g->head_dim, .k_rows = ATTN_ROWS_F32, .v_rows = ATTN_ROWS_F32,
                           .window = (uint32_t)g->window, .scaling = 1.0f / sqrtf((float)Q4X_DF_HEAD),
                           .term = ATTN_TERM_NONE, .sink = ATTN_SINK_NONE, .gate = ATTN_GATE_NONE,
                           .block = ATTN_BLOCK_DRAFT, .slots = ring, .max_tokens = (uint32_t)B,
                           .max_keys = ring + (uint32_t)B};
  char err[OSMOS_ERR];
  return ok && (d->attn = attn_core_create(&ad, NULL, 0, err)) != NULL; /* (its scores in the LDS: no scratch) */
}

void df_buffers_free(DfBuf *d)
{
  attn_core_destroy(d->attn);
  gpu_bufs_free(&d->bufs); /* (its device and pinned buffers alike: the record's) */
  memset(d, 0, sizeof *d);
}

/* a layer's keys (normed a head, rotated) and values of n rows of x at positions pos into d.k, d.v */
static bool keys(const DfHost *h, const Q4xDflash *df, const Q4xDflashLayer *L, DfBuf *d, const float *x, uint32_t n,
                 const int32_t *pos)
{
  const Q4xDflashGeometry *g = &df->geo;
  const uint32_t D = (uint32_t)g->n_embd, kv = kv_of(g), kh = (uint32_t)g->n_kv_heads;
  return mat2(h, L->k, kv, d->k, kv, L->v, kv, d->v, kv, D, x, D, n) &&
         ops_rms(d->k, Q4X_DF_HEAD, Q4X_DF_HEAD, f32(h, L->k_norm), g->eps, n * kh, d->k, Q4X_DF_HEAD, h->stream) &&
         q4x_df_rope(d->k, n, kh, kv, pos, g->rope_base, h->stream);
}

bool df_seed(const DfHost *h, const Q4xDflash *df, DfBuf *d, const DfRings *r, const float *taps, int32_t from,
             uint32_t n)
{
  const Q4xDflashGeometry *g = &df->geo;
  const uint32_t D = (uint32_t)g->n_embd, kv = kv_of(g), T = (uint32_t)g->n_taps * D;
  const int32_t *pos = d->iota + from;
  bool ok = h->mat(h->ctx, df->fc, D, T, taps, T, n, d->th, D) &&
            ops_rms(d->th, D, D, f32(h, df->hidden_norm), g->eps, n, d->th, D, h->stream);
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = keys(h, df, &df->layer[l], d, d->th, n, pos) && q4x_df_ring(r->k[l], r->ring, d->k, n, kv, from, h->stream) &&
         q4x_df_ring(r->v[l], r->ring, d->v, n, kv, from, h->stream);
  return ok;
}

/* x convolved (its dynamic kernel's half `which`, its base kernel's) into y: a block's B tokens */
static bool conv(const DfHost *h, const Q4xDflash *df, DfBuf *d, const GgufTensor *base, const float *x, int which,
                 float *y, uint32_t B)
{
  const Q4xDflashGeometry *g = &df->geo;
  const uint32_t D = (uint32_t)g->n_embd, groups = D / (uint32_t)g->conv_group, half = Q4X_DFLASH_CONV * groups;
  return q4x_df_conv(x, B, D, (uint32_t)g->conv_group, f32(h, base) + (uint64_t)which * Q4X_DFLASH_CONV * D,
                     d->dyn + (uint64_t)which * half, 2 * half, y, h->stream);
}

/* a layer on the block's B tokens of d.h at positions pos0 .., the context's keys from ctx_from */
static bool block_layer(const DfHost *h, const Q4xDflash *df, DfBuf *d, const DfRings *r, int l, uint32_t B,
                        int32_t pos0, int32_t ctx_from)
{
  const Q4xDflashGeometry *g = &df->geo;
  const Q4xDflashLayer *L = &df->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, q = (uint32_t)(g->n_heads * g->head_dim), ff = (uint32_t)g->ff;
  const uint32_t dyn = 2 * Q4X_DFLASH_CONV * (D / (uint32_t)g->conv_group), H = (uint32_t)g->n_heads;
  const int32_t *pos = d->iota + pos0;
  void *s = h->stream;
  return ops_rms(d->h, D, D, f32(h, L->attn_norm), g->eps, B, d->x, D, s) &&
         h->mat(h->ctx, L->attn_conv_proj, dyn, D, d->x, D, B, d->dyn, dyn) &&
         conv(h, df, d, L->attn_conv, d->x, 0, d->xc, B) && h->mat(h->ctx, L->q, q, D, d->xc, D, B, d->q, q) &&
         ops_rms(d->q, Q4X_DF_HEAD, Q4X_DF_HEAD, f32(h, L->q_norm), g->eps, B * H, d->q, Q4X_DF_HEAD, s) &&
         q4x_df_rope(d->q, B, H, q, pos, g->rope_base, s) && keys(h, df, L, d, d->xc, B, pos) &&
         attn_core_run(d->attn,
                       &(AttnStep){.q = d->q, .n_tok = B, .pos0 = (uint32_t)pos0, .out = d->att,
                                   .keys = {.k = r->k[l], .v = r->v[l], .key0 = (uint32_t)ctx_from, .own_k = d->k,
                                            .own_v = d->v}},
                       s) &&
         h->mat(h->ctx, L->out, D, q, d->att, q, B, d->a, D) && conv(h, df, d, L->attn_conv, d->a, 1, d->x, B) &&
         ops_add(d->h, d->x, (uint64_t)B * D, s) && ops_rms(d->h, D, D, f32(h, L->ffn_norm), g->eps, B, d->x, D, s) &&
         h->mat(h->ctx, L->ffn_conv_proj, dyn, D, d->x, D, B, d->dyn, dyn) &&
         conv(h, df, d, L->ffn_conv, d->x, 0, d->xc, B) &&
         mat2(h, L->gate, ff, d->gate, ff, L->up, ff, d->up, ff, D, d->xc, D, B) &&
         q4x_swiglu(d->gate, d->up, (uint64_t)B * ff, d->gate, NULL, s) &&
         h->mat(h->ctx, L->down, D, ff, d->gate, ff, B, d->a, D) && conv(h, df, d, L->ffn_conv, d->a, 1, d->x, B) &&
         ops_add(d->h, d->x, (uint64_t)B * D, s);
}

int df_draft(const DfHost *h, const Q4xDflash *df, DfBuf *d, const DfRings *r, int32_t y, int32_t pos0,
             int32_t ctx_from, uint32_t n, float conf_min, float cum_min, int32_t *drafts, float *conf, char *err)
{
  const Q4xDflashGeometry *g = &df->geo;
  const uint32_t B = (uint32_t)g->block, D = (uint32_t)g->n_embd, V = (uint32_t)g->n_vocab, R = (uint32_t)g->sel_rank;
  const uint32_t K = (uint32_t)g->sel_top_k;
  hipStream_t s = (hipStream_t)h->stream;
  d->pin_tok[0] = y; /* (the pinned row free: the draft before synchronized at its end) */
  bool ok = hipMemcpyAsync(d->tok, d->pin_tok, 4, hipMemcpyHostToDevice, s) == hipSuccess &&
            h->embed(h->ctx, d->tok, B, d->h);
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = block_layer(h, df, d, r, l, B, pos0, ctx_from);
  /* the block's places after the anchor: the norm, the head, the selector */
  ok = ok && ops_rms(d->h, D, D, f32(h, df->output_norm), g->eps, B, d->hn, D, s);
  if (ok && h->seen)
    h->seen(h->ctx, d->hn, (uint64_t)B * D);
  ok = ok && h->head(h->ctx, d->hn + D, D, B - 1, d->logits) &&
       q4x_df_topk(d->logits, B - 1, V, K, d->val, d->idx, s) &&
       h->mat(h->ctx, df->sel_proj, R, D, d->hn + D, D, B - 1, d->hp, R) &&
       q4x_df_select(d->val, d->idx, B - 1, K, d->hp, R, h->dev(h->ctx, df->sel_pred), h->dev(h->ctx, df->sel_succ), y,
                     d->path, s) &&
       q4x_top(d->logits, V, B - 1, V, d->top_id, d->top_p, s) &&
       hipMemcpyAsync(d->pin_path, d->path, (B - 1) * 4, hipMemcpyDeviceToHost, s) == hipSuccess &&
       hipMemcpyAsync(d->pin_idx, d->idx, (B - 1) * K * 4, hipMemcpyDeviceToHost, s) == hipSuccess &&
       hipMemcpyAsync(d->pin_val, d->val, (B - 1) * K * F, hipMemcpyDeviceToHost, s) == hipSuccess &&
       hipMemcpyAsync(d->pin_p, d->top_p, (B - 1) * F, hipMemcpyDeviceToHost, s) == hipSuccess &&
       (!h->before_sync || h->before_sync(h->ctx)) && hipStreamSynchronize(s) == hipSuccess;
  if (!ok)
    return osmos_fail(err, "a GPU call failed (a draft)"), -1;
  if (h->after_sync && !h->after_sync(h->ctx, err))
    return -1;
  /* a draft's confidence: the head's probability of it at its place - the greatest's (q4x_top) scaled by its logit
     below the greatest's (both the top-k's: the pick is one of its candidates, the greatest its first) */
  float cum = 1;
  uint32_t kept = 0;
  for (; kept < n; kept++)
  {
    const int32_t *ids = d->pin_idx + (uint64_t)kept * K;
    const float *val = d->pin_val + (uint64_t)kept * K;
    uint32_t c = 0;
    while (c < K && ids[c] != d->pin_path[kept])
      c++;
    if (c == K)
      return osmos_fail(err, "the selector's pick not one of its candidates"), -1;
    const float p = d->pin_p[kept] * expf(val[c] - val[0]);
    cum *= p;
    if (p < conf_min || cum < cum_min)
      break;
    drafts[kept] = d->pin_path[kept], conf[kept] = p;
  }
  return (int)kept;
}
