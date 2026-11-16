/* the qwen35 forward (osmos/q35/forward.h) on one GPU: q4x's kernels, a layer's blocks in HF's order - the input norm,
   the mixer (Gated DeltaNet or gated attention) into the residual, the post norm, the SwiGLU into the residual. a
   matrix's own scale (NVFP4's) applied to its output, as the role reading it (quant.h). one stream, no side work: the
   first version, the reference's bits checked layer by layer before any of q4x's overlap is taken over */
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/upload.h"
#include "osmos/gpu/lsa_ops.h"
#include "osmos/gpu/mm.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q35_ops.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/lsa.h"
#include "osmos/model/qtype.h"
#include "osmos/q35/forward.h"

#define F sizeof(float)
#define READ_PIECE (64ull << 20)

struct Q35Forward
{
  const Q35Weights *w;
  uint32_t max_tokens, max_ctx;
  Q4xRope rope;
  void **dev; /* the model's tensors, by index in its table */
  float *h, *xn, *out;
  float *qkv, *z, *al, *be, *g, *b, *cv, *qkn, *ro, *fo;
  float *qf, *kr, *vr, *q, *gt, *kq, *ao, *fa, *da; /* da: the decode attention's scratch */
  float *ga, *up, *y;
  float *logits, *pin_logits;
  int32_t *tok, *pin_tok, *overflow;
  void *x16;
  q35_tap_fn tap;
  void *tap_arg;
  /* latent sparse attention (q35_forward_lsa): a layer's maps (device, LSA_IN x LSA_DIM: NULL off), the read's scratch,
     a query's index vector, its read; the host's room for a closing block's Ward */
  bool lsa;
  float *wq[Q35_MAX_LAYERS], *wk[Q35_MAX_LAYERS];
  LsaScratch *lsa_s;
  float *iq;
  uint32_t *list, *list_n; /* a read and its count (device) */
  float *h_ik, *h_cent;
  int32_t *h_mem;
};

struct Q35Seq
{
  Q35Forward *f;
  int32_t pos;
  float *conv[Q35_MAX_LAYERS], *ssm[Q35_MAX_LAYERS]; /* a DeltaNet's conv history and state (one slot: in place) */
  void *kc[Q35_MAX_LAYERS], *vc[Q35_MAX_LAYERS];      /* an attention's cache, Q8_0 rows (q4x_kv_put) */
  /* with LSA: an attention layer's keys' index vectors (max_ctx x LSA_DIM), its closed blocks' buckets */
  float *ik[Q35_MAX_LAYERS], *cent[Q35_MAX_LAYERS];
  int32_t *mem[Q35_MAX_LAYERS];
  uint32_t closed[Q35_MAX_LAYERS];
};

static bool fail(char *err, const char *why)
{
  if (err)
    snprintf(err, Q35_ERR, "%s", why);
  return false;
}

static bool alloc(void *p, uint64_t bytes) { return hipMalloc((void **)p, bytes ? bytes : 1) == hipSuccess; }

static const void *dev(const Q35Forward *f, const GgufTensor *t)
{
  return f->dev[t - f->w->model->tensors];
}
static const float *fv(const Q35Forward *f, const GgufTensor *t) { return (const float *)dev(f, t); }

/* y = s W x over n tokens: the matmul where it tiles, else the matvec; the scale after (an NVFP4 matrix's own) */
static bool mat(Q35Forward *f, const GgufTensor *t, float s, uint32_t rows, uint32_t cols, const float *x, uint32_t n,
                float *y)
{
  if (t->type == QTYPE_NVFP4 && n <= 4 && cols % 64 == 0) /* a decode's: mv_fp4, the scale on its sums */
    return mv_fp4(dev(f, t), rows, cols, x, cols, n, y, rows, s, NULL);
  const bool ok = mm_known(t->type) && mm_fits(t->type, rows, cols)
                      ? mm_tokens(t->type, dev(f, t), rows, y, rows, NULL, 0, NULL, 0, cols, x, cols, n, f->x16,
                                  f->overflow, false, NULL)
                      : mv_strided(t->type, dev(f, t), rows, cols, x, cols, n, y, rows, NULL);
  return ok && (s == 1.0f || ops_scale(y, (uint64_t)n * rows, s, NULL));
}

static bool deltanet(Q35Forward *f, Q35Seq *s, int l, uint32_t n)
{
  const Q35Geometry *g = &f->w->geo;
  const Q35Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, vh = (uint32_t)g->ssm_v_heads, in = (uint32_t)g->ssm_inner;
  const uint32_t ch = 2u * (uint32_t)g->ssm_groups * (uint32_t)g->ssm_state + in;
  return mat(f, L->alpha, L->s_alpha, vh, D, f->xn, n, f->al) && mat(f, L->beta, L->s_beta, vh, D, f->xn, n, f->be) &&
         q4x_ssm_gates(f->al, f->be, fv(f, L->dt), fv(f, L->a), n, vh, f->g, f->b, NULL) &&
         mat(f, L->qkv, L->s_qkv, ch, D, f->xn, n, f->qkv) && mat(f, L->z, L->s_z, in, D, f->xn, n, f->z) &&
         q4x_conv(f->qkv, n, ch, (uint32_t)g->ssm_conv, 1, fv(f, L->conv), s->conv[l], 1, 0, f->cv, NULL) &&
         q4x_delta(f->cv, f->g, f->b, n, (uint32_t)g->ssm_groups, vh, (uint32_t)g->ssm_state, g->rms_eps, f->qkn,
                   s->ssm[l], 1, 0, f->ro, NULL) &&
         q4x_gated_norm_silu(f->ro, fv(f, L->ssm_norm), f->z, n, vh, (uint32_t)g->ssm_state, g->rms_eps, f->fo, NULL, NULL) &&
         mat(f, L->ssm_out, L->s_ssm_out, D, in, f->fo, n, f->out);
}

/* the layer's blocks closed up to `want` (each its 512 keys indexed): its index vectors to the host, Ward there
   (src/lsa's lsa_block), its buckets back */
static bool close_blocks(Q35Forward *f, Q35Seq *s, int l, uint32_t want)
{
  for (; s->closed[l] < want; s->closed[l]++)
  {
    const uint32_t b = s->closed[l];
    if (hipMemcpy(f->h_ik, s->ik[l] + (uint64_t)b * LSA_WINDOW * LSA_DIM, LSA_WINDOW * LSA_DIM * F,
                  hipMemcpyDeviceToHost) != hipSuccess ||
        !lsa_block(f->h_ik, b, f->h_cent, f->h_mem) ||
        hipMemcpy(s->cent[l] + (uint64_t)b * LSA_BUCKETS * LSA_DIM, f->h_cent, LSA_BUCKETS * LSA_DIM * F,
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(s->mem[l] + (uint64_t)b * LSA_BUCKETS * LSA_CAP, f->h_mem, LSA_BUCKETS * LSA_CAP * 4,
                  hipMemcpyHostToDevice) != hipSuccess)
      return false;
  }
  return true;
}

/* LSA in a step: the step's keys indexed, the blocks they close bucketed; a decode step's tokens each their read
   (the blocks closed before its window), its attention over the read alone - a prompt's stays dense (lsa.py's) */
static bool lsa_attention(Q35Forward *f, Q35Seq *s, int l, uint32_t n, bool *done)
{
  const Q35Geometry *g = &f->w->geo;
  const uint32_t heads = (uint32_t)g->n_heads, kvh = (uint32_t)g->n_kv_heads, hd = (uint32_t)g->head_dim;
  const uint32_t q = heads * hd, kv = kvh * hd, rot = (uint32_t)g->rope_dim, P = (uint32_t)s->pos;
  *done = false;
  if (!lsa_index_gpu(f->kq, n, kv, kvh, 1, hd, rot, f->wk[l], true, s->ik[l] + (uint64_t)P * LSA_DIM, NULL))
    return false;
  if (n > Q35_ATTN_DECODE_MAX)
    return close_blocks(f, s, l, (P + n) / LSA_WINDOW);
  for (uint32_t t = 0; t < n; t++)
  {
    if (!close_blocks(f, s, l, (P + t) / LSA_WINDOW) ||
        !lsa_index_gpu(f->q + (uint64_t)t * q, 1, q, kvh, heads / kvh, hd, rot, f->wq[l], false, f->iq, NULL) ||
        !lsa_read_gpu(f->iq, P + t, s->cent[l], s->mem[l], s->closed[l], f->lsa_s, f->list, f->list_n, NULL) ||
        !q35_attn_read(f->q + (uint64_t)t * q, s->kc[l], s->vc[l], heads, kvh, f->list, f->list_n, LSA_READ_MAX,
                       f->gt + (uint64_t)t * q, f->da, f->ao + (uint64_t)t * q, NULL))
      return false;
  }
  *done = true;
  return close_blocks(f, s, l, (P + n) / LSA_WINDOW);
}

static bool attention(Q35Forward *f, Q35Seq *s, int l, uint32_t n)
{
  const Q35Geometry *g = &f->w->geo;
  const Q35Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, heads = (uint32_t)g->n_heads, kvh = (uint32_t)g->n_kv_heads;
  const uint32_t hd = (uint32_t)g->head_dim, q = heads * hd, kv = kvh * hd;
  const uint64_t row = q4x_kv_row_bytes(kv);
  bool read = false; /* the step's attention done by its read (LSA's decode) */
  return mat(f, L->q, L->s_q, 2 * q, D, f->xn, n, f->qf) && mat(f, L->k, L->s_k, kv, D, f->xn, n, f->kr) &&
         mat(f, L->v, L->s_v, kv, D, f->xn, n, f->vr) &&
         q4x_qk(f->qf, f->kr, n, heads, kvh, hd, &f->rope, fv(f, L->q_norm), fv(f, L->k_norm), g->rms_eps, s->pos,
                f->q, f->gt, f->kq, NULL) &&
         q4x_kv_put(f->kq, f->vr, n, kv, (uint8_t *)s->kc[l] + (uint64_t)s->pos * row,
                    (uint8_t *)s->vc[l] + (uint64_t)s->pos * row, NULL) &&
         (!f->lsa || lsa_attention(f, s, l, n, &read)) &&
         (read || (n <= Q35_ATTN_DECODE_MAX
                       ? q35_attn_decode(f->q, s->kc[l], s->vc[l], n, heads, kvh, (uint32_t)s->pos, f->gt, f->da, f->ao, NULL)
                       : q4x_attn(f->q, s->kc[l], s->vc[l], n, heads, kvh, hd, (uint32_t)s->pos, f->gt, NULL, f->fa,
                                  f->ao, NULL, NULL))) &&
         mat(f, L->out, L->s_out, D, q, f->ao, n, f->out);
}

static bool mlp(Q35Forward *f, int l, uint32_t n)
{
  const Q35Geometry *g = &f->w->geo;
  const Q35Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, ff = (uint32_t)g->n_ff;
  return mat(f, L->ffn_gate, L->s_ffn_gate, ff, D, f->xn, n, f->ga) &&
         mat(f, L->ffn_up, L->s_ffn_up, ff, D, f->xn, n, f->up) &&
         ops_swiglu(f->ga, f->up, ff, INFINITY, NULL, n, f->y, NULL) &&
         mat(f, L->ffn_down, L->s_ffn_down, D, ff, f->y, n, f->out);
}

bool q35_step(Q35Forward *f, Q35Seq *s, const int32_t *tokens, uint32_t n, uint32_t rows, char err[Q35_ERR])
{
  if (!f || !s || !tokens || !n || n > f->max_tokens || !rows || rows > n)
    return fail(err, "a step of no tokens, more than the forward's buffers, or more rows than tokens");
  if ((uint64_t)s->pos + n > f->max_ctx)
    return fail(err, "past the forward's context");
  const Q35Geometry *g = &f->w->geo;
  const uint32_t D = (uint32_t)g->n_embd, V = (uint32_t)g->n_vocab;
  memcpy(f->pin_tok, tokens, (size_t)n * 4);
  bool ok = hipMemcpyAsync(f->tok, f->pin_tok, (size_t)n * 4, hipMemcpyHostToDevice, NULL) == hipSuccess &&
            ops_embed(f->w->embd->type, dev(f, f->w->embd), D, f->tok, n, 1, f->h, NULL);
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    const Q35Layer *L = &f->w->layer[l];
    ok = ops_rms(f->h, D, D, fv(f, L->attn_norm), g->rms_eps, n, f->xn, D, NULL) &&
         (g->recurrent[l] ? deltanet(f, s, l, n) : attention(f, s, l, n)) &&
         ops_add(f->h, f->out, (uint64_t)n * D, NULL) &&
         ops_rms(f->h, D, D, fv(f, L->post_norm), g->rms_eps, n, f->xn, D, NULL) && mlp(f, l, n) &&
         ops_add(f->h, f->out, (uint64_t)n * D, NULL);
    if (ok && f->tap && (ok = hipStreamSynchronize(NULL) == hipSuccess))
      f->tap(f->tap_arg, l, f->h, n, D);
  }
  const float *last = f->h + (uint64_t)(n - rows) * D;
  ok = ok && ops_rms(last, D, D, fv(f, f->w->out_norm), g->rms_eps, rows, f->xn, D, NULL) &&
       mat(f, f->w->output, f->w->s_output, V, D, f->xn, rows, f->logits) &&
       hipMemcpyAsync(f->pin_logits, f->logits, (uint64_t)rows * V * F, hipMemcpyDeviceToHost, NULL) == hipSuccess &&
       hipStreamSynchronize(NULL) == hipSuccess;
  int32_t over = 0;
  ok = ok && hipMemcpy(&over, f->overflow, 4, hipMemcpyDeviceToHost) == hipSuccess;
  if (!ok)
    return fail(err, "a GPU call failed in the step");
  if (over)
    return fail(err, "an activation past f16's range (the step refused)");
  s->pos += (int32_t)n;
  return true;
}

const float *q35_logits(const Q35Forward *f) { return f ? f->pin_logits : NULL; }

void q35_forward_tap(Q35Forward *f, q35_tap_fn fn, void *arg) { f->tap = fn, f->tap_arg = arg; }

static bool upload(Q35Forward *f, char *err)
{
  const Gguf *g = f->w->model;
  void *pin = NULL;
  if (!(f->dev = calloc(g->n_tensors, sizeof *f->dev)) || hipHostMalloc(&pin, READ_PIECE, 0) != hipSuccess)
    return fail(err, "out of memory (the weights' table, or a pinned piece)");
  bool ok = true;
  for (size_t i = 0; ok && i < g->n_tensors; i++)
  {
    const GgufTensor *t = &g->tensors[i];
    if (strncmp(t->name, "blk.", 4) == 0 && atoi(t->name + 4) >= f->w->geo.n_layers)
      continue; /* the MTP head's layers: not this forward's */
    const char *why = gpu_read_tensor(g, t, pin, READ_PIECE, &f->dev[i]);
    ok = !why;
    if (!ok && err)
      snprintf(err, Q35_ERR, "%s: not read into GPU memory (%s)", t->name, why);
  }
  hipHostFree(pin);
  return ok;
}

void q35_forward_destroy(Q35Forward *f)
{
  if (!f)
    return;
  for (size_t i = 0; f->dev && i < f->w->model->n_tensors; i++)
    hipFree(f->dev[i]);
  free(f->dev);
  void *b[] = {f->da, f->h,  f->xn, f->out, f->qkv, f->z,  f->al, f->be, f->g,  f->b,  f->cv,     f->qkn, f->ro,
               f->fo, f->qf, f->kr,  f->vr,  f->q,  f->gt, f->kq, f->ao, f->fa, f->ga,     f->up,  f->y,
               f->logits, f->tok, f->overflow, f->x16};
  for (size_t i = 0; i < sizeof b / sizeof *b; i++)
    hipFree(b[i]);
  hipHostFree(f->pin_logits), hipHostFree(f->pin_tok);
  for (int l = 0; l < Q35_MAX_LAYERS; l++)
    hipFree(f->wq[l]), hipFree(f->wk[l]);
  lsa_scratch_destroy(f->lsa_s);
  hipFree(f->iq), hipFree(f->list), hipFree(f->list_n);
  free(f->h_ik), free(f->h_cent), free(f->h_mem);
  free(f);
}

bool q35_forward_lsa(Q35Forward *f, const char *maps, float yarn, char err[Q35_ERR])
{
  const Q35Geometry *g = &f->w->geo;
  if (f->lsa)
    return fail(err, "the read already on");
  if (!maps) /* YaRN alone: the dense model under the maps' rotation (the read's comparison) */
    return q4x_rope((uint32_t)g->rope_dim, g->rope_base, g->native_ctx, (uint32_t)(yarn * (float)g->native_ctx), &f->rope) ||
           fail(err, "the YaRN rotation refused");
  if ((uint32_t)g->n_kv_heads * (uint32_t)(g->head_dim - g->rope_dim) != LSA_IN)
    return fail(err, "the model's free key dims are not the maps' input");
  FILE *fp = fopen(maps, "rb");
  uint32_t hdr[4] = {0};
  if (!fp || fread(hdr, 4, 4, fp) != 4 || memcmp(hdr, "LSAM", 4) || hdr[2] != LSA_IN || hdr[3] != LSA_DIM)
  {
    if (fp)
      fclose(fp);
    return fail(err, "the maps file not read (tools/convert/lsa_maps.py's, LSA_IN x LSA_DIM)");
  }
  const size_t m = (size_t)LSA_IN * LSA_DIM;
  float *buf = malloc(2 * m * F);
  bool ok = buf != NULL;
  uint32_t seen = 0;
  for (uint32_t i = 0; ok && i < hdr[1]; i++)
  {
    uint32_t L = 0;
    ok = fread(&L, 4, 1, fp) == 1 && fread(buf, F, 2 * m, fp) == 2 * m && L < (uint32_t)g->n_layers &&
         !g->recurrent[L] && !f->wq[L] && alloc(&f->wq[L], m * F) && alloc(&f->wk[L], m * F) &&
         hipMemcpy(f->wq[L], buf, m * F, hipMemcpyHostToDevice) == hipSuccess &&
         hipMemcpy(f->wk[L], buf + m, m * F, hipMemcpyHostToDevice) == hipSuccess;
    seen += ok;
  }
  fclose(fp);
  free(buf);
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = g->recurrent[l] || f->wq[l] || (fail(err, "an attention layer without its maps"), false);
  if (!ok)
    return err && !*err ? fail(err, "the maps not loaded") : false;
  /* the maps' rotation: YaRN at yarn x native (their training's), whatever this forward's context */
  if (!q4x_rope((uint32_t)g->rope_dim, g->rope_base, g->native_ctx, (uint32_t)(yarn * (float)g->native_ctx), &f->rope))
    return fail(err, "the maps' YaRN rotation refused");
  ok = (f->lsa_s = lsa_scratch_create(f->max_ctx / LSA_WINDOW * LSA_BUCKETS + 1)) && alloc(&f->iq, LSA_DIM * F) &&
       alloc(&f->list, LSA_READ_MAX * 4) && alloc(&f->list_n, 4) && (f->h_ik = malloc(LSA_WINDOW * LSA_DIM * F)) &&
       (f->h_cent = malloc(LSA_BUCKETS * LSA_DIM * F)) && (f->h_mem = malloc(LSA_BUCKETS * LSA_CAP * 4));
  if (!ok)
    return fail(err, "out of memory (the read's scratch)");
  f->lsa = seen > 0;
  return true;
}

Q35Forward *q35_forward_create(const Q35Weights *w, uint32_t max_tokens, uint32_t max_ctx, char err[Q35_ERR])
{
  if (!w || !max_tokens || max_tokens > max_ctx || max_ctx > OPS_Q4X_MAX_KEYS)
    return fail(err, "no weights, no tokens, or a context past the attention's keys"), NULL;
  Q35Forward *f = calloc(1, sizeof *f);
  if (!f)
    return fail(err, "out of memory"), NULL;
  f->w = w, f->max_tokens = max_tokens, f->max_ctx = max_ctx;
  const Q35Geometry *g = &w->geo;
  const uint64_t n = max_tokens, D = (uint64_t)g->n_embd, ff = (uint64_t)g->n_ff, vh = (uint64_t)g->ssm_v_heads;
  const uint64_t in = (uint64_t)g->ssm_inner, ch = 2ull * g->ssm_groups * g->ssm_state + in;
  const uint64_t q = (uint64_t)g->n_heads * g->head_dim, kv = (uint64_t)g->n_kv_heads * g->head_dim;
  const uint64_t wide = ff > ch ? ff : ch; /* mm's widest input: the SwiGLU's down */
  const struct
  {
    void *p;
    uint64_t bytes;
  } want[] = {
      {&f->h, n * D * F},       {&f->xn, n * D * F},  {&f->out, n * D * F},     {&f->qkv, n * ch * F},
      {&f->z, n * in * F},      {&f->al, n * vh * F}, {&f->be, n * vh * F},     {&f->g, n * vh * F},
      {&f->b, n * vh * F},      {&f->cv, n * ch * F}, {&f->qkn, n * 2 * g->ssm_groups * g->ssm_state * F},
      {&f->ro, n * in * F},     {&f->fo, n * in * F}, {&f->qf, n * 2 * q * F},  {&f->kr, n * kv * F},
      {&f->vr, n * kv * F},     {&f->q, n * q * F},   {&f->gt, n * q * F},      {&f->kq, n * kv * F},
      {&f->ao, n * q * F},      {&f->fa, q4x_attn_scratch((uint32_t)g->n_heads, max_ctx) * F},
      {&f->da, q35_attn_decode_scratch((uint32_t)g->n_heads, Q35_ATTN_DECODE_MAX) * F},
      {&f->ga, n * ff * F},     {&f->up, n * ff * F}, {&f->y, n * ff * F},      {&f->logits, n * g->n_vocab * F},
      {&f->tok, n * 4},         {&f->overflow, 4},    {&f->x16, (uint64_t)MM_MAX_TOKENS * wide * 2},
  };
  bool ok = q4x_rope((uint32_t)g->rope_dim, g->rope_base, g->native_ctx, max_ctx, &f->rope) && upload(f, err);
  for (size_t i = 0; ok && i < sizeof want / sizeof *want; i++)
    ok = alloc(want[i].p, want[i].bytes) || fail(err, "GPU memory refused (a step's buffers)");
  ok = ok && hipMemset(f->overflow, 0, 4) == hipSuccess &&
       hipHostMalloc((void **)&f->pin_logits, n * g->n_vocab * F, 0) == hipSuccess &&
       hipHostMalloc((void **)&f->pin_tok, n * 4, 0) == hipSuccess;
  if (!ok)
    return q35_forward_destroy(f), (err && !*err ? fail(err, "the forward not made") : false), NULL;
  return f;
}

Q35Seq *q35_seq_create(Q35Forward *f, char err[Q35_ERR])
{
  if (!f)
    return fail(err, "no forward"), NULL;
  const Q35Geometry *g = &f->w->geo;
  Q35Seq *s = calloc(1, sizeof *s);
  if (!s)
    return fail(err, "out of memory"), NULL;
  s->f = f;
  const uint64_t ch = 2ull * g->ssm_groups * g->ssm_state + (uint64_t)g->ssm_inner;
  const uint64_t ssm = (uint64_t)g->ssm_v_heads * g->ssm_state * g->ssm_state * F, conv = (g->ssm_conv - 1) * ch * F;
  const uint64_t row = q4x_kv_row_bytes((uint32_t)(g->n_kv_heads * g->head_dim));
  bool ok = true;
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = g->recurrent[l] ? alloc(&s->ssm[l], ssm) && alloc(&s->conv[l], conv) &&
                               hipMemset(s->ssm[l], 0, ssm) == hipSuccess && hipMemset(s->conv[l], 0, conv) == hipSuccess
                         : alloc(&s->kc[l], f->max_ctx * row) && alloc(&s->vc[l], f->max_ctx * row) &&
                               (!f->lsa || (alloc(&s->ik[l], (uint64_t)f->max_ctx * LSA_DIM * F) &&
                                            alloc(&s->cent[l], (uint64_t)(f->max_ctx / LSA_WINDOW + 1) * LSA_BUCKETS * LSA_DIM * F) &&
                                            alloc(&s->mem[l], (uint64_t)(f->max_ctx / LSA_WINDOW + 1) * LSA_BUCKETS * LSA_CAP * 4)));
  if (!ok)
    return q35_seq_destroy(s), fail(err, "GPU memory refused (a sequence's state)"), NULL;
  return s;
}

void q35_seq_destroy(Q35Seq *s)
{
  if (!s)
    return;
  for (int l = 0; l < Q35_MAX_LAYERS; l++)
    hipFree(s->conv[l]), hipFree(s->ssm[l]), hipFree(s->kc[l]), hipFree(s->vc[l]), hipFree(s->ik[l]),
        hipFree(s->cent[l]), hipFree(s->mem[l]);
  free(s);
}

int32_t q35_seq_position(const Q35Seq *s) { return s ? s->pos : -1; }
