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
#include "osmos/gpu/quant.h"
#include "osmos/lsa.h"
#include "osmos/model/qtype.h"
#include "osmos/q35/forward.h"
#include "q35_impl.h"


bool q35_fail(char *err, const char *why)
{
  if (err)
    snprintf(err, Q35_ERR, "%s", why);
  return false;
}
#define fail q35_fail

bool q35_alloc(void *p, uint64_t bytes) { return hipMalloc((void **)p, bytes ? bytes : 1) == hipSuccess; }
#define alloc q35_alloc


/* y = s W x over n tokens: the matmul where it tiles, else the matvec; the scale after (an NVFP4 matrix's own) */
/* a step's runs: each sequence's tokens at rows off .. off + len - 1 of the step */
typedef struct Run
{
  Q35Seq *s;
  uint32_t off, len;
} Run;

/* y = s W x over n tokens. read (a prompt's): the matmul where it tiles (any n), else the matvec; decode: an NVFP4
   matrix mv_fp4 in steps of Q35_MV_TOKENS tokens (its own sums: a token's bits whatever the step's size) */
#define Q35_MV_TOKENS 4
static bool mat_typed(Q35Forward *f, uint32_t type, const void *w, float s, uint32_t rows, uint32_t cols,
                      const float *x, uint32_t n, float *y, bool read)
{
  if (!read && type == QTYPE_NVFP4 && cols % 64 == 0)
  {
    bool ok = true;
    for (uint32_t t = 0; ok && t < n; t += Q35_MV_TOKENS)
      ok = mv_fp4(w, rows, cols, x + (uint64_t)t * cols, cols, n - t < Q35_MV_TOKENS ? n - t : Q35_MV_TOKENS,
                  y + (uint64_t)t * rows, rows, s, NULL);
    return ok;
  }
  const bool ok = mm_known(type) && mm_fits(type, rows, cols)
                      ? mm_tokens(type, w, rows, y, rows, NULL, 0, NULL, 0, cols, x, cols, n, f->x16, f->overflow,
                                  false, NULL)
                      : mv_strided(type, w, rows, cols, x, cols, n, y, rows, NULL);
  return ok && (s == 1.0f || ops_scale(y, (uint64_t)n * rows, s, NULL));
}
static bool mat(Q35Forward *f, const GgufTensor *t, float s, uint32_t rows, uint32_t cols, const float *x, uint32_t n,
                float *y, bool read)
{
  return mat_typed(f, t->type, dev(f, t), s, rows, cols, x, n, y, read);
}

static bool deltanet(Q35Forward *f, const Run *run, uint32_t runs, uint32_t n, int l, bool read)
{
  const Q35Geometry *g = &f->w->geo;
  const Q35Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, vh = (uint32_t)g->ssm_v_heads, in = (uint32_t)g->ssm_inner;
  const uint32_t ch = 2u * (uint32_t)g->ssm_groups * (uint32_t)g->ssm_state + in;
  bool ok = mat(f, L->alpha, L->s_alpha, vh, D, f->xn, n, f->al, read) &&
            mat(f, L->beta, L->s_beta, vh, D, f->xn, n, f->be, read) &&
            q4x_ssm_gates(f->al, f->be, fv(f, L->dt), fv(f, L->a), n, vh, f->g, f->b, NULL) &&
            mat(f, L->qkv, L->s_qkv, ch, D, f->xn, n, f->qkv, read) && mat(f, L->z, L->s_z, in, D, f->xn, n, f->z, read);
  for (uint32_t r = 0; ok && r < runs; r++) /* a sequence's conv history and state its own, its tokens in order */
  {
    const uint64_t o = run[r].off;
    Q35Seq *s = run[r].s;
    ok = q4x_conv(f->qkv + o * ch, run[r].len, ch, (uint32_t)g->ssm_conv, 1, fv(f, L->conv), s->conv[l], Q35_SLOTS,
                  s->cur, f->cv + o * ch, NULL) &&
         q4x_delta(f->cv + o * ch, f->g + o * vh, f->b + o * vh, run[r].len, (uint32_t)g->ssm_groups, vh,
                   (uint32_t)g->ssm_state, g->rms_eps, f->qkn, s->ssm[l], Q35_SLOTS, s->cur, f->ro + o * in, NULL);
  }
  return ok &&
         q4x_gated_norm_silu(f->ro, fv(f, L->ssm_norm), f->z, n, vh, (uint32_t)g->ssm_state, g->rms_eps, f->fo, NULL, NULL) &&
         mat(f, L->ssm_out, L->s_ssm_out, D, in, f->fo, n, f->out, read);
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

/* LSA in a run: its keys indexed, the blocks they close bucketed; a decode's tokens each their read (the blocks
   closed before its window), its attention over the read alone - a prompt's stays dense (lsa.py's) */
static bool lsa_attention(Q35Forward *f, Q35Seq *s, int l, uint32_t off, uint32_t n, bool read, bool *done)
{
  const Q35Geometry *g = &f->w->geo;
  const uint32_t heads = (uint32_t)g->n_heads, kvh = (uint32_t)g->n_kv_heads, hd = (uint32_t)g->head_dim;
  const uint32_t q = heads * hd, kv = kvh * hd, rot = (uint32_t)g->rope_dim, P = (uint32_t)s->pos;
  *done = false;
  if (!lsa_index_gpu(f->kq + (uint64_t)off * kv, n, kv, kvh, 1, hd, rot, f->wk[l], true,
                     s->ik[l] + (uint64_t)P * LSA_DIM, NULL))
    return false;
  if (read)
    return close_blocks(f, s, l, (P + n) / LSA_WINDOW);
  for (uint32_t t = 0; t < n; t++)
  {
    const uint64_t at = (uint64_t)(off + t) * q;
    if (!close_blocks(f, s, l, (P + t) / LSA_WINDOW) ||
        !lsa_index_gpu(f->q + at, 1, q, kvh, heads / kvh, hd, rot, f->wq[l], false, f->iq, NULL) ||
        !lsa_read_gpu(f->iq, P + t, s->cent[l], s->mem[l], s->closed[l], f->lsa_s, f->list, f->list_n, NULL) ||
        !q35_attn_read(f->q + at, s->kc[l], s->vc[l], heads, kvh, f->list, f->list_n, LSA_READ_MAX, f->gt + at, f->da,
                       f->ao + at, NULL))
      return false;
  }
  *done = true;
  return close_blocks(f, s, l, (P + n) / LSA_WINDOW);
}

/* a run's attention, dense: read the prompt's kernel (q4x_attn) at any size; decode q35_attn_decode in steps of
   Q35_ATTN_DECODE_MAX tokens (a token's split softmax its own: its bits whatever the step) */
static bool dense_attention(Q35Forward *f, Q35Seq *s, int l, uint32_t off, uint32_t n, bool read)
{
  const Q35Geometry *g = &f->w->geo;
  const uint32_t heads = (uint32_t)g->n_heads, kvh = (uint32_t)g->n_kv_heads, hd = (uint32_t)g->head_dim;
  const uint64_t q = (uint64_t)heads * hd;
  if (read)
    return q4x_attn(f->q + off * q, s->kc[l], s->vc[l], n, heads, kvh, hd, (uint32_t)s->pos, f->gt + off * q, NULL,
                    f->fa, f->ao + off * q, NULL, NULL);
  bool ok = true;
  for (uint32_t t = 0; ok && t < n; t += Q35_ATTN_DECODE_MAX)
  {
    const uint64_t at = (uint64_t)(off + t) * q;
    ok = q35_attn_decode(f->q + at, s->kc[l], s->vc[l], n - t < Q35_ATTN_DECODE_MAX ? n - t : Q35_ATTN_DECODE_MAX, heads,
                         kvh, (uint32_t)s->pos + t, f->gt + at, f->da, f->ao + at, NULL);
  }
  return ok;
}

static bool attention(Q35Forward *f, const Run *run, uint32_t runs, uint32_t n, int l, bool read)
{
  const Q35Geometry *g = &f->w->geo;
  const Q35Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, heads = (uint32_t)g->n_heads, kvh = (uint32_t)g->n_kv_heads;
  const uint32_t hd = (uint32_t)g->head_dim, q = heads * hd, kv = kvh * hd;
  const uint64_t row = q4x_kv_row_bytes(kv);
  bool ok = mat(f, L->q, L->s_q, 2 * q, D, f->xn, n, f->qf, read) && mat(f, L->k, L->s_k, kv, D, f->xn, n, f->kr, read) &&
            mat(f, L->v, L->s_v, kv, D, f->xn, n, f->vr, read);
  for (uint32_t r = 0; ok && r < runs; r++) /* a sequence's rotation, cache and attention its own */
  {
    const uint64_t o = run[r].off;
    const uint32_t m = run[r].len;
    Q35Seq *s = run[r].s;
    bool done = false; /* the run's attention done by its read (LSA's decode) */
    ok = q4x_qk(f->qf + o * 2 * q, f->kr + o * kv, m, heads, kvh, hd, &f->rope, fv(f, L->q_norm), fv(f, L->k_norm),
                g->rms_eps, s->pos, f->q + o * q, f->gt + o * q, f->kq + o * kv, NULL) &&
         q4x_kv_put(f->kq + o * kv, f->vr + o * kv, m, kv, (uint8_t *)s->kc[l] + (uint64_t)s->pos * row,
                    (uint8_t *)s->vc[l] + (uint64_t)s->pos * row, NULL) &&
         (!f->lsa || lsa_attention(f, s, l, (uint32_t)o, m, read, &done)) &&
         (done || dense_attention(f, s, l, (uint32_t)o, m, read));
  }
  return ok && mat(f, L->out, L->s_out, D, q, f->ao, n, f->out, read);
}

static bool mlp(Q35Forward *f, int l, uint32_t n, bool read)
{
  const Q35Geometry *g = &f->w->geo;
  const Q35Layer *L = &f->w->layer[l];
  const uint32_t D = (uint32_t)g->n_embd, ff = (uint32_t)g->n_ff;
  return mat(f, L->ffn_gate, L->s_ffn_gate, ff, D, f->xn, n, f->ga, read) &&
         mat(f, L->ffn_up, L->s_ffn_up, ff, D, f->xn, n, f->up, read) &&
         ops_swiglu(f->ga, f->up, ff, INFINITY, NULL, n, f->y, NULL) &&
         mat(f, L->ffn_down, L->s_ffn_down, D, ff, f->y, n, f->out, read);
}

/* the step (q35_step_runs), read its math (a prompt's or a decode's); keep_from: rows below it not kept whatever the
   prompts say (q35_step's: its last `rows` rows alone) */
static bool step(Q35Forward *f, Q35Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                 bool read, uint32_t keep_from, char *err)
{
  if (!f || !seqs || !lens || !tokens || !runs || runs > Q35_MAX_RUNS)
    return fail(err, "no forward, sequences or tokens, or past Q35_MAX_RUNS runs");
  Run run[Q35_MAX_RUNS];
  uint32_t n = 0, kept = 0;
  for (uint32_t r = 0; r < runs; r++)
  {
    Q35Seq *s = seqs[r];
    if (!s || s->f != f || !lens[r])
      return fail(err, "a run of no sequence, of another forward, or of no tokens");
    if (s->broken)
      return fail(err, "a sequence after a failed step (its states part written): load it, or make another");
    for (uint32_t k = 0; k < r; k++)
      if (seqs[k] == s)
        return fail(err, "a sequence twice in one step");
    if ((uint64_t)n + lens[r] > f->max_tokens || (uint64_t)s->pos + lens[r] > f->max_ctx)
      return fail(err, "the tokens past the forward's step, or a sequence past its context");
    run[r] = (Run){s, n, lens[r]};
    for (uint32_t t = 0; t < lens[r]; t++)
      kept += n + t >= keep_from && (!s->prompt_end || s->pos + t + 1 >= s->prompt_end);
    n += lens[r];
  }
  if (kept > f->logit_rows)
    return fail(err, "a step's kept rows past the logits' (Q35Options' logit_rows)");
  const Q35Geometry *g = &f->w->geo;
  const uint32_t D = (uint32_t)g->n_embd, V = (uint32_t)g->n_vocab;
  memcpy(f->pin_tok, tokens, (size_t)n * 4);
  bool ok = hipMemcpyAsync(f->tok, f->pin_tok, (size_t)n * 4, hipMemcpyHostToDevice, NULL) == hipSuccess &&
            ops_embed(f->w->embd->type, dev(f, f->w->embd), D, f->tok, n, 1, f->h, NULL);
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    const Q35Layer *L = &f->w->layer[l];
    ok = ops_rms(f->h, D, D, fv(f, L->attn_norm), g->rms_eps, n, f->xn, D, NULL) &&
         (g->recurrent[l] ? deltanet(f, run, runs, n, l, read) : attention(f, run, runs, n, l, read)) &&
         ops_add(f->h, f->out, (uint64_t)n * D, NULL) &&
         ops_rms(f->h, D, D, fv(f, L->post_norm), g->rms_eps, n, f->xn, D, NULL) && mlp(f, l, n, read) &&
         ops_add(f->h, f->out, (uint64_t)n * D, NULL);
    if (ok && f->tap && (ok = hipStreamSynchronize(NULL) == hipSuccess))
      f->tap(f->tap_arg, l, f->h, n, D);
  }
  /* the kept rows' residuals gathered (a run's kept rows its last ones), then the norm and the head on them alone */
  uint32_t k = 0;
  for (uint32_t r = 0; r < runs; r++)
  {
    const Q35Seq *s = run[r].s;
    for (uint32_t t = 0; t < run[r].len; t++)
    {
      const uint32_t row = run[r].off + t;
      const bool keep = row >= keep_from && (!s->prompt_end || s->pos + t + 1 >= s->prompt_end);
      f->slot[row] = keep ? (int32_t)k : -1;
      if (keep && ok)
        ok = hipMemcpyAsync(f->hx + (uint64_t)k * D, f->h + (uint64_t)row * D, D * F, hipMemcpyDeviceToDevice, NULL) ==
             hipSuccess;
      k += keep;
    }
  }
  if (ok && kept)
    ok = ops_rms(f->hx, D, D, fv(f, f->w->out_norm), g->rms_eps, kept, f->xn, D, NULL) &&
         mat_typed(f, f->head_type, dev(f, f->w->output), f->w->s_output, V, D, f->xn, kept, f->logits, read);
  ok = ok && hipStreamSynchronize(NULL) == hipSuccess;
  int32_t over = 0;
  ok = ok && hipMemcpy(&over, f->overflow, 4, hipMemcpyDeviceToHost) == hipSuccess;
  f->rows = ok && !over ? n : 0;
  if (!ok || over)
  {
    for (uint32_t r = 0; r < runs; r++) /* their rings part written: never read on */
      run[r].s->broken = true;
    return fail(err, over ? "an activation past f16's range (the step refused)" : "a GPU call failed in the step");
  }
  for (uint32_t r = 0; r < runs; r++)
  {
    Q35Seq *s = run[r].s;
    s->pos += (int32_t)run[r].len, s->cur = (s->cur + run[r].len) % Q35_SLOTS;
    s->back = s->back + run[r].len < Q35_MAX_BACK ? s->back + run[r].len : Q35_MAX_BACK;
  }
  return true;
}

bool q35_step_runs(Q35Forward *f, Q35Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                   char err[Q35_ERR])
{
  return step(f, seqs, lens, runs, tokens, false, 0, err);
}

bool q35_step_read(Q35Forward *f, Q35Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                   char err[Q35_ERR])
{
  return step(f, seqs, lens, runs, tokens, true, 0, err);
}

/* one sequence, its last `rows` rows' logits on the host (q35_logits): a step past Q35_MV_TOKENS tokens a prompt's
   math, the rest a decode's (the checks against HF: both) */
bool q35_step(Q35Forward *f, Q35Seq *s, const int32_t *tokens, uint32_t n, uint32_t rows, char err[Q35_ERR])
{
  if (!f || !s || !n || !rows || rows > n || rows > f->logit_rows)
    return fail(err, "a step of no tokens, or more rows than tokens or the logits'");
  if (!step(f, &s, &n, 1, tokens, n > Q35_MV_TOKENS, n - rows, err))
    return false;
  const uint32_t V = (uint32_t)f->w->geo.n_vocab;
  return hipMemcpy(f->pin_logits, f->logits, (uint64_t)rows * V * F, hipMemcpyDeviceToHost) == hipSuccess ||
         fail(err, "a GPU call failed (the logits' copy)");
}

const float *q35_logits(const Q35Forward *f) { return f ? f->pin_logits : NULL; }

const float *q35_row_logits(const Q35Forward *f, uint32_t row)
{
  return f && row < f->rows && f->slot[row] >= 0 ? f->logits + (uint64_t)f->slot[row] * f->w->geo.n_vocab : NULL;
}

bool q35_greedy(Q35Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[Q35_ERR])
{
  if (!f || !ids || !rows || row0 + rows > f->rows)
    return fail(err, "no forward or ids, or rows past the last step's");
  for (uint32_t r = 0; r < rows; r++) /* kept rows' slots in their order: rows row0 .. one run of slots */
    if (f->slot[row0 + r] < 0 || f->slot[row0 + r] != f->slot[row0] + (int32_t)r)
      return fail(err, "a row the last step kept no logits of");
  const uint32_t V = (uint32_t)f->w->geo.n_vocab;
  if (!ops_argmax(f->logits + (uint64_t)f->slot[row0] * V, V, rows, V, f->ids, f->argmax, NULL) ||
      hipMemcpy(ids, f->ids, rows * 4, hipMemcpyDeviceToHost) != hipSuccess)
    return fail(err, "a GPU call failed (the greedy pick)");
  for (uint32_t r = 0; r < rows; r++)
    if (ids[r] < 0)
      return fail(err, "a NaN in a row's logits");
  return true;
}
