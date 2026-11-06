#include "q4x_impl.h"

/* a layer's blocks, each as its test holds it to llama.cpp's (tests/gpu/test_q4x_*_gpu): the mixer into b.mixed and
   b.inj, a block's output into b.out, the n-gram layer's into b.h itself */

#define TY(t) ((t)->type)

/* y = W x for n tokens, x's rows xs apart, y's ys apart */
static bool mat(const Q4xForward *f, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t xs,
                uint32_t n, float *y, uint32_t ys)
{
  return mv_strided(TY(t), q4x_dev(f, t), rows, cols, x, xs, n, y, ys, NULL);
}

static const float *fv(const Q4xForward *f, const GgufTensor *t) { return (const float *)q4x_dev(f, t); }

bool q4x_mix(Q4xForward *f, const GgufTensor *norm, const GgufTensor *down, const GgufTensor *up,
             const GgufTensor *inject, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t hc = (uint32_t)g->hc, D = (uint32_t)g->n_embd, hd = hc * D, r = (uint32_t)g->hc_rank;
  return q4x_hc_norm(b->h, fv(f, norm), n, hc, D, g->rms_eps, b->xn, NULL) && mat(f, down, r, hd, b->xn, hd, n, b->lo, r) &&
         q4x_scale_silu(b->lo, (uint64_t)n * r, 1.0f / (float)hc, NULL) && mat(f, up, hd, r, b->lo, r, n, b->gate, hd) &&
         q4x_hc_gated_mean(b->xn, b->gate, n, hc, D, b->mixed, NULL) &&
         (!inject || mat(f, inject, hc, hd, b->xn, hd, n, b->inj, hc));
}

bool q4x_deltanet(Q4xForward *f, Q4xSeq *s, int l, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xLayer *L = &f->w->layer[l];
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, vh = (uint32_t)g->ssm_v_heads, in = (uint32_t)g->ssm_inner;
  const uint32_t ch = 2u * (uint32_t)g->ssm_groups * (uint32_t)g->ssm_state + in;
  return mat(f, L->qkv, ch, D, b->mixed, D, n, b->qkv, ch) && mat(f, L->gate, in, D, b->mixed, D, n, b->z, in) &&
         mat(f, L->alpha, vh, D, b->mixed, D, n, b->al, vh) && mat(f, L->beta, vh, D, b->mixed, D, n, b->be, vh) &&
         q4x_ssm_gates(b->al, b->be, fv(f, L->dt), fv(f, L->a), n, vh, b->g, b->b, NULL) &&
         q4x_conv(b->qkv, n, ch, (uint32_t)g->ssm_conv, 1, fv(f, L->conv), s->conv[l], Q4X_SLOTS, s->cur, b->cv,
                  NULL) &&
         q4x_delta(b->cv, b->g, b->b, n, (uint32_t)g->ssm_groups, vh, (uint32_t)g->ssm_state, g->rms_eps, s->ssm[l],
                   Q4X_SLOTS, s->cur, b->ro, NULL) &&
         q4x_gated_norm(b->ro, fv(f, L->ssm_norm), b->z, n, vh, (uint32_t)g->ssm_state, g->rms_eps, b->fo, NULL) &&
         mat(f, L->ssm_out, D, in, b->fo, in, n, b->out, D);
}

bool q4x_attention(Q4xForward *f, const Q4xLayer *L, float *kc, float *vc, int32_t pos, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, heads = (uint32_t)g->n_heads, kvh = (uint32_t)g->n_kv_heads;
  const uint32_t hd = (uint32_t)g->head_dim, q = heads * hd, kv = kvh * hd;
  return mat(f, L->q, 2 * q, D, b->mixed, D, n, b->qf, 2 * q) && mat(f, L->k, kv, D, b->mixed, D, n, b->kr, kv) &&
         mat(f, L->v, kv, D, b->mixed, D, n, vc + (uint64_t)pos * kv, kv) &&
         q4x_qk(b->qf, b->kr, n, heads, kvh, hd, (uint32_t)g->rope_dim, fv(f, L->q_norm), fv(f, L->k_norm), g->rms_eps,
                pos, g->rope_base, b->q, b->gt, kc + (uint64_t)pos * kv, NULL) &&
         q4x_attn(b->q, kc, vc, n, heads, kvh, hd, (uint32_t)pos, b->gt, b->ao, NULL) &&
         mat(f, L->out, D, q, b->ao, q, n, b->out, D);
}

/* the cache rows alone (the MTP head's seeding): k and v of b.mixed's n tokens at pos .., k normed and roped */
bool q4x_attention_kv(Q4xForward *f, const Q4xLayer *L, float *kc, float *vc, int32_t pos, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, kvh = (uint32_t)g->n_kv_heads, hd = (uint32_t)g->head_dim, kv = kvh * hd;
  return mat(f, L->k, kv, D, b->mixed, D, n, b->kr, kv) && mat(f, L->v, kv, D, b->mixed, D, n, vc + (uint64_t)pos * kv, kv) &&
         q4x_qk(b->qf, b->kr, n, 0, kvh, hd, (uint32_t)g->rope_dim, fv(f, L->q_norm), fv(f, L->k_norm), g->rms_eps, pos,
                g->rope_base, b->q, b->gt, kc + (uint64_t)pos * kv, NULL);
}

/* an expert tensor's matvecs over the grouping's two tables: its runs of tokens, then its single tokens */
static bool experts(const GgufTensor *t, const void **ptr, const Q4xBuf *b, uint32_t multi, uint32_t single,
                    uint32_t per, uint32_t rows, uint32_t cols, const float *x, float *y)
{
  return (!multi || mv_group_tok(TY(t), ptr, b->cnt, b->row, multi, per, rows, cols, x, cols, y, rows, NULL)) &&
         mv_group_tok(TY(t), ptr + multi, b->cnt + multi, b->row + multi, single, 1, rows, cols, x, cols, y, rows, NULL);
}

bool q4x_moe(Q4xForward *f, const Q4xLayer *L, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, E = (uint32_t)g->n_experts, k = (uint32_t)g->n_used, ff = (uint32_t)g->ff_exp;
  /* a prompt's step grouped, MV_MAX_TOKENS tokens an expert's read; a decode or a verify step (MV_MAX_TOKENS tokens at
     most) a slot an entry: its tokens share a third of their experts (4 of them), but read for several tokens an
     expert costs that back (DESIGN.md, "Qwen's speed": 61.8 ms a 4-token step so, 64.8-67.4 grouped) */
  const uint32_t sff = (uint32_t)g->ff_shexp, e = n * k, per = n > MV_MAX_TOKENS ? MV_MAX_TOKENS : 1;
  uint32_t multi, single;
  q4x_expert_bounds(e, E, per, &multi, &single);
  const GgufTensor *t[Q4X_GROUP_TENSORS] = {L->exp_gate, L->exp_up, L->exp_down};
  Q4xExpertTensors et;
  for (int i = 0; i < Q4X_GROUP_TENSORS; i++)
    et.base[i] = q4x_dev(f, t[i]), et.bytes[i] = t[i]->bytes / E, et.ptr[i] = b->ptr + (uint64_t)i * (multi + single);
  /* each expert read once for the step's tokens that picked it (q4x_expert_groups), its input rows gathered */
  return mat(f, L->router, E, D, b->mixed, D, n, b->rl, E) && q4x_route(b->rl, n, E, k, b->ids, b->w, b->rows, NULL) &&
         q4x_expert_groups(b->ids, e, k, E, per, &et, b->cnt, b->row, b->etok, b->where, NULL) &&
         q4x_gather(b->mixed, b->etok, e, D, b->xe, NULL) &&
         experts(t[0], et.ptr[0], b, multi, single, per, ff, D, b->xe, b->ga) &&
         experts(t[1], et.ptr[1], b, multi, single, per, ff, D, b->xe, b->up) &&
         q4x_swiglu(b->ga, b->up, (uint64_t)e * ff, b->ga, NULL) &&
         experts(t[2], et.ptr[2], b, multi, single, per, D, ff, b->ga, b->y) &&
         mat(f, L->sh_gate, sff, D, b->mixed, D, n, b->shg, sff) && mat(f, L->sh_up, sff, D, b->mixed, D, n, b->shu, sff) &&
         q4x_swiglu(b->shg, b->shu, (uint64_t)n * sff, b->shg, NULL) && mat(f, L->sh_down, D, sff, b->shg, sff, n, b->sh, D) &&
         mat(f, L->sh_gate_inp, 1, D, b->mixed, D, n, b->sl, 1) &&
         q4x_moe_sum(b->y, b->where, b->w, n, k, D, b->sh, b->sl, b->out, NULL);
}

/* the step's tokens' rows already in b.ple_rows */
bool q4x_ple(Q4xForward *f, Q4xSeq *s, int l, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xLayer *L = &f->w->layer[l];
  const GgufTensor *tab = f->w->ple_table;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, hc = (uint32_t)g->hc, hd = hc * D, heads = (uint32_t)g->ple_heads;
  return q4x_rows(TY(tab), q4x_dev(f, tab), tab->bytes / tab->dims[1], (uint32_t)tab->dims[0], b->ple_rows, n * heads,
                  b->emb, NULL) &&
         mat(f, L->ple_key, hd, D, b->emb, D, n, b->key, hd) && mat(f, L->ple_value, D, D, b->emb, D, n, b->val, D) &&
         q4x_hc_norm(b->key, fv(f, L->ple_norm_key), n, hc, D, g->rms_eps, b->kn, NULL) &&
         q4x_hc_norm(b->h, fv(f, L->ple_norm_query), n, hc, D, g->rms_eps, b->qn, NULL) &&
         q4x_ple_gate(b->kn, b->qn, b->val, n, hc, D, b->pg, b->gated, NULL) &&
         q4x_hc_norm(b->gated, fv(f, L->ple_norm_conv), n, hc, D, g->rms_eps, b->nrm, NULL) &&
         q4x_conv(b->nrm, n, hd, (uint32_t)g->ple_conv, (uint32_t)g->ple_ngram, fv(f, L->ple_conv), s->ple_hist,
                  Q4X_SLOTS, s->cur, b->conv, NULL) &&
         q4x_add2(b->h, b->gated, b->conv, (uint64_t)n * hd, NULL);
}
