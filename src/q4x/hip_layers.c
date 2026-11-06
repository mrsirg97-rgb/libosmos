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

bool q4x_attention(Q4xForward *f, Q4xSeq *s, int l, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xLayer *L = &f->w->layer[l];
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, heads = (uint32_t)g->n_heads, kvh = (uint32_t)g->n_kv_heads;
  const uint32_t hd = (uint32_t)g->head_dim, q = heads * hd, kv = kvh * hd;
  float *kc = s->kc[l] + (uint64_t)s->pos * kv, *vc = s->vc[l] + (uint64_t)s->pos * kv;
  return mat(f, L->q, 2 * q, D, b->mixed, D, n, b->qf, 2 * q) && mat(f, L->k, kv, D, b->mixed, D, n, b->kr, kv) &&
         mat(f, L->v, kv, D, b->mixed, D, n, vc, kv) &&
         q4x_qk(b->qf, b->kr, n, heads, kvh, hd, (uint32_t)g->rope_dim, fv(f, L->q_norm), fv(f, L->k_norm), g->rms_eps,
                s->pos, g->rope_base, b->q, b->gt, kc, NULL) &&
         q4x_attn(b->q, s->kc[l], s->vc[l], n, heads, kvh, hd, (uint32_t)s->pos, b->gt, b->ao, NULL) &&
         mat(f, L->out, D, q, b->ao, q, n, b->out, D);
}

bool q4x_moe(Q4xForward *f, int l, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xLayer *L = &f->w->layer[l];
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, E = (uint32_t)g->n_experts, k = (uint32_t)g->n_used, ff = (uint32_t)g->ff_exp;
  const uint32_t sff = (uint32_t)g->ff_shexp, e = n * k;
  const struct
  {
    const GgufTensor *t;
    uint32_t rows, cols;
    const float *x;
    float *y;
  } ex[] = {{L->exp_gate, ff, D, b->xe, b->ga}, {L->exp_up, ff, D, b->xe, b->up}};
  bool ok = mat(f, L->router, E, D, b->mixed, D, n, b->rl, E) && q4x_route(b->rl, n, E, k, b->ids, b->w, b->rows, NULL) &&
            q4x_gather(b->mixed, b->rows, e, D, b->xe, NULL);
  for (int i = 0; ok && i < 2; i++)
    ok = q4x_expert_table(b->ids, e, q4x_dev(f, ex[i].t), ex[i].t->bytes / E, b->ptr, b->cnt, b->row, NULL) &&
         mv_group_tok(TY(ex[i].t), b->ptr, b->cnt, b->row, e, 1, ex[i].rows, ex[i].cols, ex[i].x, ex[i].cols, ex[i].y,
                      ex[i].rows, NULL);
  return ok && q4x_swiglu(b->ga, b->up, (uint64_t)e * ff, b->ga, NULL) &&
         q4x_expert_table(b->ids, e, q4x_dev(f, L->exp_down), L->exp_down->bytes / E, b->ptr, b->cnt, b->row, NULL) &&
         mv_group_tok(TY(L->exp_down), b->ptr, b->cnt, b->row, e, 1, D, ff, b->ga, ff, b->y, D, NULL) &&
         mat(f, L->sh_gate, sff, D, b->mixed, D, n, b->shg, sff) && mat(f, L->sh_up, sff, D, b->mixed, D, n, b->shu, sff) &&
         q4x_swiglu(b->shg, b->shu, (uint64_t)n * sff, b->shg, NULL) && mat(f, L->sh_down, D, sff, b->shg, sff, n, b->sh, D) &&
         mat(f, L->sh_gate_inp, 1, D, b->mixed, D, n, b->sl, 1) &&
         q4x_moe_sum(b->y, b->w, n, k, D, b->sh, b->sl, b->out, NULL);
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
