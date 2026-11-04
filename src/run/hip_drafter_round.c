#include <math.h>

#include "drafter_impl.h"

/* a round's three blocks on its DRAFTER_BLOCK rows (DeepSeek's DSparkBlock: the trunk's layer in small) */

static bool attention(Drafter *d, const DBlock *k, const uint8_t *ring)
{
  const Geometry *g = d->f->g;
  const uint32_t T = DRAFTER_BLOCK, D = g->n_embd, H = g->n_heads, HD = g->head_dim, RD = g->rope_dim;
  const uint32_t QL = g->q_lora, G = g->o_groups, OL = g->o_lora, oin = H * HD / G, W = (uint32_t)g->window;
  const float *fr = d->f->freqs_win;
  return mv_strided(DRAFTER_Q8_0, k->wq_a, QL, D, d->x, D, T, d->qa, QL, NULL) &&
         ops_rms(d->qa, QL, QL, k->q_norm, g->rms_eps, T, d->qa, QL, NULL) &&
         mv_strided(DRAFTER_Q8_0, k->wq_b, H * HD, QL, d->qa, QL, T, d->q, H * HD, NULL) &&
         ops_rope(d->q, H, HD, RD, H * HD, fr, d->pos_q, false, T, NULL) &&
         mv_strided(DRAFTER_Q8_0, k->wkv, HD, D, d->x, D, T, d->kv, HD, NULL) &&
         ops_rms(d->kv, HD, HD, k->kv_norm, g->rms_eps, T, d->kv, HD, NULL) &&
         ops_rope(d->kv, 1, HD, RD, HD, fr, d->pos_q, false, T, NULL) && ops_fp8_round(d->kv, (uint64_t)T * HD, NULL, 0, NULL) &&
         /* the ring's rows before p0 (every row's window: positions p0 - window .. p0 - 1), then the block's own T
            rows - every row's, no causal mask - as the selected rows */
         ops_sparse_attn(d->q, ring, KV_FP8, W, d->pos_a, W, 0, d->kv, KV_F32, d->sel, d->n_sel, OPS_MAX_SELECT, T, k->sink,
                         (float)(1.0 / sqrt((double)HD)), H, HD, T, d->att, d->o, NULL) &&
         ops_rope(d->o, H, HD, RD, H * HD, fr, d->pos_q, true, T, NULL) &&
         ops_fp8_round(d->o, (uint64_t)T * H * HD, NULL, 0, NULL) &&
         mv_group(DRAFTER_Q8_0, k->oa_tab, T * G, OL, oin, d->o, oin, 1, d->oa, NULL) &&
         mv_strided(DRAFTER_Q8_0, k->wo_b, D, G * OL, d->oa, G * OL, T, d->y, D, NULL);
}

/* the routed experts (DRAFTER_TOPK a row, their tables gathered on the device from the router's picks: 8-byte
   pointers as rows of two floats, a copy of their bits) and the shared one */
static bool moe(Drafter *d, const DBlock *k)
{
  const Geometry *g = d->f->g;
  const uint32_t T = DRAFTER_BLOCK, D = g->n_embd, FF = g->expert_ff, P = DRAFTER_PAIRS, K = DRAFTER_TOPK;
  const uint32_t ptr = sizeof(void *) / sizeof(float);
  bool ok = mv_strided(DRAFTER_BF16, k->gate, DRAFTER_EXPERTS, D, d->xf, D, T, d->logits, DRAFTER_EXPERTS, NULL) &&
            ops_router(d->logits, k->gate_bias, DRAFTER_EXPERTS, K, g->expert_scale, T, d->ridx, d->rw, NULL);
  for (int r = 0; ok && r < 3; r++)
    ok = ops_gather_rows((float *)(d->tab + (uint64_t)r * P), (const float *)k->ex[r], d->ridx, P, ptr, NULL);
  return ok && mv_group(d->ex_type, d->tab, P, FF, D, d->xf, D, K, d->ga, NULL) &&
         mv_group(d->ex_type, d->tab + P, P, FF, D, d->xf, D, K, d->up, NULL) &&
         ops_swiglu(d->ga, d->up, FF, DRAFTER_SWIGLU_LIMIT, d->rw, P, d->hh, NULL) &&
         mv_group(d->ex_type, d->tab + 2 * P, P, D, FF, d->hh, FF, 1, d->ye, NULL) &&
         ops_gather_sum(d->moe, d->ye, d->pair, K, T, D, NULL) &&
         mv_strided(DRAFTER_Q8_0, k->sh1, FF, D, d->xf, D, T, d->sga, FF, NULL) &&
         mv_strided(DRAFTER_Q8_0, k->sh3, FF, D, d->xf, D, T, d->sup, FF, NULL) &&
         ops_swiglu(d->sga, d->sup, FF, DRAFTER_SWIGLU_LIMIT, NULL, T, d->shh, NULL) &&
         mv_strided(DRAFTER_Q8_0, k->sh2, D, FF, d->shh, FF, T, d->sye, D, NULL) &&
         ops_add(d->moe, d->sye, (uint64_t)T * D, NULL);
}

bool drafter_blocks(Drafter *d, Seq *s, int32_t p0, char *err)
{
  (void)p0; /* the positions are d->pos_q / pos_a's, put by the caller */
  Forward *f = d->f;
  const Geometry *g = f->g;
  const uint32_t T = DRAFTER_BLOCK, D = g->n_embd, hc = g->hc, W = (uint32_t)g->window, hd = g->head_dim;
  bool ok = ops_embed(f->w->embed->type, fwd_dev(f, f->w->embed), D, d->ids, T, hc, d->h, NULL);
  const float *pre = d->pre0; /* the first block's: stream 0 */
  for (int b = 0; ok && b < DRAFTER_BLOCKS; b++)
  {
    const DBlock *k = &d->b[b];
    ok = ops_hc_dot(k->hca_fn, d->h, hc, D, T, d->hcp, NULL) &&
         ops_hc_mix(d->hcp, k->hca_scale, k->hca_base, hc, D, g->rms_eps, g->hc_eps, g->sinkhorn_iters, T, d->a_pre,
                    d->a_post, d->a_comb, NULL) &&
         ops_hc_pre_rms(d->h, pre, hc, D, k->attn_norm, g->rms_eps, T, d->x, NULL) &&
         attention(d, k, s->dring + (uint64_t)b * W * kv_row_bytes(KV_FP8, hd)) &&
         ops_hc_post(d->y, d->h, d->a_post, d->a_comb, hc, D, T, d->mid, NULL) &&
         ops_hc_dot(k->hcf_fn, d->mid, hc, D, T, d->hcp, NULL) &&
         ops_hc_mix(d->hcp, k->hcf_scale, k->hcf_base, hc, D, g->rms_eps, g->hc_eps, g->sinkhorn_iters, T, d->f_pre,
                    d->f_post, d->f_comb, NULL) &&
         ops_hc_pre_rms(d->mid, d->a_pre, hc, D, k->ffn_norm, g->rms_eps, T, d->xf, NULL) && moe(d, k) &&
         ops_hc_post(d->moe, d->mid, d->f_post, d->f_comb, hc, D, T, d->h, NULL);
    pre = d->f_pre; /* the next block's attention pre: this one's FFN mix (the trunk's lag) */
  }
  ok = ok && ops_hc_pre(d->h, d->f_pre, hc, D, T, d->xc, NULL);
  return ok || fwd_fail(err, "a GPU call failed (the drafter's blocks)");
}
