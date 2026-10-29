#include <math.h>

#include <hip/hip_runtime_api.h>

#include "forward_impl.h"
#include "substrate/gpu/mv.h"
#include "substrate/gpu/ops.h"

#define FP4_LATENT_BLOCK 16 /* ref.py: the compressed rows' fp4_round(lat, 16, True) */
#define FP4_INDEX_BLOCK 32  /* and the indexer's keys and queries', fp4_round(x, 32, False) */

static const float *F(const Forward *f, const GgufTensor *t) { return (const float *)fwd_dev(f, t); }

/* a KV source's new compressed row, when the position completes one: the indexer's key from it (index sources), then
   the row itself, each roped at the row's first position and FP4-rounded */
static bool compress_row(Forward *f, Seq *s, int l, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t r = (uint32_t)g->compress[l], hd = g->head_dim, D = g->n_embd, RD = g->rope_dim, ih = g->ix_head_dim;
  const int p = s->pos;
  bool ok;
  if (r == 1)
    ok = mv_run(L->comp_kv->type, fwd_dev(f, L->comp_kv), hd, D, b->x, 1, b->lat, NULL);
  else
  {
    ok = mv_run(L->comp_kv->type, fwd_dev(f, L->comp_kv), hd, D, b->x, 1, s->pend_kvc[l] + (p % r) * hd, NULL) &&
         mv_run(L->comp_gate->type, fwd_dev(f, L->comp_gate), hd, D, b->x, 1, s->pend_sc[l] + (p % r) * hd, NULL);
    if (ok && (uint32_t)(p + 1) % r)
      return true; /* the row is not complete yet */
    ok = ok && ops_compress(s->pend_kvc[l], s->pend_sc[l], r, hd, b->lat, NULL);
  }
  const int32_t m = p / (int32_t)r, at = m * (int32_t)r;
  ok = ok && ops_rms(b->lat, hd, hd, F(f, L->comp_norm), g->rms_eps, 1, b->lat, hd, NULL) &&
       ((f->pin->rpos[l] = at), fwd_put_async(b->rpos, &f->pin->rpos[l], 4));
  if (ok && g->ix_source[l])
    ok = mv_run(L->ix_k->type, fwd_dev(f, L->ix_k), ih, hd, b->lat, 1, b->ikr, NULL) &&
         ops_rms(b->ikr, ih, ih, F(f, L->ix_k_norm), g->rms_eps, 1, b->ikr, ih, NULL) &&
         ops_rope(b->ikr, 1, ih, RD, ih, f->freqs_comp, b->rpos, false, 1, NULL) &&
         ops_fp4_round(b->ikr, ih, FP4_INDEX_BLOCK, false, NULL) &&
         fwd_copy(s->ik[l] + (uint64_t)m * ih, b->ikr, ih * sizeof(float));
  float *row = s->ckv[l] + (uint64_t)m * hd;
  ok = ok && fwd_copy(row, b->lat, hd * sizeof(float)) &&
       ops_rope(row, 1, hd, RD, hd, f->freqs_comp, b->rpos, false, 1, NULL) &&
       ops_fp4_round(row, hd, FP4_LATENT_BLOCK, true, NULL);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the compressor)");
  fwd_probe(f, "ckv", l, m, row, hd, false);
  return true;
}

/* an index source's rows for this position: its query against the source's keys, weighted a head, the top k */
static bool select_rows(Forward *f, Seq *s, int l, int src, uint32_t n_vis, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t nh = g->ix_heads, ih = g->ix_head_dim, D = g->n_embd, RD = g->rope_dim;
  if (!n_vis)
    return hipMemsetAsync(b->n_sel, 0, 4, NULL) == hipSuccess || fwd_fail(err, "a GPU call failed (the indexer)");
  const uint32_t blocks = (n_vis + g->cand_block - 1) / g->cand_block;
  if ((l == g->cand_source || g->cand_source < l) && blocks > (uint32_t)g->cand_blocks)
    return fwd_fail(err, "a context past the candidate blocks (the candidate masking is not built)");
  const float w_scale = (float)(pow(ih, -0.5) * pow(nh, -0.5));
  const bool ok = mv_run(L->ix_q_b->type, fwd_dev(f, L->ix_q_b), nh * ih, g->q_lora, b->qa, 1, b->qix, NULL) &&
                  ops_rope(b->qix, nh, ih, RD, nh * ih, f->freqs_comp, b->pos, false, 1, NULL) &&
                  ops_fp4_round(b->qix, nh * ih, FP4_INDEX_BLOCK, false, NULL) &&
                  mv_run(L->ix_proj->type, fwd_dev(f, L->ix_proj), nh, D, b->x, 1, b->wix, NULL) &&
                  ops_scale(b->wix, nh, w_scale, NULL) &&
                  ops_index_score(b->qix, s->ik[src], b->wix, n_vis, nh, ih, b->score, NULL) &&
                  ops_topk(b->score, n_vis, (uint32_t)g->ix_topk, b->sel, b->n_sel, NULL);
  return ok || fwd_fail(err, "a GPU call failed (the indexer)");
}

bool fwd_attention(Forward *f, Seq *s, int l, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t H = g->n_heads, HD = g->head_dim, RD = g->rope_dim, QL = g->q_lora, D = g->n_embd;
  const uint32_t OG = g->o_groups, OL = g->o_lora, oin = H * HD / OG, ratio = (uint32_t)g->compress[l];
  const float *freqs = ratio ? f->freqs_comp : f->freqs_win;
  const int p = s->pos;
  bool ok = mv_run(L->q_a->type, fwd_dev(f, L->q_a), QL, D, b->x, 1, b->qa, NULL) &&
            ops_rms(b->qa, QL, QL, F(f, L->q_a_norm), g->rms_eps, 1, b->qa, QL, NULL) &&
            mv_run(L->q_b->type, fwd_dev(f, L->q_b), H * HD, QL, b->qa, 1, b->q, NULL) &&
            ops_rope(b->q, H, HD, RD, H * HD, freqs, b->pos, false, 1, NULL) &&
            mv_run(L->kv->type, fwd_dev(f, L->kv), HD, D, b->x, 1, b->kv, NULL) &&
            ops_rms(b->kv, HD, HD, F(f, L->kv_norm), g->rms_eps, 1, b->kv, HD, NULL) &&
            ops_rope(b->kv, 1, HD, RD, HD, freqs, b->pos, false, 1, NULL) && ops_fp8_round(b->kv, HD, NULL);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (q, kv)");
  fwd_probe(f, "q", l, p, b->q, H * HD, false);
  fwd_probe(f, "kv", l, p, b->kv, HD, false);
  if (!fwd_copy(s->ring[l] + (uint64_t)(p % f->slots) * HD, b->kv, HD * sizeof(float)))
    return fwd_fail(err, "a GPU call failed (the window's keys)");
  const float *ckv = NULL;
  if (ratio)
  {
    const int src = g->kv_from[l];
    const uint32_t n_vis = (uint32_t)(p + 1) / ratio;
    if (g->kv_source[l] && !compress_row(f, s, l, err))
      return false;
    if (g->ix_source[l] && !select_rows(f, s, l, src, n_vis, err))
      return false;
    ckv = s->ckv[src];
    const uint32_t n_sel = n_vis < (uint32_t)g->ix_topk ? n_vis : (uint32_t)g->ix_topk;
    fwd_probe(f, "sel", l, p, b->sel, n_sel, true);
  }
  const float scale = (float)pow(HD, -0.5);
  ok = ops_sparse_attn(b->q, s->ring[l], f->slots, b->pos, g->window, ckv, ckv ? b->sel : NULL, ckv ? b->n_sel : NULL,
                       OPS_MAX_SELECT, F(f, L->sinks), scale, H, HD, 1, b->o, NULL);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the attention)");
  fwd_probe(f, "o", l, p, b->o, H * HD, false);
  ok = ops_rope(b->o, H, HD, RD, H * HD, freqs, b->pos, true, 1, NULL) && ops_fp8_round(b->o, H * HD, NULL) &&
       mv_group(L->out_a->type, f->oa_tab + (size_t)l * OG, OG, OL, oin, b->o, oin, b->oa, NULL) &&
       mv_run(L->out_b->type, fwd_dev(f, L->out_b), D, OG * OL, b->oa, 1, b->y, NULL);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the attention's output)");
  fwd_probe(f, "attn_out", l, p, b->y, D, false);
  return true;
}
