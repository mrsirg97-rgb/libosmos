#include <math.h>

#include <hip/hip_runtime_api.h>

#include "forward_impl.h"

#define FP4_LATENT_BLOCK 16 /* ref.py: the compressed rows' fp4_round(lat, 16, True) */
#define FP4_INDEX_BLOCK 32  /* and the indexer's keys and queries', fp4_round(x, 32, False) */

static const float *F(const Forward *f, const GgufTensor *t) { return (const float *)fwd_dev(f, t); }

/* a KV source's new compressed row m of token i's sequence, from its latent (lat, rms'd): the indexer's key from it
   (index sources), then the row itself, each roped at the row's first position and FP4-rounded */
static bool finish_row(Forward *f, int l, uint32_t i, int32_t m, const float *lat, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  Seq *s = f->bs[i];
  const uint32_t hd = g->head_dim, RD = g->rope_dim, ih = g->ix_head_dim;
  f->pin->rpos[l][i] = m * g->compress[l];
  bool ok = fwd_put_async(b->rpos + i, &f->pin->rpos[l][i], 4);
  if (ok && g->ix_source[l])
    ok = fwd_mat(f, L->ix_k, fwd_dev(f, L->ix_k), ih, hd, lat, hd, 1, b->ikr, ih) &&
         ops_rms(b->ikr, ih, ih, F(f, L->ix_k_norm), g->rms_eps, 1, b->ikr, ih, NULL) &&
         ops_rope(b->ikr, 1, ih, RD, ih, f->freqs_comp, b->rpos + i, false, 1, NULL) &&
         ops_fp4_round(b->ikr, ih, FP4_INDEX_BLOCK, false, NULL) &&
         fwd_copy(s->ik[l] + (uint64_t)m * ih, b->ikr, ih * sizeof(float));
  float *row = s->ckv[l] + (uint64_t)m * hd;
  ok = ok && fwd_copy(row, lat, hd * sizeof(float)) &&
       ops_rope(row, 1, hd, RD, hd, f->freqs_comp, b->rpos + i, false, 1, NULL) &&
       ops_fp4_round(row, hd, FP4_LATENT_BLOCK, true, NULL);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the compressor)");
  fwd_probe(f, "ckv", l, m, row, hd, false);
  return true;
}

/* a KV source's compressor over the batch: the projections batched, then each token in order - its position into
   its sequence's pending rows, a row finished when the position completes one (a later token of the same sequence
   then sees it) */
static bool compress_rows(Forward *f, int l, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t r = (uint32_t)g->compress[l], hd = g->head_dim, D = g->n_embd, n = f->n;
  bool ok = fwd_mat(f, L->comp_kv, fwd_dev(f, L->comp_kv), hd, D, b->x, D, n, r == 1 ? b->lat : b->kvc, hd) &&
            (r == 1 ? ops_rms(b->lat, hd, hd, F(f, L->comp_norm), g->rms_eps, n, b->lat, hd, NULL)
                    : fwd_mat(f, L->comp_gate, fwd_dev(f, L->comp_gate), hd, D, b->x, D, n, b->scg, hd));
  for (uint32_t i = 0; ok && i < n; i++)
  {
    const int32_t p = f->pin->pos[i];
    Seq *s = f->bs[i];
    if (r == 1)
    {
      ok = finish_row(f, l, i, p, b->lat + (uint64_t)i * hd, err);
      continue;
    }
    ok = fwd_copy(s->pend_kvc[l] + (p % r) * hd, b->kvc + (uint64_t)i * hd, hd * sizeof(float)) &&
         fwd_copy(s->pend_sc[l] + (p % r) * hd, b->scg + (uint64_t)i * hd, hd * sizeof(float));
    if (ok && (uint32_t)(p + 1) % r == 0)
    {
      float *lat = b->lat + (uint64_t)i * hd;
      ok = ops_compress(s->pend_kvc[l], s->pend_sc[l], r, hd, lat, NULL) &&
           ops_rms(lat, hd, hd, F(f, L->comp_norm), g->rms_eps, 1, lat, hd, NULL) &&
           finish_row(f, l, i, p / (int32_t)r, lat, err);
    }
  }
  if (!ok && !err[0])
    return fwd_fail(err, "a GPU call failed (the compressor)");
  return ok;
}

/* an index source's rows for each token: its query against its sequence's source's keys, weighted a head, the top k
   of the rows its position sees */
static bool select_rows(Forward *f, int l, int src, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t nh = g->ix_heads, ih = g->ix_head_dim, D = g->n_embd, RD = g->rope_dim, n = f->n;
  const uint32_t ratio = (uint32_t)g->compress[l];
  const float w_scale = (float)(pow(ih, -0.5) * pow(nh, -0.5));
  bool ok = fwd_mat(f, L->ix_q_b, fwd_dev(f, L->ix_q_b), nh * ih, g->q_lora, b->qa, g->q_lora, n, b->qix, nh * ih) &&
            ops_rope(b->qix, nh, ih, RD, nh * ih, f->freqs_comp, b->pos, false, n, NULL) &&
            ops_fp4_round(b->qix, (uint64_t)n * nh * ih, FP4_INDEX_BLOCK, false, NULL) &&
            fwd_mat(f, L->ix_proj, fwd_dev(f, L->ix_proj), nh, D, b->x, D, n, b->wix, nh) &&
            ops_scale(b->wix, (uint64_t)n * nh, w_scale, NULL);
  for (uint32_t i = 0; ok && i < n; i++)
  {
    const uint32_t n_vis = (uint32_t)(f->pin->pos[i] + 1) / ratio;
    int32_t *sel = b->sel + (uint64_t)i * OPS_MAX_SELECT;
    if (!n_vis)
    {
      ok = hipMemsetAsync(b->n_sel + i, 0, 4, NULL) == hipSuccess;
      continue;
    }
    const uint32_t blocks = (n_vis + g->cand_block - 1) / g->cand_block;
    if ((l == g->cand_source || g->cand_source < l) && blocks > (uint32_t)g->cand_blocks)
      return fwd_fail(err, "a context past the candidate blocks (the candidate masking is not built)");
    ok = ops_index_score(b->qix + (uint64_t)i * nh * ih, f->bs[i]->ik[src], b->wix + (uint64_t)i * nh, n_vis, nh, ih,
                         b->score, NULL) &&
         ops_topk(b->score, n_vis, (uint32_t)g->ix_topk, sel, b->n_sel + i, NULL);
  }
  return ok || fwd_fail(err, "a GPU call failed (the indexer)");
}

bool fwd_attention(Forward *f, int l, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t H = g->n_heads, HD = g->head_dim, RD = g->rope_dim, QL = g->q_lora, D = g->n_embd, n = f->n;
  const uint32_t OG = g->o_groups, OL = g->o_lora, oin = H * HD / OG, ratio = (uint32_t)g->compress[l];
  const float *freqs = ratio ? f->freqs_comp : f->freqs_win;
  bool ok = fwd_mat(f, L->q_a, fwd_dev(f, L->q_a), QL, D, b->x, D, n, b->qa, QL) &&
            ops_rms(b->qa, QL, QL, F(f, L->q_a_norm), g->rms_eps, n, b->qa, QL, NULL) &&
            fwd_mat(f, L->q_b, fwd_dev(f, L->q_b), H * HD, QL, b->qa, QL, n, b->q, H * HD) &&
            ops_rope(b->q, H, HD, RD, H * HD, freqs, b->pos, false, n, NULL) &&
            fwd_mat(f, L->kv, fwd_dev(f, L->kv), HD, D, b->x, D, n, b->kv, HD) &&
            ops_rms(b->kv, HD, HD, F(f, L->kv_norm), g->rms_eps, n, b->kv, HD, NULL) &&
            ops_rope(b->kv, 1, HD, RD, HD, freqs, b->pos, false, n, NULL) &&
            ops_fp8_round(b->kv, (uint64_t)n * HD, NULL);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (q, kv)");
  fwd_probe(f, "q", l, f->at, b->q, n * H * HD, false);
  fwd_probe(f, "kv", l, f->at, b->kv, n * HD, false);
  /* every token's keys into its sequence's window before any attends: a later one of a sequence reads an earlier's */
  for (uint32_t i = 0; ok && i < n; i++)
    ok = fwd_copy(f->bs[i]->ring[l] + (uint64_t)(f->pin->pos[i] % f->slots) * HD, b->kv + (uint64_t)i * HD,
                  HD * sizeof(float));
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the window's keys)");
  const int src = g->kv_from[l];
  if (ratio && g->kv_source[l] && !compress_rows(f, l, err))
    return false;
  if (ratio && g->ix_source[l] && !select_rows(f, l, src, err))
    return false;
  if (ratio)
  {
    const uint32_t vis = (uint32_t)(f->at + 1) / ratio;
    fwd_probe(f, "sel", l, f->at, b->sel, vis < (uint32_t)g->ix_topk ? vis : (uint32_t)g->ix_topk, true);
  }
  const float scale = (float)pow(HD, -0.5);
  for (uint32_t i = 0, run; ok && i < n; i += run) /* a launch a run of one sequence's tokens (a block a token) */
  {
    for (run = 1; i + run < n && f->bs[i + run] == f->bs[i]; run++)
      ;
    const float *ckv = ratio ? f->bs[i]->ckv[src] : NULL;
    ok = ops_sparse_attn(b->q + (uint64_t)i * H * HD, f->bs[i]->ring[l], f->slots, b->pos + i, g->window, ckv,
                         ckv ? b->sel + (uint64_t)i * OPS_MAX_SELECT : NULL, ckv ? b->n_sel + i : NULL, OPS_MAX_SELECT,
                         (uint32_t)g->ix_topk, F(f, L->sinks), scale, H, HD, run, b->att, b->o + (uint64_t)i * H * HD,
                         NULL);
  }
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the attention)");
  fwd_probe(f, "o", l, f->at, b->o, n * H * HD, false);
  ok = ops_rope(b->o, H, HD, RD, H * HD, freqs, b->pos, true, n, NULL) &&
       ops_fp8_round(b->o, (uint64_t)n * H * HD, NULL);
  /* wo_a: the tokens' attention to f16 once, then every group in one launch, group k over its 4,096 of each (the
     groups' token counts the step's: forward_batch sends them) */
  ok = ok && mm_x16(b->o, H * HD, n, H * HD, b->x16, b->overflow, NULL) &&
       mm_group(L->out_a->type, f->oa_tab + (size_t)l * OG, b->oa_cnt, OG, n, OL, oin, b->x16, f->oa_xoff, H * HD, b->oa,
                f->oa_yoff, OG * OL, NULL);
  ok = ok && fwd_mat(f, L->out_b, fwd_dev(f, L->out_b), D, OG * OL, b->oa, OG * OL, n, b->y, D);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the attention's output)");
  fwd_probe(f, "attn_out", l, f->at, b->y, n * D, false);
  return true;
}
