#include <math.h>

#include <hip/hip_runtime_api.h>

#include "forward_impl.h"

#define FP4_LATENT_BLOCK 16 /* ref.py: the compressed rows' fp4_round(lat, 16, True) */
#define FP4_INDEX_BLOCK 32  /* and the indexer's keys and queries', fp4_round(x, 32, False) */

static const float *F(const Forward *f, const GgufTensor *t) { return (const float *)fwd_dev(f, t); }

/* each run of one sequence's tokens (i, run) in turn: each(f, i, run, arg) */
static bool runs(Forward *f, bool (*each)(Forward *, uint32_t, uint32_t, void *), void *arg)
{
  bool ok = true;
  for (uint32_t i = 0, run; ok && i < f->n; i += run)
  {
    for (run = 1; i + run < f->n && f->bs[i + run] == f->bs[i]; run++)
      ;
    ok = each(f, i, run, arg);
  }
  return ok;
}

typedef struct
{
  int l;
  uint32_t r;
} Layer;

/* a run's positions into its sequence's pending rows, then the rows they complete (b->lat) */
static bool pend_run(Forward *f, uint32_t i, uint32_t run, void *arg)
{
  const Layer *a = arg;
  Scratch *b = &f->b;
  Seq *s = f->bs[i];
  const uint32_t hd = f->g->head_dim, pend = FWD_PEND_ROWS(f->step, a->r);
  return ops_put_rows(s->pend_kvc[a->l], pend, b->kvc + (uint64_t)i * hd, b->pos + i, 1, hd, run, NULL) &&
         ops_put_rows(s->pend_sc[a->l], pend, b->scg + (uint64_t)i * hd, b->pos + i, 1, hd, run, NULL) &&
         ops_compress(s->pend_kvc[a->l], s->pend_sc[a->l], pend, b->pos + i, a->r, hd, run, b->lat + (uint64_t)i * hd,
                      NULL);
}

/* a run's finished rows into its sequence's: the indexer's keys (index sources), then the rows */
static bool keep_run(Forward *f, uint32_t i, uint32_t run, void *arg)
{
  const Layer *a = arg;
  const Geometry *g = f->g;
  Scratch *b = &f->b;
  Seq *s = f->bs[i];
  const uint32_t hd = g->head_dim, ih = g->ix_head_dim, rows = f->max_ctx / a->r + 1;
  return (!g->ix_source[a->l] ||
          ops_put_rows(s->ik[a->l], rows, b->ikr + (uint64_t)i * ih, b->pos + i, a->r, ih, run, NULL)) &&
         ops_put_rows(s->ckv[a->l], rows, b->lat + (uint64_t)i * hd, b->pos + i, a->r, hd, run, NULL);
}

/* a KV source's compressor over the batch, every step batched: the projections; each sequence's positions into its
   pending rows and the rows they complete (a token whose position completes one: the ratio positions ending at it);
   the rows' indexer keys (index sources) and the rows, each roped at the row's first position and FP4-rounded, into
   their sequences' - before any token selects (a later token of the same sequence then sees them) */
static bool compress_rows(Forward *f, int l, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t r = (uint32_t)g->compress[l], hd = g->head_dim, D = g->n_embd, n = f->n;
  const uint32_t RD = g->rope_dim, ih = g->ix_head_dim;
  Layer a = {l, r};
  bool ok = fwd_mat(f, L->comp_kv, fwd_dev(f, L->comp_kv), hd, D, b->x, D, n, r == 1 ? b->lat : b->kvc, hd) &&
            (r == 1 || (fwd_mat(f, L->comp_gate, fwd_dev(f, L->comp_gate), hd, D, b->x, D, n, b->scg, hd) &&
                        runs(f, pend_run, &a))) &&
            ops_rms(b->lat, hd, hd, F(f, L->comp_norm), g->rms_eps, n, b->lat, hd, NULL);
  for (uint32_t i = 0; i < n; i++) /* a row's first position (a token completing none: unused) */
    f->pin->rpos[l][i] = f->pin->pos[i] + 1 - (int32_t)r;
  ok = ok && fwd_put_async(b->rpos, f->pin->rpos[l], (uint64_t)n * 4);
  if (ok && g->ix_source[l])
    ok = fwd_mat(f, L->ix_k, fwd_dev(f, L->ix_k), ih, hd, b->lat, hd, n, b->ikr, ih) &&
         ops_rms(b->ikr, ih, ih, F(f, L->ix_k_norm), g->rms_eps, n, b->ikr, ih, NULL) &&
         ops_rope(b->ikr, 1, ih, RD, ih, f->freqs_comp, b->rpos, false, n, NULL) &&
         ops_fp4_round(b->ikr, (uint64_t)n * ih, FP4_INDEX_BLOCK, false, NULL);
  ok = ok && ops_rope(b->lat, 1, hd, RD, hd, f->freqs_comp, b->rpos, false, n, NULL) &&
       ops_fp4_round(b->lat, (uint64_t)n * hd, FP4_LATENT_BLOCK, true, NULL) && runs(f, keep_run, &a);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the compressor)");
  for (uint32_t i = 0; f->probe && i < n; i++)
    if ((uint32_t)(f->pin->pos[i] + 1) % r == 0)
    {
      const int32_t m = (f->pin->pos[i] + 1) / (int32_t)r - 1;
      fwd_probe(f, "ckv", l, m, f->bs[i]->ckv[l] + (uint64_t)m * hd, hd, false);
    }
  return true;
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
  for (uint32_t i = 0; i < n; i++)
  {
    const uint32_t blocks = ((uint32_t)(f->pin->pos[i] + 1) / ratio + g->cand_block - 1) / g->cand_block;
    if ((l == g->cand_source || g->cand_source < l) && blocks > (uint32_t)g->cand_blocks)
      return fwd_fail(err, "a context past the candidate blocks (the candidate masking is not built)");
  }
  for (uint32_t i = 0, run; ok && i < n; i += run) /* a launch a run of one sequence's tokens (their keys its own) */
  {
    for (run = 1; i + run < n && f->bs[i + run] == f->bs[i]; run++)
      ;
    const uint32_t max_n = (uint32_t)(f->pin->pos[i + run - 1] + 1) / ratio; /* a run's positions ascend */
    ok = !max_n || (ops_index_score(b->qix + (uint64_t)i * nh * ih, f->bs[i]->ik[src], b->wix + (uint64_t)i * nh,
                                    b->pos + i, ratio, max_n, nh, ih, run, b->score + (uint64_t)i * f->score_stride,
                                    f->score_stride, NULL));
  }
  ok = ok && ops_topk(b->score, f->score_stride, b->pos, ratio, (uint32_t)g->ix_topk, n, b->sel, OPS_MAX_SELECT,
                      b->n_sel, NULL);
  return ok || fwd_fail(err, "a GPU call failed (the indexer)");
}

/* a run's keys into its sequence's window, as they are (a probe's: after it may have stood the oracle's in) */
static bool put_run(Forward *f, uint32_t i, uint32_t run, void *arg)
{
  const int l = *(const int *)arg;
  const uint32_t HD = f->g->head_dim;
  return ops_put_rows(f->bs[i]->ring[l], f->slots, f->b.kv + (uint64_t)i * HD, f->b.pos + i, 1, HD, run, NULL);
}

/* a run's keys finished - normed, roped, FP8 - and into its sequence's window, one launch (ops_kv_finish) */
static bool ring_run(Forward *f, uint32_t i, uint32_t run, void *arg)
{
  const int l = *(const int *)arg;
  const Geometry *g = f->g;
  const uint32_t HD = g->head_dim;
  const float *freqs = g->compress[l] ? f->freqs_comp : f->freqs_win;
  return ops_kv_finish(f->b.kv + (uint64_t)i * HD, HD, F(f, f->w->layer[l].kv_norm), g->rms_eps, g->rope_dim, freqs,
                       f->b.pos + i, f->bs[i]->ring[l], f->slots, run, NULL);
}

bool fwd_attention(Forward *f, int l, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t H = g->n_heads, HD = g->head_dim, RD = g->rope_dim, QL = g->q_lora, D = g->n_embd, n = f->n;
  const uint32_t OG = g->o_groups, OL = g->o_lora, oin = H * HD / OG, ratio = (uint32_t)g->compress[l];
  const float *freqs = ratio ? f->freqs_comp : f->freqs_win;
  /* q_a and kv on the same input, one launch on the x16 layer_in's mix left (ops_hc_pre_rms_x16; not with a probe,
     which may have stood the oracle's x in); q_a's norm writes q_b's x16 (ops_rms_x16) */
  bool ok = fwd_mat2(f, L->q_a, fwd_dev(f, L->q_a), QL, b->qa, QL, L->kv, fwd_dev(f, L->kv), HD, b->kv, HD, D, b->x, D,
                     n, !f->probe) &&
            ops_rms_x16(b->qa, QL, QL, F(f, L->q_a_norm), g->rms_eps, n, b->qa, QL, b->x16, b->overflow, NULL) &&
            fwd_mat_ready(f, L->q_b, fwd_dev(f, L->q_b), H * HD, QL, b->qa, QL, n, b->q, H * HD, true) &&
            ops_rope(b->q, H, HD, RD, H * HD, freqs, b->pos, false, n, NULL);
  /* every token's keys finished and into its sequence's window before any attends (a later one of a sequence reads
     an earlier's): one launch a run; with a probe the steps apart, the keys probed (or stood in) before the window
     takes them */
  if (f->probe)
  {
    ok = ok && ops_rms(b->kv, HD, HD, F(f, L->kv_norm), g->rms_eps, n, b->kv, HD, NULL) &&
         ops_rope(b->kv, 1, HD, RD, HD, freqs, b->pos, false, n, NULL) &&
         ops_fp8_round(b->kv, (uint64_t)n * HD, NULL);
    fwd_probe(f, "q", l, f->at, b->q, n * H * HD, false);
    fwd_probe(f, "kv", l, f->at, b->kv, n * HD, false);
    ok = ok && runs(f, put_run, &l);
  }
  else
    ok = ok && runs(f, ring_run, &l);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (q, the window's keys)");
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
    const int32_t floor = f->dec >= 0 && l > f->dec ? f->bs[i]->dec_from : 0; /* the decoder's: hip_skip.c */
    ok = ops_sparse_attn(b->q + (uint64_t)i * H * HD, f->bs[i]->ring[l], f->slots, b->pos + i, g->window, floor, ckv,
                         ckv ? b->sel + (uint64_t)i * OPS_MAX_SELECT : NULL, ckv ? b->n_sel + i : NULL, OPS_MAX_SELECT,
                         (uint32_t)g->ix_topk, F(f, L->sinks), scale, H, HD, run, b->att, b->o + (uint64_t)i * H * HD,
                         NULL);
  }
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the attention)");
  fwd_probe(f, "o", l, f->at, b->o, n * H * HD, false);
  /* wo_a: the tokens' attention to f16 once - its rope back and FP8 on the way, o read once (mm_x16_attn) - then
     every group in one launch, group k over its 4,096 of each (the groups' token counts the step's: forward_batch
     sends them) */
  ok = mm_x16_attn(b->o, H * HD, n, H * HD, HD, RD, freqs, b->pos, b->x16, b->overflow, NULL) &&
       mm_group(fwd_type(f, L->out_a), f->oa_tab + (size_t)l * OG, b->oa_cnt, OG, n, OL, oin, b->x16, f->oa_xoff,
                H * HD, b->oa, f->oa_yoff, OG * OL, NULL);
  ok = ok && fwd_mat(f, L->out_b, fwd_dev(f, L->out_b), D, OG * OL, b->oa, OG * OL, n, b->y, D);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the attention's output)");
  fwd_probe(f, "attn_out", l, f->at, b->y, n * D, false);
  return true;
}
