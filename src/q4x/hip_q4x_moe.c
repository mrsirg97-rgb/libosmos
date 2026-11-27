/* Qwen's MoE and PLE blocks (q4x_impl.h), llama.cpp's math: the routed experts (the
   membrane's, a host's tiles where a run is small), the shared expert, the n-gram layer's rows into b.h */
#include <math.h>
#include <string.h>

#include "osmos/mem/membrane.h"
#include "osmos/mem/moe_stage.h"
#include "osmos/model/moe_route.h"
#include "osmos/gpu/mm.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/q8.h"
#include "osmos/gpu/route.h"
#include "osmos/model/qtype.h"
#include "q4x_impl.h"
#include "osmos/err.h"

#define TY(t) ((t)->type)
#define mat(f, t, rows, cols, x, xs, n, y, ys) q4x_mat(f, t, rows, cols, x, xs, n, y, ys, false)

/* an expert tensor over the tables' runs, a launch each (mm_group: a part's slab decoded once for its 16 << r tokens) */
static bool experts(const GgufTensor *t, const void **ptr, const int32_t *cnt, const Q4xBuf *b,
                    const Q4xExpertBounds *bd, uint32_t rows, uint32_t cols, const uint32_t *xo, float *y,
                    const uint32_t *yo)
{
  bool ok = true;
  for (int r = 0; ok && r < Q4X_TILE_RUNS; r++)
    if (bd->count[r])
    {
      const uint32_t at = bd->first[r];
      ok = mm_group(TY(t), ptr + at, cnt + at, bd->count[r], MM_TILE << r, rows, cols, b->xe16, xo + at, cols, y,
                    yo + at, rows, NULL);
    }
  return ok;
}

/* a streamed layer's tables on the host, q4x_expert_tiles' layout (its bounds' runs: an entry a part of at most 128
   of an expert's rows, in the run of its tiles, experts ascending in each; slots -1 where padded): the picks by expert
   (moe_route), each run's expert from the membrane (visited - the GPU past the last layer's experts: the picks' copy
   waited for it - scored at a decode, held, waited for). false for a pick out of range, a refusal */
static int run_of(uint32_t c) { return c <= MM_TILE ? 0 : c <= 2 * MM_TILE ? 1 : c <= 4 * MM_TILE ? 2 : 3; }

/* the picks by expert on the host (moe_route), each run's expert from the membrane (visited - the GPU past the last
   layer's experts: the picks' copy waited for it - scored at a decode, every distinct pick held first, then each
   waited for: its addresses f->h_ex[run]) */
static bool host_route(Q4xForward *f, int l, uint32_t e, MoeRoute *rt)
{
  Q4xBuf *b = &f->b;
  const uint32_t E = (uint32_t)f->w->geo.n_experts, k = (uint32_t)f->w->geo.n_used;
  if (hipMemcpyAsync(f->h_pin_ids, b->ids, (size_t)e * 4, hipMemcpyDeviceToHost, NULL) != hipSuccess ||
      hipStreamSynchronize(NULL) != hipSuccess)
    return false;
  *rt = (MoeRoute){f->h_by, f->h_row, f->h_run, f->h_run + E, f->h_run + 2 * E, f->h_run + 3 * E, f->h_gidx, f->h_hidx,
                   f->h_est, 0, 0};
  char merr[OSMOS_ERR] = "";
  if (!moe_route(f->h_pin_ids, e, k, E, MM_TILE, rt) || !membrane_visit(f->mem, (uint32_t)l, e >= E, merr))
    return false;
  for (uint32_t m = 0; e < E && m < e; m++)
    membrane_score(f->mem, (uint32_t)l * E + (uint32_t)f->h_pin_ids[m]);
  for (uint32_t j = 0; j < rt->runs; j++) /* the cold ones' reads under way together */
    if (!membrane_hold(f->mem, (uint32_t)l * E + rt->expert[j], merr))
      return false;
  for (uint32_t j = 0; j < rt->runs; j++)
    if (!membrane_wait(f->mem, (uint32_t)l * E + rt->expert[j], f->h_ex[j], merr))
      return false;
  return true;
}

static bool host_tiles(Q4xForward *f, int l, uint32_t e, Q4xExpertBounds *bd, Q4xExpertTiles *et, uint32_t D, uint32_t ff)
{
  const uint32_t part = MM_TILE << (Q4X_TILE_RUNS - 1);
  MoeRoute rt;
  if (!host_route(f, l, e, &rt))
    return false;
  uint32_t rcnt[Q4X_TILE_RUNS] = {0}, ne = 0;
  for (uint32_t j = 0; j < rt.runs; j++)
    for (uint32_t off = 0; off < rt.count[j]; off += part)
    {
      const uint32_t c = rt.count[j] - off < part ? rt.count[j] - off : part;
      rcnt[run_of(c)]++, ne++;
    }
  uint32_t at[Q4X_TILE_RUNS];
  for (int r = 0, pos = 0; r < Q4X_TILE_RUNS; r++)
    bd->first[r] = at[r] = (uint32_t)pos, bd->count[r] = rcnt[r], pos += (int)rcnt[r];
  bd->entries = ne, bd->slots = rt.slots;
  MoeStage st = {f->moe_build, f->moe_dev, 0, f->moe_cap};
  int32_t *cnt = moe_stage_carve(&st, (uint64_t)ne * 4, (void **)&et->cnt);
  uint32_t *xg = moe_stage_carve(&st, (uint64_t)ne * 4, (void **)&et->xg);
  uint32_t *yg = moe_stage_carve(&st, (uint64_t)ne * 4, (void **)&et->yg);
  uint32_t *xd = moe_stage_carve(&st, (uint64_t)ne * 4, (void **)&et->xd);
  uint32_t *yd = moe_stage_carve(&st, (uint64_t)ne * 4, (void **)&et->yd);
  const void **ptr[Q4X_GROUP_TENSORS];
  for (int t = 0; t < Q4X_GROUP_TENSORS; t++)
    ptr[t] = moe_stage_carve(&st, (uint64_t)ne * sizeof *ptr[t], (void **)&et->ptr[t]);
  int32_t *gidx = moe_stage_carve(&st, (uint64_t)rt.slots * 4, (void **)&et->gidx);
  int32_t *hidx = moe_stage_carve(&st, (uint64_t)rt.slots * 4, (void **)&et->hidx);
  int32_t *where = moe_stage_carve(&st, (uint64_t)e * 4, (void **)&et->where);
  if (!cnt || !xg || !yg || !xd || !yd || !ptr[0] || !ptr[1] || !ptr[2] || !gidx || !hidx || !where)
    return false;
  for (uint32_t j = 0; j < rt.runs; j++)
  {
    for (uint32_t off = 0; off < rt.count[j]; off += part)
    {
      const uint32_t c = rt.count[j] - off < part ? rt.count[j] - off : part, en = at[run_of(c)]++;
      const uint32_t slot = rt.slot0[j] + off, place = rt.start[j] + off;
      cnt[en] = (int32_t)c, xg[en] = slot * D, xd[en] = slot * ff, yg[en] = place * ff, yd[en] = place * D;
      for (int t = 0; t < Q4X_GROUP_TENSORS; t++)
        ptr[t][en] = f->h_ex[j][t];
    }
  }
  memcpy(gidx, f->h_gidx, (size_t)rt.slots * 4);
  memcpy(hidx, f->h_hidx, (size_t)rt.slots * 4);
  memcpy(where, f->h_row, (size_t)e * 4);
  memcpy(f->moe_pin, f->moe_build, st.used);
  return hipMemcpyAsync(f->moe_dev, f->moe_pin, st.used, hipMemcpyHostToDevice, NULL) == hipSuccess;
}

/* a streamed layer's int8 tables on the host (a decode's: n <= MV_MAX_TOKENS tokens, an expert's picks one entry):
   host_route's runs sorted descending by count, staged compact via MoeStage in one async pinned copy */
static bool host_q8(Q4xForward *f, int l, uint32_t n, uint32_t e, Q8Experts *q8n, const int32_t **sidx_out)
{
  const uint32_t k = (uint32_t)f->w->geo.n_used;
  MoeRoute rt;
  if (!host_route(f, l, e, &rt))
    return false;
  const uint32_t R = rt.runs;
  MoeStage st = {f->moe_build, f->moe_dev, 0, f->moe_cap};
  int32_t *dcnt, *dxrow, *dgidx, *dsidx;
  const void **dwg, **dwu, **dwd;
  int32_t *cnt = moe_stage_carve(&st, (uint64_t)R * 4, (void **)&dcnt);
  int32_t *rows = moe_stage_carve(&st, (uint64_t)R * 4, (void **)&dxrow);
  int32_t *tok = moe_stage_carve(&st, (uint64_t)e * 4, (void **)&dgidx);
  int32_t *sidx = moe_stage_carve(&st, (uint64_t)e * 4, (void **)&dsidx);
  const void **wg = moe_stage_carve(&st, (uint64_t)R * sizeof *wg, (void **)&dwg);
  const void **wu = moe_stage_carve(&st, (uint64_t)R * sizeof *wu, (void **)&dwu);
  const void **wd = moe_stage_carve(&st, (uint64_t)R * sizeof *wd, (void **)&dwd);
  if (!cnt || !rows || !tok || !sidx || !wg || !wu || !wd)
    return false;
  uint32_t m = 0;
  for (uint32_t c = n; c >= 1; c--)
    for (uint32_t j = 0; j < R; j++)
      if (rt.count[j] == c)
      {
        f->h_cnt[m] = cnt[m] = (int32_t)c, rows[m] = (int32_t)rt.start[j];
        wg[m] = f->h_ex[j][0], wu[m] = f->h_ex[j][1], wd[m] = f->h_ex[j][2], m++;
      }
  if (m != R)
    return false;
  for (uint32_t q = 0; q < e; q++)
    tok[q] = (int32_t)(rt.by[q] / k), sidx[q] = (int32_t)rt.row[q];
  memcpy(f->moe_pin, f->moe_build, st.used);
  if (hipMemcpyAsync(f->moe_dev, f->moe_pin, st.used, hipMemcpyHostToDevice, NULL) != hipSuccess)
    return false;
  q8n->wg = dwg, q8n->wu = dwu, q8n->wd = dwd, q8n->cnt = dcnt, q8n->xrow = dxrow, q8n->xin = dgidx;
  q8n->n = R, q8n->h_cnt = f->h_cnt, q8n->by_count = true;
  *sidx_out = dsidx;
  return true;
}

void q4x_forward_reading(Q4xForward *f, bool on)
{
  if (f)
    f->reading = on;
}

bool q4x_forward_int8(const Q4xForward *f) { return f && !f->no_int8 && q8_enabled(); }

bool q4x_moe(Q4xForward *f, const Q4xLayer *L, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, E = (uint32_t)g->n_experts, k = (uint32_t)g->n_used, ff = (uint32_t)g->ff_exp;
  const uint32_t e = n * k;
  Q4xExpertBounds bd;
  q4x_expert_bounds(e, E, &bd);
  const GgufTensor *t[Q4X_GROUP_TENSORS] = {L->exp_gate, L->exp_up, L->exp_down};
  Q4xExpertTiles et = {.cnt = b->cnt, .gidx = b->gidx, .hidx = b->hidx, .xg = b->xg, .yg = b->yg, .xd = b->xd,
                       .yd = b->yd, .where = b->where, .off = b->eoff};
  /* a layer of the model's with its experts on the membrane: their addresses the table's (the MTP head's layer, or a
     forward without one: its tensors' own) */
  const bool model_layer = L >= f->w->layer && L < f->w->layer + g->n_layers;
  const bool tab = f->exp_tab && model_layer, streamed = f->mem && !f->exp_tab && model_layer;
  for (int i = 0; i < Q4X_GROUP_TENSORS; i++)
  {
    et.ptr[i] = b->ptr + (uint64_t)i * bd.entries;
    if (tab)
      et.tab[i] = (const void *const *)f->exp_tab + ((uint64_t)(L - f->w->layer) * Q4X_GROUP_TENSORS + (uint64_t)i) * E;
    else if (!streamed)
      et.base[i] = q4x_dev(f, t[i]), et.bytes[i] = t[i]->bytes / E;
  }
  if (!mat(f, L->router, E, D, b->mixed, D, n, b->rl, E) || !q4x_route(b->rl, n, E, k, b->ids, b->w, b->rows, NULL))
    return false;
  /* a decode's or a verify's experts on int8 (q8_experts): every expert in memory route_tok's tables (each expert's
     picks one entry; its fault - a pick out of range or twice, q4x_route's never - b.overflow's: the step refused),
     streamed the host's (host_q8: the membrane's addresses); each token's row
     rounded once, read by token; the activation unweighted (Qwen's weights at the sum), every row's output where sidx
     puts its pick. a prompt's on the matrix cores (f16, below): the two computed apart */
  if (n <= MV_MAX_TOKENS && !f->reading && (tab || streamed) && !f->no_int8 && q8_enabled() && TY(t[0]) == TY(t[1]) &&
      q8_takes(TY(t[0])) && q8_takes(TY(t[2])) && D % 32 == 0 && ff % 32 == 0)
  {
    const uint32_t chunk = n;
    Q8Experts q8n = {.gate_type = TY(t[0]), .up_type = TY(t[1]), .down_type = TY(t[2]), .wg = b->q_wptr,
                     .wu = b->q_wptr + e, .wd = b->q_wptr + 2 * (uint64_t)e, .cnt = b->q_cnt, .xrow = b->q_xrow,
                     .xin = b->q_gidx, .n = e, .max_cnt = chunk, .by_count = true, .d = D, .ff = ff, .x = b->mixed,
                     .x_rows = n, .xq = b->q_x, .xqd = b->q_xd, .lim = INFINITY, .ga = b->ga, .up = b->up,
                     .h = b->ga, .h_rows = e, .hq = b->q_h, .hqd = b->q_hd, .y = b->y};
    const int32_t *sidx = b->q_sidx;
    const bool tables = tab ? route_tok(b->ids, b->w, n, k, E, chunk, et.tab[0], b->q_gidx, b->q_sidx, b->q_wsel,
                                        b->q_cnt, b->q_xrow, b->q_wptr, b->overflow, NULL)
                            : host_q8(f, (int)(L - f->w->layer), n, e, &q8n, &sidx);
    const bool wait_sh = f->moe_side;
    f->moe_side = false;
    return tables && q8_experts(&q8n, NULL) &&
           (!wait_sh || hipStreamWaitEvent(NULL, f->e_out, 0) == hipSuccess) && /* the shared expert's done */
           q4x_moe_sum(b->y, sidx, b->w, n, k, D, b->sh, b->sl, b->out, NULL);
  }
  /* each expert read once for up to 128 of the step's tokens that picked it (q4x_expert_tiles, mm_group) */
  if ((streamed ? !host_tiles(f, (int)(L - f->w->layer), e, &bd, &et, D, ff)
                : !q4x_expert_tiles(b->ids, e, k, E, D, ff, &et, NULL)))
    return false;
  /* the NPU's share of a prompt step's whole entries, taken from the tables (hip_q4x_npu.c): each kind's packed
     before the GPU's matmuls of it are queued, run beside them, its rows where theirs go */
  const int npu_res = q4x_npu_take(f, L, n, &bd, D, ff);
  if (npu_res < 0)
    return false;
  const bool npu = npu_res > 0;
  const bool wait_sh = f->moe_side;
  f->moe_side = false;
  return mm_x16_rows(b->mixed, D, et.gidx, bd.slots, D, b->xe16, b->overflow, NULL) &&
         experts(t[0], et.ptr[0], et.cnt, b, &bd, ff, D, et.xg, b->ga, et.yg) &&
         experts(t[1], et.ptr[1], et.cnt, b, &bd, ff, D, et.xg, b->up, et.yg) &&
         (!npu || q4x_npu_gate_up(f, L, D, ff)) &&
         q4x_swiglu_rows(b->ga, b->up, ff, et.hidx, bd.slots, b->xe16, b->overflow, NULL) &&
         (!npu || q4x_npu_pack_down(f, ff)) &&
         experts(t[2], et.ptr[2], et.cnt, b, &bd, D, ff, et.xd, b->y, et.yd) &&
         (!npu || q4x_npu_down(f, L, D, ff)) &&
         (!wait_sh || hipStreamWaitEvent(NULL, f->e_out, 0) == hipSuccess) && /* the shared expert's done */
         q4x_moe_sum(b->y, et.where, b->w, n, k, D, b->sh, b->sl, b->out, NULL);
}

/* the shared expert and its gate's logit (b.sh, b.sl) - after the FFN's mixer, b.x16 its mixed (q4x_mix): a token's
   own, so with the layer's part (a step past MM_MAX_TOKENS: a sub-batch at a time, b.x16 the mixer's still) */
bool q4x_moe_shared(Q4xForward *f, const Q4xLayer *L, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, sff = (uint32_t)g->ff_shexp;
  Q4xX16 sd;
  /* a small step's (decode, a draft's verify): the shared expert on the second stream, beside the routed experts -
     both latency-bound at a few tokens, together nearer the memory's bandwidth. it reads the mixer's outputs alone
     (b.mixed, b.x16) and writes b.sh, b.sl and its own f16 into b.x16; skip s2 if L->router would also use b.x16
     via mm_tokens on the main stream */
  const bool router_uses_x16 = mm_known(TY(L->router)) && mm_fits(TY(L->router), (uint32_t)g->n_experts, D);
  const bool side_stream = n <= Q4X_SIDE_STEP && !router_uses_x16;
  f->moe_side = false;
  if (side_stream && (hipEventRecord(f->e_in, NULL) != hipSuccess || hipStreamWaitEvent(f->s2, f->e_in, 0) != hipSuccess))
    return false;
  if (side_stream)
    f->st = f->s2;
  if (side_stream && f->side_spin && !q4x_spin(f->side_spin, f->s2)) /* side_spin: the shared expert last */
  {
    f->st = NULL;
    return false;
  }
  const bool ok = q4x_mat2(f, L->sh_gate, sff, b->shg, sff, L->sh_up, sff, b->shu, sff, D, b->mixed, D, n,
                           n <= MM_MAX_TOKENS) &&
                  q4x_swiglu(b->shg, b->shu, (uint64_t)n * sff, b->shg, q4x_side(f, n, sff, &sd), f->st) &&
                  q4x_mat(f, L->sh_down, D, sff, b->shg, sff, n, b->sh, D, n <= MM_MAX_TOKENS) &&
                  mat(f, L->sh_gate_inp, 1, D, b->mixed, D, n, b->sl, 1);
  f->st = NULL;
  if (ok && side_stream)
  {
    if (hipEventRecord(f->e_out, f->s2) != hipSuccess)
      return false;
    f->moe_side = true;
  }
  return ok;
}

/* the step's tokens' rows already in b.ple_rows */
/* stream_ngram: the step's n tokens' rows (b.pin_rows', ple_heads a token) begun on the readers into the stage - once
   the last step's copy of it is done (ple_copied); a step that failed before its n-gram layer: its rows waited for first */
bool q4x_ple_issue(Q4xForward *f, uint32_t n)
{
  const GgufTensor *tab = f->w->ple_table;
  const uint64_t rb = tab->bytes / tab->dims[1];
  const uint32_t rows = n * (uint32_t)f->w->geo.ple_heads;
  if (!rowstream_wait(&f->ple_rs) || hipEventSynchronize(f->ple_copied) != hipSuccess) /* (a failed step's batch first) */
    return false;
  for (uint32_t i = 0; i < rows; i++)
    f->ple_off[i] = tab->offset + (uint64_t)f->b.pin_rows[i] * rb;
  f->ple_rs.read = f->ple_read, f->ple_rs.fd = f->ple_fd, f->ple_rs.off = f->ple_off, f->ple_rs.stage = f->ple_stage;
  if (!rowstream_issue(&f->ple_rs, 0, rows, rb))
    return false;
  f->ple_n = rows;
  return true;
}

/* the n-gram table's rows of the n tokens: the held table's own, or (stream_ngram) the stage's - read for the step,
   copied in whole at its first part (sub-batch 0: b's views at their start), each part's in its view of ple_sbuf */
static bool ple_table_rows(Q4xForward *f, uint32_t n)
{
  const GgufTensor *tab = f->w->ple_table;
  Q4xBuf *b = &f->b;
  const uint64_t rb = tab->bytes / tab->dims[1];
  const uint32_t heads = (uint32_t)f->w->geo.ple_heads;
  if (!f->stream_ngram)
    return q4x_rows(TY(tab), q4x_dev(f, tab), rb, (uint32_t)tab->dims[0], b->ple_rows, n * heads, b->emb, NULL);
  if (f->ple_rs.pending) /* (the step's first part: its rows waited for and copied in whole) */
  {
    if (!rowstream_wait(&f->ple_rs) ||
        hipMemcpyAsync(b->ple_sbuf, f->ple_stage, (uint64_t)f->ple_n * rb, hipMemcpyHostToDevice, NULL) != hipSuccess ||
        hipEventRecord(f->ple_copied, NULL) != hipSuccess)
      return false;
  }
  return q4x_rows(TY(tab), b->ple_sbuf, rb, (uint32_t)tab->dims[0], b->ple_iota, n * heads, b->emb, NULL);
}

bool q4x_ple(Q4xForward *f, const Q4xRun *r, uint32_t runs, int l, uint32_t n)
{
  const Q4xGeometry *g = &f->w->geo;
  const Q4xLayer *L = &f->w->layer[l];
  Q4xBuf *b = &f->b;
  const uint32_t D = (uint32_t)g->n_embd, hc = (uint32_t)g->hc, hd = hc * D;
  bool ok = ple_table_rows(f, n) &&
         q4x_mat2(f, L->ple_key, hd, b->key, hd, L->ple_value, D, b->val, D, D, b->emb, D, n, false) &&
         q4x_hc_norm(b->key, q4x_fv(f, L->ple_norm_key), n, hc, D, g->rms_eps, b->kn, NULL, NULL) &&
         q4x_hc_norm(b->h, q4x_fv(f, L->ple_norm_query), n, hc, D, g->rms_eps, b->qn, NULL, NULL) &&
         q4x_ple_gate(b->kn, b->qn, b->val, n, hc, D, b->pg, b->gated, NULL) &&
         q4x_hc_norm(b->gated, q4x_fv(f, L->ple_norm_conv), n, hc, D, g->rms_eps, b->nrm, NULL, NULL);
  for (uint32_t i = 0; ok && i < runs; i++) /* a sequence's conv history over its own tokens */
  {
    const uint64_t t0 = r[i].t0 * (uint64_t)hd;
    ok = q4x_conv(b->nrm + t0, r[i].n, hd, (uint32_t)g->ple_conv, (uint32_t)g->ple_ngram, q4x_fv(f, L->ple_conv),
                  r[i].s->ple_hist, Q4X_SLOTS, r[i].s->cur, b->conv + t0, NULL);
  }
  return ok && q4x_add2(b->h, b->gated, b->conv, (uint64_t)n * hd, NULL);
}
