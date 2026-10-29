#include <stdlib.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>

#include "forward_impl.h"
#include "substrate/gpu/mv.h"
#include "substrate/gpu/ops.h"
#include "substrate/gpu/quant.h"

/* the nth Engram layer's rows for each token, read from the Engram file, decoded (BF16, as ref.py's), projected and
   gated into the streams */
bool fwd_engram(Forward *f, int l, int nth, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const GgufTensor *t = L->eg_embd;
  const uint32_t cols = (uint32_t)g->engram_cols, hd = (uint32_t)g->engram_head_dim, D = g->n_embd, hc = g->hc;
  const uint32_t n = f->n;
  const uint64_t rb = gguf_bytes(t->type, hd);
  uint8_t *stage = f->stage + (uint64_t)nth * FWD_MAX_BATCH * cols * rb;
  for (uint32_t i = 0; i < n; i++)
  {
    const int64_t *rows = f->rows + (uint64_t)i * WEIGHTS_MAX_LAYERS * cols + (uint64_t)nth * cols;
    for (uint32_t c = 0; c < cols; c++)
    {
      uint8_t *to = stage + ((uint64_t)i * cols + c) * rb;
      uint64_t got = 0;
      while (got < rb)
      {
        const ssize_t r = pread(f->eg_fd, to + got, rb - got, (off_t)(t->offset + (uint64_t)rows[c] * rb + got));
        if (r <= 0)
          return fwd_fail(err, "an Engram row could not be read");
        got += (uint64_t)r;
      }
    }
  }
  const uint64_t vals = (uint64_t)n * cols * hd;
  const bool ok = fwd_put_async(b->erow, stage, (uint64_t)n * cols * rb) &&
                  quant_dequant(t->type, b->erow, vals, b->erow_f, NULL) && ops_round_bf16(b->erow_f, vals, NULL) &&
                  fwd_mat(f, L->eg_wkv, fwd_dev(f, L->eg_wkv), (hc + 1) * D, cols * hd, b->erow_f, cols * hd, n,
                          b->ekv, (hc + 1) * D) &&
                  ops_engram_gate(b->h, b->ekv, fwd_dev(f, L->eg_q), fwd_dev(f, L->eg_k), hc, D, g->rms_eps, n, NULL);
  return ok || fwd_fail(err, "a GPU call failed (Engram)");
}

static int cmp_expert(const void *a, const void *b)
{
  const int32_t *x = a, *y = b;
  return (x[0] > y[0]) - (x[0] < y[0]);
}

/* the shared expert, batched, into its own buffers: queued before the router's result is read, so the GPU has it to
   do while the host pins the routed ones */
static bool shared(Forward *f, int l)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t D = g->n_embd, FF = g->expert_ff, n = f->n;
  return fwd_mat(f, L->sh_gate, fwd_dev(f, L->sh_gate), FF, D, b->xf, D, n, b->sga, FF) &&
         fwd_mat(f, L->sh_up, fwd_dev(f, L->sh_up), FF, D, b->xf, D, n, b->sup, FF) &&
         ops_swiglu(b->sga, b->sup, FF, g->clamp_shexp[l], NULL, n, b->shh, NULL) &&
         fwd_mat(f, L->sh_down, fwd_dev(f, L->sh_down), D, FF, b->shh, FF, n, b->sye, D);
}

/* the routed experts: each distinct one held in the pool (their reads at once) and read once for all the tokens routed
   to it - gate, up, swiglu (each pair its weight, before the down projection, as ref.py) and down, a launch each
   (mm_group); each token's summed in ascending expert id (ref.py's index_add order), the shared expert last.
   nothing here waits for the GPU but the routing event; the experts are released at the next one (or the step's
   end), when the stream is past their kernels */
bool fwd_moe(Forward *f, int l, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t D = g->n_embd, FF = g->expert_ff, E = g->n_experts, K = g->n_used, n = f->n, P = n * K;
  bool ok = fwd_mat(f, L->router, fwd_dev(f, L->router), E, D, b->xf, D, n, b->logits, E);
  if (ok)
    fwd_probe(f, "router_logits", l, f->at, b->logits, (uint64_t)n * E, false);
  /* the picks to the host and marked, then the shared expert queued: the GPU runs it while the host waits */
  ok = ok && ops_router(b->logits, (const float *)fwd_dev(f, L->router_bias), E, K, g->expert_scale, n, b->ridx, b->rw,
                        NULL) &&
       hipMemcpyAsync(f->pin->ids, b->ridx, P * sizeof(int32_t), hipMemcpyDeviceToHost, NULL) == hipSuccess &&
       hipEventRecord(f->routed, NULL) == hipSuccess && shared(f, l) && hipEventSynchronize(f->routed) == hipSuccess;
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the router, the shared expert)");
  fwd_release_held(f); /* the stream is past the last layer's experts: the router came after them */
  const int32_t *ids = f->pin->ids;
  fwd_probe(f, "route_idx", l, f->at, b->ridx, P, true);
  fwd_probe(f, "route_w", l, f->at, b->rw, P, false);
  /* the distinct experts in ascending id, each with its pairs (in token order: at most one a token): an expert's
     rows base .. base + cnt, its slots those tokens; one launch of mm_group reads each once for all of its tokens */
  uint32_t nd = 0, by[FWD_MAX_PAIRS];
  for (uint32_t m = 0; m < P; m++)
  {
    if (ids[m] < 0 || (uint32_t)ids[m] >= E)
      return fwd_fail(err, "the router picked an expert the model does not have");
    by[m] = m;
  }
  for (uint32_t i = 1; i < P; i++) /* by expert, then pair: an insertion sort of at most FWD_MAX_PAIRS */
    for (uint32_t j = i; j > 0 && (ids[by[j]] < ids[by[j - 1]] || (ids[by[j]] == ids[by[j - 1]] && by[j] < by[j - 1]));
         j--)
    {
      const uint32_t t = by[j];
      by[j] = by[j - 1], by[j - 1] = t;
    }
  Pinned *pin = f->pin;
  uint32_t row_of[FWD_MAX_PAIRS];
  for (uint32_t r = 0; r < P; r++)
  {
    const uint32_t m = by[r];
    if (r == 0 || ids[m] != ids[by[r - 1]])
    {
      pin->base[nd] = (int32_t)r, pin->cnt[nd] = 0;
      for (uint32_t j = 0; j < MM_TILE; j++)
        pin->gidx[nd * MM_TILE + j] = -1, pin->hidx[nd * MM_TILE + j] = -1;
      const uint32_t unit = (uint32_t)l * E + (uint32_t)ids[m];
      if (pool_pin(f->pool, unit, (const void *[POOL_PARTS]){0}) == POOL_FULL)
        return fwd_fail(err, "an expert the budget cannot hold beside the layers' weights");
      f->held[f->n_held++] = unit;
      nd++;
    }
    const uint32_t j = nd - 1, slot = (uint32_t)pin->cnt[j]++;
    pin->gidx[j * MM_TILE + slot] = (int32_t)(m / K), pin->hidx[j * MM_TILE + slot] = (int32_t)r;
    pin->pidx[r] = (int32_t)m, row_of[m] = r;
  }
  const void **tab = pin->tab;
  for (uint32_t j = 0; ok && j < nd; j++)
  {
    const void *ex[POOL_PARTS];
    ok = fwd_pinned(f, f->held[j], ex, err);
    tab[j] = ex[0], tab[nd + j] = ex[1], tab[2 * nd + j] = ex[2];
  }
  for (uint32_t i = 0; i < n; i++) /* each token's rows in ascending expert id */
  {
    int32_t ord[OPS_MAX_TOPK][2];
    for (uint32_t k = 0; k < K; k++)
      ord[k][0] = ids[i * K + k], ord[k][1] = (int32_t)row_of[i * K + k];
    qsort(ord, K, sizeof *ord, cmp_expert);
    for (uint32_t k = 0; k < K; k++)
      pin->order[i * K + k] = ord[k][1];
  }
  uint32_t *xg = pin->offs, *yg = pin->offs + FWD_MAX_PAIRS, *xd = pin->offs + 2 * FWD_MAX_PAIRS;
  uint32_t *yd = pin->offs + 3 * FWD_MAX_PAIRS;
  for (uint32_t j = 0; j < nd; j++) /* expert j: its 16 gathered slots, its rows from base */
    xg[j] = j * MM_TILE * D, yg[j] = (uint32_t)pin->base[j] * FF, xd[j] = j * MM_TILE * FF,
    yd[j] = (uint32_t)pin->base[j] * D;
  const uint32_t *dxg = b->offs, *dyg = b->offs + FWD_MAX_PAIRS, *dxd = b->offs + 2 * FWD_MAX_PAIRS;
  const uint32_t *dyd = b->offs + 3 * FWD_MAX_PAIRS;
  ok = ok && fwd_put_async(b->wtab, tab, 3 * nd * sizeof *tab) && fwd_put_async(b->order, pin->order, P * 4) &&
       fwd_put_async(b->cnt, pin->cnt, nd * 4) && fwd_put_async(b->offs, pin->offs, sizeof pin->offs) &&
       fwd_put_async(b->gidx, pin->gidx, nd * MM_TILE * 4) && fwd_put_async(b->hidx, pin->hidx, nd * MM_TILE * 4) &&
       fwd_put_async(b->pidx, pin->pidx, P * 4) && ops_gather_rows(b->rwp, b->rw, b->pidx, P, 1, NULL) &&
       mm_x16_rows(b->xf, D, b->gidx, nd, D, b->xe16, b->overflow, NULL) &&
       mm_group(L->exp_gate->type, b->wtab, b->cnt, nd, FF, D, b->xe16, dxg, D, b->ga, dyg, FF, NULL) &&
       mm_group(L->exp_up->type, b->wtab + nd, b->cnt, nd, FF, D, b->xe16, dxg, D, b->up, dyg, FF, NULL) &&
       ops_swiglu(b->ga, b->up, FF, g->clamp_exp[l], b->rwp, P, b->hh, NULL) &&
       mm_x16_rows(b->hh, FF, b->hidx, nd, FF, b->xe16, b->overflow, NULL) &&
       mm_group(L->exp_down->type, b->wtab + 2 * nd, b->cnt, nd, D, FF, b->xe16, dxd, FF, b->ye, dyd, D, NULL) &&
       ops_gather_sum(b->moe, b->ye, b->order, K, n, D, NULL) && ops_add(b->moe, b->sye, (uint64_t)n * D, NULL);
  if (!ok)
    return err[0] ? false : fwd_fail(err, "a GPU call failed (the experts)");
  fwd_probe(f, "moe_out", l, f->at, b->moe, (uint64_t)n * D, false);
  return true;
}
