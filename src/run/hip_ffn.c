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
                  mv_run(L->eg_wkv->type, fwd_dev(f, L->eg_wkv), (hc + 1) * D, cols * hd, b->erow_f, n, b->ekv, NULL) &&
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
  return mv_run(L->sh_gate->type, fwd_dev(f, L->sh_gate), FF, D, b->xf, n, b->sga, NULL) &&
         mv_run(L->sh_up->type, fwd_dev(f, L->sh_up), FF, D, b->xf, n, b->sup, NULL) &&
         ops_swiglu(b->sga, b->sup, FF, g->clamp_shexp[l], NULL, n, b->shh, NULL) &&
         mv_run(L->sh_down->type, fwd_dev(f, L->sh_down), D, FF, b->shh, n, b->sye, NULL);
}

/* the routed experts as (token, slot) pairs, pair i * K + k: each held in the pool (their reads at once), the
   matrices into the group table, then gate, up, swiglu (each its weight, before the down projection, as ref.py) and
   down, a launch each for every pair; each token's summed in ascending expert id (ref.py's index_add order), the
   shared expert last. nothing here waits for the GPU but the routing event; the experts are released at the next
   one (or the step's end), when the stream is past their kernels */
bool fwd_moe(Forward *f, int l, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t D = g->n_embd, FF = g->expert_ff, E = g->n_experts, K = g->n_used, n = f->n, P = n * K;
  bool ok = mv_run(L->router->type, fwd_dev(f, L->router), E, D, b->xf, n, b->logits, NULL);
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
  for (uint32_t m = 0; m < P; m++)
  {
    if (ids[m] < 0 || (uint32_t)ids[m] >= E)
      return fwd_fail(err, "the router picked an expert the model does not have");
    if (pool_pin(f->pool, (uint32_t)l * E + (uint32_t)ids[m], (const void *[POOL_PARTS]){0}) == POOL_FULL)
      return fwd_fail(err, "an expert the budget cannot hold beside the layers' weights");
    f->held[f->n_held++] = (uint32_t)l * E + (uint32_t)ids[m];
  }
  const void **tab = f->pin->tab;
  for (uint32_t m = 0; ok && m < P; m++)
  {
    const void *ex[POOL_PARTS];
    ok = fwd_pinned(f, f->held[m], ex, err);
    tab[m] = ex[0], tab[P + m] = ex[1], tab[2 * P + m] = ex[2];
  }
  for (uint32_t i = 0; i < n; i++) /* each token's pairs in ascending expert id */
  {
    int32_t by[OPS_MAX_TOPK][2];
    for (uint32_t k = 0; k < K; k++)
      by[k][0] = ids[i * K + k], by[k][1] = (int32_t)(i * K + k);
    qsort(by, K, sizeof *by, cmp_expert);
    for (uint32_t k = 0; k < K; k++)
      f->pin->order[i * K + k] = by[k][1];
  }
  ok = ok && fwd_put_async(b->wtab, tab, 3 * P * sizeof *tab) && fwd_put_async(b->order, f->pin->order, P * 4) &&
       mv_group(L->exp_gate->type, b->wtab, P, FF, D, b->xf, D, K, b->ga, NULL) &&
       mv_group(L->exp_up->type, b->wtab + P, P, FF, D, b->xf, D, K, b->up, NULL) &&
       ops_swiglu(b->ga, b->up, FF, g->clamp_exp[l], b->rw, P, b->hh, NULL) &&
       mv_group(L->exp_down->type, b->wtab + 2 * P, P, D, FF, b->hh, FF, 1, b->ye, NULL) &&
       ops_gather_sum(b->moe, b->ye, b->order, K, n, D, NULL) && ops_add(b->moe, b->sye, (uint64_t)n * D, NULL);
  if (!ok)
    return err[0] ? false : fwd_fail(err, "a GPU call failed (the experts)");
  fwd_probe(f, "moe_out", l, f->at, b->moe, (uint64_t)n * D, false);
  return true;
}
