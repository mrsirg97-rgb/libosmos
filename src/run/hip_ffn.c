#include <stdlib.h>
#include <unistd.h>

#include <hip/hip_runtime_api.h>

#include "forward_impl.h"
#include "substrate/gpu/mv.h"
#include "substrate/gpu/ops.h"
#include "substrate/gpu/quant.h"

/* the nth Engram layer's rows for this position, read from the Engram file, decoded (BF16, as ref.py's), projected
   and gated into the streams */
bool fwd_engram(Forward *f, int l, int nth, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const GgufTensor *t = L->eg_embd;
  const uint32_t cols = (uint32_t)g->engram_cols, hd = (uint32_t)g->engram_head_dim, D = g->n_embd, hc = g->hc;
  const uint64_t rb = gguf_bytes(t->type, hd);
  const int64_t *rows = f->rows + (uint64_t)nth * cols;
  uint8_t *stage = f->stage + (uint64_t)nth * cols * rb;
  for (uint32_t c = 0; c < cols; c++)
  {
    uint64_t got = 0;
    while (got < rb)
    {
      const off_t at = (off_t)(t->offset + (uint64_t)rows[c] * rb + got);
      const ssize_t n = pread(f->eg_fd, stage + c * rb + got, rb - got, at);
      if (n <= 0)
        return fwd_fail(err, "an Engram row could not be read");
      got += (uint64_t)n;
    }
  }
  const bool ok = fwd_put_async(b->erow, stage, cols * rb) &&
                  quant_dequant(t->type, b->erow, (uint64_t)cols * hd, b->erow_f, NULL) &&
                  ops_round_bf16(b->erow_f, (uint64_t)cols * hd, NULL) &&
                  mv_run(L->eg_wkv->type, fwd_dev(f, L->eg_wkv), (hc + 1) * D, cols * hd, b->erow_f, 1, b->ekv, NULL) &&
                  ops_engram_gate(b->h, b->ekv, fwd_dev(f, L->eg_q), fwd_dev(f, L->eg_k), hc, D, g->rms_eps, 1, NULL);
  return ok || fwd_fail(err, "a GPU call failed (Engram)");
}

static int cmp_expert(const void *a, const void *b)
{
  const int32_t *x = a, *y = b;
  return (x[0] > y[0]) - (x[0] < y[0]);
}

/* the shared expert, into its own buffers: queued before the router's result is read, so the GPU has it to do while
   the host pins the routed ones */
static bool shared(Forward *f, int l)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t D = g->n_embd, FF = g->expert_ff;
  return mv_run(L->sh_gate->type, fwd_dev(f, L->sh_gate), FF, D, b->xf, 1, b->sga, NULL) &&
         mv_run(L->sh_up->type, fwd_dev(f, L->sh_up), FF, D, b->xf, 1, b->sup, NULL) &&
         ops_swiglu(b->sga, b->sup, FF, g->clamp_shexp[l], NULL, 1, b->shh, NULL) &&
         mv_run(L->sh_down->type, fwd_dev(f, L->sh_down), D, FF, b->shh, 1, b->sye, NULL);
}

/* the routed experts: held in the pool (their reads at once), their matrices into the group table, then gate, up,
   swiglu (each its weight, before the down projection, as ref.py) and down, all of them a launch each; summed in
   ascending id (ref.py's index_add order), the shared expert last. nothing here waits for the GPU but the routing
   event; the experts are released at the next one (or the step's end), when the stream is past their kernels */
bool fwd_moe(Forward *f, int l, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t D = g->n_embd, FF = g->expert_ff, E = g->n_experts, K = g->n_used;
  bool ok = mv_run(L->router->type, fwd_dev(f, L->router), E, D, b->xf, 1, b->logits, NULL);
  if (ok)
    fwd_probe(f, "router_logits", l, f->at, b->logits, E, false);
  /* the picks to the host and marked, then the shared expert queued: the GPU runs it while the host waits */
  ok = ok && ops_router(b->logits, (const float *)fwd_dev(f, L->router_bias), E, K, g->expert_scale, 1, b->ridx, b->rw,
                        NULL) &&
       hipMemcpyAsync(f->pin->ids, b->ridx, K * sizeof(int32_t), hipMemcpyDeviceToHost, NULL) == hipSuccess &&
       hipEventRecord(f->routed, NULL) == hipSuccess && shared(f, l) && hipEventSynchronize(f->routed) == hipSuccess;
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the router, the shared expert)");
  fwd_release_held(f); /* the stream is past the last layer's experts: the router came after them */
  const int32_t *ids = f->pin->ids;
  fwd_probe(f, "route_idx", l, f->at, b->ridx, K, true);
  fwd_probe(f, "route_w", l, f->at, b->rw, K, false);
  for (uint32_t k = 0; k < K; k++)
  {
    if (ids[k] < 0 || (uint32_t)ids[k] >= E)
      return fwd_fail(err, "the router picked an expert the model does not have");
    if (pool_pin(f->pool, (uint32_t)l * E + (uint32_t)ids[k], (const void *[POOL_PARTS]){0}) == POOL_FULL)
      return fwd_fail(err, "an expert the budget cannot hold beside the layers' weights");
    f->held[f->n_held++] = (uint32_t)l * E + (uint32_t)ids[k];
  }
  const void **tab = f->pin->tab;
  for (uint32_t k = 0; ok && k < K; k++)
  {
    const void *ex[POOL_PARTS];
    ok = fwd_pinned(f, f->held[k], ex, err);
    tab[k] = ex[0], tab[K + k] = ex[1], tab[2 * K + k] = ex[2];
  }
  int32_t order[OPS_MAX_TOPK][2]; /* (expert, slot), ascending expert */
  for (uint32_t k = 0; k < K; k++)
    order[k][0] = ids[k], order[k][1] = (int32_t)k;
  qsort(order, K, sizeof *order, cmp_expert);
  ok = ok && fwd_put_async(b->wtab, tab, 3 * K * sizeof *tab) &&
       mv_group(L->exp_gate->type, b->wtab, K, FF, D, b->xf, 0, b->ga, NULL) &&
       mv_group(L->exp_up->type, b->wtab + K, K, FF, D, b->xf, 0, b->up, NULL) &&
       ops_swiglu(b->ga, b->up, FF, g->clamp_exp[l], b->rw, K, b->hh, NULL) &&
       mv_group(L->exp_down->type, b->wtab + 2 * K, K, D, FF, b->hh, FF, b->ye, NULL) &&
       hipMemsetAsync(b->moe, 0, D * sizeof(float), NULL) == hipSuccess;
  for (uint32_t k = 0; ok && k < K; k++)
    ok = ops_add(b->moe, b->ye + (uint64_t)order[k][1] * D, D, NULL);
  ok = ok && ops_add(b->moe, b->sye, D, NULL); /* the experts stay held until the stream is past them */
  if (!ok)
    return err[0] ? false : fwd_fail(err, "a GPU call failed (the experts)");
  fwd_probe(f, "moe_out", l, f->at, b->moe, D, false);
  return true;
}
