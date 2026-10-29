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
  for (uint32_t c = 0; c < cols; c++)
  {
    uint64_t got = 0;
    while (got < rb)
    {
      const off_t at = (off_t)(t->offset + (uint64_t)rows[c] * rb + got);
      const ssize_t n = pread(f->eg_fd, f->stage + c * rb + got, rb - got, at);
      if (n <= 0)
        return fwd_fail(err, "an Engram row could not be read");
      got += (uint64_t)n;
    }
  }
  const bool ok = fwd_put(b->erow, f->stage, cols * rb) &&
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

/* the router's experts, in ascending id (ref.py's index_add order), each its weight before the down projection; then
   the shared expert */
bool fwd_moe(Forward *f, int l, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  Scratch *b = &f->b;
  const uint32_t D = g->n_embd, FF = g->expert_ff, E = g->n_experts, K = g->n_used;
  bool ok = mv_run(L->router->type, fwd_dev(f, L->router), E, D, b->xf, 1, b->logits, NULL) &&
            ops_router(b->logits, (const float *)fwd_dev(f, L->router_bias), E, K, g->expert_scale, 1, b->ridx, b->rw,
                       NULL);
  int32_t pick[OPS_MAX_TOPK][2];
  int32_t ids[OPS_MAX_TOPK];
  ok = ok && hipMemcpy(ids, b->ridx, K * sizeof(int32_t), hipMemcpyDeviceToHost) == hipSuccess;
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the router)");
  fwd_probe(f, "route_idx", l, f->at, b->ridx, K, true);
  fwd_probe(f, "route_w", l, f->at, b->rw, K, false);
  for (uint32_t k = 0; k < K; k++)
    pick[k][0] = ids[k], pick[k][1] = (int32_t)k;
  qsort(pick, K, sizeof *pick, cmp_expert);
  const uint64_t gs = L->exp_gate->bytes / E, us = L->exp_up->bytes / E, ds = L->exp_down->bytes / E;
  ok = hipMemset(b->moe, 0, D * sizeof(float)) == hipSuccess;
  for (uint32_t k = 0; ok && k < K; k++)
  {
    const uint64_t e = (uint64_t)pick[k][0];
    ok = mv_run(L->exp_gate->type, fwd_dev(f, L->exp_gate) + e * gs, FF, D, b->xf, 1, b->ga, NULL) &&
         mv_run(L->exp_up->type, fwd_dev(f, L->exp_up) + e * us, FF, D, b->xf, 1, b->up, NULL) &&
         ops_swiglu(b->ga, b->up, FF, g->clamp_exp[l], b->rw + pick[k][1], 1, b->hh, NULL) &&
         mv_run(L->exp_down->type, fwd_dev(f, L->exp_down) + e * ds, D, FF, b->hh, 1, b->ye, NULL) &&
         ops_add(b->moe, b->ye, D, NULL);
  }
  ok = ok && mv_run(L->sh_gate->type, fwd_dev(f, L->sh_gate), FF, D, b->xf, 1, b->ga, NULL) &&
       mv_run(L->sh_up->type, fwd_dev(f, L->sh_up), FF, D, b->xf, 1, b->up, NULL) &&
       ops_swiglu(b->ga, b->up, FF, g->clamp_shexp[l], NULL, 1, b->hh, NULL) &&
       mv_run(L->sh_down->type, fwd_dev(f, L->sh_down), D, FF, b->hh, 1, b->ye, NULL) &&
       ops_add(b->moe, b->ye, D, NULL);
  if (!ok)
    return fwd_fail(err, "a GPU call failed (the experts)");
  fwd_probe(f, "moe_out", l, f->at, b->moe, D, false);
  return true;
}
