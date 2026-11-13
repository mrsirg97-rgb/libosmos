/* 0.7.0: the NPU beside the GPU in a prompt step's routed experts (forward.h's forward_npu; npu.h). an eligible
   layer (gate and up one format the NPU has a kernel for: IQ2_XXS, IQ1_S) gives the NPU a fixed share of its pairs in whole NPU_TOKENS
   parts of its hottest experts - chosen from the routing alone (npu_pick), so a prompt's bits are the same each run;
   the GPU packs their tokens' rows (npu_pack_a), the NPU runs them while the GPU runs the layer's other gate and up
   rows, and the GPU unpacks theirs into ga / upr before the swiglu (fwd_npu_run, fwd_npu_unpack) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "forward_impl.h"
#include "osmos/model/qtype.h"

static void npu_free(Forward *f)
{
  npu_close(f->npu), f->npu = NULL;
  if (f->npu_packed)
    hipEventDestroy(f->npu_packed), f->npu_packed = NULL;
  hipHostFree(f->npu_tok), hipHostFree(f->npu_row), f->npu_tok = NULL, f->npu_row = NULL;
  free(f->npu_wg), free(f->npu_wu), f->npu_wg = f->npu_wu = NULL;
}

void fwd_npu_destroy(Forward *f) { npu_free(f); }

bool forward_npu(Forward *f, const char *dir, float share, char err[FORWARD_ERR])
{
  const Geometry *g = f->g;
  if (!dir || !(share > 0.0f && share < 1.0f))
    return fwd_fail(err, "forward_npu: a directory and a share in (0, 1)");
  if (g->n_embd != 5120 || g->expert_ff != 2304 || g->n_experts > OPS_MAX_EXPERTS)
    return fwd_fail(err, "forward_npu: the kernel is DeepSeek's (5120 -> 2304)");
  npu_free(f);
  const uint64_t pairs = (uint64_t)f->max_tokens * (uint64_t)g->n_used;
  const uint32_t max = (uint32_t)(share * (double)pairs / NPU_TOKENS) + 1;
  char nerr[NPU_ERR] = "";
  void *hb = host_slots_base(f->hs), *sb = host_slots_base(f->hs_stream);
  if (!(f->npu = npu_open(dir, g->n_embd, g->expert_ff, max, nerr)) ||
      /* the cache's pages before its tail (always registered), and the stream's: every expert a prompt step holds
         while the tail is lent (fwd_npu_run checks each) */
      !npu_region(f->npu, hb, host_slots_bytes(f->hs) - host_slots_tail(f->hs), nerr) ||
      !npu_region(f->npu, sb, host_slots_bytes(f->hs_stream), nerr))
  {
    npu_free(f);
    return fwd_fail(err, nerr);
  }
  const unsigned mapped = hipHostMallocMapped | hipHostMallocCoherent;
  if (hipHostMalloc((void **)&f->npu_tok, (size_t)max * NPU_TOKENS * 4, mapped) != hipSuccess ||
      hipHostMalloc((void **)&f->npu_row, (size_t)max * 4, mapped) != hipSuccess ||
      !(f->npu_wg = malloc(max * sizeof *f->npu_wg)) || !(f->npu_wu = malloc(max * sizeof *f->npu_wu)) ||
      hipEventCreateWithFlags(&f->npu_packed, hipEventDisableTiming) != hipSuccess)
  {
    npu_free(f);
    return fwd_fail(err, "forward_npu: out of memory");
  }
  f->npu_share = share, f->npu_max = max;
  return true;
}

bool fwd_npu_layer(const Forward *f, int l, uint32_t pairs)
{
  const LayerWeights *L = &f->w->layer[l];
  return f->npu && pairs >= (uint32_t)f->g->n_experts && L->exp_gate->type == L->exp_up->type &&
         npu_has(f->npu, L->exp_gate->type);
}

/* the layer's NPU parts: floor(share x pairs / NPU_TOKENS) at most, from its experts with a whole part, the most
   picked first (ties the lower id), each expert's parts from its first rows. nq[j] each distinct expert j's; returns
   their count. by the routing alone: the same picks the same parts */
uint32_t fwd_npu_pick(const Forward *f, uint32_t pairs, const int32_t *cnt, uint32_t nd, uint32_t *nq)
{
  uint32_t want = (uint32_t)(f->npu_share * (double)pairs / NPU_TOKENS), got = 0;
  if (want > f->npu_max)
    want = f->npu_max;
  for (uint32_t j = 0; j < nd; j++)
    nq[j] = 0;
  while (got < want)
  {
    int32_t best = -1; /* the distinct expert with the most rows still whole parts past its taken ones */
    uint32_t left_best = 0;
    for (uint32_t j = 0; j < nd; j++)
    {
      const uint32_t left = (uint32_t)cnt[j] - nq[j] * NPU_TOKENS;
      if (left >= NPU_TOKENS && left > left_best)
        best = (int32_t)j, left_best = left;
    }
    if (best < 0)
      break;
    nq[best]++, got++;
  }
  return got;
}

/* the parts' tokens and rows into the tables, their packing queued (after the layer's gathers on the stream) */
bool fwd_npu_pack(Forward *f, uint32_t parts, const uint32_t *nq, uint32_t nd, const uint32_t *base,
                  const uint32_t *slot0, const int32_t *gidx, const void *const *ex_gate, const void *const *ex_up)
{
  uint32_t p = 0;
  for (uint32_t j = 0; j < nd; j++)
    for (uint32_t q = 0; q < nq[j]; q++, p++)
    {
      for (uint32_t t = 0; t < NPU_TOKENS; t++)
        f->npu_tok[p * NPU_TOKENS + t] = gidx[slot0[j] + q * NPU_TOKENS + t];
      f->npu_row[p] = base[j] + q * NPU_TOKENS;
      f->npu_wg[p] = ex_gate[j], f->npu_wu[p] = ex_up[j];
    }
  return p == parts && npu_pack_a(f->b.xf, f->g->n_embd, f->npu_tok, parts, npu_a(f->npu), NULL);
}

/* the NPU's launches, once the packing is done (the GPU going on with what was queued after it) */
bool fwd_npu_run(Forward *f, int l, uint32_t parts, hipEvent_t packed, char *err)
{
  const Geometry *g = f->g;
  const uint32_t qt = f->w->layer[l].exp_gate->type;
  char nerr[NPU_ERR] = "";
  if (hipEventSynchronize(packed) != hipSuccess)
    return fwd_fail(err, "the NPU's packing failed");
  const uint64_t blk = qt == QTYPE_IQ2_XXS ? QTYPE_IQ2_XXS_BYTES : QTYPE_IQ1_S_BYTES; /* npu_has's formats */
  if (!npu_gate_up(f->npu, qt, parts, f->npu_wg, f->npu_wu, (uint64_t)g->expert_ff * (g->n_embd / 256) * blk, nerr))
    return fwd_fail(err, nerr);
  return true;
}

bool fwd_npu_unpack(Forward *f, uint32_t parts)
{
  return npu_unpack_c(npu_c(f->npu), f->g->expert_ff, NPU_GATE_UP_ROWS, 2, parts, f->npu_row, f->b.ga, f->b.upr, NULL);
}

void forward_npu_stats(const Forward *f, uint64_t *layers, uint64_t *parts, double *wait_us)
{
  const NpuStats st = npu_stats(f ? f->npu : NULL);
  *layers = st.layers, *parts = st.parts, *wait_us = st.wait_us;
}
