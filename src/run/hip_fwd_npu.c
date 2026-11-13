/* 0.7.0: the NPU beside the GPU in a prompt step's routed experts (forward.h's forward_npu; npu.h). a layer whose
   gate and up share a format the NPU has a kernel for, or whose down has one, gives the NPU a fixed share of its pairs
   in whole NPU_TOKENS parts of its hottest experts - chosen from the routing alone (fwd_npu_pick), so a prompt's bits
   are the same each run. for each matrix it takes the GPU packs the parts' rows (npu_pack_a: of xf for gate and up,
   of hh for the down), and the NPU runs them while the GPU runs the layer's other rows - writing theirs where the GPU
   reads them, a part's 128 rows of ga / upr (before the swiglu) or ye (before the sum): no copy back */
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
  if (f->npu_gpu_done)
    hipEventDestroy(f->npu_gpu_done), hipEventDestroy(f->npu_resumed), f->npu_gpu_done = f->npu_resumed = NULL;
  hipHostFree(f->npu_tok), hipHostFree(f->npu_row), f->npu_tok = NULL, f->npu_row = NULL;
  free(f->npu_wg), free(f->npu_wu), f->npu_wg = f->npu_wu = NULL;
  free(f->npu_c0), free(f->npu_c1), f->npu_c0 = f->npu_c1 = NULL;
}

void fwd_npu_destroy(Forward *f) { npu_free(f); }

bool forward_npu(Forward *f, const char *dir, float share, char err[FORWARD_ERR])
{
  const Geometry *g = f->g;
  if (!dir || !(share > 0.0f && share < 1.0f))
    return fwd_fail(err, "forward_npu: a directory and a share in (0, 1)");
  /* the kernels' shapes: 2304 = 3 passes x 32 cores x NPU_GATE_UP_ROWS, 5120 = 10 x 32 x NPU_DOWN_ROWS */
  if (g->n_embd % (NPU_CORES * NPU_DOWN_ROWS) || g->expert_ff % (NPU_CORES * NPU_GATE_UP_ROWS) || g->n_embd != 5120 ||
      g->expert_ff != 2304 || g->n_experts > OPS_MAX_EXPERTS)
    return fwd_fail(err, "forward_npu: the kernels are DeepSeek's (5120 -> 2304 -> 5120)");
  npu_free(f);
  const uint64_t pairs = (uint64_t)f->max_tokens * (uint64_t)g->n_used;
  const uint32_t max = (uint32_t)(share * (double)pairs / NPU_TOKENS) + 1;
  char nerr[NPU_ERR] = "";
  void *hb = host_slots_base(f->hs), *sb = host_slots_base(f->hs_stream);
  if (!(f->npu = npu_open(dir, g->n_embd, g->expert_ff, max, nerr)) ||
      /* the cache's pages and the stream's: every expert a step holds (npu_run refuses one outside them) */
      !npu_region(f->npu, hb, host_slots_bytes(f->hs), nerr) ||
      !npu_region(f->npu, sb, host_slots_bytes(f->hs_stream), nerr) ||
      /* where its rows go: the step's gate, up and down outputs (their sizes hip_forward.c's buffers()') */
      !npu_output(f->npu, f->b.ga, pairs * g->expert_ff * sizeof(float), nerr) ||
      !npu_output(f->npu, f->b.upr, pairs * g->expert_ff * sizeof(float), nerr) ||
      !npu_output(f->npu, f->b.ye, pairs * g->n_embd * sizeof(float), nerr))
  {
    npu_free(f);
    return fwd_fail(err, nerr);
  }
  const unsigned mapped = hipHostMallocMapped | hipHostMallocCoherent;
  if (hipHostMalloc((void **)&f->npu_tok, (size_t)max * NPU_TOKENS * 4, mapped) != hipSuccess ||
      hipHostMalloc((void **)&f->npu_row, (size_t)max * 4, mapped) != hipSuccess ||
      !(f->npu_wg = malloc(max * sizeof *f->npu_wg)) || !(f->npu_wu = malloc(max * sizeof *f->npu_wu)) ||
      !(f->npu_c0 = malloc(max * sizeof *f->npu_c0)) || !(f->npu_c1 = malloc(max * sizeof *f->npu_c1)) ||
      hipEventCreateWithFlags(&f->npu_packed, hipEventDisableTiming) != hipSuccess ||
      hipEventCreate(&f->npu_gpu_done) != hipSuccess || hipEventCreate(&f->npu_resumed) != hipSuccess)
  {
    npu_free(f);
    return fwd_fail(err, "forward_npu: out of memory");
  }
  f->npu_share = share, f->npu_max = max;
  return true;
}

/* what the NPU takes of layer l in a step of `pairs`: bit 0 its gate and up, bit 1 its down; a prompt's steps only */
uint32_t fwd_npu_layer(const Forward *f, int l, uint32_t pairs)
{
  if (!f->npu || pairs < (uint32_t)f->g->n_experts)
    return 0;
  const LayerWeights *L = &f->w->layer[l];
  return (uint32_t)(L->exp_gate->type == L->exp_up->type && npu_has(f->npu, NPU_GATE_UP, L->exp_gate->type)) |
         (uint32_t)npu_has(f->npu, NPU_DOWN, L->exp_down->type) << 1;
}

/* the layer's NPU parts: floor(share x pairs / NPU_TOKENS) at most, from its experts with a whole part, the most
   rows left first (ties the lower id), each expert's parts from its first rows. nq[j] each distinct expert j's;
   returns their count. by the routing alone: the same picks the same parts */
uint32_t fwd_npu_pick(const Forward *f, uint32_t pairs, const int32_t *cnt, uint32_t nd, uint32_t *nq)
{
  uint32_t want = (uint32_t)(f->npu_share * (double)pairs / NPU_TOKENS), got = 0;
  if (want > f->npu_max)
    want = f->npu_max;
  for (uint32_t j = 0; j < nd; j++)
    nq[j] = 0;
  while (got < want)
  {
    int32_t best = -1;
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

/* the parts' rows and weights into the tables, their packing queued and npu_packed recorded after it: gate and up
   from xf (a part's tokens, gidx), the down from hh (its pairs' rows). w1 the up's, NULL for the down */
bool fwd_npu_pack(Forward *f, NpuKind kind, uint32_t parts, const uint32_t *nq, uint32_t nd, const uint32_t *base,
                  const uint32_t *slot0, const int32_t *gidx, const void *const *w0, const void *const *w1)
{
  uint32_t p = 0;
  for (uint32_t j = 0; j < nd; j++)
    for (uint32_t q = 0; q < nq[j]; q++, p++)
    {
      const uint32_t r0 = base[j] + q * NPU_TOKENS;
      for (uint32_t t = 0; t < NPU_TOKENS; t++)
        f->npu_tok[p * NPU_TOKENS + t] = kind == NPU_GATE_UP ? gidx[slot0[j] + q * NPU_TOKENS + t] : (int32_t)(r0 + t);
      f->npu_row[p] = r0;
      f->npu_wg[p] = w0[j], f->npu_wu[p] = w1 ? w1[j] : NULL;
    }
  const bool gu = kind == NPU_GATE_UP;
  return p == parts &&
         npu_pack_a(gu ? f->b.xf : f->b.hh, gu ? f->g->n_embd : f->g->expert_ff, f->npu_tok, parts, npu_a(f->npu), NULL) &&
         hipEventRecord(f->npu_packed, NULL) == hipSuccess;
}

/* the NPU's launches, once the packing is done (the GPU going on with what was queued after it) */
bool fwd_npu_run(Forward *f, NpuKind kind, int l, uint32_t parts, char *err)
{
  const Geometry *g = f->g;
  const LayerWeights *L = &f->w->layer[l];
  const uint32_t qt = kind == NPU_GATE_UP ? L->exp_gate->type : L->exp_down->type;
  const QTypeInfo *qi = qtype_info(qt);
  char nerr[NPU_ERR] = "";
  if (f->npu_timed) /* the last call's pair, long past: the GPU's idle between them */
  {
    float ms = 0;
    if (hipEventElapsedTime(&ms, f->npu_gpu_done, f->npu_resumed) == hipSuccess)
      f->npu_idle_ms += ms;
    f->npu_timed = false;
  }
  if (!qi || hipEventRecord(f->npu_gpu_done, NULL) != hipSuccess || hipEventSynchronize(f->npu_packed) != hipSuccess)
    return fwd_fail(err, "the NPU's packing failed");
  const uint64_t w_bytes = (uint64_t)g->expert_ff * g->n_embd / qi->values * qi->bytes; /* either matrix: D x FF */
  for (uint32_t p = 0; p < parts; p++) /* a part's rows where the GPU reads them: from its first row's */
    if (kind == NPU_GATE_UP)
      f->npu_c0[p] = f->b.ga + (uint64_t)f->npu_row[p] * g->expert_ff,
      f->npu_c1[p] = f->b.upr + (uint64_t)f->npu_row[p] * g->expert_ff;
    else
      f->npu_c0[p] = f->npu_c1[p] = f->b.ye + (uint64_t)f->npu_row[p] * g->n_embd;
  const bool ahead = hipEventQuery(f->npu_gpu_done) == hipErrorNotReady; /* the GPU still busy as the NPU begins */
  if (!npu_run(f->npu, kind, qt, parts, f->npu_wg, f->npu_wu, w_bytes, f->npu_c0, f->npu_c1, nerr))
    return fwd_fail(err, nerr);
  f->npu_gpu_ahead_ms += ahead && hipEventQuery(f->npu_gpu_done) == hipErrorNotReady; /* still busy as it ends: a count */
  if (hipEventRecord(f->npu_resumed, NULL) != hipSuccess)
    return fwd_fail(err, "the NPU's resume event");
  f->npu_timed = true;
  return true;
}

void forward_npu_stats(const Forward *f, uint64_t *calls, uint64_t *parts, double *wait_us, double *gpu_idle_ms,
                       uint64_t *gpu_outlasted)
{
  const NpuStats st = npu_stats(f ? f->npu : NULL);
  *calls = st.calls, *parts = st.parts, *wait_us = st.wait_us;
  *gpu_idle_ms = f ? f->npu_idle_ms : 0, *gpu_outlasted = f ? (uint64_t)f->npu_gpu_ahead_ms : 0;
}
