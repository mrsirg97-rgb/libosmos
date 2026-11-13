/* 0.7.0: the NPU beside the GPU in a Qwen prompt step's routed experts (forward.h's q4x_forward_npu; npu.h). Qwen's
   tables are built on the GPU (q4x_expert_tiles), so the NPU's parts are chosen there too: npu_select_q4x takes the
   first whole 128-token entries of the tables' 128-token run - a layer's share of its picks, by the routing alone, so
   a prompt's bits are the same each run - zeroes their counts (mm_group skips them: their gate, up and down the
   NPU's) and packs their tokens' rows of b.mixed. the host waits for the packing alone, the GPU going on with its
   gate and up, and the NPU writes its rows where the GPU reads them (b.ga, b.up before the swiglu; b.y before the
   sum) - then the same for the down, from b.xe16's slots. no copy back. the weights read in place: the membrane's
   host pages (q4x_forward_whole), or each resident tensor's hipMalloc imported */
#include <stdio.h>
#include <stdlib.h>

#include "osmos/gpu/mm.h"
#include "osmos/mem/host_slots.h"
#include "osmos/model/qtype.h"
#include "q4x_impl.h"

/* a part's NPU time by its gate / up format, against IQ3_XXS's (spikes/npu/RESEARCH.md: gate + up, then the down -
   IQ3_XXS 281 + 139 us, IQ2_S 603 + 139): the share of an IQ2_S layer scaled down, the NPU no longer on either */
static float cost(uint32_t gate_type)
{
  return gate_type == QTYPE_IQ2_S ? 742.0f / 420.0f : 1.0f;
}

void q4x_npu_destroy(Q4xForward *f)
{
  npu_close(f->npu), f->npu = NULL;
  if (f->npu_packed)
    (void)hipEventDestroy(f->npu_packed), f->npu_packed = NULL;
  (void)hipHostFree(f->npu_sel), (void)hipHostFree(f->npu_nsel), f->npu_sel = NULL, f->npu_nsel = NULL;
  (void)hipFree(f->npu_tok), (void)hipFree(f->npu_slot), f->npu_tok = f->npu_slot = NULL;
  free(f->npu_w0), free(f->npu_w1), free(f->npu_c0), free(f->npu_c1);
  f->npu_w0 = f->npu_w1 = NULL, f->npu_c0 = f->npu_c1 = NULL;
}

bool q4x_forward_npu(Q4xForward *f, const char *dir, float share, char err[Q4X_FWD_ERR])
{
  const Q4xGeometry *g = &f->w->geo;
  if (!dir || !(share > 0.0f && share < 1.0f))
    return q4x_fail(err, "q4x_forward_npu: a directory and a share in (0, 1)");
  /* the kernels' shapes (fused_ds.py MODEL=qwen): 2560 -> 640 -> 2560 */
  if (g->n_embd != 2560 || g->ff_exp != 640)
    return q4x_fail(err, "q4x_forward_npu: the kernels are Qwen3.8-Flash-Next's (2560 -> 640 -> 2560)");
  q4x_npu_destroy(f);
  const uint64_t n = f->max_tokens, k = (uint64_t)g->n_used, D = (uint64_t)g->n_embd, ff = (uint64_t)g->ff_exp;
  char nerr[NPU_ERR] = "";
  Q4xBuf *b = &f->b;
  bool ok = (f->npu = npu_open(dir, (uint32_t)D, (uint32_t)ff, NPU_SELECT_MAX, nerr)) &&
            /* where its rows go: the step's gate, up and down outputs (their sizes hip_forward.c's buffers()') */
            npu_device(f->npu, b->ga, n * k * ff * sizeof(float), nerr) &&
            npu_device(f->npu, b->up, n * k * ff * sizeof(float), nerr) &&
            npu_device(f->npu, b->y, n * k * D * sizeof(float), nerr) &&
            /* the experts on a membrane (q4x_forward_whole): its host pages, every expert in them */
            (!f->mem || (npu_region(f->npu, host_slots_base(f->exp_slots), host_slots_bytes(f->exp_slots), nerr) &&
                         npu_region(f->npu, host_slots_base(f->exp_stream), host_slots_bytes(f->exp_stream), nerr)));
  int layers = 0;
  for (int l = 0; ok && l < g->n_layers; l++)
  {
    const Q4xLayer *L = &f->w->layer[l];
    f->npu_layer[l] = L->exp_gate && L->exp_up && L->exp_down && L->exp_gate->type == L->exp_up->type &&
                      npu_has(f->npu, NPU_GATE_UP, L->exp_gate->type) && npu_has(f->npu, NPU_DOWN, L->exp_down->type);
    if (!f->npu_layer[l])
      continue;
    f->npu_cost[l] = cost(L->exp_gate->type), layers++;
    const GgufTensor *t[3] = {L->exp_gate, L->exp_up, L->exp_down};
    for (int i = 0; ok && !f->mem && i < 3; i++) /* resident: its weights read in place, the tensor's own hipMalloc */
      ok = npu_device(f->npu, (void *)q4x_dev(f, t[i]), t[i]->bytes, nerr);
  }
  if (!ok)
  {
    q4x_npu_destroy(f);
    return q4x_fail(err, nerr);
  }
  if (!layers)
  {
    q4x_npu_destroy(f);
    return q4x_fail(err, "q4x_forward_npu: no layer's experts in formats the kernels take");
  }
  const unsigned mapped = hipHostMallocMapped | hipHostMallocCoherent;
  const size_t rows = (size_t)NPU_SELECT_MAX * NPU_TOKENS * sizeof(int32_t), m = NPU_SELECT_MAX;
  if (hipHostMalloc((void **)&f->npu_sel, m * sizeof *f->npu_sel, mapped) != hipSuccess ||
      hipHostMalloc((void **)&f->npu_nsel, sizeof *f->npu_nsel, mapped) != hipSuccess ||
      hipMalloc((void **)&f->npu_tok, rows) != hipSuccess || hipMalloc((void **)&f->npu_slot, rows) != hipSuccess ||
      !(f->npu_w0 = malloc(m * sizeof *f->npu_w0)) || !(f->npu_w1 = malloc(m * sizeof *f->npu_w1)) ||
      !(f->npu_c0 = malloc(m * sizeof *f->npu_c0)) || !(f->npu_c1 = malloc(m * sizeof *f->npu_c1)) ||
      hipEventCreateWithFlags(&f->npu_packed, hipEventDisableTiming) != hipSuccess)
  {
    q4x_npu_destroy(f);
    return q4x_fail(err, "q4x_forward_npu: out of memory");
  }
  f->npu_share = share;
  return true;
}

uint32_t q4x_npu_take(Q4xForward *f, const Q4xLayer *L, uint32_t n, const Q4xExpertBounds *bd, uint32_t D, uint32_t ff)
{
  const Q4xGeometry *g = &f->w->geo;
  const int l = (int)(L - f->w->layer);
  /* a prompt's steps, a layer of the model's (not the MTP head's) with kernels, a 128-token run */
  if (!f->npu || l < 0 || l >= g->n_layers || !f->npu_layer[l] || n <= Q4X_SIDE_STEP || !bd->count[Q4X_TILE_RUNS - 1])
    return 0;
  uint32_t want = (uint32_t)(f->npu_share / f->npu_cost[l] * (double)n * (double)g->n_used / NPU_TOKENS);
  if (want > NPU_SELECT_MAX)
    want = NPU_SELECT_MAX;
  if (!want)
    return 0;
  Q4xBuf *b = &f->b;
  const int r = Q4X_TILE_RUNS - 1;
  if (!npu_select_q4x(b->cnt, bd->first[r], bd->count[r], want, (const void *const *)b->ptr, bd->entries, b->xg, b->yg,
                      b->xd, b->yd, b->gidx, D, ff, f->npu_sel, f->npu_nsel, f->npu_tok, f->npu_slot, NULL) ||
      !npu_pack_a(b->mixed, D, f->npu_tok, want, npu_a(f->npu), NULL) ||
      hipEventRecord(f->npu_packed, NULL) != hipSuccess)
    return 0;
  f->npu_want = want, f->npu_wanted += want;
  return want;
}

/* the parts' launches of one kind, once their packing is done (the GPU going on with what was queued after it) */
static bool run(Q4xForward *f, const Q4xLayer *L, NpuKind kind, uint32_t D, uint32_t ff)
{
  char nerr[NPU_ERR] = "";
  if (hipEventSynchronize(f->npu_packed) != hipSuccess)
    return fprintf(stderr, "q4x: the NPU's packing failed\n"), false;
  const uint32_t parts = *f->npu_nsel;
  if (kind == NPU_GATE_UP)
    f->npu_taken += parts;
  if (!parts)
    return true;
  const GgufTensor *t = kind == NPU_GATE_UP ? L->exp_gate : L->exp_down;
  const QTypeInfo *qi = qtype_info(t->type);
  if (!qi)
    return false;
  const uint64_t w_bytes = (uint64_t)ff * D / qi->values * qi->bytes; /* an expert's matrix: D x ff either way */
  Q4xBuf *b = &f->b;
  for (uint32_t p = 0; p < parts; p++)
  {
    const NpuPart *s = &f->npu_sel[p];
    if (kind == NPU_GATE_UP)
      f->npu_w0[p] = s->gate, f->npu_w1[p] = s->up, f->npu_c0[p] = b->ga + s->yg, f->npu_c1[p] = b->up + s->yg;
    else
      f->npu_w0[p] = f->npu_w1[p] = s->down, f->npu_c0[p] = f->npu_c1[p] = b->y + s->yd;
  }
  if (!npu_run(f->npu, kind, t->type, parts, f->npu_w0, f->npu_w1, w_bytes, f->npu_c0, f->npu_c1, nerr))
    return fprintf(stderr, "q4x: the NPU: %s\n", nerr), false;
  return true;
}

bool q4x_npu_gate_up(Q4xForward *f, const Q4xLayer *L, uint32_t D, uint32_t ff)
{
  return run(f, L, NPU_GATE_UP, D, ff);
}

/* the down's parts packed from b.xe16's slots (the swiglu's, queued before) - before the GPU's down is queued */
bool q4x_npu_pack_down(Q4xForward *f, uint32_t ff)
{
  return !*f->npu_nsel || /* none taken: nothing to pack */
         (npu_pack_a_x16(f->b.xe16, ff, f->npu_slot, f->npu_want, true, npu_a(f->npu), NULL) &&
          hipEventRecord(f->npu_packed, NULL) == hipSuccess);
}

bool q4x_npu_down(Q4xForward *f, const Q4xLayer *L, uint32_t D, uint32_t ff)
{
  return !*f->npu_nsel || run(f, L, NPU_DOWN, D, ff);
}

void q4x_forward_npu_stats(const Q4xForward *f, uint64_t *calls, uint64_t *parts, double *wait_us, uint64_t *wanted,
                           uint64_t *taken)
{
  const NpuStats st = npu_stats(f ? f->npu : NULL);
  *calls = st.calls, *parts = st.parts, *wait_us = st.wait_us;
  *wanted = f ? f->npu_wanted : 0, *taken = f ? f->npu_taken : 0;
}
