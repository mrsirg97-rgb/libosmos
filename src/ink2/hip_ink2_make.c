/* Inkling-2-Small's forward made (osmos/ink2/forward.h): a step's buffers, the build in order
   (hip_ink2_resident.c's parts), destroyed in reverse - the profile's totals said; its reads; the profile's timer */
#define _GNU_SOURCE
#include <fcntl.h>
#include <hip/hip_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "ink2_impl.h"
#include "osmos/model/tensor.h"
#include "osmos/gpu/ink2_ops.h"
#include "osmos/gpu/mm.h"
#include "osmos/mem/experts.h"
#include "osmos/model/moe_route.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/kept.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/gpu/quant.h"
#include "osmos/mem/arena.h"
#include "osmos/err.h"

/* the profile's sections' names (PF_*, ink2_impl.h), and its timer: each section after a synchronize */
static const char *const kPfName[PF_N] = {"inputs",         "attn sliding",  "attn full",     "mid",           "moe",
                                          "  of it waits", "  of it router", "  of it kernels", "head"};

double ink2_pf_now(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
void ink2_pf(Ink2Forward *f, int b)
{
  if (!f->pf.on)
    return;
  hipDeviceSynchronize();
  const double t = ink2_pf_now();
  if (b >= 0)
    f->pf.step[b] += t - f->pf.at;
  f->pf.at = t;
}

/* mm's: its table's types, and BF16 where the model's dense weights are BF16 (f->bf16_mm: Mini's, as released) - mm's
   by name, not its table's. a model whose BF16 is a side tensor (Inkling-2-Small's sliding layers' sink projections)
   keeps its matvec and its bits; the other families' BF16 matrices their matvec's. by the model, the type and the
   shape alone: a decode's and a prompt's the same */
bool ink2_mm_takes(const Ink2Forward *f, uint32_t type, uint32_t rows, uint32_t cols)
{
  return (mm_known(type) || (type == QTYPE_BF16 && f->bf16_mm)) && mm_fits(type, rows, cols);
}

/* a layer kind's attention (full: the cache's Q8 keys, Q5 or Q8 values, every key; else f16 rows, the window) -
   Inkling's: its relative bias (a sliding layer's out to its window), its sink, its gate a value's */
static AttnCoreDesc attn_desc(const Ink2Forward *f, bool full)
{
  const Ink2Geometry *g = &f->w->geo;
  const uint32_t window = (uint32_t)g->window, D = (uint32_t)g->head_dim;
  return (AttnCoreDesc){.heads = (uint32_t)g->n_heads,
                        .kv_heads = (uint32_t)g->n_kv_heads,
                        .dk = D,
                        .dv = D,
                        .k_rows = full ? ATTN_ROWS_KV8 : ATTN_ROWS_F16,
                        .v_rows = full ? (f->v8 ? ATTN_ROWS_KV8 : ATTN_ROWS_KV5) : ATTN_ROWS_F16,
                        .window = full ? 0u : window,
                        .scaling = 1.0f / sqrtf((float)D),
                        .term = ATTN_TERM_REL,
                        .d_rel = (uint32_t)g->d_rel,
                        .extent = full ? (uint32_t)g->rel_extent : window,
                        .sink = ATTN_SINK_TOKEN,
                        .gate = ATTN_GATE_VALUE,
                        .block = ATTN_BLOCK_TOKENS,
                        .max_tokens = f->max_tokens,
                        .max_keys = full ? f->max_ctx : window + f->max_tokens + ATTN_TILE}; /* (a window's keys: tile-aligned) */
}

static bool buffers(Ink2Forward *f)
{
  const Ink2Geometry *g = &f->w->geo;
  const uint64_t T = f->max_tokens, C = (uint64_t)g->n_embd, H = (uint64_t)g->n_heads,
                 KV = (uint64_t)g->n_kv_heads, D = (uint64_t)g->head_dim, R = (uint64_t)g->d_rel,
                 taps = (uint64_t)g->conv, E = (uint64_t)g->n_experts, K = (uint64_t)g->n_used,
                 F = (uint64_t)g->ff_exp, VR = f->w->output->dims[1],
                 CH0 = H * D > C ? H * D : C, CH = g->loop.passes && 2 * C > CH0 ? 2 * C : CH0,
                 X16 = (T + MM_TILE - 1) / MM_TILE * MM_TILE; /* a step's rows: its producers' f16 whole (mm_tokens_x16) */
  const uint32_t head = ink2_type(f->w->output);
  f->head_rows = ink2_mm_takes(f, head, (uint32_t)VR, (uint32_t)C) ? MM_TILE * mm_col_tiles(head, MM_MAX_TOKENS) : MV_MAX_TOKENS;
  /* the experts' down rows and a decode's split attention share one buffer: the attention's partials are dead before
     the layer's MoE writes the down rows, and those are summed before the next layer's attention */
  const AttnCoreDesc full = attn_desc(f, true), win = attn_desc(f, false);
  const uint64_t fa_full = attn_core_scratch(&full), fa_win = attn_core_scratch(&win),
                 fa_bytes = fa_full > fa_win ? fa_full : fa_win, down_bytes = T * K * C * 4,
                 ed_bytes = fa_bytes > down_bytes ? fa_bytes : down_bytes;
  struct
  {
    void **p;
    uint64_t bytes;
  } b[] = {
      {(void **)&f->ids_d, T * 4}, {(void **)&f->h, T * C * 4}, {(void **)&f->blocks, T * f->maxb * C * 4},
      {(void **)&f->a3, 3 * T * C * 4}, {(void **)&f->xn, T * C * 4}, {(void **)&f->kn, T * C * 4},
      {(void **)&f->vn, T * C * 4}, {(void **)&f->q, T * H * D * 4}, {(void **)&f->k, T * KV * D * 4},
      {(void **)&f->v, T * KV * D * 4}, {(void **)&f->kc, T * KV * D * 4}, {(void **)&f->vc, T * KV * D * 4},
      {(void **)&f->rel, T * H * R * 4}, {(void **)&f->gate, T * H * D * 4}, {(void **)&f->sink, T * H * 4},
      {(void **)&f->att, T * H * D * 4}, {(void **)&f->dyn, 2 * T * KV * taps * 4}, {(void **)&f->o, T * C * 4},
      {(void **)&f->r, T * C * 4}, {(void **)&f->moe, T * C * 4}, {(void **)&f->logit, T * E * 4},
      {(void **)&f->tmp, T * C * 4}, {(void **)&f->pick, 2 * T * K * 4},
      {(void **)&f->gpick, T * K * 4}, {(void **)&f->gw, T * K * 4},
      {(void **)&f->eg, T * K * F * 4}, {(void **)&f->eu, T * K * F * 4}, {(void **)&f->ed, ed_bytes}, {&f->m_dev, ink2_table_bytes(T * K, E)}, {&f->x16, X16 * CH * 2},
      {&f->xe16, ink2_table_slots(T * K, E) * C * 2}, {(void **)&f->overflow, 4},
      {(void **)&f->head, f->head_rows * VR * 4}, {(void **)&f->logits, (uint64_t)f->logit_rows * f->vocab * 4},
      {(void **)&f->gid, T * 4}, {&f->amax, ops_argmax_scratch((uint32_t)T)},
      {(void **)&f->res_sc, INK2_RES_SCRATCH * 4}, {&f->bx16, INK2_BX16 * MM_BATCH_TOKENS * CH * 2},
      /* a decode's int8 experts: x and the activations rounded (MV_MAX_TOKENS tokens, their picks) */
      {(void **)&f->q_x, MV_MAX_TOKENS * C}, {(void **)&f->q_xd, MV_MAX_TOKENS * (C / 32) * 4},
      {(void **)&f->q_h, MV_MAX_TOKENS * K * F}, {(void **)&f->q_hd, MV_MAX_TOKENS * K * (F / 32) * 4},
  };
  for (size_t i = 0; i < sizeof b / sizeof *b; i++)
    if (!ink2_dalloc(f, b[i].p, b[i].bytes))
      return false;
  if (g->loop.passes) /* the loop's */
  {
    const uint64_t core = (uint64_t)(g->loop.end - g->loop.start), rank = (uint64_t)g->loop.gate_rank;
    f->maxbank = core / (uint64_t)g->res_block > 1 ? (uint32_t)(core / (uint64_t)g->res_block - 1) : 1;
    struct
    {
      float **p;
      uint64_t bytes;
    } lb[] = {
        {&f->pre_h, T * C * 4}, {&f->bank, T * f->maxbank * C * 4}, {&f->bank_out, T * C * 4},
        {&f->msg, INK2_CTRLS * T * C * 4}, {&f->ctl, 3 * T * C * 4}, {&f->ccat, 2 * T * C * 4},
        {&f->cgate, T * rank * 4}, {&f->cgain, T * C * 4}, {&f->cforget, T * C * 4},
        {&f->ccat3, 3 * MM_BATCH_TOKENS * 2 * C * 4}, {&f->cgate3, 3 * MM_BATCH_TOKENS * rank * 4},
        {&f->cgain3, 3 * MM_BATCH_TOKENS * C * 4}, {&f->cforget3, 3 * MM_BATCH_TOKENS * C * 4},
    };
    for (size_t i = 0; i < sizeof lb / sizeof *lb; i++)
      if (!ink2_dalloc(f, (void **)lb[i].p, lb[i].bytes))
        return false;
  }
  f->x16_rows = (uint32_t)((T + MM_TILE - 1) / MM_TILE * MM_TILE);
  char why[OSMOS_ERR] = "";
  for (int k = 0; k < 2; k++) /* (the down rows' buffer, ed_bytes: the cores' scratch) */
    if (!(f->attn[k] = attn_core_create(k ? &full : &win, f->ed, ed_bytes, why)))
      return fprintf(stderr, "ink2: %s\n", why), false;
  if (!gpu_bufs_host(&f->bufs, f->ledger, &f->hpick, 2 * T * K * 4) ||
      !gpu_bufs_host(&f->bufs, f->ledger, &f->hgpick, T * K * 4) ||
      !gpu_bufs_host(&f->bufs, f->ledger, &f->m_host, ink2_table_bytes(T * K, E)) ||
      !gpu_bufs_host(&f->bufs, f->ledger, &f->pin_tok, T * 4) ||
      !gpu_bufs_host(&f->bufs, f->ledger, &f->pin_overflow, 4))
    return false;
  *f->pin_overflow = 0;
  f->ovf = (OverflowFlag){f->overflow, 4, false, f->pin_overflow};
  if (!overflow_reset(&f->ovf, NULL))
    return false;
  f->slot = malloc(T * sizeof *f->slot);
  f->m_at = malloc((E + 1) * sizeof *f->m_at), f->m_by = malloc(T * K * sizeof *f->m_by);
  f->m_key = malloc(ink2_table_entries(T * K, E) * sizeof *f->m_key);
  f->m_ent = malloc(ink2_table_entries(T * K, E) * sizeof *f->m_ent), f->m_run = malloc(4 * E * sizeof *f->m_run);
  f->m_ex = malloc(E * sizeof *f->m_ex), f->m_cold = malloc(E * sizeof *f->m_cold);
  f->m_gidx = malloc(ink2_table_slots(T * K, E) * 4), f->m_hidx = malloc(ink2_table_slots(T * K, E) * 4);
  f->m_build = malloc(ink2_table_bytes(T * K, E)); /* the tables built in cached memory, then one copy into the pinned */
  return f->hpick && f->slot && f->hgpick && f->m_at && f->m_by && f->m_host && f->m_key && f->m_build &&
         f->m_ent && f->m_run && f->m_ex && f->m_cold && f->m_gidx && f->m_hidx;
}

Ink2Forward *ink2_forward_create(const Ink2Weights *w, const Ink2Options *o, char err[OSMOS_ERR])
{
  if (!w || !o || !o->max_ctx || !o->max_tokens)
    return osmos_fail(err, "no weights, or max_ctx / max_tokens 0"), NULL;
  Ink2Forward *f = calloc(1, sizeof *f);
  if (!f)
    return osmos_fail(err, "out of memory"), NULL;
  f->ledger = o->ledger;
  f->pf.on = o->profile;
  f->no_int8 = o->no_int8;
  f->bf16_mm = w->layer[0].q && ink2_type(w->layer[0].q) == QTYPE_BF16; /* (mm_takes': BF16 the model's dense weights) */
  f->w = w, f->v8 = o->values_q8, f->max_ctx = o->max_ctx, f->max_tokens = o->max_tokens, f->vocab = (uint32_t)w->geo.n_vocab;
  f->logit_rows = o->logit_rows && o->logit_rows < o->max_tokens ? o->logit_rows : o->max_tokens;
  f->maxb = (uint32_t)((w->geo.n_layers + w->geo.res_block - 1) / w->geo.res_block);
  const uint32_t C = (uint32_t)w->geo.n_embd, F = (uint32_t)w->geo.ff_exp;
  for (int l = 0; l < w->geo.n_layers; l++) /* the experts on mm at every step (moe) */
  {
    const Ink2Layer *L = &w->layer[l];
    if (!ink2_mm_takes(f, L->exp_gate->type, F, C) || !ink2_mm_takes(f, L->exp_up->type, F, C) || !ink2_mm_takes(f, L->exp_down->type, C, F))
      return osmos_fail(err, "an expert tensor of a type or shape mm has no kernel for"),
             ink2_forward_destroy(f), NULL;
  }
  if (!ink2_upload(f))
    return osmos_fail(err, "a dense tensor not read or uploaded (GPU memory, or its file)"),
           ink2_forward_destroy(f), NULL;
  if (!buffers(f))
    return osmos_fail(err, "GPU memory refused (the step's buffers)"), ink2_forward_destroy(f), NULL;
  f->mtp = o->mtp;
  if (f->mtp && !ink2_mtp_up(f, err))
    return ink2_forward_destroy(f), NULL;
  if (!(f->picks = calloc((size_t)w->geo.n_layers * w->geo.n_experts, sizeof *f->picks)))
    return osmos_fail(err, "out of memory"), ink2_forward_destroy(f), NULL;
  f->fixed_bytes = f->bufs.bytes; /* the fixed part, counted: its record (the dense weights, the buffers, the drafter) -
                                     not the GPU's free memory before and after, which anything else moves */
  if (!ink2_experts(f, o->room, o->budget, o->picks, err) || !ink2_whole_tables(f, err))
    return ink2_forward_destroy(f), NULL;
  return f;
}

void ink2_forward_destroy(Ink2Forward *f)
{
  if (!f)
    return;
  for (int k = 0; f->pf.on && k < 2; k++)
  {
    double all = 0;
    for (int b = 0; b < PF_N; b++)
      all += b == PF_WAIT || b == PF_ROUTE || b == PF_XK ? 0 : f->pf.total[k][b];
    if (!f->pf.steps[k])
      continue;
    fprintf(stderr, "ink2 profile (%s steps): %llu steps, %llu tokens, %.3f s\n", k ? "prompt" : "decode",
            (unsigned long long)f->pf.steps[k], (unsigned long long)f->pf.tokens[k], all);
    for (int b = 0; b < PF_N; b++)
      fprintf(stderr, "  %-13s %8.2f ms a step  %5.1f%%\n", kPfName[b], 1e3 * f->pf.total[k][b] / (double)f->pf.steps[k],
              100 * f->pf.total[k][b] / all);
  }
  if (f->pf.on && (f->pf.steps[0] || f->pf.steps[1]))
  {
    MembraneStats ms;
    membrane_stats(f->mem, &ms);
    fprintf(stderr, "  membrane (all steps): %llu picks not in memory, %llu of them guessed; %llu guesses read; waits "
                    "%llu unguessed %.1f s, %llu guessed %.1f s\n",
            (unsigned long long)ms.streamed, (unsigned long long)ms.guessed, (unsigned long long)ms.guess_read,
            (unsigned long long)ms.waits[0], ms.wait_ns[0] * 1e-9, (unsigned long long)ms.waits[1],
            ms.wait_ns[1] * 1e-9);
  }
  hipDeviceSynchronize();
  attn_core_destroy(f->attn[0]), attn_core_destroy(f->attn[1]);
  for (int l = 0; l < INK2_MTP_MAX_LAYERS; l++)
    attn_core_destroy(f->md.attn[l]);
  membrane_destroy(f->mem);
  host_slots_destroy(f->slots), host_slots_destroy(f->stream);
  for (int i = 0; f->fds && i < f->w->model->n_files; i++)
    if (f->fds[i] >= 0)
      close(f->fds[i]);
  free(f->fds);
  for (int l = 0; l < INK2_MAX_LAYERS; l++)
    for (int k = 0; k < 3; k++)
      free(f->L[l].scale[k]);
  gpu_bufs_free(&f->bufs); /* every GPU and pinned buffer, the forward's and its drafter's: from the record */
  free(f->picks), free(f->picks_all), free(f->slot), free(f->m_at), free(f->m_by);
  free(f->m_key), free(f->m_build), free(f->m_ent), free(f->m_run), free(f->m_ex), free(f->m_cold);
  free(f->m_gidx), free(f->m_hidx);
  free(f);
}

uint64_t ink2_forward_bytes(const Ink2Forward *f) { return f ? f->fixed_bytes + f->pinned_bytes : 0; }

bool ink2_forward_whole(const Ink2Forward *f) { return f && f->whole; }

/* the host's counts, plus (a whole membrane's) the GPU's: copied at the call */
const uint64_t *ink2_forward_picks(const Ink2Forward *f)
{
  if (!f || !f->whole)
    return f ? f->picks : NULL;
  const size_t n = (size_t)f->w->geo.n_layers * f->w->geo.n_experts;
  uint32_t *d = malloc(n * 4);
  if (!d || hipMemcpy(d, f->picks_d, n * 4, hipMemcpyDeviceToHost) != hipSuccess)
    return free(d), NULL;
  for (size_t u = 0; u < n; u++)
    f->picks_all[u] = f->picks[u] + d[u];
  free(d);
  return f->picks_all;
}
