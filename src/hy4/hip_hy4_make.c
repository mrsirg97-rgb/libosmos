/* Hy4's forward made (osmos/hy4/forward.h): the build in order (hip_hy4_resident.c's parts), the head's tables,
   destroyed in reverse; its reads */
#define _GNU_SOURCE
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime_api.h>
#include "osmos/model/tensor.h"
#include "osmos/gpu/hy4_ops.h"
#include "osmos/gpu/mm.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/quant.h"
#include "osmos/gpu/bufs.h"
#include "osmos/gpu/upload.h"
#include "osmos/mem/arena.h"
#include "osmos/mem/host_slots.h"
#include "osmos/model/expert_units.h"
#include "osmos/model/fnv.h"
#include "osmos/model/qtype.h"
#include "osmos/gpu/q8.h"
#include "hy4_impl.h"
#include "osmos/err.h"
#include "osmos/err_hip.h"

void hy4_forward_destroy(Hy4Forward *f)
{
  if (!f)
    return;
  if (f->st)
    hipStreamSynchronize((hipStream_t)f->st), hipStreamDestroy((hipStream_t)f->st);
  hy4_dfly_free(f);
  for (int l = 0; l < HY4_MAX_LAYERS; l++)
    attn_core_destroy(f->mla[l]);
  if (f->pf.on && f->pf.steps)
    fprintf(stderr,
            "hy4 profile: %llu steps, %llu tokens; a step %.2f ms; the Engram read %.3f ms a step (preads %.3f, copy and "
            "decode %.3f), %.1f rows and %.1f KB a step\n",
            (unsigned long long)f->pf.steps, (unsigned long long)f->pf.tokens, 1e3 * f->pf.step_s / (double)f->pf.steps,
            1e3 * (f->pf.rowread_s + f->pf.copy_s) / (double)f->pf.steps, 1e3 * f->pf.rowread_s / (double)f->pf.steps,
            1e3 * f->pf.copy_s / (double)f->pf.steps, (double)f->pf.rows / (double)f->pf.steps,
            (double)f->pf.bytes / 1024.0 / (double)f->pf.steps);
  if (f->pf.on > 1 && f->pf.dsteps)
  {
    static const char *const part[24] = {"engram update", "iHC attn", "attention", "iHC ffn", "experts", "dense ffn",
                                         "embed", "head", "  proj in", "  q_b", "  kv, q absorb", "  indexer",
                                         "  mla attn", "  v absorb", "  proj out", "  x: router, down, shared in",
                                         "  router picks", "  route, gather", "  experts gate", "  experts up",
                                         "  swiglu, down", "  routed up", "  shared down, post", "  gather sum"};
    for (int k = 0; k < 24; k++)
      fprintf(stderr, "  %-14s %7.2f ms a %s step\n", part[k], 1e3 * f->pf.part_s[k] / (double)f->pf.dsteps,
              f->pf.on > 2 ? "prompt" : "decode");
    static const char *const dpart[4] = {"draft: embed, fusion", "draft: 5 layers", "draft: corrections x5",
                                         "draft: head + pick x5"};
    for (int k = 0; f->pf.drafts && k < 4; k++)
      fprintf(stderr, "  %-22s %7.2f ms a draft\n", dpart[k], 1e3 * f->pf.part_s[24 + k] / (double)f->pf.drafts);
  }
  experts_destroy(&f->ex); /* its readers stopped, then its memory */
  for (int i = 0; f->xfds && i < f->w->model->n_files; i++)
    if (f->xfds[i] >= 0)
      close(f->xfds[i]);
  free(f->xfds);
  free((void *)f->h_tab), free(f->h_ex);
  gpu_free_table(f->ledger, f->w->model, f->dev, f->in_joint); /* (a joint's members its buffer's) */
  rowread_destroy(f->eg_read); /* (its pending reads into the pinned stage waited for, before the record frees it) */
  gpu_bufs_free(&f->bufs);     /* every GPU and pinned buffer the forward made: from the record (before the joints'
                                  table, whose slots it NULLs) */
  free(f->joints), free(f->in_joint);
  engram_destroy(f->eg);
  free(f->eg_off), free(f->slot), free(f->h_sidx);
  free(f->h_rcnt), free(f->rkey);
  free(f->h_rgidx), free(f->h_rhidx), free(f->moe_build), free(f->h_estart), free(f->h_by), free(f->h_run), free(f->h_ent);
  free(f);
}

/* a read's per-head absorption (k_b: q_nope to the latent; v_b: the latent to the head's values) as one mm_group a
   layer over its heads: each layer's heads' matrices, and where head h reads (its columns of the tiled x16: the q /
   latent rows converted once) and writes - fixed for the forward's life */
/* layer l's MLA attention (hy4_mla_attn's, now the core's): one latent row for every head, keys the row (latent and
   rope) and values its latent, a head's sink; a sliding layer's window over the ring, a sparse one's groups */
static AttnCoreDesc mla_desc(const Hy4Forward *f, int l)
{
  const Hy4Geometry *g = &f->w->geo;
  const bool sw = g->sliding[l];
  const uint32_t lat = (uint32_t)g->kv_lora, rd = (uint32_t)g->rope_dim, qk = (uint32_t)(g->nope_dim + g->rope_dim);
  return (AttnCoreDesc){.heads = (uint32_t)g->n_heads[l], .kv_heads = 1, .dk = lat + rd, .dv = lat,
                        .k_rows = f->kv8 ? ATTN_ROWS_MLA_Q8 : ATTN_ROWS_MLA_F32,
                        .v_rows = f->kv8 ? ATTN_ROWS_MLA_Q8 : ATTN_ROWS_MLA_F32, .window = sw ? (uint32_t)g->window : 0,
                        .scaling = 1.0f / sqrtf((float)qk), .term = ATTN_TERM_NONE, .sink = ATTN_SINK_HEAD,
                        .gate = ATTN_GATE_NONE, .block = ATTN_BLOCK_HEADS, .ratio = sw ? 0 : (uint32_t)g->ix_pool,
                        .unit = 32, .slots = sw ? f->ring : 0, .apart = true,
                        .max_tokens = f->max_tokens, .max_keys = HY4_MLA_MAX_KEYS};
}
/* the cores' scratch: the most heads' (every layer's core borrows it, a step's alone) */
static uint64_t mla_scratch(const Hy4Forward *f, uint32_t heads)
{
  AttnCoreDesc d = mla_desc(f, 0);
  d.heads = heads;
  return attn_core_scratch(&d);
}

static bool head_tables(Hy4Forward *f)
{
  const Hy4Geometry *g = &f->w->geo;
  const uint32_t L = (uint32_t)g->n_layers, M = HY4_MAX_HEADS, lat = (uint32_t)g->kv_lora, vd = (uint32_t)g->v_dim;
  const void **vb = calloc((size_t)L * M, sizeof *vb);
  uint32_t xo[HY4_MAX_HEADS], yo[HY4_MAX_HEADS];
  for (uint32_t h = 0; h < M; h++)
    xo[h] = h * lat * MM_TILE, yo[h] = h * vd;
  bool ok = vb != NULL;
  for (uint32_t l = 0; ok && l < L; l++)
  {
    const Hy4Layer *Ly = &f->w->layer[l];
    if (Ly->k_b->type != QTYPE_Q8_0 || Ly->v_b->type != QTYPE_Q8_0 ||
        (!g->sliding[l] && (Ly->ix_pool_value->type != QTYPE_Q8_0 || Ly->ix_pool_gate->type != QTYPE_Q8_0)))
    {
      ok = false;
      break;
    }
    for (uint32_t h = 0; h < (uint32_t)g->n_heads[l]; h++)
      vb[(size_t)l * M + h] = (const uint8_t *)hy4_dev(f, Ly->v_b) + (uint64_t)h * vd * lat / 32 * 34;
  }
  const uint64_t tb = (uint64_t)L * M * sizeof *vb;
  ok = ok && hy4_alloc(&f->bufs, &f->vb_tab, tb) &&
       hy4_alloc(&f->bufs, &f->vb_xo, M * 4) && hy4_alloc(&f->bufs, &f->vb_yo, M * 4) &&
       hy4_alloc(&f->bufs, &f->hcnt, M * 4) &&
       hipMemcpy(f->vb_tab, vb, tb, hipMemcpyHostToDevice) == hipSuccess &&
       hipMemcpy(f->vb_xo, xo, M * 4, hipMemcpyHostToDevice) == hipSuccess &&
       hipMemcpy(f->vb_yo, yo, M * 4, hipMemcpyHostToDevice) == hipSuccess;
  free(vb);
  return ok;
}

Hy4Forward *hy4_forward_create(const Hy4Weights *w, const int *fds, const Hy4Options *o, char err[OSMOS_ERR])
{
  if (!w || !fds || !o || !o->max_tokens || o->max_tokens > o->max_ctx)
    return osmos_fail(err, "no weights, files or options, or a step past the context"), NULL;
  if (w->geo.engram_ngram - 1 > HY4_STATE_HISTORY)
    return osmos_fail(err, "an n-gram longer than a saved state's history keeps"), NULL;
  Hy4Forward *f = calloc(1, sizeof *f);
  if (!f)
    return osmos_fail(err, "out of memory"), NULL;
  f->ledger = o->ledger, f->bufs.ledger = o->ledger;
  const Hy4Geometry *g = &w->geo;
  f->w = w, f->max_tokens = o->max_tokens, f->max_ctx = o->max_ctx, f->kv8 = !o->kv_f32;
  f->logit_rows = o->logit_rows && o->logit_rows < o->max_tokens ? o->logit_rows : o->max_tokens;
  f->ring = (uint32_t)g->window + o->max_tokens + HY4_MAX_BACK;
  f->kring = (o->max_tokens + (uint32_t)g->ix_pool + HY4_MAX_BACK + (uint32_t)g->ix_pool - 1) / (uint32_t)g->ix_pool *
             (uint32_t)g->ix_pool;
  f->eg_fd = -1;
  f->pf.on = getenv("HY4_PROFILE") ? (atoi(getenv("HY4_PROFILE")) > 2 ? 3 : atoi(getenv("HY4_PROFILE")) > 1 ? 2 : 1) : 0;
  f->q8 = !o->no_int8; /* a decode's experts over int8 inputs (the pack's; OSMOS_Q8=0 at any step: F32) */
  for (int l = 0; l < HY4_MAX_LAYERS; l++)
    f->dfly_slot[l] = -1;
  int hmax = 0;
  for (int l = 0; l < g->n_layers; l++)
    hmax = g->n_heads[l] > hmax ? g->n_heads[l] : hmax;
  const uint64_t n = o->max_tokens, lr = f->logit_rows, D = (uint64_t)g->n_embd, hc = (uint64_t)g->hc;
  const uint64_t H = (uint64_t)hmax, qk = (uint64_t)(g->nope_dim + g->rope_dim), row = (uint64_t)(g->kv_lora + g->rope_dim);
  const uint64_t K = (uint64_t)g->n_used, L = (uint64_t)g->latent, ffe = (uint64_t)g->ff_exp;
  const uint64_t eg_in = (uint64_t)(g->engram_ngram - 1) * g->engram_heads * g->engram_dim;
  const uint64_t entries = n * K; /* mv_group_tok's entries at most: a token's picks each its own at worst */
  /* (xg and eg_h a chunk's rows past the last: mv_group_tok reads an entry's MV_MAX_TOKENS rows, uses its count) */
  const GpuWant want[] = {
      {&f->h, n * hc * D * F}, {&f->xn, n * D * F}, {&f->y, n * D * F}, {&f->post, n * hc * F}, {&f->hcs, hy4_hc_scratch((uint32_t)n) * F},
      {&f->table, hy4_mla_table_bytes((uint32_t)n, (uint32_t)H)}, {&f->attn_part, mla_scratch(f, (uint32_t)H)},
      {&f->qr, n * g->q_lora * F}, {&f->qres, n * g->q_lora * F}, {&f->q, n * H * qk * F}, {&f->kva, n * row * F},
      {&f->gt, n * H * g->v_dim * F}, {&f->qa, n * H * row * F}, {&f->olat, n * H * g->kv_lora * F},
      {&f->ao, n * H * g->v_dim * F}, {&f->out, n * D * F}, {&f->ixk, n * g->ix_dim * F}, {&f->ixkn, n * g->ix_dim * F},
      {&f->ixq, n * g->ix_heads * g->ix_dim * F}, {&f->ixw, n * g->ix_heads * F},
      {&f->scores, (n < HY4_SCORE_ROWS ? n : HY4_SCORE_ROWS) * (o->max_ctx / g->ix_pool + 1) * F}, {&f->sel, n * (g->ix_topk / g->ix_pool) * 4},
      {&f->n_sel, n * 4}, {&f->emb, n * eg_in * F}, {&f->ekv, n * (hc + 1) * D * F}, {&f->ga, n * g->n_ff * F},
      {&f->up, n * g->n_ff * F}, {&f->act, n * g->n_ff * F}, {&f->logit_r, n * g->n_experts * F}, {&f->wtok, n * K * F},
      {&f->rin, n * L * F}, {&f->xg, (n * K + MV_MAX_TOKENS) * L * F}, {&f->eg_g, n * K * ffe * F},
      {&f->eg_u, n * K * ffe * F}, {&f->eg_h, (n * K + MV_MAX_TOKENS) * ffe * F}, {&f->eg_y, n * K * L * F}, {&f->wsel, n * K * F}, {&f->rsum, n * L * F},
      {&f->xq, n * L}, {&f->xqd, n * (L / 32) * F}, {&f->hq, (n * K + MV_MAX_TOKENS) * ffe},
      {&f->hqd, (n * K + MV_MAX_TOKENS) * (ffe / 32) * F},
      {&f->sh, n * D * F}, {&f->idx, n * K * 4}, {&f->gidx, n * K * 4}, {&f->sidx, n * K * 4},
      {&f->cnt, entries * 4}, {&f->xrow, entries * 4}, {&f->wptr, 3 * entries * sizeof(void *)},
      {&f->tok, n * 4}, {&f->pos, n * 4}, {&f->hx, lr * hc * D * F}, {&f->logits, lr * g->n_vocab * F},
      {&f->ids, lr * 4}, {&f->argmax, ops_argmax_scratch((uint32_t)lr)},
  };
  /* the read math's: the widest matrix input an mm reads (o_proj's heads x v_dim, the dense down's n_ff, Engram's) */
  uint64_t wide = (uint64_t)g->n_ff > H * g->v_dim ? (uint64_t)g->n_ff : H * g->v_dim;
  wide = wide > H * qk ? wide : H * qk, wide = wide > H * g->kv_lora ? wide : H * g->kv_lora;
  if (D > wide)
    wide = D;
  if (ffe * (uint64_t)g->n_shexp > wide)
    wide = ffe * (uint64_t)g->n_shexp;
  const uint64_t ecap = hy4_read_entries(n, g), scap = hy4_read_slots(n, g);
  const uint64_t xe = L > ffe ? L : ffe;
  /* compact MoE staging capacity: covers prefill (2*nk + 3*ecap ptrs + 5*ecap u32 + 2*scap u32) and decode */
  f->moe_cap = (3 * n * K + 2 * scap + 5 * ecap) * 4 + 3 * ecap * sizeof(void *) + 16 * MOE_STAGE_ALIGN;
  const struct
  {
    void *p;
    uint64_t bytes;
  } read_want[] = {
      {&f->x16, (uint64_t)MM_MAX_TOKENS * wide * 2}, {&f->xe16, scap * xe * 2}, {&f->overflow, 8},
      {&f->moe_dev, f->moe_cap},
  };
  f->skip_output = o->head_q8 && w->output->type == QTYPE_BF16;
  bool ok = (uint64_t)hmax <= HY4_MAX_HEADS || osmos_fail(err, "more heads than this build's tables");
  ok = ok && (hipStreamCreate((hipStream_t *)&f->st) == hipSuccess || osmos_fail(err, "the step's stream not made"));
  ok = ok && hy4_upload(f, err);
  if (ok)
    ok = head_tables(f) || osmos_fail(err, "a GPU allocation refused (the heads' tables)");
  if (ok && f->skip_output)
  { /* the head's Q8_0 copy (Hy4Options' head_q8): BF16 output.weight was never uploaded */
    const GgufTensor *out = f->w->output;
    float *x = tensor_f32(f->w->model, out, &(uint64_t){0});
    ok = (x && hy4_q8_up(&f->bufs, x, out->dims[1], (uint32_t)out->dims[0], &f->head8)) ||
         osmos_fail(err, "the head's Q8_0 copy not made");
    free(x);
  }
  uint32_t joint_most = 0;
  ok = ok && (((f->in_joint = calloc(w->model->n_tensors, sizeof *f->in_joint)) && hy4_joints_make(f, &joint_most) &&
               hy4_alloc(&f->bufs, &f->jo_attn, (uint64_t)joint_most * F) && hy4_alloc(&f->bufs, &f->jo_ffn, (uint64_t)joint_most * F)) ||
              osmos_fail(err, "the joint matrices not made (GPU memory)"));
  ok = ok && ledger_check(f->ledger, hy4_alloc(&f->bufs, &f->ix_spread, hy4_ix_spread_bytes()) ? hipSuccess
                                                                                             : hipErrorOutOfMemory,
                          err, HY4_IX_SPREAD); /* (whole: SCOPE.md) */
  for (size_t i = 0; ok && i < sizeof read_want / sizeof *read_want; i++)
    ok = hy4_alloc(&f->bufs, read_want[i].p, read_want[i].bytes) || osmos_fail(err, "GPU memory refused (the read's buffers)");
  ok = ok && hy4_pinned(&f->bufs, &f->pin_flags, 8) &&
       (f->ovf = (OverflowFlag){f->overflow, 8, false, f->pin_flags}, overflow_reset(&f->ovf, err)) &&
       (f->fault = f->overflow + 1) && (f->h_rcnt = malloc(ecap * 4)) && (f->rkey = malloc(ecap * 8)) &&
       (f->h_rgidx = malloc(scap * 4)) && (f->h_rhidx = malloc(scap * 4)) &&
       (f->moe_build = malloc(f->moe_cap)) && hy4_pinned(&f->bufs, &f->moe_pin, f->moe_cap) &&
       hy4_pinned(&f->bufs, &f->pin_idx, n * K * 4) && hy4_pinned(&f->bufs, &f->pin_w, n * K * F) &&
       (f->h_estart = malloc(((size_t)g->n_experts + 1) * 4)) && (f->h_run = malloc((size_t)g->n_experts * 16)) &&
       (f->h_ent = malloc(ecap * sizeof *f->h_ent));
  for (int l = 0; ok && l < g->n_layers; l++)
    ok = (hy4_as_f32(f, w->layer[l].hc_attn_fn, &f->fn_attn[l]) && hy4_as_f32(f, w->layer[l].hc_ffn_fn, &f->fn_ffn[l])) ||
         osmos_fail(err, "iHC's mixes not made F32");
  ok = ok && (hy4_as_f32(f, w->out_hc_fn, &f->fn_out) || osmos_fail(err, "the head's iHC not made F32")) &&
       ((hy4_freqs(f, g->rope_base_swa, (uint32_t)g->rope_dim, &f->freqs_swa) &&
         hy4_freqs(f, g->rope_base, (uint32_t)g->rope_dim, &f->freqs_sparse)) ||
        osmos_fail(err, "the ropes' tables not made")) &&
       hy4_engram_reader(f, fds, err);
  ok = ok && (gpu_bufs_alloc(&f->bufs, f->ledger, want, sizeof want / sizeof *want) ||
              osmos_fail(err, "GPU memory refused (a step's buffers)"));
  ok = ok && hy4_pinned(&f->bufs, &f->pin_tok, n * 4) && hy4_pinned(&f->bufs, &f->pin_pos, n * 4) &&
       (f->slot = malloc(n * 4)) && (f->h_sidx = malloc(n * K * 4)) && (f->h_by = malloc(n * K * 4)) &&
       hipDeviceSynchronize() == hipSuccess;
  for (int l = 0; ok && l < g->n_layers; l++)
  { /* a layer's MLA core: its heads, a sliding one's window over the ring, a sparse one's groups of ix_pool */
    const AttnCoreDesc d = mla_desc(f, l);
    ok = (f->mla[l] = attn_core_create(&d, f->attn_part, mla_scratch(f, (uint32_t)H), err)) != NULL;
  }
  ok = ok && (!o->dfly || hy4_dfly_load(f, o->dfly, o->dfly_q8, err)); /* (whole: its drafter part of the build) */
  /* the fixed part, counted (experts_plan's): the table read, the records' buffers, DFly2's - not the GPU's free
     memory before and after, which anything else moves (DFly2 loaded after the experts: outside the budget) */
  if (ok)
    f->fixed_bytes = gpu_table_bytes(w->model, f->dev, f->in_joint) + f->bufs.bytes + hy4_dfly_bytes(f);
  ok = ok && hy4_experts(f, o->budget == FIT_BUDGET_NOW ? experts_fit_now(f->ledger, f->fixed_bytes) : o->budget, err);
  if (!ok)
    return hy4_forward_destroy(f), (err && !*err ? osmos_fail(err, "the forward not made") : false), NULL;
  return f;
}

uint64_t hy4_forward_bytes(const Hy4Forward *f) { return f ? f->fixed_bytes + f->pinned_bytes : 0; }

bool hy4_forward_whole(const Hy4Forward *f) { return f && f->whole; }

uint64_t hy4_identity(const Hy4Forward *f)
{
  if (!f)
    return 0;
  uint64_t h = fnv(&f->w->geo, sizeof f->w->geo, FNV_SEED);
  h = fnv_u64(f->ring, h); /* a sliding layer's ring: its rows' places another forward's */
  h = fnv_u64(f->kv8, h);  /* the cache rows' format */
  for (int i = 0; i < f->w->model->n_files; i++)
    h = fnv_file(f->w->model->paths[i], h);
  return h;
}
