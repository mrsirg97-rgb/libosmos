/* DeepSeek's forward made (osmos/ds4/forward.h): its scratch, sub-buffers and rope, the build in order
   (hip_ds4_resident.c's parts), destroyed in reverse; its reads (budget, room, stream, picks, waits) and the kept rows'
   view */
#define _GNU_SOURCE /* O_DIRECT */
#include <fcntl.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <hip/hip_runtime_api.h>
#include "osmos/model/tensor.h"
#include "osmos/mem/ledger_hip.h"
#include "osmos/run/fit.h"
#include "ds4_impl.h"
#include "osmos/mem/experts.h"
#include "osmos/mem/arena.h"
#include "osmos/mem/hip_slots.h"
#include "osmos/gpu/kvfmt.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/model/qtype.h"
#include "osmos/model/rope.h"
#include "osmos/err.h"
#include "osmos/model/dspark.h"

#define FWD_ROW_THREADS 64  /* Engram's readers: past 64 threads no faster (DESIGN.md) */

static bool scratch(Forward *f)
{
  const Geometry *g = f->g;
  Scratch *b = &f->b;
  const uint64_t D = g->n_embd, hc = g->hc, F = sizeof(float), HQ = (uint64_t)g->n_heads * g->head_dim;
  const uint64_t row = gguf_bytes(f->w->layer[0].eg_embd ? f->w->layer[0].eg_embd->type : 0, g->engram_head_dim);
  const uint64_t cols = g->engram_cols, K = (uint64_t)g->n_used, FF = (uint64_t)g->expert_ff, hd = g->head_dim;
  const uint64_t ih = g->ix_head_dim, nh = g->ix_heads, egv = cols * (uint64_t)g->engram_head_dim;
  const uint64_t B = f->step, Fs = OPS_MAX_SELECT, NS = f->n_subs;
  const uint64_t S = f->max_tokens, P = S * K; /* the experts': a layer's sub-batches at once (forward's max_tokens) */
  uint64_t widest = (uint64_t)g->o_groups * g->o_lora > D ? (uint64_t)g->o_groups * g->o_lora : D;
  widest = HQ > widest ? HQ : widest; /* the attention's output: wo_a's groups read it whole */
  const struct
  {
    void *field;
    uint64_t bytes;
  } want[] = {
      {&b->h, B * hc * D * F},    {&b->hcp, B * ((2 + hc) * hc + 1) * OPS_HC_PARTS * F}, {&b->a_pre, B * hc * F},
      {&b->a_post, B * hc * F},   {&b->a_comb, B * hc * hc * F},    {&b->f_pre, B * hc * F},
      {&b->f_post, NS * B * hc * F}, {&b->f_comb, NS * B * hc * hc * F}, {&b->pre0, B * hc * F},
      {&b->x, B * D * F},         {&b->qa, B * (uint64_t)g->q_lora * F}, {&b->q, B * HQ * F},
      {&b->kv, B * hd * F},       {&b->o, B * HQ * F},  {&b->oa, B * (uint64_t)g->o_groups * g->o_lora * F},
      {&b->y, B * D * F},         {&b->mid, NS * B * hc * D * F},   {&b->xf, S * D * F},
      {&b->logits, S * (uint64_t)g->n_experts * F}, {&b->vocab, (f->logit_rows && f->logit_rows < B ? f->logit_rows : B) * (uint64_t)g->n_vocab * F},
      {&b->rw, S * K * F},        {&b->ga, P * FF * F},             {&b->upr, P * FF * F},
      {&b->hh, P * FF * F},       {&b->ye, P * D * F},              {&b->moe, S * D * F},
      {&b->sga, S * FF * F},      {&b->sup, S * FF * F},            {&b->shh, S * FF * F},
      {&b->sye, S * D * F},       {&b->up, FWD_UP_BYTES(P)}, {&b->lat, B * hd * F},
      {&b->kvc, B * hd * F},      {&b->scg, B * hd * F},            {&b->ikr, B * ih * F},
      {&b->qix, B * nh * ih * F}, {&b->wix, B * nh * F},
      {&b->kv_p, B * kv_row_bytes(KV_FP8, (uint32_t)hd)}, {&b->lat_p, B * kv_row_bytes(KV_FP4_E4M3, (uint32_t)hd)},
      {&b->ikr_p, B * kv_row_bytes(KV_FP4_POW2, (uint32_t)ih)}, {&b->row_f, hd * F},
      {&b->ckv_f, (uint64_t)f->ckv_f_rows * hd * F}, {&b->ring_f, (uint64_t)f->slots * hd * F},
      {&b->score, (f->score_rows * (uint64_t)f->score_stride > B * Fs ? f->score_rows * (uint64_t)f->score_stride
                                                                        : B * Fs) * F}, /* the indexer's; hip_ds4_skip.c's */
      {&b->att, OPS_ATTN_SCRATCH(B, g->n_heads, (uint64_t)g->window + g->ix_topk) * F},
      {&b->erow, B * cols * row}, {&b->erow_f, B * egv * F},        {&b->ekv, B * (hc + 1) * D * F},
      {&b->ids, B * 4},           {&b->pos, B * 4},                 {&b->rpos, B * 4},
      {&b->ridx, S * K * 4},      {&b->sel, B * Fs * 4},            {&b->n_sel, B * 4},
      {&b->cand, B * f->cand_words * 4},
      {&b->clist, f->cand_words ? (uint64_t)f->score_rows * (uint64_t)g->cand_blocks * 4 : 0},
      {&b->clist_n, f->cand_words ? (uint64_t)f->score_rows * 4 : 0},
      {&b->bmax, f->cand_words ? (uint64_t)f->score_rows * ((f->score_stride + g->cand_block - 1) / g->cand_block) * F : 0},
      {&b->rwp, P * F},                 {&b->x16, B * (widest > egv ? widest : egv) * 2},
      /* past the last slot, a part's tiles read whole: group_gemm_kernel loads a block's every column tile's x (the
         WMMAs past an expert's tokens skipped), up to 8 tiles past its own - as Qwen's (src/q4x/hip_q4x_make.c) */
      {&b->xe16, (FWD_MAX_SLOTS(P, g->n_experts) + 8 * MM_TILE) * (D > FF ? D : FF) * 2},
      {&b->overflow, 4},          {&b->oa_cnt, FWD_MAX_GROUPS * 4},
      {&b->keep, B * 4},          {&b->back, B * 4},
      {&b->gw, S * FWD_GUESS_K * F}, /* a layer's tokens: a prompt's read ahead guesses them all (ahead.h) */
  };
  _Static_assert(sizeof want / sizeof *want == sizeof(Scratch) / sizeof(void *), "every buffer of Scratch, once");
  for (size_t i = 0; i < sizeof want / sizeof *want; i++)
    if (!fwd_alloc(&f->bufs, want[i].field, want[i].bytes))
      return false;
  float pre0[FWD_MAX_BATCH * OPS_MAX_HC] = {0}; /* the first block's pre, a token: stream 0 */
  for (uint64_t i = 0; i < B; i++)
    pre0[i * hc] = 1.0f;
  int n_eg = 0;
  for (int l = 0; l < g->n_layers; l++)
    n_eg += g->engram[l];
  f->xf0 = b->xf, f->moe0 = b->moe, f->mid0 = b->mid, f->f_post0 = b->f_post, f->f_comb0 = b->f_comb;
  f->ovf = (OverflowFlag){b->overflow, 4, false, NULL}; /* (zeroed: osmos/gpu/bufs.h) */
  return fwd_put(b->pre0, pre0, B * hc * F) && fwd_alloc(&f->bufs, &f->argmax_part, ops_argmax_scratch((uint32_t)B)) &&
         (!f->n_tap || fwd_alloc(&f->bufs, &f->taps, (uint64_t)f->step * f->n_tap * f->g->n_embd * F)) &&
         fwd_mapped(&f->bufs, &f->r_ids, P * 4) &&
         hipHostGetDevicePointer((void **)&f->r_ids_dev, f->r_ids, 0) == hipSuccess &&
         fwd_mapped(&f->bufs, &f->r_guess, S * FWD_GUESS_K * 4) &&
         hipHostGetDevicePointer((void **)&f->r_guess_dev, f->r_guess, 0) == hipSuccess &&
         fwd_pinned(&f->bufs, &f->r_up, FWD_UP_BYTES(P)) &&
         (f->m_by = malloc(P * 4)) && (f->m_row = malloc(P * 4)) && (f->m_base = malloc(P * 4)) &&
         (f->m_slot0 = malloc(P * 4)) &&
         (f->m_gidx = malloc(FWD_MAX_SLOTS(P, g->n_experts) * 4)) &&
         (f->m_hidx = malloc(FWD_MAX_SLOTS(P, g->n_experts) * 4)) &&
         fwd_pinned(&f->bufs, &f->stage, NS * B * (uint64_t)(f->n_eg = n_eg ? n_eg : 1) * cols * row) &&
         fwd_pinned(&f->bufs, &f->pin, sizeof *f->pin) && fwd_subs(f) &&
         (f->slot = malloc(f->max_tokens * sizeof *f->slot)) && memset(f->slot, 0xff, f->max_tokens * sizeof *f->slot) &&
         hipEventCreateWithFlags(&f->routed, hipEventDisableTiming) == hipSuccess &&
         hipEventCreateWithFlags(&f->passed, hipEventDisableTiming) == hipSuccess;
}

/* the sub-batches' own state (ds4_impl.h's Sub): the first the Forward's own buffers, the others their own */
bool fwd_subs(Forward *f)
{
  const Geometry *g = f->g;
  Scratch *b = &f->b;
  const uint64_t B = f->step, hc = g->hc, F = sizeof(float);
  Sub *s0 = &f->sub[0];
  s0->h = b->h, s0->f_pre = b->f_pre, s0->pos = b->pos, s0->sel = b->sel, s0->n_sel = b->n_sel, s0->cand = b->cand;
  s0->oa_cnt = b->oa_cnt, s0->keep = b->keep, s0->back = b->back, s0->pin = f->pin, s0->rows = f->rows;
  bool ok = true;
  for (uint32_t k = 1; ok && k < f->n_subs; k++)
  {
    Sub *s = &f->sub[k];
    ok = fwd_alloc(&f->bufs, &s->h, B * hc * g->n_embd * F) && fwd_alloc(&f->bufs, &s->f_pre, B * hc * F) && fwd_alloc(&f->bufs, &s->pos, B * 4) &&
         fwd_alloc(&f->bufs, &s->sel, B * OPS_MAX_SELECT * 4) && fwd_alloc(&f->bufs, &s->n_sel, B * 4) &&
         fwd_alloc(&f->bufs, &s->cand, B * f->cand_words * 4) &&
         fwd_alloc(&f->bufs, &s->oa_cnt, FWD_MAX_GROUPS * 4) && fwd_alloc(&f->bufs, &s->keep, B * 4) && fwd_alloc(&f->bufs, &s->back, B * 4) &&
         fwd_pinned(&f->bufs, &s->pin, sizeof *s->pin) &&
         (s->rows = calloc(B * WEIGHTS_MAX_LAYERS * g->engram_cols, sizeof *s->rows)) != NULL;
  }
  return ok;
}

static bool freqs(Forward *f)
{
  const Geometry *g = f->g;
  float a[ROPE_MAX_DIM / 2], c[ROPE_MAX_DIM / 2];
  const Rope win = {g->rope_dim, g->rope_base, 0, g->rope_factor, g->yarn_beta_fast, g->yarn_beta_slow};
  const Rope comp = {g->rope_dim, g->compress_rope_base, g->rope_original, g->rope_factor, g->yarn_beta_fast,
                     g->yarn_beta_slow};
  const uint64_t bytes = (uint64_t)g->rope_dim / 2 * sizeof(float);
  return rope_freqs(&win, a) && rope_freqs(&comp, c) && fwd_alloc(&f->bufs, &f->freqs_win, bytes) &&
         fwd_alloc(&f->bufs, &f->freqs_comp, bytes) && fwd_put(f->freqs_win, a, bytes) && fwd_put(f->freqs_comp, c, bytes);
}

Forward *forward_create(const Weights *w, int n_layers, uint32_t max_ctx, uint64_t budget, char err[OSMOS_ERR])
{
  return forward_create_ranked(w, n_layers, max_ctx, budget, NULL, err);
}

Forward *forward_create_ranked(const Weights *w, int n_layers, uint32_t max_ctx, uint64_t budget,
                               const uint64_t *picks, char err[OSMOS_ERR])
{
  const ForwardOptions o = {n_layers, max_ctx, budget, picks, FORWARD_MAX_TOKENS, false, false, 0, 0, 0, NULL,
                            false, 0, NULL, NULL, 0, NULL, false};
  return forward_create_with(w, &o, err);
}

Forward *forward_create_with(const Weights *w, const ForwardOptions *o, char err[OSMOS_ERR])
{
  const int n_layers = o ? o->n_layers : 0;
  const uint32_t max_ctx = o ? o->max_ctx : 0, max_tokens = o && o->max_tokens ? o->max_tokens : FORWARD_MAX_TOKENS;
  const uint64_t *picks = o ? o->picks : NULL;
  const uint64_t budget = o ? o->budget : 0;
  if (!w || n_layers < 1 || n_layers > w->geo.n_layers || !max_ctx || max_tokens > FORWARD_MAX_TOKENS)
    return osmos_fail(err, "no weights, or layers, a context or a step it cannot run"), NULL;
  if (w->n_files > FWD_MAX_FILES)
    return osmos_fail(err, "more files than this build maps"), NULL;
  Forward *f = calloc(1, sizeof *f);
  if (!f)
    return osmos_fail(err, "out of memory"), NULL;
  f->ledger = o->ledger, f->bufs.ledger = o->ledger;
  f->w = w, f->g = &w->geo, f->n_layers = n_layers, f->max_ctx = max_ctx, f->no_guess = o->no_guess, f->budget = budget;
  f->ahead_limit = o->ahead_limit, f->logit_rows = o->logit_rows;
  f->ram_kept = o->fill_set ? o->fill : FIT_RAM_FREE;
  if (max_tokens < FWD_MAX_BATCH) /* a small step's server: every buffer of a step at its size (DESIGN.md) */
    f->n_subs = 1, f->max_tokens = max_tokens, f->step = (max_tokens + MM_TILE - 1) / MM_TILE * MM_TILE;
  else
    f->n_subs = (max_tokens + FWD_MAX_BATCH - 1) / FWD_MAX_BATCH, f->max_tokens = f->n_subs * FWD_MAX_BATCH,
    f->step = FWD_MAX_BATCH;
  f->slots = w->geo.window + f->step;
  fwd_decoder(f);
  f->score_stride = 1; /* no compressed layer: never read */
  for (int l = 0; l < n_layers; l++)
    if (w->geo.compress[l] && max_ctx / (uint32_t)w->geo.compress[l] > f->score_stride)
      f->score_stride = max_ctx / (uint32_t)w->geo.compress[l];
  f->ckv_f_rows = f->score_stride < FWD_CKV_F32_ROWS ? f->score_stride : FWD_CKV_F32_ROWS;
  const uint64_t fit = FWD_SCORE_BYTES / ((uint64_t)f->score_stride * sizeof(float));
  f->score_rows = fit >= f->step ? f->step : fit ? (uint32_t)fit : 1;
  if (w->geo.cand_source >= 0 && w->geo.cand_source < n_layers && w->geo.compress[w->geo.cand_source] &&
      (max_ctx / (uint32_t)w->geo.compress[w->geo.cand_source] + w->geo.cand_block - 1) / w->geo.cand_block >
          (uint32_t)w->geo.cand_blocks) /* a context this long passes the candidate blocks: their bits a token */
    f->cand_words = ((f->score_stride + w->geo.cand_block - 1) / w->geo.cand_block + 31) / 32;
  f->eg_fd = -1;
  f->pf.on = getenv("DS4_PROFILE") != NULL;
  f->no_int8 = o && o->no_int8;
  for (int i = 0; i < FWD_MAX_FILES; i++)
    f->fds[i] = f->dfds[i] = -1;
  /* a drafter's plan before any GPU memory: its index read, its target layers the model's (a drafter refused here,
     not after the build) - the taps then made with the step's buffers */
  DsparkIndex *dix = NULL;
  bool at[WEIGHTS_MAX_LAYERS] = {false};
  if (o->drafter && !fwd_drafter_plan(f, o->drafter, &dix, at, err))
    return forward_destroy(f), NULL;
  if (o->drafter)
    fwd_taps_plan(f, at);
  bool ok = true, direct = true;
  for (int i = 0; ok && i < w->n_files; i++) /* the model's files, then an overlay's (weights.h) */
    ok = (f->fds[i] = open(w->paths[i], O_RDONLY)) >= 0 &&
         (direct = (f->dfds[i] = open(w->paths[i], O_RDONLY | O_DIRECT)) >= 0);
  char eerr[OSMOS_ERR] = "";
  if (!ok && !direct) /* refused loudly: the pool's reads are direct (no silent buffered path) */
    osmos_fail(err, "a model file could not be opened O_DIRECT (its filesystem: tmpfs and some FUSE ones refuse it)");
  else if (!ok)
    osmos_fail(err, "a model file could not be opened");
  else if ((f->eg_fd = fwd_engram_fd(f, w)) < 0)
    ok = osmos_fail(err, "the Engram's file could not be opened");
  else if (!(f->hash = engram_create(w, f->fds, eerr)))
    ok = osmos_fail(err, "%s", eerr);
  else if (!fwd_gates(f, w, err) || !fwd_guess_routers(f, w, err))
    ok = false;
  else if (!(f->rows = calloc((size_t)FWD_MAX_BATCH * WEIGHTS_MAX_LAYERS * f->g->engram_cols, sizeof *f->rows)) ||
           !(f->eg_off = calloc((size_t)FWD_MAX_SUBS * FWD_MAX_BATCH * fwd_engram_layers(w, n_layers) *
                                    f->g->engram_cols + 1,
                                sizeof *f->eg_off)))
    ok = osmos_fail(err, "out of memory");
  else if (!(f->eg_read = rowread_create(FWD_ROW_THREADS)))
    ok = osmos_fail(err, "Engram's readers could not start");
  const size_t nu = (size_t)n_layers * w->geo.n_experts;
  ok = ok && fwd_dense(f, err) && fwd_oa_tables(f, err) &&
       ((f->picks = calloc(nu, sizeof *f->picks)) ||
        osmos_fail(err, "out of memory"));
  if (ok && !(freqs(f) && scratch(f)))
    ok = osmos_fail(err, "a GPU allocation refused");
  if (ok)
  { /* the attention: a token's window over the ring, then its selected compressed rows (ix_topk at most) */
    const Geometry *g = f->g;
    const AttnCoreDesc d = {.heads = g->n_heads, .kv_heads = 1, .dk = g->head_dim, .dv = g->head_dim,
                            .k_rows = ATTN_ROWS_FP8, .v_rows = ATTN_ROWS_FP8, .s_rows = ATTN_ROWS_FP4,
                            .max_sel = (uint32_t)g->ix_topk, .window = (uint32_t)g->window, .slots = f->slots,
                            .scaling = (float)pow(g->head_dim, -0.5), .term = ATTN_TERM_NONE, .sink = ATTN_SINK_HEAD,
                            .gate = ATTN_GATE_NONE, .block = ATTN_BLOCK_HEADS, .max_tokens = (uint32_t)f->step,
                            .max_keys = (uint32_t)g->window + (uint32_t)g->ix_topk};
    ok = (f->latent = attn_core_create(&d, f->b.att, OPS_ATTN_SCRATCH(f->step, g->n_heads, (uint64_t)g->window + g->ix_topk) * sizeof(float),
                                       err)) != NULL;
  }
  /* made whole (ForwardOptions'): the lane and the drafter, then the experts in what the whole build's budget leaves
     past them, then the NPU (it maps the experts' host slots) */
  ok = ok && fwd_lane_make(f, o->cpu_threads, o->cpu_cores, err);
  if (ok)
    f->lane = o->lane;
  if (ok && dix)
    ok = (f->drafter = fwd_drafter_make(f, dix, err)) != NULL, dix = NULL; /* (the drafter's: freed with it) */
  dspark_index_destroy(dix);
  ok = ok && fwd_experts(f, budget, picks, err) && (!o->npu || fwd_npu_make(f, o->npu, o->npu_share, err));
  if (!ok)
  {
    forward_destroy(f);
    return NULL;
  }
  return f;
}

struct Drafter *forward_drafter(const Forward *f) { return f ? f->drafter : NULL; }

uint64_t forward_budget(const Forward *f) { return f ? f->budget : 0; }

bool forward_room(const Forward *f, uint64_t *bytes)
{
  size_t free_b = 0, total = 0;
  if (!f || hipMemGetInfo(&free_b, &total) != hipSuccess)
    return false;
  *bytes = fit_room((uint64_t)free_b, fit_ram_available(), f->ram_kept); /* the pool's memory taken at create, in free_b */
  return true;
}

void forward_destroy(Forward *f)
{
  if (!f)
    return;
  static const char *const part[8] = {"embed", "engram", "attn mixes", "attention", "ffn mixes", "experts",
                                      "layer out", "head + settle"};
  for (int k = 0; f->pf.on && k < 2; k++)
  {
    if (!f->pf.steps[k])
      continue;
    fprintf(stderr, "ds4 profile (%s steps): %llu steps, %llu tokens, %.2f ms a step\n", k ? "prompt" : "small",
            (unsigned long long)f->pf.steps[k], (unsigned long long)f->pf.tokens[k],
            1e3 * f->pf.step_s[k] / (double)f->pf.steps[k]);
    for (int p = 0; p < 8; p++)
      fprintf(stderr, "  %-14s %8.2f ms a step\n", part[p], 1e3 * f->pf.s[k][p] / (double)f->pf.steps[k]);
  }
  (void)hipDeviceSynchronize(); /* a failed step's kernels done: what is freed or unlocked, the GPU is past */
  fwd_drafter_destroy(f->drafter); /* first: it borrows the forward (its taps, its head) */
  attn_core_destroy(f->latent);
  membrane_destroy(f->mem);
  host_slots_destroy(f->hs), host_slots_destroy(f->hs_stream); /* after the membrane: no unit still in them */
  resident_destroy(f->res);
  hip_slots_destroy(f->gpu_slots); /* after the resident set: its memory released through it */
  rowread_destroy(f->eg_read);
  free(f->eg_off);
  free(f->tdev);
  free(f->ttype);
  if (f->xf0) /* the views back at the whole buffers: the record frees what was made */
    f->b.xf = f->xf0, f->b.moe = f->moe0, f->b.mid = f->mid0, f->b.f_post = f->f_post0, f->b.f_comb = f->f_comb0;
  free(f->m_by), free(f->m_row), free(f->m_base), free(f->m_slot0), free(f->m_gidx), free(f->m_hidx);
  free(f->picks), free(f->units);
  fwd_cpu_destroy(f);
  fwd_npu_destroy(f);
  if (f->sub[0].h) /* the first sub-batch's buffers back where they came from: the record frees them */
  {
    const Sub *s = &f->sub[0];
    f->b.h = s->h, f->b.f_pre = s->f_pre, f->b.pos = s->pos, f->b.sel = s->sel, f->b.n_sel = s->n_sel;
    f->b.cand = s->cand;
    f->b.oa_cnt = s->oa_cnt, f->b.keep = s->keep, f->b.back = s->back, f->pin = s->pin, f->rows = s->rows;
  }
  for (int k = 1; k < FWD_MAX_SUBS; k++)
    free(f->sub[k].rows);
  gpu_bufs_free(&f->bufs); /* every GPU and pinned buffer the forward made - Scratch's, the subs', the lane's, the taps',
                              the tables': from the record */
  if (f->routed)
    hipEventDestroy(f->routed);
  if (f->passed)
    hipEventDestroy(f->passed);
  for (int i = 0; i < FWD_MAX_FILES; i++)
  {
    if (f->fds[i] >= 0)
      close(f->fds[i]);
    if (f->dfds[i] >= 0)
      close(f->dfds[i]);
  }
  if (f->eg_fd >= 0)
    close(f->eg_fd);
  engram_destroy(f->hash);
  free(f->slot);
  free(f->rows);
  free(f);
}

void forward_probe(Forward *f, forward_probe_fn fn, void *arg)
{
  f->probe = fn;
  f->probe_arg = arg;
}

const float *forward_stream(const Forward *f) { return f->b.h; }
const uint64_t *forward_picks(const Forward *f) { return f->picks; }
void forward_guesses(const Forward *f, uint64_t *streamed, uint64_t *guessed, uint64_t *read)
{
  MembraneStats s;
  membrane_stats(f->mem, &s);
  *streamed = s.streamed, *guessed = s.guessed;
  if (read)
    *read = s.guess_read;
}
void forward_waits(const Forward *f, uint64_t n[2], uint64_t ns[2])
{
  MembraneStats s;
  membrane_stats(f->mem, &s);
  n[0] = s.waits[0], n[1] = s.waits[1], ns[0] = s.wait_ns[0], ns[1] = s.wait_ns[1];
}
const float *forward_logits(const Forward *f) { return f->b.vocab; }
uint32_t forward_logit_rows(const Forward *f) { return f ? f->vocab_rows : 0; }

static bool settle(void *f) { return fwd_settle(f); } /* (the forward's wait: a decode's spins - hip_ds4_wait.c) */

void forward_kept_view(const Forward *f, KeptView *k)
{
  *k = (KeptView){0};
  if (f) /* (the picks' scratch: b.ids, free between steps - the next one's tokens written at its start) */
    *k = (KeptView){f->slot,   f->step_n,       f->b.vocab,        (uint32_t)f->g->n_vocab, f->b.ids, f->argmax_part,
                    NULL,      f->pin->greedy,  settle,            (void *)f};
}

bool forward_greedy(Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR])
{
  KeptView k;
  forward_kept_view(f, &k);
  const char *why = kept_greedy(&k, row0, rows, ids);
  return !why || osmos_fail(err, "%s", why);
}
