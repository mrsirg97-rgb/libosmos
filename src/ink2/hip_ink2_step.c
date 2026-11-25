/* Inkling-2-Small's forward, a step (osmos/ink2/forward.h): a sequence's rows through every layer, the head, the
   kept rows' logits and their pick (osmos/gpu/kept.h) */
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

static bool run(Ink2Forward *f, Ink2Seq *s, const int32_t *tok, uint32_t n, uint32_t row0, char *err)
{
  const Ink2Weights *W = f->w;
  const Ink2Geometry *g = &W->geo;
  const uint32_t C = (uint32_t)g->n_embd, H = (uint32_t)g->n_heads, KV = (uint32_t)g->n_kv_heads,
                 D = (uint32_t)g->head_dim, R = (uint32_t)g->d_rel, taps = (uint32_t)g->conv, maxb = f->maxb,
                 pos = (uint32_t)s->pos;
  const bool bare = g->keys == INK2_KEYS_UNSLOTH_SMALL; /* its convs' residual and scale not applied */
  const float res_scale = 1.0f / sqrtf((float)C), conv_res = bare ? 0.0f : 1.0f,
              conv_scale = bare ? 1.0f : 1.0f / sqrtf((float)taps + 1),
              dyn_scale = bare ? 1.0f : 1.0f / sqrtf(2.0f * taps + 1);
#define GPU(x)                                                                                                         \
  do                                                                                                                   \
  {                                                                                                                    \
    if (!(x))                                                                                                          \
      return osmos_fail(err, "a GPU call failed: " #x);                                                       \
  } while (0)
  ink2_pf(f, -1);
  GPU(hipMemcpy(f->ids_d, tok, n * 4, hipMemcpyHostToDevice) == hipSuccess);
  GPU(ops_embed(W->embed->type, f->embed, C, f->ids_d, n, 1, f->h, f->st));
  GPU(ops_rms(f->h, C, C, f->embed_norm, g->rms_eps, n, f->h, C, f->st));
  for (int c = 0; c < INK2_CONVS; c++) /* every tail's room first: the step's reads and writes in one place */
    for (int sl = 0; sl < g->n_slots; sl++)
      GPU(ink2_tail_room(&s->conv[sl][c], n));
  uint32_t nb = 0;
  ink2_pf(f, PF_HEAD);
  if (f->mtp) /* the drafter's taps: this run's rows of the step's */
    s->tap_row0 = row0, s->tap_pos0 = (int32_t)pos, s->tap_step = f->steps;
  /* the slots in their order: the prelude, the core `passes` times (each from the prelude's output and blocks; a pass
     after the first its own slots and its layers' messages from the pass before's bank), the tail */
  const int core = g->loop.passes ? g->loop.end - g->loop.start : 0, passes = core ? g->loop.passes : 1,
            cs = core ? g->loop.start : g->n_layers;
  const uint64_t nC = (uint64_t)n * C;
  uint32_t pre_nb = 0, nbank = 0;
  for (int step = 0; step < g->n_slots; step++)
  {
    const int in = step - cs, pass = core && in >= 0 && in < passes * core ? in / core : -1;
    const int l = pass >= 0 ? cs + in % core : step < cs ? step : step - (passes - 1) * core;
    const int slot = pass > 0 ? g->n_layers + (pass - 1) * core + in % core : l;
    const bool looped = pass > 0;
    if (core && step == cs) /* the prelude's output and blocks: each pass's start */
    {
      GPU(hipMemcpyAsync(f->pre_h, f->h, nC * 4, hipMemcpyDeviceToDevice, f->st) == hipSuccess);
      pre_nb = nb;
    }
    if (looped && l == cs) /* the pass before's bank (its blocks after its first) and output; then the prelude's */
    {
      nbank = nb - pre_nb - 1;
      if (nbank > f->maxbank)
        return osmos_fail(err, "a pass's blocks past the loop's bank");
      if (nbank)
        GPU(hipMemcpy2DAsync(f->bank, (uint64_t)f->maxbank * C * 4, f->blocks + (uint64_t)(pre_nb + 1) * C,
                             (uint64_t)maxb * C * 4, (uint64_t)nbank * C * 4, n, hipMemcpyDeviceToDevice,
                             f->st) == hipSuccess);
      GPU(hipMemcpyAsync(f->bank_out, f->h, nC * 4, hipMemcpyDeviceToDevice, f->st) == hipSuccess &&
          hipMemcpyAsync(f->h, f->pre_h, nC * 4, hipMemcpyDeviceToDevice, f->st) == hipSuccess);
      nb = pre_nb;
    }
    const Ink2Layer *L = &W->layer[l];
    const Ink2Dev *d = &f->L[l];
    const bool full = !g->sliding[l], boundary = l % g->res_block == 0;
    if (looped) /* the layer's messages: q, k, v, ffn */
      GPU(ink2_attn_res(d->msg_q, INK2_CTRLS, f->bank, nbank, f->maxbank, f->bank_out, n, C, g->rms_eps, res_scale,
                        f->msg, f->res_sc, f->st));
    for (int k = 0; f->mtp && k + 1 < f->mtp->n_taps; k++) /* h entering a target layer: the attention residual's
                                                              running partial */
      if (f->mtp->targets[k] == step) /* (a step of the slots' order: unlooped, the layer) */
        GPU(ink2_mtp_tap(f, k, f->h, n, row0));
    const float *aq = f->h, *ak = f->h, *av = f->h;
    if (d->attn_res_q)
    {
      GPU(ink2_attn_res(d->attn_res_q, full ? 3 : 1, f->blocks, nb, maxb, f->h, n, C, g->rms_eps, res_scale, f->a3,
                        f->res_sc, f->st));
      aq = f->a3, ak = full ? f->a3 + (uint64_t)n * C : f->a3, av = full ? f->a3 + (uint64_t)2 * n * C : f->a3;
    }
    if (looped) /* each input by its message: a sliding layer's q, k and v then apart */
    {
      const float *const xs[3] = {aq, ak, av};
      bool cok = false;
      if (ink2_controls3(f, L, d, xs, f->msg, n, f->ctl, &cok))
        GPU(cok);
      else
        GPU(ink2_control(f, L, d, INK2_CTRL_Q, aq, f->msg, n, f->ctl) &&
            ink2_control(f, L, d, INK2_CTRL_K, ak, f->msg + nC, n, f->ctl + nC) &&
            ink2_control(f, L, d, INK2_CTRL_V, av, f->msg + 2 * nC, n, f->ctl + 2 * nC));
      aq = f->ctl, ak = f->ctl + nC, av = f->ctl + 2 * nC;
    }
    const bool apart = full || looped; /* q, k and v each their own norm's */
    const float *k_norm = full ? d->k_norm : d->attn_norm, *v_norm = full ? d->v_norm : d->attn_norm;
    bool ready, pok = false;
    const float *q_norm = full ? d->q_norm : d->attn_norm;
    const float *xk = apart ? f->kn : f->xn, *xv = apart ? f->vn : f->xn;
    float *dyn_k = full ? f->dyn : NULL, *dyn_v = full ? f->dyn + (uint64_t)n * KV * taps : NULL;
    if (ink2_proj_batch(f, L, d, full, apart, aq, ak, av, q_norm, k_norm, v_norm, n, dyn_k, dyn_v, &pok))
      GPU(pok && ink2_mat(f, L->sink, d->sink, f->xn, n, f->sink));
    else
    {
      GPU(ink2_rms_for(f, L->q, aq, q_norm, n, f->xn, &ready));
      const bool xn16 = ready; /* x16 xn's f16, every row (the q norm's): the other matrices on xn too, not converted again */
      /* the projections on xn in pairs through its one f16 (the norm's; a sliding layer's k and v too: xn theirs),
         then a full layer's k on kn and v on vn, each its norm's f16 - each matrix's bits its own launch's */
      GPU(ink2_mat2(f, L->q, d->q, f->q, L->gate, d->gate, f->gate, f->xn, n, ready) &&
          ink2_mat2(f, L->rel, d->rel, f->rel, NULL, NULL, NULL, f->xn, n, xn16));
      if (apart)
      {
        GPU(ink2_rms_for(f, L->k, ak, k_norm, n, f->kn, &ready) &&
            ink2_mat2(f, L->k, d->k, f->k, NULL, NULL, NULL, xk, n, ready));
        GPU(ink2_rms_for(f, L->v, av, v_norm, n, f->vn, &ready) &&
            ink2_mat2(f, L->v, d->v, f->v, NULL, NULL, NULL, xv, n, ready));
      }
      else
        GPU(ink2_mat2(f, L->k, d->k, f->k, L->v, d->v, f->v, f->xn, n, xn16));
      GPU(apart ? ink2_mat(f, L->sink, d->sink, f->xn, n, f->sink) /* (x16 vn's now: xn converted) */
                : ink2_mat2(f, L->sink, d->sink, f->sink, NULL, NULL, NULL, f->xn, n, xn16));
      if (full)
      {
        GPU(ink2_mat(f, L->k_conv_proj, d->k_conv_proj, xk, n, dyn_k));
        GPU(ink2_mat(f, L->v_conv_proj, d->v_conv_proj, xv, n, dyn_v));
      }
    }
    /* the keys: a full layer's every row to its position, a sliding layer's window (its last window - 1 and these) */
    /* a decode's split attention: its fold writes the out matrix's f16 too (an mm one's) */
    const bool att16 = n <= ATTN_TILE && L->out->dims[0] == (uint64_t)H * D &&
                       ink2_mm_takes(f, ink2_type(L->out), (uint32_t)L->out->dims[1], H * D);
    Ink2Tail *tk = &s->kw[slot], *tv = &s->vw[slot];
    uint32_t n_keys = pos + n;
    if (full)
      tk->held = tv->held = pos; /* the rows to its position, then these */
    else
    {
      GPU(ink2_tail_room(tk, n) && ink2_tail_room(tv, n));
      /* the window's keys from a position a multiple of ATTN_TILE: a query's tiles the same in every step */
      const uint32_t want = tk->held < (uint32_t)g->window - 1 ? tk->held : (uint32_t)g->window - 1;
      n_keys = want + (pos - want) % ATTN_TILE + n;
    }
    /* the convs, clips, v's norm and the caches' rows in one launch (ink2_attn_prep); a conv tail past its room
       takes its rows after (ink2_tail_put) */
    Ink2Tail *ck = &s->conv[slot][INK2_CONV_K], *cv = &s->conv[slot][INK2_CONV_V];
    const float *hk = ink2_tail_last(ck, taps - 1), *hv = ink2_tail_last(cv, taps - 1);
    float *keep_k = ink2_tail_claim(ck, n), *keep_v = ink2_tail_claim(cv, n);
    void *store_k = ink2_tail_claim(tk, n), *store_v = ink2_tail_claim(tv, n);
    if (!store_k || !store_v)
      return osmos_fail(err, "a layer's keys past its cache's room");
    const Ink2Prep prep = {n, H, KV, D, R, taps, f->q, f->rel, d->q_clip, d->rel_clip, g->q_clip, g->rel_clip,
                           g->k_clip, g->rms_eps, conv_res, full ? dyn_scale : conv_scale,
                           (Ink2Tau){pos, full ? g->log_alpha : 0.0f, (float)g->log_floor}, full, f->v8,
                           {f->k, d->k_conv, dyn_k, hk, f->kc, keep_k, store_k},
                           {f->v, d->v_conv, dyn_v, hv, f->vc, keep_v, store_v}};
    GPU(ink2_attn_prep(&prep, f->st));
    GPU((keep_k || ink2_tail_put(ck, f->k, n)) && (keep_v || ink2_tail_put(cv, f->v, n)));
    const void *keys = full ? tk->d : ink2_tail_last(tk, n_keys), *vals = full ? tv->d : ink2_tail_last(tv, n_keys);
    /* key 0's position: a full layer's 0, a window's its first's (a tile's start); a decode's units split */
    const uint32_t pos0 = full ? 0 : pos + n - n_keys;
    ink2_pf(f, PF_IN);
    /* the layer kind's core (its rows: a full layer's Q8 keys, Q5 or Q8 values - Ink2Options' values_q8; a window's
       f16): the step's queries at pos, its keys from pos0 */
    const AttnStep step = {f->q, n, pos, NULL, {keys, vals, pos0, n_keys, NULL, NULL, NULL, NULL, 0, NULL, 0, false, false, NULL, NULL}, f->rel, d->rel_proj, f->sink, f->gate,
                           f->att, att16 ? f->x16 : NULL, att16 ? f->overflow : NULL, 0, false};
    const bool attended = attn_core_run(f->attn[full], &step, f->st);
    GPU(attended);
    ink2_pf(f, full ? PF_FULL : PF_ATTN);
    GPU(ink2_mat2(f, L->out, d->out, f->o, NULL, NULL, NULL, f->att, n, att16));
    Ink2Tail *co = &s->conv[slot][INK2_CONV_OUT], *cf = &s->conv[slot][INK2_CONV_FFN];
    /* r: the output's conv, plus h within a block (a block's first layer's h a block state of its own) */
    if (boundary)
    {
      if (nb >= maxb)
        return osmos_fail(err, "more blocks than the geometry's");
      GPU(hipMemcpy2DAsync(f->blocks + (uint64_t)nb * C, (uint64_t)maxb * C * 4, f->h, (uint64_t)C * 4,
                           (uint64_t)C * 4, n, hipMemcpyDeviceToDevice, f->st) == hipSuccess);
      nb++;
    }
    GPU(ink2_conv_put(co, f->o, n, C, taps, d->out_conv, 0, NULL, conv_res, conv_scale, boundary ? NULL : f->h, f->r, f->st));
    GPU(ink2_attn_res(d->ffn_res_q, 1, f->blocks, nb, maxb, f->r, n, C, g->rms_eps, res_scale, f->tmp, f->res_sc,
                      f->st));
    /* a looped layer's FFN control, its last step (x - tanh(forget) x + tanh(gain) m) in the router's norm's launch
       where the router is mm's (ops_rms_x16_ctrl: ink2_loop_ctrl then rms_for's bits) */
    const bool ctl16 = looped && n <= MM_MAX_TOKENS &&
                       ink2_mm_takes(f, ink2_type(L->router), (uint32_t)L->router->dims[1], (uint32_t)L->router->dims[0]);
    if (looped)
      GPU(ink2_control(f, L, d, INK2_CTRL_FFN, f->tmp, f->msg + 3 * nC, n, ctl16 ? NULL : f->tmp));
    if (ctl16)
    {
      const OpsCtrl ctl = {f->tmp, f->msg + 3 * nC, f->cgain, f->cforget, f->tmp};
      GPU(ops_rms_x16_ctrl(&ctl, C, d->ffn_norm, g->rms_eps, n, f->xn, C, f->x16, f->overflow, f->st));
      f->router16 = true;
    }
    else
      GPU(ink2_rms_for(f, L->router, f->tmp, d->ffn_norm, n, f->xn, &f->router16)); /* (the router's x16: moe's first) */
    ink2_pf(f, PF_MID);
    if (!ink2_moe(f, l, n, err))
      return false;
    ink2_pf(f, PF_MOE);
    /* the layer's output, r + the MoE's conv: the next layer's h - the experts' sums in the conv's launch, or
       (its tail past its room) summed first, the tail's rows */
    {
      const uint32_t K = (uint32_t)g->n_used;
      const float *hist = ink2_tail_last(cf, taps - 1);
      float *keep = ink2_tail_claim(cf, n);
      if (keep)
        GPU(ink2_moe_conv(f->ed, f->m_a, f->m_row, K, n, C, taps, d->ffn_out_conv, hist, conv_res, conv_scale, f->r,
                          f->h, keep, f->st));
      else
        GPU(ink2_moe_sum(f->moe, f->ed, f->m_a, f->m_row, K, n, C, f->st) &&
            ink2_conv_put(cf, f->moe, n, C, taps, d->ffn_out_conv, 0, NULL, conv_res, conv_scale, f->r, f->h, f->st));
    }
    if (f->tap)
      f->tap(f->tap_arg, slot, "l_out", f->h, n, C);
  }
  ink2_pf(f, PF_MID);
  /* the head: the rows from the prompt's end - 1 on */
  GPU(ink2_attn_res(f->out_res_q, 1, f->blocks, nb, maxb, f->h, n, C, g->rms_eps, res_scale, f->tmp, f->res_sc,
                    f->st));
  GPU(ops_rms(f->tmp, C, C, f->out_norm, g->rms_eps, n, f->xn, C, f->st));
  if (f->mtp) /* the final normed hidden: the last tap */
    GPU(ink2_mtp_tap(f, f->mtp->n_taps - 1, f->xn, n, row0));
  const uint32_t VR = (uint32_t)W->output->dims[1], V = f->vocab;
  uint32_t first = 0;
  if (s->prompt_end && s->prompt_end - 1 > pos)
    first = s->prompt_end - 1 - pos < n ? s->prompt_end - 1 - pos : n;
  if (f->n_slots + (n - first) > f->logit_rows)
    return osmos_fail(err, "a step's kept rows past the logits' (Ink2Options' logit_rows)");
  const uint32_t slot0 = f->n_slots; /* this run's kept rows the next slots, in order */
  for (uint32_t t = 0; t < n; t++)
    f->slot[row0 + t] = t >= first ? (int32_t)(slot0 + t - first) : -1;
  f->n_slots += n - first;
  for (uint32_t at = first; at < n; at += f->head_rows)
  {
    const uint32_t k = n - at < f->head_rows ? n - at : f->head_rows;
    GPU(ink2_mat(f, W->output, f->output, f->xn + (uint64_t)at * C, k, f->head));
    GPU(hipMemcpy2DAsync(f->logits + (uint64_t)(slot0 + at - first) * V, (uint64_t)V * 4, f->head, (uint64_t)VR * 4,
                         (uint64_t)V * 4, k, hipMemcpyDeviceToDevice, f->st) == hipSuccess);
  }
  if (first < n)
    GPU(ink2_softcap(f->logits + (uint64_t)slot0 * V, (uint64_t)(n - first) * V, g->logit_softcap, f->st));
#undef GPU
  ink2_pf(f, PF_HEAD);
  const int kind = n > MV_MAX_TOKENS;
  for (int b = 0; f->pf.on && b < PF_N; b++)
    f->pf.total[kind][b] += f->pf.step[b], f->pf.step[b] = 0;
  if (f->pf.on)
    f->pf.tokens[kind] += n, f->pf.steps[kind]++;
  s->pos += (int32_t)n;
  s->reach = s->reach + n < INK2_MAX_BACK ? s->reach + n : INK2_MAX_BACK;
  return true;
}

void ink2_forward_tap(Ink2Forward *f, ink2_tap_fn fn, void *arg)
{
  if (f)
    f->tap = fn, f->tap_arg = arg;
}

bool ink2_step(Ink2Forward *f, Ink2Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
               char err[OSMOS_ERR])
{
  if (!f || !runs || runs > INK2_MAX_RUNS)
    return osmos_fail(err, "a step's runs 1 .. INK2_MAX_RUNS");
  if (!overflow_settle(&f->ovf, err)) /* (a reset the last step owes: refused with its cause while it fails) */
    return false;
  uint64_t total = 0, kept = 0;
  for (uint32_t i = 0; i < runs; i++)
  {
    if (!seqs[i] || seqs[i]->f != f || seqs[i]->failed || !lens[i])
      return osmos_fail(err, "a run of no tokens, or a sequence of another forward or failed");
    for (uint32_t j = 0; j < i; j++)
      if (seqs[j] == seqs[i])
        return osmos_fail(err, "a sequence twice in a step");
    if ((uint64_t)seqs[i]->pos + lens[i] > f->max_ctx)
      return osmos_fail(err, "a sequence past max_ctx");
    total += lens[i];
    /* the rows its logits are kept of: from its prompt's end - 1 on (run()'s) - counted before any work, so a step
       past the logits' rows is refused with its sequences as they were, not failed after the layers ran */
    const uint32_t pos = (uint32_t)seqs[i]->pos, end = seqs[i]->prompt_end; /* (a position: never negative) */
    const uint32_t first = end && end - 1 > pos ? (end - 1 - pos < lens[i] ? end - 1 - pos : lens[i]) : 0;
    kept += lens[i] - first;
  }
  if (total > f->max_tokens)
    return osmos_fail(err, "a step's tokens past max_tokens");
  if (kept > f->logit_rows)
    return osmos_fail(err, "a step's kept rows past the logits' (Ink2Options' logit_rows)");
  f->n = 0;
  for (uint32_t r = 0; r < f->max_tokens; r++) /* (no row kept until the step keeps it) */
    f->slot[r] = -1;
  f->n_slots = 0;
  f->steps++;
  bool ok = true;
  for (uint32_t i = 0, at = 0; ok && i < runs; at += lens[i++])
    ok = run(f, seqs[i], tokens + at, lens[i], at, err);
  ok = ok && (hipDeviceSynchronize() == hipSuccess || osmos_fail(err, "a GPU call failed"));
  hipDeviceSynchronize();
  membrane_end(f->mem);
  int32_t overflow = 0;
  if (ok && hipMemcpy(&overflow, f->overflow, 4, hipMemcpyDeviceToHost) != hipSuccess)
    ok = osmos_fail(err, "a GPU call failed (the step's overflow flag)");
  if (ok && overflow)
    ok = osmos_fail(err, "an activation past f16's range: the step's results are not to be trusted");
  if (!ok) /* what the failed step may have set: not the next step's to answer for */
    { char rerr_[OSMOS_ERR] = ""; overflow_reset(&f->ovf, rerr_); } /* (failing: owed, the next step says so) */
  if (!ok)
  {
    for (uint32_t i = 0; i < runs; i++)
      ink2_seq_fail(seqs[i]);
    return false;
  }
  f->n = (uint32_t)total;
  return true;
}

const float *ink2_logits(const Ink2Forward *f) { return f ? f->logits : NULL; }

void ink2_kept_view(const Ink2Forward *f, KeptView *k)
{
  *k = (KeptView){0};
  if (f)
    *k = (KeptView){f->slot, f->n, f->logits, f->vocab, f->gid, f->amax, f->st, NULL, NULL, NULL};
}

const float *ink2_row_logits(const Ink2Forward *f, uint32_t row)
{
  KeptView k;
  ink2_kept_view(f, &k);
  return kept_row(&k, row);
}

bool ink2_kept(const Ink2Forward *f, uint32_t row) { return ink2_row_logits(f, row) != NULL; }

bool ink2_greedy(Ink2Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR])
{
  KeptView k;
  ink2_kept_view(f, &k);
  const char *why = kept_greedy(&k, row0, rows, ids);
  return why ? osmos_fail(err, "%s", why) : true;
}
