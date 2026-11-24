/* Inkling-2-Small's forward, a layer's controls (osmos/ink2/forward.h): the convs and their tails, the projections
   batched where their shapes agree */
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

bool ink2_conv_put(Ink2Tail *t, const float *x, uint32_t n, uint32_t ch, uint32_t taps, const float *w,
                     uint32_t heads, const float *dyn, float res, float scale, const float *add, float *y, void *st)
{
  const float *hist = ink2_tail_last(t, taps - 1);
  float *keep = ink2_tail_claim(t, n);
  const bool ok = dyn ? !add && ink2_conv_dyn_keep(x, n, ch, taps, heads, w, dyn, hist, res, scale, y, keep, st)
                      : ink2_conv_keep(x, n, ch, taps, w, hist, res, scale, add, y, keep, st);
  return ok && (keep || ink2_tail_put(t, x, n));
}

/* a run: n tokens of s at its next positions through the model, their logits at rows row0 .. (those before the
   prompt's end - 1 not) */
/* f->bx16's input i: MM_BATCH_TOKENS rows of the widest input (buffers' CH) */
static void *bx16_at(const Ink2Forward *f, int i)
{
  const Ink2Geometry *g = &f->w->geo;
  const uint64_t C = (uint64_t)g->n_embd, HD = (uint64_t)g->n_heads * (uint64_t)g->head_dim,
                 CH0 = HD > C ? HD : C, CH = g->loop.passes && 2 * C > CH0 ? 2 * C : CH0;
  return (uint8_t *)f->bx16 + (uint64_t)i * MM_BATCH_TOKENS * CH * 2;
}

/* matrices of one type and cols on mm (mm_takes): an mm_batch's */
static bool batchable(const Ink2Forward *f, const GgufTensor *const *t, int count, uint32_t cols)
{
  const uint32_t type = ink2_type(t[0]);
  for (int i = 0; i < count; i++)
    if (t[i] && (ink2_type(t[i]) != type || (uint32_t)t[i]->dims[0] != cols ||
                 !ink2_mm_takes(f, type, (uint32_t)t[i]->dims[1], cols)))
      return false;
  return true;
}

/* a decode's (n <= MM_BATCH_TOKENS) norms in one launch (q's into xn and f->x16; apart: k's into kn and bx16 0,
   v's into vn and bx16 1 - each rms_for's bits) and its projections on them in one mm_batch: q, gate and rel on q's
   norm, k and v on theirs (else q's), a full layer's conv projections on k's and v's - each matrix's bits mat2's
   (mm_run's tiles; the norms' f16 mm_x16's bytes). false: not taken (a matrix not the batch's type on mm), nothing
   launched; *ok the launches' */
bool ink2_proj_batch(Ink2Forward *f, const Ink2Layer *L, const Ink2Dev *d, bool full, bool apart, const float *aq,
                       const float *ak, const float *av, const float *q_norm, const float *k_norm,
                       const float *v_norm, uint32_t n, float *dyn_k, float *dyn_v, bool *ok)
{
  const uint32_t C = (uint32_t)f->w->geo.n_embd;
  const GgufTensor *t[7] = {L->q, L->gate, L->rel, L->k, L->v, full ? L->k_conv_proj : NULL,
                            full ? L->v_conv_proj : NULL};
  if (n > MM_BATCH_TOKENS || !batchable(f, t, 7, C))
    return false;
  void *x16k = apart ? bx16_at(f, 0) : f->x16, *x16v = apart ? bx16_at(f, 1) : f->x16;
  const OpsRmsPart norms[3] = {{aq, q_norm, f->xn, f->x16, C, C, C, 0},
                               {ak, k_norm, f->kn, x16k, C, C, C, 0},
                               {av, v_norm, f->vn, x16v, C, C, C, 0}};
  *ok = ops_rms_x16_many(norms, apart ? 3 : 1, C, f->w->geo.rms_eps, n, f->overflow, f->st);
  const void *w[7] = {d->q, d->gate, d->rel, d->k, d->v, d->k_conv_proj, d->v_conv_proj};
  const void *x[7] = {f->x16, f->x16, f->x16, x16k, x16v, x16k, x16v};
  float *y[7] = {f->q, f->gate, f->rel, f->k, f->v, dyn_k, dyn_v};
  MmPart p[7];
  uint32_t count = 0;
  for (int i = 0; i < 7; i++)
    if (t[i])
    {
      const uint32_t rows = (uint32_t)t[i]->dims[1];
      p[count++] = (MmPart){w[i], rows, x[i], y[i], rows};
    }
  *ok = *ok && mm_batch(ink2_type(L->q), C, p, count, n, f->st);
  return true;
}

/* a loop's control of input x by message m (upstream's control): gate = down(concat(rms(x), rms(m))), out = x -
   tanh(forget(gate)) x + tanh(gain(gate)) m; out may be x, NULL: the gain and forget alone (the caller applies them:
   ops_rms_x16_ctrl) */
bool ink2_control(Ink2Forward *f, const Ink2Layer *L, const Ink2Dev *d, int c, const float *x, const float *m,
                    uint32_t n, float *out)
{
  const uint32_t C = (uint32_t)f->w->geo.n_embd;
  const float eps = f->w->geo.rms_eps;
  /* the two norms' f16 side by side in x16 where down is mm's: its input's, not converted again (ops_rms's bits) */
  const bool r16 = n <= MM_MAX_TOKENS && ink2_mm_takes(f, ink2_type(L->ctrl_down[c]), (uint32_t)L->ctrl_down[c]->dims[1],
                                                  (uint32_t)L->ctrl_down[c]->dims[0]);
  /* the down's gate and its f16 in one launch (mm_run_x16: mm_run then mm_x16's bytes), the gain and forget on that
     f16 (bx16 5's: the controls' gates', free by the FFN's) - its norms the f16 alone (no reader of their F32: x16's
     the down's input, x and m loop_ctrl's); else the norms' F32 (ccat), the down, then the pair converting its gate */
  const uint32_t G = (uint32_t)f->w->geo.loop.gate_rank;
  const GgufTensor *gf[2] = {L->ctrl_gain[c], L->ctrl_forget[c]};
  void *g16 = bx16_at(f, 5);
  if (r16 && batchable(f, gf, 2, G))
  {
    const OpsRmsPart norms16[2] = {{x, NULL, NULL, f->x16, C, 2 * C, 2 * C, 0}, {m, NULL, NULL, f->x16, C, 2 * C, 2 * C, C}};
    if (!ops_rms_x16_many(norms16, 2, C, eps, n, f->overflow, f->st))
      return false;
    if (mm_run_x16(ink2_type(L->ctrl_down[c]), d->ctrl_down[c], G, 2 * C, f->x16, n, f->cgate, G, g16, f->overflow,
                   f->st))
      return mm_tokens(ink2_type(gf[0]), d->ctrl_gain[c], C, f->cgain, C, d->ctrl_forget[c], C, f->cforget, C, G,
                       f->cgate, G, n, g16, f->overflow, true, f->st) &&
             (!out || ink2_loop_ctrl(x, m, f->cgain, f->cforget, (uint64_t)n * C, out, f->st));
    /* mm_run_x16 refused the shape: the path below, its norms again with their F32 (the same values, deterministic) */
  }
  const OpsRmsPart norms[2] = {{x, NULL, f->ccat, f->x16, C, 2 * C, 2 * C, 0},
                               {m, NULL, f->ccat + C, f->x16, C, 2 * C, 2 * C, C}};
  const bool normed = r16 ? ops_rms_x16_many(norms, 2, C, eps, n, f->overflow, f->st)
                          : ops_rms(x, C, C, NULL, eps, n, f->ccat, 2 * C, f->st) &&
                                ops_rms(m, C, C, NULL, eps, n, f->ccat + C, 2 * C, f->st);
  if (!normed)
    return false;
  return ink2_mat2(f, L->ctrl_down[c], d->ctrl_down[c], f->cgate, NULL, NULL, NULL, f->ccat, n, r16) &&
         ink2_mat2(f, L->ctrl_gain[c], d->ctrl_gain[c], f->cgain, L->ctrl_forget[c], d->ctrl_forget[c], f->cforget,
              f->cgate, n, false) &&
         (!out || ink2_loop_ctrl(x, m, f->cgain, f->cforget, (uint64_t)n * C, out, f->st));
}

/* a decode's three controls (q, k, v: control's for each) in batches: their norms each control's (x16s bx16 2 .. 4),
   the three downs one mm_batch, their gates' f16 (bx16 5 .. 7), the gains and forgets one mm_batch, the three
   ink2_loop_ctrl - each value control's bits. false: not taken (n past MM_BATCH_TOKENS, a matrix not on mm) */
bool ink2_controls3(Ink2Forward *f, const Ink2Layer *L, const Ink2Dev *d, const float *const *x, const float *m,
                      uint32_t n, float *out, bool *ok)
{
  const uint32_t C = (uint32_t)f->w->geo.n_embd, G = (uint32_t)f->w->geo.loop.gate_rank;
  const float eps = f->w->geo.rms_eps;
  const uint64_t nC = (uint64_t)n * C;
  const GgufTensor *down[3] = {L->ctrl_down[INK2_CTRL_Q], L->ctrl_down[INK2_CTRL_K], L->ctrl_down[INK2_CTRL_V]};
  const GgufTensor *gf[6] = {L->ctrl_gain[INK2_CTRL_Q],   L->ctrl_forget[INK2_CTRL_Q], L->ctrl_gain[INK2_CTRL_K],
                             L->ctrl_forget[INK2_CTRL_K], L->ctrl_gain[INK2_CTRL_V],   L->ctrl_forget[INK2_CTRL_V]};
  if (n > MM_BATCH_TOKENS || !batchable(f, down, 3, 2 * C) || !batchable(f, gf, 6, G))
    return false;
  MmPart pd[3], pg[6];
  OpsRmsPart norms[6];
  const uint64_t n16 = (n + MM_TILE - 1) / MM_TILE * MM_TILE;
  for (int c = 0; c < 3; c++)
  {
    const int k = INK2_CTRL_Q + c;
    /* the gates side by side in a row (gate c at c G, rows 3 G apart): one mm_x16_heads their three f16s, gate c's at
       c n16 G halves of bx16 5's (each mm_x16's bits) */
    float *cat = f->ccat3 + (uint64_t)c * MM_BATCH_TOKENS * 2 * C, *gate = f->cgate3 + (uint64_t)c * G;
    void *x16c = bx16_at(f, 2 + c), *x16g = (uint8_t *)bx16_at(f, 5) + (uint64_t)c * n16 * G * 2;
    norms[2 * c] = (OpsRmsPart){x[c], NULL, cat, x16c, C, 2 * C, 2 * C, 0};
    norms[2 * c + 1] = (OpsRmsPart){m + c * nC, NULL, cat + C, x16c, C, 2 * C, 2 * C, C};
    pd[c] = (MmPart){d->ctrl_down[k], G, x16c, gate, 3 * G};
    float *gain = f->cgain3 + (uint64_t)c * MM_BATCH_TOKENS * C, *forget = f->cforget3 + (uint64_t)c * MM_BATCH_TOKENS * C;
    pg[2 * c] = (MmPart){d->ctrl_gain[k], C, x16g, gain, C};
    pg[2 * c + 1] = (MmPart){d->ctrl_forget[k], C, x16g, forget, C};
  }
  *ok = ops_rms_x16_many(norms, 6, C, eps, n, f->overflow, f->st) && mm_batch(ink2_type(down[0]), 2 * C, pd, 3, n, f->st);
  *ok = *ok && mm_x16_heads(f->cgate3, 3 * G, n, 3, G, G, bx16_at(f, 5), f->overflow, f->st);
  *ok = *ok && mm_batch(ink2_type(gf[0]), G, pg, 6, n, f->st);
  Ink2LoopCtrlPart lc[3];
  for (int c = 0; c < 3; c++)
    lc[c] = (Ink2LoopCtrlPart){x[c], m + c * nC, f->cgain3 + (uint64_t)c * MM_BATCH_TOKENS * C,
                               f->cforget3 + (uint64_t)c * MM_BATCH_TOKENS * C, out + c * nC};
  *ok = *ok && ink2_loop_ctrl_many(lc, 3, nC, f->st);
  return true;
}
