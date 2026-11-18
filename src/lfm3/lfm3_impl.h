#ifndef OSMOS_LFM3_IMPL_H
#define OSMOS_LFM3_IMPL_H

/* the lfm3 forward's state (osmos/lfm3/forward.h), shared by hip_lfm3_make.c (the weights, buffers, sequences) and
   hip_lfm3.c (the step) */

#include "osmos/gpu/q4x_ops.h"
#include "osmos/lfm3/forward.h"

#define F 4ull               /* a float's bytes */
#define LFM3_CONV_SLOTS 1    /* a convolution's states a sequence keeps (step 1: the last; rollback's step 2's) */

struct Lfm3Forward
{
  const Lfm3Weights *w;
  void **dev; /* each tensor's device copy, by the model's index */
  uint32_t max_tokens, max_ctx, logit_rows;
  Q4xRope rope_full, rope_swa;
  void *st;
  /* a step's buffers: max_tokens rows each */
  int32_t *tok, *pin_tok;
  float *h, *xn, *y, *bcx, *u, *qf, *kf, *vf, *qr, *kr, *g, *att, *ga, *up, *act, *attn_scratch;
  float *xf, *logits, *pin_logits; /* the kept rows' final norm and logits */
  lfm3_tap_fn tap;
  void *tap_arg;
};

struct Lfm3Seq
{
  Lfm3Forward *f;
  int32_t pos;
  float *conv[LFM3_MAX_LAYERS];         /* a convolution layer's state ring (LFM3_CONV_SLOTS x (taps - 1) x D) */
  void *kc[LFM3_MAX_LAYERS], *vc[LFM3_MAX_LAYERS]; /* an attention layer's Q8_0 cache: a row a position */
};

bool lfm3_fail(char *err, const char *why);
bool lfm3_alloc(void *p, uint64_t bytes);
static inline const void *lfm3_dev(const Lfm3Forward *f, const GgufTensor *t)
{
  return f->dev[t - f->w->model->tensors];
}
static inline const float *lfm3_fv(const Lfm3Forward *f, const GgufTensor *t) { return (const float *)lfm3_dev(f, t); }

#endif
