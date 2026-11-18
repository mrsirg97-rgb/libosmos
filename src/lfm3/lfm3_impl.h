#ifndef OSMOS_LFM3_IMPL_H
#define OSMOS_LFM3_IMPL_H

/* the lfm3 forward's state (osmos/lfm3/forward.h), shared by hip_lfm3_make.c (the weights, buffers, sequences) and
   hip_lfm3.c (the step) */

#include "osmos/gpu/q4x_ops.h"
#include "osmos/lfm3/forward.h"

#define F 4ull               /* a float's bytes */
#define LFM3_CONV_SLOTS (LFM3_MAX_BACK + 1) /* a convolution's states a sequence keeps: the current, and a rollback's */

struct Lfm3Forward
{
  const Lfm3Weights *w;
  void **dev; /* each tensor's device copy, by the model's index */
  uint32_t max_tokens, max_ctx, logit_rows;
  uint32_t ring; /* a sliding layer's cache rows: window + max_tokens + LFM3_MAX_BACK (DESIGN.md, step 2) */
  Q4xRope rope_full, rope_swa;
  const void *head;   /* the tied head as the step reads it: head_q8 (LFM3_HEAD_Q8_0) or the embedding's table */
  uint32_t head_type;
  void *head_q8;      /* its Q8_0 copy (owned), NULL for the file's */
  /* a one-token step's joints: an attention layer's q, k, v and gate one matrix, every layer's FFN gate and up one -
     their rows concatenated at load (a row's sums its own: the same bits), each tensor's device pointer into it */
  const void *j_attn[LFM3_MAX_LAYERS], *j_ffn[LFM3_MAX_LAYERS]; /* NULL: not joined (types unlike) */
  void *joints[2 * LFM3_MAX_LAYERS];
  uint32_t n_joints;
  bool *in_joint;     /* a tensor's bytes inside a joint (not freed alone) */
  float *jo;          /* a joint's outputs (a token's) */
  void *x16;          /* a read's matrices' f16 input (MM_MAX_TOKENS rows of the widest) */
  int32_t *overflow;  /* an activation past f16's range in a read (mm_x16's): the step refused */
  void *st;
  /* a step's buffers: max_tokens rows each */
  int32_t *tok, *pin_tok;
  float *h, *xn, *y, *bcx, *u, *qf, *kf, *vf, *qr, *kr, *g, *att, *ga, *up, *act, *attn_scratch;
  float *xf, *logits, *pin_logits; /* the kept rows' final norm and logits */
  float *hk;                        /* the kept rows' residuals, gathered */
  int32_t *slot;                    /* each row of the last step its logits' row, -1 not kept (host) */
  uint32_t rows;                    /* the last step's rows (0 after a failed one) */
  int32_t *ids;                     /* lfm3_greedy's: rows' picks (device) */
  void *argmax;                     /* ops_argmax's scratch */
  lfm3_tap_fn tap;
  void *tap_arg;
};

struct Lfm3Seq
{
  Lfm3Forward *f;
  int32_t pos;
  uint32_t back, prompt_end, cur; /* the reach; the prompt's end (0: none); the convolutions' current slot */
  bool broken;
  float *conv[LFM3_MAX_LAYERS];         /* a convolution layer's state ring (LFM3_CONV_SLOTS x (taps - 1) x D) */
  void *kc[LFM3_MAX_LAYERS], *vc[LFM3_MAX_LAYERS]; /* an attention layer's Q8_0 cache: a full layer's a row a
                                                       position, a sliding layer's a ring of f->ring rows */
};

/* a step's runs: each sequence's tokens at rows off .. off + len - 1 of the step */
typedef struct Lfm3Run
{
  Lfm3Seq *s;
  uint32_t off, len;
} Lfm3Run;

bool lfm3_fail(char *err, const char *why);
bool lfm3_alloc(void *p, uint64_t bytes);
static inline const void *lfm3_dev(const Lfm3Forward *f, const GgufTensor *t)
{
  return f->dev[t - f->w->model->tensors];
}
static inline const float *lfm3_fv(const Lfm3Forward *f, const GgufTensor *t) { return (const float *)lfm3_dev(f, t); }

#endif
