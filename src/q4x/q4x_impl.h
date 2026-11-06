/* the Qwen forward's shapes (osmos/q4x/forward.h): its weights on the device, a step's buffers, a sequence's state.
   hip_forward.c builds them, hip_layers.c runs a layer's blocks on them, hip_step.c a step */
#pragma once

#include <hip/hip_runtime_api.h>

#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/q4x/forward.h"

/* a step's buffers, for max_tokens tokens */
typedef struct Q4xBuf
{
  float *h, *xn, *lo, *gate, *mixed, *inj, *out, *x;                  /* the residual, a mixer's, a block's output */
  float *qkv, *z, *al, *be, *g, *b, *cv, *ro, *fo;                    /* a DeltaNet's */
  float *qf, *kr, *q, *gt, *ao;                                       /* an attention's */
  float *rl, *w, *xe, *ga, *up, *y, *shg, *shu, *sh, *sl;             /* the MoE's */
  int32_t *ids, *rows, *cnt, *row;
  const void **ptr;
  float *emb, *key, *val, *kn, *qn, *pg, *gated, *nrm, *conv;         /* the n-gram layer's */
  int32_t *ple_rows, *tok;                                            /* device: its rows, the step's tokens */
  int32_t *pin_rows, *pin_tok;                                        /* their pinned host copies */
  float *logits;
} Q4xBuf;

struct Q4xForward
{
  const Q4xWeights *w;
  void **dev; /* a tensor's bytes on the device, by its index in the model */
  uint32_t max_ctx, max_tokens;
  Q4xBuf b;
  q4x_probe_fn probe;
  void *probe_arg;
};

struct Q4xSeq
{
  Q4xForward *f;
  int32_t pos;
  float *ssm[Q4X_MAX_LAYERS], *conv[Q4X_MAX_LAYERS]; /* a DeltaNet layer's S and conv history */
  float *kc[Q4X_MAX_LAYERS], *vc[Q4X_MAX_LAYERS];    /* an attention layer's cache */
  float *ple_hist;                                   /* the n-gram layer's conv history */
  int32_t last[Q4X_MAX_NGRAM];                       /* its last tokens, the latest first */
  int n_last;
};

/* a sequence's buffers' bytes: a DeltaNet layer's state and conv history, an attention cache's row, the n-gram
   layer's conv history */
typedef struct Q4xSeqBytes
{
  uint64_t ssm, conv, kv_row, ple;
} Q4xSeqBytes;

Q4xSeqBytes q4x_seq_bytes(const Q4xGeometry *g);
bool q4x_seq_reset(Q4xSeq *s); /* hip_state.c: empty, position 0 (its recurrent states and histories zeroed) */

bool q4x_fail(char *err, const char *why);
const void *q4x_dev(const Q4xForward *f, const GgufTensor *t);
void q4x_seen(const Q4xForward *f, const char *step, int layer, float *dev, uint64_t n);

/* hip_layers.c: a layer's blocks on n tokens of b.h (the residual), its output into b.h */
bool q4x_mix(Q4xForward *f, const GgufTensor *norm, const GgufTensor *down, const GgufTensor *up,
             const GgufTensor *inject, uint32_t n);
bool q4x_deltanet(Q4xForward *f, Q4xSeq *s, int l, uint32_t n);
bool q4x_attention(Q4xForward *f, Q4xSeq *s, int l, uint32_t n);
bool q4x_moe(Q4xForward *f, int l, uint32_t n);
bool q4x_ple(Q4xForward *f, Q4xSeq *s, int l, uint32_t n);
