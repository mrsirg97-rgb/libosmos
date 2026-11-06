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
  float *qkv, *z, *al, *be, *g, *b, *cv, *qkn, *ro, *fo;              /* a DeltaNet's */
  float *qf, *kr, *kq, *vr, *q, *gt, *ao, *fa; /* an attention's (kq, vr: k and v to the cache; fa: its chunks') */
  float *rl, *w, *ga, *up, *y, *shg, *shu, *sh, *sl;                  /* the MoE's */
  int32_t *ids, *rows, *cnt, *gidx, *hidx, *where; /* the picks, mm_group's tables (q4x_expert_tiles) */
  uint32_t *xg, *yg, *xd, *yd;
  const void **ptr;                                /* Q4X_GROUP_TENSORS tables of entries */
  void *x16, *xe16;                                /* mm's f16 activations: a matrix's, the experts' tiles */
  int32_t *overflow;                               /* an activation past f16's: the step refused */
  float *emb, *key, *val, *kn, *qn, *pg, *gated, *nrm, *conv;         /* the n-gram layer's */
  int32_t *ple_rows, *tok;                                            /* device: its rows, the step's tokens */
  int32_t *pin_rows, *pin_tok, *pin_overflow;                         /* their pinned host copies */
  float *logits;
} Q4xBuf;

/* the MTP head's buffers (with a head): a step's hidden rows, a draft's inputs and logits */
typedef struct Q4xMtpBuf
{
  float *hid;   /* [max_tokens][hc n_embd]: each step's final residual (the head's h) at its rows */
  float *hin;   /* [max_tokens][hc n_embd]: the head's h rows (a position's before it) */
  float *emb;   /* [max_tokens][n_embd]: the head's tokens' embedding, then normed */
  float *hn;    /* [max_tokens][hc n_embd]: h normed */
  float *join;  /* [max_tokens][hc][2 n_embd]: eh_proj's input */
  float *logits;
  int32_t *top_id;
  float *top_p;
  int32_t *pin_id; /* pinned: a draft's token and its probability */
  float *pin_p;
} Q4xMtpBuf;

struct Q4xForward
{
  const Q4xWeights *w;
  const Q4xMtp *mtp; /* or NULL */
  void **dev, **mtp_dev; /* a tensor's bytes on the device, by its index in its model */
  uint32_t max_ctx, max_tokens;
  Q4xBuf b;
  Q4xMtpBuf m;
  q4x_probe_fn probe;
  void *probe_arg;
};

#define Q4X_SLOTS (Q4X_MAX_BACK + 1) /* a ring's states: the latest and those behind it */

struct Q4xSeq
{
  Q4xForward *f;
  int32_t pos;
  float *ssm[Q4X_MAX_LAYERS], *conv[Q4X_MAX_LAYERS]; /* a DeltaNet layer's S and conv history, Q4X_SLOTS each */
  uint8_t *kc[Q4X_MAX_LAYERS], *vc[Q4X_MAX_LAYERS];  /* an attention layer's cache (Q8_0 rows: q4x_kv_put) */
  float *ple_hist;                                   /* the n-gram layer's conv history, Q4X_SLOTS of them */
  uint32_t cur, back;                                /* the rings' latest slot, and the states kept behind it */
  int32_t last[Q4X_MAX_NGRAM + Q4X_MAX_BACK];        /* its last tokens, the latest first (a rollback's too) */
  int n_last;
  /* with an MTP head: its cache, the main model's h of position pend_pos (-1: none, zeros), its last step's place */
  uint8_t *mkc, *mvc;
  float *pend;
  int32_t pend_pos, step_pos0;
  uint32_t step_row0, step_n;
  int32_t *step_tok; /* [max_tokens], host */
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
/* y = W x for n tokens (rows xs, ys apart): mm (f16 activations, WMMA) for a matrix it tiles, else the matvec - by the
   matrix alone, so a token's bits are its own whatever its batch (hip_layers.c). `ready`: b.x16 is this x's already
   (the call before made it); q4x_mat2: two matrices on one x, one launch when mm takes both (mm_run2: each the bits it
   takes alone) */
bool q4x_mat(Q4xForward *f, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t xs,
             uint32_t n, float *y, uint32_t ys, bool ready);
bool q4x_mat2(Q4xForward *f, const GgufTensor *t0, uint32_t rows0, float *y0, uint32_t ys0, const GgufTensor *t1,
              uint32_t rows1, float *y1, uint32_t ys1, uint32_t cols, const float *x, uint32_t xs, uint32_t n,
              bool ready);
/* the step's mm activations all within f16: the flag queued to its pinned copy (q4x_overflow_read, before the sync a
   step makes anyway - no round trip of its own), then read after it (q4x_overflow_ok: false with why, the flag
   cleared, when set). the flag accumulates: a call that queues no read (the MTP head's seeding) is caught at the next */
bool q4x_overflow_read(Q4xForward *f);
bool q4x_overflow_ok(Q4xForward *f, char *err);
const void *q4x_dev(const Q4xForward *f, const GgufTensor *t);
void q4x_seen(const Q4xForward *f, const char *step, int layer, float *dev, uint64_t n);

/* hip_layers.c: a layer's blocks on n tokens of b.h (the residual), its output into b.h */
bool q4x_mix(Q4xForward *f, const GgufTensor *norm, const GgufTensor *down, const GgufTensor *up,
             const GgufTensor *inject, uint32_t n);
bool q4x_deltanet(Q4xForward *f, Q4xSeq *s, int l, uint32_t n);
bool q4x_attention(Q4xForward *f, const Q4xLayer *L, uint8_t *kc, uint8_t *vc, int32_t pos, uint32_t n);
bool q4x_attention_kv(Q4xForward *f, const Q4xLayer *L, uint8_t *kc, uint8_t *vc, int32_t pos, uint32_t n);
bool q4x_moe(Q4xForward *f, const Q4xLayer *L, uint32_t n);
bool q4x_ple(Q4xForward *f, Q4xSeq *s, int l, uint32_t n);
