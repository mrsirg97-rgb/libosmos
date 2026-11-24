#ifndef OSMOS_Q4X_DFLASH_CORE_H
#define OSMOS_Q4X_DFLASH_CORE_H

/* DFlash2's drafter for any target (hip_dflash_core.c: z-lab's DFlash2DraftModel on the q4x_df_* kernels): its
   buffers, its context's seed and its block's draft - the math one, the target's own calls (its matrices, its
   embedding and head, its stream, its checks) through a DfHost: q4x's (src/q4x/hip_q4x_draft_dflash.c), LFM3's
   (src/lfm3/hip_lfm3_draft.c) */

#include <stdbool.h>
#include <stdint.h>

#include "osmos/gpu/bufs.h"
#include "osmos/mem/ledger_hip.h"
#include "osmos/q4x/dflash.h"

/* the drafter's buffers: a seed's max_tokens context rows, a block's, the selector's, pinned copies out - each made
   through one ledger and freed by name, never as an array of the struct's first pointers */
typedef struct DfBuf
{
  float *th;            /* [max_tokens][n_embd]: the context's taps through fc, then its hidden norm */
  float *k, *v;         /* [max(max_tokens, block)][kv]: a layer's keys and values (the context's, the block's) */
  float *h, *x, *xc, *dyn, *q, *att, *a, *gate, *up, *hn; /* the block's: residual, normed, convolved, .. its out */
  float *logits, *hp, *val; /* the head's at the block's drafts' places, the selector's projection, the top-k */
  float *top_p;             /* each place's greatest token's probability (q4x_top: a draft's confidence's) */
  int32_t *idx, *path, *top_id;
  int32_t *tok, *iota; /* device: the block's tokens (y, then masks); 0 .. max_ctx + block (a run's positions) */
  int32_t *pin_tok, *pin_path, *pin_idx; /* pinned: the block's anchor in; its path, the top-k's ids, values, p out */
  float *pin_val, *pin_p;
  GpuBufs bufs;        /* the device buffers' record (osmos/gpu/bufs.h: its ledger the pinned ones' too) */
} DfBuf;

/* a target's calls the drafter's math goes through - each on its own stream, its own matrices' path */
typedef struct DfHost
{
  void *ctx;
  void *stream;
  const void *(*dev)(void *ctx, const GgufTensor *t); /* a drafter tensor's bytes on the device */
  bool (*mat)(void *ctx, const GgufTensor *t, uint32_t rows, uint32_t cols, const float *x, uint32_t xs, uint32_t n,
              float *y, uint32_t ys);
  /* two matrices on one x (a fused call where the target has one), or NULL: two mats */
  bool (*mat2)(void *ctx, const GgufTensor *t0, uint32_t rows0, float *y0, uint32_t ys0, const GgufTensor *t1,
               uint32_t rows1, float *y1, uint32_t ys1, uint32_t cols, const float *x, uint32_t xs, uint32_t n);
  bool (*embed)(void *ctx, const int32_t *tok, uint32_t n, float *h);                 /* the target's embedding */
  bool (*head)(void *ctx, const float *x, uint32_t xs, uint32_t n, float *logits);    /* the target's head */
  void (*seen)(void *ctx, const float *hn, uint64_t n);   /* the block's normed output (a probe), or NULL */
  bool (*before_sync)(void *ctx);                         /* queued before the draft's sync, or NULL */
  bool (*after_sync)(void *ctx, char *err);               /* checked after it (an overflow's), or NULL */
} DfHost;

/* a sequence's drafter rings: each layer's keys and values (ring rows, position p at p % ring) */
typedef struct DfRings
{
  float *const *k, *const *v;
  uint32_t ring;
} DfRings;

bool df_buffers(DfBuf *d, OsmosLedger *l, const Q4xDflashGeometry *g, uint32_t max_tokens, uint32_t max_ctx);
void df_buffers_free(DfBuf *d);
bool df_seed(const DfHost *h, const Q4xDflash *df, DfBuf *d, const DfRings *r, const float *taps, int32_t from,
             uint32_t n);
int df_draft(const DfHost *h, const Q4xDflash *df, DfBuf *d, const DfRings *r, int32_t y, int32_t pos0,
             int32_t ctx_from, uint32_t n, float conf_min, float cum_min, int32_t *drafts, float *conf, char *err);

#endif

/*
df_buffers, df_buffers_free
  every buffer for a drafter of geometry g on a forward of max_tokens a step and max_ctx positions (the block's
  tokens after its anchor the mask, iota 0 .. max_ctx + block on the device); false where the GPU refused one (the
  caller's message). free: each NULL or freed, the struct then zeros.

df_seed
  the context at positions from .. from + n - 1 from their taps (n rows of n_taps x n_embd): fc and the hidden norm,
  then each layer's k (normed, rotated) and v into the rings - the GPU's part of q4x_dflash_seed's contract, its
  caller's checks and context bounds its own. false for a GPU call failed.

df_draft
  a block at pos0 - y and the mask after it - through the drafter's layers (attention over the rings from ctx_from
  within its window, and the block's own), the target's head at the block's places after y, the selector's greedy
  path from y: up to n drafts (n at most block - 1, the caller's caps applied), each kept while its confidence is at
  least conf_min and their product at least cum_min - the count, or -1 with why.
*/
