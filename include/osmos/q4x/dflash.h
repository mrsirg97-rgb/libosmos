#ifndef OSMOS_Q4X_DFLASH_H
#define OSMOS_Q4X_DFLASH_H

#include <stdint.h>

#include "osmos/model/gguf.h"
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"

/* the drafter's kernels' fixed shapes: a file past them refused at load, by name (as geo.c's Q4X_* for the model) */
#define Q4X_DFLASH_MAX_LAYERS 8    /* its layers at most (DFlash2's 5) */
#define Q4X_DFLASH_HEAD_DIM 128    /* its attention's head dim: the kernel's */
#define Q4X_DFLASH_CONV 2          /* its dynamic convolutions' taps: the kernel's (the token and the one before) */
#define Q4X_DFLASH_BLOCK_MAX 8     /* a block's tokens at most: the anchor and its drafts (DFlash2's 8) */
#define Q4X_DFLASH_TOPK_MAX 16     /* the selector's candidates a position at most (DFlash2's 16) */
#define Q4X_DFLASH_WINDOW_MAX 8192 /* its sliding window at most: a sequence's context ring (DFlash2's 2,048) */

typedef struct Q4xDflashGeometry
{
  int n_layers, n_embd, n_heads, n_kv_heads, head_dim, ff, n_vocab;
  int window;      /* the attention's sliding window, both ways (its layers are not causal) */
  int block;       /* a round's block: the anchor and block - 1 drafts */
  int mask_token;  /* the drafts' input token */
  int n_taps;      /* the target's layers its features are (Q4xOptions' tap_layers) */
  int32_t taps[Q4X_MAX_TAPS];
  int conv_group;  /* the dynamic convolutions' channels a group (a group's own kernel per token) */
  int sel_rank, sel_top_k;
  float eps, rope_base;
} Q4xDflashGeometry;

typedef struct Q4xDflashLayer
{
  const GgufTensor *attn_norm, *q, *k, *v, *out, *q_norm, *k_norm, *ffn_norm, *gate, *up, *down;
  const GgufTensor *attn_conv, *attn_conv_proj, *ffn_conv, *ffn_conv_proj;
} Q4xDflashLayer;

/* DFlash2 (incoai's Qwen3.8-Flash-Next-DFlash2, tools/convert/dflash2_gguf.py's GGUF): a block-diffusion drafter of
   the target's own tokens - the target's embedding and output borrowed - fed the target's taps */
typedef struct Q4xDflash
{
  const Gguf *model;
  Q4xDflashGeometry geo;
  const GgufTensor *fc, *hidden_norm, *output_norm, *sel_proj, *sel_pred, *sel_succ;
  Q4xDflashLayer layer[Q4X_DFLASH_MAX_LAYERS];
} Q4xDflash;

Q4xDflash *q4x_dflash_create(const Gguf *g, const Q4xWeights *target, char err[Q4X_ERR]);
void q4x_dflash_destroy(Q4xDflash *d);

#endif

/*
q4x_dflash_create
  the drafter in g for the target w: its geometry from its metadata (dflash2.*) - its width and vocabulary w's, its
  target's layer count w's, its taps w's layers ascending (Q4X_MAX_TAPS at most), its mask token in the vocabulary -
  within its kernels' (Q4X_DFLASH_*: the head dim, the convolutions' taps, a block, the selector's candidates, the
  window; the conv group dividing the width, the heads a multiple of the KV heads), then every tensor given its role,
  checked as q4x_weights_create checks the model's: its name, dims, a type its reader takes (a matrix the matvec's
  and the matmul's, the codebooks a row gather's, the norms and the conv kernels F32), its bytes; a tensor no role
  takes refused. NULL with why, the key or the tensor named. the drafter points into g's tensors: g outlives it.

q4x_dflash_destroy
  frees it. NULL is nothing to do.
*/
