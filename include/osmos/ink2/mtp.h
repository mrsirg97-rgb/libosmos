#ifndef OSMOS_INK2_MTP_H
#define OSMOS_INK2_MTP_H

#include "osmos/err.h"
#include <stdbool.h>
#include <stdint.h>

#include "osmos/ink2/weights.h"
#include "osmos/model/gguf.h"

#define INK2_MTP_MAX_LAYERS 8 /* the drafter's layers at most (Inkling-2's 5) */
#define INK2_MTP_MAX_TAPS 8   /* the target's taps at most (Inkling-2's 8: seven layers' entries, the final hidden) */
#define INK2_MTP_HEAD 128     /* its heads' dim: the kernels' */
#define INK2_MTP_BLOCK 8      /* a round's block at most: the anchor and its drafts */
#define INK2_MTP_RANK 256     /* the Markov head's rank: the kernels' */
#define INK2_MTP_ATTN_ROWS 48 /* a block's queries times the heads a KV head serves, at most: the attention's (8 x 6) */

typedef struct Ink2MtpLayer
{
  const GgufTensor *in_norm, *post_norm, *dflare_norm, *q_norm, *k_norm, *k_norm_ctx;
  const GgufTensor *q, *k, *v, *k_ctx, *v_ctx, *o, *gate, *up, *down;
} Ink2MtpLayer;

/* Inkling-2's drafter (tools/convert/ink2_mtp_gguf.py's GGUF): DSpark with the RNN Markov head - the target's taps,
   embedding and head borrowed */
typedef struct Ink2Mtp
{
  const Gguf *model;
  int n_layers, n_embd, n_heads, n_kv_heads, ff, window, block, rank, n_taps;
  float eps, rope_base;
  bool local[INK2_MTP_MAX_LAYERS];   /* a layer's attention over its window (else every position) */
  int32_t targets[INK2_MTP_MAX_TAPS]; /* the target's slots each tap is the entry of, in the order they run (a loop's
                                         passes unrolled); the last is n_slots: the final hidden (the head's input) */
  Ink2MtpLayer layer[INK2_MTP_MAX_LAYERS];
  const GgufTensor *fuse, *mask, *norm, *m_dflare, *m_hidden_w, *m_hidden_b, *m_w1, *m_rnn_w, *m_rnn_b;
} Ink2Mtp;

Ink2Mtp *ink2_mtp_create(const Gguf *g, const Ink2Weights *target, char err[OSMOS_ERR]);
void ink2_mtp_destroy(Ink2Mtp *m);

#endif

/*
ink2_mtp_create
  the drafter in g for the target, its geometry from inkling2_mtp.*: its width the target's, its taps the target's
  slots ascending and the last n_slots, heads of INK2_MTP_HEAD, a block of at most INK2_MTP_BLOCK, rank INK2_MTP_RANK;
  every tensor's dims and type checked (a matrix the matvec's, markov_w1 BF16 rows, the rest F32). NULL with why,
  naming the key or tensor. the drafter points into g: g outlives it.
*/
