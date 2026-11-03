#ifndef DRAFTER_IMPL_H
#define DRAFTER_IMPL_H

#include "forward_impl.h"
#include "substrate/gpu/quant.h"
#include "substrate/model/dspark.h"
#include "substrate/run/drafter.h"

#define DRAFTER_BLOCKS 3          /* mtp.0 .. mtp.2 */
#define DRAFTER_EXPERTS 128       /* a block's routed experts */
#define DRAFTER_SWIGLU_LIMIT 10.0f /* DeepSeek's config.json swiglu_limit, the routed and shared experts' */
#define DRAFTER_PAIRS (DRAFTER_BLOCK * DRAFTER_TOPK)
/* the GGUF type ids of the drafter's weights (dspark.json's: its dense matrices Q8_0, gate and heads BF16, experts
   MXFP4, vectors F32 or BF16) */
#define DRAFTER_F32 0
#define DRAFTER_Q8_0 8
#define DRAFTER_BF16 30
#define DRAFTER_MXFP4 39

/* a block's weights in the drafter's GPU memory: the matrices (Q8_0, the gate BF16) where they lie; its F32 vectors
   where they lie; its BF16 vectors widened to F32; its experts' w1 (gate), w3 (up), w2 (down) each a table of
   DRAFTER_EXPERTS device pointers on the device (the router's picks gather a round's from it) */
typedef struct DBlock
{
  const void *wq_a, *wq_b, *wkv, *wo_b, *sh1, *sh3, *sh2, *gate;
  const float *sink, *gate_bias, *hca_fn, *hca_base, *hca_scale, *hcf_fn, *hcf_base, *hcf_scale;
  float *q_norm, *kv_norm, *attn_norm, *ffn_norm;
  const void **ex[3];    /* [w1, w3, w2][expert] (device) */
  const void **oa_tab;   /* wo_a's 8 groups, [row][group], DRAFTER_BLOCK rows (device): mv_group's */
} DBlock;

/* host memory the GPU copies to and from (pinned) */
typedef struct DPin
{
  int32_t ids[DRAFTER_BLOCK], pos_q[DRAFTER_BLOCK], pos_a[DRAFTER_BLOCK];
  int32_t tap_pos[FWD_MAX_BATCH];
  float conf;
  float row[]; /* a row of the vocabulary's logits */
} DPin;

struct Drafter
{
  Forward *f;
  DsparkIndex *ix;
  int fd;
  Resident *res;
  DBlock b[DRAFTER_BLOCKS];
  const void *main_proj, *mk_embed, *mk_head;
  uint32_t main_type, rank;
  float *main_norm, *norm, *conf_w; /* widened */
  /* a round's device scratch, DRAFTER_BLOCK rows */
  float *h, *mid, *x, *qa, *q, *kv, *o, *oa, *y, *xf, *logits, *rw, *ga, *up, *hh, *ye, *moe, *sga, *sup, *shh,
      *sye, *hcp, *a_pre, *a_post, *a_comb, *f_pre, *f_post, *f_comb, *pre0, *att, *xc, *xn, *vocab, *mk, *cat,
      *cval, *main_x, *kvs;
  uint8_t *kvs_p; /* those rows packed as the ring keeps them (kvfmt.h's KV_FP8) */
  int32_t *ridx, *sel, *n_sel, *pos_q, *pos_a, *ids, *pair, *tap_pos;
  const void **tab; /* [w1, w3, w2][pair]: a round's picks' pointers (gathered on the device) */
  DPin *pin;
};

/* hip_drafter_round.c: a round's blocks on the d->ids rows at p0, into d->xc (the collapse) */
bool drafter_blocks(Drafter *d, Seq *s, int32_t p0, char *err);

#endif
