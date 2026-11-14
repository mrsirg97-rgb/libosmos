/* Ink2Forward's and Ink2Seq's insides (osmos/ink2/forward.h): src/ink2/hip_forward.c the weights, the membrane and
   the step; src/ink2/hip_seq.c a sequence's tails, its saves and marks */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "osmos/ink2/forward.h"
#include "osmos/mem/host_slots.h"
#include "osmos/mem/membrane.h"

/* a layer's resident tensors on the device: its vectors as F32 (the norms, clips, convs, queries, the router's
   bias), its matrices as stored (mv_run reads their type) */
typedef struct Ink2Dev
{
  float *attn_res_q, *ffn_res_q, *attn_norm, *q_norm, *k_norm, *v_norm, *ffn_norm;
  void *q, *k, *v, *rel, *out, *gate, *sink, *router, *k_conv_proj, *v_conv_proj;
  float *q_clip, *rel_clip, *rel_proj, *router_bias;
  float *k_conv, *v_conv, *out_conv, *ffn_out_conv; /* the static convs: a full layer's k / v [taps][ch], tap 0 first */
  float *scale[3]; /* host: each expert's gate, up, down scale (1 where the file has none) */
} Ink2Dev;

/* a sequence's rows of one kind, oldest first: `held` of them at d, room for cap; the last `keep` (what a later
   token reads, and a rollback's) moved to the front when a step would pass cap (cap 2 keep + a step's: the move's
   source and target apart) */
typedef struct Ink2Tail
{
  float *d;
  uint32_t row; /* floats a row */
  uint32_t keep, cap, held;
} Ink2Tail;

enum
{
  INK2_CONV_K,
  INK2_CONV_V,
  INK2_CONV_OUT,
  INK2_CONV_FFN,
  INK2_CONVS
};

struct Ink2Seq
{
  Ink2Forward *f;
  int32_t pos;
  uint32_t reach, prompt_end;
  bool failed;
  Ink2Tail conv[INK2_MAX_LAYERS][INK2_CONVS]; /* each layer's short convs' inputs */
  Ink2Tail kw[INK2_MAX_LAYERS], vw[INK2_MAX_LAYERS]; /* a sliding layer's window (a full layer's d: max_ctx rows) */
};

struct Ink2Forward
{
  const Ink2Weights *w;
  uint32_t max_ctx, max_tokens, maxb, vocab;
  Ink2Dev L[INK2_MAX_LAYERS];
  void *embed, *output;
  float *embed_norm, *out_res_q, *out_norm;
  uint64_t dense_bytes, pinned_bytes;
  /* the experts */
  int *fds;
  HostSlots *slots, *stream;
  Membrane *mem;
  /* the step's buffers (max_tokens rows) */
  int32_t *ids_d;
  float *h, *blocks, *a3, *xn, *kn, *vn, *q, *k, *v, *kc, *vc, *rel, *gate, *sink, *att, *dyn, *o, *r, *moe, *logit,
      *tmp, *pw, *eg, *eu, *eh, *ed, *head, *logits;
  int32_t *pick, *hpick, *gid;
  float *hw, *xs; /* xs: the pairs' inputs in expert order */
  /* moe's tables: the counting sort's, the region built on the host and its device copy */
  uint32_t *m_at, *m_by;
  uint8_t *m_host;
  void *m_dev;
  void *amax;
  bool *kept;
  uint32_t n; /* the last step's rows */
};

bool ink2_fail(char *err, const char *why);
/* the tails: room for n more rows (the move), the last k rows, n rows appended (queued on the null stream) */
bool ink2_tail_room(Ink2Tail *t, uint32_t n);
const float *ink2_tail_last(const Ink2Tail *t, uint32_t k);
bool ink2_tail_put(Ink2Tail *t, const float *x, uint32_t n);
