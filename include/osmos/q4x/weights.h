#ifndef OSMOS_Q4X_WEIGHTS_H
#define OSMOS_Q4X_WEIGHTS_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/mem/pool.h"
#include "osmos/model/gguf.h"

#define Q4X_ERR 256
#define Q4X_MAX_LAYERS 64
#define Q4X_MAX_HC 8
#define Q4X_MAX_NGRAM 4     /* the n-gram table's longest n-gram */
#define Q4X_MAX_PLE_HEADS 16
#define Q4X_ROPE_SECTIONS 4

/* Qwen3.8-Flash-Next's shape (general.architecture qwen4exp), from its metadata */
typedef struct Q4xGeometry
{
  int n_layers, n_embd, n_vocab;
  uint32_t native_ctx;
  float rms_eps;
  /* the hyper-connections: hc streams of n_embd, mixed through a rank of hc_rank */
  int hc, hc_rank;
  /* a layer is a Gated DeltaNet (recurrent) or Qwen's sparse attention, every attn_interval-th */
  bool recurrent[Q4X_MAX_LAYERS];
  int attn_interval;
  /* the sparse attention: heads, KV heads, a head's dim; the rope's dims and sections; its indexer */
  int n_heads, n_kv_heads, head_dim, rope_dim;
  int rope_sections[Q4X_ROPE_SECTIONS];
  float rope_base;
  int ix_heads, ix_dim, ix_top_k, ix_ratio; /* ix_ratio: the keys' pooling block (compress_ratios', one for all) */
  /* the Gated DeltaNet: its conv's taps, a head's state (k and v dims), key heads (groups), value heads, inner */
  int ssm_conv, ssm_state, ssm_groups, ssm_v_heads, ssm_inner;
  /* the MoE: experts, picked a token, an expert's width, the shared expert's */
  int n_experts, n_used, ff_exp, ff_shexp;
  /* the n-gram table (PLE) on one layer: n-grams up to ple_ngram, heads_per_ngram heads each, a head's dim, its
     conv's taps, the hash's multipliers, each head's rows in the table */
  int ple_layer, ple_ngram, ple_heads_per_ngram, ple_heads, ple_dim, ple_conv;
  int32_t ple_eos, ple_image;
  int64_t ple_mult[Q4X_MAX_NGRAM];
  uint64_t ple_offset[Q4X_MAX_PLE_HEADS], ple_rows[Q4X_MAX_PLE_HEADS];
} Q4xGeometry;

/* a layer's tensors; NULL where its kind has none (a DeltaNet's attention, an attention's DeltaNet, PLE elsewhere) */
typedef struct Q4xLayer
{
  /* the hyper-connections before the mixer and before the MoE: a norm a stream, the rank's down and up, the inject */
  const GgufTensor *hc_attn_norm, *hc_attn_down, *hc_attn_up, *hc_attn_inject;
  const GgufTensor *hc_ffn_norm, *hc_ffn_down, *hc_ffn_up, *hc_ffn_inject;
  /* the sparse attention */
  const GgufTensor *q, *k, *v, *out, *q_norm, *k_norm;
  const GgufTensor *ix_q, *ix_k, *ix_q_norm, *ix_k_norm;
  /* the Gated DeltaNet */
  const GgufTensor *qkv, *gate, *conv, *dt, *a, *alpha, *beta, *ssm_norm, *ssm_out;
  /* the n-gram table's layer */
  const GgufTensor *ple_key, *ple_value, *ple_norm_key, *ple_norm_query, *ple_norm_conv, *ple_conv;
  /* the MoE */
  const GgufTensor *router, *exp_gate, *exp_up, *exp_down;
  const GgufTensor *sh_gate_inp, *sh_gate, *sh_up, *sh_down;
} Q4xLayer;

typedef struct Q4xWeights
{
  Q4xGeometry geo;
  const Gguf *model;
  const Gguf *overlay; /* tensors standing in for the model's of their names (an abliteration's), or NULL */
  const GgufTensor *embed, *output, *out_hc_norm, *out_hc_down, *out_hc_up, *ple_table;
  Q4xLayer layer[Q4X_MAX_LAYERS];
} Q4xWeights;

/* the MTP draft head (a file of its own: unsloth's mtp-*-shared-*.gguf): one more attention layer, blk.<n_layers>,
   and the nextn glue; the token embedding and the output the main model's */
typedef struct Q4xMtp
{
  const Gguf *model;
  Q4xLayer layer;
  const GgufTensor *eh_proj, *enorm, *hnorm, *head_norm, *head_down, *head_up;
} Q4xMtp;

Q4xWeights *q4x_weights_create(const Gguf *model, const Gguf *overlay, char err[Q4X_ERR]);
void q4x_weights_destroy(Q4xWeights *w);
Q4xMtp *q4x_mtp_create(const Gguf *mtp, const Q4xWeights *w, char err[Q4X_ERR]);
void q4x_mtp_destroy(Q4xMtp *m);
size_t q4x_weights_experts(const Q4xWeights *w, const int *model_fds, PoolUnit *units, size_t cap);

#endif

/*
q4x_weights_experts (DESIGN.md, "0.6.4: Qwen on the membrane")
  the routed experts as the pool's units, unit = layer * n_experts + expert, each three parts - the expert's slice of
  the layer's exp_gate, exp_up and exp_down (gate, up, down), a tensor's bytes / n_experts each - in the model's files,
  `model_fds[file]` the caller's open descriptors of the model's paths. writes at most `cap` units; returns how many
  there are (n_layers * n_experts) - 0, nothing written, for no weights, a tensor whose bytes are not n_experts equal
  slices, or an expert tensor an overlay stands in for (its file is not the model's: such a model's experts are not
  units).

q4x_mtp_create
  the MTP head of w's model: its metadata w's geometry (the same architecture, widths, heads, experts; one nextn layer
  after w's; its embedding and output borrowed from w - nextn_shared_target_tensors), its layer's tensors an attention
  layer's and the MoE's (the roles q4x_weights_create checks, at index n_layers) and the nextn ones: eh_proj
  (2 n_embd -> n_embd), enorm (n_embd), hnorm (hc n_embd), the head's mixer (norm, down, up). NULL with why as
  q4x_weights_create refuses: a key or a tensor named.

q4x_weights_create
  the model's geometry read from its metadata and every tensor given its role, checked: its name, its dims (the
  geometry's), its type (one the role's reader takes). NULL with why - the key or the tensor named - for a model of
  another architecture, a key missing or out of range, a tensor missing, of another shape or a type the role does not
  read, or a tensor no role takes (a file of another layout is refused, not half read). the weights point into the
  model's tensors: the model outlives them. an overlay (or NULL): its tensors stand in for the model's of their names
  (tools/install/overlay_q4x.py's: huihui-ai's abliteration), each checked against its role as the model's would be;
  refused besides - an overlay tensor no role reads, or one the model has no tensor of. the overlay outlives them too.
*/
