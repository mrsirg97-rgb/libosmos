#ifndef OSMOS_INK2_WEIGHTS_H
#define OSMOS_INK2_WEIGHTS_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/mem/pool.h"
#include "osmos/model/gguf.h"

#define INK2_ERR 256
#define INK2_MAX_LAYERS 64
#define INK2_CONV_TAPS_MAX 8 /* a short convolution's taps (Inkling-2's: 4) */
#define INK2_RES_QUERIES 3   /* a full layer's attention residual: one query each for q, k and v */

/* Inkling-2's text model (general.architecture inkling2, tools/convert/ink2_ref_gguf.py's keys), from its metadata */
typedef struct Ink2Geometry
{
  int n_layers, n_embd;
  int n_vocab, n_vocab_rows; /* the real tokens (the head's columns read); the embedding's rows (padded past them) */
  uint32_t native_ctx;
  float rms_eps;
  /* every layer's attention: heads, KV heads, a head's dim (q, k, v alike); a layer sliding (its window) or full */
  int n_heads, n_kv_heads, head_dim, window;
  bool sliding[INK2_MAX_LAYERS];
  /* the relative bias: each head's d_rel states mixed into a bias by distance, out to rel_extent (a sliding layer's:
     its window); the full layers' log scaling of q and the relative states past log_floor positions */
  int d_rel, rel_extent, log_floor;
  float log_alpha;
  float q_clip, k_clip, rel_clip; /* the L2 clips' thresholds */
  int conv;                       /* the short convolutions' taps */
  int res_block;                  /* the attention residual's block: a layer every res_block starts one */
  /* the MoE (every layer's; no shared expert): experts, picked a token (sigmoid scores, the bias for the pick alone,
     the weights normalized then scaled), an expert's width, the SwiGLU's softcaps */
  int n_experts, n_used, ff_exp;
  float route_scale, gate_softcap, up_softcap;
  float logit_softcap;
} Ink2Geometry;

/* a layer's tensors; NULL where its kind has none */
typedef struct Ink2Layer
{
  /* the attention residuals before the attention (none at layer 0; a full layer's three queries: q, k, v) and the MoE */
  const GgufTensor *attn_res_query, *ffn_res_query;
  /* the input norms: a sliding layer's one, a full layer's for q, k and v */
  const GgufTensor *attn_norm, *q_norm, *k_norm, *v_norm;
  const GgufTensor *q, *k, *v, *rel, *out, *gate, *sink;
  const GgufTensor *q_clip, *rel_clip, *rel_proj;
  /* k's and v's short convolutions: a sliding layer's static (taps x channels), a full layer's static weights plus
     its per-token projection (the taps of each KV head) */
  const GgufTensor *k_conv, *v_conv, *k_conv_static, *v_conv_static, *k_conv_proj, *v_conv_proj;
  const GgufTensor *out_conv, *ffn_out_conv; /* after the attention and after the MoE */
  /* the MoE: its norm, router and the router's bias, the experts, and each expert's own scale (NVFP4's: one an
     expert, gate and up the same; NULL for a type whose blocks are its whole values - MXFP8's) */
  const GgufTensor *ffn_norm, *router, *router_bias;
  const GgufTensor *exp_gate, *exp_up, *exp_down, *exp_gate_scale, *exp_up_scale, *exp_down_scale;
} Ink2Layer;

typedef struct Ink2Weights
{
  Ink2Geometry geo;
  const Gguf *model;
  const GgufTensor *embed, *embed_norm, *output_res_query, *output_norm, *output;
  Ink2Layer layer[INK2_MAX_LAYERS];
} Ink2Weights;

bool ink2_geometry(const Gguf *model, Ink2Geometry *g, char err[INK2_ERR]);
Ink2Weights *ink2_weights_create(const Gguf *model, char err[INK2_ERR]);
void ink2_weights_destroy(Ink2Weights *w);
size_t ink2_weights_experts(const Ink2Weights *w, const int *model_fds, PoolUnit *units, size_t cap);

#endif

/*
ink2_geometry
  the model's shape from its metadata, each key checked (its range, and what this build implements: the router's
  sigmoid gating with normalized weights, a short convolution of at most INK2_CONV_TAPS_MAX taps, the sliding
  layers a list of distinct layers with at least one full layer among them). false with the key and why in err.
  n_vocab_rows is not metadata: ink2_weights_create sets it from the embedding.

ink2_weights_create
  the geometry, then every tensor given its role and checked: its name, its dims (the geometry's), its type (one the
  role's reader takes). NULL with why - the key or the tensor named - for a model of another architecture, a key
  missing or out of range, a tensor missing, of another shape or a type its role does not read, an expert tensor of
  a type that needs a scale with none beside it (NVFP4) or a scale beside one that has none, or a tensor no role
  takes. the drafter's tensors (mtp.*: the DSpark head, its own roles to come) are left to it, not refused. the
  weights point into the model's tensors: the model outlives them.

ink2_weights_experts
  the routed experts as the pool's units, unit = layer * n_experts + expert, each three parts - the expert's slice of
  gate, up and down (a tensor's bytes / n_experts each) - in the model's files, `model_fds[file]` the caller's open
  descriptors of its paths. an expert's scales are not parts: a few bytes a layer, read with the layer. writes at most
  `cap` units; returns how many there are (n_layers * n_experts) - 0, nothing written, for no weights or a tensor whose
  bytes are not n_experts equal slices.
*/
