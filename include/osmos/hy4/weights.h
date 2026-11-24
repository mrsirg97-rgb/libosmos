#ifndef OSMOS_HY4_WEIGHTS_H
#define OSMOS_HY4_WEIGHTS_H

#include "osmos/err.h"
#include <stdbool.h>
#include <stdint.h>

#include "osmos/model/engram.h"
#include "osmos/model/gguf.h"

#define HY4_MAX_LAYERS 64
#define HY4_MAX_HC 4      /* the residual's streams (Hy4-flash's iHC: 4) */
#define HY4_MAX_ENGRAM 8  /* the Engram layers (Hy4-flash's: 4) */

/* Tencent's Hy4-flash text model (general.architecture hy_v4_flash, llama.cpp's keys), from its metadata */
typedef struct Hy4Geometry
{
  int n_layers;  /* the trunk's (block_count less the drafter's nextn layers) */
  int n_nextn;   /* the MTP layers after the trunk (blk.n_layers ..), the drafter's - not read by the forward */
  int n_embd, n_vocab, n_ff;
  uint32_t native_ctx;
  float rms_eps;
  /* iHC: hc streams; pre's and post's epsilon, post's magnitude */
  int hc;
  float hc_eps, hc_magnitude;
  /* gated MLA: a layer's heads (window layers wider), the latent ranks, a head's dims; the cache's row kv_lora +
     rope_dim (absorbed) */
  int n_heads[HY4_MAX_LAYERS];
  int q_lora, kv_lora, nope_dim, rope_dim, v_dim;
  /* a layer sliding (its window: the keys back, its own counted) or sparse (its indexer); the ropes' bases */
  bool sliding[HY4_MAX_LAYERS];
  int window;
  float rope_base, rope_base_swa;
  /* the sparse layers' indexer: heads, a key's dim, the keys a query reads at most, the tokens a pooled group */
  int ix_heads, ix_dim, ix_topk, ix_pool;
  /* the MLP: the first n_dense layers dense SwiGLU (n_ff), the rest the latent MoE - experts, picked a token, an
     expert's width (its input and output the latent's), the shared experts' count, the weights' scale, each layer's
     clamp on the routed experts' SwiGLU */
  int n_dense, n_experts, n_used, ff_exp, latent, n_shexp;
  float expert_scale;
  float clamp[HY4_MAX_LAYERS];
  /* Engram: before these layers' iHC; n-grams up to engram_ngram, engram_heads an order, a row's dim */
  bool engram[HY4_MAX_LAYERS];
  int n_engram, engram_ngram, engram_heads, engram_dim;
  int32_t engram_pad;
} Hy4Geometry;

/* a layer's tensors; NULL where its kind has none */
typedef struct Hy4Layer
{
  /* iHC before the attention and before the MLP: the mixes (2 hc x hc n_embd), their bases (2 hc), scales (pre, post) */
  const GgufTensor *hc_attn_fn, *hc_attn_base, *hc_attn_scale, *hc_ffn_fn, *hc_ffn_base, *hc_ffn_scale;
  /* gated MLA, with its sandwich norm (post_attn_norm) */
  const GgufTensor *attn_norm, *q_a, *q_a_norm, *q_b, *kv_a, *kv_a_norm, *k_b, *v_b, *gate, *out, *sinks;
  const GgufTensor *post_attn_norm;
  /* the indexer (a sparse layer's): its query from q_a's output, the token keys and their norm, the heads' weights;
     the pooling of a group's keys (value, gate, the slots' bias, the norm) */
  const GgufTensor *ix_q_b, *ix_k, *ix_k_norm, *ix_proj, *ix_pool_value, *ix_pool_gate, *ix_pool_ape, *ix_pool_norm;
  /* the MLP, with its sandwich norm (post_ffn_norm): a dense layer's SwiGLU; a MoE's router and its bias, the experts
     in the latent, the latent's projections, the shared expert */
  const GgufTensor *ffn_norm, *post_ffn_norm;
  const GgufTensor *ffn_gate, *ffn_up, *ffn_down;
  const GgufTensor *router, *router_bias, *exp_gate, *exp_up, *exp_down, *routed_down, *routed_up;
  const GgufTensor *sh_gate, *sh_up, *sh_down;
  /* Engram (an Engram layer's): the fused keys and value (hc keys, then the value) and their biases, the folded
     norms of the gate, the rows' per-head norms, the value's gain, the streams' scales */
  const GgufTensor *eg_kv, *eg_kv_b, *eg_gate, *eg_lookup_norm, *eg_pre_gamma, *eg_hc_scale;
} Hy4Layer;

typedef struct Hy4Weights
{
  Hy4Geometry geo;
  const Gguf *model;
  const GgufTensor *embed, *output_norm, *output;
  const GgufTensor *out_hc_fn, *out_hc_base, *out_hc_scale; /* the head's iHC: pre alone */
  const GgufTensor *engram_table;                           /* per_layer_token_embd: every Engram layer's rows */
  uint64_t engram_rows[HY4_MAX_ENGRAM];                     /* each Engram layer's rows of it (its columns' span) */
  Hy4Layer layer[HY4_MAX_LAYERS];
} Hy4Weights;

bool hy4_geometry(const Gguf *model, Hy4Geometry *g, char err[OSMOS_ERR]);
Hy4Weights *hy4_weights_create(const Gguf *model, char err[OSMOS_ERR]);
void hy4_weights_destroy(Hy4Weights *w);

/* the Engram hash's shape (osmos/model/engram.h) from the file's metadata */
bool hy4_engram_shape(const Hy4Weights *w, EngramShape *s, char err[OSMOS_ERR]);

#endif

/*
hy4_geometry
  the model's shape from its metadata, each key checked (its range, and what this build implements: sigmoid routing
  with normalized weights and the bias for the pick alone, one latent geometry for both attention kinds, a pooled
  group dividing the indexer's keys, the dense layers before every MoE one, the Engram layers within the trunk).
  per-layer lists cover block_count (the trunk and the nextn layers); the geometry keeps the trunk's. false with the
  key and why in err.

hy4_weights_create
  the geometry, then every tensor given its role and checked: its name, its dims (the geometry's), its type (one the
  role's reader takes: a matrix any type the table's matvec reads, the routed experts any its experts' path reads, a
  norm, bias, sink, router or iHC parameter F32). NULL with why - the key or the tensor named - for another
  architecture, a key missing or out of range, a tensor missing, of another shape or a type its role does not read,
  or a tensor no role takes. the nextn layers' tensors (blk.n_layers ..) are left to the drafter, not refused. the
  weights point into the model's tensors: the model outlives them.

hy4_engram_shape
  the hash as the reference computes it (modeling_hy_v4_flash.py's HYV4FlashEngramLayer.hash_ngrams), from the
  file's lists (ple.*, engram.*). the token map's covered length: llama.cpp's converter extends the checkpoint's
  lookup table by identity to the vocabulary; the reference reads a token past its table as the pad. the extension is
  the map's trailing run of i -> i past every earlier entry (the release's table: 120,818 tokens, 92,620 compressed
  ids); the compressed count the covered entries' greatest + 1. false with why for lists not the geometry's lengths.
*/
