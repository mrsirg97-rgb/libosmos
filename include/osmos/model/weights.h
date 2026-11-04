#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "osmos/mem/pool.h"
#include "osmos/model/gguf.h"

#define WEIGHTS_ERR 256
#define WEIGHTS_MAX_LAYERS 64
#define WEIGHTS_SLICE_NAME 64

typedef struct Geometry
{
  int n_layers;
  int n_vocab;
  int n_embd;
  int n_heads;
  int head_dim;
  int rope_dim;
  int q_lora;
  int o_lora;
  int o_groups;
  int window;
  int n_experts;
  int n_used;
  int expert_ff;
  int hc;
  int sinkhorn_iters;
  int ix_heads;
  int ix_head_dim;
  int ix_topk;
  int cand_blocks;
  int cand_block;
  int cand_source;
  int compress[WEIGHTS_MAX_LAYERS];
  int kv_from[WEIGHTS_MAX_LAYERS];
  bool kv_source[WEIGHTS_MAX_LAYERS];
  bool ix_source[WEIGHTS_MAX_LAYERS];
  bool engram[WEIGHTS_MAX_LAYERS];
  bool dspark_target[WEIGHTS_MAX_LAYERS];
  float clamp_exp[WEIGHTS_MAX_LAYERS];
  float clamp_shexp[WEIGHTS_MAX_LAYERS];
  uint64_t engram_rows[WEIGHTS_MAX_LAYERS];
  bool engram_own; /* the Engram in the model's own files (llama.cpp's ple.*: one table tensor, the gate folded) */
  int engram_heads;
  int engram_head_dim;
  int engram_max_ngram;
  int engram_cols;
  int engram_pad;
  int engram_vocab;
  float rms_eps;
  float hc_eps;
  float rope_base;
  float compress_rope_base;
  float rope_factor;
  int rope_original;
  float yarn_beta_fast;
  float yarn_beta_slow;
  float expert_scale;
} Geometry;

typedef struct LayerWeights
{
  const GgufTensor *attn_norm, *q_a, *q_a_norm, *q_b, *kv, *kv_norm, *out_a, *out_b, *sinks;
  const GgufTensor *comp_kv, *comp_gate, *comp_norm;
  const GgufTensor *ix_k, *ix_k_norm, *ix_q_b, *ix_proj;
  const GgufTensor *eg_q, *eg_k, *eg_gate, *eg_wkv, *eg_embd;
  const GgufTensor *hc_attn_fn, *hc_attn_scale, *hc_attn_base, *hc_ffn_fn, *hc_ffn_scale, *hc_ffn_base;
  const GgufTensor *ffn_norm, *router, *router_bias;
  const GgufTensor *sh_gate, *sh_up, *sh_down;
  const GgufTensor *exp_gate, *exp_up, *exp_down;
} LayerWeights;

typedef struct Weights
{
  const Gguf *model;
  const Gguf *engram;
  Geometry geo;
  const GgufTensor *embed, *output, *output_norm;
  const GgufTensor *eg_token_map, *eg_primes, *eg_offsets, *eg_multipliers;
  LayerWeights layer[WEIGHTS_MAX_LAYERS];
  GgufTensor eg_slice[WEIGHTS_MAX_LAYERS]; /* the Engram's own: a layer's rows of per_layer_token_embd (eg_embd) */
  char eg_slice_name[WEIGHTS_MAX_LAYERS][WEIGHTS_SLICE_NAME];
} Weights;

Weights *weights_create(const Gguf *model, const Gguf *engram, char err[WEIGHTS_ERR]);
void weights_destroy(Weights *w);

size_t weights_experts(const Weights *w, const int *model_fds, PoolUnit *units, size_t cap);

#endif

/*
Weights
  DeepSeek-V4.1-Flash's tensors, by role, from its GGUF (`model`, the split model) and its Engram tables (`engram`, the
  model's `deepseek41.engram.file`): each a GgufTensor of theirs - borrowed, not owned; both Ggufs must outlive the
  Weights (libtrdr's scope rule). `geo` is the model's metadata, read once and checked; `kv_from[l]` the KV source a
  compressed layer reads (the last at or before it, of its ratio; -1 uncompressed), `engram_cols` the hash columns an
  Engram position reads ((max_ngram - 1) * heads), `engram_pad` the token a hash reads before a sequence's start and
  `engram_vocab` the compressed tokens the hash's map gives. a field a layer does not have
  is NULL (`comp_gate` off the ratio-2 sources, `ix_k` off the KV sources, `eg_*` off the Engram layers, ...).
  the Engram in two forms (geo.engram_own): ds1017's, a second file - each layer's table its own tensor, the gate as
  q and k (BF16), the hash's tables tensors; llama.cpp's (unsloth's UD), the model's own - the layers' tables rows of
  one tensor (per_layer_token_embd; each layer's `eg_embd` a slice of it in `eg_slice`, owned by the Weights), the
  gate folded (`eg_gate`, F32 q k), the hash's tables metadata (engram.c). a field of the other form is NULL.

weights_create
  the model as this build knows it, or nothing: every tensor a layer must have, present, of its exact dims (from the
  geometry) and of a type its role allows - a role read through the kernels any type the type table gives its path
  (qtype.h), a role read raw (a norm, a bias, the gate's halves) its one type; an optional one (exp_probs_b_vl: the
  router's bias for images, checked and never read) present or not; none it must not have; no tensor of a
  name it does not know; the geometry's metadata present and consistent (its layer lists within the layers, a KV source
  compressed, an index source among the KV sources or reading one). anything else refuses the whole model - NULL, the
  tensor or key and why in `err`. DSpark's blocks are not in the GGUF (maya's dspark.bin) and not read here.

weights_destroy
  the Weights alone (the Ggufs are the caller's). NULL is nothing to do.

weights_experts
  the routed experts as the pool's units, unit = layer * n_experts + expert, each three parts - the expert's slice of
  ffn_gate_exps, ffn_up_exps, ffn_down_exps (gate, up, down) - in the model's files, `model_fds[file]` the caller's
  open descriptors of `model->paths`. writes at most `cap` units; returns how many there are (n_layers * n_experts).
*/
