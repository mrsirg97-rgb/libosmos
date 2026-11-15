#ifndef OSMOS_Q35_FORWARD_H
#define OSMOS_Q35_FORWARD_H

/* Qwen3.5 / 3.8's dense models (general.architecture qwen35: Qwen3.8-27B, the 10M flagship's family - DESIGN.md,
   "the 10M agent"): every attn_interval-th layer gated attention, the rest Gated DeltaNet, each followed by a dense
   SwiGLU. the q4x family's kernels (q4x_ops.h) on one GPU, the weights whole in its memory; a sequence's state its
   attention layers' cache (Q8_0) and its DeltaNets' conv history and recurrent state */

#include <stdbool.h>
#include <stdint.h>

#include "osmos/model/gguf.h"

#define Q35_ERR 256
#define Q35_MAX_LAYERS 128

typedef struct Q35Geometry
{
  int n_layers, n_embd, n_ff, n_vocab;
  int n_heads, n_kv_heads, head_dim, rope_dim;
  int attn_interval;
  bool recurrent[Q35_MAX_LAYERS]; /* a Gated DeltaNet, else gated attention */
  int ssm_conv, ssm_state, ssm_groups, ssm_v_heads, ssm_inner;
  float rms_eps, rope_base;
  uint32_t native_ctx;
} Q35Geometry;

bool q35_geometry(const Gguf *model, Q35Geometry *g, char err[Q35_ERR]);

/* a layer's tensors (the ones its kind has; the rest NULL) and each matrix's own scale (NVFP4's: its .scale beside it,
   1 for any other type) */
typedef struct Q35Layer
{
  const GgufTensor *attn_norm, *post_norm, *ffn_gate, *ffn_up, *ffn_down;
  float s_ffn_gate, s_ffn_up, s_ffn_down;
  /* Gated DeltaNet */
  const GgufTensor *qkv, *z, *alpha, *beta, *conv, *dt, *a, *ssm_norm, *ssm_out;
  float s_qkv, s_z, s_alpha, s_beta, s_ssm_out;
  /* gated attention */
  const GgufTensor *q, *k, *v, *q_norm, *k_norm, *out;
  float s_q, s_k, s_v, s_out;
} Q35Layer;

typedef struct Q35Weights
{
  Q35Geometry geo;
  const Gguf *model;
  const GgufTensor *embd, *out_norm, *output;
  float s_output;
  Q35Layer layer[Q35_MAX_LAYERS];
} Q35Weights;

/* every tensor the forward reads, found and its shape checked against the geometry; NULL and err otherwise */
Q35Weights *q35_weights_create(const Gguf *model, char err[Q35_ERR]);
void q35_weights_destroy(Q35Weights *w);

/* the forward: the weights in the GPU's memory, a step's buffers for max_tokens tokens, contexts to max_ctx (past the
   model's native length its rotation is YaRN's, factor max_ctx / native, as q4x_rope) */
typedef struct Q35Forward Q35Forward;
typedef struct Q35Seq Q35Seq;

Q35Forward *q35_forward_create(const Q35Weights *w, uint32_t max_tokens, uint32_t max_ctx, char err[Q35_ERR]);
void q35_forward_destroy(Q35Forward *f);
Q35Seq *q35_seq_create(Q35Forward *f, char err[Q35_ERR]);
void q35_seq_destroy(Q35Seq *s);
int32_t q35_seq_position(const Q35Seq *s);

/* n tokens (n <= max_tokens) after the sequence's own: its state advanced, the logits of the last `rows` of them on
   the host (q35_logits: rows x n_vocab); a tap, when set, gets every layer's residual after it (n x n_embd, the GPU's) */
typedef void (*q35_tap_fn)(void *arg, int layer, const float *dev_h, uint32_t n, uint32_t dim);
void q35_forward_tap(Q35Forward *f, q35_tap_fn fn, void *arg);
bool q35_step(Q35Forward *f, Q35Seq *s, const int32_t *tokens, uint32_t n, uint32_t rows, char err[Q35_ERR]);
const float *q35_logits(const Q35Forward *f);

#endif
