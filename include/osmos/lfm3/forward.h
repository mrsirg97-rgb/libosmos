#ifndef OSMOS_LFM3_FORWARD_H
#define OSMOS_LFM3_FORWARD_H

/* LiquidAI's LFM3 (general.architecture lfm3: LFM3-1.7B, vibes-flash's family - DESIGN.md, "vibes-flash: LFM3"): a
   short convolution or attention a layer (full, or a window of keys), each followed by a dense SwiGLU, each block's
   output normed before the residual. q35's shape on one GPU, the weights whole in its memory; a sequence's state its
   convolutions' last rows and its attention layers' cache (Q8_0: the full layers' every row, the window's ring).
   step 1's surface: the model and one sequence's forward - the engine's contracts (runs, rollback, saved state) are
   step 2's, beside q35's */

#include <stdbool.h>
#include <stdint.h>

#include "osmos/model/gguf.h"

#define LFM3_ERR 256
#define LFM3_MAX_LAYERS 64

typedef enum Lfm3Kind
{
  LFM3_CONV,    /* the short convolution: y = out_proj(C * conv(B * x)) */
  LFM3_FULL,    /* attention over every key; its rotation the zero tail's (rope_pairs of head_dim / 2) */
  LFM3_SLIDING, /* attention over the last `window` keys; every pair rotated */
} Lfm3Kind;

typedef struct Lfm3Geometry
{
  int n_layers, n_embd, n_ff, n_vocab;
  int n_heads, n_kv_heads, head_dim; /* n_kv_heads: an attention layer's (a conv layer has none) */
  Lfm3Kind kind[LFM3_MAX_LAYERS];
  int conv_taps; /* shortconv.l_cache: a convolution's taps (its state the last conv_taps - 1 rows) */
  int window;    /* attention.sliding_window: a sliding layer's keys, its own the last */
  int rope_pairs; /* the full layers' rotated pairs: rope_freqs' finite factors (the rest unrotated) */
  float rms_eps, rope_base, rope_base_swa;
  uint32_t native_ctx;
} Lfm3Geometry;

/* from the file's metadata, every field checked: the layer kinds by the per-layer head_count_kv (0: a convolution)
   and the window pattern; refused - an unknown architecture, a kind the file does not say, a field past the kernels'
   shapes (q4x_ops.h's Q4X_CONV_TAPS_MAX, Q4X_GQA_MAX) */
bool lfm3_geometry(const Gguf *model, Lfm3Geometry *g, char err[LFM3_ERR]);

/* a layer's tensors (the ones its kind has; the rest NULL) and each matrix's own scale (NVFP4's: its .scale beside it,
   1 for any other type) */
typedef struct Lfm3Layer
{
  const GgufTensor *attn_norm, *post_attn_norm, *ffn_norm, *post_ffn_norm, *ffn_gate, *ffn_up, *ffn_down;
  float s_ffn_gate, s_ffn_up, s_ffn_down;
  /* the short convolution */
  const GgufTensor *in_proj, *conv, *out_proj;
  float s_in_proj, s_out_proj;
  /* attention, full or sliding */
  const GgufTensor *q, *k, *v, *q_norm, *k_norm, *gate, *out;
  float s_q, s_k, s_v, s_gate, s_out;
} Lfm3Layer;

typedef struct Lfm3Weights
{
  Lfm3Geometry geo;
  const Gguf *model;
  const GgufTensor *embd, *out_norm, *rope_freqs; /* the head the embedding: tied, never a tensor of its own */
  Lfm3Layer layer[LFM3_MAX_LAYERS];
} Lfm3Weights;

/* every tensor the forward reads, found and its shape and type checked against the geometry; NULL and err otherwise
   (an NVFP4 matrix without its one F32 .scale, a tensor no role takes, a role's tensor missing) */
Lfm3Weights *lfm3_weights_create(const Gguf *model, char err[LFM3_ERR]);
void lfm3_weights_destroy(Lfm3Weights *w);

typedef struct Lfm3Forward Lfm3Forward;
typedef struct Lfm3Seq Lfm3Seq;

/* the weights in the GPU's memory, a step's buffers for max_tokens tokens, contexts to max_ctx (past the native
   length the full layers' rotation is YaRN's, factor max_ctx / native, the sliding layers' never), logit_rows the
   rows a step keeps the logits of at most (0: max_tokens) */
typedef struct Lfm3Options
{
  uint32_t max_tokens, max_ctx, logit_rows;
} Lfm3Options;
Lfm3Forward *lfm3_forward_create(const Lfm3Weights *w, const Lfm3Options *o, char err[LFM3_ERR]);
void lfm3_forward_destroy(Lfm3Forward *f);

Lfm3Seq *lfm3_seq_create(Lfm3Forward *f, char err[LFM3_ERR]);
void lfm3_seq_destroy(Lfm3Seq *s);
int32_t lfm3_seq_position(const Lfm3Seq *s);

/* n tokens (n <= max_tokens) after the sequence's own: its state advanced, the logits of the last `rows` of them on
   the host (lfm3_logits: rows x n_vocab); refused past max_ctx, a token past the vocabulary, rows > n or past
   logit_rows - the sequence then unchanged */
bool lfm3_step(Lfm3Forward *f, Lfm3Seq *s, const int32_t *tokens, uint32_t n, uint32_t rows, char err[LFM3_ERR]);
const float *lfm3_logits(const Lfm3Forward *f);

/* the per-op checks' (llama-eval-callback's tensors): a tap, when set, gets each layer's named outputs - "operator"
   (the conv's or attention's, before its post-norm) and "residual" (the layer's output) - n x n_embd, the GPU's */
typedef void (*lfm3_tap_fn)(void *arg, int layer, const char *what, const float *dev, uint32_t n, uint32_t dim);
void lfm3_forward_tap(Lfm3Forward *f, lfm3_tap_fn fn, void *arg);

#endif
