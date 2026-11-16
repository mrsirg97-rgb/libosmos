#ifndef OSMOS_Q35_FORWARD_H
#define OSMOS_Q35_FORWARD_H

/* Qwen3.5 / 3.8's dense models (general.architecture qwen35: Qwen3.8-27B, the 10M flagship's family - DESIGN.md,
   "the 10M agent"): every attn_interval-th layer gated attention, the rest Gated DeltaNet, each followed by a dense
   SwiGLU. the q4x family's kernels (q4x_ops.h) on one GPU, the weights whole in its memory; a sequence's state its
   attention layers' cache (Q8_0) and its DeltaNets' conv history and recurrent state */

#include <stdbool.h>
#include <stdint.h>

#include "osmos/model/gguf.h"
#include "osmos/model/state_io.h"

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
/* the same, logit_rows the logits' rows a step keeps at most (0: max_tokens, every row) - a server's prompt step keeps
   its last row alone (q35_seq_prompt), so its table is a round's rows, not max_tokens x 248K floats */
typedef struct Q35Options
{
  uint32_t max_tokens, max_ctx, logit_rows;
} Q35Options;
Q35Forward *q35_forward_create_with(const Q35Weights *w, const Q35Options *o, char err[Q35_ERR]);
void q35_forward_destroy(Q35Forward *f);
/* latent sparse attention (osmos/lsa.h): the maps file (tools/convert/lsa_maps.py's: W_Q and W_K an attention layer),
   before any sequence; from then every attention layer's keys indexed and each closed block bucketed (a prompt's
   steps and a decode's), a decode step's (n <= Q35_ATTN_DECODE_MAX) attention its read's alone, a prompt's dense (as
   lsa.py's quality: the read's cost measured on what follows a prompt). the rotation YaRN's at yarn x native, the
   maps' own (-vibes': 40). maps NULL: the rotation alone, the read off (the dense model it is measured against) */
bool q35_forward_lsa(Q35Forward *f, const char *maps, float yarn, char err[Q35_ERR]);
Q35Seq *q35_seq_create(Q35Forward *f, char err[Q35_ERR]);
void q35_seq_destroy(Q35Seq *s);
int32_t q35_seq_position(const Q35Seq *s);

/* n tokens (n <= max_tokens) after the sequence's own: its state advanced, the logits of the last `rows` of them on
   the host (q35_logits: rows x n_vocab); a tap, when set, gets every layer's residual after it (n x n_embd, the GPU's) */
typedef void (*q35_tap_fn)(void *arg, int layer, const float *dev_h, uint32_t n, uint32_t dim);
void q35_forward_tap(Q35Forward *f, q35_tap_fn fn, void *arg);
bool q35_step(Q35Forward *f, Q35Seq *s, const int32_t *tokens, uint32_t n, uint32_t rows, char err[Q35_ERR]);
const float *q35_logits(const Q35Forward *f);

/* the engine's (DESIGN.md, "0.9.0: q35 behind the engine"): q4x's contracts (osmos/q4x/forward.h) on q35's state */
#define Q35_MAX_BACK 8                 /* the tokens a sequence can drop (a drafted round's) */
#define Q35_SLOTS (Q35_MAX_BACK + 1)   /* a DeltaNet's states and conv histories kept: a ring */
#define Q35_MAX_RUNS 64                /* a step's sequences at most */
/* runs sequences' tokens in one step (seqs[i] the next lens[i] tokens, consecutive; each once): every matrix read once
   for all of them, each sequence's DeltaNet, conv, rotation, cache and attention its own - a token's bits the same
   however the sequences share the step. the kept rows' logits (q35_row_logits): a row kept unless its token is
   before its sequence's prompt end - 1 (q35_seq_prompt). false with why, every sequence unchanged, for a refusal
   before the work; false after it began (a GPU call failed, an activation past f16's range): the step's sequences
   refused from then on (q35_seq_fail's), until a load */
bool q35_step_runs(Q35Forward *f, Q35Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                   char err[Q35_ERR]);
/* the same step in a prompt's math (the engine's read: the matmul at any size, attention dense), its bits a prompt's
   however it is split - not a decode's (two maths, as DeepSeek's: DESIGN.md) */
bool q35_step_read(Q35Forward *f, Q35Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                   char err[Q35_ERR]);
/* row's logits (device, n_vocab floats) of the last q35_step_runs, NULL for a row not kept */
const float *q35_row_logits(const Q35Forward *f, uint32_t row);
/* the greatest token of each of rows row0 .. row0 + rows - 1 (the lowest among equals); false for a row not kept */
bool q35_greedy(Q35Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[Q35_ERR]);
/* the prompt ends at end (its last token end - 1): rows before it not kept (the head not run for them) */
bool q35_seq_prompt(Q35Seq *s, uint32_t end);
/* the tokens a sequence can drop (Q35_MAX_BACK at most; 0 after a load, a mark's load, or a new one; with LSA no
   further back than its last closed block), and the drop: every state as before them, the bits of never having run
   them. false past the reach, unchanged */
uint32_t q35_seq_reach(const Q35Seq *s);
bool q35_seq_rollback(Q35Seq *s, uint32_t n);
/* a sequence refused (every step, rollback, save) until a load: an engine's batch that failed past its step */
void q35_seq_fail(Q35Seq *s);
/* a 64-bit key of what a saved state depends on (its format, the geometry, the model's files, the rotation) */
uint64_t q35_identity(const Q35Forward *f);
/* a sequence as one byte string (osmos/gpu/state_parts.h: a header, the DeltaNets' current states and conv
   histories, the attention rows below the position, LSA's index and buckets, then a sum) - load refuses another
   identity or format, a size not its position's, a damaged string: the sequence then empty */
uint64_t q35_seq_state_bytes(const Q35Seq *s);
bool q35_seq_save_to(const Q35Seq *s, StateSink sink, void *ctx, char err[Q35_ERR]);
bool q35_seq_load_from(Q35Seq *s, uint64_t bytes, StateSource src, void *ctx, char err[Q35_ERR]);
/* a mark (a point the sequence goes back to: what a later token changes - the DeltaNets' states and conv histories,
   the position - not the attention rows below it, which only grow): its bytes, save, load (refused past the
   position or another's size; with LSA, below the last closed block) */
uint64_t q35_seq_mark_bytes(const Q35Seq *s);
bool q35_seq_mark_save(const Q35Seq *s, void *out, uint64_t bytes, char err[Q35_ERR]);
bool q35_seq_mark_load(Q35Seq *s, const void *in, uint64_t bytes, char err[Q35_ERR]);

#endif
