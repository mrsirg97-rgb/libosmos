#ifndef OSMOS_LFM3_FORWARD_H
#define OSMOS_LFM3_FORWARD_H

/* LiquidAI's LFM3 (general.architecture lfm3: LFM3-1.7B, vibes-flash's family - DESIGN.md, "vibes-flash: LFM3"): a
   short convolution or attention a layer (full, or a window of keys), each followed by a dense SwiGLU, each block's
   output normed before the residual. q35's shape on one GPU, the weights whole in its memory; a sequence's state its
   convolutions' last rows and its attention layers' cache (Q8_0: the full layers' every row, the window's ring).
   the engine's contracts beside it (DESIGN.md, "LFM3 step 2"): runs, rollback, the prompt's end, a saved state */

#include <stdbool.h>
#include <stdint.h>

#include "osmos/model/gguf.h"
#include "osmos/model/state_io.h"

#define LFM3_ERR 256
#define LFM3_MAX_LAYERS 64
#define LFM3_HEAD_DIM 128 /* the attention kernels' head (q35's widened to it: DESIGN.md) */
#define LFM3_MAX_BACK 8   /* the tokens a sequence can drop (a drafted round's) */
#define LFM3_MAX_RUNS 64  /* a step's sequences at most */

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
   shapes (q4x_ops.h's Q4X_CONV_TAPS_MAX, Q4X_GQA_MAX; LFM3_HEAD_DIM). rope_pairs is lfm3_weights_create's (the
   rope_freqs tensor's: rotated pairs a factor 1, the rest at least 1e30 - any other factor refused) */
bool lfm3_geometry(const Gguf *model, Lfm3Geometry *g, char err[LFM3_ERR]);

/* a layer's tensors (the ones its kind has; the rest NULL): every matrix the file's one matrix type (vibes1.7b's
   files: Q4_0, or BF16) */
typedef struct Lfm3Layer
{
  const GgufTensor *attn_norm, *post_attn_norm, *ffn_norm, *post_ffn_norm, *ffn_gate, *ffn_up, *ffn_down;
  /* the short convolution */
  const GgufTensor *in_proj, *conv, *out_proj;
  /* attention, full or sliding */
  const GgufTensor *q, *k, *v, *q_norm, *k_norm, *gate, *out;
} Lfm3Layer;

typedef struct Lfm3Weights
{
  Lfm3Geometry geo;
  const Gguf *model;
  const GgufTensor *embd, *out_norm, *rope_freqs; /* the head the embedding: tied, never a tensor of its own */
  Lfm3Layer layer[LFM3_MAX_LAYERS];
} Lfm3Weights;

/* every tensor the forward reads, found and its shape and type checked against the geometry; NULL and err otherwise
   (a matrix of another type than the file's others, a type the matvec does not take, a tensor no role takes, a
   role's tensor missing) */
Lfm3Weights *lfm3_weights_create(const Gguf *model, char err[LFM3_ERR]);
void lfm3_weights_destroy(Lfm3Weights *w);

typedef struct Lfm3Forward Lfm3Forward;
typedef struct Lfm3Seq Lfm3Seq;

/* the weights in the GPU's memory, a step's buffers for max_tokens tokens, contexts to max_ctx (past the native
   length the full layers' rotation is YaRN's, factor max_ctx / native, the sliding layers' never), logit_rows the
   rows a step keeps the logits of at most (0: max_tokens) */
/* the tied head as the step reads it: a Q8_0 copy made at load (ggml's quantizer: a row's blocks of 32, d = amax /
   127) - half the BF16 table's bytes a token reads, the embedding still the file's - or the file's BF16 itself */
typedef enum Lfm3Head
{
  LFM3_HEAD_Q8_0,
  LFM3_HEAD_FILE,
} Lfm3Head;

/* lsa: latent sparse attention's maps (the release's vibes1.7b-lsa-maps.safetensors: a full layer's W_Q_idx and
   W_K_idx, F32 [256 x 64] - its 4 KV heads' zero-tail dims), or NULL: every full layer dense. with them a decode's
   full layers read incarnate's served read (osmos/lsa.h: 512-key blocks, 64 buckets each, the top 1,024 buckets' keys
   cut at 4,096, the window and the sinks); a prompt's stay dense (lsa.py's) */
/* yarn_from: the YaRN stages' base (0: the native context) - a sequence's full layers rotated at the least factor 2^s
   whose span (yarn_from 2^s) holds its positions: plain rope below the base, and crossing a span its cached keys
   turned to the next stage's (DESIGN.md, "vibes-flash: 2M"); the stages to max_ctx's, LFM3_MAX_STAGES at most */
#define LFM3_MAX_STAGES 8
#define LFM3_MAX_CTX 2097152 /* a context's most positions: 2M (the stages to YaRN x8 over the native 262,144) */
typedef struct Lfm3Options
{
  uint32_t max_tokens, max_ctx, logit_rows;
  Lfm3Head head;
  const char *lsa;
  uint32_t yarn_from;
  bool yarn_static;         /* every position the top stage's rotation (YaRN at max_ctx over yarn_from), no stages */
  uint32_t lsa_sparse_from; /* with LSA, a prompt's tokens from this position read sparse (0: LFM3_LSA_SPARSE_FROM) */
} Lfm3Options;
/* a prompt token's read sparse from here: under LSA_READ_MAX positions a read is every key (the dense attention's) */
#define LFM3_LSA_SPARSE_FROM LSA_READ_MAX
/* the model's open files' paths are its Gguf's (the identity hashes them) */
Lfm3Forward *lfm3_forward_create(const Lfm3Weights *w, const Lfm3Options *o, char err[LFM3_ERR]);
void lfm3_forward_destroy(Lfm3Forward *f);

Lfm3Seq *lfm3_seq_create(Lfm3Forward *f, char err[LFM3_ERR]);
void lfm3_seq_destroy(Lfm3Seq *s);
int32_t lfm3_seq_position(const Lfm3Seq *s);

/* runs sequences' tokens in one step (seqs[i] the next lens[i] tokens, consecutive; each once): every matrix read
   once for all of them, each sequence's convolution states and caches its own - a token's bits the same however the
   sequences share the step or a sequence's tokens are split into steps. the kept rows' logits (lfm3_row_logits): a
   row kept unless its token is before its sequence's prompt end - 1 (lfm3_seq_prompt). false with why, every sequence
   unchanged, for a refusal before the work; false after it began (a GPU call failed): the step's sequences refused
   from then on (lfm3_seq_fail's), until a load */
bool lfm3_step_runs(Lfm3Forward *f, Lfm3Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                    char err[LFM3_ERR]);
/* the same step in a prompt's math (the engine's read): every matrix on the matrix cores (mm_tokens: f16 inputs), its
   bits a prompt's however it is split - not a decode's (two maths, as q35's and hy4's: DESIGN.md). false past it as
   lfm3_step_runs, and for an activation past f16's range (the step refused) */
bool lfm3_step_read(Lfm3Forward *f, Lfm3Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                    char err[LFM3_ERR]);
/* one sequence, its last `rows` rows' logits on the host (lfm3_logits: rows x n_vocab) - the checks' form: a step past
   LFM3_MV_TOKENS tokens a prompt's math, the rest a decode's */
#define LFM3_MV_TOKENS 4
bool lfm3_step(Lfm3Forward *f, Lfm3Seq *s, const int32_t *tokens, uint32_t n, uint32_t rows, char err[LFM3_ERR]);
const float *lfm3_logits(const Lfm3Forward *f);
/* row's logits (device, n_vocab floats) of the last step, NULL for a row not kept */
const float *lfm3_row_logits(const Lfm3Forward *f, uint32_t row);
/* the greatest token of each of rows row0 .. row0 + rows - 1 (the lowest among equals); false for a row not kept */
bool lfm3_greedy(Lfm3Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[LFM3_ERR]);

/* the prompt ends at end (its last token end - 1): rows before it not kept (the head not run for them) */
bool lfm3_seq_prompt(Lfm3Seq *s, uint32_t end);
/* the tokens a sequence can drop (LFM3_MAX_BACK at most; 0 after a load or a new one; with LSA never across a block's
   close: its buckets made of the keys; never across a YaRN stage's change: its keys turned), and the drop: every state as before them - the bits of never having run
   them. false past the reach, unchanged */
uint32_t lfm3_seq_reach(const Lfm3Seq *s);
bool lfm3_seq_rollback(Lfm3Seq *s, uint32_t n);
/* a sequence refused (every step, rollback, save) until a load: an engine's batch that failed past its step */
void lfm3_seq_fail(Lfm3Seq *s);
/* a 64-bit key of what a saved state depends on (its format, the geometry, the ring, the model's file) */
uint64_t lfm3_identity(const Lfm3Forward *f);
/* a sequence as one byte string (a header, each convolution's current state, the full layers' rows below the
   position, the sliding layers' rings, then a sum) - load refuses another identity or format, a size not its
   position's, a damaged string: the sequence then empty */
uint64_t lfm3_seq_state_bytes(const Lfm3Seq *s);
bool lfm3_seq_save_to(const Lfm3Seq *s, StateSink sink, void *ctx, char err[LFM3_ERR]);
bool lfm3_seq_load_from(Lfm3Seq *s, uint64_t bytes, StateSource src, void *ctx, char err[LFM3_ERR]);

/* the sequence's YaRN stage (0: plain rope) */
uint32_t lfm3_seq_stage(const Lfm3Seq *s);

/* the per-op checks' (llama-eval-callback's tensors): a tap, when set, gets each layer's named outputs - "operator"
   (the conv's or attention's, before its post-norm) and "residual" (the layer's output) - n x n_embd, the GPU's */
typedef void (*lfm3_tap_fn)(void *arg, int layer, const char *what, const float *dev, uint32_t n, uint32_t dim);
void lfm3_forward_tap(Lfm3Forward *f, lfm3_tap_fn fn, void *arg);

#endif
