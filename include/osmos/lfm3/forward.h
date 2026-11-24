#ifndef OSMOS_LFM3_FORWARD_H
#define OSMOS_LFM3_FORWARD_H

/* LiquidAI's LFM3 (general.architecture lfm3): a short convolution or attention a layer (full, or a window of keys),
   each followed by a dense SwiGLU, on one GPU with the weights whole in its memory. a sequence's state is its
   convolutions' last rows and its attention layers' Q8_0 cache (the full layers' every row, the window's ring) */

#include "osmos/err.h"
#include <stdbool.h>
#include <stdint.h>

#include "osmos/model/gguf.h"
#include "osmos/gpu/kept.h"
#include "osmos/model/state_io.h"

#define LFM3_MAX_LAYERS 64
#define LFM3_HEAD_DIM 128 /* the attention kernels' head */
#define LFM3_MAX_BACK 8   /* the tokens a sequence can drop (a drafted round's) */
#define LFM3_MAX_RUNS 64  /* a step's sequences at most */
#define LFM3_SEQ_RESET "a sequence's reset" /* its GPU calls' name (osmos_hip: the fault seam's too) */

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

/* from the file's metadata, every field checked: a layer's kind by its head_count_kv (0: a convolution) and the
   window pattern. false with why for an unknown architecture, a kind the file does not say, a field past the kernels'
   shapes (Q4X_CONV_TAPS_MAX, Q4X_GQA_MAX, LFM3_HEAD_DIM). rope_pairs is set by lfm3_weights_create from rope_freqs
   (rotated pairs a factor 1, the rest at least 1e30; any other factor refused) */
bool lfm3_geometry(const Gguf *model, Lfm3Geometry *g, char err[OSMOS_ERR]);

/* a layer's tensors (the ones its kind has; the rest NULL): every matrix the file's one matrix type */
typedef struct Lfm3Layer
{
  const GgufTensor *attn_norm, *post_attn_norm, *ffn_norm, *post_ffn_norm, *ffn_gate, *ffn_up, *ffn_down;
  const GgufTensor *in_proj, *conv, *out_proj;
  const GgufTensor *q, *k, *v, *q_norm, *k_norm, *gate, *out;
} Lfm3Layer;

typedef struct Lfm3Weights
{
  Lfm3Geometry geo;
  const Gguf *model;
  const GgufTensor *embd, *out_norm, *rope_freqs; /* the head the embedding: tied, never a tensor of its own */
  Lfm3Layer layer[LFM3_MAX_LAYERS];
} Lfm3Weights;

/* every tensor the forward reads, its shape and type checked against the geometry. NULL with why for a matrix of
   another type than the file's others, a type the matvec does not take, a tensor no role takes, a role's tensor
   missing. the weights point into the model: it outlives them */
Lfm3Weights *lfm3_weights_create(const Gguf *model, char err[OSMOS_ERR]);
void lfm3_weights_destroy(Lfm3Weights *w);

typedef struct Lfm3Forward Lfm3Forward;
typedef struct Lfm3Seq Lfm3Seq;

/* the tied head as the step reads it: a Q8_0 copy made at load (ggml's quantizer: a row's blocks of 32, d = amax /
   127; the embedding still the file's), or the file's BF16 itself */
typedef enum Lfm3Head
{
  LFM3_HEAD_Q8_0,
  LFM3_HEAD_FILE,
} Lfm3Head;

/* a YaRN stage: a sequence past `from` positions has its full layers rotated YaRN at `factor` over yarn_from's
   positions (factor 1: the native rotation); a sequence's stage is the last whose `from` its length passes, a crossing
   turns its cached keys. a table: the first from 0, from ascending, factors 1 .. 64, LFM3_MAX_STAGES at most */
#define LFM3_MAX_STAGES 8
typedef struct Lfm3Stage
{
  uint32_t from, factor;
} Lfm3Stage;
/* Lfm3Options' stages when NULL: the native rotation throughout - one stage, its keys never turned */
extern const Lfm3Stage lfm3_stages_default[];
extern const uint32_t lfm3_stages_default_n;
#define LFM3_MAX_CTX 2097152 /* a context's most positions (YaRN x8 over the native 262,144) */
typedef struct Lfm3Options
{
  uint32_t max_tokens, max_ctx, logit_rows;
  Lfm3Head head;
  const char *lsa; /* latent sparse attention's maps (a full layer's W_Q_idx and W_K_idx, F32 [256 x 64]), or NULL:
                      every full layer dense. with them a decode's full layers read sparse (osmos/lsa.h) */
  uint32_t yarn_from;        /* YaRN's original positions (0: the model's native context) */
  const Lfm3Stage *stages;   /* NULL: lfm3_stages_default */
  uint32_t n_stages;
  uint32_t yarn_static_ctx; /* every position YaRN at yarn_static_ctx over yarn_from (64 times it at most), no stages;
                               0: the stages */
  uint32_t lsa_sparse_from; /* with LSA, a prompt's tokens from this position read sparse (0: LFM3_LSA_SPARSE_FROM) */
  const struct Q4xDflash *dflash; /* a DFlash2 drafter (q4x_dflash_create_for against these weights): its taps every
                                     step's, its rings each sequence's; NULL none */
  struct OsmosLedger *ledger;     /* every allocation of the forward and its sequences counted in it; borrowed, outlives
                                     the forward; NULL counts nothing */
} Lfm3Options;
/* under LSA_READ_MAX positions a sparse read is every key: the dense attention's */
#define LFM3_LSA_SPARSE_FROM LSA_READ_MAX
/* a stages table from "from:factor,from:factor,..." (--rope-stages): each a 32-bit decimal, LFM3_MAX_STAGES at most,
   nothing else; false for any other string (t, n then meaningless). lfm3_forward_create checks order and factors */
bool lfm3_stages_parse(const char *c, Lfm3Stage t[LFM3_MAX_STAGES], uint32_t *n);
/* the weights in the GPU's memory, a step's buffers for max_tokens tokens, contexts to max_ctx; logit_rows the rows a
   step keeps logits of at most (0: max_tokens) */
Lfm3Forward *lfm3_forward_create(const Lfm3Weights *w, const Lfm3Options *o, char err[OSMOS_ERR]);
void lfm3_forward_destroy(Lfm3Forward *f);

Lfm3Seq *lfm3_seq_create(Lfm3Forward *f, char err[OSMOS_ERR]);
void lfm3_seq_destroy(Lfm3Seq *s);
bool lfm3_seq_reset(Lfm3Seq *s, char err[OSMOS_ERR]); /* oseq_reset's contract (osmos/run/engine.h) */
int32_t lfm3_seq_position(const Lfm3Seq *s);

/* one step over sequences' tokens (seqs[i] the next lens[i] tokens, consecutive; each sequence once): a token's bits
   the same however the sequences share the step or a sequence's tokens are split into steps. a row's logits kept
   unless its token is before its sequence's prompt end - 1 (lfm3_seq_prompt). false with why and every sequence
   unchanged for a refusal before the work; false after it began (a GPU call failed): the step's sequences refused
   (lfm3_seq_fail) until a load */
bool lfm3_step_runs(Lfm3Forward *f, Lfm3Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                    char err[OSMOS_ERR]);
/* the same step in a prompt's math: every matrix on the matrix cores (f16 inputs), its bits the same however it is
   split but not a decode's. false as lfm3_step_runs, and for an activation past f16's range (the step refused) */
bool lfm3_step_read(Lfm3Forward *f, Lfm3Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                    char err[OSMOS_ERR]);
/* one sequence, its last `rows` rows' logits on the host (lfm3_logits: rows x n_vocab): a step past LFM3_MV_TOKENS
   tokens a prompt's math, the rest a decode's */
#define LFM3_MV_TOKENS 4
bool lfm3_step(Lfm3Forward *f, Lfm3Seq *s, const int32_t *tokens, uint32_t n, uint32_t rows, char err[OSMOS_ERR]);
const float *lfm3_logits(const Lfm3Forward *f);
void lfm3_kept(const Lfm3Forward *f, KeptView *k);
/* a row's logits (device, n_vocab floats) of the last step, NULL for a row not kept */
const float *lfm3_row_logits(const Lfm3Forward *f, uint32_t row);
/* the greatest token of each of rows row0 .. row0 + rows - 1 (the lowest among equals); false for a row not kept */
bool lfm3_greedy(Lfm3Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR]);

/* the prompt ends at end (its last token end - 1): rows before it not kept (the head not run for them) */
bool lfm3_seq_prompt(Lfm3Seq *s, uint32_t end);
/* the tokens a sequence can drop (LFM3_MAX_BACK at most; 0 after a load or a new one; never across an LSA block's
   close or a YaRN stage's change), and the drop: every state the bits of never having run them. false past the
   reach, unchanged */
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
bool lfm3_seq_save_to(const Lfm3Seq *s, StateSink sink, void *ctx, char err[OSMOS_ERR]);
bool lfm3_seq_load_from(Lfm3Seq *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR]);

/* DFlash2 (a forward made with Lfm3Options' dflash): seed the drafter's context at positions from .. to - 1 from the
   sequence's last step's taps, and draft up to max tokens after y (the token at the sequence's position) -
   q4x_dflash_seed / q4x_dflash_draft's contracts, the drafts capped where their verify would leave the reach
   (lfm3_seq_reach). neither moves a bit of the sequence's own state */
bool lfm3_dflash_seed(Lfm3Forward *f, Lfm3Seq *s, int32_t from, int32_t to, char err[OSMOS_ERR]);
int lfm3_dflash_draft(Lfm3Forward *f, Lfm3Seq *s, int32_t y, uint32_t max, float conf_min, float cum_min,
                      int32_t *drafts, float *conf, char err[OSMOS_ERR]);

/* the sequence's YaRN stage (0: plain rope) */
uint32_t lfm3_seq_stage(const Lfm3Seq *s);

/* a tap, when set, gets each layer's named outputs on the GPU, n x n_embd: "operator" (the conv's or attention's,
   before its post-norm) and "residual" (the layer's output) */
typedef void (*lfm3_tap_fn)(void *arg, int layer, const char *what, const float *dev, uint32_t n, uint32_t dim);
void lfm3_forward_tap(Lfm3Forward *f, lfm3_tap_fn fn, void *arg);

#endif
