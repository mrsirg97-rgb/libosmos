#ifndef OSMOS_HY4_FORWARD_H
#define OSMOS_HY4_FORWARD_H

/* Hy4-flash's forward on one GPU (DESIGN.md, "0.10.0 step 2"): the weights whole in its memory but the Engram table,
   whose rows come from the SSD a step at a time; a sequence's state its sparse layers' latent cache, token keys and
   pooled groups, its sliding layers' rings, its last tokens (the next step's n-grams). two maths, each a
   token's bits however its sequence is split into steps: a prompt's (hy4_step_read) the matrix cores', a decode's
   (hy4_step_runs) the matvecs' */

#include "osmos/err.h"
#include <stdbool.h>
#include <stdint.h>

#include "osmos/hy4/weights.h"
#include "osmos/model/state_io.h"

#define HY4_MAX_BACK 8                 /* the tokens a sequence can drop (a drafted round's) */
#define HY4_IX_SPREAD "the indexer's spread scratch" /* its allocation's name (osmos_hip: the fault seam's too) */
#define HY4_MAX_RUNS 64                /* a step's sequences at most */
#define HY4_EXPERT_READERS 12          /* the experts' membrane's readers: DeepSeek's and Inkling's (FWD_FETCH_THREADS) */

typedef struct Hy4Forward Hy4Forward;
typedef struct Hy4Seq Hy4Seq;

typedef struct Hy4Options
{
  uint32_t max_tokens; /* a step's tokens at most */
  uint32_t max_ctx;    /* a sequence's positions at most (its state sized for them) */
  uint32_t logit_rows; /* the logits' rows a step keeps at most (0: max_tokens) */
  bool kv_f32;         /* MLA's cache rows F32 (the reference's checks); else (0.11.0's default) the latent Q8_0 and
                          the rope part F32 - 800 bytes a row against 2,304 (hy4_mla_row_bytes) */
  bool head_q8;        /* a BF16 head read as its Q8_0 copy (ggml's rounding, made at load): half its reads */
  uint64_t budget;     /* the build's bytes, the routed experts' room what the rest leaves (experts_plan): streamed
                          through the membrane past it; 0 sized to this machine as it stands (experts_fit_now: every
                          expert resident where there is room) */
  struct OsmosLedger *ledger; /* the forward's, its sequences' and its drafter's every allocation counted in it
                                 (osmos/mem/ledger.h: borrowed - the engine's, outliving the forward); NULL nothing */
} Hy4Options;

/* the weights to the GPU (every tensor but the Engram table, read from `fds`' files as the step needs its rows: the
   model's open descriptors, by file), a step's buffers; NULL and err for a model refused, memory the GPU refuses */
Hy4Forward *hy4_forward_create(const Hy4Weights *w, const int *fds, const Hy4Options *o, char err[OSMOS_ERR]);
void hy4_forward_destroy(Hy4Forward *f);
/* the build's bytes (its fixed part - weights, buffers - and its experts' pinned host slots), and whether every expert
   is resident (decode's tables then the GPU's) */
uint64_t hy4_forward_bytes(const Hy4Forward *f);
bool hy4_forward_whole(const Hy4Forward *f);

Hy4Seq *hy4_seq_create(Hy4Forward *f, char err[OSMOS_ERR]);
void hy4_seq_destroy(Hy4Seq *s);
bool hy4_seq_reset(Hy4Seq *s, char err[OSMOS_ERR]); /* oseq_reset's contract (osmos/run/engine.h) */
int32_t hy4_seq_position(const Hy4Seq *s);

/* runs sequences' tokens in one step (seqs[i] the next lens[i] tokens, consecutive; each once): every matrix read once
   for all of them, each sequence's caches, keys, rings and n-grams its own - a token's bits the same however the
   sequences share the step or a sequence's tokens are split into steps. the kept rows' logits (hy4_row_logits): a
   row kept unless its token is before its sequence's prompt end - 1 (hy4_seq_prompt). false with why, every sequence
   unchanged, for a refusal before the work; false after it began (a GPU call or a read failed): the step's sequences
   refused from then on (hy4_seq_fail's), until a load */
bool hy4_step_runs(Hy4Forward *f, Hy4Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                   char err[OSMOS_ERR]);
/* the same step in a prompt's math (the engine's read): the matrices and the experts on the matrix cores (mm), its
   bits a prompt's however it is split - not a decode's (two maths, as DeepSeek's and q35's: DESIGN.md). false past
   it as hy4_step_runs, and for an activation past f16's range (the step refused) */
bool hy4_step_read(Hy4Forward *f, Hy4Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                   char err[OSMOS_ERR]);
/* one sequence, its last `rows` rows' logits on the host (hy4_logits: rows x n_vocab) - the checks' form: a step past
   HY4_MV_TOKENS tokens a prompt's math, the rest a decode's */
#define HY4_MV_TOKENS 4
bool hy4_step(Hy4Forward *f, Hy4Seq *s, const int32_t *tokens, uint32_t n, uint32_t rows, char err[OSMOS_ERR]);
const float *hy4_logits(const Hy4Forward *f);
/* row's logits (device, n_vocab floats) of the last hy4_step_runs, NULL for a row not kept */
const float *hy4_row_logits(const Hy4Forward *f, uint32_t row);
/* the greatest token of each of rows row0 .. row0 + rows - 1 (the lowest among equals); false for a row not kept */
bool hy4_greedy(Hy4Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR]);
/* a tap, when set: every layer's streams after it (n x hc x n_embd, the GPU's) - the checks against the reference */
typedef void (*hy4_tap_fn)(void *arg, int layer, const float *dev_h, uint32_t n, uint32_t hc, uint32_t dim);
void hy4_forward_tap(Hy4Forward *f, hy4_tap_fn fn, void *arg);

/* the prompt ends at end (its last token end - 1): rows before it not kept (the head not run for them) */
bool hy4_seq_prompt(Hy4Seq *s, uint32_t end);
/* the tokens a sequence can drop (HY4_MAX_BACK at most; 0 after a load or a new one), and the drop: every state as
   before them - the bits of never having run them (a group their keys completed pooled again when its keys land
   again). false past the reach, unchanged */
uint32_t hy4_seq_reach(const Hy4Seq *s);
bool hy4_seq_rollback(Hy4Seq *s, uint32_t n);
/* a sequence refused (every step, rollback, save) until a load: an engine's batch that failed past its step */
void hy4_seq_fail(Hy4Seq *s);
/* a 64-bit key of what a saved state depends on (its format, the geometry, the model's files) */
uint64_t hy4_identity(const Hy4Forward *f);
/* a sequence as one byte string (a header, every layer's rows below the position, the rings, the last tokens, then a
   sum) - load refuses another identity or format, a size not its position's, a damaged string: the sequence then
   empty */
/* DFly2, the release's drafter (0.11.0; DESIGN.md, "0.11.0: DFly2"): its five layers read from the release's own
   safetensors (model-00148-of-00148.safetensors, BF16) onto the GPU beside the forward; a draft block the anchor (the
   sampled token, at the sequence's position) and HY4_DFLY_DRAFTS mask tokens, each layer attending to its target
   sparse layer's keys for the last accepted token (that layer's indexer picks, kept by every step) and the block's
   own rows (written into that layer's cache past the position: a verify's step writes its own over them) - then the
   hidden correction's chain, each draft through the target's head. hy4_dfly_load: before any sequence drafts;
   false, err said, for a file not the drafter or not this model's shapes. q8: its matrices and the drafts' head
   (the target's, copied) Q8_0 (ggml's rounding of the BF16) - half the reads; a draft's precision moves how many
   are accepted, never the tokens. hy4_seq_draft: up to `most` drafts after
   y (HY4_DFLY_DRAFTS at most), each kept while its probability under the draft's own logits is at least conf_min and
   their product at least cum_min - the count; 0 when the last step kept no snapshot of the position before (a load,
   a prompt not run to its end) or the block would pass max_ctx; -1 with why on a GPU failure */
#define HY4_DFLY_DRAFTS 5
#define HY4_DFLY_LAYERS 5
bool hy4_dfly_load(Hy4Forward *f, const char *path, bool q8, char err[OSMOS_ERR]);
int hy4_seq_draft(Hy4Forward *f, Hy4Seq *s, int32_t y, uint32_t most, float conf_min, float cum_min, int32_t *drafts,
                  float *conf, char err[OSMOS_ERR]);

uint64_t hy4_seq_state_bytes(const Hy4Seq *s);
bool hy4_seq_save_to(const Hy4Seq *s, StateSink sink, void *ctx, char err[OSMOS_ERR]);
bool hy4_seq_load_from(Hy4Seq *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR]);

#endif
