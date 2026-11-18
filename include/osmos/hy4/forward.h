#ifndef OSMOS_HY4_FORWARD_H
#define OSMOS_HY4_FORWARD_H

/* Hy4-flash's forward on one GPU (DESIGN.md, "0.10.0 step 2"): the weights whole in its memory but the Engram table,
   whose rows come from the SSD a step at a time; a sequence's state its sparse layers' latent cache, token keys and
   pooled groups, its sliding layers' rings, its last tokens (the next step's n-grams). one math for prompts and
   decode: a token's bits the same however its sequence is split into steps */

#include <stdbool.h>
#include <stdint.h>

#include "osmos/hy4/weights.h"
#include "osmos/model/state_io.h"

#define HY4_MAX_BACK 8                 /* the tokens a sequence can drop (a drafted round's) */
#define HY4_MAX_RUNS 64                /* a step's sequences at most */

typedef struct Hy4Forward Hy4Forward;
typedef struct Hy4Seq Hy4Seq;

typedef struct Hy4Options
{
  uint32_t max_tokens; /* a step's tokens at most */
  uint32_t max_ctx;    /* a sequence's positions at most (its state sized for them) */
  uint32_t logit_rows; /* the logits' rows a step keeps at most (0: max_tokens) */
} Hy4Options;

/* the weights to the GPU (every tensor but the Engram table, read from `fds`' files as the step needs its rows: the
   model's open descriptors, by file), a step's buffers; NULL and err for a model refused, memory the GPU refuses */
Hy4Forward *hy4_forward_create(const Hy4Weights *w, const int *fds, const Hy4Options *o, char err[HY4_ERR]);
void hy4_forward_destroy(Hy4Forward *f);

Hy4Seq *hy4_seq_create(Hy4Forward *f, char err[HY4_ERR]);
void hy4_seq_destroy(Hy4Seq *s);
int32_t hy4_seq_position(const Hy4Seq *s);

/* runs sequences' tokens in one step (seqs[i] the next lens[i] tokens, consecutive; each once): every matrix read once
   for all of them, each sequence's caches, keys, rings and n-grams its own - a token's bits the same however the
   sequences share the step or a sequence's tokens are split into steps. the kept rows' logits (hy4_row_logits): a
   row kept unless its token is before its sequence's prompt end - 1 (hy4_seq_prompt). false with why, every sequence
   unchanged, for a refusal before the work; false after it began (a GPU call or a read failed): the step's sequences
   refused from then on (hy4_seq_fail's), until a load */
bool hy4_step_runs(Hy4Forward *f, Hy4Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
                   char err[HY4_ERR]);
/* one sequence, its last `rows` rows' logits on the host (hy4_logits: rows x n_vocab) - the checks' form */
bool hy4_step(Hy4Forward *f, Hy4Seq *s, const int32_t *tokens, uint32_t n, uint32_t rows, char err[HY4_ERR]);
const float *hy4_logits(const Hy4Forward *f);
/* row's logits (device, n_vocab floats) of the last hy4_step_runs, NULL for a row not kept */
const float *hy4_row_logits(const Hy4Forward *f, uint32_t row);
/* the greatest token of each of rows row0 .. row0 + rows - 1 (the lowest among equals); false for a row not kept */
bool hy4_greedy(Hy4Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[HY4_ERR]);
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
uint64_t hy4_seq_state_bytes(const Hy4Seq *s);
bool hy4_seq_save_to(const Hy4Seq *s, StateSink sink, void *ctx, char err[HY4_ERR]);
bool hy4_seq_load_from(Hy4Seq *s, uint64_t bytes, StateSource src, void *ctx, char err[HY4_ERR]);

#endif
