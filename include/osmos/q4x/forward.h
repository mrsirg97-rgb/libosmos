#ifndef OSMOS_Q4X_FORWARD_H
#define OSMOS_Q4X_FORWARD_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/model/state_io.h"
#include "osmos/q4x/weights.h"

#define Q4X_FWD_ERR 256
/* a sequence's states kept behind its latest: a rollback's reach - a drafter's most drafts a round (OSMOS_MAX_BLOCK,
   the engine's), each a DeltaNet layer's S (3.1 MB) a slot: ~1 GB a sequence at 8 */
#define Q4X_MAX_BACK 8
/* the MTP head's most drafts a round: measured 2-7 on mm (q4x_bench, 192 tokens: 33.6 34.1 35.2 35.7 34.6 34.8 tok/s;
   the engine's confidence floors end most rounds before 6) */
#define Q4X_MTP_BLOCK 5
/* the MTP head's draft scored over the vocabulary's first ids alone (BPE's order: the frequent first - 97.8% of a
   text's tokens below it): its output projection 521 -> 137 MB a draft, drafted decode 35.7 -> 40.5 tok/s, the
   acceptance the same (q4x_bench, 192 tokens; 32,768: 39.3, 100,000: 38.5). a draft is a proposal: the answer's bits
   are the verify step's, whatever the drafts */
#define Q4X_MTP_VOCAB 65536
#define Q4X_MAX_TAPS 8 /* Q4xOptions' tap layers at most (DFlash2's 5) */

typedef struct Q4xForward Q4xForward;
typedef struct Q4xSeq Q4xSeq;
typedef struct Q4xDflash Q4xDflash; /* osmos/q4x/dflash.h */

typedef void (*q4x_probe_fn)(const char *step, int layer, float *dev, uint64_t n, void *arg);

typedef struct Q4xOptions
{
  uint32_t max_ctx;    /* a sequence's most positions */
  uint32_t max_tokens; /* a step's most tokens */
  bool stream_ngram;   /* the n-gram table read from its file a step's rows at a time, not held in GPU memory */
  const int32_t *tap_layers; /* layers whose output a step keeps (q4x_taps: a drafter's features), ascending */
  uint32_t n_taps;           /* how many (Q4X_MAX_TAPS at most); 0: none */
  const Q4xDflash *dflash;   /* a DFlash2 drafter (q4x_dflash_*), its taps the step's; NULL: none */
} Q4xOptions;

Q4xForward *q4x_forward_create(const Q4xWeights *w, const Q4xMtp *mtp, uint32_t max_ctx, uint32_t max_tokens,
                               char err[Q4X_FWD_ERR]);
Q4xForward *q4x_forward_create_with(const Q4xWeights *w, const Q4xMtp *mtp, const Q4xOptions *o, char err[Q4X_FWD_ERR]);
void q4x_forward_destroy(Q4xForward *f);
void q4x_forward_probe(Q4xForward *f, q4x_probe_fn fn, void *arg);
bool q4x_forward_whole(const Q4xForward *f);
void q4x_forward_side_spin(Q4xForward *f, uint32_t us);
/* 0.7.0: the NPU beside the GPU (src/q4x/hip_npu.c): dir its kernels (npu_open's: iq3xxs/, iq2s/, iq4nl-down/), share
   the part of an eligible layer's prompt picks it takes (whole 128-token entries; an IQ2_S layer's scaled by its
   cost). another forward's bits: q4x_identity tells them apart */
bool q4x_forward_npu(Q4xForward *f, const char *dir, float share, char err[Q4X_FWD_ERR]);
/* npu_run's calls, launches and the host's time in them; the parts wanted, and those the tables' whole entries gave */
void q4x_forward_npu_stats(const Q4xForward *f, uint64_t *calls, uint64_t *parts, double *wait_us, uint64_t *wanted,
                           uint64_t *taken);

Q4xSeq *q4x_seq_create(Q4xForward *f, char err[Q4X_FWD_ERR]);
void q4x_seq_destroy(Q4xSeq *s);
int32_t q4x_seq_position(const Q4xSeq *s);
uint32_t q4x_seq_reach(const Q4xSeq *s);
bool q4x_seq_rollback(Q4xSeq *s, uint32_t n);
bool q4x_seq_prompt(Q4xSeq *s, uint32_t end);
void q4x_seq_fail(Q4xSeq *s);
uint64_t q4x_identity(const Q4xForward *f);
uint64_t q4x_seq_state_bytes(const Q4xSeq *s);
bool q4x_seq_save_to(const Q4xSeq *s, StateSink sink, void *ctx, char err[Q4X_FWD_ERR]);
bool q4x_seq_load_from(Q4xSeq *s, uint64_t bytes, StateSource src, void *ctx, char err[Q4X_FWD_ERR]);
uint64_t q4x_seq_mark_bytes(const Q4xSeq *s);
bool q4x_seq_mark_save(const Q4xSeq *s, void *out, uint64_t bytes, char err[Q4X_FWD_ERR]);
bool q4x_seq_mark_load(Q4xSeq *s, const void *in, uint64_t bytes, char err[Q4X_FWD_ERR]);

#define Q4X_MAX_RUNS 64 /* a step's sequences at most (q4x_step) */
bool q4x_step(Q4xForward *f, Q4xSeq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
              uint32_t row0, char err[Q4X_FWD_ERR]);
bool q4x_batch(Q4xForward *f, Q4xSeq *s, const int32_t *tokens, uint32_t n, uint32_t row0, char err[Q4X_FWD_ERR]);
const float *q4x_logits(const Q4xForward *f);
bool q4x_kept(const Q4xForward *f, uint32_t row);
const float *q4x_taps(const Q4xForward *f);
bool q4x_greedy(Q4xForward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[Q4X_FWD_ERR]);

bool q4x_mtp_seed(Q4xForward *f, Q4xSeq *s, int32_t from, int32_t to, char err[Q4X_FWD_ERR]);
int q4x_mtp_draft(Q4xForward *f, Q4xSeq *s, int32_t y, uint32_t max, float conf_min, float cum_min, int32_t *drafts,
                  float *conf, char err[Q4X_FWD_ERR]);
bool q4x_mtp_draft_all(Q4xForward *f, Q4xSeq *const *s, const int32_t *y, uint32_t n, uint32_t max, float conf_min,
                       float cum_min, int32_t *drafts, float *conf, uint32_t stride, int *counts,
                       char err[Q4X_FWD_ERR]);

bool q4x_dflash_seed(Q4xForward *f, Q4xSeq *s, int32_t from, int32_t to, char err[Q4X_FWD_ERR]);
int q4x_dflash_draft(Q4xForward *f, Q4xSeq *s, int32_t y, uint32_t max, float conf_min, float cum_min, int32_t *drafts,
                     float *conf, char err[Q4X_FWD_ERR]);

#endif

/*
Q4xForward
  Qwen3.8-Flash-Next's forward on the GPU (DESIGN.md, "Qwen's forward, op by op"): every weight resident (the file's
  bytes, ~78 GB), its steps the kernels q4x_ops.h names, F32 between them. the attention dense over every position
  below OPS_Q4X_MAX_KEYS (llama.cpp's QSA keeps every token below its 2,048-token budget, a selection past it: dense
  past it measured its equal - 2.7136 over 8,192 tokens, QSA's 2.7193 - the selection not built).

q4x_forward_create, q4x_forward_create_with
  the weights uploaded and the step's buffers for max_tokens tokens; NULL with why where the GPU's memory or a read
  refuses. max_ctx: a sequence's most positions (at most OPS_Q4X_MAX_KEYS). create is create_with's held table
  (stream_ngram false). stream_ngram: the n-gram layer's table (per_layer_token_embd, 28.8 GB of UD-IQ3_XXS) stays in
  its file - a step's rows (its tokens' ple_heads each, ~1.4 KB a token) read at its start by rowread's workers into a
  pinned stage while the GPU runs the layers before the n-gram layer, copied in and gathered as the held table's: the
  same bits; its bytes of GPU memory left to the system. tap_layers: each layer's output (its residual's hc streams'
  mean, q4x_tap) kept for every token of a step (q4x_taps); NULL with why for a layer past the model's, out of order,
  or more than Q4X_MAX_TAPS. taking them changes no bit of the step's logits. dflash: the drafter's buffers, its
  weights uploaded (~0.6 GB at Q8_0), each sequence its rings; its taps the step's (tap_layers then NULL, or the
  drafter's own); NULL with why for a drafter not of these weights (q4x_dflash_create's) or its codebooks not Q8_0.

q4x_taps
  the last step's taps on the device, [token][n_taps][n_embd] F32 (the step's tokens in its order); NULL for a forward
  made with none.

q4x_forward_side_spin
  a test's: each small step's shared expert (the second stream's: q4x_moe_shared) after ~us of that stream's time -
  so the routed experts end first, and anything that read the shared expert without waiting would read it stale.
  0 (the default): none.

q4x_forward_probe
  fn sees each named step's output (the dump's names: l_last a layer, result_norm) on the device, n floats,
  synchronized: a test's window - and may write it, the step going on from what it wrote (a reference's values stood
  in: a layer's error alone). NULL stops it.

q4x_seq_create .. q4x_seq_position
  a sequence: its DeltaNet states and conv histories, the n-gram layer's conv history, its attention caches (max_ctx
  rows, Q8_0: q4x_kv_put), its last tokens; its position the tokens it holds.

q4x_seq_fail
  the sequence refused as after a failed step (q4x_step) - every step, rollback, save and draft until a load: for a
  caller whose batch failed past the sequence's own step (the engine's batch of more than Q4X_MAX_RUNS sequences, run
  as several steps), so its tokens are never fed twice.

q4x_seq_reach, q4x_seq_rollback
  how many of its last tokens a sequence can drop (its states kept behind: Q4X_MAX_BACK at most, 0 when new or just
  loaded) - and the drop: its last n tokens gone, every state as it was before them (the bits of never having run
  them). false for n past the reach, the sequence unchanged.

q4x_identity
  a 64-bit key of what a saved sequence depends on: the state's format, the geometry, the model's files (each one's
  name and size) - a state saved under another is refused.

q4x_seq_state_bytes .. q4x_seq_load_from
  a sequence as one byte string: a header (format, identity, position, last tokens), the n-gram conv history, each
  DeltaNet layer's state and conv history, each attention layer's cache rows below the position, then a 64-bit
  FNV-1a of everything before it - its bytes, save (to a sink in pieces of at most STATE_CHUNK: osmos/model/
  state_io.h), load (from a source, the same). load refuses another identity or format, a size not the state's of
  its position, a position past max_ctx, a damaged state (the checksum read last, the parts sent as they come): the
  sequence then empty, position 0 (never part one state and part another).

q4x_seq_mark_bytes .. q4x_seq_mark_load
  a point the sequence goes back to (0.7.5: a conversation's prompt end, osmos-serve's): what a later token changes -
  the position and last tokens, the n-gram conv history, each DeltaNet layer's state and conv history, each attention
  layer's raw keys' ring - not the attention rows and pooled keys below the position, which only grow. so a mark is
  the sequence's own while it holds the same tokens below the mark's position (no load, reset or rollback under it
  since): load refuses a mark past the position or of another sequence's size, a broken sequence; the caller keeps
  the tokens. ~118 MB for Qwen3.8-Flash-Next, whatever the position (a whole state: 2.9 GB at 190K).

q4x_step
  runs sequences' tokens in one step (seqs[i] the next lens[i] tokens, consecutive; each sequence once, runs 1 ..
  Q4X_MAX_RUNS): every weight read once for all of them (the matmuls and the experts on the step's tokens; each
  sequence's own - its n-gram history, conv histories, DeltaNet state, rotation, cache and attention - on its tokens
  alone), the logits at rows row0 .. in the tokens' order (a sequence told its prompt's end: q4x_seq_prompt's rows
  alone). each sequence as a q4x_batch of its tokens: the same bits,
  however the sequences share the step. false with why - every sequence then unchanged in position - for no runs, a
  run of no tokens, a sequence twice or of another forward, the tokens past max_tokens or a sequence past max_ctx.
  false after its work began (a GPU call failed, an activation past f16's range): its sequences' DeltaNet and n-gram
  rings part written, so each is refused from then on - every step, rollback, save and draft - until a state is
  loaded into it (or another sequence is made), never read on from a half-written state.

q4x_batch
  q4x_step of one sequence: n tokens (1 .. max_tokens) of it at its next positions; the logits of every one of them after it, rows
  row0 .. row0 + n - 1 of q4x_logits ([max_tokens][vocab] on the device: the rows outside left as they were - several
  sequences' steps one after another fill one table, row0 + n at most max_tokens). a token's results are the same bits
  however its sequence's tokens are split into batches. false with why for n out of range or past max_ctx; the
  sequence then unchanged in position (its state not to be read on: a refusal before any step runs).

q4x_forward_whole (DESIGN.md, "0.6.5: Qwen on the membrane")
  the forward's experts are a membrane's units, every one in memory and none to be evicted (membrane_whole): read
  from the model's files O_DIRECT into memory the GPU reads in place, their addresses a table on the device the MoE's
  entries are made from (q4x_expert_tiles' tab) - written once, a step asks the membrane nothing. false for a
  forward whose experts lie in their tensors as uploaded: a model an overlay's expert tensor stands in for.

q4x_seq_prompt, q4x_kept
  q4x_seq_prompt: the sequence's prompt ends at position end (its last token end - 1; 0, a new sequence's: none).
  the caller reads no logits before end - 1: from then on a step leaves its tokens' rows at positions before end - 1
  as they were (the output head not run for them - a server reads a prompt's last row alone), and writes those at
  end - 1 and later, the bits of a run never told (DESIGN.md, "0.6.3"). nothing of the sequence's state reads it: any
  end up to max_ctx is taken, lower than the last too; not saved, cleared by a load or a refused one. false for no
  sequence or an end past max_ctx.
  q4x_kept: whether row's logits are its last step's - false for a row whose token was before its prompt's end - 1
  (or past max_tokens, or of no step yet).

q4x_greedy
  rows row0 .. row0 + rows - 1's greatest token each (ops_argmax's: the lowest among equals). false for a row not
  kept (q4x_kept).

q4x_mtp_seed, q4x_mtp_draft (a forward built with an MTP head; DESIGN.md, "Qwen's MTP head")
  seed: the head's cache at positions from .. to - 1, of the sequence's last step (its rows, read before another
  step): position p's row from its token and the main model's h of p - 1 (the step's row before it, or the pending h -
  zeros when none is held: a load, a first step) - only what the cache reads (eh_proj, the attention's mixer, k and
  v); the pending h then position to - 1's. draft: up to max tokens after y (the token at the sequence's position),
  greedily: draft i the head's greatest among the first Q4X_MTP_VOCAB tokens at position pos + i from token i (y, then
  draft i - 1) and h (the pending, then the head's own residual), each kept while its probability (q4x_top's) is at
  least conf_min and their product at least cum_min - the count, or -1 with why. neither moves a bit of the sequence's
  own state. draft_all: n sequences' (1 .. min(max_tokens, Q4X_MAX_RUNS), each once), y[i] sequence i's, its drafts
  and confidences at row i (stride apart), its count counts[i] - each the bits of its q4x_mtp_draft: a draft's depth
  one head step for the sequences still drafting (the head's weights read once for all). false with why.

q4x_dflash_seed, q4x_dflash_draft (a forward built with Q4xOptions' dflash; DESIGN.md, "DFlash2: a block drafter
  for Qwen")
  seed: the drafter's context at positions from .. to - 1 from the sequence's last step's taps (its rows of that
  step, below the sequence's position: a rollback's dropped tokens refused) - each through the drafter's fc and hidden
  norm, then each layer's k (normed, rotated) and v into the sequence's rings (the window's rows, position p at
  p % rows). the context is the positions seeded since the last gap: a seed past where the context reaches starts it
  over at from (a prompt's window, a load - which keeps none - or a rollback below it), one before it seeds those
  positions again and the context ends at to. draft: a block at the sequence's position - y (the token there) and the
  mask token after it - through the drafter's layers (attention over the context within its window, and the block's
  own, not causal), Qwen's output head at the block's places after y, the selector's greedy path from y: up to
  min(max, block - 1) drafts, each kept while its confidence (the head's probability of it there) is at least
  conf_min and their product at least cum_min - the count, or -1 with why. the context must reach the sequence's
  position. neither moves a bit of the sequence's own state; a draft is a proposal (the verify step's bits are the
  answer's).
*/
