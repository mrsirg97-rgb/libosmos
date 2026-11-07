#ifndef OSMOS_Q4X_FORWARD_H
#define OSMOS_Q4X_FORWARD_H

#include <stdbool.h>
#include <stdint.h>

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

typedef struct Q4xForward Q4xForward;
typedef struct Q4xSeq Q4xSeq;

typedef void (*q4x_probe_fn)(const char *step, int layer, float *dev, uint64_t n, void *arg);

Q4xForward *q4x_forward_create(const Q4xWeights *w, const Q4xMtp *mtp, uint32_t max_ctx, uint32_t max_tokens,
                               char err[Q4X_FWD_ERR]);
void q4x_forward_destroy(Q4xForward *f);
void q4x_forward_probe(Q4xForward *f, q4x_probe_fn fn, void *arg);
void q4x_forward_side_spin(Q4xForward *f, uint32_t us);

Q4xSeq *q4x_seq_create(Q4xForward *f, char err[Q4X_FWD_ERR]);
void q4x_seq_destroy(Q4xSeq *s);
int32_t q4x_seq_position(const Q4xSeq *s);
uint32_t q4x_seq_reach(const Q4xSeq *s);
bool q4x_seq_rollback(Q4xSeq *s, uint32_t n);
uint64_t q4x_identity(const Q4xForward *f);
uint64_t q4x_seq_state_bytes(const Q4xSeq *s);
bool q4x_seq_save(const Q4xSeq *s, void *out, uint64_t bytes, char err[Q4X_FWD_ERR]);
bool q4x_seq_load(Q4xSeq *s, const void *in, uint64_t bytes, char err[Q4X_FWD_ERR]);

#define Q4X_MAX_RUNS 64 /* a step's sequences at most (q4x_step) */
bool q4x_step(Q4xForward *f, Q4xSeq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
              uint32_t row0, char err[Q4X_FWD_ERR]);
bool q4x_batch(Q4xForward *f, Q4xSeq *s, const int32_t *tokens, uint32_t n, uint32_t row0, char err[Q4X_FWD_ERR]);
const float *q4x_logits(const Q4xForward *f);
bool q4x_greedy(Q4xForward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[Q4X_FWD_ERR]);

bool q4x_mtp_seed(Q4xForward *f, Q4xSeq *s, int32_t from, int32_t to, char err[Q4X_FWD_ERR]);
int q4x_mtp_draft(Q4xForward *f, Q4xSeq *s, int32_t y, uint32_t max, float conf_min, float cum_min, int32_t *drafts,
                  float *conf, char err[Q4X_FWD_ERR]);
bool q4x_mtp_draft_all(Q4xForward *f, Q4xSeq *const *s, const int32_t *y, uint32_t n, uint32_t max, float conf_min,
                       float cum_min, int32_t *drafts, float *conf, uint32_t stride, int *counts,
                       char err[Q4X_FWD_ERR]);

#endif

/*
Q4xForward
  Qwen3.8-Flash-Next's forward on the GPU (DESIGN.md, "Qwen's forward, op by op"): every weight resident (the file's
  bytes, ~78 GB), its steps the kernels q4x_ops.h names, F32 between them. the attention dense over every position
  below OPS_Q4X_MAX_KEYS (llama.cpp's QSA keeps every token below its 2,048-token budget, a selection past it: dense
  past it measured its equal - 2.7136 over 8,192 tokens, QSA's 2.7193 - the selection not built).

q4x_forward_create
  the weights uploaded and the step's buffers for max_tokens tokens; NULL with why where the GPU's memory or a read
  refuses. max_ctx: a sequence's most positions (at most OPS_Q4X_MAX_KEYS).

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

q4x_seq_reach, q4x_seq_rollback
  how many of its last tokens a sequence can drop (its states kept behind: Q4X_MAX_BACK at most, 0 when new or just
  loaded) - and the drop: its last n tokens gone, every state as it was before them (the bits of never having run
  them). false for n past the reach, the sequence unchanged.

q4x_identity
  a 64-bit key of what a saved sequence depends on: the state's format, the geometry, the model's files (each one's
  name and size) - a state saved under another is refused.

q4x_seq_state_bytes .. q4x_seq_load
  a sequence as one byte string: a header (format, identity, position, last tokens), the n-gram conv history, each
  DeltaNet layer's state and conv history, each attention layer's cache rows below the position, then a 64-bit
  FNV-1a of everything before it - its bytes, save (into exactly that many), load. load refuses another identity
  or format, a size not the state's of its position, a position past max_ctx, a damaged state: the sequence then
  empty, position 0 (never part one state and part another).

q4x_step
  runs sequences' tokens in one step (seqs[i] the next lens[i] tokens, consecutive; each sequence once, runs 1 ..
  Q4X_MAX_RUNS): every weight read once for all of them (the matmuls and the experts on the step's tokens; each
  sequence's own - its n-gram history, conv histories, DeltaNet state, rotation, cache and attention - on its tokens
  alone), the logits at rows row0 .. in the tokens' order. each sequence as a q4x_batch of its tokens: the same bits,
  however the sequences share the step. false with why - every sequence then unchanged in position - for no runs, a
  run of no tokens, a sequence twice or of another forward, the tokens past max_tokens or a sequence past max_ctx.

q4x_batch
  q4x_step of one sequence: n tokens (1 .. max_tokens) of it at its next positions; the logits of every one of them after it, rows
  row0 .. row0 + n - 1 of q4x_logits ([max_tokens][vocab] on the device: the rows outside left as they were - several
  sequences' steps one after another fill one table, row0 + n at most max_tokens). a token's results are the same bits
  however its sequence's tokens are split into batches. false with why for n out of range or past max_ctx; the
  sequence then unchanged in position (its state not to be read on: a refusal before any step runs).

q4x_greedy
  rows row0 .. row0 + rows - 1's greatest token each (ops_argmax's: the lowest among equals).

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
*/
