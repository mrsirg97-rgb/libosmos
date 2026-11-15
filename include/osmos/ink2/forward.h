#ifndef OSMOS_INK2_FORWARD_H
#define OSMOS_INK2_FORWARD_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/ink2/weights.h"
#include "osmos/model/state_io.h"

#define INK2_FWD_ERR 256
/* a sequence's tokens a rollback can drop: a drafted round's (OSMOS_MAX_BLOCK) - its windows and conv histories keep
   that many rows past what the next token reads */
#define INK2_MAX_BACK 8
#define INK2_MAX_RUNS 64 /* a step's sequences at most (ink2_step) */
#define INK2_EXPERT_READERS 12 /* the experts' membrane's readers: DeepSeek's (FWD_FETCH_THREADS) */

typedef struct Ink2Forward Ink2Forward;
typedef struct Ink2Seq Ink2Seq;

typedef struct Ink2Options
{
  uint32_t max_ctx;    /* a sequence's most positions */
  uint32_t max_tokens; /* a step's most tokens */
  uint64_t room;       /* the experts' cache (the membrane's pool), bytes; 0: every expert */
  const uint64_t *picks; /* a picks profile ([layer][expert] counts): the warm set the most picked; NULL: uniform */
} Ink2Options;

Ink2Forward *ink2_forward_create(const Ink2Weights *w, const Ink2Options *o, char err[INK2_FWD_ERR]);
void ink2_forward_destroy(Ink2Forward *f);
uint64_t ink2_forward_bytes(const Ink2Forward *f);
const uint64_t *ink2_forward_picks(const Ink2Forward *f);
uint64_t ink2_identity(const Ink2Forward *f);

Ink2Seq *ink2_seq_create(Ink2Forward *f, char err[INK2_FWD_ERR]);
void ink2_seq_destroy(Ink2Seq *s);
int32_t ink2_seq_position(const Ink2Seq *s);
uint32_t ink2_seq_reach(const Ink2Seq *s);
bool ink2_seq_rollback(Ink2Seq *s, uint32_t n);
bool ink2_seq_prompt(Ink2Seq *s, uint32_t end);
void ink2_seq_fail(Ink2Seq *s);
uint64_t ink2_seq_state_bytes(const Ink2Seq *s);
bool ink2_seq_save_to(const Ink2Seq *s, StateSink sink, void *ctx, char err[INK2_FWD_ERR]);
bool ink2_seq_load_from(Ink2Seq *s, uint64_t bytes, StateSource src, void *ctx, char err[INK2_FWD_ERR]);
uint64_t ink2_seq_mark_bytes(const Ink2Seq *s);
bool ink2_seq_mark_save(const Ink2Seq *s, void *out, uint64_t bytes, char err[INK2_FWD_ERR]);
bool ink2_seq_mark_load(Ink2Seq *s, const void *in, uint64_t bytes, char err[INK2_FWD_ERR]);

bool ink2_step(Ink2Forward *f, Ink2Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
               char err[INK2_FWD_ERR]);
const float *ink2_logits(const Ink2Forward *f);
bool ink2_kept(const Ink2Forward *f, uint32_t row);
bool ink2_greedy(Ink2Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[INK2_FWD_ERR]);

#endif

/*
Ink2Forward (DESIGN.md, "0.8.0: Inkling's engine")
  Inkling-2-Small's forward with a cache: tools/bench/ink2_run's math (transformers' modeling_inkling2.py, held to its
  oracle) a step's tokens at a time - every weight but the routed experts resident (the attention's, the router's, the
  convs', the embedding and head: ~6 GB), the experts a membrane's units (osmos/mem/membrane.h) read from the model's
  files O_DIRECT into pinned host memory the GPU reads in place, `room` of them kept. the attention ink2_fa's (WMMA).
  a token's bits are the same however its sequence's tokens are split into steps, and ink2_run's (ATTN=fa) for the
  same tokens.

ink2_forward_create
  the dense weights uploaded, the membrane made and warmed (room's units: given picks the most picked first, else an
  expert of every layer at a time), the step's buffers for max_tokens. NULL with why where memory or a read refuses,
  or max_ctx / max_tokens is 0. ink2_forward_bytes: the GPU and pinned memory it took (the server's budget).
  ink2_forward_picks: every unit's picks since create ([layer][expert], uint64: a picks file's layout).

ink2_identity
  a 64-bit key of what a saved sequence depends on: the state's format, the geometry, max_ctx's place in it (none:
  a full layer's rows are its position's), the model's files (each one's name and size).

ink2_seq_create .. ink2_seq_position
  a sequence: each full layer's keys and values (max_ctx rows, F32, after their conv, clip and norm), each sliding
  layer's last window - 1 + INK2_MAX_BACK of them, each layer's four short convs' inputs (k, v, the attention's output,
  the MoE's: conv - 1 + INK2_MAX_BACK rows; zeros before the first token). its position the tokens it holds.

ink2_seq_reach, ink2_seq_rollback
  how many of its last tokens a sequence can drop (INK2_MAX_BACK at most, its position at most) - and the drop, the
  bits of never having run them: nothing of Inkling's state is recurrent, each part the rows of its tokens. false
  for n past the reach.

ink2_seq_prompt, ink2_kept
  the prompt ends at position end: a step leaves the rows of its tokens before end - 1 unwritten (the head not run
  for them: 201,024 columns), the rest written as ever. not saved; cleared by a load. kept: whether row's logits are
  the last step's.

ink2_seq_fail
  the sequence refused (every step, rollback, save) until a load: a caller's batch failed past its own step.

ink2_seq_state_bytes .. ink2_seq_load_from
  a sequence as one byte string: a header (format, identity, position), each layer's conv tails, each sliding
  layer's window tail, each full layer's rows below the position, an FNV-1a of all before it. load refuses another
  identity, a position past max_ctx, a size not its position's, a damaged state - the sequence then empty.

ink2_seq_mark_bytes .. ink2_seq_mark_load
  a point the sequence goes back to (osmos-serve's prompt end, 0.7.5): the position, the conv tails and the sliding
  windows - not the full layers' rows below the position, which only grow. load refuses a mark past the position or
  of another size; the caller keeps the tokens.

ink2_step
  sequences' tokens (seqs[i] the next lens[i] tokens, each sequence once, 1 .. INK2_MAX_RUNS runs, their tokens at
  most max_tokens), each run through the model in turn: its logits at its rows in the tokens' order (ink2_logits:
  [max_tokens][n_vocab] on the device, softcapped). false with why - the sequences unchanged - for a sequence twice,
  of another forward or failed, or past max_ctx; false after work began: the step's sequences failed.

ink2_greedy
  rows row0 .. row0 + rows - 1's greatest token (the lowest among equals). false for a row not kept.
*/
