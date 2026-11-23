#ifndef OSMOS_INK2_FORWARD_H
#define OSMOS_INK2_FORWARD_H

#include "osmos/err.h"
#include <stdbool.h>
#include <stdint.h>

#include "osmos/ink2/mtp.h"
#include "osmos/ink2/weights.h"
#include "osmos/gpu/kept.h"
#include "osmos/model/state_io.h"

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
  uint64_t room;       /* the experts' cache (the membrane's pool), bytes; 0: the least (a layer's) */
  const uint64_t *picks; /* a picks profile ([layer][expert] counts): the warm set the most picked; NULL: uniform */
  const Ink2Mtp *mtp;    /* the drafter (osmos/ink2/mtp.h), or NULL: none */
  uint64_t budget;       /* the whole build's bytes (osmos's budget): the experts' room what the dense weights, the
                            step's buffers, the drafter and the stream leave of it (room then unread); 0: room's */
  uint32_t logit_rows;   /* the logits' rows a step keeps at most, slots in the order computed (a prompt's last row,
                            a decode's or a drafted round's every row, each lane's); 0: max_tokens (every row) */
  bool values_q8;        /* a full layer's cached values Q8 records (ink2_fa8), not Q5 (ink2_fa85): a value's bytes
                            its key's - for a model with the room (osmos-serve's --kv-q8) */
  struct OsmosLedger *ledger; /* the forward's and its sequences' every allocation counted in it (osmos/mem/ledger.h:
                                 borrowed - the engine's, outliving the forward); NULL counts nothing */
  bool profile;          /* a step's sections timed, each after a synchronize (its bits the same), their totals said
                            at its destroy (stderr) - a measurement's build (0.16.0: INK2_PROFILE) */
} Ink2Options;

Ink2Forward *ink2_forward_create(const Ink2Weights *w, const Ink2Options *o, char err[OSMOS_ERR]);
void ink2_forward_destroy(Ink2Forward *f);
uint64_t ink2_forward_bytes(const Ink2Forward *f);
const uint64_t *ink2_forward_picks(const Ink2Forward *f);
/* every expert in memory (the experts' membrane whole): the MoE's tables built on the GPU, the picks never on the host -
   the same bits as the host's tables */
bool ink2_forward_whole(const Ink2Forward *f);
uint64_t ink2_identity(const Ink2Forward *f);

Ink2Seq *ink2_seq_create(Ink2Forward *f, char err[OSMOS_ERR]);
void ink2_seq_destroy(Ink2Seq *s);
bool ink2_seq_reset(Ink2Seq *s, char err[OSMOS_ERR]); /* oseq_reset's contract (osmos/run/engine.h) */
int32_t ink2_seq_position(const Ink2Seq *s);
uint32_t ink2_seq_reach(const Ink2Seq *s);
bool ink2_seq_rollback(Ink2Seq *s, uint32_t n);
bool ink2_seq_prompt(Ink2Seq *s, uint32_t end);
void ink2_seq_fail(Ink2Seq *s);
uint64_t ink2_seq_state_bytes(const Ink2Seq *s);
bool ink2_seq_save_to(const Ink2Seq *s, StateSink sink, void *ctx, char err[OSMOS_ERR]);
bool ink2_seq_load_from(Ink2Seq *s, uint64_t bytes, StateSource src, void *ctx, char err[OSMOS_ERR]);
uint64_t ink2_seq_mark_bytes(const Ink2Seq *s);
bool ink2_seq_mark_save(const Ink2Seq *s, void *out, uint64_t bytes, char err[OSMOS_ERR]);
bool ink2_seq_mark_load(Ink2Seq *s, const void *in, uint64_t bytes, char err[OSMOS_ERR]);

bool ink2_step(Ink2Forward *f, Ink2Seq *const *seqs, const uint32_t *lens, uint32_t runs, const int32_t *tokens,
               char err[OSMOS_ERR]);
const float *ink2_logits(const Ink2Forward *f);

/* the per-slot checks' (oracle/qwen_dump.cpp's l_out-<slot>, upstream llama.cpp's tensors): a tap, when set, gets
   each cache slot's output - "l_out", the layer's stream after its MoE (its attn_res block's partial: a block's first
   layer's its own outputs alone), n x n_embd, the GPU's - a step's run of rows at a time */
typedef void (*ink2_tap_fn)(void *arg, int slot, const char *what, const float *dev, uint32_t n, uint32_t dim);
void ink2_forward_tap(Ink2Forward *f, ink2_tap_fn fn, void *arg);
/* the last step's kept rows (osmos/gpu/kept.h): the engine reads and picks a row through it */
void ink2_kept_view(const Ink2Forward *f, KeptView *k);
const float *ink2_row_logits(const Ink2Forward *f, uint32_t row);
bool ink2_kept(const Ink2Forward *f, uint32_t row);
bool ink2_greedy(Ink2Forward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[OSMOS_ERR]);

bool ink2_seq_seed(Ink2Forward *f, Ink2Seq *s, int32_t from, int32_t to, char err[OSMOS_ERR]);
int ink2_seq_draft(Ink2Forward *f, Ink2Seq *s, int32_t y, uint32_t max, float conf_min, float cum_min, int32_t *drafts,
                   float *conf, char err[OSMOS_ERR]);

#endif

/*
Ink2Forward (DESIGN.md, "0.8.0: Inkling's engine")
  Inkling-2-Small's forward with a cache: tools/bench/ink2_run's math (transformers' modeling_inkling2.py, held to its
  oracle) a step's tokens at a time - every weight but the routed experts resident (the attention's, the router's, the
  convs', the embedding and head: ~6 GB), the experts a membrane's units (osmos/mem/membrane.h) read from the model's
  files O_DIRECT into pinned host memory the GPU reads in place, `room` of them kept. the attention ink2_fa's, the
  matrices mm's where it has their type (WMMA, f16 in: the experts and the quantized dense) and the matvec's for the
  rest (F32, BF16) - at every step size, so a token's bits are the same however its sequence's tokens are split into
  steps, and ink2_run's (ATTN=fa MAT=mm) for the same tokens. a step's activation past f16's range refuses it.

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
  most max_tokens), each run through the model in turn: its kept rows' logits (a run's rows from its prompt's end - 1)
  in slots in the order computed (ink2_row_logits: a row's, on the device, softcapped; ink2_logits the slots'
  [logit_rows][n_vocab] - with logit_rows 0 and every row kept, row r's slot r). false with why - the sequences unchanged - for a sequence twice,
  of another forward or failed, or past max_ctx; false after work began: the step's sequences failed.

ink2_greedy
  rows row0 .. row0 + rows - 1's greatest token (the lowest among equals). false for a row not kept, or rows not
  one run's (their slots not consecutive).

ink2_seq_seed, ink2_seq_draft (a forward built with a drafter: DSpark with the RNN Markov head, src/ink2/hip_mtp.c)
  seed: the drafter's context at positions from .. to - 1 from the sequence's last step's taps (its rows of that
  step) - each drafter layer's mixture of them normed, its ctx keys and values into the sequence's rings. the
  context is the positions seeded since the last gap (a seed past where it reaches starts it over; a load or a
  rollback below it, its end). draft: a block at the sequence's position - y (the token there, its normed embedding)
  and mask rows - through the drafter, the target's head on its rows after y, the Markov steps' picks: up to max
  (block - 1 at most) drafts, each kept while its probability is at least conf_min and their product cum_min - the
  count, or -1 with why. the context must reach the sequence's position. neither moves a bit of the sequence's own
  state; a draft is a proposal (the verify step's bits are the answer's).
*/
