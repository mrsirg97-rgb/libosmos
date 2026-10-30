#ifndef FORWARD_H
#define FORWARD_H

#include <stdbool.h>
#include <stdint.h>

#include "substrate/gpu/mm.h"
#include "substrate/model/weights.h"

#define FORWARD_ERR 256
/* a batch's sub-batches of MM_MAX_TOKENS: maya's 8,192-token chunk. each layer runs them all before the next, so the
   experts it streams are read once for all of them (DESIGN.md, "the memory, revised") */
#define FORWARD_SUBS 4
#define FORWARD_MAX_TOKENS (FORWARD_SUBS * MM_MAX_TOKENS)
#define SEQ_MAX_BACK 16 /* seq_rollback's deepest: a verify's rejected drafts */

typedef struct Forward Forward;
typedef struct Seq Seq;

typedef void (*forward_probe_fn)(const char *step, int layer, int index, void *dev, uint64_t n, bool ints,
                                 void *arg);

Forward *forward_create(const Weights *w, int n_layers, uint32_t max_ctx, uint64_t budget, char err[FORWARD_ERR]);
Forward *forward_create_ranked(const Weights *w, int n_layers, uint32_t max_ctx, uint64_t budget,
                               const uint64_t *picks, char err[FORWARD_ERR]);
void forward_destroy(Forward *f);
void forward_probe(Forward *f, forward_probe_fn fn, void *arg);

Seq *seq_create(Forward *f, char err[FORWARD_ERR]);
void seq_destroy(Seq *s);
bool seq_prompt(Seq *s, uint32_t end);
bool seq_rollback(Seq *s, uint32_t n);

bool forward_step(Forward *f, Seq *s, int32_t token, char err[FORWARD_ERR]);
bool forward_batch(Forward *f, Seq *const *seqs, const int32_t *tokens, uint32_t n, char err[FORWARD_ERR]);
const float *forward_stream(const Forward *f);
const float *forward_logits(const Forward *f);
const uint64_t *forward_picks(const Forward *f);
uint64_t forward_swaps(const Forward *f);
void forward_guesses(const Forward *f, uint64_t *streamed, uint64_t *guessed);

#endif

/*
Forward
  DeepSeek-V4.1-Flash's forward on the GPU, a token at a time (the decode), its first `n_layers` layers (all of them,
  or fewer for a test). the weights are copies in GPU memory, read from their files straight in: a resident set
  (substrate/mem/resident.h) of every tensor the layers use but the routed experts and as many experts as the budget
  holds, about an equal share of each layer's (its most-picked first: forward_create_ranked); the other experts
  streamed by the pool (substrate/mem/pool.h), held when the router picks them and let go, the one wanted furthest
  ahead first, when it needs the room. the quantized weights' products are mm's (WMMA over f16 activations, one
  kernel for 1 .. 16 tokens), the F32 and BF16 weights' the matvec's. `budget` is the GPU memory of the resident set
  (the pool's own room is two layers' streamed experts). the Engram rows are read from the Engram file a position at
  a time (pread). `max_ctx` bounds a sequence: its
  compressed rows and indexer keys are held for that many positions.

forward_create
  NULL, with why in err, for a model or a size it cannot run: n_layers outside 1 .. the model's, max_ctx 0, the
  layers' weights more than the budget, a GPU allocation refused, a weight's read failed.

forward_create_ranked
  forward_create, the resident experts chosen by picks ([layer][expert] for the n_layers layers, n_experts a layer:
  the router's picks counted on calibration text, forward_picks's) - a rank at a time, each layer's most-picked
  first, ties to the lower id. picks NULL: by id, as forward_create. which experts are resident changes no bit.

forward_destroy
  everything it allocated (its Seqs first: seq_destroy). NULL is nothing to do.

forward_probe
  fn is called after each step the forward names - "h_in" (the stream into a layer, after Engram), "a_pre", "a_post",
  "a_comb", "attn_x", "q", "kv" (the window's new row), "ckv" (a new compressed row, index = its row), "sel" (the
  compressed rows a position attends, ints), "o", "attn_out", "h_mid", "f_pre", "f_post", "f_comb", "ffn_x",
  "router_logits", "route_idx" (ints), "route_w", "moe_out", "h_out", "logits" (layer -1) - with the layer, the
  position (or the row), the device buffer and its length. a float step's buffer is what the forward reads next: a
  probe may overwrite it (a test's teacher forcing: each step then runs on the oracle's input). the GPU is waited for
  before each call, so a probe slows the forward; NULL (the default) none.

Seq
  one sequence's state: its position, its last tokens (Engram's hash), each layer's window of keys (a ring of window +
  MV_MAX_TOKENS rows), each KV source's compressed rows and indexer keys and the positions it has not yet compressed.

seq_create / seq_destroy
  a sequence at position 0 / its state freed (NULL nothing to do). a Seq is its Forward's and goes before it.

seq_prompt
  the sequence's prompt ends at position end (its last token end - 1): the caller reads no logits before end - 1 and
  steps nothing from before it. its tokens before that, beyond the decoder's reach of the end - the last token's
  windows through the layers past the model's last KV source, (those layers) (window - 1) + 1 positions - then skip
  those layers (DeepSeek-V4.1's decoder, whose state of a prompt token is its windows); the last KV source still runs
  for every token. the logits of every token from end - 1 on, and every stream and logit after, are the bits of the
  whole run. raised as the prompt grows (a later turn's); false for no sequence, an end past max_ctx, or one lower
  than before.

seq_rollback
  the sequence's last n positions undone: its next position n back, its last tokens (Engram's n-gram) as they were
  there - the state a forward of those positions leaves (each its window row, its compressor's pending row, a
  compressed row it completed) is rewritten when the positions are stepped again, before anything reads it, so the
  rows after are the bits of a run that never stepped them (a verify's rejected drafts). false, nothing changed, for
  n past SEQ_MAX_BACK or the tokens it holds, or a position before the prompt's end (seq_prompt).

forward_step
  forward_batch of one: the token at the sequence's next position through the layers; the stream out in
  forward_stream. false with why in err for a token outside the vocabulary, a sequence at max_ctx, a context past
  what this build selects from (more candidate blocks than the model's candidate_topk_blocks: the candidate masking
  is not built), a GPU call that failed, or an expert the budget cannot hold - the sequence is then not to be
  stepped again.

forward_batch
  n tokens (1 .. FORWARD_MAX_TOKENS) at once, token i at seqs[i]'s next position: a sequence may be given several times,
  its tokens then at its next positions in the order given (a chunk of its prompt, or drafts to verify), each seeing
  the ones before it. a token's results are the same bits however it is batched - alone, beside other sequences, or
  in a chunk of its own: every step a token takes is summed in an order of its own, never the batch's. past
  MM_MAX_TOKENS the batch runs as sub-batches of MM_MAX_TOKENS (the last the rest), each layer for every one of them
  in turn before the next layer - the same bits as a step a sub-batch, the experts read once a layer. false, with
  why, as forward_step, for a batch of 0 or more than FORWARD_MAX_TOKENS tokens, or a sequence of another forward; on
  a refusal no sequence has moved. probes see a sub-batch at a time (its rows; index: its first token's position).

forward_picks
  each expert's picks by the router since forward_create, [layer][expert] (n_layers x n_experts): a profile for
  forward_create_ranked. the host's; valid until forward_destroy.

forward_swaps
  the experts moved into the resident set so far: as a step's layer is done with them, its hottest streamed experts
  take the places of its coldest resident ones not picked in the step, when much hotter (their picks, halved over
  the steps) - a copy in GPU memory each, the set following the text from where forward_create_ranked started it.
  it changes no bit.

forward_guesses
  in steps that pick fewer experts than a layer has (a decode's, a verify's): the picks of experts not resident so
  far, and of them those guessed a layer ahead - each layer's next experts guessed by the next layer's router on its
  own FFN input and their reads begun then, under the layer's experts and the next one's attention.

forward_stream
  the device buffer of the last step's stream out ([n][hc][n_embd] F32) - the last sub-batch's rows, for a batch
  past MM_MAX_TOKENS; a token that skipped the decoder (seq_prompt): its row unspecified.

forward_logits
  the device buffer of the last step's logits ([n][n_vocab] F32) - the last sub-batch's rows, for a batch past
  MM_MAX_TOKENS: the head over the stream, when n_layers is all of the model's (else not computed); a token before its
  sequence's prompt end - 1 (seq_prompt): its row not computed.
*/
