#ifndef FORWARD_H
#define FORWARD_H

#include <stdbool.h>
#include <stdint.h>

#include "substrate/model/weights.h"

#define FORWARD_ERR 256

typedef struct Forward Forward;
typedef struct Seq Seq;

typedef void (*forward_probe_fn)(const char *step, int layer, int index, void *dev, uint64_t n, bool ints,
                                 void *arg);

Forward *forward_create(const Weights *w, int n_layers, uint32_t max_ctx, uint64_t budget, char err[FORWARD_ERR]);
void forward_destroy(Forward *f);
void forward_probe(Forward *f, forward_probe_fn fn, void *arg);

Seq *seq_create(Forward *f, char err[FORWARD_ERR]);
void seq_destroy(Seq *s);

bool forward_step(Forward *f, Seq *s, int32_t token, char err[FORWARD_ERR]);
const float *forward_stream(const Forward *f);
const float *forward_logits(const Forward *f);

#endif

/*
Forward
  DeepSeek-V4.1-Flash's forward on the GPU, a token at a time (the decode), its first `n_layers` layers (all of them,
  or fewer for a test). the weights are read where they lie, through one pool (substrate/mem/pool.h): every tensor
  the layers use but the routed experts a unit pinned for the forward's life, and each layer's experts units the
  pool pins when the router picks them and unpins, least recently used first, when it needs the room - read by
  fetchq's workers, nothing copied, none of it counted against the GPU's own memory. `budget` is the RAM the pool may
  pin: the layers' weights and a cache of experts. the Engram rows are read from the Engram file a position at a
  time (pread). `max_ctx` bounds a sequence: its compressed rows and indexer keys are held for that many positions.

forward_create
  NULL, with why in err, for a model or a size it cannot run: n_layers outside 1 .. the model's, max_ctx 0, the
  layers' weights more than the budget, a GPU allocation or registration refused.

forward_destroy
  everything it registered, mapped and allocated (its Seqs first: seq_destroy). NULL is nothing to do.

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

forward_step
  the token at the sequence's next position through the layers; the stream out in forward_stream. false with why in
  err for a token outside the vocabulary, a sequence at max_ctx, a context past what this build selects from (more
  candidate blocks than the model's candidate_topk_blocks: the candidate masking is not built), or a GPU call that
  failed, or an expert the budget cannot hold - the sequence is then not to be stepped again.

forward_stream
  the device buffer of the last step's stream out ([hc][n_embd] F32).

forward_logits
  the device buffer of the last step's logits ([n_vocab] F32): the head over the stream, when n_layers is all of the
  model's (else not computed).
*/
