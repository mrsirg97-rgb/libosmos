#ifndef OSMOS_Q4X_FORWARD_H
#define OSMOS_Q4X_FORWARD_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/q4x/weights.h"

#define Q4X_FWD_ERR 256

typedef struct Q4xForward Q4xForward;
typedef struct Q4xSeq Q4xSeq;

typedef void (*q4x_probe_fn)(const char *step, int layer, float *dev, uint64_t n, void *arg);

Q4xForward *q4x_forward_create(const Q4xWeights *w, uint32_t max_ctx, uint32_t max_tokens, char err[Q4X_FWD_ERR]);
void q4x_forward_destroy(Q4xForward *f);
void q4x_forward_probe(Q4xForward *f, q4x_probe_fn fn, void *arg);

Q4xSeq *q4x_seq_create(Q4xForward *f, char err[Q4X_FWD_ERR]);
void q4x_seq_destroy(Q4xSeq *s);
int32_t q4x_seq_position(const Q4xSeq *s);

bool q4x_batch(Q4xForward *f, Q4xSeq *s, const int32_t *tokens, uint32_t n, char err[Q4X_FWD_ERR]);
const float *q4x_logits(const Q4xForward *f);
bool q4x_greedy(Q4xForward *f, uint32_t row0, uint32_t rows, int32_t *ids, char err[Q4X_FWD_ERR]);

#endif

/*
Q4xForward
  Qwen3.8-Flash-Next's forward on the GPU (DESIGN.md, "Qwen's forward, op by op"): every weight resident (the file's
  bytes, ~78 GB), its steps the kernels q4x_ops.h names, F32 between them. the attention dense: a sequence's
  positions below OPS_Q4X_MAX_KEYS (llama.cpp's QSA keeps every token below its 2,048-token budget; past it, the
  selection is not built yet).

q4x_forward_create
  the weights uploaded and the step's buffers for max_tokens tokens; NULL with why where the GPU's memory or a read
  refuses. max_ctx: a sequence's most positions (at most OPS_Q4X_MAX_KEYS).

q4x_forward_probe
  fn sees each named step's output (the dump's names: l_last a layer, result_norm) on the device, n floats,
  synchronized: a test's window - and may write it, the step going on from what it wrote (a reference's values stood
  in: a layer's error alone). NULL stops it.

q4x_seq_create .. q4x_seq_position
  a sequence: its DeltaNet states and conv histories, the n-gram layer's conv history, its attention caches (max_ctx
  rows), its last tokens; its position the tokens it holds.

q4x_batch
  n tokens (1 .. max_tokens) of one sequence at its next positions; the logits of every one of them after it
  (q4x_logits: [n][vocab] on the device). a token's results are the same bits however its sequence's tokens are
  split into batches. false with why for n out of range or past max_ctx; the sequence then unchanged in position
  (its state not to be read on: a refusal before any step runs).

q4x_greedy
  rows row0 .. row0 + rows - 1's greatest token each (ops_argmax's: the lowest among equals).
*/
