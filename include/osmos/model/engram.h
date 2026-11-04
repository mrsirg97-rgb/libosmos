#ifndef ENGRAM_H
#define ENGRAM_H

#include <stdint.h>

#include "osmos/model/weights.h"

#define ENGRAM_ERR 256
#define ENGRAM_MAX_NGRAM 8

typedef struct Engram Engram;

Engram *engram_create(const Weights *w, const int *model_fds, char err[ENGRAM_ERR]);
void engram_destroy(Engram *e);

int engram_cols(const Engram *e);
int engram_ngram(const Engram *e);
bool engram_rows(const Engram *e, const int32_t *last, int n_last, int64_t *rows);

#endif

/*
Engram
  DeepSeek-V4.1's n-gram hash: the table rows a position reads in each Engram layer. its four tables - the token map,
  the multipliers, the primes, the offsets (the model's engram_* tensors) - read once from the model's files
  (`model_fds[file]`, the caller's descriptors of the Weights' model) into its own memory and checked: every map entry
  a compressed token (below the geometry's engram_vocab), every multiplier odd, positive and small enough that no
  product overflows, every prime positive, and each column's rows (offset .. offset + prime) inside its layer's table
  - so every row engram_rows gives is one the table has. NULL with why in err otherwise.

engram_destroy
  the tables. NULL is nothing to do.

engram_cols / engram_ngram
  the rows a position reads a layer ((ngram - 1) * heads), and the n-gram's length (the tokens a position's hash reads).

engram_rows
  `last[s]` the token s positions back (last[0] the position's own), `n_last` of them there (fewer than the n-gram at a
  sequence's start: the rest are the pad); `rows` gets each Engram layer's cols rows, layer by layer in the model's
  order. ref.py's engram_hash.Hasher: the products xor-folded, a column's row the fold mod its prime plus its offset.
  false, rows untouched past the point, for a token outside the model's vocabulary.
*/
