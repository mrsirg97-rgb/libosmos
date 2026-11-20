#ifndef ENGRAM_H
#define ENGRAM_H

#include <stdint.h>

#include "osmos/model/weights.h"

#define ENGRAM_ERR 256
#define ENGRAM_MAX_NGRAM 8

typedef struct Engram Engram;

/* a hash in a model's metadata (llama.cpp's form: engram.token_map, ple.layer_multipliers, ple.head_vocab_sizes,
   ple.head_offsets under the model's architecture), as a family states it */
typedef struct EngramShape
{
  const Gguf *model;
  const char *arch;          /* the keys' prefix (general.architecture's) */
  int n_layers;              /* the Engram layers */
  int ngram, heads;          /* n-grams up to ngram (orders 2 .. ngram), heads an order */
  uint64_t n_vocab;          /* the map's entries: one a token */
  uint64_t covered;          /* the tokens the map covers; a token at or past it reads as the pad (n_vocab: all) */
  int32_t pad;               /* the token a hash reads before a sequence's start */
  int64_t compressed;        /* the compressed tokens (0: the covered entries' greatest + 1) */
  const uint64_t *rows;      /* each Engram layer's table rows */
} EngramShape;

Engram *engram_create(const Weights *w, const int *model_fds, char err[ENGRAM_ERR]);
Engram *engram_create_shape(const EngramShape *s, char err[ENGRAM_ERR]);
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

engram_create_shape
  the same tables and checks for a hash a family states (its keys' prefix, its counts, the map's covered length):
  DeepSeek's own-file form is one of these. the map's entries past `covered` read as the pad's (the reference's
  lookup past its table), so every map entry engram_rows reads is a compressed token.

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
