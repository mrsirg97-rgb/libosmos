#ifndef ROPE_H
#define ROPE_H

#include <stdbool.h>

#define ROPE_MAX_DIM 256

typedef struct Rope
{
  int dim;
  float base;
  int original;
  float factor;
  float beta_fast;
  float beta_slow;
} Rope;

bool rope_freqs(const Rope *r, float *freqs);

#endif

/*
Rope
  a rotary embedding: the last `dim` values of a head rotated as dim / 2 adjacent pairs, pair i by position * freq[i].
  `original` 0: plain (the window's: base 10,000); else YaRN past `original` positions (the compressed rows': base
  160,000, factor 16, beta 32 / 1).

rope_freqs
  the dim / 2 frequencies into freqs, in F32 as ref.py's freqs_cis computes them (float32 pow, reciprocal and blend:
  the same bits; tests/test_rope.c holds torch's). false for a dim not even, or past ROPE_MAX_DIM, a base not above 1,
  or YaRN with a factor or betas not positive.
*/
