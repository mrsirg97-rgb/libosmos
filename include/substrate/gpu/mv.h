#ifndef MV_H
#define MV_H

#include <stdbool.h>
#include <stdint.h>

#define MV_MAX_TOKENS 8

bool mv_run(uint32_t type, const void *w, uint32_t rows, uint32_t cols, const float *x, uint32_t n_tok, float *y,
            void *stream);

#endif

/*
mv
  the decode's matrix-vector products: y = W x for up to MV_MAX_TOKENS tokens at once (a step and DSpark's drafts),
  W in any of the weights' types (quant.h) or F32, as the GGUF stores it: `rows` rows of `cols` values, row-major.

mv_run
  y[t * rows + r] = sum over c of W[r, c] * x[t * cols + c], for t < n_tok, queued on `stream` (a hipStream_t; NULL
  the default) and not waited for. `w`, `x`, `y` device-readable; x and y F32. a row is summed the same way whatever
  n_tok, the stream or the run: one wave a row, each lane its chunks in order (fmaf), the lanes by a fixed tree - so a
  token's y is the same bits alone or beside others (schedule invariance) and every run (determinism). false, nothing
  queued, for a type not known, cols not whole blocks, n_tok 0 or past MV_MAX_TOKENS, or a launch refused.
*/
