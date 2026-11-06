#ifndef MV_H
#define MV_H

#include <stdbool.h>
#include <stdint.h>

#define MV_MAX_TOKENS 8
#define MV_MAX_GROUP 65535

bool mv_run(uint32_t type, const void *w, uint32_t rows, uint32_t cols, const float *x, uint32_t n_tok, float *y,
            void *stream);
bool mv_group(uint32_t type, const void *const *w, uint32_t n, uint32_t rows, uint32_t cols, const float *x,
              uint32_t x_mat, uint32_t x_div, float *y, void *stream);
bool mv_group_tok(uint32_t type, const void *const *w, const int32_t *cnt, const int32_t *xrow, uint32_t n,
                  uint32_t max_cnt, uint32_t rows, uint32_t cols, const float *x, uint32_t xs, float *y, uint32_t ys,
                  void *stream);
bool mv_strided(uint32_t type, const void *w, uint32_t rows, uint32_t cols, const float *x, uint32_t x_stride,
                uint32_t n_tok, float *y, uint32_t y_stride, void *stream);

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

mv_group
  n matrices of one type and shape at once, one vector each: y[m * rows + r] = sum over c of W_m[r, c] *
  x[(m / x_div) * x_mat + c], W_m at w[m] (`w` a device-readable table of n pointers - experts in the pool, say;
  x_mat 0: one x for all; x_div k: k matrices a vector, a batch's tokens' k experts each). a row summed as mv_run
  sums it: the same bits as n mv_run calls, in one launch. false, nothing queued, as mv_run, and for n 0 or past
  MV_MAX_GROUP, or x_div 0.

mv_group_tok
  n matrices of one type and shape, each over its own tokens: matrix m (w[m]) on cnt[m] (1 .. max_cnt <= MV_MAX_TOKENS;
  0: the entry skipped, nothing read or written - a table launched to its bound) rows of x from row xrow[m] (rows xs floats apart), into y's rows from xrow[m] (ys apart) - an expert read once for
  the tokens that picked it. each row mv_run's sums: a token's bits are those it takes alone. `w`, `cnt`, `xrow` on
  the device; x's rows past a matrix's are read, not used (rows up to xrow[m] + max_cnt must be readable). the routed
  experts' types and Q8_0, MXFP4.

mv_strided
  mv_run with token t's x at x + t * x_stride and its y at y + t * y_stride (a slice of a wider row: the attention's
  grouped wo_a reads each group's 4,096 of a token's 32,768); the same sums, so the same bits. any n_tok: past
  MV_MAX_TOKENS (a prompt's), tiles of rows and tokens that share each decoded chunk of W and of x - each token's
  sums still its own alone, so the same bits as one call a token. also false for a stride that is not a multiple of
  8, or, with tokens > 1, that is shorter than its row.
*/
