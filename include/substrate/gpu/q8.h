#ifndef Q8_H
#define Q8_H

#include <stdbool.h>
#include <stdint.h>

bool q8_round(const float *x, uint32_t rows, uint32_t cols, uint32_t xs, int8_t *q, float *d, void *stream);
bool q8_group_tok(uint32_t type, const void *const *w, const int32_t *cnt, const int32_t *xrow, uint32_t n,
                  uint32_t max_cnt, uint32_t rows, uint32_t cols, const int8_t *q, const float *d, float *y,
                  uint32_t ys, void *stream);

#endif

/*
q8
  the lane's routed experts over int8 inputs (kernels/lane.h's contract): each row of x rounded to int8 a block of
  LANE_Q8_VALUES values, a weight's half-blocks against them integer dots, then F32 scales in one order - the order the
  CPU repeats (src/cpu/expert.c), so an expert gives the same bits on either side. the experts' IQ types only.

q8_round
  rows of x (cols values, rows xs floats apart) to q (row r at q + r * cols, int8) and d (row r's block scales at
  d + r * (cols / LANE_Q8_VALUES)): each block q = rint(x * (127 / amax)), d = amax / 127 (a block of zeros: q 0,
  d 0). queued on `stream` (a hipStream_t; NULL the default). false, nothing queued, for cols not whole blocks, xs
  shorter than a row, rows 0, or a launch refused.

q8_group_tok
  mv_group_tok (substrate/gpu/mv.h) over q8_round's rows: matrix m (w[m], `type` IQ2_XXS, IQ2_XS or IQ1_M; `rows`
  rows of `cols`) on its cnt[m] (1 .. max_cnt <= MV_MAX_TOKENS) rows of q / d from row xrow[m], into y's rows from
  xrow[m] (ys floats apart). a row's sum: its blocks to LANE_Q8_LANES(cols) lanes in turn, each block's two halves'
  lane_q8_part added, times the block's scale fmaf'd onto the lane's sum, then the lanes' xor tree - so a token's bits
  are its own alone, whatever max_cnt, the matrices beside it or the run. q / d rows up to xrow[m] + max_cnt are read
  (past cnt[m], not used). false, nothing queued, for a type it does not take, cols not whole blocks, n 0 or past
  MV_MAX_GROUP, max_cnt 0 or past MV_MAX_TOKENS, or a launch refused.
*/
