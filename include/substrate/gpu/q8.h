#ifndef Q8_H
#define Q8_H

#include <stdbool.h>
#include <stdint.h>

bool q8_round(const float *x, uint32_t rows, uint32_t cols, uint32_t xs, int8_t *q, float *d, void *stream);
bool q8_group_tok(uint32_t type, const void *const *w, const int32_t *cnt, const int32_t *xrow, const int32_t *xin,
                  uint32_t n, uint32_t max_cnt, uint32_t rows, uint32_t cols, const int8_t *q, const float *d, float *y,
                  uint32_t ys, void *stream);
bool q8_gate_up(uint32_t type, const void *const *wg, const void *const *wu, const int32_t *cnt, const int32_t *xrow,
                const int32_t *xin, const int32_t *win, uint32_t n, uint32_t max_cnt, uint32_t rows, uint32_t cols, const int8_t *q, const float *d,
                const float *scale, float lim, float *h, uint32_t hs, void *stream);

#endif

/*
q8
  the lane's routed experts over int8 inputs (kernels/lane.h's contract): each row of x rounded to int8 a block of
  LANE_Q8_VALUES values, a weight's half-blocks against them integer dots, then F32 scales in one order - the order the
  CPU repeats (src/cpu/q8.c), so an expert gives the same bits on either side. the experts' IQ types only.

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
  are its own alone, whatever max_cnt, the matrices beside it or the run. output row r's input: q / d row r (xin NULL),
  else row xin[r] - the lane's rows by token, each token's row rounded once for all its experts, not a copy a pair.
  the rows past cnt[m] are summed on the count's last input (not stored): no row nor xin entry past it is read. false, nothing queued, for a type it does not take, cols not whole blocks, n 0 or past
  MV_MAX_GROUP, max_cnt 0 or past MV_MAX_TOKENS, or a launch refused.

q8_gate_up
  an expert's gate and up and the activation in one launch: q8_group_tok's sums of wg[m] and of wu[m] (both `type`)
  on the same rows of q / d - x read once for both - and, rather than writing them, h's row r (hs floats apart) the
  lane's activation on them: lane_swiglu(gate, up, lim, scale[r]) (scale[win[r]] with a win: the router's weights by
  pair, not gathered) - ops_swiglu_lane's bits on q8_group_tok's. inputs by xin as q8_group_tok's. false,
  nothing queued, as q8_group_tok, and for no scale or hs shorter than a row.
*/
