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
bool q8_takes(uint32_t type);
bool q8_enabled(void);

/* a table of routed experts through gate, up, the activation and down over int8 inputs (q8_experts) */
#define Q8_ACT_LANE 0 /* lane_swiglu(g, u, lim, scale[win ? win[r] : r]; NULL 1): DeepSeek, Hy4, Qwen */
#define Q8_ACT_INK2 1 /* Inkling's: gcap tanh(g' / gcap) sigmoid(g') ucap tanh(u' / ucap), g' = g sg[r], u' = u su[r] */
typedef struct Q8Experts
{
  int act; /* the fused activation (Q8_ACT_*; 0 the lane's) */
  uint32_t gate_type, up_type, down_type;
  const void *const *wg, *const *wu, *const *wd; /* entry m's three matrices (device tables) */
  const int32_t *cnt, *xrow;                     /* entry m's rows: cnt[m] (0 .. max_cnt) from row xrow[m] */
  const int32_t *xin;                            /* an entry row's input row (NULL: its own) */
  uint32_t n, max_cnt;                           /* the entries, the most rows an entry */
  const int32_t *h_cnt; /* the counts on the host in runs of one count (a launch a run), or NULL: exact */
  uint32_t d, ff;       /* the input's width (down's rows), an expert's width (gate / up's rows) */
  const float *x;       /* x_rows rows of d floats rounded first into xq / xqd, or NULL: xq / xqd already */
  uint32_t x_rows;
  int8_t *xq;
  float *xqd;
  const float *scale; /* the activation's weight: scale[win ? win[r] : r], NULL 1 (types unlike: given, win NULL) */
  const int32_t *win;
  float lim;
  const float *sg, *su; /* Q8_ACT_INK2's: each output row's gate and up tensor scales */
  float gcap, ucap;     /* its soft caps */
  float *ga, *up;          /* gate and up's sums when their types differ (h_rows x ff each); else unused */
  float *h;                /* the activations: h_rows rows of ff, each rounded into hq / hqd for down */
  uint32_t h_rows;
  int8_t *hq;
  float *hqd;
  float *y; /* down's rows: entry row r's at y + r * d */
} Q8Experts;
bool q8_experts(const Q8Experts *e, void *stream);

#endif

/*
q8
  the lane's routed experts over int8 inputs (kernels/lane.h's contract): each row of x rounded to int8 a block of
  LANE_Q8_VALUES values, a weight's half-blocks against them integer dots, then F32 scales in one order - the order the
  CPU repeats (src/cpu/q8.c), so an expert gives the same bits on either side (the CPU's types: lane_q8_type's). the
  GPU's types: q8_takes's.

q8_round
  rows of x (cols values, rows xs floats apart) to q (row r at q + r * cols, int8) and d (row r's block scales at
  d + r * (cols / LANE_Q8_VALUES)): each block q = rint(x * (127 / amax)), d = amax / 127 (a block of zeros: q 0,
  d 0). queued on `stream` (a hipStream_t; NULL the default). false, nothing queued, for cols not whole blocks, xs
  shorter than a row, rows 0, or a launch refused.

q8_group_tok
  mv_group_tok (osmos/gpu/mv.h) over q8_round's rows: matrix m (w[m], `type` one q8_takes takes; `rows` rows of
  `cols`) on its cnt[m] (1 .. max_cnt <= MV_MAX_TOKENS) rows of q / d from row xrow[m], into y's rows from xrow[m] (ys
  floats apart). a row's sum: its blocks to LANE_Q8_LANES(cols) lanes in turn, each block's two halves' lane_q8_part
  added, times the block's scale fmaf'd onto the lane's sum, then the lanes' xor tree - so a token's bits are its own
  alone, whatever max_cnt, the matrices beside it or the run. output row r's input: q / d row r (xin NULL), else row
  xin[r] - the lane's rows by token, each token's row rounded once for all its experts, not a copy a pair. the rows past
  cnt[m] are summed on the count's last input (not stored): no row nor xin entry past it is read. false, nothing queued,
  for a type it does not take, cols not whole blocks, n 0 or past MV_MAX_GROUP, max_cnt 0 or past MV_MAX_TOKENS, or a
  launch refused.

q8_gate_up
  an expert's gate and up and the activation in one launch: q8_group_tok's sums of wg[m] and of wu[m] (both `type`)
  on the same rows of q / d - x read once for both - and, rather than writing them, h's row r (hs floats apart) the
  lane's activation on them: lane_swiglu(gate, up, lim, scale[r]) (scale[win[r]] with a win: the router's weights by
  pair, not gathered; scale NULL: 1 - the weights applied at the sum) - ops_swiglu_lane's bits on q8_group_tok's.
  inputs by xin as q8_group_tok's. false, nothing queued, as q8_group_tok, and for hs shorter than a row.

q8_takes
  whether the GPU's int8 lane decodes `type` exactly (kernels/q8.h's codes; the table's QTYPE_Q8 rows: every
  QTYPE_MV_TOK type but MXFP8, whose
  E4M3 codes have no int8 form; F32 / BF16 not block codes). test_q8_types_gpu proves each against its F32 decoder.

q8_enabled
  whether a family takes its routed experts' int8 path where it has one (q8_takes its types): yes, unless the
  environment's OSMOS_Q8 is 0 - the engine-wide switch an A/B turns off (ds4's lane its own: lane_q8_types, the CPU's
  bits).

q8_experts
  a family's routed experts on int8 inputs, ds4's lane order: x rounded (unless xq / xqd given), gate and up and the
  activation (one launch for both, q8_gate_up's, when they are one type; else each, then ops_swiglu_lane over h_rows
  rows), h_rows rows of h rounded, then down into y. each role over the table either by h_cnt's runs (a launch a run of
  one count, at it: the caller's table in runs) or exact - a launch a count 1 .. max_cnt, each its entries of exactly
  that count, a table in any order with counts of 0 (nothing read for them). a token's sums its own either way: the
  same bits. a decode's or a verify's experts (max_cnt <= MV_MAX_TOKENS): a family's prompt its own path (its
  matrix cores, f16) - the two computed apart, as Hy4's and DeepSeek's. false, perhaps part queued, for max_cnt past
  MV_MAX_TOKENS, a type q8_takes refuses, an argument missing, or a launch refused.
*/
