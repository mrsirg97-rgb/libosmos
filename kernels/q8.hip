// q8.h's matvec and rounding behind include/osmos/gpu/q8.h (the driver: q8_experts.hip)
#include <cstdlib>

#include "q8_launch.h"

extern "C" {
#include "osmos/gpu/mv.h"
#include "osmos/gpu/q8.h"
#include "osmos/model/qtype.h"
}

static_assert(LANE_Q8_VALUES == q8::kWave, "q8_round: a lane a value of a block");
static_assert(MV_MAX_TOKENS == 8, "q8_group_tok instantiates 1 .. 8 tokens");

namespace q8 {

// rows of x to int8 a block at a time: a wave a block (lane i its value i), the block's max magnitude by the wave's
// tree, each lane's code, four lanes' bytes a word; d the block's scale
__global__ __launch_bounds__(kWave* kWavesPerBlock) void round_kernel(const float* __restrict__ x, uint32_t rows,
                                                                       uint32_t cols, uint32_t xs,
                                                                       int8_t* __restrict__ q, float* __restrict__ d) {
  const uint32_t nblk = cols / LANE_Q8_VALUES, blk = blockIdx.x * kWavesPerBlock + threadIdx.x / kWave;
  const int lane = threadIdx.x % kWave;
  if (blk >= rows * nblk) return;  // the whole wave
  const uint32_t r = blk / nblk, s = blk % nblk;
  const float v = x[(uint64_t) r * xs + (uint64_t) s * LANE_Q8_VALUES + lane];
  float a = fabsf(v);
  for (int o = kWave / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, kWave));
  q[(uint64_t) r * cols + (uint64_t) s * LANE_Q8_VALUES + lane] = (int8_t) lane_q8_round(v, lane_q8_inv(a));
  if (lane == 0) d[(uint64_t) r * nblk + s] = lane_q8_scale(a);
}

}  // namespace q8

namespace {

template <quant::Type T, int NT, int L, int E>
bool launch(const void* const* w, const int32_t* cnt, const int32_t* xrow, const int32_t* xin, uint32_t n, uint32_t rows, uint32_t cols,
            const int8_t* q, const float* d, float* y, uint32_t ys, hipStream_t s) {
  const uint32_t per_block = q8::kWavesPerBlock * (q8::kWave / L);
  q8::group_tok_kernel<T, NT, L, E><<<dim3((rows + per_block - 1) / per_block, n), q8::kWave * q8::kWavesPerBlock, 0,
                                      s>>>((const uint8_t* const*) w, cnt, xrow, xin, rows, cols, q, d, y, ys);
  return hipGetLastError() == hipSuccess;
}

template <quant::Type T, int NT, int E>
bool by_lanes(const void* const* w, const int32_t* cnt, const int32_t* xrow, const int32_t* xin, uint32_t n, uint32_t rows, uint32_t cols,
              const int8_t* q, const float* d, float* y, uint32_t ys, hipStream_t s) {
  if (LANE_Q8_LANES(cols) == 32) return launch<T, NT, 32, E>(w, cnt, xrow, xin, n, rows, cols, q, d, y, ys, s);
  return launch<T, NT, 16, E>(w, cnt, xrow, xin, n, rows, cols, q, d, y, ys, s);
}

// at most max_cnt rows a matrix: one launch at max_cnt (every matrix's rows 1 .. max_cnt), or exact - a table in any
// order, counts 0 among them: the count-1 matrices a launch at 1 (a decode's or a verify's: nearly all), then the
// rest a launch at max_cnt (an expert two tokens picked, three) - not a launch a count, each a grid of every matrix
// that the others' leave empty (Mini's 7-token verify: 10% of its experts' time). by_count (exact, the table by count
// the most first, its counts' sum at most n: route_tok's): a launch a range of counts at the range's most - 1, 2,
// 3 .. 4, 5 .. max_cnt - each over the first n / (its least) matrices alone (a count past lo among them: the picks are
// n at most). a shared expert summed at its count's width, not max_cnt's (real verifies share: Hy4's 6-token ones 57%
// of their picks in shared experts). a token's sums its own at any width: the same bits either way
template <quant::Type T, int LO>
bool range(const void* const* w, const int32_t* cnt, const int32_t* xrow, const int32_t* xin, uint32_t n,
           uint32_t nt, uint32_t rows, uint32_t cols, const int8_t* q, const float* d, float* y, uint32_t ys,
           hipStream_t s) {
  const uint32_t m = n / (LO + 1);
  if (!m) return true;
  switch (nt) {
#define TOK(t) \
  case t: return by_lanes<T, t, LO>(w, cnt, xrow, xin, m, rows, cols, q, d, y, ys, s);
    TOK(1) TOK(2) TOK(3) TOK(4) TOK(5) TOK(6) TOK(7) TOK(8)
#undef TOK
    default: return false;
  }
}

template <quant::Type T>
bool by_tokens(const void* const* w, const int32_t* cnt, const int32_t* xrow, const int32_t* xin, uint32_t n, uint32_t max_cnt,
               uint32_t rows, uint32_t cols, const int8_t* q, const float* d, float* y, uint32_t ys, bool exact,
               bool by_count, hipStream_t s) {
  if (cols % quant::Block<T>::values) return false;  // whole blocks of the type (Qwen's down: 640 of IQ4_NL)
  bool ok = true;
  if (exact && by_count) {
    ok = range<T, 0>(w, cnt, xrow, xin, n, 1, rows, cols, q, d, y, ys, s);
    if (max_cnt >= 2) ok = ok && range<T, 1>(w, cnt, xrow, xin, n, 2, rows, cols, q, d, y, ys, s);
    if (max_cnt >= 3) ok = ok && range<T, 2>(w, cnt, xrow, xin, n, max_cnt < 4 ? max_cnt : 4, rows, cols, q, d, y, ys, s);
    if (max_cnt >= 5) ok = ok && range<T, 4>(w, cnt, xrow, xin, n, max_cnt, rows, cols, q, d, y, ys, s);
    return ok;
  }
  if (exact) {
    ok = by_lanes<T, 1, 0>(w, cnt, xrow, xin, n, rows, cols, q, d, y, ys, s);
    switch (max_cnt) {
#define TOK(nt) \
  case nt: return ok && by_lanes<T, nt, 1>(w, cnt, xrow, xin, n, rows, cols, q, d, y, ys, s);
      TOK(2) TOK(3) TOK(4) TOK(5) TOK(6) TOK(7) TOK(8)
#undef TOK
      default: return ok;
    }
  }
  switch (max_cnt) {
#define TOK(nt) \
  case nt: return by_lanes<T, nt, q8::kSelAll>(w, cnt, xrow, xin, n, rows, cols, q, d, y, ys, s);
    TOK(1) TOK(2) TOK(3) TOK(4) TOK(5) TOK(6) TOK(7) TOK(8)
#undef TOK
    default: return false;
  }
}

template <quant::Type T, int NT, int L, int E, int ACT>
bool launch_gate_up(const q8::GateUp& g, hipStream_t s) {
  const uint32_t per_block = q8::kWavesPerBlock * (q8::kWave / L);
  q8::gate_up_kernel<T, NT, L, E, ACT><<<dim3((g.rows + per_block - 1) / per_block, g.n),
                                         q8::kWave * q8::kWavesPerBlock, 0, s>>>(
      (const uint8_t* const*) g.wg, (const uint8_t* const*) g.wu, g.cnt, g.xrow, g.xin, g.rows, g.cols, g.q, g.d,
      g.args, g.h, g.hs);
  return hipGetLastError() == hipSuccess;
}

template <quant::Type T, int NT, int E>
bool gate_up_lanes(const q8::GateUp& g, hipStream_t s) {
  if (g.act == q8::kActInk2)
    return LANE_Q8_LANES(g.cols) == 32 ? launch_gate_up<T, NT, 32, E, q8::kActInk2>(g, s)
                                       : launch_gate_up<T, NT, 16, E, q8::kActInk2>(g, s);
  return LANE_Q8_LANES(g.cols) == 32 ? launch_gate_up<T, NT, 32, E, q8::kActLane>(g, s)
                                     : launch_gate_up<T, NT, 16, E, q8::kActLane>(g, s);
}

template <quant::Type T, int LO>
bool gate_up_range(const q8::GateUp& g, uint32_t nt, hipStream_t s) {
  q8::GateUp r = g;
  r.n = g.n / (LO + 1);
  if (!r.n) return true;
  switch (nt) {
#define TOK(t) \
  case t: return gate_up_lanes<T, t, LO>(r, s);
    TOK(1) TOK(2) TOK(3) TOK(4) TOK(5) TOK(6) TOK(7) TOK(8)
#undef TOK
    default: return false;
  }
}

template <quant::Type T>
bool gate_up_tokens(const q8::GateUp& g, uint32_t max_cnt, bool exact, bool by_count, hipStream_t s) {
  if (g.cols % quant::Block<T>::values) return false;
  bool ok = true;
  if (exact && by_count) {  // (by_tokens' ranges)
    ok = gate_up_range<T, 0>(g, 1, s);
    if (max_cnt >= 2) ok = ok && gate_up_range<T, 1>(g, 2, s);
    if (max_cnt >= 3) ok = ok && gate_up_range<T, 2>(g, max_cnt < 4 ? max_cnt : 4, s);
    if (max_cnt >= 5) ok = ok && gate_up_range<T, 4>(g, max_cnt, s);
    return ok;
  }
  if (exact) {
    ok = gate_up_lanes<T, 1, 0>(g, s);
    switch (max_cnt) {
#define TOK(nt) \
  case nt: return ok && gate_up_lanes<T, nt, 1>(g, s);
      TOK(2) TOK(3) TOK(4) TOK(5) TOK(6) TOK(7) TOK(8)
#undef TOK
      default: return ok;
    }
  }
  switch (max_cnt) {
#define TOK(nt) \
  case nt: return gate_up_lanes<T, nt, q8::kSelAll>(g, s);
    TOK(1) TOK(2) TOK(3) TOK(4) TOK(5) TOK(6) TOK(7) TOK(8)
#undef TOK
    default: return false;
  }
}

}  // namespace

bool q8::group(uint32_t type, const void* const* w, const int32_t* cnt, const int32_t* xrow, const int32_t* xin,
               uint32_t n, uint32_t max_cnt, uint32_t rows, uint32_t cols, const int8_t* q, const float* d, float* y,
               uint32_t ys, bool exact, bool by_count, hipStream_t s) {
  if (!n || n > MV_MAX_GROUP || !rows || !max_cnt || max_cnt > MV_MAX_TOKENS || !cols || cols % LANE_Q8_VALUES)
    return false;
  switch (type) {
#define CASE(t) \
  case quant::t: return by_tokens<quant::t>(w, cnt, xrow, xin, n, max_cnt, rows, cols, q, d, y, ys, exact, by_count, s);
    Q8_TYPES(CASE)
#undef CASE
    default: return false;
  }
}

bool q8::gate_up(uint32_t type, const GateUp& g, uint32_t max_cnt, bool exact, bool by_count, hipStream_t s) {
  if (!g.n || g.n > MV_MAX_GROUP || !g.rows || !max_cnt || max_cnt > MV_MAX_TOKENS || !g.cols || g.cols % LANE_Q8_VALUES ||
      g.hs < g.rows || (g.act == kActInk2 && (!g.args.sg || !g.args.su)))
    return false;
  switch (type) {
#define CASE(t) \
  case quant::t: return gate_up_tokens<quant::t>(g, max_cnt, exact, by_count, s);
    Q8_TYPES(CASE)
#undef CASE
    default: return false;
  }
}

extern "C" bool q8_round(const float* x, uint32_t rows, uint32_t cols, uint32_t xs, int8_t* q, float* d,
                         void* stream) {
  if (!rows || !cols || cols % LANE_Q8_VALUES || xs < cols) return false;
  const uint64_t blocks = (uint64_t) rows * (cols / LANE_Q8_VALUES);
  q8::round_kernel<<<(unsigned) ((blocks + q8::kWavesPerBlock - 1) / q8::kWavesPerBlock),
                     q8::kWave * q8::kWavesPerBlock, 0, (hipStream_t) stream>>>(x, rows, cols, xs, q, d);
  return hipGetLastError() == hipSuccess;
}

// the table's QTYPE_Q8 rows (qtype.h) - test_qtype_gpu holds them to Q8_TYPES' kernels
#define Q8_ROW(name, id, values, bytes, paths) || (type == (id) && ((paths) & QTYPE_Q8))
extern "C" bool q8_takes(uint32_t type) { return false QTYPES(Q8_ROW); }
#undef Q8_ROW

extern "C" bool q8_enabled(void) {
  const char* v = getenv("OSMOS_Q8");
  return !v || atoi(v) != 0;
}

extern "C" bool q8_group_tok(uint32_t type, const void* const* w, const int32_t* cnt, const int32_t* xrow,
                             const int32_t* xin, uint32_t n, uint32_t max_cnt, uint32_t rows, uint32_t cols,
                             const int8_t* q, const float* d, float* y, uint32_t ys, void* stream) {
  return q8::group(type, w, cnt, xrow, xin, n, max_cnt, rows, cols, q, d, y, ys, false, false, (hipStream_t) stream);
}

extern "C" bool q8_gate_up(uint32_t type, const void* const* wg, const void* const* wu, const int32_t* cnt,
                           const int32_t* xrow, const int32_t* xin, const int32_t* win, uint32_t n,
                           uint32_t max_cnt, uint32_t rows, uint32_t cols,
                           const int8_t* q, const float* d, const float* scale, float lim, float* h, uint32_t hs,
                           void* stream) {
  const q8::GateUp g = {wg, wu, cnt, xrow, xin, n, rows, cols, q, d, q8::kActLane,
                        {scale, win, lim, nullptr, nullptr, 0.0f, 0.0f}, h, hs};
  return q8::gate_up(type, g, max_cnt, false, false, (hipStream_t) stream);
}
