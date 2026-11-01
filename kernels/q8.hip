// q8.h's kernels behind include/substrate/gpu/q8.h
#include "q8.h"

extern "C" {
#include "substrate/gpu/mv.h"
#include "substrate/gpu/q8.h"
}

static_assert(LANE_Q8_VALUES == q8::kWave, "q8_round: a lane a value of a block");
static_assert(MV_MAX_TOKENS == 8, "q8_group_tok instantiates 1 .. 8 tokens");

namespace {

// q8_gate_up's arguments, carried through the dispatch
struct GateUp {
  const void* const *wg, *const *wu;
  const int32_t *cnt, *xrow;
  uint32_t n, rows, cols;
  const int8_t* q;
  const float *d, *scale;
  float lim;
  float* h;
  uint32_t hs;
};

template <quant::Type T, int NT, int L>
bool launch(const void* const* w, const int32_t* cnt, const int32_t* xrow, uint32_t n, uint32_t rows, uint32_t cols,
            const int8_t* q, const float* d, float* y, uint32_t ys, hipStream_t s) {
  const uint32_t per_block = q8::kWavesPerBlock * (q8::kWave / L);
  q8::group_tok_kernel<T, NT, L><<<dim3((rows + per_block - 1) / per_block, n), q8::kWave * q8::kWavesPerBlock, 0,
                                   s>>>((const uint8_t* const*) w, cnt, xrow, rows, cols, q, d, y, ys);
  return hipGetLastError() == hipSuccess;
}

template <quant::Type T, int NT>
bool by_lanes(const void* const* w, const int32_t* cnt, const int32_t* xrow, uint32_t n, uint32_t rows, uint32_t cols,
              const int8_t* q, const float* d, float* y, uint32_t ys, hipStream_t s) {
  if (LANE_Q8_LANES(cols) == 32) return launch<T, NT, 32>(w, cnt, xrow, n, rows, cols, q, d, y, ys, s);
  return launch<T, NT, 16>(w, cnt, xrow, n, rows, cols, q, d, y, ys, s);
}

template <quant::Type T>
bool by_tokens(const void* const* w, const int32_t* cnt, const int32_t* xrow, uint32_t n, uint32_t max_cnt,
               uint32_t rows, uint32_t cols, const int8_t* q, const float* d, float* y, uint32_t ys, hipStream_t s) {
  switch (max_cnt) {
#define TOK(nt) \
  case nt: return by_lanes<T, nt>(w, cnt, xrow, n, rows, cols, q, d, y, ys, s);
    TOK(1) TOK(2) TOK(3) TOK(4) TOK(5) TOK(6) TOK(7) TOK(8)
#undef TOK
    default: return false;
  }
}

template <quant::Type T, int NT, int L>
bool launch_gate_up(const GateUp& g, hipStream_t s) {
  const uint32_t per_block = q8::kWavesPerBlock * (q8::kWave / L);
  q8::gate_up_kernel<T, NT, L><<<dim3((g.rows + per_block - 1) / per_block, g.n), q8::kWave * q8::kWavesPerBlock, 0,
                                 s>>>((const uint8_t* const*) g.wg, (const uint8_t* const*) g.wu, g.cnt, g.xrow, g.rows,
                                      g.cols, g.q, g.d, g.scale, g.lim, g.h, g.hs);
  return hipGetLastError() == hipSuccess;
}

template <quant::Type T, int NT>
bool gate_up_lanes(const GateUp& g, hipStream_t s) {
  return LANE_Q8_LANES(g.cols) == 32 ? launch_gate_up<T, NT, 32>(g, s) : launch_gate_up<T, NT, 16>(g, s);
}

template <quant::Type T>
bool gate_up_tokens(const GateUp& g, uint32_t max_cnt, hipStream_t s) {
  switch (max_cnt) {
#define TOK(nt) \
  case nt: return gate_up_lanes<T, nt>(g, s);
    TOK(1) TOK(2) TOK(3) TOK(4) TOK(5) TOK(6) TOK(7) TOK(8)
#undef TOK
    default: return false;
  }
}

}  // namespace

extern "C" bool q8_round(const float* x, uint32_t rows, uint32_t cols, uint32_t xs, int8_t* q, float* d,
                         void* stream) {
  if (!rows || !cols || cols % LANE_Q8_VALUES || xs < cols) return false;
  const uint64_t blocks = (uint64_t) rows * (cols / LANE_Q8_VALUES);
  q8::round_kernel<<<(unsigned) ((blocks + q8::kWavesPerBlock - 1) / q8::kWavesPerBlock),
                     q8::kWave * q8::kWavesPerBlock, 0, (hipStream_t) stream>>>(x, rows, cols, xs, q, d);
  return hipGetLastError() == hipSuccess;
}

extern "C" bool q8_group_tok(uint32_t type, const void* const* w, const int32_t* cnt, const int32_t* xrow, uint32_t n,
                             uint32_t max_cnt, uint32_t rows, uint32_t cols, const int8_t* q, const float* d, float* y,
                             uint32_t ys, void* stream) {
  if (!n || n > MV_MAX_GROUP || !rows || !max_cnt || max_cnt > MV_MAX_TOKENS || !cols || cols % quant::kQK)
    return false;
  const hipStream_t s = (hipStream_t) stream;
  switch (type) {
    case quant::kIQ2_XXS: return by_tokens<quant::kIQ2_XXS>(w, cnt, xrow, n, max_cnt, rows, cols, q, d, y, ys, s);
    case quant::kIQ2_XS: return by_tokens<quant::kIQ2_XS>(w, cnt, xrow, n, max_cnt, rows, cols, q, d, y, ys, s);
    case quant::kIQ1_M: return by_tokens<quant::kIQ1_M>(w, cnt, xrow, n, max_cnt, rows, cols, q, d, y, ys, s);
    default: return false;
  }
}

extern "C" bool q8_gate_up(uint32_t type, const void* const* wg, const void* const* wu, const int32_t* cnt,
                           const int32_t* xrow, uint32_t n, uint32_t max_cnt, uint32_t rows, uint32_t cols,
                           const int8_t* q, const float* d, const float* scale, float lim, float* h, uint32_t hs,
                           void* stream) {
  if (!n || n > MV_MAX_GROUP || !rows || !max_cnt || max_cnt > MV_MAX_TOKENS || !cols || cols % quant::kQK || !scale ||
      hs < rows)
    return false;
  const GateUp g = {wg, wu, cnt, xrow, n, rows, cols, q, d, scale, lim, h, hs};
  const hipStream_t s = (hipStream_t) stream;
  switch (type) {
    case quant::kIQ2_XXS: return gate_up_tokens<quant::kIQ2_XXS>(g, max_cnt, s);
    case quant::kIQ2_XS: return gate_up_tokens<quant::kIQ2_XS>(g, max_cnt, s);
    case quant::kIQ1_M: return gate_up_tokens<quant::kIQ1_M>(g, max_cnt, s);
    default: return false;
  }
}
