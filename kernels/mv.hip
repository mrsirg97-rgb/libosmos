// mv_run (include/osmos/gpu/mv.h): the kernel is mv.h's, its shape picked here
#include "mv.h"

extern "C" {
#include "osmos/gpu/mv.h"
}

namespace {

// rows a wave (R) and chunks in flight a lane (U), by type and tokens: the best of R 1-2, U 1-8 measured on gfx1151
// on ds1017's shapes (2026-10-28, an expert's 5120 x 2304 and 2304 x 5120, q_b, out_b, q_a, shexp); 3 tokens take
// 2's, 5-7 take 8's. neither changes a bit (mv.h)
struct Tune {
  int r, u;
};

constexpr int band(int nt) { return nt == 1 ? 0 : nt <= 2 ? 1 : nt <= 4 ? 2 : 3; }

constexpr Tune tune(quant::Type t, int nt) {
  switch (t) {
    case quant::kIQ1_M:
    case quant::kIQ1_S:  // IQ1_S and IQ2_S: their sibling's (IQ1_M's, IQ2_XS's), not measured
    case quant::kIQ2_XXS:
    case quant::kIQ2_XS:
    case quant::kIQ2_S: return (Tune[]){{1, 4}, {1, 2}, {2, 2}, {2, 2}}[band(nt)];
    case quant::kQ6_K: return (Tune[]){{1, 4}, {1, 2}, {1, 1}, {2, 1}}[band(nt)];
    case quant::kQ5_K:
    case quant::kQ5_KR: return {2, 1};
    case quant::kQ8_0: return nt > 4 ? Tune{2, 1} : Tune{1, 1};
    // an F32 router (unsloth's: 384 x 5120) and its guess's BF16 copy, one token: 4 chunks in flight a lane - F32
    // 53.5 -> 38.8 us (147 -> 202 GB/s), BF16 22.8 -> 19.5 (2026-11-04, 40 routers cold; R 1-4, U 1-8 swept, the
    // same bits each). more tokens not measured
    case quant::kF32:
    case quant::kBF16: return nt == 1 ? Tune{1, 4} : Tune{1, 1};
    default: return {1, 1};
  }
}

// the wide tile: R rows a wave, NW tokens a block - the best of R 1-8, NW 4-32 on ds1017's router (BF16 5120 x 384,
// 8,192 tokens: 15.2 ms in chunks of 8, 5.5 here; R 3 NW 12 5.1, 6 or 8 rows 6.6-7.1, 16+ tokens 6.3-16)
struct Wide {
  int r, nw;
};
constexpr Wide kWide = {4, 8};

template <quant::Type T, int NT>
bool launch(const void* w, uint32_t rows, uint32_t cols, const float* x, uint32_t xs, float* y, uint32_t ys,
            hipStream_t s) {
  constexpr Tune k = tune(T, NT);
  return mv::launch<T, NT, k.r, k.u>(w, rows, cols, x, xs, y, ys, s);
}

template <quant::Type T>
bool by_tokens(const void* w, uint32_t rows, uint32_t cols, const float* x, uint32_t xs, uint32_t n_tok, float* y,
               uint32_t ys, hipStream_t s) {
  if (cols % quant::Block<T>::values || cols % quant::kChunk) return false;
  if ((n_tok > 1 && (xs < cols || ys < rows)) || xs % quant::kChunk) return false;  // tokens apart; x's float4s whole
  if (n_tok > MV_MAX_TOKENS)  // a prompt's: mv.h's wide tiles, each token's sums its own alone
    return mv::wide_launch<T, kWide.r, kWide.nw>(w, rows, cols, x, xs, n_tok, y, ys, s);
  switch (n_tok) {
    case 1: return launch<T, 1>(w, rows, cols, x, xs, y, ys, s);
    case 2: return launch<T, 2>(w, rows, cols, x, xs, y, ys, s);
    case 3: return launch<T, 3>(w, rows, cols, x, xs, y, ys, s);
    case 4: return launch<T, 4>(w, rows, cols, x, xs, y, ys, s);
    case 5: return launch<T, 5>(w, rows, cols, x, xs, y, ys, s);
    case 6: return launch<T, 6>(w, rows, cols, x, xs, y, ys, s);
    case 7: return launch<T, 7>(w, rows, cols, x, xs, y, ys, s);
    case 8: return launch<T, 8>(w, rows, cols, x, xs, y, ys, s);
    default: return false;
  }
}

}  // namespace

static_assert(MV_MAX_TOKENS == 8, "by_tokens instantiates 1 .. 8");

template <quant::Type T>
bool group(const void* const* w, uint32_t n, uint32_t rows, uint32_t cols, const float* x, uint32_t x_mat,
           uint32_t x_div, float* y, hipStream_t s) {
  if (cols % quant::Block<T>::values || cols % quant::kChunk || x_mat % quant::kChunk || !x_div) return false;
  constexpr Tune k = tune(T, 1);
  return mv::group_launch<T, k.r, k.u>(w, n, rows, cols, x, x_mat, x_div, y, s);
}

// the lane's one-token launches, several experts at once (mv_group_tok, 6 experts of ds1017's IQ types): the IQ2s
// best at 2 rows a wave, 4 chunks in flight (gate 122.5 -> 110 us, down 149.8 -> 125.3); IQ1_M its short rows (the
// down's 2,304 columns) at 2 and 2 (123.3 -> 118.5), its long as tune's (2026-10-31, DESIGN.md). no bit changes
constexpr uint32_t kShortCols = 4096;  // fewer than 16 chunks a lane
constexpr Tune tune_group1(quant::Type t, bool short_row) {
  switch (t) {
    case quant::kIQ2_XXS:
    case quant::kIQ2_XS:
    case quant::kIQ2_S: return {2, 4};
    case quant::kIQ1_M:
    case quant::kIQ1_S: return short_row ? Tune{2, 2} : tune(t, 1);
    default: return tune(t, 1);
  }
}

template <quant::Type T, int NT, int R, int U>
bool group_tok_k(const void* const* w, const int32_t* cnt, const int32_t* xrow, uint32_t n, uint32_t rows,
                 uint32_t cols, const float* x, uint32_t xs, float* y, uint32_t ys, hipStream_t s) {
  const uint32_t per_block = mv::kWavesPerBlock * R;
  const dim3 grid((rows + per_block - 1) / per_block, n);
  mv::group_tok_kernel<T, NT, R, U><<<grid, mv::kWave * mv::kWavesPerBlock, 0, s>>>(
      (const uint8_t* const*) w, cnt, xrow, rows, cols, x, xs, y, ys);
  return hipGetLastError() == hipSuccess;
}

template <quant::Type T, int NT>
bool group_tok_nt(const void* const* w, const int32_t* cnt, const int32_t* xrow, uint32_t n, uint32_t rows,
                  uint32_t cols, const float* x, uint32_t xs, float* y, uint32_t ys, hipStream_t s) {
  if constexpr (NT == 1) {
    constexpr Tune a = tune_group1(T, true), b = tune_group1(T, false);
    if (cols < kShortCols) return group_tok_k<T, 1, a.r, a.u>(w, cnt, xrow, n, rows, cols, x, xs, y, ys, s);
    return group_tok_k<T, 1, b.r, b.u>(w, cnt, xrow, n, rows, cols, x, xs, y, ys, s);
  }
  constexpr Tune k = tune(T, NT);
  return group_tok_k<T, NT, k.r, k.u>(w, cnt, xrow, n, rows, cols, x, xs, y, ys, s);
}

template <quant::Type T>
bool group_tok(const void* const* w, const int32_t* cnt, const int32_t* xrow, uint32_t n, uint32_t max_cnt,
               uint32_t rows, uint32_t cols, const float* x, uint32_t xs, float* y, uint32_t ys, hipStream_t s) {
  if (cols % quant::Block<T>::values || cols % quant::kChunk || xs < cols || xs % quant::kChunk) return false;
  switch (max_cnt) {
#define TOK(nt) \
  case nt: return group_tok_nt<T, nt>(w, cnt, xrow, n, rows, cols, x, xs, y, ys, s);
    TOK(1) TOK(2) TOK(3) TOK(4) TOK(5) TOK(6) TOK(7) TOK(8)
#undef TOK
    default: return false;
  }
}

extern "C" bool mv_group_tok(uint32_t type, const void* const* w, const int32_t* cnt, const int32_t* xrow, uint32_t n,
                             uint32_t max_cnt, uint32_t rows, uint32_t cols, const float* x, uint32_t xs, float* y,
                             uint32_t ys, void* stream) {
  if (!n || n > MV_MAX_GROUP || !rows || !max_cnt || max_cnt > MV_MAX_TOKENS) return false;
  const hipStream_t s = (hipStream_t) stream;
  switch (type) {
    case quant::kQ8_0: return group_tok<quant::kQ8_0>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kIQ2_XXS: return group_tok<quant::kIQ2_XXS>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kIQ2_XS: return group_tok<quant::kIQ2_XS>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kIQ1_M: return group_tok<quant::kIQ1_M>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kIQ1_S: return group_tok<quant::kIQ1_S>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kIQ2_S: return group_tok<quant::kIQ2_S>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kMXFP4: return group_tok<quant::kMXFP4>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    default: return false;
  }
}

extern "C" bool mv_group(uint32_t type, const void* const* w, uint32_t n, uint32_t rows, uint32_t cols, const float* x,
                         uint32_t x_mat, uint32_t x_div, float* y, void* stream) {
  if (!n || n > MV_MAX_GROUP || !rows) return false;
  const hipStream_t s = (hipStream_t) stream;
  switch (type) {
    case quant::kF32: return group<quant::kF32>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kQ8_0: return group<quant::kQ8_0>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kQ5_K: return group<quant::kQ5_K>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kQ6_K: return group<quant::kQ6_K>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kIQ2_XXS: return group<quant::kIQ2_XXS>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kIQ2_XS: return group<quant::kIQ2_XS>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kIQ4_NL: return group<quant::kIQ4_NL>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kIQ1_M: return group<quant::kIQ1_M>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kIQ1_S: return group<quant::kIQ1_S>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kIQ2_S: return group<quant::kIQ2_S>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kBF16: return group<quant::kBF16>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kMXFP4: return group<quant::kMXFP4>(w, n, rows, cols, x, x_mat, x_div, y, s);
    default: return false;
  }
}

extern "C" bool mv_run(uint32_t type, const void* w, uint32_t rows, uint32_t cols, const float* x, uint32_t n_tok,
                       float* y, void* stream) {
  if (n_tok > MV_MAX_TOKENS) return false;  // the decode's: a step and its drafts
  return mv_strided(type, w, rows, cols, x, cols, n_tok, y, rows, stream);
}

extern "C" bool mv_strided(uint32_t type, const void* w, uint32_t rows, uint32_t cols, const float* x, uint32_t xs,
                           uint32_t n_tok, float* y, uint32_t ys, void* stream) {
  const hipStream_t s = (hipStream_t) stream;
  switch (type) {
    case quant::kF32: return by_tokens<quant::kF32>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kQ8_0: return by_tokens<quant::kQ8_0>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kQ5_K: return by_tokens<quant::kQ5_K>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kQ5_KR: return by_tokens<quant::kQ5_KR>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kQ6_K: return by_tokens<quant::kQ6_K>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kIQ2_XXS: return by_tokens<quant::kIQ2_XXS>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kIQ2_XS: return by_tokens<quant::kIQ2_XS>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kIQ4_NL: return by_tokens<quant::kIQ4_NL>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kIQ1_M: return by_tokens<quant::kIQ1_M>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kIQ1_S: return by_tokens<quant::kIQ1_S>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kIQ2_S: return by_tokens<quant::kIQ2_S>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kBF16: return by_tokens<quant::kBF16>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kMXFP4: return by_tokens<quant::kMXFP4>(w, rows, cols, x, xs, n_tok, y, ys, s);
    default: return false;
  }
}
