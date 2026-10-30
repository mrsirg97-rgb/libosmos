// mv_run (include/substrate/gpu/mv.h): the kernel is mv.h's, its shape picked here
#include "mv.h"

extern "C" {
#include "substrate/gpu/mv.h"
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
    case quant::kIQ2_XXS:
    case quant::kIQ2_XS: return (Tune[]){{1, 4}, {1, 2}, {2, 2}, {2, 2}}[band(nt)];
    case quant::kQ6_K: return (Tune[]){{1, 4}, {1, 2}, {1, 1}, {2, 1}}[band(nt)];
    case quant::kQ5_K: return {2, 1};
    case quant::kQ8_0: return nt > 4 ? Tune{2, 1} : Tune{1, 1};
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
    case quant::kQ6_K: return by_tokens<quant::kQ6_K>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kIQ2_XXS: return by_tokens<quant::kIQ2_XXS>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kIQ2_XS: return by_tokens<quant::kIQ2_XS>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kIQ4_NL: return by_tokens<quant::kIQ4_NL>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kIQ1_M: return by_tokens<quant::kIQ1_M>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kBF16: return by_tokens<quant::kBF16>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kMXFP4: return by_tokens<quant::kMXFP4>(w, rows, cols, x, xs, n_tok, y, ys, s);
    default: return false;
  }
}
