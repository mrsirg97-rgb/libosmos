#include <cstdlib>
// mv_run (include/osmos/gpu/mv.h): the kernel is mv.h's, its shape picked here
#include "mv.h"

extern "C" {
#include "osmos/gpu/mv.h"
}

namespace {

// rows a wave (R) and chunks in flight a lane (U), by type and tokens, tuned on gfx1151; 3 tokens take 2's, 5-7 take
// 8's. neither changes a bit (mv.h)
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
    // an F32 router and its guess's BF16 copy: one token 4 chunks in flight a lane, 2 to 6 tokens (a verify's) 2
    case quant::kF32:
    case quant::kBF16: return nt == 1 ? Tune{1, 4} : Tune{1, 2};
    // Qwen3.8's expert types read for several tokens (an expert two or more of a step's tokens picked). one token:
    // tune_group1's
    case quant::kIQ4_NL:
    case quant::kQ4_0:  // not measured: IQ4_NL's (its layout, its block)
    case quant::kQ2_0:
    case quant::kQ2_K:  // not measured: Q2_0's
    case quant::kNVFP4:  // not measured: Q2_0's (a 64-value block too)
    case quant::kIQ3_XXS:
    case quant::kIQ4_XS: return {2, 2}; /* 4 for IQ3_XXS and IQ4_XS at 4 tokens, but a group shares one R */
    default: return {1, 1};
  }
}

// the wide tile: R rows a wave, NW tokens a block
struct Wide {
  int r, nw;
};
constexpr Wide kWide = {4, 8};
// F32's own; and at most kNarrow.r rows (the mixers' inject, the shared expert's gate) mv.h's narrow tile, a wave all
// the rows of its own tokens. a tile's sums are its rows' and tokens' own: any of them the same bits
constexpr Wide kWideF32 = {3, 12}, kNarrow = {4, 1};

constexpr uint64_t kQ8Wide = 8u << 20;  // a Q8_0 matrix's values from which a token's matvec takes 4 chunks in flight

// the device's CUs (0: not read)
static uint32_t cus() {
  static int n = -1;
  if (n < 0 && hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, 0) != hipSuccess) n = 0;
  return (uint32_t) n;
}

// an F32 matrix too short for a block of 8 waves on every CU (Inkling's sink, 48 x 6144, 6 blocks on 40 CUs): a wave
// a block, its waves over the CUs - each row's sums rows_of's whatever its block, so the same bits
template <quant::Type T, int NT>
bool launch(const void* w, uint32_t rows, uint32_t cols, const float* x, uint32_t xs, float* y, uint32_t ys,
            hipStream_t s) {
  constexpr Tune k = tune(T, NT);
  if constexpr (T == quant::kF32)
    if (rows < (uint64_t) cus() * mv::kWavesPerBlock * k.r) return mv::launch<T, NT, k.r, k.u, 1>(w, rows, cols, x, xs, y, ys, s);
  // osmos1.7b v-r15's BF16 2,048-column rows, one token: 8 chunks in flight a lane - the row's whole share at once
  // (16.3 -> 15.0 ms a token's matrices, DESIGN.md "0.17.1"); by its columns alone (Mini's BF16 none of them). the
  // same bits (a lane's chunks summed in the same order)
  if constexpr (T == quant::kBF16 && NT == 1)
    if (cols == 2048) return mv::launch<T, NT, 1, 8>(w, rows, cols, x, xs, y, ys, s);
  // a large Q8_0 matrix, one token: 4 chunks in flight a lane (under kQ8Wide: 1). the same bits (a lane's chunks
  // summed in the same order)
  if constexpr (T == quant::kQ8_0 && NT == 1)
    if ((uint64_t) rows * cols >= kQ8Wide) return mv::launch<T, NT, 1, 4>(w, rows, cols, x, xs, y, ys, s);
  // a verify's few tokens on a large Q8_0 matrix: tall 1 row 2 in flight to 4 tokens, tune's past them; wide 4 rows 1
  // to 4 tokens, 1 and 1 at 5, 2 and 2 at 6. the same bits
  if constexpr (T == quant::kQ8_0 && NT > 1)
    if ((uint64_t) rows * cols >= kQ8Wide) {
      if (rows >= cols) {
        if constexpr (NT <= 4) return mv::launch<T, NT, 1, 2>(w, rows, cols, x, xs, y, ys, s);
      } else if constexpr (NT <= 4) return mv::launch<T, NT, 4, 1>(w, rows, cols, x, xs, y, ys, s);
      else if constexpr (NT == 5) return mv::launch<T, NT, 1, 1>(w, rows, cols, x, xs, y, ys, s);
      else if constexpr (NT == 6) return mv::launch<T, NT, 2, 2>(w, rows, cols, x, xs, y, ys, s);
    }
  return mv::launch<T, NT, k.r, k.u>(w, rows, cols, x, xs, y, ys, s);
}

template <quant::Type T>
bool by_tokens(const void* w, uint32_t rows, uint32_t cols, const float* x, uint32_t xs, uint32_t n_tok, float* y,
               uint32_t ys, hipStream_t s) {
  if (cols % quant::Block<T>::values || cols % quant::kChunk) return false;
  if ((n_tok > 1 && (xs < cols || ys < rows)) || xs % quant::kChunk) return false;  // tokens apart; x's float4s whole
  if (n_tok > MV_MAX_TOKENS) {  // a prompt's: mv.h's wide tiles, each token's sums its own alone
    if constexpr (T == quant::kF32) {
      if (rows <= (uint32_t) kNarrow.r) return mv::narrow_launch<T, kNarrow.r, kNarrow.nw>(w, rows, cols, x, xs, n_tok, y, ys, s);
      return mv::wide_launch<T, kWideF32.r, kWideF32.nw>(w, rows, cols, x, xs, n_tok, y, ys, s);
    }
    return mv::wide_launch<T, kWide.r, kWide.nw>(w, rows, cols, x, xs, n_tok, y, ys, s);
  }
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

// the lane's one-token launches, several experts at once (mv_group_tok): the IQ2s 2 rows a wave, 4 chunks in flight;
// IQ1_M its short rows at 2 and 2, its long as tune's. no bit changes
constexpr uint32_t kShortCols = 4096;  // fewer than 16 chunks a lane
constexpr Tune tune_group1(quant::Type t, bool short_row) {
  switch (t) {
    case quant::kIQ2_XXS:
    case quant::kIQ2_XS:
    case quant::kIQ2_S: return {2, 4};
    // Hy4-flash's experts (mvsweep hy4)
    case quant::kQ3_K: return {2, 2};
    case quant::kQ4_K: return {1, 2};
    case quant::kIQ1_M:
    case quant::kIQ1_S: return short_row ? Tune{2, 2} : tune(t, 1);
    // Qwen3.8's expert types (tools/bench/mvsweep)
    case quant::kIQ4_NL:
    case quant::kQ4_0:  // not measured: IQ4_NL's
    case quant::kIQ3_XXS:
    case quant::kIQ4_XS:
    case quant::kQ2_0:
    case quant::kNVFP4:
    case quant::kQ2_K: return {2, 2};  // (Q2_K and NVFP4 not measured: Q2_0's)
    default: return tune(t, 1);
  }
}

template <quant::Type T, int NT, int R, int U, bool EXACT>
bool group_tok_k(const void* const* w, const int32_t* cnt, const int32_t* xrow, uint32_t n, uint32_t rows,
                 uint32_t cols, const float* x, uint32_t xs, float* y, uint32_t ys, hipStream_t s) {
  const uint32_t per_block = mv::kWavesPerBlock * R;
  const dim3 grid((rows + per_block - 1) / per_block, n);
  if constexpr (EXACT)
    mv::group_tok_exact_kernel<T, NT, R, U><<<grid, mv::kWave * mv::kWavesPerBlock, 0, s>>>(
        (const uint8_t* const*) w, cnt, xrow, rows, cols, x, xs, y, ys);
  else
    mv::group_tok_kernel<T, NT, R, U><<<grid, mv::kWave * mv::kWavesPerBlock, 0, s>>>(
        (const uint8_t* const*) w, cnt, xrow, rows, cols, x, xs, y, ys);
  return hipGetLastError() == hipSuccess;
}

template <quant::Type T, int NT, bool EXACT>
bool group_tok_nt(const void* const* w, const int32_t* cnt, const int32_t* xrow, uint32_t n, uint32_t rows,
                  uint32_t cols, const float* x, uint32_t xs, float* y, uint32_t ys, hipStream_t s) {
  if constexpr (NT == 1) {
    constexpr Tune a = tune_group1(T, true), b = tune_group1(T, false);
    if (cols < kShortCols) return group_tok_k<T, 1, a.r, a.u, EXACT>(w, cnt, xrow, n, rows, cols, x, xs, y, ys, s);
    return group_tok_k<T, 1, b.r, b.u, EXACT>(w, cnt, xrow, n, rows, cols, x, xs, y, ys, s);
  }
  constexpr Tune k = tune(T, NT);
  return group_tok_k<T, NT, k.r, k.u, EXACT>(w, cnt, xrow, n, rows, cols, x, xs, y, ys, s);
}

// a table of at most max_cnt tokens a matrix: one launch a count (1 .. max_cnt), each its matrices of exactly that many
// - a verify's experts mostly one token each, not each at the table's most (16 one-token Q3_K experts: 122.7 us at a
// count of 1, 224.2 at 6). a token's sums its own at any count (rows_of's): the same bits
template <quant::Type T>
bool group_tok(const void* const* w, const int32_t* cnt, const int32_t* xrow, uint32_t n, uint32_t max_cnt,
               uint32_t rows, uint32_t cols, const float* x, uint32_t xs, float* y, uint32_t ys, hipStream_t s) {
  if (cols % quant::Block<T>::values || cols % quant::kChunk || xs < cols || xs % quant::kChunk) return false;
  if (max_cnt == 1) return group_tok_nt<T, 1, false>(w, cnt, xrow, n, rows, cols, x, xs, y, ys, s);
  if (max_cnt > MV_MAX_TOKENS) return false;
  bool ok = true;
#define TOK(nt) \
  if (ok && max_cnt >= nt) ok = group_tok_nt<T, nt, true>(w, cnt, xrow, n, rows, cols, x, xs, y, ys, s);
  TOK(1) TOK(2) TOK(3) TOK(4) TOK(5) TOK(6) TOK(7) TOK(8)
#undef TOK
  return ok;
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
    case quant::kMXFP8: return group_tok<quant::kMXFP8>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kIQ4_NL: return group_tok<quant::kIQ4_NL>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kQ4_0: return group_tok<quant::kQ4_0>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kIQ3_XXS: return group_tok<quant::kIQ3_XXS>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kQ3_K: return group_tok<quant::kQ3_K>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kQ2_K: return group_tok<quant::kQ2_K>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kIQ4_XS: return group_tok<quant::kIQ4_XS>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kQ4_K: return group_tok<quant::kQ4_K>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kQ5_K: return group_tok<quant::kQ5_K>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kQ6_K: return group_tok<quant::kQ6_K>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kQ5_1: return group_tok<quant::kQ5_1>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kQ2_0: return group_tok<quant::kQ2_0>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
    case quant::kNVFP4: return group_tok<quant::kNVFP4>(w, cnt, xrow, n, max_cnt, rows, cols, x, xs, y, ys, s);
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
    case quant::kQ4_0: return group<quant::kQ4_0>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kIQ1_M: return group<quant::kIQ1_M>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kIQ1_S: return group<quant::kIQ1_S>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kIQ2_S: return group<quant::kIQ2_S>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kBF16: return group<quant::kBF16>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kMXFP4: return group<quant::kMXFP4>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kMXFP8: return group<quant::kMXFP8>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kIQ3_XXS: return group<quant::kIQ3_XXS>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kQ3_K: return group<quant::kQ3_K>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kQ2_K: return group<quant::kQ2_K>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kIQ4_XS: return group<quant::kIQ4_XS>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kQ4_K: return group<quant::kQ4_K>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kQ5_1: return group<quant::kQ5_1>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kQ2_0: return group<quant::kQ2_0>(w, n, rows, cols, x, x_mat, x_div, y, s);
    case quant::kNVFP4: return group<quant::kNVFP4>(w, n, rows, cols, x, x_mat, x_div, y, s);
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
    case quant::kQ4_0: return by_tokens<quant::kQ4_0>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kIQ1_M: return by_tokens<quant::kIQ1_M>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kIQ1_S: return by_tokens<quant::kIQ1_S>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kIQ2_S: return by_tokens<quant::kIQ2_S>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kBF16: return by_tokens<quant::kBF16>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kMXFP4: return by_tokens<quant::kMXFP4>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kMXFP8: return by_tokens<quant::kMXFP8>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kIQ3_XXS: return by_tokens<quant::kIQ3_XXS>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kQ3_K: return by_tokens<quant::kQ3_K>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kQ2_K: return by_tokens<quant::kQ2_K>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kIQ4_XS: return by_tokens<quant::kIQ4_XS>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kQ4_K: return by_tokens<quant::kQ4_K>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kQ5_1: return by_tokens<quant::kQ5_1>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kQ2_0: return by_tokens<quant::kQ2_0>(w, rows, cols, x, xs, n_tok, y, ys, s);
    case quant::kNVFP4: return by_tokens<quant::kNVFP4>(w, rows, cols, x, xs, n_tok, y, ys, s);
    default: return false;
  }
}
