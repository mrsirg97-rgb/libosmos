// mm (include/substrate/gpu/mm.h): the kernel is mm.h's
#include <cmath>

#include "mm.h"
#include "round.h"

extern "C" {
#include "substrate/gpu/mm.h"
}

namespace {

constexpr int kThreads = 256;

// a type's tile (mm.h's Shape) and its most column tiles: measured on gfx1151 (2026-10-29). a shape is a type's for
// every token count - its K split is the sums' order, so a token's bits would otherwise depend on its batch. Q5_K's
// decode is the heaviest: small slabs, so more waves hide it (q_b 88 -> 163 GB/s, the head 123 -> 157); Q8_0 and Q6_K
// were near their matvec's at SL 128. the IQ experts: 4 waves of 128-value slabs, each wave a long run of slabs and a
// 4-way sum at the end (16 waves of 64 ran 5 slabs a wave to 5120 columns, then a 16-way sum through LDS a tile; the
// counters: the waves 52-76% waiting, 14% at the barrier). a real step's experts (each run of the same column tiles a
// launch) 17.6 -> 13.9 ms (blk.24's gate), 22.6 -> 15.7 (blk.10's down); a decode token 74.0 -> 71.6 ms; a 4096-token
// prompt ~294 -> ~314 tok/s. the column tiles at 1024 tokens (4096 x 4096, the tiled x16): 8 for all but Q8_0, whose
// 4 is 1.30 ms to 8's 1.62
template <quant::Type T> struct Tuned { using S = mm::Shape<128, 8, 8>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kQ8_0> { using S = mm::Shape<128, 8, 8>; static constexpr int kMaxCt = 4; };
template <> struct Tuned<quant::kQ5_K> { using S = mm::Shape<32, 8, 2>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kQ5_KR> : Tuned<quant::kQ5_K> {};
template <> struct Tuned<quant::kIQ1_M> { using S = mm::Shape<128, 4, 8>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kIQ2_XXS> { using S = mm::Shape<128, 4, 8>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kIQ2_XS> { using S = mm::Shape<128, 4, 8>; static constexpr int kMaxCt = 8; };
using mm::kF16Max;

// a thread a value: to f16 (the rows past n zero)
__global__ void x16_kernel(const float* x, uint32_t xs, uint32_t n, uint32_t cols, _Float16* x16, int32_t* overflow) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  const uint32_t t = (uint32_t) (i / cols), c = (uint32_t) (i % cols);
  if (t >= (n + mm::kTile - 1) / mm::kTile * mm::kTile) return;
  mm::x16_put(x16, t, c, cols, t < n ? x[(uint64_t) t * xs + c] : 0.0f, overflow);  // past f16: refused at the end
}

// the attention's output to f16 on the way to wo_a: a wave an FP8 block of a token's row, a lane a value - each
// head's last rope_dim values rotated back (a pair's lanes trading theirs), the block rounded to FP8, then to f16.
// ops_rope's, ops_fp8_round's and x16_kernel's expressions in their order, so the same bits as the three (the rows
// past n zero, as x16_kernel's)
__global__ void x16_attn_kernel(const float* x, uint32_t xs, uint32_t n, uint32_t cols, uint32_t head_dim,
                                uint32_t rope_dim, const float* freqs, const int32_t* pos, _Float16* x16,
                                int32_t* overflow) {
  static_assert(kThreads % rnd::kFp8Block == 0, "whole waves");
  const uint64_t blk = (uint64_t) blockIdx.x * (kThreads / rnd::kFp8Block) + threadIdx.x / rnd::kFp8Block;
  const uint32_t lane = threadIdx.x % rnd::kFp8Block, per_row = cols / rnd::kFp8Block;
  const uint32_t t = (uint32_t) (blk / per_row), c = (uint32_t) (blk % per_row) * rnd::kFp8Block + lane;
  if (t >= (n + mm::kTile - 1) / mm::kTile * mm::kTile) return;  // the whole wave: a block is one token's
  float v = 0.0f;
  if (t < n) {
    v = x[(uint64_t) t * xs + c];
    const uint32_t d = c % head_dim, r0 = head_dim - rope_dim;
    if (d >= r0) {  // the whole block in the rope's part (r0 a multiple of the block: the launch checks)
      const float other = __shfl_xor(v, 1, rnd::kFp8Block);
      const float ang = (float) pos[t] * freqs[(d - r0) / 2];
      const float cs = cosf(ang), sn = -sinf(ang);
      const float a = lane & 1 ? other : v, b = lane & 1 ? v : other;
      v = lane & 1 ? a * sn + b * cs : a * cs - b * sn;
    }
    v = rnd::fp8_block(v);
  }
  if (!(fabsf(v) <= kF16Max)) atomicOr(overflow, 1);
  x16[mm::x16_at(t, c, cols)] = (_Float16) v;
}

// slot s: row idx[s] of x (-1: zero), to f16
__global__ void x16_rows_kernel(const float* x, uint32_t xs, const int32_t* idx, uint32_t slots, uint32_t cols,
                                _Float16* x16, int32_t* overflow) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) slots * cols) return;
  const int32_t r = idx[i / cols];
  const float v = r >= 0 ? x[(uint64_t) r * xs + i % cols] : 0.0f;
  if (!(fabsf(v) <= kF16Max)) atomicOr(overflow, 1);
  x16[mm::x16_at((uint32_t) (i / cols), (uint32_t) (i % cols), cols)] = (_Float16) v;
}

// the column tiles a block takes (a weight slab decoded once for CT WMMAs): as many as the tokens need, up to the
// type's most
template <quant::Type T>
uint32_t col_tiles(uint32_t n) {
  const uint32_t want = n <= mm::kTile ? 1 : n <= 2 * mm::kTile ? 2 : n <= 4 * mm::kTile ? 4 : 8;
  return want < (uint32_t) Tuned<T>::kMaxCt ? want : (uint32_t) Tuned<T>::kMaxCt;
}

template <quant::Type T, int CT>
bool group_ct(const void* const* w, const int32_t* cnt, uint32_t n_mat, uint32_t max_cnt, uint32_t rows,
              uint32_t cols, const void* x16, const uint32_t* xoff, uint32_t xts, float* y, const uint32_t* yoff,
              uint32_t ys, hipStream_t s) {
  using S = typename Tuned<T>::S;
  const uint32_t z = (max_cnt + mm::kTile * CT - 1) / (mm::kTile * CT);
  mm::group_kernel<T, S, CT><<<dim3(rows / mm::kTile, n_mat, z), mm::kWave * S::kWaves, 0, s>>>(
      (const uint8_t* const*) w, cnt, cols, (const _Float16*) x16, xoff, xts, y, yoff, ys);
  return hipGetLastError() == hipSuccess;
}

template <quant::Type T>
bool group_launch(const void* const* w, const int32_t* cnt, uint32_t n_mat, uint32_t max_cnt, uint32_t rows,
                  uint32_t cols, const void* x16, const uint32_t* xoff, uint32_t xts, float* y, const uint32_t* yoff,
                  uint32_t ys, hipStream_t s) {
  using S = typename Tuned<T>::S;
  if (rows % mm::kTile || cols % quant::Block<T>::values || cols % S::kSlab || xts < cols || xts % mm::kTile ||
      !max_cnt || max_cnt > MM_MAX_TOKENS)
    return false;
  switch (col_tiles<T>(max_cnt)) {
    case 1: return group_ct<T, 1>(w, cnt, n_mat, max_cnt, rows, cols, x16, xoff, xts, y, yoff, ys, s);
    case 2: return group_ct<T, 2>(w, cnt, n_mat, max_cnt, rows, cols, x16, xoff, xts, y, yoff, ys, s);
    case 4: return group_ct<T, 4>(w, cnt, n_mat, max_cnt, rows, cols, x16, xoff, xts, y, yoff, ys, s);
    default: return group_ct<T, Tuned<T>::kMaxCt>(w, cnt, n_mat, max_cnt, rows, cols, x16, xoff, xts, y, yoff, ys, s);
  }
}

/* one matrix (w[1] NULL) or a pair on the same x16 */
struct Mats {
  const void* w[2];
  uint32_t rows[2];
  float* y[2];
  uint32_t ys[2];
};

template <quant::Type T, int CT>
bool launch_ct(const Mats& m, uint32_t cols, const void* x16, uint32_t n, hipStream_t s) {
  using S = typename Tuned<T>::S;
  const uint32_t z = (n + mm::kTile * CT - 1) / (mm::kTile * CT), rows = m.rows[0] > m.rows[1] ? m.rows[0] : m.rows[1];
  mm::kernel<T, S, CT><<<dim3(rows / mm::kTile, m.w[1] ? 2 : 1, z), mm::kWave * S::kWaves, 0, s>>>(
      (const uint8_t*) m.w[0], (const uint8_t*) m.w[1], m.rows[0], m.rows[1], cols, (const _Float16*) x16, n, m.y[0],
      m.y[1], m.ys[0], m.ys[1]);
  return hipGetLastError() == hipSuccess;
}

template <quant::Type T>
bool launch(const Mats& m, uint32_t cols, const void* x16, uint32_t n, hipStream_t s) {
  using S = typename Tuned<T>::S;
  for (int i = 0; i < (m.w[1] ? 2 : 1); i++)
    if (m.rows[i] % mm::kTile || (n > 1 && m.ys[i] < m.rows[i])) return false;
  if (cols % quant::Block<T>::values || cols % S::kSlab) return false;
  switch (col_tiles<T>(n)) {
    case 1: return launch_ct<T, 1>(m, cols, x16, n, s);
    case 2: return launch_ct<T, 2>(m, cols, x16, n, s);
    case 4: return launch_ct<T, 4>(m, cols, x16, n, s);
    default: return launch_ct<T, Tuned<T>::kMaxCt>(m, cols, x16, n, s);
  }
}

bool run(uint32_t type, const Mats& m, uint32_t cols, const void* x16, uint32_t n, hipStream_t s) {
  if (!n || n > MM_MAX_TOKENS || !m.rows[0] || (m.w[1] && !m.rows[1])) return false;
  switch (type) {
    case quant::kQ8_0: return launch<quant::kQ8_0>(m, cols, x16, n, s);
    case quant::kQ5_K: return launch<quant::kQ5_K>(m, cols, x16, n, s);
    case quant::kQ5_KR: return launch<quant::kQ5_KR>(m, cols, x16, n, s);
    case quant::kQ6_K: return launch<quant::kQ6_K>(m, cols, x16, n, s);
    case quant::kIQ1_M: return launch<quant::kIQ1_M>(m, cols, x16, n, s);
    case quant::kIQ2_XXS: return launch<quant::kIQ2_XXS>(m, cols, x16, n, s);
    case quant::kIQ2_XS: return launch<quant::kIQ2_XS>(m, cols, x16, n, s);
    default: return false;
  }
}

}  // namespace

// the table's types, and the repacked Q5_K the forward makes at load (no file's type: no row)
extern "C" bool mm_known(uint32_t type) { return type == quant::kQ5_KR || (qtype_paths(type) & QTYPE_MM); }

extern "C" uint32_t mm_col_tiles(uint32_t type, uint32_t n) {
  switch (type) {
#define TILES(t) \
  case quant::t: return col_tiles<quant::t>(n);
    TILES(kQ8_0) TILES(kQ5_K) TILES(kQ5_KR) TILES(kQ6_K) TILES(kIQ1_M) TILES(kIQ2_XXS) TILES(kIQ2_XS)
#undef TILES
    default: return 0;
  }
}

extern "C" bool mm_x16(const float* x, uint32_t xs, uint32_t n, uint32_t cols, void* x16, int32_t* overflow,
                       void* stream) {
  if (!n || n > MM_MAX_TOKENS || !cols || cols % mm::kTile || (n > 1 && xs < cols)) return false;
  const uint64_t work = (uint64_t) (n + mm::kTile - 1) / mm::kTile * mm::kTile * cols;  // whole tiles: past n, zero
  x16_kernel<<<(unsigned) ((work + kThreads - 1) / kThreads), kThreads, 0, (hipStream_t) stream>>>(
      x, xs, n, cols, (_Float16*) x16, overflow);
  return hipGetLastError() == hipSuccess;
}

extern "C" bool mm_x16_attn(const float* x, uint32_t xs, uint32_t n, uint32_t cols, uint32_t head_dim,
                            uint32_t rope_dim, const float* freqs, const int32_t* pos, void* x16, int32_t* overflow,
                            void* stream) {
  const uint32_t blk = rnd::kFp8Block;
  if (!n || n > MM_MAX_TOKENS || !cols || cols % mm::kTile || (n > 1 && xs < cols) || !head_dim || cols % head_dim ||
      head_dim % blk || rope_dim > head_dim || rope_dim % 2 || (head_dim - rope_dim) % blk)
    return false;
  const uint64_t work = (uint64_t) (n + mm::kTile - 1) / mm::kTile * mm::kTile * cols;
  x16_attn_kernel<<<(unsigned) ((work + kThreads - 1) / kThreads), kThreads, 0, (hipStream_t) stream>>>(
      x, xs, n, cols, head_dim, rope_dim, freqs, pos, (_Float16*) x16, overflow);
  return hipGetLastError() == hipSuccess;
}

extern "C" bool mm_run(uint32_t type, const void* w, uint32_t rows, uint32_t cols, const void* x16, uint32_t n,
                       float* y, uint32_t ys, void* stream) {
  return run(type, Mats{{w, nullptr}, {rows, 0}, {y, nullptr}, {ys, 0}}, cols, x16, n, (hipStream_t) stream);
}

extern "C" bool mm_run2(uint32_t type, const void* w0, uint32_t rows0, float* y0, uint32_t ys0, const void* w1,
                        uint32_t rows1, float* y1, uint32_t ys1, uint32_t cols, const void* x16, uint32_t n,
                        void* stream) {
  return w0 && w1 &&
         run(type, Mats{{w0, w1}, {rows0, rows1}, {y0, y1}, {ys0, ys1}}, cols, x16, n, (hipStream_t) stream);
}

extern "C" bool mm_x16_rows(const float* x, uint32_t xs, const int32_t* idx, uint32_t slots, uint32_t cols, void* x16,
                            int32_t* overflow, void* stream) {
  if (!slots || !cols || cols % mm::kTile || xs < cols) return false;
  const uint64_t work = (uint64_t) slots * cols;
  x16_rows_kernel<<<(unsigned) ((work + kThreads - 1) / kThreads), kThreads, 0, (hipStream_t) stream>>>(
      x, xs, idx, slots, cols, (_Float16*) x16, overflow);
  return hipGetLastError() == hipSuccess;
}

extern "C" bool mm_group(uint32_t type, const void* const* w, const int32_t* cnt, uint32_t n_mat, uint32_t max_cnt,
                         uint32_t rows, uint32_t cols, const void* x16, const uint32_t* xoff, uint32_t xts, float* y,
                         const uint32_t* yoff, uint32_t ys, void* stream) {
  if (!n_mat || n_mat > MM_MAX_GROUP) return false;
  const hipStream_t s = (hipStream_t) stream;
  switch (type) {
#define GROUP(t)                                                                                                     \
  case quant::t:                                                                                                     \
    return group_launch<quant::t>(w, cnt, n_mat, max_cnt, rows, cols, x16, xoff, xts, y, yoff, ys, s);
    GROUP(kQ8_0) GROUP(kQ5_K) GROUP(kQ5_KR) GROUP(kQ6_K) GROUP(kIQ1_M) GROUP(kIQ2_XXS) GROUP(kIQ2_XS)
#undef GROUP
    default: return false;
  }
}
