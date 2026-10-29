// mm (include/substrate/gpu/mm.h): the kernel is mm.h's
#include <cmath>

#include "mm.h"

extern "C" {
#include "substrate/gpu/mm.h"
}

namespace {

constexpr int kThreads = 256;

// a type's tile (mm.h's Shape): measured on gfx1151 on ds1017's matrices, one token (2026-10-29). Q5_K's decode is the
// heaviest: small slabs, so more waves hide it (q_b 88 -> 163 GB/s, the head 123 -> 157); the IQ experts' wants
// bytes in flight (SL 64, 16 waves); Q8_0 and Q6_K were near their matvec's at SL 128
template <quant::Type T> struct Tuned { using S = mm::Shape<128, 8, 8>; };
template <> struct Tuned<quant::kQ5_K> { using S = mm::Shape<32, 8, 2>; };
template <> struct Tuned<quant::kIQ1_M> { using S = mm::Shape<64, 16, 4>; };
template <> struct Tuned<quant::kIQ2_XXS> { using S = mm::Shape<64, 16, 4>; };
template <> struct Tuned<quant::kIQ2_XS> { using S = mm::Shape<64, 16, 4>; };
constexpr float kF16Max = 65504.0f;

// a thread a value: to f16 (the rows past n zero)
__global__ void x16_kernel(const float* x, uint32_t xs, uint32_t n, uint32_t cols, _Float16* x16, int32_t* overflow) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) mm::kTile * cols) return;
  const uint32_t t = (uint32_t) (i / cols), c = (uint32_t) (i % cols);
  const float v = t < n ? x[(uint64_t) t * xs + c] : 0.0f;
  if (!(fabsf(v) <= kF16Max)) atomicOr(overflow, 1);  // past f16, or NaN: refused at the step's end
  x16[i] = (_Float16) v;
}

// matrix m's slot j: row idx[m * 16 + j] of x (-1: zero), to f16
__global__ void x16_rows_kernel(const float* x, uint32_t xs, const int32_t* idx, uint32_t n_mat, uint32_t cols,
                                _Float16* x16, int32_t* overflow) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_mat * mm::kTile * cols) return;
  const int32_t r = idx[i / cols];
  const float v = r >= 0 ? x[(uint64_t) r * xs + i % cols] : 0.0f;
  if (!(fabsf(v) <= kF16Max)) atomicOr(overflow, 1);
  x16[i] = (_Float16) v;
}

template <quant::Type T>
bool group_launch(const void* const* w, const int32_t* cnt, uint32_t n_mat, uint32_t rows, uint32_t cols,
                  const void* x16, const uint32_t* xoff, uint32_t xts, float* y, const uint32_t* yoff, uint32_t ys,
                  hipStream_t s) {
  using S = typename Tuned<T>::S;
  if (rows % mm::kTile || cols % quant::Block<T>::values || cols % S::kSlab || xts < cols || xts % mm::kTile)
    return false;
  mm::group_kernel<T, S><<<dim3(rows / mm::kTile, n_mat), mm::kWave * S::kWaves, 0, s>>>(
      (const uint8_t* const*) w, cnt, cols, (const _Float16*) x16, xoff, xts, y, yoff, ys);
  return hipGetLastError() == hipSuccess;
}

template <quant::Type T>
bool launch(const void* w, uint32_t rows, uint32_t cols, const void* x16, uint32_t n, float* y, uint32_t ys,
            hipStream_t s) {
  using S = typename Tuned<T>::S;
  if (rows % mm::kTile || cols % quant::Block<T>::values || cols % S::kSlab) return false;
  mm::kernel<T, S><<<rows / mm::kTile, mm::kWave * S::kWaves, 0, s>>>((const uint8_t*) w, cols, (const _Float16*) x16,
                                                                     n, y, ys);
  return hipGetLastError() == hipSuccess;
}

}  // namespace

extern "C" bool mm_known(uint32_t type) {
  switch (type) {
    case quant::kQ8_0: case quant::kQ5_K: case quant::kQ6_K: case quant::kIQ1_M: case quant::kIQ2_XXS:
    case quant::kIQ2_XS: return true;
    default: return false;
  }
}

extern "C" bool mm_x16(const float* x, uint32_t xs, uint32_t n, uint32_t cols, void* x16, int32_t* overflow,
                       void* stream) {
  if (!n || n > MM_TILE || !cols || cols % mm::kTile || (n > 1 && xs < cols)) return false;
  const uint64_t work = (uint64_t) mm::kTile * cols;
  x16_kernel<<<(unsigned) ((work + kThreads - 1) / kThreads), kThreads, 0, (hipStream_t) stream>>>(
      x, xs, n, cols, (_Float16*) x16, overflow);
  return hipGetLastError() == hipSuccess;
}

extern "C" bool mm_run(uint32_t type, const void* w, uint32_t rows, uint32_t cols, const void* x16, uint32_t n,
                       float* y, uint32_t ys, void* stream) {
  if (!n || n > MM_TILE || (n > 1 && ys < rows)) return false;
  const hipStream_t s = (hipStream_t) stream;
  switch (type) {
    case quant::kQ8_0: return launch<quant::kQ8_0>(w, rows, cols, x16, n, y, ys, s);
    case quant::kQ5_K: return launch<quant::kQ5_K>(w, rows, cols, x16, n, y, ys, s);
    case quant::kQ6_K: return launch<quant::kQ6_K>(w, rows, cols, x16, n, y, ys, s);
    case quant::kIQ1_M: return launch<quant::kIQ1_M>(w, rows, cols, x16, n, y, ys, s);
    case quant::kIQ2_XXS: return launch<quant::kIQ2_XXS>(w, rows, cols, x16, n, y, ys, s);
    case quant::kIQ2_XS: return launch<quant::kIQ2_XS>(w, rows, cols, x16, n, y, ys, s);
    default: return false;
  }
}

extern "C" bool mm_x16_rows(const float* x, uint32_t xs, const int32_t* idx, uint32_t n_mat, uint32_t cols, void* x16,
                            int32_t* overflow, void* stream) {
  if (!n_mat || !cols || cols % mm::kTile || xs < cols) return false;
  const uint64_t work = (uint64_t) n_mat * mm::kTile * cols;
  x16_rows_kernel<<<(unsigned) ((work + kThreads - 1) / kThreads), kThreads, 0, (hipStream_t) stream>>>(
      x, xs, idx, n_mat, cols, (_Float16*) x16, overflow);
  return hipGetLastError() == hipSuccess;
}

extern "C" bool mm_group(uint32_t type, const void* const* w, const int32_t* cnt, uint32_t n_mat, uint32_t rows,
                         uint32_t cols, const void* x16, const uint32_t* xoff, uint32_t xts, float* y,
                         const uint32_t* yoff, uint32_t ys, void* stream) {
  if (!n_mat || n_mat > MM_MAX_GROUP) return false;
  const hipStream_t s = (hipStream_t) stream;
  switch (type) {
#define GROUP(t) \
  case quant::t: return group_launch<quant::t>(w, cnt, n_mat, rows, cols, x16, xoff, xts, y, yoff, ys, s);
    GROUP(kQ8_0) GROUP(kQ5_K) GROUP(kQ6_K) GROUP(kIQ1_M) GROUP(kIQ2_XXS) GROUP(kIQ2_XS)
#undef GROUP
    default: return false;
  }
}
