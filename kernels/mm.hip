// mm (include/substrate/gpu/mm.h): the kernel is mm.h's
#include <cmath>

#include "mm.h"

extern "C" {
#include "substrate/gpu/mm.h"
}

namespace {

constexpr int kThreads = 256;
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

template <quant::Type T>
bool launch(const void* w, uint32_t rows, uint32_t cols, const void* x16, uint32_t n, float* y, uint32_t ys,
            hipStream_t s) {
  if (rows % mm::kTile || cols % quant::Block<T>::values || cols % mm::kSlab) return false;
  mm::kernel<T><<<rows / mm::kTile, mm::kWave * mm::kWavesPerBlock, 0, s>>>((const uint8_t*) w, cols,
                                                                           (const _Float16*) x16, n, y, ys);
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
