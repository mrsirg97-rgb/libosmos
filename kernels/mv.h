// the decode's matrix-vector products (mv.hip; include/substrate/gpu/mv.h): a wave R rows, a lane its 8-value chunks
// lane, lane + 32, lane + 64, ... of each, in that order, U of them loaded before any is summed. a row's sum is the
// same sequence of fmaf whatever R, U and NT - they only change what is in flight - so its bits are too.
#pragma once
#include "quant.h"

namespace mv {

constexpr int kWave = 32;
constexpr int kWavesPerBlock = 8;

// U chunks c0, c0 + 32, ... of R rows into acc, for NT tokens: x loaded, then the weights decoded, then summed
template <quant::Type T, int NT, int R, int U>
__device__ inline void step(const uint8_t* const* base, uint32_t c0, uint32_t xs, const float* __restrict__ x,
                            float (&acc)[R][NT]) {
  using B = quant::Block<T>;
  constexpr int per = B::values / quant::kChunk;
  float4 xa[U][NT], xb[U][NT];
  for (int u = 0; u < U; u++)
    for (int t = 0; t < NT; t++) {
      const float4* xp = (const float4*) (x + (uint64_t) t * xs + (uint64_t) (c0 + u * kWave) * quant::kChunk);
      xa[u][t] = xp[0], xb[u][t] = xp[1];
    }
  float v[U][R][quant::kChunk];
  for (int u = 0; u < U; u++) {
    const uint32_t c = c0 + u * kWave;
    for (int q = 0; q < R; q++) quant::decode<T>(base[q] + (c / per) * B::bytes, (int) (c % per), v[u][q]);
  }
  for (int u = 0; u < U; u++)
    for (int t = 0; t < NT; t++) {
      const float4 a = xa[u][t], b = xb[u][t];
      for (int q = 0; q < R; q++) {
        const float* vv = v[u][q];
        float s = acc[q][t];
        s = fmaf(vv[0], a.x, s), s = fmaf(vv[1], a.y, s), s = fmaf(vv[2], a.z, s), s = fmaf(vv[3], a.w, s);
        s = fmaf(vv[4], b.x, s), s = fmaf(vv[5], b.y, s), s = fmaf(vv[6], b.z, s), s = fmaf(vv[7], b.w, s);
        acc[q][t] = s;
      }
    }
}

template <quant::Type T, int NT, int R, int U>
__device__ inline void rows_of(const uint8_t* __restrict__ w, uint32_t rows, uint32_t cols, const float* __restrict__ x,
                               uint32_t xs, float* __restrict__ y, uint32_t ys) {
  using B = quant::Block<T>;
  const uint32_t row0 = (blockIdx.x * kWavesPerBlock + threadIdx.x / kWave) * R;
  const int lane = threadIdx.x % kWave;
  if (row0 >= rows) return;  // the whole wave: no lane of it is left to a shuffle
  const uint64_t row_bytes = (uint64_t) (cols / B::values) * B::bytes;
  const uint8_t* base[R];
  for (int q = 0; q < R; q++) base[q] = w + (row0 + q < rows ? row0 + q : row0) * row_bytes;  // past the end: unused
  const uint32_t chunks = cols / quant::kChunk;
  float acc[R][NT];
  for (int q = 0; q < R; q++)
    for (int t = 0; t < NT; t++) acc[q][t] = 0.0f;
  // whole groups of U chunks, nothing branching inside one, so all U's loads (x's first: they depend on nothing) go
  // out before any is waited for; then the rest one chunk at a time. either way each row's chunks in order
  const uint32_t whole = chunks / (kWave * U) * (kWave * U);
  for (uint32_t c0 = lane; c0 < whole; c0 += kWave * U) step<T, NT, R, U>(base, c0, xs, x, acc);
  for (uint32_t c0 = whole + lane; c0 < chunks; c0 += kWave) step<T, NT, R, 1>(base, c0, xs, x, acc);
  for (int q = 0; q < R; q++)
    for (int t = 0; t < NT; t++) {
      float s = acc[q][t];
      for (int o = kWave / 2; o > 0; o >>= 1) s += __shfl_xor(s, o, kWave);  // a + b = b + a: every lane the same
      if (lane == 0 && row0 + q < rows) y[(uint64_t) t * ys + row0 + q] = s;
    }
}

template <quant::Type T, int NT, int R, int U>
__global__ __launch_bounds__(kWave* kWavesPerBlock) void kernel(const uint8_t* __restrict__ w, uint32_t rows,
                                                                 uint32_t cols, const float* __restrict__ x,
                                                                 uint32_t xs, float* __restrict__ y, uint32_t ys) {
  rows_of<T, NT, R, U>(w, rows, cols, x, xs, y, ys);
}

// n matrices of the same shape and type at once (blockIdx.y the matrix): matrix m's w from the table, its x at
// x + (m / x_div) * x_mat (x_mat 0: one x for all), its y at y + m * rows; each row the sums of rows_of, so the bits
// of kernel's
template <quant::Type T, int R, int U>
__global__ __launch_bounds__(kWave* kWavesPerBlock) void group_kernel(const uint8_t* const* __restrict__ w,
                                                                       uint32_t rows, uint32_t cols,
                                                                       const float* __restrict__ x, uint32_t x_mat,
                                                                       uint32_t x_div, float* __restrict__ y) {
  const uint32_t m = blockIdx.y;
  rows_of<T, 1, R, U>(w[m], rows, cols, x + (uint64_t) (m / x_div) * x_mat, cols, y + (uint64_t) m * rows, rows);
}

template <quant::Type T, int NT, int R, int U>
bool launch(const void* w, uint32_t rows, uint32_t cols, const float* x, uint32_t xs, float* y, uint32_t ys,
            hipStream_t s) {
  const uint32_t per_block = kWavesPerBlock * R;
  kernel<T, NT, R, U><<<(rows + per_block - 1) / per_block, kWave * kWavesPerBlock, 0, s>>>((const uint8_t*) w, rows,
                                                                                            cols, x, xs, y, ys);
  return hipGetLastError() == hipSuccess;
}

template <quant::Type T, int R, int U>
bool group_launch(const void* const* w, uint32_t n, uint32_t rows, uint32_t cols, const float* x, uint32_t x_mat,
                  uint32_t x_div, float* y, hipStream_t s) {
  const uint32_t per_block = kWavesPerBlock * R;
  group_kernel<T, R, U><<<dim3((rows + per_block - 1) / per_block, n), kWave * kWavesPerBlock, 0, s>>>(
      (const uint8_t* const*) w, rows, cols, x, x_mat, x_div, y);
  return hipGetLastError() == hipSuccess;
}

}  // namespace mv
