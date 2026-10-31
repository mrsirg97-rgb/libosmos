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
__device__ inline void step(const uint8_t* const* base, const uint32_t* rid, uint32_t c0, uint32_t xs,
                            const float* __restrict__ x, float (&acc)[R][NT]) {
  float4 xa[U][NT], xb[U][NT];
  for (int u = 0; u < U; u++)
    for (int t = 0; t < NT; t++) {
      const float4* xp = (const float4*) (x + (uint64_t) t * xs + (uint64_t) (c0 + u * kWave) * quant::kChunk);
      xa[u][t] = xp[0], xb[u][t] = xp[1];
    }
  float v[U][R][quant::kChunk];
  for (int u = 0; u < U; u++) {
    const uint32_t c = c0 + u * kWave;
    for (int q = 0; q < R; q++) {
      int k;
      const uint8_t* b = quant::chunk_at<T>(base[q], rid[q], c, &k);
      quant::decode<T>(b, k, v[u][q]);
    }
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
                               uint32_t xs, float* __restrict__ y, uint32_t ys, uint32_t live = NT) {
  const uint32_t row0 = (blockIdx.x * kWavesPerBlock + threadIdx.x / kWave) * R;
  const int lane = threadIdx.x % kWave;
  if (row0 >= rows) return;  // the whole wave: no lane of it is left to a shuffle
  const uint8_t* base[R];
  uint32_t rid[R];
  for (int q = 0; q < R; q++)  // past the end: unused
    rid[q] = row0 + q < rows ? row0 + q : row0, base[q] = quant::row_base<T>(w, rid[q], cols);
  const uint32_t chunks = cols / quant::kChunk;
  float acc[R][NT];
  for (int q = 0; q < R; q++)
    for (int t = 0; t < NT; t++) acc[q][t] = 0.0f;
  // whole groups of U chunks, nothing branching inside one, so all U's loads (x's first: they depend on nothing) go
  // out before any is waited for; then the rest one chunk at a time. either way each row's chunks in order
  const uint32_t whole = chunks / (kWave * U) * (kWave * U);
  for (uint32_t c0 = lane; c0 < whole; c0 += kWave * U) step<T, NT, R, U>(base, rid, c0, xs, x, acc);
  for (uint32_t c0 = whole + lane; c0 < chunks; c0 += kWave) step<T, NT, R, 1>(base, rid, c0, xs, x, acc);
  for (int q = 0; q < R; q++)
    for (int t = 0; t < NT; t++) {
      float s = acc[q][t];
      for (int o = kWave / 2; o > 0; o >>= 1) s += __shfl_xor(s, o, kWave);  // a + b = b + a: every lane the same
      if (lane == 0 && row0 + q < rows && (uint32_t) t < live) y[(uint64_t) t * ys + row0 + q] = s;
    }
}

template <quant::Type T, int NT, int R, int U>
__global__ __launch_bounds__(kWave* kWavesPerBlock) void kernel(const uint8_t* __restrict__ w, uint32_t rows,
                                                                 uint32_t cols, const float* __restrict__ x,
                                                                 uint32_t xs, float* __restrict__ y, uint32_t ys) {
  rows_of<T, NT, R, U>(w, rows, cols, x, xs, y, ys);
}

// many tokens at once (a prompt's): a wave R rows of NW tokens, the block's 8 waves 8 R rows of the same NW tokens,
// their x shared through LDS a step of 32 chunks at a time. lane l still sums its chunks l, l + 32, ... of each row in
// order, 8 fmaf a chunk, then the tree - rows_of's sums, so a token's bits are its own alone. blockIdx.y a tile of NW
// tokens; a tile past n_tok computes its last token again in the spare places and stores none of them
template <quant::Type T, int R, int NW>
__global__ __launch_bounds__(kWave* kWavesPerBlock) void wide_kernel(const uint8_t* __restrict__ w, uint32_t rows,
                                                                      uint32_t cols, const float* __restrict__ x,
                                                                      uint32_t xs, uint32_t n_tok,
                                                                      float* __restrict__ y, uint32_t ys) {
  constexpr int threads = kWave * kWavesPerBlock;
  __shared__ float4 xl[NW][2][kWave];  // [token][half of a chunk][chunk of the step]: a lane's float4s side by side
  const uint32_t t0 = blockIdx.y * NW, row0 = (blockIdx.x * kWavesPerBlock + threadIdx.x / kWave) * R;
  const int lane = threadIdx.x % kWave;
  const uint8_t* base[R];  // a row past the end reads row 0 and stores nothing (the wave still joins the barriers)
  uint32_t rid[R];
  for (int q = 0; q < R; q++) rid[q] = row0 + q < rows ? row0 + q : 0, base[q] = quant::row_base<T>(w, rid[q], cols);
  const uint32_t chunks = cols / quant::kChunk;
  float acc[R][NW];
  for (int q = 0; q < R; q++)
    for (int t = 0; t < NW; t++) acc[q][t] = 0.0f;
  for (uint32_t s = 0; s < chunks; s += kWave) {
    for (int i = threadIdx.x; i < NW * 2 * kWave; i += threads) {
      const int t = i / (2 * kWave), c = (i / 2) % kWave, h = i % 2;
      const uint32_t tok = t0 + t < n_tok ? t0 + t : n_tok - 1;
      const float* xp = x + (uint64_t) tok * xs + (uint64_t) (s + c) * quant::kChunk;
      xl[t][h][c] = s + c < chunks ? ((const float4*) xp)[h] : float4{0, 0, 0, 0};
    }
    __syncthreads();
    const uint32_t c = s + lane;
    if (c < chunks) {
      float v[R][quant::kChunk];
      for (int q = 0; q < R; q++) {
        int k;
        const uint8_t* b = quant::chunk_at<T>(base[q], rid[q], c, &k);
        quant::decode<T>(b, k, v[q]);
      }
      for (int t = 0; t < NW; t++) {
        const float4 a = xl[t][0][lane], b = xl[t][1][lane];
        for (int q = 0; q < R; q++) {
          const float* vv = v[q];
          float m = acc[q][t];
          m = fmaf(vv[0], a.x, m), m = fmaf(vv[1], a.y, m), m = fmaf(vv[2], a.z, m), m = fmaf(vv[3], a.w, m);
          m = fmaf(vv[4], b.x, m), m = fmaf(vv[5], b.y, m), m = fmaf(vv[6], b.z, m), m = fmaf(vv[7], b.w, m);
          acc[q][t] = m;
        }
      }
    }
    __syncthreads();
  }
  for (int q = 0; q < R; q++)
    for (int t = 0; t < NW; t++) {
      float m = acc[q][t];
      for (int o = kWave / 2; o > 0; o >>= 1) m += __shfl_xor(m, o, kWave);
      if (lane == 0 && row0 + q < rows && t0 + t < n_tok) y[(uint64_t) (t0 + t) * ys + row0 + q] = m;
    }
}

template <quant::Type T, int R, int NW>
bool wide_launch(const void* w, uint32_t rows, uint32_t cols, const float* x, uint32_t xs, uint32_t n_tok, float* y,
                 uint32_t ys, hipStream_t s) {
  const uint32_t per_block = kWavesPerBlock * R;
  wide_kernel<T, R, NW><<<dim3((rows + per_block - 1) / per_block, (n_tok + NW - 1) / NW), kWave * kWavesPerBlock, 0,
                          s>>>((const uint8_t*) w, rows, cols, x, xs, n_tok, y, ys);
  return hipGetLastError() == hipSuccess;
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

// matrices of one shape, each with its own few tokens (blockIdx.y the matrix m): its first cnt[m] (<= NT) tokens'
// rows from row xrow[m] of x and of y - an expert once for every token that picked it (forward_lane's verify). its
// rows NT .. past cnt[m] are summed and not stored (x is read there: the caller's rows past the last are its own);
// each row the sums of rows_of, so the bits of a token alone
template <quant::Type T, int NT, int R, int U>
__global__ __launch_bounds__(kWave* kWavesPerBlock) void group_tok_kernel(const uint8_t* const* __restrict__ w,
                                                                           const int32_t* __restrict__ cnt,
                                                                           const int32_t* __restrict__ xrow,
                                                                           uint32_t rows, uint32_t cols,
                                                                           const float* __restrict__ x, uint32_t xs,
                                                                           float* __restrict__ y, uint32_t ys) {
  const uint32_t m = blockIdx.y;
  const uint64_t r0 = (uint64_t) xrow[m];
  rows_of<T, NT, R, U>(w[m], rows, cols, x + r0 * xs, xs, y + r0 * ys, ys, (uint32_t) cnt[m]);
}

template <quant::Type T, int NT, int R, int U>
bool launch(const void* w, uint32_t rows, uint32_t cols, const float* x, uint32_t xs, float* y, uint32_t ys,
            hipStream_t s) {
  const uint32_t per_block = kWavesPerBlock * R;
  kernel<T, NT, R, U><<<(rows + per_block - 1) / per_block, kWave * kWavesPerBlock, 0, s>>>(
      (const uint8_t*) w, rows, cols, x, xs, y, ys);
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
