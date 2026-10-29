// the matmul for the quantized weights, every step's however many tokens (1 .. 16): a wave a tile of 16 rows x 16
// tokens, K in steps of 16 - the rows' 16 weights at that step decoded (quant.h: exact) and rounded to f16, one WMMA
// (f32 <- f16, 16 x 16 x 16) accumulating against the tokens' f16 activations. a column's results are the same bits
// whatever the other columns hold (measured: test_mm_gpu), so a token's are whatever it is batched with.
#pragma once
#include "quant.h"

namespace mm {

constexpr int kTile = 16;
constexpr int kWave = 32;

typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));

constexpr int kPad = 8;  // f16s past a row in LDS: rows 2 banks apart, so the reads do not collide

// a type's tile: SL values a slab, W waves a block (K split W ways), U chunks in flight a lane - the decode's latency
// is what a single token's matmul waits on, and LDS bounds the waves that hide it, so each type takes the slab and
// waves that fit it (mm.hip: measured)
template <int SL, int W, int U>
struct Shape {
  static constexpr int kSlab = SL, kWaves = W, kUnroll = U;
  static constexpr int kRowLds = SL + kPad;
  static constexpr int kChunks = kTile * SL / quant::kChunk;  // a slab's 8-value chunks
  static constexpr int kLane = kChunks / kWave;               // a lane's of them
  static_assert(kLane % U == 0, "a lane's chunks a whole number of U");
  static constexpr int kSlabHalves = W * kTile * kRowLds, kPartHalves = W * 8 * kWave * 2;
  static constexpr int kLdsHalves = kSlabHalves > kPartHalves ? kSlabHalves : kPartHalves;
};

// a block's 16-row tile (row0 ..) x the 16 token columns (token t's x at x16 + t * xts, f16) into y (token t's row r
// at y[t * ys + r], the first n tokens). K split over the block's waves: wave w the slabs w, w + W, ... - each decoded
// as the matvec decodes (lanes along a row: the loads coalesced) into the wave's LDS as f16, then read in WMMA's
// layout (lane l: row l % 16) a step of 16 at a time; the waves' tiles then summed in w's order, through the same LDS.
// the split is the shape's, never the tokens': a token's sums are the same whatever the batch
template <quant::Type T, class S>
__device__ __attribute__((always_inline)) inline void tile(const uint8_t* __restrict__ w, uint32_t row0,
                                                           uint32_t cols, const _Float16* __restrict__ x16,
                                                           uint32_t xts, uint32_t n, float* __restrict__ y,
                                                           uint32_t ys, _Float16* lds_all) {
  using B = quant::Block<T>;
  constexpr int per = B::values / quant::kChunk, SL = S::kSlab;
  const int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave, half = lane / kTile, col = lane % kTile;
  _Float16* lds = lds_all + wave * kTile * S::kRowLds;
  const uint64_t row_bytes = (uint64_t) (cols / B::values) * B::bytes;
  f8 acc = {};
  for (uint32_t k0 = wave * SL; k0 < cols; k0 += S::kWaves * SL) {
#pragma unroll 1
    for (int i0 = 0; i0 < S::kLane; i0 += S::kUnroll) {
#pragma unroll
      for (int i = i0; i < i0 + S::kUnroll; i++) {
        const int d = lane + kWave * i, r = d / (SL / quant::kChunk), c = d % (SL / quant::kChunk);
        float v[quant::kChunk];
        const uint32_t k = k0 / quant::kChunk + c;  // the chunk of the row
        quant::decode<T>(w + (uint64_t) (row0 + r) * row_bytes + (k / per) * B::bytes, (int) (k % per), v);
        h8 hv;
        for (int j = 0; j < quant::kChunk; j++) hv[j] = (_Float16) v[j];
        *(h8*) (lds + r * S::kRowLds + c * quant::kChunk) = hv;  // 16 bytes, aligned: kRowLds * 2 a multiple of 16
      }
    }
    __builtin_amdgcn_wave_barrier();  // the wave's LDS in order: its writes, then its reads
    for (int g = 0; g < SL / kTile; g++) {
      const h16 a = *(const h16*) (lds + col * S::kRowLds + g * kTile);
      const h16 bx = *(const h16*) (x16 + (uint64_t) col * xts + k0 + (uint64_t) g * kTile);
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bx, acc);
    }
    __builtin_amdgcn_wave_barrier();
  }
  __syncthreads();  // every wave's slabs done: their LDS takes the partial tiles
  float* part = (float*) lds_all;
  for (int j = 0; j < 8; j++) part[(wave * 8 + j) * kWave + lane] = acc[j];
  __syncthreads();
  if (wave != 0) return;
  for (int j = 0; j < 8; j++) {
    float s = part[j * kWave + lane];
    for (int v = 1; v < S::kWaves; v++) s += part[(v * 8 + j) * kWave + lane];
    if ((uint32_t) col < n) y[(uint64_t) col * ys + row0 + 2 * j + half] = s;
  }
}

template <quant::Type T, class S>
__global__ __launch_bounds__(kWave* S::kWaves) void kernel(const uint8_t* __restrict__ w, uint32_t cols,
                                                            const _Float16* __restrict__ x16, uint32_t n,
                                                            float* __restrict__ y, uint32_t ys) {
  __shared__ __attribute__((aligned(32))) _Float16 lds[S::kLdsHalves];
  tile<T, S>(w, blockIdx.x * kTile, cols, x16, cols, n, y, ys, lds);
}

// n matrices at once (blockIdx.y the matrix m): its w from the table, its token t's x at x16 + xoff[m] + t * xts,
// its first cnt[m] tokens' rows at y + yoff[m] + t * ys - a step's distinct experts, each read once for its tokens,
// or a layer's wo_a groups, each over its slice of the tokens' attention
template <quant::Type T, class S>
__global__ __launch_bounds__(kWave* S::kWaves) void group_kernel(const uint8_t* const* __restrict__ w,
                                                                  const int32_t* __restrict__ cnt, uint32_t cols,
                                                                  const _Float16* __restrict__ x16,
                                                                  const uint32_t* __restrict__ xoff, uint32_t xts,
                                                                  float* __restrict__ y,
                                                                  const uint32_t* __restrict__ yoff, uint32_t ys) {
  __shared__ __attribute__((aligned(32))) _Float16 lds[S::kLdsHalves];
  const uint32_t m = blockIdx.y;
  tile<T, S>(w[m], blockIdx.x * kTile, cols, x16 + xoff[m], xts, (uint32_t) cnt[m], y + yoff[m], ys, lds);
}

}  // namespace mm
