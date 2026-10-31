// the matmul for the quantized weights, every step's however many tokens (1 .. 16): a wave a tile of 16 rows x 16
// tokens, K in steps of 16 - the rows' 16 weights at that step decoded to f16 (quant.h's decode_h), one WMMA
// (f32 <- f16, 16 x 16 x 16) accumulating against the tokens' f16 activations. a column's results are the same bits
// whatever the other columns hold (measured: test_mm_gpu), so a token's are whatever it is batched with.
#pragma once
#include "quant.h"

namespace mm {

constexpr int kTile = 16;
constexpr int kWave = 32;

typedef _Float16 h16 __attribute__((ext_vector_type(16)));
using quant::h8;
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

// the activations' layout (x16): tiled as WMMA reads them - token t's column c at (t / 16) 16 xts + (c / 16) 256 +
// (t % 16) 16 + c % 16, xts the row's length: a wave's fragment (16 tokens x 16 columns) 512 contiguous bytes, so a
// load touches 4 lines, not 16 (a token's row each) - measured, the activations' loads were what bound many tokens
constexpr int kFrag = kTile * kTile;
__host__ __device__ inline uint64_t x16_at(uint32_t t, uint32_t c, uint32_t xts) {
  return (uint64_t) (t / kTile) * kTile * xts + (c / kTile) * kFrag + (t % kTile) * kTile + c % kTile;
}

// a value into the tiled x16 as mm_x16 puts it: f16, nearest; past f16's range (or NaN) the overflow flag set - every
// kernel that writes an x16 (mm_x16's own, and the ops that write theirs beside their F32 output) through this
constexpr float kF16Max = 65504.0f;
__device__ inline void x16_put(_Float16* x16, uint32_t t, uint32_t c, uint32_t xts, float v, int32_t* overflow) {
  if (!(fabsf(v) <= kF16Max)) atomicOr(overflow, 1);
  x16[x16_at(t, c, xts)] = (_Float16) v;
}

// the tiles' x double-buffered (each tile's fragments loaded under the one before's WMMAs) for a slab of few
// fragments: a tile's G fragments are 8 G VGPRs, so a second set is cheap at G 2 (Q5_K's slabs of 32: wo_a 7.24 ->
// 5.69 ms at 2048 tokens, the shared expert's gate 2.01 -> 1.87) and ruinous at 8 (the IQ experts' 128: a real layer
// 15.2 -> 22.0 ms). the order of the sums is the same either way: the same bits
template <int G, int CT>
constexpr bool kPipe = CT > 1 && G <= 2;

// one column tile (a decode's 1-16 tokens) of a type of short slabs: each lane decodes its row of WMMA's A operand
// itself (lane l: row l % 16; lanes 0-15 a step's even chunk, 16-31 its odd, the halves traded across the lane
// halves), no LDS between the decode and the WMMA - the same f16 in the same places, so the same sums (scratch
// m1.hip, one token: Q5_K's q_b 181 -> 143 us, wo_a's 8 groups 185 -> 147, the shared expert's 62 -> 60; the IQ
// experts' and Q6_K's, Q8_0's slower or the same: their slabs' LDS layout is the better)
template <quant::Type T>
constexpr bool kRegA = T == quant::kQ5_K || T == quant::kQ5_KR;

__device__ inline h8 lane_other_half(h8 v) {  // lane l ^ 16's 8 halves
  typedef uint32_t u4 __attribute__((ext_vector_type(4)));
  const u4 w = __builtin_bit_cast(u4, v);
  u4 o;
  for (int i = 0; i < 4; i++) o[i] = (uint32_t) __shfl_xor((int) w[i], kTile, kWave);
  return __builtin_bit_cast(h8, o);
}

// a block's 16-row tile (row0 ..) x CT tiles of 16 token columns, tokens c0 .. c0 + 16 CT (the tiled x16 above, xts
// its row length) into y (token t's row r at y[t * ys + r], the first n tokens). K split over the block's waves: wave
// w the slabs w, w + W, ... - each decoded as the matvec decodes (lanes along a row: the loads coalesced) into the
// wave's LDS as f16 once, read in WMMA's layout (lane l: row l % 16) a step of 16 at a time; then a column tile at a
// time its steps' x loaded together and its WMMAs in step order. the waves' tiles summed in w's order, through the
// same LDS. the split is the shape's and a column's sums are its own: a token's bits are the same whatever the batch,
// its place in it, or CT
template <quant::Type T, class S, int CT>
__device__ __attribute__((always_inline)) inline void tile(const uint8_t* __restrict__ w, uint32_t row0,
                                                           uint32_t cols, const _Float16* __restrict__ x16,
                                                           uint32_t xts, uint32_t c0, uint32_t n,
                                                           float* __restrict__ y, uint32_t ys, _Float16* lds_all) {
  constexpr int SL = S::kSlab, G = SL / kTile;
  const int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave, half = lane / kTile, col = lane % kTile;
  _Float16* lds = lds_all + wave * kTile * S::kRowLds;
  const int tiles = (int) ((n - c0 + kTile - 1) / kTile) < CT ? (int) ((n - c0 + kTile - 1) / kTile) : CT;
  const _Float16* xb[CT];  // each tile's fragments of this lane: its column, a step kFrag on
  for (int ct = 0; ct < CT; ct++) xb[ct] = x16 + x16_at(c0 + ct * kTile + col, 0, xts);
  f8 acc[CT] = {};
  if constexpr (CT == 1 && kRegA<T>) {  // the A operand straight from the decode (kRegA)
    for (uint32_t k0 = wave * SL; k0 < cols; k0 += S::kWaves * SL) {
      h16 a[G], bx[G];
#pragma unroll
      for (int g = 0; g < G; g++) {
        const uint32_t k = k0 / quant::kChunk + 2 * g + half;  // this lane's chunk of the step: the even, or the odd
        int kk;
        const uint8_t* p = quant::at<T>(w, row0 + col, cols, k, &kk);
        const h8 mine = quant::decode_h<T>(p, kk), theirs = lane_other_half(mine);
        for (int j = 0; j < quant::kChunk; j++)
          a[g][j] = half ? theirs[j] : mine[j], a[g][quant::kChunk + j] = half ? mine[j] : theirs[j];
      }
      for (int g = 0; g < G; g++) bx[g] = *(const h16*) (xb[0] + (uint64_t) (k0 / kTile + g) * kFrag);
      for (int g = 0; g < G; g++) acc[0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[g], bx[g], acc[0]);
    }
  } else
  for (uint32_t k0 = wave * SL; k0 < cols; k0 += S::kWaves * SL) {
#pragma unroll 1
    for (int i0 = 0; i0 < S::kLane; i0 += S::kUnroll) {
#pragma unroll
      for (int i = i0; i < i0 + S::kUnroll; i++) {
        const int d = lane + kWave * i, r = d / (SL / quant::kChunk), c = d % (SL / quant::kChunk);
        const uint32_t k = k0 / quant::kChunk + c;  // the chunk of the row
        int kk;
        const h8 hv = quant::decode_h<T>(quant::at<T>(w, row0 + r, cols, k, &kk), kk);
        *(h8*) (lds + r * S::kRowLds + c * quant::kChunk) = hv;  // 16 bytes, aligned: kRowLds * 2 a multiple of 16
      }
    }
    __builtin_amdgcn_wave_barrier();  // the wave's LDS in order: its writes, then its reads
    h16 a[G];
    for (int g = 0; g < G; g++) a[g] = *(const h16*) (lds + col * S::kRowLds + g * kTile);
    __builtin_amdgcn_wave_barrier();
    if (kPipe<G, CT> && tiles == CT) {  // every tile real: the next tile's x loaded under this one's WMMAs
      h16 bc[G], bn[G];
      for (int g = 0; g < G; g++) bc[g] = *(const h16*) (xb[0] + (uint64_t) (k0 / kTile + g) * kFrag);
#pragma unroll
      for (int ct = 0; ct < CT; ct++) {
        if (ct + 1 < CT)
          for (int g = 0; g < G; g++) bn[g] = *(const h16*) (xb[ct + 1] + (uint64_t) (k0 / kTile + g) * kFrag);
        for (int g = 0; g < G; g++) acc[ct] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[g], bc[g], acc[ct]);
        if (ct + 1 < CT)
          for (int g = 0; g < G; g++) bc[g] = bn[g];
      }
    } else
#pragma unroll
      for (int ct = 0; ct < CT; ct++)
        if (CT == 1 || ct < tiles) {  // the whole wave: past the tokens, nothing
          h16 bx[G];
          for (int g = 0; g < G; g++) bx[g] = *(const h16*) (xb[ct] + (uint64_t) (k0 / kTile + g) * kFrag);
          for (int g = 0; g < G; g++) acc[ct] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[g], bx[g], acc[ct]);
        }
  }
  float* part = (float*) lds_all;
#pragma unroll
  for (int ct = 0; ct < CT; ct++) {
    if (CT > 1 && ct >= tiles) break;  // the block's: every wave the same tiles
    __syncthreads();  // the LDS free: the slabs done, or the tile before summed
    for (int j = 0; j < 8; j++) part[(wave * 8 + j) * kWave + lane] = acc[ct][j];
    __syncthreads();
    if (CT == 1 && wave != 0) return;
    if (wave == 0) {
      const uint32_t t = c0 + ct * kTile + col;
      for (int j = 0; j < 8; j++) {
        float s = part[j * kWave + lane];
        for (int v = 1; v < S::kWaves; v++) s += part[(v * 8 + j) * kWave + lane];
        if (t < n) y[(uint64_t) t * ys + row0 + 2 * j + half] = s;
      }
    }
  }
}

// one matrix, or two of one type on the same x16 (blockIdx.y the matrix: a pair's blocks in one launch - the shared
// expert's gate and up, q_a and kv), each tile as alone; the grid's rows the larger's, a block past its matrix's
// rows nothing
template <quant::Type T, class S, int CT>
__global__ __launch_bounds__(kWave* S::kWaves) void kernel(const uint8_t* __restrict__ w0,
                                                            const uint8_t* __restrict__ w1, uint32_t rows0,
                                                            uint32_t rows1, uint32_t cols,
                                                            const _Float16* __restrict__ x16, uint32_t n,
                                                            float* __restrict__ y0, float* __restrict__ y1,
                                                            uint32_t ys0, uint32_t ys1) {
  __shared__ __attribute__((aligned(32))) _Float16 lds[S::kLdsHalves];
  const uint32_t row0 = blockIdx.x * kTile;
  if (row0 >= (blockIdx.y ? rows1 : rows0)) return;  // the whole block
  tile<T, S, CT>(blockIdx.y ? w1 : w0, row0, cols, x16, cols, blockIdx.z * kTile * CT, n, blockIdx.y ? y1 : y0,
                 blockIdx.y ? ys1 : ys0, lds);
}

// n matrices at once (blockIdx.y the matrix m): its w from the table, its x the tiled x16 from x16 + xoff[m] (rows xts
// long: a token tile's offset t / 16 * 16 xts, a column's c * 16),
// its first cnt[m] tokens' rows at y + yoff[m] + t * ys - a step's distinct experts, each read once for its tokens,
// or a layer's wo_a groups, each over its slice of the tokens' attention. a block past its matrix's tokens: nothing
template <quant::Type T, class S, int CT>
__global__ __launch_bounds__(kWave* S::kWaves) void group_kernel(const uint8_t* const* __restrict__ w,
                                                                  const int32_t* __restrict__ cnt, uint32_t cols,
                                                                  const _Float16* __restrict__ x16,
                                                                  const uint32_t* __restrict__ xoff, uint32_t xts,
                                                                  float* __restrict__ y,
                                                                  const uint32_t* __restrict__ yoff, uint32_t ys) {
  __shared__ __attribute__((aligned(32))) _Float16 lds[S::kLdsHalves];
  const uint32_t m = blockIdx.y, c0 = blockIdx.z * kTile * CT;
  if (c0 >= (uint32_t) cnt[m]) return;  // the whole block
  tile<T, S, CT>(w[m], blockIdx.x * kTile, cols, x16 + xoff[m], xts, c0, (uint32_t) cnt[m], y + yoff[m], ys, lds);
}

}  // namespace mm
