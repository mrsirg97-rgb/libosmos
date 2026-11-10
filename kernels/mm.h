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

// one column tile (a decode's 1-16 tokens): each lane decodes its row of WMMA's A operand itself (lane l: row l % 16;
// lanes 0-15 a step's even chunk, 16-31 its odd, the halves traded across the lane halves), no LDS between the
// decode and the WMMA - the same f16 in the same places, so the same sums. measured against the slab through LDS at a
// token, each matrix read from DRAM (dec1 / dec_ds / exp1, low-power, 2026-11-07): Q6_K's long rows (128-value
// slabs) +6-16% (Qwen's DeltaNet pair 183 -> 207 GB/s, attn_q 172 -> 200, its logits 179 -> 203; DeepSeek's wo_b
// 193 -> 204, wo_a 191 -> 207), Q8_0 +13% (Qwen's hc mixers 156 / 147 -> 178 / 166), IQ4_NL +7% (142 -> 151),
// IQ4_XS +5%; Q6_K's short rows (64-value slabs: DeepSeek's q_b, 1,280 columns) -7%, IQ2_S -10%, IQ3_XXS and Q2_0
// level - those keep the LDS's. a token's dense reads Qwen 23.2 -> 20.7 ms, DeepSeek 34.2 -> 33.1 (Q5_K: m1.hip's
// q_b 181 -> 143 us, its first). UD-Q4_K_XL's expert types (mmsweep q4, a token's 10 experts cold, low-power,
// 2026-11-09, two runs): Q4_K at its 128 x 8 x 8 103-107 -> 78-80 us, Q5_1 111 -> 72-75; a 512-token prompt 247 ->
// 409 tok/s, its ids' hash the same
template <quant::Type T>
constexpr bool kRegA = T == quant::kQ5_K || T == quant::kQ5_KR || T == quant::kQ6_K || T == quant::kQ8_0 ||
                       T == quant::kIQ4_NL || T == quant::kIQ4_XS || T == quant::kQ4_K || T == quant::kQ5_1;
template <quant::Type T, class S>
constexpr bool kRegAShape = kRegA<T> && !(T == quant::kQ6_K && S::kSlab < 128);
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
// an epilogue's: y = silu(y scale) - q4x_scale_silu's expression - and its f16 into x16 (cols wide), for the next
// matmul (the mixers' down, then up: one launch fewer a mixer)
struct Silu {
  float scale;
  _Float16* x16;
  int32_t* overflow;
  uint32_t cols;
};

template <quant::Type T, class S, int CT, bool E = false>
__device__ __attribute__((always_inline)) inline void tile(const uint8_t* __restrict__ w, uint32_t row0,
                                                           uint32_t cols, const _Float16* __restrict__ x16,
                                                           uint32_t xts, uint32_t c0, uint32_t n,
                                                           float* __restrict__ y, uint32_t ys, _Float16* lds_all,
                                                           Silu ep = {}) {
  constexpr int SL = S::kSlab, G = SL / kTile;
  const int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave, half = lane / kTile, col = lane % kTile;
  _Float16* lds = lds_all + wave * kTile * S::kRowLds;
  const int tiles = (int) ((n - c0 + kTile - 1) / kTile) < CT ? (int) ((n - c0 + kTile - 1) / kTile) : CT;
  const _Float16* xb[CT];  // each tile's fragments of this lane: its column, a step kFrag on
  for (int ct = 0; ct < CT; ct++) xb[ct] = x16 + x16_at(c0 + ct * kTile + col, 0, xts);
  f8 acc[CT] = {};
  if constexpr (CT == 1 && kRegAShape<T, S>) {  // the A operand straight from the decode (kRegA)
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
        if constexpr (E) {
          const float u = s * ep.scale;
          s = u / (1.0f + expf(-u));
          if (t < n) x16_put(ep.x16, t, row0 + 2 * j + half, ep.cols, s, ep.overflow);
        }
        if (t < n) y[(uint64_t) t * ys + row0 + 2 * j + half] = s;
      }
    }
  }
}

// a prompt's GEMM (cols >= the shape's W slabs, many tokens): a block BM = 16 RM WR rows x BN = 16 RN WC tokens, its
// 256 threads' waves WR x WC, wave (wr, wc) the RM x RN 16 x 16 tiles at rows 16 RM wr .., tokens 16 RN wc ... each
// slab (BM rows x SL values of K) decoded once into LDS by the block - every wave's A - and each wave's x fragments
// its own; two slabs' LDS, the next decoded under this one's WMMAs, a barrier a slab. the order is tile's: partition p (tile's wave p) its slabs p, p + W, ... each in its 16-value steps, one
// WMMA chain a 16 x 16 tile from zero; the partitions in turn, a running sum (run = p 0's, then run += p's) - tile's
// part[0] + part[1] + ... - so a token's bits are tile's, whatever the batch (test_mm_gpu)
template <quant::Type T, class S, int RM, int RN, int WR, int WC>
__global__ __launch_bounds__(256) void gemm_kernel(const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
                                                    uint32_t rows0, uint32_t rows1, uint32_t cols,
                                                    const _Float16* __restrict__ x16, uint32_t n,
                                                    float* __restrict__ y0, float* __restrict__ y1, uint32_t ys0,
                                                    uint32_t ys1) {
  static_assert(WR * WC * kWave == 256, "8 waves");
  constexpr int SL = S::kSlab, G = SL / kTile, W = S::kWaves, BM = kTile * RM * WR, BN = kTile * RN * WC;
  constexpr int RL = SL + kPad, CPR = SL / quant::kChunk, CH = BM * CPR / 256;  // a thread's chunks a slab
  static_assert((BM * CPR) % 256 == 0, "a slab's chunks whole");
  __shared__ __attribute__((aligned(32))) _Float16 lds2[2][BM * RL];
  const uint8_t* w = blockIdx.y ? w1 : w0;
  const uint32_t rows = blockIdx.y ? rows1 : rows0, ys = blockIdx.y ? ys1 : ys0;
  float* y = blockIdx.y ? y1 : y0;
  const uint32_t row0 = blockIdx.x * BM, c0 = blockIdx.z * BN;
  if (row0 >= rows) return;  // the whole block
  const int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave, half = lane / kTile, col = lane % kTile;
  const int wr = wave / WC, wc = wave % WC;
  const _Float16* xb[RN];
  bool live[RN];
  for (int j = 0; j < RN; j++) {
    const uint32_t t0 = c0 + (wc * RN + j) * kTile;
    live[j] = t0 < n, xb[j] = x16 + x16_at(t0 + col, 0, cols);
  }
  f8 run[RM][RN], acc[RM][RN];
  const uint32_t slabs = cols / SL;  // >= W (the launch's): every partition a slab at least
  auto decode = [&](uint32_t k0, h8* out) {
#pragma unroll
    for (int c = 0; c < CH; c++) {
      const int d = threadIdx.x + 256 * c, r = d / CPR, ck = d % CPR;
      const uint32_t row = row0 + r < rows ? row0 + r : rows - 1;
      int kk;
      out[c] = quant::decode_h<T>(quant::at<T>(w, row, cols, k0 / quant::kChunk + ck, &kk), kk);
    }
  };
  auto put = [&](_Float16* buf, const h8* v) {
#pragma unroll
    for (int c = 0; c < CH; c++) {
      const int d = threadIdx.x + 256 * c, r = d / CPR, ck = d % CPR;
      *(h8*) (buf + r * RL + ck * quant::kChunk) = v[c];
    }
  };
  for (int i = 0; i < RM; i++)
    for (int j = 0; j < RN; j++) acc[i][j] = f8{};
  {
    h8 v[CH];
    decode(0, v);
    put(lds2[0], v);
  }
  __syncthreads();
  int p = 0, cur = 0;
  uint32_t sl = 0;
  for (;;) {
    const uint32_t k0 = sl * SL;
    // the slab after this one in the order: the partition's next, else the next partition's first
    const bool more_p = sl + W < slabs;
    const int np = more_p ? p : p + 1;
    const uint32_t nsl = more_p ? sl + W : (uint32_t) np;
    const bool next = np < W;
    h8 v[CH];
    if (next) decode(nsl * SL, v);
    const _Float16* lds = lds2[cur];
#pragma unroll
    for (int g = 0; g < G; g++) {
      h16 a[RM], b[RN];
      for (int i = 0; i < RM; i++) a[i] = *(const h16*) (lds + ((wr * RM + i) * kTile + col) * RL + g * kTile);
      for (int j = 0; j < RN; j++) b[j] = *(const h16*) (xb[j] + (uint64_t) (k0 / kTile + g) * kFrag);
      for (int j = 0; j < RN; j++)
        if (live[j])
          for (int i = 0; i < RM; i++) acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[i], b[j], acc[i][j]);
    }
    if (!more_p) {  // the partition's sum done: into the running sum, the next from zero
      for (int i = 0; i < RM; i++)
        for (int j = 0; j < RN; j++) {
          if (p == 0) run[i][j] = acc[i][j];
          else
            for (int e = 0; e < 8; e++) run[i][j][e] += acc[i][j][e];
          acc[i][j] = f8{};
        }
    }
    if (!next) break;
    put(lds2[cur ^ 1], v);  // the other buffer: read last before the barrier ending the slab before
    __syncthreads();
    cur ^= 1, p = np, sl = nsl;
  }
  for (int j = 0; j < RN; j++) {
    const uint32_t t = c0 + (wc * RN + j) * kTile + col;
    if (t >= n) continue;
    for (int i = 0; i < RM; i++) {
      const uint32_t r = row0 + (wr * RM + i) * kTile;
      if (r >= rows) continue;
      for (int e = 0; e < 8; e++) y[(uint64_t) t * ys + r + 2 * e + half] = run[i][j][e];
    }
  }
}

// RM row tiles a wave, a multi-tile step (the experts' prompt-sized matmul): tile's partition a wave (slabs w, w + W,
// ...), its slab decoded for 16 RM rows into the wave's LDS, each x fragment then feeding RM WMMAs - tile loads a
// fragment a WMMA. every output's sums in tile's order (its partition's slabs in turn, each its 16-value steps; the
// partitions summed in w's order): tile's bits (test_mm_gpu, mm_group at 37 / 64 / 128 tokens). x double-buffered as
// tile's kPipe. LDS: the slabs' 16 RM rows a wave, or the partial sums RM times tile's
// the chunks a lane decodes together (quant::decode_hn): Q4_K's 2 of a group (its scales once), else 1 - when a
// slab's row (CPR chunks) and a lane's share (CH) both hold whole runs
#ifndef RUN_MAX
#define RUN_MAX 4
#endif
template <quant::Type T, int CPR, int CH>
constexpr int run_chunks() {
  return T != quant::kQ4_K ? 1 : CPR % RUN_MAX == 0 && CH % RUN_MAX == 0 ? RUN_MAX : CPR % 2 == 0 && CH % 2 == 0 ? 2 : 1;
}

template <class S, int RM>
struct RmLds {
  static constexpr int kSlab = S::kWaves * RM * kTile * S::kRowLds, kPart = S::kWaves * RM * 8 * kWave * 2;
  static constexpr int kHalves = kSlab > kPart ? kSlab : kPart;
};
template <quant::Type T, class S, int CT, int RM, bool PIPE = false>
__device__ __attribute__((always_inline)) inline void tile_rm(const uint8_t* __restrict__ w, uint32_t row0,
                                                              uint32_t cols, const _Float16* __restrict__ x16,
                                                              uint32_t xts, uint32_t c0, uint32_t n,
                                                              float* __restrict__ y, uint32_t ys, _Float16* lds_all) {
  static_assert(CT > 1 && RM > 1, "tile's own for a column tile or a row tile");
  constexpr int SL = S::kSlab, G = SL / kTile, CPR = SL / quant::kChunk, CH = RM * S::kLane;
  constexpr int RN_ = run_chunks<T, CPR, CH>();
  const int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave, half = lane / kTile, col = lane % kTile;
  _Float16* lds = lds_all + wave * RM * kTile * S::kRowLds;
  const int tiles = (int) ((n - c0 + kTile - 1) / kTile) < CT ? (int) ((n - c0 + kTile - 1) / kTile) : CT;
  const _Float16* xb[CT];
  for (int ct = 0; ct < CT; ct++) xb[ct] = x16 + x16_at(c0 + ct * kTile + col, 0, xts);
  f8 acc[RM][CT] = {};
  for (uint32_t k0 = wave * SL; k0 < cols; k0 += S::kWaves * SL) {
#pragma unroll
    for (int i = 0; i < CH / RN_; i++) {  /* the whole slab's chunks in flight: RM times tile's; a lane RN_ of a row's */
      const int d = lane + kWave * i, r = d / (CPR / RN_), c = (d % (CPR / RN_)) * RN_;
      int kk;
      h8 hv[RN_];
      quant::decode_hn<T, RN_>(quant::at<T>(w, row0 + r, cols, k0 / quant::kChunk + c, &kk), kk, hv);
      for (int u = 0; u < RN_; u++) *(h8*) (lds + r * S::kRowLds + (c + u) * quant::kChunk) = hv[u];
    }
    __builtin_amdgcn_wave_barrier();
    h16 a[RM][G];
    for (int rm = 0; rm < RM; rm++)
      for (int g = 0; g < G; g++) a[rm][g] = *(const h16*) (lds + (rm * kTile + col) * S::kRowLds + g * kTile);
    __builtin_amdgcn_wave_barrier();
    if (PIPE && kPipe<G, CT> && tiles == CT) {  // every tile real: the next tile's x loaded under this one's WMMAs
      h16 bc[G], bn[G];
      for (int g = 0; g < G; g++) bc[g] = *(const h16*) (xb[0] + (uint64_t) (k0 / kTile + g) * kFrag);
#pragma unroll
      for (int ct = 0; ct < CT; ct++) {
        if (ct + 1 < CT)
          for (int g = 0; g < G; g++) bn[g] = *(const h16*) (xb[ct + 1] + (uint64_t) (k0 / kTile + g) * kFrag);
        for (int rm = 0; rm < RM; rm++)
          for (int g = 0; g < G; g++) acc[rm][ct] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[rm][g], bc[g], acc[rm][ct]);
        if (ct + 1 < CT)
          for (int g = 0; g < G; g++) bc[g] = bn[g];
      }
    } else
#pragma unroll
      for (int ct = 0; ct < CT; ct++)
        if (ct < tiles) {
          h16 bx[G];
          for (int g = 0; g < G; g++) bx[g] = *(const h16*) (xb[ct] + (uint64_t) (k0 / kTile + g) * kFrag);
          for (int rm = 0; rm < RM; rm++)
            for (int g = 0; g < G; g++) acc[rm][ct] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[rm][g], bx[g], acc[rm][ct]);
        }
  }
  float* part = (float*) lds_all;
#pragma unroll
  for (int ct = 0; ct < CT; ct++) {
    if (ct >= tiles) break;  // the block's: every wave the same tiles
    __syncthreads();         // the LDS free: the slabs done, or the tile before summed
    for (int rm = 0; rm < RM; rm++)
      for (int j = 0; j < 8; j++) part[((wave * RM + rm) * 8 + j) * kWave + lane] = acc[rm][ct][j];
    __syncthreads();
    if (wave == 0) {
      const uint32_t t = c0 + ct * kTile + col;
      for (int rm = 0; rm < RM; rm++)
        for (int j = 0; j < 8; j++) {
          float s = part[(rm * 8 + j) * kWave + lane];
          for (int v = 1; v < S::kWaves; v++) s += part[((v * RM + rm) * 8 + j) * kWave + lane];
          if (t < n) y[(uint64_t) t * ys + row0 + rm * kTile + 2 * j + half] = s;
        }
    }
  }
}

// group_kernel with RM row tiles a wave (tile_rm): its grid's rows RM times fewer
template <quant::Type T, class S, int CT, int RM, bool PIPE = false>
__global__ __launch_bounds__(kWave* S::kWaves) void group_rm_kernel(const uint8_t* const* __restrict__ w,
                                                                     const int32_t* __restrict__ cnt, uint32_t cols,
                                                                     const _Float16* __restrict__ x16,
                                                                     const uint32_t* __restrict__ xoff, uint32_t xts,
                                                                     float* __restrict__ y,
                                                                     const uint32_t* __restrict__ yoff, uint32_t ys) {
  __shared__ __attribute__((aligned(32))) _Float16 lds[RmLds<S, RM>::kHalves];
  const uint32_t m = blockIdx.y, c0 = blockIdx.z * kTile * CT;
  if (c0 >= (uint32_t) cnt[m]) return;  // a block past its matrix's tokens: nothing
  tile_rm<T, S, CT, RM, PIPE>(w[m], blockIdx.x * kTile * RM, cols, x16 + xoff[m], xts, c0, (uint32_t) cnt[m], y + yoff[m],
                        ys, lds);
}

// the experts' prompt-sized matmul as gemm_kernel's blocks: n matrices of one shape (blockIdx.y the matrix m: its
// weights wt[m], its first cnt[m] tokens, its x the tiled x16 from x16a + xoff[m] with rows xts long, its y at
// ya + yoff[m] + t * ys) - group_kernel's addressing on gemm_kernel's body, so its tiles' sums in tile's order: the
// same bits as group_kernel at every count (a block past the matrix's tokens nothing)
// a stage: up to kStageValues of one partition's slabs in its order (partition p's slabs p, p + W, ...) - a slab of
// 32 values alone would be a barrier every 8 WMMAs; the sums' order the partition's either way
constexpr int kStageValues = 64;
template <class S>
constexpr int stage_slabs() {
  return S::kSlab >= kStageValues ? 1 : kStageValues / S::kSlab;
}

template <quant::Type T, class S, int RM, int RN, int WR, int WC>
__global__ __launch_bounds__(256) void group_gemm_kernel(const uint8_t* const* __restrict__ wt, const int32_t* __restrict__ cnt,
                                                          uint32_t rows, uint32_t cols, const _Float16* __restrict__ x16a,
                                                          const uint32_t* __restrict__ xoff, uint32_t xts,
                                                          float* __restrict__ ya, const uint32_t* __restrict__ yoff,
                                                          uint32_t ys) {
  static_assert(WR * WC * kWave == 256, "8 waves");
  constexpr int SL = S::kSlab, G = SL / kTile, W = S::kWaves, BM = kTile * RM * WR, BN = kTile * RN * WC;
  constexpr int SP = stage_slabs<S>();
  constexpr int RL = SP * SL + kPad, CPR = SL / quant::kChunk, CH = BM * CPR / 256;  // a thread's chunks a slab
  static_assert((BM * CPR) % 256 == 0, "a slab's chunks whole");
  __shared__ __attribute__((aligned(32))) _Float16 lds2[2][BM * RL];
  const uint32_t m = blockIdx.y, n = (uint32_t) cnt[m];
  const uint8_t* w = wt[m];
  const _Float16* x16 = x16a + xoff[m];
  float* y = ya + yoff[m];
  const uint32_t row0 = blockIdx.x * BM, c0 = blockIdx.z * BN;
  if (row0 >= rows || c0 >= n) return;  // the whole block: past the matrix's rows or its tokens
  const int lane = threadIdx.x % kWave, wave = threadIdx.x / kWave, half = lane / kTile, col = lane % kTile;
  const int wr = wave / WC, wc = wave % WC;
  const _Float16* xb[RN];
  bool live[RN];
  for (int j = 0; j < RN; j++) {
    const uint32_t t0 = c0 + (wc * RN + j) * kTile;
    live[j] = t0 < n, xb[j] = x16 + x16_at(t0 + col, 0, xts);
  }
  f8 run[RM][RN], acc[RM][RN];
  const uint32_t slabs = cols / SL;  // >= W (the launch's): every partition a slab at least
  auto in_part = [&](int p) { return (int) ((slabs - (uint32_t) p + W - 1) / W); };  // partition p's slabs
  // a stage: partition p's slabs q .. q + ns - 1 (slab p + i W)
  // Q4_K: a thread NR chunks of one row's slab (quant::decode_hn, the group's scales once) - the stage's same chunks
  constexpr int NR = T == quant::kQ4_K && CPR % 2 == 0 && (BM * SP * CPR / 2) % 256 == 0 ? 2 : 1;
  constexpr int RUNS = BM * SP * CPR / NR / 256;  // a thread's runs a stage
  auto decode_runs = [&](int p, int q, int ns, h8 (*out)[NR]) {
#pragma unroll
    for (int c = 0; c < RUNS; c++) {
      const int u = threadIdx.x + 256 * c, r = u / (SP * CPR / NR), e = u % (SP * CPR / NR), i = e / (CPR / NR);
      const int ck = (e % (CPR / NR)) * NR;
      if (i < ns) {
        const uint32_t k0 = ((uint32_t) p + (uint32_t) (q + i) * W) * SL;
        const uint32_t row = row0 + r < rows ? row0 + r : rows - 1;
        int kk;
        quant::decode_hn<T, NR>(quant::at<T>(w, row, cols, k0 / quant::kChunk + ck, &kk), kk, out[c]);
      }
    }
  };
  auto put_runs = [&](_Float16* buf, int ns, h8 (*v)[NR]) {
#pragma unroll
    for (int c = 0; c < RUNS; c++) {
      const int u = threadIdx.x + 256 * c, r = u / (SP * CPR / NR), e = u % (SP * CPR / NR), i = e / (CPR / NR);
      const int ck = (e % (CPR / NR)) * NR;
      if (i < ns)
        for (int x = 0; x < NR; x++) *(h8*) (buf + r * RL + i * SL + (ck + x) * quant::kChunk) = v[c][x];
    }
  };
  auto decode = [&](int p, int q, int ns, h8 (*out)[CH]) {
#pragma unroll
    for (int i = 0; i < SP; i++)
      if (i < ns) {
        const uint32_t k0 = ((uint32_t) p + (uint32_t) (q + i) * W) * SL;
#pragma unroll
        for (int c = 0; c < CH; c++) {
          const int d = threadIdx.x + 256 * c, r = d / CPR, ck = d % CPR;
          const uint32_t row = row0 + r < rows ? row0 + r : rows - 1;
          int kk;
          out[i][c] = quant::decode_h<T>(quant::at<T>(w, row, cols, k0 / quant::kChunk + ck, &kk), kk);
        }
      }
  };
  auto put = [&](_Float16* buf, int ns, h8 (*v)[CH]) {
#pragma unroll
    for (int i = 0; i < SP; i++)
      if (i < ns)
#pragma unroll
        for (int c = 0; c < CH; c++) {
          const int d = threadIdx.x + 256 * c, r = d / CPR, ck = d % CPR;
          *(h8*) (buf + r * RL + i * SL + ck * quant::kChunk) = v[i][c];
        }
  };
  for (int i = 0; i < RM; i++)
    for (int j = 0; j < RN; j++) acc[i][j] = f8{};
  int p = 0, q = 0, ns = in_part(0) < SP ? in_part(0) : SP, cur = 0;
  if constexpr (NR > 1) {
    h8 v[RUNS][NR];
    decode_runs(p, q, ns, v);
    put_runs(lds2[0], ns, v);
  } else {
    h8 v[SP][CH];
    decode(p, q, ns, v);
    put(lds2[0], ns, v);
  }
  __syncthreads();
  for (;;) {
    // the stage after this one: the partition's next slabs, else the next partition's first
    const bool more_p = q + ns < in_part(p);
    const int np = more_p ? p : p + 1, nq = more_p ? q + ns : 0;
    const bool next = np < W;
    const int left = next ? in_part(np) - nq : 0, nns = left < SP ? left : SP;
    h8 v[NR > 1 ? RUNS : SP][NR > 1 ? NR : CH];
    if (next) {
      if constexpr (NR > 1) decode_runs(np, nq, nns, v);
      else decode(np, nq, nns, v);
    }
    const _Float16* lds = lds2[cur];
#pragma unroll
    for (int i = 0; i < SP; i++)
      if (i < ns) {
        const uint32_t k0 = ((uint32_t) p + (uint32_t) (q + i) * W) * SL;
#pragma unroll
        for (int g = 0; g < G; g++) {
          h16 a[RM], b[RN];
          for (int r = 0; r < RM; r++) a[r] = *(const h16*) (lds + ((wr * RM + r) * kTile + col) * RL + i * SL + g * kTile);
          for (int j = 0; j < RN; j++) b[j] = *(const h16*) (xb[j] + (uint64_t) (k0 / kTile + g) * kFrag);
          for (int j = 0; j < RN; j++)
            if (live[j])
              for (int r = 0; r < RM; r++) acc[r][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[r], b[j], acc[r][j]);
        }
      }
    if (!more_p) {  // the partition's sum done: into the running sum, the next from zero
      for (int i = 0; i < RM; i++)
        for (int j = 0; j < RN; j++) {
          if (p == 0) run[i][j] = acc[i][j];
          else
            for (int e = 0; e < 8; e++) run[i][j][e] += acc[i][j][e];
          acc[i][j] = f8{};
        }
    }
    if (!next) break;
    if constexpr (NR > 1) put_runs(lds2[cur ^ 1], nns, v);  // the other buffer: read last before the barrier ending
    else put(lds2[cur ^ 1], nns, v);                         // the stage before
    __syncthreads();
    cur ^= 1, p = np, q = nq, ns = nns;
  }
  for (int j = 0; j < RN; j++) {
    const uint32_t t = c0 + (wc * RN + j) * kTile + col;
    if (t >= n) continue;
    for (int i = 0; i < RM; i++) {
      const uint32_t r = row0 + (wr * RM + i) * kTile;
      if (r >= rows) continue;
      for (int e = 0; e < 8; e++) y[(uint64_t) t * ys + r + 2 * e + half] = run[i][j][e];
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

// one matrix with the silu epilogue (mm_run_silu)
template <quant::Type T, class S, int CT>
__global__ __launch_bounds__(kWave* S::kWaves) void kernel_silu(const uint8_t* __restrict__ w, uint32_t rows,
                                                                 uint32_t cols, const _Float16* __restrict__ x16,
                                                                 uint32_t n, float* __restrict__ y, uint32_t ys,
                                                                 Silu ep) {
  __shared__ __attribute__((aligned(32))) _Float16 lds[S::kLdsHalves];
  const uint32_t row0 = blockIdx.x * kTile;
  if (row0 >= rows) return;
  tile<T, S, CT, true>(w, row0, cols, x16, cols, blockIdx.z * kTile * CT, n, y, ys, lds, ep);
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
