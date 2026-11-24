// mm (include/osmos/gpu/mm.h): the kernel is mm.h's
#include <cmath>

#include "mm.h"
#include "round.h"

extern "C" {
#include "osmos/gpu/mm.h"
}

namespace {

constexpr int kThreads = 256;

// a type's tile (mm.h's Shape) and its most column tiles, tuned on gfx1151. a shape is a type's for every token count
// - its K split is the sums' order, so a token's bits would otherwise depend on its batch. Q5_K's decode is the
// heaviest: small slabs, so more waves hide it. the IQ experts: 4 waves of 128-value slabs, each wave a long run of
// slabs and a 4-way sum at the end
template <quant::Type T> struct Tuned { using S = mm::Shape<128, 8, 8>; static constexpr int kMaxCt = 8; };
// a short row's shape (fewer than kShortCols columns): a type's own where measured, else S. a matrix's shape is its
// type's and its columns' alone, never its tokens' - the same sums at every count. Q6_K's: 8 waves of 64-value slabs,
// not 4 of 128 (a wave's longer run, a larger error)
constexpr uint32_t kShortCols = 2048;
template <quant::Type T> struct Short { using S = typename Tuned<T>::S; };
template <> struct Short<quant::kQ6_K> { using S = mm::Shape<64, 8, 4>; };
// BF16's short rows (Mini's loop gains and forgets: 64 columns, one slab - its one wave the row's sum)
#ifndef MM_BF16_SHORT
#define MM_BF16_SHORT 64, 1, 4
#endif
template <> struct Short<quant::kBF16> { using S = mm::Shape<MM_BF16_SHORT>; };
// a width the type's slab does not divide (Qwen3.8's mixers' up: 320 columns of Q8_0): 64-value slabs - a shape only
// such a width takes, so no matrix that tiled before changes
template <quant::Type T> struct Odd { using S = typename Short<T>::S; };
template <> struct Odd<quant::kQ8_0> { using S = mm::Shape<64, 2, 4>; };
template <quant::Type T>
constexpr bool odd(uint32_t cols) {
  return cols % (cols < kShortCols ? Short<T>::S::kSlab : Tuned<T>::S::kSlab) != 0;
}
template <> struct Tuned<quant::kQ8_0> { using S = mm::Shape<128, 8, 8>; static constexpr int kMaxCt = 4; };
template <> struct Tuned<quant::kQ5_K> { using S = mm::Shape<32, 8, 2>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kQ5_KR> : Tuned<quant::kQ5_K> {};
template <> struct Tuned<quant::kIQ1_M> { using S = mm::Shape<128, 4, 8>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kIQ2_XXS> { using S = mm::Shape<128, 4, 8>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kIQ2_XS> { using S = mm::Shape<128, 4, 8>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kIQ1_S> : Tuned<quant::kIQ1_M> {};   // not measured: the sibling's
template <> struct Tuned<quant::kIQ2_S> : Tuned<quant::kIQ2_XS> {};  // not measured: the sibling's
// Qwen3.8's expert types (tools/bench/mmsweep)
template <> struct Tuned<quant::kIQ3_XXS> { using S = mm::Shape<128, 8, 8>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kIQ4_NL> { using S = mm::Shape<64, 4, 4>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kIQ4_XS> { using S = mm::Shape<64, 4, 4>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kQ4_0> : Tuned<quant::kIQ4_NL> {};  // not measured: IQ4_NL's (its layout, its block)
template <> struct Tuned<quant::kQ2_0> { using S = mm::Shape<64, 8, 4>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kQ2_K> : Tuned<quant::kQ2_0> {};
template <> struct Tuned<quant::kQ3_K> : Tuned<quant::kQ2_0> {};  // not measured: Q2_K's (a 256-value K block)
// NVFP4 (Inkling-2's experts: 4,608 and 2,304 columns, whole 64-value slabs) not measured: Q2_0's, a 64-value block too.
// its values exact in f16 (6 significant bits, 2^-10 .. 2,688: decode_h rounds none); a tensor's own scale its role's
template <> struct Tuned<quant::kNVFP4> : Tuned<quant::kQ2_0> {};
// MXFP8 (Inkling-2's experts: Small's layers 40-47, Mini's 20-23; first unsloth's Engram tables, decode alone). its
// values exact in f16 where e4m3 x 2^(e - 127) is one: Inkling's (e -15 .. -10, |w| 2^-22 .. 0.375) all are -
// ink2_ref_gguf.py checks every expert's
template <> struct Tuned<quant::kMXFP8> { using S = mm::Shape<64, 4, 4>; static constexpr int kMaxCt = 8; };
// BF16 (Inkling-2-Mini's dense weights, as released): a value f16's where it is one - all but those below f16's
// normal range (each within 2^-25 of its own); not in mm_known's table (QTYPE_MM): a family takes it by name (ink2's
// mm_takes), the others' BF16 matrices their matvec's
#ifndef MM_BF16_SHAPE
#define MM_BF16_SHAPE 128, 8, 8
#endif
template <> struct Tuned<quant::kBF16> { using S = mm::Shape<MM_BF16_SHAPE>; static constexpr int kMaxCt = 8; };
// a long BF16 row whose 128-value slabs 9 waves divide evenly (Mini's 2,304 columns: q, gate, k, v, rel, the router,
// the head; the loop controls' down 4,608): 9 waves, each 2 slabs (4 for down); 3,072 columns (24 slabs: the output)
// keep 128 x 8 x 8. by its columns alone, so a matrix's every count
struct Bf16Nine { using S = mm::Shape<128, 9, 8>; };
constexpr bool bf16_nine(uint32_t cols) { return cols >= kShortCols && cols % (128 * 9) == 0; }
// UD-Q4_K_XL's expert types, on the register-A path (mm.h's kRegA): Q4_K 32 x 16 x 2 (its 2,560-column rows' 80
// slabs for 16 waves to share: the decode's latency hidden; a 32-value slab's lane holds 2 chunks, so U 2 at most),
// Q5_1 32 x 8 x 2 (its 640-column down: 20 slabs too few for 16). the 64-value slabs through LDS were not taken: they
// moved the perplexity up (DESIGN.md, "UD-Q4_K_XL")
template <> struct Tuned<quant::kQ4_K> { using S = mm::Shape<32, 16, 2>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kQ5_1> { using S = mm::Shape<32, 8, 2>; static constexpr int kMaxCt = 8; };
using mm::kF16Max;

// a thread a value: to f16 (the rows past n zero)
__global__ void x16_kernel(const float* x, uint32_t xs, uint32_t n, uint32_t cols, _Float16* x16, int32_t* overflow) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  const uint32_t t = (uint32_t) (i / cols), c = (uint32_t) (i % cols);
  if (t >= (n + mm::kTile - 1) / mm::kTile * mm::kTile) return;
  mm::x16_put(x16, t, c, cols, t < n ? x[(uint64_t) t * xs + c] : 0.0f, overflow);  // past f16: refused at the end
}

// the same values 8 a thread (x 16-byte aligned, xs a whole float4s): two float4 loads, the 8 halves one 16-byte
// store (a fragment row's half: contiguous in the tiled layout), 32-bit indices, the overflow flag once - each value
// x16_put's conversion: the same bits
constexpr uint32_t kX16Vec = 8;
__global__ void x16_vec_kernel(const float* x, uint32_t xs, uint32_t n, uint32_t cols, _Float16* x16, int32_t* overflow) {
  const uint32_t i = blockIdx.x * kThreads + threadIdx.x, per = cols / kX16Vec, t = i / per, c = i % per * kX16Vec;
  if (t >= (n + mm::kTile - 1) / mm::kTile * mm::kTile) return;
  float4 a = make_float4(0, 0, 0, 0), b = a;
  if (t < n) {
    const float4* p = (const float4*) (x + (uint64_t) t * xs + c);
    a = p[0], b = p[1];
  }
  const float v[kX16Vec] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  __attribute__((aligned(16))) _Float16 h[kX16Vec];
  bool bad = false;
#pragma unroll
  for (uint32_t e = 0; e < kX16Vec; e++) bad |= !(fabsf(v[e]) <= mm::kF16Max), h[e] = (_Float16) v[e];
  if (bad) atomicOr(overflow, 1);
  *(uint4*) (x16 + mm::x16_at(t, c, cols)) = *(const uint4*) h;
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

// slot s: row idx[s] of x (-1: zero), to f16 - a thread 8 adjacent columns of a slot (half a fragment's row: 16
// bytes, x16_at's contiguous run), threads in the tiled order so a wave's stores fill whole lines; each value
// x16_put's rounding and flag, so the same bytes as a value a thread (whose 2-byte stores scatter)
constexpr uint32_t kRun = 8;
__global__ void x16_rows_kernel(const float* x, uint32_t xs, const int32_t* idx, uint32_t slots, uint32_t cols,
                                _Float16* x16, int32_t* overflow) {
  const uint64_t o = ((uint64_t) blockIdx.x * kThreads + threadIdx.x) * kRun;  // the run's first half, tiled
  const uint64_t tile = (uint64_t) mm::kTile * cols;
  const uint32_t in = (uint32_t) (o % tile), frag = in % mm::kFrag;
  const uint32_t t = (uint32_t) (o / tile) * mm::kTile + frag / mm::kTile;
  if (t >= slots) return;
  const uint32_t c = in / mm::kFrag * mm::kTile + frag % mm::kTile;
  const int32_t r = idx[t];
  float v[kRun] = {};
  if (r >= 0) {
    const float4* p = (const float4*) (x + (uint64_t) r * xs + c);
    const float4 a = p[0], b = p[1];
    v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
  }
  bool over = false;
  _Float16 h[kRun];
  for (uint32_t j = 0; j < kRun; j++) over |= !(fabsf(v[j]) <= mm::kF16Max), h[j] = (_Float16) v[j];
  if (over) atomicOr(overflow, 1);
  __builtin_memcpy(x16 + o, h, sizeof h);
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

// the experts' multi-tile steps by type, each the bits of group_ct's (the sums in tile's order): 2-4 column tiles
// with 2 row tiles a wave (tile_rm: an x fragment two WMMAs), 8 as gemm_kernel's 64 x 128 blocks - chosen a type at a
// time (tools/bench/mmsweep's experts). IQ3_XXS's 128 x 8 shape takes no 2 row tiles (70 KB of LDS); the codebook
// types none (twice a lane's grid lookups). the types not listed: group_ct's
template <quant::Type T>
struct GroupPlan {
  static constexpr bool kRm = T == quant::kQ4_K || T == quant::kQ5_1 || T == quant::kIQ4_XS || T == quant::kQ2_K || T == quant::kQ3_K;
  static constexpr bool kGemm = T == quant::kQ4_K || T == quant::kQ5_1 || T == quant::kIQ4_NL || T == quant::kQ4_0 || T == quant::kQ2_0 || T == quant::kNVFP4 ||
                                T == quant::kIQ3_XXS || T == quant::kQ2_K || T == quant::kQ3_K || T == quant::kIQ2_S || T == quant::kIQ1_S;
};
constexpr int kGroupRm = 2, kGroupGemmRm = 2, kGroupGemmRn = 2, kGroupGemmWr = 2, kGroupGemmWc = 4;

template <quant::Type T, int CT>
bool group_rm(const void* const* w, const int32_t* cnt, uint32_t n_mat, uint32_t max_cnt, uint32_t rows,
              uint32_t cols, const void* x16, const uint32_t* xoff, uint32_t xts, float* y, const uint32_t* yoff,
              uint32_t ys, hipStream_t s) {
  using S = typename Tuned<T>::S;
  const uint32_t z = (max_cnt + mm::kTile * CT - 1) / (mm::kTile * CT);
  mm::group_rm_kernel<T, S, CT, kGroupRm><<<dim3(rows / (mm::kTile * kGroupRm), n_mat, z), mm::kWave * S::kWaves, 0, s>>>(
      (const uint8_t* const*) w, cnt, cols, (const _Float16*) x16, xoff, xts, y, yoff, ys);
  return hipGetLastError() == hipSuccess;
}

template <quant::Type T>
bool group_gemm(const void* const* w, const int32_t* cnt, uint32_t n_mat, uint32_t max_cnt, uint32_t rows,
                uint32_t cols, const void* x16, const uint32_t* xoff, uint32_t xts, float* y, const uint32_t* yoff,
                uint32_t ys, hipStream_t s) {
  using S = typename Tuned<T>::S;
  constexpr uint32_t BM = mm::kTile * kGroupGemmRm * kGroupGemmWr, BN = mm::kTile * kGroupGemmRn * kGroupGemmWc;
  mm::group_gemm_kernel<T, S, kGroupGemmRm, kGroupGemmRn, kGroupGemmWr, kGroupGemmWc>
      <<<dim3(rows / BM, n_mat, (max_cnt + BN - 1) / BN), 256, 0, s>>>((const uint8_t* const*) w, cnt, rows, cols,
                                                                      (const _Float16*) x16, xoff, xts, y, yoff, ys);
  return hipGetLastError() == hipSuccess;
}

// the codebook types' one-tile steps on mm.h's group_tile_kernel (group_ct<T, 1>'s bits): as many blocks as the GPU
// holds at once, each taking tiles until none are left. what a WGP holds: its VGPRs' blocks or its LDS's, the fewer -
// the LDS both CUs' (WGP mode; the runtime's figure, and its occupancy call, are one CU's 64 KB)
#if defined(__HIP__)  // mm.h's group_tile_kernel: RDNA's alone (kOneTile false on CUDA)
template <quant::Type T, int W>
uint32_t resident_blocks() {
  static const uint32_t n = [] {
    int dev = 0;
    hipDeviceProp_t p;
    hipFuncAttributes a;
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess ||
        hipFuncGetAttributes(&a, (const void*) mm::group_tile_kernel<T, W>) != hipSuccess || a.numRegs < 1)
      return 0u;
    const uint64_t regs = (uint64_t) (a.numRegs + 7) / 8 * 8 * mm::kWave * W;  // VGPRs come in 8s a lane
    const uint64_t by_regs = (uint64_t) p.regsPerMultiprocessor / regs;
    const uint64_t by_lds = a.sharedSizeBytes ? 2 * (uint64_t) p.maxSharedMemoryPerMultiProcessor / a.sharedSizeBytes
                                              : by_regs;
    const uint64_t by_waves = (uint64_t) p.maxThreadsPerMultiProcessor / (mm::kWave * W);
    uint64_t per = by_regs < by_lds ? by_regs : by_lds;
    per = per < by_waves ? per : by_waves;
    return (uint32_t) (per * (uint64_t) p.multiProcessorCount);
  }();
  return n;
}

template <quant::Type T>
bool group_one_tile(const void* const* w, const int32_t* cnt, uint32_t n_mat, uint32_t rows, uint32_t cols,
                    const void* x16, const uint32_t* xoff, uint32_t xts, float* y, const uint32_t* yoff, uint32_t ys,
                    hipStream_t s) {
  using S = typename Tuned<T>::S;
  static_assert(S::kSlab == 128, "group_tile_kernel's slab");
  const uint32_t tiles = rows / mm::kTile * n_mat, held = resident_blocks<T, S::kWaves>();
  if (!held) return false;
  mm::group_tile_kernel<T, S::kWaves><<<tiles < held ? tiles : held, mm::kWave * S::kWaves, 0, s>>>(
      (const uint8_t* const*) w, cnt, n_mat, rows, cols, (const _Float16*) x16, xoff, xts, y, yoff, ys);
  return hipGetLastError() == hipSuccess;
}
#endif

template <quant::Type T>
bool group_launch(const void* const* w, const int32_t* cnt, uint32_t n_mat, uint32_t max_cnt, uint32_t rows,
                  uint32_t cols, const void* x16, const uint32_t* xoff, uint32_t xts, float* y, const uint32_t* yoff,
                  uint32_t ys, hipStream_t s) {
  using S = typename Tuned<T>::S;
  if (rows % mm::kTile || cols % quant::Block<T>::values || cols % S::kSlab || xts < cols || xts % mm::kTile ||
      !max_cnt || max_cnt > MM_MAX_TOKENS)
    return false;
  constexpr uint32_t gemm_rows = mm::kTile * kGroupGemmRm * kGroupGemmWr;
  const bool rm = GroupPlan<T>::kRm && rows % (mm::kTile * kGroupRm) == 0;
  const bool gemm = GroupPlan<T>::kGemm && rows % gemm_rows == 0 && cols / S::kSlab >= (uint32_t) S::kWaves;
  switch (col_tiles<T>(max_cnt)) {
    case 1:
#if defined(__HIP__)
      if constexpr (mm::kOneTile<T>) return group_one_tile<T>(w, cnt, n_mat, rows, cols, x16, xoff, xts, y, yoff, ys, s);
#endif
      return group_ct<T, 1>(w, cnt, n_mat, max_cnt, rows, cols, x16, xoff, xts, y, yoff, ys, s);
    case 2:
      if constexpr (GroupPlan<T>::kRm)
        if (rm) return group_rm<T, 2>(w, cnt, n_mat, max_cnt, rows, cols, x16, xoff, xts, y, yoff, ys, s);
      return group_ct<T, 2>(w, cnt, n_mat, max_cnt, rows, cols, x16, xoff, xts, y, yoff, ys, s);
    case 4:
      if constexpr (GroupPlan<T>::kRm)
        if (rm) return group_rm<T, 4>(w, cnt, n_mat, max_cnt, rows, cols, x16, xoff, xts, y, yoff, ys, s);
      return group_ct<T, 4>(w, cnt, n_mat, max_cnt, rows, cols, x16, xoff, xts, y, yoff, ys, s);
    default:
      if constexpr (GroupPlan<T>::kGemm)
        if (gemm) return group_gemm<T>(w, cnt, n_mat, max_cnt, rows, cols, x16, xoff, xts, y, yoff, ys, s);
      return group_ct<T, Tuned<T>::kMaxCt>(w, cnt, n_mat, max_cnt, rows, cols, x16, xoff, xts, y, yoff, ys, s);
  }
}

/* one matrix (w[1] NULL) or a pair on the same x16 */
struct Mats {
  const void* w[2];
  uint32_t rows[2];
  float* y[2];
  uint32_t ys[2];
  float scale[2] = {1.0f, 1.0f};  // each matrix's sums times it (mm_tokens_scaled)
};

template <quant::Type T, int CT, typename S>
bool launch_shape(const Mats& m, uint32_t cols, const void* x16, uint32_t n, hipStream_t s) {
  const uint32_t z = (n + mm::kTile * CT - 1) / (mm::kTile * CT), rows = m.rows[0] > m.rows[1] ? m.rows[0] : m.rows[1];
  mm::kernel<T, S, CT><<<dim3(rows / mm::kTile, m.w[1] ? 2 : 1, z), mm::kWave * S::kWaves, 0, s>>>(
      (const uint8_t*) m.w[0], (const uint8_t*) m.w[1], m.rows[0], m.rows[1], cols, (const _Float16*) x16, n, m.y[0],
      m.y[1], m.ys[0], m.ys[1], m.scale[0], m.scale[1]);
  return hipGetLastError() == hipSuccess;
}

// a prompt's matrix on mm.h's gemm_kernel: Q6_K, Q8_0 and Q5_K (the models' dense), on its rows' own shape (tile's: Short
// below kShortCols; every partition a slab) and more than 64 tokens (CT 8's: below, tile's own); the same bits as tile
// at every count. 64 rows x 256 tokens a block, a wave 32 x 64 (128 x 128: past the LDS)
template <quant::Type T>
constexpr bool kGemm = T == quant::kQ6_K || T == quant::kQ8_0 || T == quant::kNVFP4 || T == quant::kBF16 ||
                       T == quant::kQ5_K || T == quant::kQ5_KR;
constexpr int kGemmRm = 2, kGemmRn = 4, kGemmWr = 2, kGemmWc = 4;
// a type's gemm blocks (kGemm*'s but where measured): its rows and tokens a wave in tiles, its waves' grid
template <quant::Type T> struct GemmPlan { static constexpr int Rm = kGemmRm, Rn = kGemmRn, Wr = kGemmWr, Wc = kGemmWc; };
#ifdef GEMM_NVFP4
template <> struct GemmPlan<quant::kNVFP4> { static constexpr int Rm = GEMM_RM, Rn = GEMM_RN, Wr = GEMM_WR, Wc = GEMM_WC; };
#endif
constexpr uint32_t kGemmMinTokens = 4 * mm::kTile;  // past it: CT 8's counts

template <quant::Type T, class S = typename Tuned<T>::S>
bool gemm_takes(const Mats& m, uint32_t cols, uint32_t n) {
  return kGemm<T> && n > kGemmMinTokens && !odd<T>(cols) && cols / S::kSlab >= (uint32_t) S::kWaves;
}
// the gemm on a row's own shape (the tile's: a short row's Short, else Tuned) - its sums' order at every count
template <quant::Type T>
bool gemm_on(const Mats& m, uint32_t cols, uint32_t n) {
  return cols < kShortCols ? gemm_takes<T, typename Short<T>::S>(m, cols, n) : gemm_takes<T>(m, cols, n);
}

template <quant::Type T, class S = typename Tuned<T>::S>
bool launch_gemm(const Mats& m, uint32_t cols, const void* x16, uint32_t n, hipStream_t s) {
  using P = GemmPlan<T>;
  constexpr uint32_t BM = mm::kTile * P::Rm * P::Wr, BN = mm::kTile * P::Rn * P::Wc;
  const uint32_t rows = m.rows[0] > m.rows[1] ? m.rows[0] : m.rows[1];
  mm::gemm_kernel<T, S, P::Rm, P::Rn, P::Wr, P::Wc>
      <<<dim3((rows + BM - 1) / BM, m.w[1] ? 2 : 1, (n + BN - 1) / BN), 256, 0, s>>>(
          (const uint8_t*) m.w[0], (const uint8_t*) m.w[1], m.rows[0], m.rows[1], cols, (const _Float16*) x16, n,
          m.y[0], m.y[1], m.ys[0], m.ys[1], m.scale[0], m.scale[1]);
  return hipGetLastError() == hipSuccess;
}

template <quant::Type T, int CT>
bool launch_ct(const Mats& m, uint32_t cols, const void* x16, uint32_t n, hipStream_t s) {
  if constexpr (T == quant::kBF16)
    if (bf16_nine(cols)) {  // (the gemm and the tile the same shape: its sums' order at every count)
      using S9 = Bf16Nine::S;
      if (gemm_takes<T, S9>(m, cols, n)) return launch_gemm<T, S9>(m, cols, x16, n, s);
      return launch_shape<T, CT, S9>(m, cols, x16, n, s);
    }
  if constexpr (kGemm<T>)
    if (gemm_on<T>(m, cols, n))
      return cols < kShortCols ? launch_gemm<T, typename Short<T>::S>(m, cols, x16, n, s)
                               : launch_gemm<T>(m, cols, x16, n, s);
  if (odd<T>(cols)) return launch_shape<T, CT, typename Odd<T>::S>(m, cols, x16, n, s);
  return cols < kShortCols ? launch_shape<T, CT, typename Short<T>::S>(m, cols, x16, n, s)
                           : launch_shape<T, CT, typename Tuned<T>::S>(m, cols, x16, n, s);
}


template <quant::Type T>
bool launch(const Mats& m, uint32_t cols, const void* x16, uint32_t n, hipStream_t s) {
  for (int i = 0; i < (m.w[1] ? 2 : 1); i++)
    if (m.rows[i] % mm::kTile || (n > 1 && m.ys[i] < m.rows[i])) return false;
  const int slab = odd<T>(cols) ? Odd<T>::S::kSlab : cols < kShortCols ? Short<T>::S::kSlab : Tuned<T>::S::kSlab;
  if (cols % quant::Block<T>::values || cols % slab) return false;
  switch (col_tiles<T>(n)) {
    case 1: return launch_ct<T, 1>(m, cols, x16, n, s);
    case 2: return launch_ct<T, 2>(m, cols, x16, n, s);
    case 4: return launch_ct<T, 4>(m, cols, x16, n, s);
    default: return launch_ct<T, Tuned<T>::kMaxCt>(m, cols, x16, n, s);
  }
}

template <quant::Type T, int CT, typename S>
bool silu_shape(const void* w, uint32_t rows, uint32_t cols, const void* x16, uint32_t n, float* y, uint32_t ys,
                mm::Silu ep, hipStream_t s) {
  const uint32_t z = (n + mm::kTile * CT - 1) / (mm::kTile * CT);
  mm::kernel_silu<T, S, CT><<<dim3(rows / mm::kTile, 1, z), mm::kWave * S::kWaves, 0, s>>>(
      (const uint8_t*) w, rows, cols, (const _Float16*) x16, n, y, ys, ep);
  return hipGetLastError() == hipSuccess;
}

// mm_run's launch with the silu epilogue: the shape launch_ct takes (the same sums), a matrix the gemm would take
// refused (its caller's two launches instead)
template <quant::Type T, int CT>
bool silu_ct(const void* w, uint32_t rows, uint32_t cols, const void* x16, uint32_t n, float* y, uint32_t ys,
             mm::Silu ep, hipStream_t s) {
  const Mats m = {{w, nullptr}, {rows, 0}, {y, nullptr}, {ys, 0}};
  if constexpr (kGemm<T>)
    if (gemm_on<T>(m, cols, n)) return false;
  if (odd<T>(cols)) return silu_shape<T, CT, typename Odd<T>::S>(w, rows, cols, x16, n, y, ys, ep, s);
  return cols < kShortCols ? silu_shape<T, CT, typename Short<T>::S>(w, rows, cols, x16, n, y, ys, ep, s)
                           : silu_shape<T, CT, typename Tuned<T>::S>(w, rows, cols, x16, n, y, ys, ep, s);
}

template <quant::Type T>
bool silu_launch(const void* w, uint32_t rows, uint32_t cols, const void* x16, uint32_t n, float* y, uint32_t ys,
                 mm::Silu ep, hipStream_t s) {
  const int slab = odd<T>(cols) ? Odd<T>::S::kSlab : cols < kShortCols ? Short<T>::S::kSlab : Tuned<T>::S::kSlab;
  if (rows % mm::kTile || (n > 1 && ys < rows) || cols % quant::Block<T>::values || cols % slab) return false;
  switch (col_tiles<T>(n)) {
    case 1: return silu_ct<T, 1>(w, rows, cols, x16, n, y, ys, ep, s);
    case 2: return silu_ct<T, 2>(w, rows, cols, x16, n, y, ys, ep, s);
    case 4: return silu_ct<T, 4>(w, rows, cols, x16, n, y, ys, ep, s);
    default: return silu_ct<T, Tuned<T>::kMaxCt>(w, rows, cols, x16, n, y, ys, ep, s);
  }
}

// mm_batch's launch: the shape launch_ct takes at its counts (n <= kGemmMinTokens: the tile's, never the gemm)
template <quant::Type T, int CT, typename S>
bool batch_shape(const mm::Parts& p, uint32_t count, uint32_t rows, uint32_t cols, uint32_t n, hipStream_t s) {
  const uint32_t z = (n + mm::kTile * CT - 1) / (mm::kTile * CT);
  mm::batch_kernel<T, S, CT><<<dim3(rows / mm::kTile, count, z), mm::kWave * S::kWaves, 0, s>>>(p, cols, n);
  return hipGetLastError() == hipSuccess;
}

template <quant::Type T, int CT>
bool batch_ct(const mm::Parts& p, uint32_t count, uint32_t rows, uint32_t cols, uint32_t n, hipStream_t s) {
  if constexpr (T == quant::kBF16)
    if (bf16_nine(cols)) return batch_shape<T, CT, Bf16Nine::S>(p, count, rows, cols, n, s);
  if (odd<T>(cols)) return batch_shape<T, CT, typename Odd<T>::S>(p, count, rows, cols, n, s);
  return cols < kShortCols ? batch_shape<T, CT, typename Short<T>::S>(p, count, rows, cols, n, s)
                           : batch_shape<T, CT, typename Tuned<T>::S>(p, count, rows, cols, n, s);
}

template <quant::Type T>
bool batch(const mm::Parts& p, uint32_t count, uint32_t cols, uint32_t n, hipStream_t s) {
  uint32_t rows = 0;
  for (uint32_t i = 0; i < count; i++) {  // launch()'s refusals, each part
    if (!p.rows[i] || p.rows[i] % mm::kTile || (n > 1 && p.ys[i] < p.rows[i]) || !p.w[i] || !p.x16[i]) return false;
    rows = p.rows[i] > rows ? p.rows[i] : rows;
  }
  const int slab = odd<T>(cols) ? Odd<T>::S::kSlab : cols < kShortCols ? Short<T>::S::kSlab : Tuned<T>::S::kSlab;
  if (cols % quant::Block<T>::values || cols % slab) return false;
  switch (col_tiles<T>(n)) {
    case 1: return batch_ct<T, 1>(p, count, rows, cols, n, s);
    case 2: return batch_ct<T, 2>(p, count, rows, cols, n, s);
    case 4: return batch_ct<T, 4>(p, count, rows, cols, n, s);
    default: return batch_ct<T, Tuned<T>::kMaxCt>(p, count, rows, cols, n, s);
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
    case quant::kIQ1_S: return launch<quant::kIQ1_S>(m, cols, x16, n, s);
    case quant::kIQ2_S: return launch<quant::kIQ2_S>(m, cols, x16, n, s);
    case quant::kIQ3_XXS: return launch<quant::kIQ3_XXS>(m, cols, x16, n, s);
    case quant::kQ2_K: return launch<quant::kQ2_K>(m, cols, x16, n, s);
    case quant::kQ3_K: return launch<quant::kQ3_K>(m, cols, x16, n, s);
    case quant::kIQ4_NL: return launch<quant::kIQ4_NL>(m, cols, x16, n, s);
    case quant::kQ4_0: return launch<quant::kQ4_0>(m, cols, x16, n, s);
    case quant::kIQ4_XS: return launch<quant::kIQ4_XS>(m, cols, x16, n, s);
    case quant::kQ2_0: return launch<quant::kQ2_0>(m, cols, x16, n, s);
    case quant::kNVFP4: return launch<quant::kNVFP4>(m, cols, x16, n, s);
    case quant::kQ4_K: return launch<quant::kQ4_K>(m, cols, x16, n, s);
    case quant::kQ5_1: return launch<quant::kQ5_1>(m, cols, x16, n, s);
    case quant::kMXFP4: return launch<quant::kMXFP4>(m, cols, x16, n, s);
    case quant::kMXFP8: return launch<quant::kMXFP8>(m, cols, x16, n, s);
    case quant::kBF16: return launch<quant::kBF16>(m, cols, x16, n, s);
    default: return false;
  }
}

}  // namespace

// the table's types, and the repacked Q5_K the forward makes at load (no file's type: no row)
extern "C" bool mm_known(uint32_t type) { return type == quant::kQ5_KR || (qtype_paths(type) & QTYPE_MM); }

template <quant::Type T>
bool fits(uint32_t rows, uint32_t cols) {
  const int slab = odd<T>(cols) ? Odd<T>::S::kSlab : cols < kShortCols ? Short<T>::S::kSlab : Tuned<T>::S::kSlab;
  return rows && cols && !(rows % mm::kTile) && !(cols % quant::Block<T>::values) && !(cols % slab);
}

extern "C" bool mm_fits(uint32_t type, uint32_t rows, uint32_t cols) {
  switch (type) {
#define FITS(t) \
  case quant::t: return fits<quant::t>(rows, cols);
    FITS(kQ8_0) FITS(kQ5_K) FITS(kQ5_KR) FITS(kQ6_K) FITS(kIQ1_M) FITS(kIQ2_XXS) FITS(kIQ2_XS) FITS(kIQ1_S)
    FITS(kIQ2_S) FITS(kIQ3_XXS) FITS(kIQ4_NL) FITS(kQ4_0) FITS(kIQ4_XS) FITS(kQ2_0) FITS(kQ2_K) FITS(kQ3_K) FITS(kQ4_K) FITS(kQ5_1)
    FITS(kMXFP4) FITS(kMXFP8) FITS(kNVFP4) FITS(kBF16)
#undef FITS
    default: return false;
  }
}

template <quant::Type T>
bool gemm_runs(uint32_t rows, uint32_t cols, uint32_t n) {
  const Mats m = {{nullptr, nullptr}, {rows, 0}, {nullptr, nullptr}, {rows, 0}};
  if (!fits<T>(rows, cols) || n > MM_MAX_TOKENS) return false;
  if constexpr (T == quant::kBF16)
    if (bf16_nine(cols)) return gemm_takes<T, Bf16Nine::S>(m, cols, n);
  if constexpr (kGemm<T>) return gemm_on<T>(m, cols, n);
  return false;
}

extern "C" bool mm_gemm(uint32_t type, uint32_t rows, uint32_t cols, uint32_t n) {
  switch (type) {
#define GEMM(t) \
  case quant::t: return gemm_runs<quant::t>(rows, cols, n);
    GEMM(kQ8_0) GEMM(kQ5_K) GEMM(kQ5_KR) GEMM(kQ6_K) GEMM(kNVFP4) GEMM(kBF16)
#undef GEMM
    default: return false;
  }
}

extern "C" uint32_t mm_col_tiles(uint32_t type, uint32_t n) {
  switch (type) {
#define TILES(t) \
  case quant::t: return col_tiles<quant::t>(n);
    TILES(kQ8_0) TILES(kQ5_K) TILES(kQ5_KR) TILES(kQ6_K) TILES(kIQ1_M) TILES(kIQ2_XXS) TILES(kIQ2_XS) TILES(kIQ1_S)
    TILES(kIQ2_S) TILES(kIQ3_XXS) TILES(kIQ4_NL) TILES(kQ4_0) TILES(kIQ4_XS) TILES(kQ2_0) TILES(kQ2_K) TILES(kQ3_K) TILES(kQ4_K) TILES(kQ5_1)
    TILES(kMXFP4) TILES(kMXFP8) TILES(kNVFP4) TILES(kBF16)
#undef TILES
    default: return 0;
  }
}

extern "C" bool mm_x16(const float* x, uint32_t xs, uint32_t n, uint32_t cols, void* x16, int32_t* overflow,
                       void* stream) {
  if (!n || n > MM_MAX_TOKENS || !cols || cols % mm::kTile || (n > 1 && xs < cols)) return false;
  const uint64_t work = (uint64_t) (n + mm::kTile - 1) / mm::kTile * mm::kTile * cols;  // whole tiles: past n, zero
  if ((uintptr_t) x % 16 == 0 && xs % 4 == 0 && (uintptr_t) x16 % 16 == 0 && work / kX16Vec < (1ull << 32))
    x16_vec_kernel<<<(unsigned) ((work / kX16Vec + kThreads - 1) / kThreads), kThreads, 0, (hipStream_t) stream>>>(
        x, xs, n, cols, (_Float16*) x16, overflow);
  else
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

// mm_run_x16's launch: launch_ct's shape at its counts (BF16's nine waves where it takes them: the same sums), a step
// the gemm would take refused (its caller's mm_run and mm_x16 instead)
template <quant::Type T, int CT>
bool x16_ct(const void* w, uint32_t rows, uint32_t cols, const void* x16, uint32_t n, float* y, uint32_t ys,
            mm::Silu ep, hipStream_t s) {
  if constexpr (T == quant::kBF16)
    if (bf16_nine(cols)) {
      const Mats m = {{w, nullptr}, {rows, 0}, {y, nullptr}, {ys, 0}};
      if (gemm_takes<T, Bf16Nine::S>(m, cols, n)) return false;
      return silu_shape<T, CT, Bf16Nine::S>(w, rows, cols, x16, n, y, ys, ep, s);
    }
  return silu_ct<T, CT>(w, rows, cols, x16, n, y, ys, ep, s);
}

template <quant::Type T>
bool x16_launch(const void* w, uint32_t rows, uint32_t cols, const void* x16, uint32_t n, float* y, uint32_t ys,
                mm::Silu ep, hipStream_t s) {
  int slab = odd<T>(cols) ? Odd<T>::S::kSlab : cols < kShortCols ? Short<T>::S::kSlab : Tuned<T>::S::kSlab;
  if constexpr (T == quant::kBF16)
    if (bf16_nine(cols)) slab = Bf16Nine::S::kSlab;
  if (rows % mm::kTile || (n > 1 && ys < rows) || cols % quant::Block<T>::values || cols % slab) return false;
  switch (col_tiles<T>(n)) {
    case 1: return x16_ct<T, 1>(w, rows, cols, x16, n, y, ys, ep, s);
    case 2: return x16_ct<T, 2>(w, rows, cols, x16, n, y, ys, ep, s);
    case 4: return x16_ct<T, 4>(w, rows, cols, x16, n, y, ys, ep, s);
    default: return x16_ct<T, Tuned<T>::kMaxCt>(w, rows, cols, x16, n, y, ys, ep, s);
  }
}

extern "C" bool mm_run_x16(uint32_t type, const void* w, uint32_t rows, uint32_t cols, const void* x16, uint32_t n,
                           float* y, uint32_t ys, void* x16_out, int32_t* overflow, void* stream) {
  if (!w || !x16 || !y || !x16_out || x16_out == x16 || !overflow || !n || n > MM_SILU_MAX_TOKENS || !rows) return false;
  const mm::Silu ep = {1.0f, (_Float16*) x16_out, overflow, rows, true};
  switch (type) {
    case quant::kBF16: return x16_launch<quant::kBF16>(w, rows, cols, x16, n, y, ys, ep, (hipStream_t) stream);
    case quant::kQ8_0: return x16_launch<quant::kQ8_0>(w, rows, cols, x16, n, y, ys, ep, (hipStream_t) stream);
    default: return false;  /* Inkling's control downs' types, so far */
  }
}

extern "C" bool mm_run_silu(uint32_t type, const void* w, uint32_t rows, uint32_t cols, const void* x16, uint32_t n,
                            float* y, uint32_t ys, float scale, void* x16_out, int32_t* overflow, void* stream) {
  if (!w || !x16 || !y || !x16_out || x16_out == x16 || !overflow || !n || n > MM_SILU_MAX_TOKENS || !rows) return false;
  const mm::Silu ep = {scale, (_Float16*) x16_out, overflow, rows, false};
  switch (type) {
    case quant::kQ8_0: return silu_launch<quant::kQ8_0>(w, rows, cols, x16, n, y, ys, ep, (hipStream_t) stream);
    default: return false;  /* the mixers' down's type alone, so far */
  }
}

// mm_run_gated_mean's launch: the shape launch_ct takes for the up's rows (the same sums), a matrix the gemm would take
// refused (the caller's mm_run and q4x_hc_gated_mean then)
template <quant::Type T, int CT, typename S>
bool gm_shape(const void* w, uint32_t cols, const void* x16, uint32_t n, const mm::GatedMean& gm, hipStream_t s) {
  const uint32_t z = (n + mm::kTile * CT - 1) / (mm::kTile * CT);
  mm::kernel_gated_mean<T, S, CT><<<dim3(gm.dim / mm::kTile, 1, z), mm::kWave * S::kWaves, 0, s>>>(
      (const uint8_t*) w, cols, (const _Float16*) x16, n, gm);
  return hipGetLastError() == hipSuccess;
}

template <quant::Type T, int CT>
bool gm_ct(const void* w, uint32_t cols, const void* x16, uint32_t n, const mm::GatedMean& gm, hipStream_t s) {
  const uint32_t rows = gm.hc * gm.dim;
  const Mats m = {{w, nullptr}, {rows, 0}, {nullptr, nullptr}, {rows, 0}};
  if constexpr (kGemm<T>)
    if (gemm_on<T>(m, cols, n)) return false;
  if (odd<T>(cols)) return gm_shape<T, CT, typename Odd<T>::S>(w, cols, x16, n, gm, s);
  return cols < kShortCols ? gm_shape<T, CT, typename Short<T>::S>(w, cols, x16, n, gm, s)
                           : gm_shape<T, CT, typename Tuned<T>::S>(w, cols, x16, n, gm, s);
}

template <quant::Type T>
bool gm_launch(const void* w, uint32_t cols, const void* x16, uint32_t n, const mm::GatedMean& gm, hipStream_t s) {
  const int slab = odd<T>(cols) ? Odd<T>::S::kSlab : cols < kShortCols ? Short<T>::S::kSlab : Tuned<T>::S::kSlab;
  if (cols % quant::Block<T>::values || cols % slab) return false;
  switch (col_tiles<T>(n)) {
    case 1: return gm_ct<T, 1>(w, cols, x16, n, gm, s);
    case 2: return gm_ct<T, 2>(w, cols, x16, n, gm, s);
    case 4: return gm_ct<T, 4>(w, cols, x16, n, gm, s);
    default: return gm_ct<T, Tuned<T>::kMaxCt>(w, cols, x16, n, gm, s);
  }
}

extern "C" bool mm_run_gated_mean(uint32_t type, const void* w, uint32_t hc, uint32_t dim, uint32_t cols,
                                  const void* x16, uint32_t n, const float* xn, float* out, void* out16, uint32_t row0,
                                  int32_t* overflow, void* stream) {
  if (!w || !x16 || !xn || !out || out16 == x16 || (out16 && !overflow) || !n || n > MM_MAX_TOKENS || !hc ||
      hc > MM_GATED_MEAN_MAX_HC || !dim || dim % mm::kTile)
    return false;
  const mm::GatedMean gm = {xn, out, (_Float16*) out16, overflow, row0, hc, dim};
  switch (type) {
    case quant::kQ8_0: return gm_launch<quant::kQ8_0>(w, cols, x16, n, gm, (hipStream_t) stream);
    default: return false;  /* the mixers' up's type alone, so far */
  }
}

extern "C" bool mm_run(uint32_t type, const void* w, uint32_t rows, uint32_t cols, const void* x16, uint32_t n,
                       float* y, uint32_t ys, void* stream) {
  return run(type, Mats{{w, nullptr}, {rows, 0}, {y, nullptr}, {ys, 0}}, cols, x16, n, (hipStream_t) stream);
}

extern "C" bool mm_batch(uint32_t type, uint32_t cols, const MmPart* parts, uint32_t count, uint32_t n,
                         void* stream) {
  static_assert(MM_BATCH_MAX == mm::kBatch && MM_BATCH_TOKENS == kGemmMinTokens, "mm.h's batch: its parts, its counts");
  if (!parts || !count || count > MM_BATCH_MAX || !n || n > MM_MAX_TOKENS) return false;
  const hipStream_t s = (hipStream_t) stream;
  if (n > kGemmMinTokens) {  // (the gemm's counts: each part its own launch, the gemm where it takes it)
    bool ok = true;
    for (uint32_t i = 0; ok && i < count; i++)
      ok = run(type, Mats{{parts[i].w, nullptr}, {parts[i].rows, 0}, {parts[i].y, nullptr}, {parts[i].ys, 0}}, cols,
               parts[i].x16, n, s);
    return ok;
  }
  mm::Parts p = {};
  for (uint32_t i = 0; i < count; i++)
    p.w[i] = (const uint8_t*) parts[i].w, p.x16[i] = (const _Float16*) parts[i].x16, p.y[i] = parts[i].y,
    p.rows[i] = parts[i].rows, p.ys[i] = parts[i].ys;
  switch (type) {
#define BATCH(t) \
  case quant::t: return batch<quant::t>(p, count, cols, n, s);
    BATCH(kQ8_0) BATCH(kQ5_K) BATCH(kQ5_KR) BATCH(kQ6_K) BATCH(kIQ1_M) BATCH(kIQ2_XXS) BATCH(kIQ2_XS) BATCH(kIQ1_S)
    BATCH(kIQ2_S) BATCH(kIQ3_XXS) BATCH(kQ2_K) BATCH(kQ3_K) BATCH(kIQ4_NL) BATCH(kQ4_0) BATCH(kIQ4_XS) BATCH(kQ2_0)
    BATCH(kNVFP4) BATCH(kQ4_K) BATCH(kQ5_1) BATCH(kMXFP4) BATCH(kMXFP8) BATCH(kBF16)
#undef BATCH
    default: return false;
  }
}

extern "C" bool mm_run2(uint32_t type, const void* w0, uint32_t rows0, float* y0, uint32_t ys0, const void* w1,
                        uint32_t rows1, float* y1, uint32_t ys1, uint32_t cols, const void* x16, uint32_t n,
                        void* stream) {
  return w0 && w1 &&
         run(type, Mats{{w0, w1}, {rows0, rows1}, {y0, y1}, {ys0, ys1}}, cols, x16, n, (hipStream_t) stream);
}

extern "C" bool mm_tokens_scaled(uint32_t type, const void* w0, uint32_t rows0, float* y0, uint32_t ys0, float s0,
                                 const void* w1, uint32_t rows1, float* y1, uint32_t ys1, float s1, uint32_t cols,
                                 const float* x, uint32_t xs, uint32_t n, void* x16, int32_t* overflow, bool ready,
                                 void* stream) {
  if (!w0) return false;
  bool ok = true;
  for (uint32_t k0 = 0; ok && k0 < n; k0 += MM_MAX_TOKENS) {
    const uint32_t m = n - k0 < MM_MAX_TOKENS ? n - k0 : MM_MAX_TOKENS;
    Mats ms{{w0, w1}, {rows0, w1 ? rows1 : 0}, {y0 + (uint64_t) k0 * ys0, w1 ? y1 + (uint64_t) k0 * ys1 : nullptr},
            {ys0, w1 ? ys1 : 0}};
    ms.scale[0] = s0, ms.scale[1] = s1;
    ok = ((ready && n <= MM_MAX_TOKENS) || mm_x16(x + (uint64_t) k0 * xs, xs, m, cols, x16, overflow, stream)) &&
         run(type, ms, cols, x16, m, (hipStream_t) stream);
  }
  return ok;
}

extern "C" bool mm_x16_whole(const float* x, uint32_t xs, uint32_t n, uint32_t cols, void* x16, int32_t* overflow,
                             void* stream) {
  if (!x || !x16 || !n || n > MM_X16_ROWS_MAX) return false;
  bool ok = true;
  for (uint32_t k0 = 0; ok && k0 < n; k0 += MM_MAX_TOKENS) {
    const uint32_t m = n - k0 < MM_MAX_TOKENS ? n - k0 : MM_MAX_TOKENS;
    ok = mm_x16(x + (uint64_t) k0 * xs, xs, m, cols, (_Float16*) x16 + mm::x16_at(k0, 0, cols), overflow, stream);
  }
  return ok;
}

extern "C" bool mm_tokens_x16(uint32_t type, const void* w0, uint32_t rows0, float* y0, uint32_t ys0, const void* w1,
                              uint32_t rows1, float* y1, uint32_t ys1, uint32_t cols, const void* x16, uint32_t n,
                              void* stream) {
  if (!w0 || !x16 || !n || n > MM_X16_ROWS_MAX || !cols || cols % mm::kTile) return false;
  bool ok = true;
  for (uint32_t k0 = 0; ok && k0 < n; k0 += MM_MAX_TOKENS) {  // (k0 a whole tile: a slice x16 + k0 cols)
    const uint32_t m = n - k0 < MM_MAX_TOKENS ? n - k0 : MM_MAX_TOKENS;
    Mats ms{{w0, w1}, {rows0, w1 ? rows1 : 0}, {y0 + (uint64_t) k0 * ys0, w1 ? y1 + (uint64_t) k0 * ys1 : nullptr},
            {ys0, w1 ? ys1 : 0}};
    ms.scale[0] = 1.0f, ms.scale[1] = 1.0f;
    ok = run(type, ms, cols, (const _Float16*) x16 + mm::x16_at(k0, 0, cols), m, (hipStream_t) stream);
  }
  return ok;
}

extern "C" bool mm_tokens(uint32_t type, const void* w0, uint32_t rows0, float* y0, uint32_t ys0, const void* w1,
                          uint32_t rows1, float* y1, uint32_t ys1, uint32_t cols, const float* x, uint32_t xs,
                          uint32_t n, void* x16, int32_t* overflow, bool ready, void* stream) {
  return mm_tokens_scaled(type, w0, rows0, y0, ys0, 1.0f, w1, rows1, y1, ys1, 1.0f, cols, x, xs, n, x16, overflow,
                          ready, stream);
}

// every head's slice of x's rows to its own tiled x16 in one pass (x read along its rows: a head's slice converted
// alone reads 2,048 rows 64 KB apart, a page each): token t's head h, column c (c < cols, at h hs + c of
// its row) to x16 + h n16 cols at x16_at(t, c, cols), n16 n in whole tiles (the rows past n zero). x16_put's
// conversion: the same bits as mm_x16 a head at a time
__global__ void x16_heads_kernel(const float* x, uint32_t xs, uint32_t n, uint32_t heads, uint32_t hs, uint32_t cols,
                                 _Float16* x16, int32_t* overflow) {
  const uint32_t per = cols / kX16Vec, i = blockIdx.x * kThreads + threadIdx.x, t = i / (heads * per);
  const uint32_t h = i / per % heads, c = i % per * kX16Vec, n16 = (n + mm::kTile - 1) / mm::kTile * mm::kTile;
  if (t >= n16) return;
  float4 a = make_float4(0, 0, 0, 0), b = a;
  if (t < n) {
    const float4* p = (const float4*) (x + (uint64_t) t * xs + (uint64_t) h * hs + c);
    a = p[0], b = p[1];
  }
  const float v[kX16Vec] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  __attribute__((aligned(16))) _Float16 o[kX16Vec];
  bool bad = false;
#pragma unroll
  for (uint32_t e = 0; e < kX16Vec; e++) bad |= !(fabsf(v[e]) <= mm::kF16Max), o[e] = (_Float16) v[e];
  if (bad) atomicOr(overflow, 1);
  *(uint4*) (x16 + (uint64_t) h * n16 * cols + mm::x16_at(t, c, cols)) = *(const uint4*) o;
}

extern "C" bool mm_x16_heads(const float* x, uint32_t xs, uint32_t n, uint32_t heads, uint32_t hs, uint32_t cols,
                             void* x16, int32_t* overflow, void* stream) {
  if (!n || n > MM_MAX_TOKENS || !heads || !cols || cols % mm::kTile || hs % 4 || xs % 4 || (uintptr_t) x % 16 ||
      (uintptr_t) x16 % 16 || (heads - 1) * (uint64_t) hs + cols > xs)
    return false;
  const uint64_t work = (uint64_t) (n + mm::kTile - 1) / mm::kTile * mm::kTile * heads * (cols / kX16Vec);
  if (work >= (1ull << 32)) return false;
  x16_heads_kernel<<<(unsigned) ((work + kThreads - 1) / kThreads), kThreads, 0, (hipStream_t) stream>>>(
      x, xs, n, heads, hs, cols, (_Float16*) x16, overflow);
  return hipGetLastError() == hipSuccess;
}

extern "C" bool mm_x16_rows(const float* x, uint32_t xs, const int32_t* idx, uint32_t slots, uint32_t cols, void* x16,
                            int32_t* overflow, void* stream) {
  // a run's two float4 loads: x's rows 16-byte aligned (hipMalloc's, a stride of 4s)
  if (!slots || !cols || cols % mm::kTile || xs < cols || xs % 4 || (uintptr_t) x % 16) return false;
  const uint64_t runs = (uint64_t) (slots + mm::kTile - 1) / mm::kTile * mm::kTile * cols / kRun;
  x16_rows_kernel<<<(unsigned) ((runs + kThreads - 1) / kThreads), kThreads, 0, (hipStream_t) stream>>>(
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
    GROUP(kQ8_0) GROUP(kQ5_K) GROUP(kQ5_KR) GROUP(kQ6_K) GROUP(kIQ1_M) GROUP(kIQ2_XXS) GROUP(kIQ2_XS) GROUP(kIQ1_S)
    GROUP(kIQ2_S) GROUP(kIQ3_XXS) GROUP(kIQ4_NL) GROUP(kQ4_0) GROUP(kIQ4_XS) GROUP(kQ2_0) GROUP(kQ2_K) GROUP(kQ3_K) GROUP(kQ4_K) GROUP(kQ5_1)
    GROUP(kMXFP4) GROUP(kMXFP8) GROUP(kNVFP4)
#undef GROUP
    default: return false;
  }
}
