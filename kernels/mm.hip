// mm (include/osmos/gpu/mm.h): the kernel is mm.h's
#include <cmath>

#include "mm.h"
#include "round.h"

extern "C" {
#include "osmos/gpu/mm.h"
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
// a short row's shape (fewer than kShortCols columns): a type's own where measured, else S. a matrix's shape is its
// type's and its columns' alone, never its tokens' - the same sums at every count. Q6_K's (unsloth's UD: q_b's 1,280
// columns, 2026-11-03, test_mm_gpu): 8 waves of 64-value slabs 170.8 -> 197.0 GB/s at a token, its error 1.6 ->
// 1.4 u|wx| (4 waves of 128: 200.0 GB/s, but 2.9 u|wx| - a wave's longer run); its long rows keep 128 x 8 (out_b's
// 8,192 columns: 202.4 GB/s against 144.1 at 64 x 8)
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
template <> struct Odd<quant::kQ8_0> { using S = mm::Shape<64, 2, 4>; };  // 10240 x 320 at a token: 32.4 -> 22.2 us (mmsweep)
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
// Qwen3.8's expert types, a token's 10 experts cold (tools/bench/mmsweep, 2026-11-06; SL 64-256, W 4-8, U 2-8):
// IQ4_NL down 69.5 -> 57.7 us at 64 x 4 x 4, IQ4_XS 68.7 -> 53.0 the same, IQ3_XXS gate / up 58.7 -> 53.5 at
// 128 x 8 x 8, Q2_0 45.3 -> 39.9 at 64 x 8 x 4
template <> struct Tuned<quant::kIQ3_XXS> { using S = mm::Shape<128, 8, 8>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kIQ4_NL> { using S = mm::Shape<64, 4, 4>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kIQ4_XS> { using S = mm::Shape<64, 4, 4>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kQ4_0> : Tuned<quant::kIQ4_NL> {};  // not measured: IQ4_NL's (its layout, its block)
template <> struct Tuned<quant::kQ2_0> { using S = mm::Shape<64, 8, 4>; static constexpr int kMaxCt = 8; };
template <> struct Tuned<quant::kQ2_K> : Tuned<quant::kQ2_0> {};
template <> struct Tuned<quant::kQ3_K> : Tuned<quant::kQ2_0> {};  // not measured: Q2_K's (a 256-value K block)  // not measured: Q2_0's (its shape its bits: fixed before a pack)
// NVFP4 (Inkling-2's experts: 4,608 and 2,304 columns, whole 64-value slabs) not measured: Q2_0's, a 64-value block too.
// its values exact in f16 (6 significant bits, 2^-10 .. 2,688: decode_h rounds none); a tensor's own scale its role's
template <> struct Tuned<quant::kNVFP4> : Tuned<quant::kQ2_0> {};
// MXFP8 (Inkling-2's experts: Small's layers 40-47, Mini's 20-23; first unsloth's Engram tables, decode alone). its
// values exact in f16 where e4m3 x 2^(e - 127) is one: Inkling's (e -15 .. -10, |w| 2^-22 .. 0.375) all are -
// ink2_ref_gguf.py checks every expert's. measured (mmsweep mini, low-power, 2026-11-20; decode_h<kMXFP8> by bits): 8
// experts at one token, Mini's 1,152 x 2,304 gate / up and 2,304 x 1,152 down - the default 128 x 8 x 8 95 / 89 GB/s
// (the generic decode_h's 57 / 48), 64 x 4 x 4 155 / 151 (a plain read 224)
template <> struct Tuned<quant::kMXFP8> { using S = mm::Shape<64, 4, 4>; static constexpr int kMaxCt = 8; };
// BF16 (Inkling-2-Mini's dense weights, as released): a value f16's where it is one - all but those below f16's
// normal range (Mini's: 322,439 of 1.93 G, each within 2^-25 of its own); not in mm_known's table (QTYPE_MM): a
// family takes it by name (ink2's mm_takes), the others' BF16 matrices their matvec's as before
#ifndef MM_BF16_SHAPE
#define MM_BF16_SHAPE 128, 8, 8
#endif
template <> struct Tuned<quant::kBF16> { using S = mm::Shape<MM_BF16_SHAPE>; static constexpr int kMaxCt = 8; };
// a long BF16 row whose 128-value slabs 9 waves divide evenly (Mini's 2,304 columns: q, gate, k, v, rel, the router,
// the head; the loop controls' down 4,608): 9 waves, each 2 slabs (4 for down) - mmsweep bf16 at a token, low-power:
// q / gate 3,072 x 2,304 197 -> 228 GB/s (the matvec's 210), k / v 1,024 x 2,304 190 -> 200 (173); 3,072 columns
// (24 slabs: the output) keep 128 x 8 x 8 (217; 9 waves 204). by its columns alone, so a matrix's every count
struct Bf16Nine { using S = mm::Shape<128, 9, 8>; };
constexpr bool bf16_nine(uint32_t cols) { return cols >= kShortCols && cols % (128 * 9) == 0; }
// UD-Q4_K_XL's expert types, on the register-A path (mm.h's kRegA), a token's 10 experts cold (mmsweep q4, low-power,
// 2026-11-09, two runs): Q4_K's 2,560-column gate / up 78-80 us at 128 x 8 x 8 -> 58-60 at 32 x 16 x 2 (80 slabs of
// 32 for 16 waves to share: the decode's latency hidden; a 32-value slab's lane holds 2 chunks, so U 2 at most), Q5_1's
// 640-column down 72-75 -> 63-64 at 32 x 8 x 2 (16 waves slower: 20 slabs too few). a 512-token prompt 409 -> 507
// tok/s, decode 22.0 -> 23.2. the perplexity the default order's within its scatter (2K 6.9661 against 6.9754, 8K
// 2.7434 against 2.7354); the 64-value slabs' 64 x 8 x 4 / 64 x 4 x 4 (through LDS) moved it up at every length
// (+0.5%, +1.3%) and were not taken (DESIGN.md, "UD-Q4_K_XL")
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
// x16_put's conversion: the same bits. (one a thread, a 64-bit divide each: ~15 GB/s, the prompt's 6.9 s at 32K)
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
// x16_put's rounding and flag, so the same bytes as a value a thread (its 2-byte stores scattered: a 2048-token
// step on the IQ3 pack 2.51 -> 1.09 ms a layer, low-power, 2026-11-10)
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

// the experts' multi-tile steps by type, each the bits of group_ct's (the sums in tile's order: test_mm_gpu's mm_group
// at 24 / 37 / 64 / 100 / 128 tokens against mm_run): 2-4 column tiles with 2 row tiles a wave (tile_rm: an x fragment
// two WMMAs), 8 as gemm_kernel's 64 x 128 blocks - measured a type at a time (tools/bench/mmsweep's experts, real
// blocks, 160 experts cold, low-power, 2026-11-09): Q4_K +22 / +42 / +17% at 24 / 37 / 64 tokens, +13% at 128; Q5_1
// +6 / +28 / +8%, +42%; IQ4_XS +7 / +6 / +9% (its 128s level: group_ct's); IQ4_NL and Q2_0 level at 24-64, +16% and
// +60% at 128. IQ3_XXS's 128 x 8 shape takes no 2 row tiles (70 KB of LDS). DeepSeek's (2,304 x 5,120, 64 experts):
// 2 row tiles 25-68% slower for every codebook type (IQ2_XXS / XS / S, IQ1_S: twice a lane's grid lookups), Q2_K
// +30 / +7 / +7%; the gemm at 128 IQ3_XXS +41%, Q2_K +35%, IQ2_S +16%, IQ1_S +6-10%, IQ2_XXS and IQ2_XS -4 .. +6% by
// shape (not taken); Qwen's IQ3_XXS gate level (-1%). the other types not measured: group_ct's
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
// holds at once, each taking tiles until none are left (IQ1_S, a block a tile: 204 against 188 us; mmsweep ink2).
// what a WGP holds: its VGPRs' blocks or its LDS's, the fewer - the LDS both CUs' (WGP mode; the runtime's figure,
// and its occupancy call, one CU's 64 KB: 7 blocks a WGP, 211 us, against the 12 the VGPRs allow, 190 us)
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

// a prompt's matrix on mm.h's gemm_kernel: Q6_K and Q8_0 (both models' dense), rows of at least kShortCols (tile's
// shape, every partition a slab) and more than 64 tokens (CT 8's: below, tile's own); the same bits as tile at every
// count (test_mm_gpu: 2,048, 64 and 33 tokens against 16). 64 rows x 256 tokens a block, a wave 32 x 64: 10240 x
// 2560 Q6_K at 2,048 tokens 20.6 -> 27.0 TFLOP/s, Q8_0 20.5 -> 28.7 (low-power; 64 x 128: 22.8, 32 x 256: 22.4,
// 128 x 128: past the LDS; without the second slab's LDS 25.5)
template <quant::Type T>
constexpr bool kGemm = T == quant::kQ6_K || T == quant::kQ8_0 || T == quant::kNVFP4 || T == quant::kBF16;
constexpr int kGemmRm = 2, kGemmRn = 4, kGemmWr = 2, kGemmWc = 4;
// a type's gemm blocks (kGemm*'s but where measured): its rows and tokens a wave in tiles, its waves' grid
template <quant::Type T> struct GemmPlan { static constexpr int Rm = kGemmRm, Rn = kGemmRn, Wr = kGemmWr, Wc = kGemmWc; };
#ifdef GEMM_NVFP4
template <> struct GemmPlan<quant::kNVFP4> { static constexpr int Rm = GEMM_RM, Rn = GEMM_RN, Wr = GEMM_WR, Wc = GEMM_WC; };
#endif
constexpr uint32_t kGemmMinTokens = 4 * mm::kTile;  // past it: CT 8's counts

template <quant::Type T, class S = typename Tuned<T>::S>
bool gemm_takes(const Mats& m, uint32_t cols, uint32_t n) {
  return kGemm<T> && n > kGemmMinTokens && !odd<T>(cols) && cols >= kShortCols &&
         cols / S::kSlab >= (uint32_t) S::kWaves;
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
    if (gemm_takes<T>(m, cols, n)) return launch_gemm<T>(m, cols, x16, n, s);
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
    if (gemm_takes<T>(m, cols, n)) return false;
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

extern "C" bool mm_run_silu(uint32_t type, const void* w, uint32_t rows, uint32_t cols, const void* x16, uint32_t n,
                            float* y, uint32_t ys, float scale, void* x16_out, int32_t* overflow, void* stream) {
  if (!w || !x16 || !y || !x16_out || x16_out == x16 || !overflow || !n || n > MM_SILU_MAX_TOKENS || !rows) return false;
  const mm::Silu ep = {scale, (_Float16*) x16_out, overflow, rows};
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
    if (gemm_takes<T>(m, cols, n)) return false;
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

extern "C" bool mm_tokens(uint32_t type, const void* w0, uint32_t rows0, float* y0, uint32_t ys0, const void* w1,
                          uint32_t rows1, float* y1, uint32_t ys1, uint32_t cols, const float* x, uint32_t xs,
                          uint32_t n, void* x16, int32_t* overflow, bool ready, void* stream) {
  return mm_tokens_scaled(type, w0, rows0, y0, ys0, 1.0f, w1, rows1, y1, ys1, 1.0f, cols, x, xs, n, x16, overflow,
                          ready, stream);
}

// every head's slice of x's rows to its own tiled x16 in one pass (x read along its rows: a head's slice converted
// alone reads 2,048 rows 64 KB apart, a page each - ~15 GB/s): token t's head h, column c (c < cols, at h hs + c of
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
