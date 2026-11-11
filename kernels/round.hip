// ops (include/osmos/gpu/ops.h): the reference kernels' activation roundings - act_quant's FP8, fp4_act_quant's
// FP4 - and BF16, each value as ref.py rounds it
#include <cmath>
#include "launch.h"

#include "kvload.h"
#include "round.h"

extern "C" {
#include "osmos/gpu/ops.h"
}

namespace {

using launches::kThreads;
using launches::grid;
using launches::st;
using launches::ok;

using namespace rnd;

constexpr int kWave = 32;
constexpr int kWaves = kThreads / kWave;
constexpr float kFp4Max = 6.0f;          // E2M1's largest
constexpr int kE4m3ScaleFloorExp = -9;   // fp4_act_quant's E4M3-scale amax floor: 6 * 2^-9
constexpr int kPow2ScaleFloorExp = -126; // and its power-of-two one: 6 * 2^-126

unsigned wave_grid(uint64_t blocks) { return (unsigned) ((blocks + kWaves - 1) / kWaves); }

// a wave a block, a lane a value (fp4's longer blocks: lanes strided): the loads coalesced, the max round.h's xor tree
// packed (or NULL): rows of dim as kvfmt.h's KV_FP8 - a value's code at its place, a block's scale after the row's codes
__global__ void fp8_kernel(float* x, uint64_t blocks, uint8_t* packed, uint32_t dim) {
  static_assert(kWave == kRndWave, "round.h's wave");
  const uint64_t b = (uint64_t) blockIdx.x * kWaves + threadIdx.x / kWave;
  if (b >= blocks) return;
  const uint64_t i = b * kFp8Block + threadIdx.x % kWave;
  uint32_t code, scale;
  x[i] = fp8_block_coded(x[i], code, scale);
  if (packed) {
    uint8_t* row = packed + i / dim * kv_row_bytes(KV_FP8, dim);
    const uint32_t j = (uint32_t) (i % dim);
    row[j] = (uint8_t) code;
    if (j % kFp8Block == 0) row[dim + j / kFp8Block] = (uint8_t) scale;
  }
}

// a block of `block` values (at most a wave's: a lane a value); packed: rows of dim, two codes a byte (the even
// value's the low nibble), a block's scale after the row's codes - its E4M3 code (e4m3_scale) or its UE8M0
__global__ void fp4_kernel(float* x, uint64_t blocks, uint32_t block, bool e4m3_scale, uint8_t* packed,
                           uint32_t dim) {
  const uint64_t b = (uint64_t) blockIdx.x * kWaves + threadIdx.x / kWave;
  if (b >= blocks) return;
  float* v = x + b * block;
  const uint32_t lane = threadIdx.x % kWave;
  float amax = 0;
  for (uint32_t j = lane; j < block; j += kWave) amax = fmaxf(amax, fabsf(v[j]));
  amax = wave_amax(amax);
  float s;
  if (e4m3_scale)
    s = e4m3(fmaxf(amax, kFp4Max * ldexpf(1.0f, kE4m3ScaleFloorExp)) / kFp4Max);
  else
    s = exp2f(ceilf(log2f(fmaxf(amax, kFp4Max * ldexpf(1.0f, kPow2ScaleFloorExp)) / kFp4Max)));
  uint32_t code = 0;
  for (uint32_t j = lane; j < block; j += kWave) {
    const float q = e2m1(fminf(fmaxf(v[j] / s, -kFp4Max), kFp4Max));
    v[j] = q * s, code = e2m1_code(q);
  }
  if (!packed) return;
  const uint32_t pair = __shfl_xor(code, 1, kWave);  // (a wave the whole block: every lane here)
  const uint64_t i = b * block + lane;
  uint8_t* row = packed + i / dim * kv_row_bytes(e4m3_scale ? KV_FP4_E4M3 : KV_FP4_POW2, dim);
  const uint32_t j = (uint32_t) (i % dim);
  if (lane < block && lane % 2 == 0) row[j / 2] = (uint8_t) (code | pair << 4);
  /* an E4M3 scale past E4M3's 448 (a block's amax past 6 x 448: none of the model's own) kept as E4M3's NaN, so its
     values read back NaN - loud (forward_greedy refuses a row with one), never a wrong value */
  if (lane == 0)
    row[dim / 2 + j / block] = (uint8_t) (e4m3_scale ? (s > kFp8Max ? kE4m3Nan : e4m3_code(s)) : ue8m0_code(s));
}

// values on a format's grid already (each its code's value times one scale a block: another's rounding, the
// oracle's) packed so each reads back as itself: a wave a block (at most 32: a lane a value), the block's scale the
// largest whose every value is its code's value times it, exactly - E4M3's 126 finite scales, or a power of two's
// 254. a block with none (not on the grid) packs NaN's code (E4M3's 0x7F, UE8M0's 0xFF): its values read back NaN
constexpr uint32_t kUe8m0Nan = 0xFF;
__global__ void pack_kernel(uint32_t fmt, const float* src, uint64_t blocks, uint32_t block, uint32_t dim,
                            uint8_t* dst) {
  const uint64_t b = (uint64_t) blockIdx.x * kWaves + threadIdx.x / kWave;
  if (b >= blocks) return;
  const uint32_t lane = threadIdx.x % kWave;
  const uint64_t i = b * block + lane;
  const bool mine = lane < block;
  const float v = mine ? src[i] : 0.0f;
  const bool e4m3_scale = fmt == KV_FP4_E4M3, fp8 = fmt == KV_FP8;
  const int n_codes = e4m3_scale ? 0x7F : kUe8m0Nan;  // the finite scales, codes 0 .. n - 1
  uint32_t scale = e4m3_scale ? kE4m3Nan : kUe8m0Nan, code = 0;
  for (int c = n_codes - 1; c >= 0; c--) {
    const float s = e4m3_scale ? e4m3_value((uint32_t) c) : ue8m0_value((uint32_t) c);
    if (!(s > 0.0f) || !isfinite(s)) continue;
    const float q = v / s;
    const float back = fp8 ? e4m3(fminf(fmaxf(q, -kFp8Max), kFp8Max)) : e2m1(fminf(fmaxf(q, -kFp4Max), kFp4Max));
    const bool exact = !mine || (fabsf(q) <= (fp8 ? kFp8Max : kFp4Max) && back * s == v &&
                                 signbit(back * s) == signbit(v));
    if (__all(exact)) {
      scale = (uint32_t) c, code = fp8 ? e4m3_code(back) : e2m1_code(back);
      break;
    }
  }
  uint8_t* row = dst + i / dim * kv_row_bytes((KvFmt) fmt, dim);
  const uint32_t j = (uint32_t) (i % dim);
  if (fp8) {
    if (mine) row[j] = (uint8_t) code;
    if (!lane) row[dim + j / block] = (uint8_t) scale;
    return;
  }
  const uint32_t pair = __shfl_xor(code, 1, kWave);
  if (mine && lane % 2 == 0) row[j / 2] = (uint8_t) (code | pair << 4);
  if (!lane) row[dim / 2 + j / block] = (uint8_t) scale;
}

// four values a thread: kvload.h's reading
__global__ void unpack_kernel(uint32_t fmt, const uint8_t* src, uint64_t quads, uint32_t dim, float* dst) {
  const uint64_t q = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (q >= quads) return;
  const uint64_t i = q * 4, r = i / dim;
  *(kv::f4*) (dst + i) = kv::load4(fmt, src + r * kv_row_bytes((KvFmt) fmt, dim), dim, (uint32_t) (i % dim));
}

__global__ void bf16_kernel(float* x, uint64_t n) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) x[i] = bf16_round(x[i]);
}

}  // namespace

// a packed format's rows: dim whole blocks, n whole rows; the FP4 blocks a wave's at most (packed: a lane a value)
static bool rows_ok(uint64_t n, uint32_t dim, uint32_t block) {
  return dim && dim % block == 0 && dim % 4 == 0 && n % dim == 0 && block <= (uint32_t) kWave;
}

extern "C" bool ops_fp4_round(float* x, uint64_t n, uint32_t block, bool e4m3_scale, void* packed, uint32_t dim,
                              void* stream) {
  if (!n || !block || n % block) return false;
  if (packed && (!rows_ok(n, dim, block) || block != (e4m3_scale ? KV_FP4_E4M3_BLOCK : KV_FP4_POW2_BLOCK)))
    return false;
  fp4_kernel<<<wave_grid(n / block), kThreads, 0, st(stream)>>>(x, n / block, block, e4m3_scale, (uint8_t*) packed,
                                                                dim);
  return ok();
}

extern "C" bool ops_fp8_round(float* x, uint64_t n, void* packed, uint32_t dim, void* stream) {
  if (!n || n % kFp8Block || (packed && !rows_ok(n, dim, kFp8Block))) return false;
  fp8_kernel<<<wave_grid(n / kFp8Block), kThreads, 0, st(stream)>>>(x, n / kFp8Block, (uint8_t*) packed, dim);
  return ok();
}

extern "C" bool ops_kv_pack(uint32_t fmt, const float* src, uint64_t rows, uint32_t dim, void* dst, void* stream) {
  if (fmt < KV_FP8 || fmt > KV_FP4_POW2 || !rows || !rows_ok(rows * dim, dim, kv_block((KvFmt) fmt))) return false;
  const uint32_t block = kv_block((KvFmt) fmt);
  const uint64_t blocks = rows * dim / block;
  pack_kernel<<<wave_grid(blocks), kThreads, 0, st(stream)>>>(fmt, src, blocks, block, dim, (uint8_t*) dst);
  return ok();
}

extern "C" bool ops_kv_unpack(uint32_t fmt, const void* src, uint64_t rows, uint32_t dim, float* dst, void* stream) {
  if (fmt > KV_FP4_POW2 || !rows || !dim || dim % kv_block((KvFmt) fmt) || dim % 4) return false;
  const uint64_t quads = rows * dim / 4;
  unpack_kernel<<<grid(quads), kThreads, 0, st(stream)>>>(fmt, (const uint8_t*) src, quads, dim, dst);
  return ok();
}

extern "C" bool ops_round_bf16(float* x, uint64_t n, void* stream) {
  if (!n) return false;
  bf16_kernel<<<grid(n), kThreads, 0, st(stream)>>>(x, n);
  return ok();
}

