// ops (include/substrate/gpu/ops.h): the reference kernels' activation roundings - act_quant's FP8, fp4_act_quant's
// FP4 - and BF16, each value as ref.py rounds it
#include <cmath>

#include "round.h"

extern "C" {
#include "substrate/gpu/ops.h"
}

namespace {

using namespace rnd;

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kWaves = kThreads / kWave;
constexpr float kFp4Max = 6.0f;          // E2M1's largest
constexpr int kE4m3ScaleFloorExp = -9;   // fp4_act_quant's E4M3-scale amax floor: 6 * 2^-9
constexpr int kPow2ScaleFloorExp = -126; // and its power-of-two one: 6 * 2^-126
constexpr float kFp8Floor = 1e-4f;  // act_quant's amax floor
constexpr int kFp8Block = 32;

unsigned grid(uint64_t n) { return (unsigned) ((n + kThreads - 1) / kThreads); }
unsigned wave_grid(uint64_t blocks) { return (unsigned) ((blocks + kWaves - 1) / kWaves); }
bool ok() { return hipGetLastError() == hipSuccess; }
hipStream_t st(void* s) { return (hipStream_t) s; }

// a wave a block, a lane a value (fp4's longer blocks: lanes strided): the loads coalesced, the max an xor tree - exact
// in any order, so the same bits as a value at a time
__device__ inline float wave_amax(float m) {
  for (int off = kWave / 2; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor(m, off, kWave));
  return m;
}

__global__ void fp8_kernel(float* x, uint64_t blocks) {
  static_assert(kFp8Block == kWave, "a lane a value");
  const uint64_t b = (uint64_t) blockIdx.x * kWaves + threadIdx.x / kWave;
  if (b >= blocks) return;
  float* v = x + b * kFp8Block + threadIdx.x % kWave;
  const float amax = wave_amax(fabsf(*v));
  const float s = exp2f(ceilf(log2f(fmaxf(amax, kFp8Floor) / kFp8Max)));
  *v = e4m3(fminf(fmaxf(*v / s, -kFp8Max), kFp8Max)) * s;
}

__global__ void fp4_kernel(float* x, uint64_t blocks, uint32_t block, bool e4m3_scale) {
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
  for (uint32_t j = lane; j < block; j += kWave) v[j] = e2m1(fminf(fmaxf(v[j] / s, -kFp4Max), kFp4Max)) * s;
}

__global__ void bf16_kernel(float* x, uint64_t n) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) x[i] = bf16_round(x[i]);
}

}  // namespace

extern "C" bool ops_fp4_round(float* x, uint64_t n, uint32_t block, bool e4m3_scale, void* stream) {
  if (!n || !block || n % block) return false;
  fp4_kernel<<<wave_grid(n / block), kThreads, 0, st(stream)>>>(x, n / block, block, e4m3_scale);
  return ok();
}

extern "C" bool ops_fp8_round(float* x, uint64_t n, void* stream) {
  if (!n || n % kFp8Block) return false;
  fp8_kernel<<<wave_grid(n / kFp8Block), kThreads, 0, st(stream)>>>(x, n / kFp8Block);
  return ok();
}

extern "C" bool ops_round_bf16(float* x, uint64_t n, void* stream) {
  if (!n) return false;
  bf16_kernel<<<grid(n), kThreads, 0, st(stream)>>>(x, n);
  return ok();
}

