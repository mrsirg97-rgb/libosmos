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
constexpr float kFp4Max = 6.0f;          // E2M1's largest
constexpr int kE4m3ScaleFloorExp = -9;   // fp4_act_quant's E4M3-scale amax floor: 6 * 2^-9
constexpr int kPow2ScaleFloorExp = -126; // and its power-of-two one: 6 * 2^-126
constexpr float kFp8Floor = 1e-4f;  // act_quant's amax floor
constexpr int kFp8Block = 32;

unsigned grid(uint64_t n) { return (unsigned) ((n + kThreads - 1) / kThreads); }
bool ok() { return hipGetLastError() == hipSuccess; }
hipStream_t st(void* s) { return (hipStream_t) s; }

__global__ void fp8_kernel(float* x, uint64_t blocks) {
  const uint64_t b = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (b >= blocks) return;
  float* v = x + b * kFp8Block;
  float amax = 0;
  for (int j = 0; j < kFp8Block; j++) amax = fmaxf(amax, fabsf(v[j]));
  const float s = exp2f(ceilf(log2f(fmaxf(amax, kFp8Floor) / kFp8Max)));
  for (int j = 0; j < kFp8Block; j++) v[j] = e4m3(fminf(fmaxf(v[j] / s, -kFp8Max), kFp8Max)) * s;
}


__global__ void fp4_kernel(float* x, uint64_t blocks, uint32_t block, bool e4m3_scale) {
  const uint64_t b = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (b >= blocks) return;
  float* v = x + b * block;
  float amax = 0;
  for (uint32_t j = 0; j < block; j++) amax = fmaxf(amax, fabsf(v[j]));
  float s;
  if (e4m3_scale)
    s = e4m3(fmaxf(amax, kFp4Max * ldexpf(1.0f, kE4m3ScaleFloorExp)) / kFp4Max);
  else
    s = exp2f(ceilf(log2f(fmaxf(amax, kFp4Max * ldexpf(1.0f, kPow2ScaleFloorExp)) / kFp4Max)));
  for (uint32_t j = 0; j < block; j++) v[j] = e2m1(fminf(fmaxf(v[j] / s, -kFp4Max), kFp4Max)) * s;
}

__global__ void bf16_kernel(float* x, uint64_t n) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) x[i] = bf16_round(x[i]);
}

}  // namespace

extern "C" bool ops_fp4_round(float* x, uint64_t n, uint32_t block, bool e4m3_scale, void* stream) {
  if (!n || !block || n % block) return false;
  fp4_kernel<<<grid(n / block), kThreads, 0, st(stream)>>>(x, n / block, block, e4m3_scale);
  return ok();
}

extern "C" bool ops_fp8_round(float* x, uint64_t n, void* stream) {
  if (!n || n % kFp8Block) return false;
  fp8_kernel<<<grid(n / kFp8Block), kThreads, 0, st(stream)>>>(x, n / kFp8Block);
  return ok();
}

extern "C" bool ops_round_bf16(float* x, uint64_t n, void* stream) {
  if (!n) return false;
  bf16_kernel<<<grid(n), kThreads, 0, st(stream)>>>(x, n);
  return ok();
}

