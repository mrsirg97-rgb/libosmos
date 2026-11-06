// a block's shared reductions and activations, one definition for every kernel file (ops.hip, q4x.hip)
#pragma once
#include <hip/hip_runtime.h>

#include <cmath>

// the sum of each thread's v over a block of NT threads, by a fixed tree (every thread gets it)
template <int NT>
__device__ inline float block_sum(float v, float* lds) {
  lds[threadIdx.x] = v;
  __syncthreads();
  for (int s = NT / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) lds[threadIdx.x] += lds[threadIdx.x + s];
    __syncthreads();
  }
  const float r = lds[0];
  __syncthreads();
  return r;
}

__device__ inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
