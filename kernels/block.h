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

// the greatest of each thread's v over a block of NT threads, by the same tree
template <int NT>
__device__ inline float block_max(float v, float* lds) {
  lds[threadIdx.x] = v;
  __syncthreads();
  for (int s = NT / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) lds[threadIdx.x] = fmaxf(lds[threadIdx.x], lds[threadIdx.x + s]);
    __syncthreads();
  }
  const float r = lds[0];
  __syncthreads();
  return r;
}

// the greatest of each thread's (v, i) over a block of NT threads - the lowest i among equal v - by the same tree
// (every thread gets it: its value into *v, its index returned)
template <int NT>
__device__ inline int block_argmax(float* v, int i, float* lds, int* ldi) {
  lds[threadIdx.x] = *v, ldi[threadIdx.x] = i;
  __syncthreads();
  for (int s = NT / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float a = lds[threadIdx.x], b = lds[threadIdx.x + s];
      const int ia = ldi[threadIdx.x], ib = ldi[threadIdx.x + s];
      if (b > a || (b == a && ib < ia)) lds[threadIdx.x] = b, ldi[threadIdx.x] = ib;
    }
    __syncthreads();
  }
  *v = lds[0];
  const int r = ldi[0];
  __syncthreads();
  return r;
}

__device__ inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ inline float silu(float x) { return x / (1.0f + expf(-x)); }
