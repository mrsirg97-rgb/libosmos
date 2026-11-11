// a block's shared reductions and activations, one definition for every kernel file (ops.hip, q4x.hip)
#pragma once
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

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

// one pick of a wave's top k (both families' routers): the wave's lanes hold a token's n keys in registers, lane l its
// R keys of ids l, l + 32, ... (v[r]: id l + 32 r); `taken` the lane's ids picked before (bit r). the greatest key
// not yet taken, the lowest id among equals - a scan in id order's pick - by shuffles, no barrier; its id returned,
// its key in *key, in every lane. a lane with none left offers kPickNone, which any id beats; the keys through
// pick_key (a NaN the least: it compares false both ways, and lanes that disagree on a pick each take their own id) -
// so the k picks are k ids below n, whatever the keys
constexpr int kPickNone = INT32_MAX;
__device__ inline float pick_key(float x) { return x == x ? x : -INFINITY; }
template <int R>
__device__ __attribute__((always_inline)) inline int wave_pick(const float* v, uint32_t n, uint32_t* taken,
                                                               float* key) {
  const uint32_t lane = threadIdx.x % 32;
  float bv = -INFINITY;
  int bi = kPickNone;
#pragma unroll
  for (int r = 0; r < R; r++) { /* ascending ids: a strict > keeps the lowest among equals */
    const uint32_t e = lane + 32u * (uint32_t) r;
    if (32u * (uint32_t) r >= n) break; /* (no lane has an id this far: the wave leaves together) */
    if (e < n && !(*taken >> r & 1) && (bi == kPickNone || v[r] > bv)) bv = v[r], bi = (int) e;
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor(bv, o, 32);
    const int oi = __shfl_xor(bi, o, 32);
    if (oi != kPickNone && (bi == kPickNone || ov > bv || (ov == bv && oi < bi))) bv = ov, bi = oi;
  }
  if (bi != kPickNone && (uint32_t) bi % 32 == lane) *taken |= 1u << ((uint32_t) bi / 32);
  *key = bv;
  return bi;
}

__device__ inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ inline float silu(float x) { return x / (1.0f + expf(-x)); }
