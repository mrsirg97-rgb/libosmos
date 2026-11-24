#pragma once
// a step's routed picks laid out as mm_group's tables (q4x_expert_tiles, and ink2's moe_prep): one block of
// kGroupThreads threads - shared by every caller, so its tables are the same however it is launched
#include <cstdint>

#include "block.h"
extern "C" {
#include "osmos/gpu/mm.h"
#include "osmos/gpu/q4x_ops.h"
}

namespace moe_tiles {
constexpr int kGroupThreads = 512;     /* a thread an expert (512 of them) */
constexpr uint32_t kMaxExperts = 1024; /* an expert's counts and scans in LDS */
// a step's picks as mm_group's tables (q4x_expert_tiles): each expert's count, then scans of its places, its slots
// and its entries in each run, then a thread an expert lays its picks out in pick order and writes its parts - the
// same layout however it runs
constexpr int kPart = MM_TILE << (Q4X_TILE_RUNS - 1);  // an entry's most tokens: the last run's 8 tiles (128)
// a step's picks at most this many: each expert's thread places its own picks itself (a scan of the step's picks, cheap
// when they are few: a decode token's 10, a draft's tens), not place_kernel - one launch, not two. the same tables
// either way
constexpr uint32_t kPlaceInline = 256;
__device__ inline int run_of(int c) { return c <= 16 ? 0 : c <= 32 ? 1 : c <= 64 ? 2 : 3; }

__device__ inline void tiles_block(const int32_t* ids, uint32_t n, uint32_t k, uint32_t n_exp, uint32_t dim,
                                   uint32_t ff, const Q4xExpertTiles& t, const Q4xExpertBounds& bd) {
  constexpr int T = MM_TILE, A = 2 + Q4X_TILE_RUNS;  // scanned: places, slots, each run's entries
  __shared__ int count[kMaxExperts], sc[A][kMaxExperts], live[A];
  for (uint32_t e = threadIdx.x; e < n_exp; e += kGroupThreads) count[e] = 0;
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < n; i += kGroupThreads) atomicAdd(&count[ids[i]], 1);
  __syncthreads();
  auto own = [&](uint32_t e, int a) {  // expert e's own share of scan a
    const int c = count[e];
    if (a == 0) return c;
    if (a == 1) return (c + T - 1) / T * T;
    const int r = a - 2, rem = c % kPart;
    return (r == Q4X_TILE_RUNS - 1 ? c / kPart : 0) + (rem && run_of(rem) == r);
  };
  for (uint32_t e = threadIdx.x; e < n_exp; e += kGroupThreads)
    for (int a = 0; a < A; a++) sc[a][e] = own(e, a);
  __syncthreads();
  {  // inclusive scans, then each less its own: a wave's 32 by shuffles, then the earlier waves' and chunks' totals -
     // integers, so the sums Hillis-Steele's whatever their order (two barriers a chunk of kGroupThreads, not 14)
    constexpr int W = kGroupThreads / 32;
    __shared__ int wsum[A][W];
    const uint32_t lane = threadIdx.x % 32, w = threadIdx.x / 32;
    int carry[A] = {};
    for (uint32_t base = 0; base < n_exp; base += kGroupThreads) {
      const uint32_t e = base + threadIdx.x;
      int v[A];
      for (int a = 0; a < A; a++) v[a] = e < n_exp ? sc[a][e] : 0;
      for (uint32_t o = 1; o < 32; o <<= 1)
        for (int a = 0; a < A; a++) {
          const int u = __shfl_up(v[a], o, 32);
          if (lane >= o) v[a] += u;
        }
      if (lane == 31)
        for (int a = 0; a < A; a++) wsum[a][w] = v[a];
      __syncthreads();
      for (int a = 0; a < A; a++) {
        int pre = carry[a], all = 0;
        for (uint32_t x = 0; x < (uint32_t) W; x++) pre += x < w ? wsum[a][x] : 0, all += wsum[a][x];
        if (e < n_exp) sc[a][e] = v[a] + pre;
        carry[a] += all;
      }
      __syncthreads();  // (wsum the next chunk's)
    }
  }
  if (threadIdx.x < A) live[threadIdx.x] = sc[threadIdx.x][n_exp - 1];
  __syncthreads();  // the totals read before the last expert's thread makes its own exclusive below
  for (uint32_t e = threadIdx.x; e < n_exp; e += kGroupThreads)
    for (int a = 0; a < A; a++) sc[a][e] -= own(e, a);
  __syncthreads();
  for (uint32_t e = threadIdx.x; e < n_exp; e += kGroupThreads) {
    const int c = count[e];
    t.off[e] = sc[0][e], t.off[n_exp + e] = sc[1][e], t.off[2 * n_exp + e] = c;  // place_kernel's
    if (!c) continue;
    const int p0 = sc[0][e], s0 = sc[1][e];
    if (n <= kPlaceInline) {  // the picks in pick order, this thread's expert's
      int q = p0;
      for (uint32_t i = 0; i < n && q < p0 + c; i++)
        if (ids[i] == (int32_t) e) t.where[i] = q, t.gidx[s0 + q - p0] = (int32_t) (i / k), t.hidx[s0 + q - p0] = q, q++;
    }
    for (int s = s0 + c; s < s0 + own(e, 1); s++) t.gidx[s] = -1, t.hidx[s] = -1;
    int at[Q4X_TILE_RUNS];
    for (int r = 0; r < Q4X_TILE_RUNS; r++) at[r] = (int) bd.first[r] + sc[2 + r][e];
    for (int part = 0; part * kPart < c; part++) {
      const int m = c - part * kPart < kPart ? c - part * kPart : kPart, en = at[run_of(m)]++;
      const int slot = s0 + part * kPart, place = p0 + part * kPart;
      t.cnt[en] = m;
      t.xg[en] = (uint32_t) slot * dim, t.xd[en] = (uint32_t) slot * ff;
      t.yg[en] = (uint32_t) place * ff, t.yd[en] = (uint32_t) place * dim;
      for (int j = 0; j < Q4X_GROUP_TENSORS; j++)  // the table's address of the expert, or its place in the tensor
        t.ptr[j][en] = t.tab[j] ? t.tab[j][e] : (const uint8_t*) t.base[j] + (uint64_t) e * t.bytes[j];
    }
  }
  for (int r = 0; r < Q4X_TILE_RUNS; r++)
    for (uint32_t i = (uint32_t) live[2 + r] + threadIdx.x; i < bd.count[r]; i += kGroupThreads) t.cnt[bd.first[r] + i] = 0;
  for (uint32_t s = (uint32_t) live[1] + threadIdx.x; s < bd.slots; s += kGroupThreads) t.gidx[s] = -1, t.hidx[s] = -1;
}
}  // namespace moe_tiles
