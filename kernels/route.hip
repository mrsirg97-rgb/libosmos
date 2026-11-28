// route_tok (include/osmos/gpu/route.h): a decode's routed picks as the expert matvecs' tables on the GPU - any
// family's (moved from hy4.hip: the same kernel, the same bits)
#include <hip/hip_runtime.h>

#include "launch.h"

using namespace launches;

extern "C" {
#include "osmos/gpu/route.h"
}

namespace {

// route_tok: one block. the picks counted by expert, the rows' prefix sum (an expert's rows from its start: experts
// ascending, an expert's tokens ascending, the host's order) and the entries' by count (an expert's full chunks one
// 32-bit sum; its rest, a count under chunk, one of kBuckets 16-bit counters packed four to a word - each at most an
// expert's one, under 65,536: the entries of each count before an expert's), then the tokens a group at a time
// (each thread a pick): a pick's row its expert's start, the group's earlier tokens that picked it, the rows placed
// before the group. the entries by count, the most first, experts ascending within one (0.18.1: q8_experts' exact
// launches each a range of counts over the entries that can hold it - an entry of more than c picks among the first
// n x k / (c + 1)); an expert's chunks of `chunk` rows its full ones (count chunk) then its rest
constexpr int kRouteThreads = 1024;
constexpr int kBuckets = ROUTE_MAX_CHUNK, kWords = kBuckets / 4;
static_assert(kRouteThreads == ROUTE_MAX_EXPERTS, "the prefix sums: a thread an expert");
static_assert(kBuckets % 4 == 0, "four 16-bit counters a word");

__device__ inline uint32_t field(const uint64_t* w, int v) {  // bucket v's counter (1 .. kBuckets)
  return (uint32_t) ((w[(v - 1) / 4] >> (((v - 1) % 4) * 16)) & 0xffff);
}

__global__ __launch_bounds__(kRouteThreads) void route_kernel(const int32_t* idx, const float* wtok, uint32_t n,
                                                              uint32_t k, uint32_t E, uint32_t chunk,
                                                              const void* const* tab,
                                                              int32_t* gidx, int32_t* sidx, float* wsel, int32_t* cnt,
                                                              int32_t* xrow, const void** wptr, int32_t* fault) {
  constexpr int kWaves = kRouteThreads / 32;
  __shared__ int32_t count[ROUTE_MAX_EXPERTS], start[ROUTE_MAX_EXPERTS];
  __shared__ int32_t run[ROUTE_MAX_EXPERTS], group[kRouteThreads];
  __shared__ int32_t w_start[kWaves], w_fp[kWaves];
  __shared__ uint64_t w_pk[kWords][kWaves];
  __shared__ int32_t bad, total, from[kBuckets + 2];
  const uint32_t tid = threadIdx.x, nk = n * k;
  for (uint32_t e = tid; e < ROUTE_MAX_EXPERTS; e += kRouteThreads) count[e] = 0, run[e] = 0;
  if (tid == 0) bad = 0;
  __syncthreads();
  for (uint32_t m = tid; m < nk; m += kRouteThreads) {
    const int32_t e = idx[m];
    bool b = e < 0 || (uint32_t) e >= E;
    for (uint32_t s = m - m % k; !b && s < m; s++) b = idx[s] == e;
    if (b)
      bad = 1;
    else
      atomicAdd(&count[e], 1);
  }
  __syncthreads();
  if (bad) {
    if (tid == 0) *fault = 1;
    return;
  }
  // an expert's entries: c / chunk full ones (count chunk), then one of the rest (c % chunk) if any - their counts'
  // counters, and the rows, prefix-summed together (intra-wave shuffle scan + 32-wave scan: fits 48 KB static LDS)
  const int32_t c_own = tid < E ? count[tid] : 0, full = c_own / (int32_t) chunk, rest = c_own % (int32_t) chunk;
  uint64_t own[kWords];
  for (int w = 0; w < kWords; w++) own[w] = 0;
  if (rest) own[(rest - 1) / 4] += (uint64_t) 1 << (((rest - 1) % 4) * 16);
  const uint32_t lane = tid & 31u, wave = tid >> 5;
  int32_t s_inc = c_own, f_inc = full;
  uint64_t p_inc[kWords];
  for (int w = 0; w < kWords; w++) p_inc[w] = own[w];
  for (uint32_t d = 1; d < 32; d <<= 1) {
    const int32_t sa = __shfl_up(s_inc, d, 32), fa = __shfl_up(f_inc, d, 32);
    if (lane >= d) s_inc += sa, f_inc += fa;
    for (int w = 0; w < kWords; w++) {
      const uint32_t lo = __shfl_up((uint32_t) p_inc[w], d, 32);
      const uint32_t hi = __shfl_up((uint32_t) (p_inc[w] >> 32), d, 32);
      if (lane >= d) p_inc[w] += ((uint64_t) hi << 32) | lo;  // (fields never carry: each under 65,536)
    }
  }
  if (lane == 31) {
    w_start[wave] = s_inc, w_fp[wave] = f_inc;
    for (int w = 0; w < kWords; w++) w_pk[w][wave] = p_inc[w];
  }
  __syncthreads();
  if (wave == 0) {
    int32_t ws = w_start[lane], wf = w_fp[lane];
    uint64_t wp[kWords];
    for (int w = 0; w < kWords; w++) wp[w] = w_pk[w][lane];
    for (uint32_t d = 1; d < 32; d <<= 1) {
      const int32_t sa = __shfl_up(ws, d, 32), fa = __shfl_up(wf, d, 32);
      if (lane >= d) ws += sa, wf += fa;
      for (int w = 0; w < kWords; w++) {
        const uint32_t lo = __shfl_up((uint32_t) wp[w], d, 32);
        const uint32_t hi = __shfl_up((uint32_t) (wp[w] >> 32), d, 32);
        if (lane >= d) wp[w] += ((uint64_t) hi << 32) | lo;
      }
    }
    w_start[lane] = ws, w_fp[lane] = wf;
    for (int w = 0; w < kWords; w++) w_pk[w][lane] = wp[w];
    if (lane == 31) {  // each count's first entry, the most first: from[c] .. from[c] + its total
      int32_t at = wf;  // the full chunks first (count chunk), then each count under it
      from[chunk] = 0;
      for (int v = (int) chunk - 1; v >= 1; v--) from[v] = at, at += (int32_t) field(wp, v);
      total = at;
    }
  }
  __syncthreads();
  const int32_t ws_pre = wave ? w_start[wave - 1] : 0;
  const int32_t wf_pre = wave ? w_fp[wave - 1] : 0;
  start[tid] = s_inc + ws_pre - c_own;
  uint64_t before[kWords];  // the entries of each count under chunk before this expert's, and its full chunks' first
  for (int w = 0; w < kWords; w++) before[w] = p_inc[w] + (wave ? w_pk[w][wave - 1] : 0) - own[w];
  const int32_t full_at = f_inc + wf_pre - full;
  __syncthreads();
  const uint32_t G = kRouteThreads / k;  // tokens a group
  for (uint32_t t0 = 0; t0 < n; t0 += G) {
    const uint32_t m = t0 * k + tid, t = t0 + tid / k;
    const bool mine = tid < G * k && t < n;
    group[tid] = mine ? idx[m] : -1;
    __syncthreads();
    if (mine) {
      const int32_t e = group[tid];
      int32_t rank = 0;
      for (uint32_t q = 0; q < (tid / k) * k; q++) rank += group[q] == e;
      const int32_t r = start[e] + run[e] + rank;
      gidx[r] = (int32_t) t, sidx[m] = r, wsel[r] = wtok[m];
    }
    __syncthreads();
    if (mine) atomicAdd(&run[group[tid]], 1);
    __syncthreads();
  }
  if (tid < E && c_own) {
    const uint32_t e = tid;
    for (int32_t j = 0; j <= full; j++) {
      const int32_t c = j < full ? (int32_t) chunk : rest;
      if (!c) break;
      const uint32_t i = (uint32_t) (j < full ? full_at + j : from[c] + (int32_t) field(before, c));
      cnt[i] = c, xrow[i] = start[e] + j * (int32_t) chunk;
      for (int q = 0; q < 3; q++) wptr[(uint64_t) q * nk + i] = tab[(uint64_t) q * E + e];
    }
  }
  for (uint32_t i = (uint32_t) total + tid; i < nk; i += kRouteThreads) cnt[i] = 0;
}

}  // namespace

extern "C" bool route_tok(const int32_t* idx, const float* wtok, uint32_t n_tok, uint32_t k, uint32_t n_exp,
                          uint32_t chunk, const void* const* tab, int32_t* gidx, int32_t* sidx, float* wsel,
                          int32_t* cnt, int32_t* xrow, const void** wptr, int32_t* fault, void* stream) {
  if (!k || k > ROUTE_MAX_K || !n_exp || n_exp > ROUTE_MAX_EXPERTS || !chunk || chunk > ROUTE_MAX_CHUNK || !n_tok || !tab || !idx ||
      !wtok || !gidx || !sidx || !wsel || !cnt || !xrow || !wptr || !fault)
    return false;
  route_kernel<<<1, kRouteThreads, 0, st(stream)>>>(idx, wtok, n_tok, k, n_exp, chunk, tab, gidx, sidx, wsel, cnt, xrow,
                                                    wptr, fault);
  return ok();
}

