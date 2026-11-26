// route_tok (include/osmos/gpu/route.h): a decode's routed picks as the expert matvecs' tables on the GPU - any
// family's (moved from hy4.hip: the same kernel, the same bits)
#include <hip/hip_runtime.h>

#include "launch.h"

using namespace launches;

extern "C" {
#include "osmos/gpu/route.h"
}

namespace {

// route_tok: one block. the picks counted by expert, the counts' and entries' prefix sums, then the tokens a group at
// a time (each thread a pick): a pick's row its expert's start, the group's earlier tokens that picked it, the rows
// placed before the group - experts ascending, an expert's tokens ascending, the host's order
constexpr int kRouteThreads = 1024;
static_assert(kRouteThreads == ROUTE_MAX_EXPERTS, "the prefix sums: a thread an expert");

__global__ __launch_bounds__(kRouteThreads) void route_kernel(const int32_t* idx, const float* wtok, uint32_t n,
                                                              uint32_t k, uint32_t E, uint32_t chunk,
                                                              const void* const* tab,
                                                              int32_t* gidx, int32_t* sidx, float* wsel, int32_t* cnt,
                                                              int32_t* xrow, const void** wptr, int32_t* fault) {
  __shared__ int32_t count[ROUTE_MAX_EXPERTS], start[ROUTE_MAX_EXPERTS], ent[ROUTE_MAX_EXPERTS];
  __shared__ int32_t run[ROUTE_MAX_EXPERTS], group[kRouteThreads];
  __shared__ int32_t bad, total;
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
  // exclusive prefix sums of the rows and the entries (each expert's rows in chunks), Hillis-Steele over the experts
  const int32_t c_own = tid < E ? count[tid] : 0, e_own = (c_own + (int32_t) chunk - 1) / (int32_t) chunk;
  start[tid] = c_own, ent[tid] = e_own;
  __syncthreads();
  for (uint32_t d = 1; d < kRouteThreads; d <<= 1) {
    const int32_t a = tid >= d ? start[tid - d] : 0, b = tid >= d ? ent[tid - d] : 0;
    __syncthreads();
    start[tid] += a, ent[tid] += b;
    __syncthreads();
  }
  start[tid] -= c_own, ent[tid] -= e_own;
  if (tid == kRouteThreads - 1) total = ent[tid] + e_own;
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
  for (uint32_t e = tid; e < E; e += kRouteThreads)
    for (int32_t j = 0, c = count[e]; j * (int32_t) chunk < c; j++) {
      const uint32_t i = (uint32_t) (ent[e] + j);
      cnt[i] = min((int32_t) chunk, c - j * (int32_t) chunk), xrow[i] = start[e] + j * (int32_t) chunk;
      for (int q = 0; q < 3; q++) wptr[(uint64_t) q * nk + i] = tab[(uint64_t) q * E + e];
    }
  for (uint32_t i = (uint32_t) total + tid; i < nk; i += kRouteThreads) cnt[i] = 0;
}

}  // namespace

extern "C" bool route_tok(const int32_t* idx, const float* wtok, uint32_t n_tok, uint32_t k, uint32_t n_exp,
                          uint32_t chunk, const void* const* tab, int32_t* gidx, int32_t* sidx, float* wsel,
                          int32_t* cnt, int32_t* xrow, const void** wptr, int32_t* fault, void* stream) {
  if (!k || k > ROUTE_MAX_K || !n_exp || n_exp > ROUTE_MAX_EXPERTS || !chunk || !n_tok || !tab || !idx ||
      !wtok || !gidx || !sidx || !wsel || !cnt || !xrow || !wptr || !fault)
    return false;
  route_kernel<<<1, kRouteThreads, 0, st(stream)>>>(idx, wtok, n_tok, k, n_exp, chunk, tab, gidx, sidx, wsel, cnt, xrow,
                                                    wptr, fault);
  return ok();
}

