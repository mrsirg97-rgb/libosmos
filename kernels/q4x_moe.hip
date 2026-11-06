// Qwen3.8-Flash-Next's MoE (include/osmos/gpu/q4x_ops.h): its routing, the experts' tables and inputs, their
// activation and their sum with the shared expert's, F32
#include <climits>

#include "block.h"

extern "C" {
#include "osmos/gpu/mm.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/q4x_ops.h"
}

namespace {

constexpr int kThreads = 256;
constexpr int kGroupThreads = 512; /* a step's grouping: a thread an expert (512 of them) */
constexpr uint32_t kMaxExperts = 1024; /* a token's probabilities in LDS */
constexpr uint32_t kMaxK = 32;
constexpr float kSumFloor = 6.103515625e-5f; /* llama.cpp's clamp of the picked weights' sum: f16's least normal */

inline unsigned grid(uint64_t n) { return (unsigned) ((n + kThreads - 1) / kThreads); }
inline hipStream_t st(void* s) { return (hipStream_t) s; }
inline bool ok() { return hipGetLastError() == hipSuccess; }

__device__ inline float silu(float x) { return x / (1.0f + expf(-x)); }

// a block a token: its softmax (ggml's: exp(x - max), times 1 / their sum), then the k greatest one at a time, each
// the block's argmax of the experts not yet taken (the lowest id among equals)
__global__ void route_kernel(const float* logits, uint32_t n_exp, uint32_t k, int32_t* ids, float* w, int32_t* rows) {
  __shared__ float p[kMaxExperts], lds[kThreads];
  __shared__ int ldi[kThreads];
  __shared__ bool taken[kMaxExperts];
  const uint32_t t = blockIdx.x;
  const float* l = logits + (uint64_t) t * n_exp;
  float m = -INFINITY;
  for (uint32_t i = threadIdx.x; i < n_exp; i += kThreads) m = fmaxf(m, l[i]);
  m = block_max<kThreads>(m, lds);
  float sum = 0;
  for (uint32_t i = threadIdx.x; i < n_exp; i += kThreads) sum += (p[i] = expf(l[i] - m)), taken[i] = false;
  const float inv = 1.0f / block_sum<kThreads>(sum, lds);
  for (uint32_t i = threadIdx.x; i < n_exp; i += kThreads) p[i] *= inv;
  __syncthreads();
  float pick[kMaxK], total = 0;
  for (uint32_t j = 0; j < k; j++) {
    float bv = -INFINITY;
    int bi = INT_MAX;
    for (uint32_t i = threadIdx.x; i < n_exp; i += kThreads)
      if (!taken[i] && (p[i] > bv || (p[i] == bv && (int) i < bi))) bv = p[i], bi = (int) i;
    const int best = block_argmax<kThreads>(&bv, bi, lds, ldi);
    if (!threadIdx.x) {
      taken[best] = true, pick[j] = bv, total += bv;
      ids[t * k + j] = best, rows[t * k + j] = (int32_t) t;
    }
    __syncthreads();
  }
  if (threadIdx.x) return;
  const float div = fmaxf(total, kSumFloor);
  for (uint32_t j = 0; j < k; j++) w[t * k + j] = pick[j] / div;
}

// a step's picks as mm_group's tables (q4x_expert_tiles): each expert's count, its first place and first entry (a
// scan), then a thread an expert lays its picks out in pick order and writes its entries - the same layout however it
// runs
__global__ __launch_bounds__(kGroupThreads) void tiles_kernel(const int32_t* ids, uint32_t n, uint32_t k,
                                                             uint32_t n_exp, uint32_t dim, uint32_t ff,
                                                             Q4xExpertTiles t, uint32_t bound) {
  __shared__ int count[kMaxExperts], start[kMaxExperts], first[kMaxExperts], live;
  constexpr int T = MM_TILE;
  for (uint32_t e = threadIdx.x; e < n_exp; e += kGroupThreads) count[e] = 0;
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < n; i += kGroupThreads) atomicAdd(&count[ids[i]], 1);
  __syncthreads();
  for (uint32_t e = threadIdx.x; e < n_exp; e += kGroupThreads) start[e] = count[e], first[e] = (count[e] + T - 1) / T;
  __syncthreads();
  for (uint32_t d = 1; d < n_exp; d <<= 1) {  // inclusive scans (Hillis-Steele), then each less its own
    int a[2][kMaxExperts / kGroupThreads];
    for (uint32_t e = threadIdx.x, j = 0; e < n_exp; e += kGroupThreads, j++)
      a[0][j] = e >= d ? start[e - d] : 0, a[1][j] = e >= d ? first[e - d] : 0;
    __syncthreads();
    for (uint32_t e = threadIdx.x, j = 0; e < n_exp; e += kGroupThreads, j++) start[e] += a[0][j], first[e] += a[1][j];
    __syncthreads();
  }
  if (!threadIdx.x) live = first[n_exp - 1];
  for (uint32_t e = threadIdx.x; e < n_exp; e += kGroupThreads)
    start[e] -= count[e], first[e] -= (count[e] + T - 1) / T;
  __syncthreads();
  for (uint32_t e = threadIdx.x; e < n_exp; e += kGroupThreads) {
    const int c = count[e];
    if (!c) continue;
    int q = start[e];
    for (uint32_t i = 0; i < n && q < start[e] + c; i++)
      if (ids[i] == (int32_t) e) t.where[i] = q, t.gidx[(first[e] + (q - start[e]) / T) * T + (q - start[e]) % T] = (int32_t) (i / k), q++;
    for (int part = 0; part * T < c; part++) {
      const int en = first[e] + part, p0 = start[e] + part * T, m = c - part * T < T ? c - part * T : T;
      t.cnt[en] = m;
      t.xg[en] = (uint32_t) en * T * dim, t.xd[en] = (uint32_t) en * T * ff;
      t.yg[en] = (uint32_t) p0 * ff, t.yd[en] = (uint32_t) p0 * dim;
      for (int j = 0; j < Q4X_GROUP_TENSORS; j++) t.ptr[j][en] = (const uint8_t*) t.base[j] + (uint64_t) e * t.bytes[j];
      for (int i = 0; i < T; i++) {
        if (i >= m) t.gidx[en * T + i] = -1;
        t.hidx[en * T + i] = i < m ? p0 + i : -1;
      }
    }
  }
  for (uint32_t en = (uint32_t) live + threadIdx.x; en < bound; en += kGroupThreads) {
    t.cnt[en] = 0;
    for (int i = 0; i < T; i++) t.gidx[en * T + i] = -1, t.hidx[en * T + i] = -1;
  }
}

__global__ void gather_kernel(const float* src, const int32_t* rows, uint32_t n, uint32_t len, float* dst) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n * len) return;
  dst[i] = src[(uint64_t) rows[i / len] * len + i % len];
}

__global__ void swiglu_kernel(const float* g, const float* u, uint64_t n, float* y) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) y[i] = silu(g[i]) * u[i];
}

__global__ void moe_sum_kernel(const float* y, const int32_t* where, const float* w, uint32_t n_tok, uint32_t k,
                               uint32_t dim, const float* shared, const float* shared_logit, float* out) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * dim) return;
  const uint64_t t = i / dim, d = i % dim;
  float sum = 0;
  for (uint32_t j = 0; j < k; j++) sum += y[(uint64_t) where[t * k + j] * dim + d] * w[t * k + j];
  out[i] = sum + shared[i] * sigmoid(shared_logit[t]);
}

}  // namespace

extern "C" bool q4x_route(const float* logits, uint32_t n_tok, uint32_t n_exp, uint32_t k, int32_t* ids, float* w,
                          int32_t* rows, void* stream) {
  if (!logits || !ids || !w || !rows || !n_tok || !n_exp || n_exp > kMaxExperts || !k || k > kMaxK || k > n_exp)
    return false;
  route_kernel<<<n_tok, kThreads, 0, st(stream)>>>(logits, n_exp, k, ids, w, rows);
  return ok();
}

extern "C" uint32_t q4x_expert_entries(uint32_t n, uint32_t n_exp) { return n / MM_TILE + (n < n_exp ? n : n_exp); }

extern "C" bool q4x_expert_tiles(const int32_t* ids, uint32_t n, uint32_t k, uint32_t n_exp, uint32_t dim, uint32_t ff,
                                 const Q4xExpertTiles* t, void* stream) {
  if (!ids || !n || !k || n % k || !n_exp || n_exp > kMaxExperts || !dim || !ff || !t || !t->cnt || !t->gidx ||
      !t->hidx || !t->xg || !t->yg || !t->xd || !t->yd || !t->where)
    return false;
  for (int j = 0; j < Q4X_GROUP_TENSORS; j++)
    if (!t->base[j] || !t->bytes[j] || !t->ptr[j]) return false;
  tiles_kernel<<<1, kGroupThreads, 0, st(stream)>>>(ids, n, k, n_exp, dim, ff, *t, q4x_expert_entries(n, n_exp));
  return ok();
}

extern "C" bool q4x_gather(const float* src, const int32_t* rows, uint32_t n, uint32_t len, float* dst, void* stream) {
  if (!src || !rows || !dst || !n || !len) return false;
  gather_kernel<<<grid((uint64_t) n * len), kThreads, 0, st(stream)>>>(src, rows, n, len, dst);
  return ok();
}

extern "C" bool q4x_swiglu(const float* g, const float* u, uint64_t n, float* y, void* stream) {
  if (!g || !u || !y || !n) return false;
  swiglu_kernel<<<grid(n), kThreads, 0, st(stream)>>>(g, u, n, y);
  return ok();
}

extern "C" bool q4x_moe_sum(const float* y, const int32_t* where, const float* w, uint32_t n_tok, uint32_t k,
                            uint32_t dim, const float* shared, const float* shared_logit, float* out, void* stream) {
  if (!y || !where || !w || !shared || !shared_logit || !out || !n_tok || !k || !dim) return false;
  moe_sum_kernel<<<grid((uint64_t) n_tok * dim), kThreads, 0, st(stream)>>>(y, where, w, n_tok, k, dim, shared,
                                                                             shared_logit, out);
  return ok();
}
