// Qwen3.8-Flash-Next's MoE (include/osmos/gpu/q4x_ops.h): its routing, the experts' tables and inputs, their
// activation and their sum with the shared expert's, F32
#include <climits>

#include "block.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

namespace {

constexpr int kThreads = 256;
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

__global__ void table_kernel(const int32_t* ids, uint32_t n, const uint8_t* base, uint64_t bytes, const void** ptr,
                             int32_t* cnt, int32_t* row) {
  const uint32_t e = blockIdx.x * kThreads + threadIdx.x;
  if (e >= n) return;
  ptr[e] = base + (uint64_t) ids[e] * bytes, cnt[e] = 1, row[e] = (int32_t) e;
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

__global__ void moe_sum_kernel(const float* y, const float* w, uint32_t n_tok, uint32_t k, uint32_t dim,
                               const float* shared, const float* shared_logit, float* out) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * dim) return;
  const uint64_t t = i / dim, d = i % dim;
  float sum = 0;
  for (uint32_t j = 0; j < k; j++) sum += y[(t * k + j) * dim + d] * w[t * k + j];
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

extern "C" bool q4x_expert_table(const int32_t* ids, uint32_t n, const void* base, uint64_t bytes, const void** ptr,
                                 int32_t* cnt, int32_t* row, void* stream) {
  if (!ids || !base || !bytes || !ptr || !cnt || !row || !n) return false;
  table_kernel<<<grid(n), kThreads, 0, st(stream)>>>(ids, n, (const uint8_t*) base, bytes, ptr, cnt, row);
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

extern "C" bool q4x_moe_sum(const float* y, const float* w, uint32_t n_tok, uint32_t k, uint32_t dim,
                            const float* shared, const float* shared_logit, float* out, void* stream) {
  if (!y || !w || !shared || !shared_logit || !out || !n_tok || !k || !dim) return false;
  moe_sum_kernel<<<grid((uint64_t) n_tok * dim), kThreads, 0, st(stream)>>>(y, w, n_tok, k, dim, shared,
                                                                             shared_logit, out);
  return ok();
}
