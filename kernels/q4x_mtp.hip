// Qwen3.8-Flash-Next's MTP head's own ops (include/osmos/gpu/q4x_ops.h): a draft's greatest token and its
// probability, the eh_proj input's join, F32
#include <climits>

#include "block.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

namespace {

constexpr int kTopThreads = 1024;  // one block a row: 248,320 logits, ~243 a thread
constexpr int kThreads = 256;

inline hipStream_t st(void* s) { return (hipStream_t) s; }
inline bool ok() { return hipGetLastError() == hipSuccess; }

// a thread's logits strided: the greatest (the lowest id among equals), the block's; then exp(x - max) summed a
// thread, then the block's tree
__global__ __launch_bounds__(kTopThreads) void top_kernel(const float* logits, uint32_t n, uint32_t stride,
                                                          int32_t* id, float* p) {
  __shared__ float lds[kTopThreads];
  const float* l = logits + (uint64_t) blockIdx.x * stride;
  __shared__ int ldi[kTopThreads];
  float bv = -INFINITY;
  int bi = INT_MAX;
  for (uint32_t i = threadIdx.x; i < n; i += kTopThreads)
    if (l[i] > bv) bv = l[i], bi = (int) i;
  const int best = block_argmax<kTopThreads>(&bv, bi, lds, ldi);
  if (!p) {
    if (!threadIdx.x) id[blockIdx.x] = best;
    return;
  }
  float sum = 0;
  for (uint32_t i = threadIdx.x; i < n; i += kTopThreads) sum += expf(l[i] - bv);
  sum = block_sum<kTopThreads>(sum, lds);
  if (!threadIdx.x) id[blockIdx.x] = best, p[blockIdx.x] = 1.0f / sum;
}

__global__ void join_kernel(const float* e, const float* h, uint32_t n_tok, uint32_t hc, uint32_t dim, float* out) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x, row = 2ull * dim;
  if (i >= (uint64_t) n_tok * hc * row) return;
  const uint64_t ts = i / row, d = i % row, t = ts / hc;  // ts: the token's stream
  out[i] = d < dim ? e[t * dim + d] : h[ts * dim + (d - dim)];
}

}  // namespace

extern "C" bool q4x_top(const float* logits, uint32_t n, uint32_t rows, uint32_t stride, int32_t* id, float* p,
                        void* stream) {
  if (!logits || !n || n > (uint32_t) INT_MAX || !rows || (rows > 1 && stride < n) || !id) return false;
  top_kernel<<<rows, kTopThreads, 0, st(stream)>>>(logits, n, stride, id, p);
  return ok();
}

extern "C" bool q4x_join(const float* e, const float* h, uint32_t n_tok, uint32_t hc, uint32_t dim, float* out,
                         void* stream) {
  if (!e || !h || !out || !n_tok || !hc || !dim) return false;
  const uint64_t n = (uint64_t) n_tok * hc * 2 * dim;
  join_kernel<<<(unsigned) ((n + kThreads - 1) / kThreads), kThreads, 0, st(stream)>>>(e, h, n_tok, hc, dim, out);
  return ok();
}
