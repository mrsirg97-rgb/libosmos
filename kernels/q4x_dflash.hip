// DFlash2's drafter's own steps (include/osmos/gpu/q4x_ops.h's q4x_df_*), F32: its rope, its grouped dynamic
// convolution, its context ring, its windowed attention, its top-k and its candidate selector
#pragma clang fp contract(off) /* each sum in the order its contract writes: the references' bits */
#include <hip/hip_fp16.h>

#include "block.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

namespace {

constexpr int kThreads = 256;
constexpr uint32_t kHalf = Q4X_DF_HEAD / 2;
constexpr uint32_t kQ8Block = 32, kQ8Bytes = 34; /* ggml's Q8_0: an f16 scale, then 32 int8 */

inline unsigned grid(uint64_t n) { return (unsigned) ((n + kThreads - 1) / kThreads); }
inline hipStream_t st(void* s) { return (hipStream_t) s; }
inline bool ok() { return hipGetLastError() == hipSuccess; }

__global__ void rope_kernel(float* x, uint32_t n_tok, uint32_t heads, uint32_t stride, const int32_t* pos, float base) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * heads * kHalf) return;
  const uint32_t d = (uint32_t) (i % kHalf), h = (uint32_t) (i / kHalf % heads), t = (uint32_t) (i / kHalf / heads);
  float* v = x + (uint64_t) t * stride + (uint64_t) h * Q4X_DF_HEAD;
  const float a = (float) pos[t] * (float) (1.0 / pow((double) base, 2.0 * d / Q4X_DF_HEAD)); /* its frequency rounded once */
  const float c = cosf(a), s = sinf(a), x1 = v[d], x2 = v[d + kHalf];
  v[d] = x1 * c - x2 * s, v[d + kHalf] = x2 * c + x1 * s;
}

__global__ void conv_kernel(const float* x, uint32_t n_tok, uint32_t dim, uint32_t group, const float* base,
                            const float* dyn, uint32_t dyn_stride, float* y) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * dim) return;
  const uint32_t c = (uint32_t) (i % dim), t = (uint32_t) (i / dim), groups = dim / group;
  float z = 0.0f;
  for (uint32_t off = 0; off < 2; off++) {
    const float v = t >= off ? x[(uint64_t) (t - off) * dim + c] : 0.0f;
    z = z + base[off * dim + c] * v;
    z = z + dyn[(uint64_t) t * dyn_stride + off * groups + c / group] * v;
  }
  y[i] = z;
}

__global__ void ring_kernel(float* ring, uint32_t ring_rows, const float* x, uint32_t n, uint32_t row, int32_t pos0) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n * row) return;
  const uint64_t t = i / row, d = i % row;
  if (n > ring_rows && t < n - ring_rows) return; /* its slot a newer row's: written once, by the newest */
  ring[(uint64_t) (((uint64_t) pos0 + t) % ring_rows) * row + d] = x[i];
}

// a block a (query, head): its visible keys' scores into the LDS (the ring's then the block's), their max and sum by
// block_sum's tree order, then each of its dims the weighted sum over the keys in order
__global__ void attn_kernel(const float* q, uint32_t n_q, int32_t pos0, const float* kb, const float* vb, const float* kr,
                            const float* vr, uint32_t ring_rows, int32_t ctx_from, uint32_t heads, uint32_t kv_heads,
                            uint32_t window, float* out) {
  extern __shared__ float lds[];
  __shared__ float qv[Q4X_DF_HEAD], red[kThreads];
  const uint32_t qi = blockIdx.x, h = blockIdx.y, g = h / (heads / kv_heads), row = kv_heads * Q4X_DF_HEAD;
  const int32_t qp = pos0 + (int32_t) qi;
  const int32_t lo = ctx_from > qp - (int32_t) window + 1 ? ctx_from : qp - (int32_t) window + 1;
  const uint32_t nctx = pos0 > lo ? (uint32_t) (pos0 - lo) : 0, nk = nctx + n_q;
  const float* qh = q + ((uint64_t) qi * heads + h) * Q4X_DF_HEAD;
  for (uint32_t d = threadIdx.x; d < Q4X_DF_HEAD; d += kThreads) qv[d] = qh[d];
  __syncthreads();
  const float scale = 1.0f / sqrtf((float) Q4X_DF_HEAD);
  float mx = -INFINITY;
  for (uint32_t j = threadIdx.x; j < nk; j += kThreads) {
    const float* k = j < nctx ? kr + (uint64_t) ((uint64_t) (lo + (int32_t) j) % ring_rows) * row + g * Q4X_DF_HEAD
                              : kb + (uint64_t) (j - nctx) * row + g * Q4X_DF_HEAD;
    float s = 0.0f;
    for (uint32_t d = 0; d < Q4X_DF_HEAD; d++) s = s + qv[d] * k[d];
    lds[j] = s * scale;
    mx = fmaxf(mx, lds[j]);
  }
  mx = block_max<kThreads>(mx, red);
  float sum = 0.0f;
  for (uint32_t j = threadIdx.x; j < nk; j += kThreads) {
    lds[j] = expf(lds[j] - mx);
    sum = sum + lds[j];
  }
  sum = block_sum<kThreads>(sum, red);
  __syncthreads();
  for (uint32_t d = threadIdx.x; d < Q4X_DF_HEAD; d += kThreads) {
    float o = 0.0f;
    for (uint32_t j = 0; j < nk; j++) {
      const float* v = j < nctx ? vr + (uint64_t) ((uint64_t) (lo + (int32_t) j) % ring_rows) * row + g * Q4X_DF_HEAD
                                : vb + (uint64_t) (j - nctx) * row + g * Q4X_DF_HEAD;
      o = o + lds[j] / sum * v[d];
    }
    out[((uint64_t) qi * heads + h) * Q4X_DF_HEAD + d] = o;
  }
}

__device__ inline bool better(float a, int32_t ia, float b, int32_t ib) { return a > b || (a == b && ia < ib); }

// a block a row: each thread its strided values' k best (insertion), then thread 0 the block's k best of those
__global__ void topk_kernel(const float* logits, uint32_t vocab, uint32_t k, float* val, int32_t* idx) {
  __shared__ float sv[kThreads * Q4X_DF_TOPK];
  __shared__ int32_t si[kThreads * Q4X_DF_TOPK];
  const float* row = logits + (uint64_t) blockIdx.x * vocab;
  float v[Q4X_DF_TOPK];
  int32_t id[Q4X_DF_TOPK];
  for (uint32_t j = 0; j < k; j++) v[j] = -INFINITY, id[j] = INT32_MAX;
  for (uint32_t i = threadIdx.x; i < vocab; i += kThreads) {
    const float x = row[i];
    if (!better(x, (int32_t) i, v[k - 1], id[k - 1])) continue; /* a NaN never better: its compares false */
    uint32_t j = k - 1;
    for (; j > 0 && better(x, (int32_t) i, v[j - 1], id[j - 1]); j--) v[j] = v[j - 1], id[j] = id[j - 1];
    v[j] = x, id[j] = (int32_t) i;
  }
  for (uint32_t j = 0; j < k; j++) sv[threadIdx.x * k + j] = v[j], si[threadIdx.x * k + j] = id[j];
  __syncthreads();
  if (threadIdx.x) return;
  for (uint32_t j = 0; j < k; j++) v[j] = -INFINITY, id[j] = INT32_MAX;
  for (uint32_t c = 0; c < kThreads * k; c++) {
    const float x = sv[c];
    const int32_t i = si[c];
    if (!better(x, i, v[k - 1], id[k - 1])) continue;
    uint32_t j = k - 1;
    for (; j > 0 && better(x, i, v[j - 1], id[j - 1]); j--) v[j] = v[j - 1], id[j] = id[j - 1];
    v[j] = x, id[j] = i;
  }
  for (uint32_t j = 0; j < k; j++) val[blockIdx.x * k + j] = v[j], idx[blockIdx.x * k + j] = id[j];
}

__device__ inline float q8(const uint8_t* rows, uint64_t row_bytes, int32_t t, uint32_t r) {
  const uint8_t* b = rows + (uint64_t) t * row_bytes + (uint64_t) (r / kQ8Block) * kQ8Bytes;
  return __half2float(*(const __half*) b) * (float) (int8_t) b[2 + r % kQ8Block];
}

// one block: a thread a candidate, its score's sum over the rank in order; thread 0 the row's pick
__global__ void select_kernel(const float* val, const int32_t* idx, uint32_t rows, uint32_t k, const float* hp,
                              uint32_t rank, const uint8_t* pred, const uint8_t* succ, int32_t anchor, int32_t* path) {
  __shared__ float score[Q4X_DF_TOPK];
  __shared__ int32_t prev;
  const uint64_t rb = (uint64_t) rank / kQ8Block * kQ8Bytes;
  if (threadIdx.x == 0) prev = anchor;
  __syncthreads();
  for (uint32_t j = 0; j < rows; j++) {
    const uint32_t c = threadIdx.x;
    if (c < k) {
      float s = 0.0f;
      for (uint32_t r = 0; r < rank; r++) s = s + q8(pred, rb, prev, r) * hp[(uint64_t) j * rank + r] * q8(succ, rb, idx[j * k + c], r);
      score[c] = val[j * k + c] + s;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t best = 0;
      for (uint32_t i = 1; i < k; i++)
        if (score[i] > score[best]) best = i;
      prev = path[j] = idx[j * k + best];
    }
    __syncthreads();
  }
}

}  // namespace

extern "C" bool q4x_df_rope(float* x, uint32_t n_tok, uint32_t heads, uint32_t stride, const int32_t* pos, float base,
                            void* stream) {
  if (!x || !pos || !n_tok || !heads || stride < heads * Q4X_DF_HEAD || !(base > 0)) return false;
  rope_kernel<<<grid((uint64_t) n_tok * heads * kHalf), kThreads, 0, st(stream)>>>(x, n_tok, heads, stride, pos, base);
  return ok();
}

extern "C" bool q4x_df_conv(const float* x, uint32_t n_tok, uint32_t dim, uint32_t group, const float* base,
                            const float* dyn, uint32_t dyn_stride, float* y, void* stream) {
  if (!x || !base || !dyn || !y || !n_tok || !group || dim % group || dyn_stride < 2 * dim / group) return false;
  conv_kernel<<<grid((uint64_t) n_tok * dim), kThreads, 0, st(stream)>>>(x, n_tok, dim, group, base, dyn, dyn_stride, y);
  return ok();
}

extern "C" bool q4x_df_ring(float* ring, uint32_t ring_rows, const float* x, uint32_t n, uint32_t row, int32_t pos0,
                            void* stream) {
  if (!ring || !x || !ring_rows || !row || pos0 < 0) return false;
  if (!n) return true;
  ring_kernel<<<grid((uint64_t) n * row), kThreads, 0, st(stream)>>>(ring, ring_rows, x, n, row, pos0);
  return ok();
}

extern "C" bool q4x_df_attn(const float* q, uint32_t n_q, int32_t pos0, const float* kb, const float* vb, const float* kr,
                            const float* vr, uint32_t ring_rows, int32_t ctx_from, uint32_t heads, uint32_t kv_heads,
                            uint32_t window, float* out, void* stream) {
  if (!q || !kb || !vb || !kr || !vr || !out || !n_q || !heads || !kv_heads || heads % kv_heads || !window ||
      window < n_q || ctx_from < 0 || ctx_from > pos0 || ctx_from < pos0 - (int32_t) ring_rows)
    return false;
  const size_t lds = ((size_t) ring_rows + n_q) * sizeof(float);
  attn_kernel<<<dim3(n_q, heads), kThreads, lds, st(stream)>>>(q, n_q, pos0, kb, vb, kr, vr, ring_rows, ctx_from, heads,
                                                                kv_heads, window, out);
  return ok();
}

extern "C" bool q4x_df_topk(const float* logits, uint32_t rows, uint32_t vocab, uint32_t k, float* val, int32_t* idx,
                            void* stream) {
  if (!logits || !val || !idx || !rows || !k || k > Q4X_DF_TOPK || vocab < k) return false;
  topk_kernel<<<rows, kThreads, 0, st(stream)>>>(logits, vocab, k, val, idx);
  return ok();
}

extern "C" bool q4x_df_select(const float* val, const int32_t* idx, uint32_t rows, uint32_t k, const float* hp,
                              uint32_t rank, const void* pred, const void* succ, int32_t anchor, int32_t* path,
                              void* stream) {
  if (!val || !idx || !hp || !pred || !succ || !path || !rows || !k || k > Q4X_DF_TOPK || !rank || rank % kQ8Block ||
      anchor < 0)
    return false;
  select_kernel<<<1, Q4X_DF_TOPK, 0, st(stream)>>>(val, idx, rows, k, hp, rank, (const uint8_t*) pred,
                                                   (const uint8_t*) succ, anchor, path);
  return ok();
}
