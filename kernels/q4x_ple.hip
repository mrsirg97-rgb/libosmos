// Qwen3.8-Flash-Next's n-gram table (include/osmos/gpu/q4x_ops.h): its rows decoded, the gate, the residual's add
#include "block.h"
#include "quant.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

namespace {

constexpr int kThreads = 256;
constexpr float kDotFloor = 1e-6f; /* the gate's |dot| floor before its square root (llama.cpp's clamp) */
constexpr uint32_t kMaxHc = 8;

inline unsigned grid(uint64_t n) { return (unsigned) ((n + kThreads - 1) / kThreads); }
inline hipStream_t st(void* s) { return (hipStream_t) s; }
inline bool ok() { return hipGetLastError() == hipSuccess; }

// a thread a chunk (8 values) of a row: row i's chunk k decoded from its block
template <quant::Type T>
__global__ void rows_kernel(const uint8_t* table, uint64_t row_bytes, uint32_t row_values, const int32_t* rows,
                            uint32_t n, float* out) {
  using B = quant::Block<T>;
  const uint32_t per_row = row_values / quant::kChunk;
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n * per_row) return;
  const uint64_t r = i / per_row, c = i % per_row, v0 = c * quant::kChunk;
  const uint8_t* row = table + (uint64_t) rows[r] * row_bytes;
  quant::decode<T>(row + v0 / B::values * B::bytes, (int) (v0 % B::values / quant::kChunk),
                   out + r * row_values + v0);
}

// a block a (stream, token): the dot, its gate, the gated value
__global__ void ple_gate_kernel(const float* key, const float* query, const float* value, uint32_t hc, uint32_t dim,
                                float* gate, float* gated) {
  __shared__ float lds[kThreads];
  const uint32_t s = blockIdx.x, t = blockIdx.y;
  const uint64_t at = ((uint64_t) t * hc + s) * dim;
  float dot = 0;
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) dot += key[at + d] * query[at + d];
  const float x = block_sum<kThreads>(dot, lds) / sqrtf((float) dim);
  const float sg = (x > 0) - (x < 0), gv = sigmoid(sg * sqrtf(fmaxf(fabsf(x), kDotFloor)));
  if (!threadIdx.x) gate[(uint64_t) t * hc + s] = gv;
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) gated[at + d] = value[(uint64_t) t * dim + d] * gv;
}

__global__ void add2_kernel(float* h, const float* a, const float* b, uint64_t n) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) h[i] += a[i] + b[i];
}

template <quant::Type T>
bool rows_launch(const void* table, uint64_t row_bytes, uint32_t row_values, const int32_t* rows, uint32_t n,
                 float* out, hipStream_t s) {
  if (row_values % quant::Block<T>::values || row_bytes != row_values / quant::Block<T>::values * quant::Block<T>::bytes)
    return false;
  rows_kernel<T><<<grid((uint64_t) n * (row_values / quant::kChunk)), kThreads, 0, s>>>(
      (const uint8_t*) table, row_bytes, row_values, rows, n, out);
  return ok();
}

__global__ void streams_kernel(const float* x, uint32_t n_tok, uint32_t hc, uint32_t dim, float* h) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * hc * dim) return;
  h[i] = x[i / ((uint64_t) hc * dim) * dim + i % dim];
}

}  // namespace

extern "C" bool q4x_streams(const float* x, uint32_t n_tok, uint32_t hc, uint32_t dim, float* h, void* stream) {
  if (!x || !h || !n_tok || !hc || hc > kMaxHc || !dim) return false;
  streams_kernel<<<grid((uint64_t) n_tok * hc * dim), kThreads, 0, st(stream)>>>(x, n_tok, hc, dim, h);
  return ok();
}

extern "C" bool q4x_rows(uint32_t type, const void* table, uint64_t row_bytes, uint32_t row_values,
                         const int32_t* rows, uint32_t n, float* out, void* stream) {
  if (!table || !rows || !out || !n || !row_values) return false;
  switch (type) {
    case quant::kIQ4_NL: return rows_launch<quant::kIQ4_NL>(table, row_bytes, row_values, rows, n, out, st(stream));
    case quant::kQ8_0: return rows_launch<quant::kQ8_0>(table, row_bytes, row_values, rows, n, out, st(stream));
    case quant::kQ6_K: return rows_launch<quant::kQ6_K>(table, row_bytes, row_values, rows, n, out, st(stream));
    default: return false;
  }
}

extern "C" bool q4x_ple_gate(const float* key, const float* query, const float* value, uint32_t n_tok, uint32_t hc,
                             uint32_t dim, float* gate, float* gated, void* stream) {
  if (!key || !query || !value || !gate || !gated || !n_tok || !hc || hc > kMaxHc || !dim) return false;
  ple_gate_kernel<<<dim3(hc, n_tok), kThreads, 0, st(stream)>>>(key, query, value, hc, dim, gate, gated);
  return ok();
}

extern "C" bool q4x_add2(float* h, const float* a, const float* b, uint64_t n, void* stream) {
  if (!h || !a || !b || !n) return false;
  add2_kernel<<<grid(n), kThreads, 0, st(stream)>>>(h, a, b, n);
  return ok();
}
