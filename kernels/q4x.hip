// Qwen3.8-Flash-Next's own steps (include/osmos/gpu/q4x_ops.h), F32
#include "block.h"
#include "x16side.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

namespace {

constexpr int kThreads = 256;
constexpr uint32_t kMaxHc = 8;

inline unsigned grid(uint64_t n) { return (unsigned) ((n + kThreads - 1) / kThreads); }
inline hipStream_t st(void* s) { return (hipStream_t) s; }
inline bool ok() { return hipGetLastError() == hipSuccess; }

__device__ inline float silu(float x) { return x / (1.0f + expf(-x)); }

// a block a (stream, token): its sum of squares by block_sum's tree, then each value normed and weighted
__global__ void hc_norm_kernel(const float* h, const float* w, uint32_t hc, uint32_t dim, float eps, float* xn,
                               Q4xX16 side) {
  __shared__ float lds[kThreads];
  const uint32_t s = blockIdx.x, t = blockIdx.y;
  const float* x = h + ((uint64_t) t * hc + s) * dim;
  float* y = xn + ((uint64_t) t * hc + s) * dim;
  float ss = 0;
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) ss += x[d] * x[d];
  const float r = 1.0f / sqrtf(block_sum<kThreads>(ss, lds) / (float) dim + eps);
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) {
    y[d] = x[d] * r * w[(uint64_t) s * dim + d];
    x16_side(side, ((uint64_t) t * hc + s) * dim + d, y[d]);
  }
}

__global__ void scale_silu_kernel(float* x, uint64_t n, float scale, Q4xX16 side) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) x[i] = silu(x[i] * scale), x16_side(side, i, x[i]);
}

__global__ void gated_mean_kernel(const float* xn, const float* gate, uint32_t n_tok, uint32_t hc, uint32_t dim,
                                  float* out, Q4xX16 side) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * dim) return;
  const uint64_t t = i / dim, d = i % dim;
  float sum = 0;
  for (uint32_t s = 0; s < hc; s++) {
    const uint64_t at = (t * hc + s) * dim + d;
    sum += xn[at] * sigmoid(gate[at]);
  }
  out[i] = sum / (float) hc;
  x16_side(side, i, out[i]);
}

__global__ void combine_kernel(float* h, const float* out, const float* inject, uint32_t n_tok, uint32_t hc,
                               uint32_t dim, float inject_scale) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * hc * dim) return;
  const uint64_t d = i % dim, ts = i / dim, t = ts / hc;
  h[i] += 2.0f * sigmoid(inject[ts] * inject_scale) * out[t * dim + d];
}

bool shape(uint32_t n_tok, uint32_t hc, uint32_t dim) { return n_tok && hc && hc <= kMaxHc && dim; }

}  // namespace

extern "C" bool q4x_hc_norm(const float* h, const float* w, uint32_t n_tok, uint32_t hc, uint32_t dim, float eps,
                            float* xn, const Q4xX16* side, void* stream) {
  if (!shape(n_tok, hc, dim) || !h || !w || !xn) return false;
  hc_norm_kernel<<<dim3(hc, n_tok), kThreads, 0, st(stream)>>>(h, w, hc, dim, eps, xn, side_of(side));
  return ok();
}

extern "C" bool q4x_scale_silu(float* x, uint64_t n, float scale, const Q4xX16* side, void* stream) {
  if (!x || !n) return false;
  scale_silu_kernel<<<grid(n), kThreads, 0, st(stream)>>>(x, n, scale, side_of(side));
  return ok();
}

extern "C" bool q4x_hc_gated_mean(const float* xn, const float* gate, uint32_t n_tok, uint32_t hc, uint32_t dim,
                                  float* out, const Q4xX16* side, void* stream) {
  if (!shape(n_tok, hc, dim) || !xn || !gate || !out) return false;
  gated_mean_kernel<<<grid((uint64_t) n_tok * dim), kThreads, 0, st(stream)>>>(xn, gate, n_tok, hc, dim, out,
                                                                             side_of(side));
  return ok();
}

extern "C" bool q4x_hc_combine(float* h, const float* out, const float* inject, uint32_t n_tok, uint32_t hc,
                               uint32_t dim, float inject_scale, void* stream) {
  if (!shape(n_tok, hc, dim) || !h || !out || !inject) return false;
  combine_kernel<<<grid((uint64_t) n_tok * hc * dim), kThreads, 0, st(stream)>>>(h, out, inject, n_tok, hc, dim,
                                                                                  inject_scale);
  return ok();
}
