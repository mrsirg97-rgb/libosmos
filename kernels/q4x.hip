// Qwen3.8-Flash-Next's own steps (include/osmos/gpu/q4x_ops.h), F32
#include "block.h"
#include "launch.h"
#include "x16side.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

namespace {

using launches::kThreads;
using launches::grid;
using launches::st;
using launches::ok;

constexpr uint32_t kMaxHc = 8;



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

// a block a (stream, token): combine_kernel's update of its values, then hc_norm_kernel's norm of them (each value
// the update's, its square summed in hc_norm_kernel's order)
__global__ void combine_norm_kernel(float* h, const float* out, const float* inject, uint32_t hc, uint32_t dim,
                                    float inject_scale, const float* w, float eps, float* xn, Q4xX16 side) {
  __shared__ float lds[kThreads];
  const uint32_t s = blockIdx.x, t = blockIdx.y;
  float* x = h + ((uint64_t) t * hc + s) * dim;
  float* y = xn + ((uint64_t) t * hc + s) * dim;
  const float g = 2.0f * sigmoid(inject[(uint64_t) t * hc + s] * inject_scale);
  float ss = 0;
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) {
    x[d] += g * out[(uint64_t) t * dim + d];
    ss += x[d] * x[d];
  }
  const float r = 1.0f / sqrtf(block_sum<kThreads>(ss, lds) / (float) dim + eps);
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) {
    y[d] = x[d] * r * w[(uint64_t) s * dim + d];
    x16_side(side, ((uint64_t) t * hc + s) * dim + d, y[d]);
  }
}

// a block a token: its hc streams in turn, each combine_norm_kernel's update (COMBINE) or hc_norm_kernel's alone, its
// norm and its f16 - their expressions, their trees, so their bits - its normed values kept in LDS; then the inject
// (an F32 hc x hc dim matrix: inj[t][q] = row q . the token's normed hc dim values) wave q's row q: lane l its 8-value
// chunks l, l + 32, ... of the token's row in order, 8 fmaf a chunk, the stream's chunks as each stream is done, then
// the tree - mv's sums (mv.h: any tile), so mv_strided's bits - without reading xn back (hc dim F32 a token)
template <bool COMBINE>
__global__ void hc_norm_inject_kernel(float* h, const float* out, const float* inject, uint32_t hc, uint32_t dim,
                                      float inject_scale, const float* w, float eps, float* xn, Q4xX16 side,
                                      const float* winj, float* inj) {
  extern __shared__ float ys[];  // a stream's normed values
  __shared__ float lds[kThreads];
  constexpr uint32_t kLanes = 32, kChunk = 8;
  const uint32_t t = blockIdx.x, wave = threadIdx.x / kLanes, lane = threadIdx.x % kLanes, cpd = dim / kChunk;
  const uint64_t hd = (uint64_t) hc * dim;
  float acc = 0;  // wave q < hc: row q's lane partial
  for (uint32_t s = 0; s < hc; s++) {
    float* x = h + ((uint64_t) t * hc + s) * dim;
    float* y = xn + ((uint64_t) t * hc + s) * dim;
    float ss = 0;
    if constexpr (COMBINE) {
      const float g = 2.0f * sigmoid(inject[(uint64_t) t * hc + s] * inject_scale);
      for (uint32_t d = threadIdx.x; d < dim; d += kThreads) {
        x[d] += g * out[(uint64_t) t * dim + d];
        ss += x[d] * x[d];
      }
    } else {
      for (uint32_t d = threadIdx.x; d < dim; d += kThreads) ss += x[d] * x[d];
    }
    const float r = 1.0f / sqrtf(block_sum<kThreads>(ss, lds) / (float) dim + eps);
    for (uint32_t d = threadIdx.x; d < dim; d += kThreads) {
      y[d] = x[d] * r * w[(uint64_t) s * dim + d];
      ys[d] = y[d];
      x16_side(side, ((uint64_t) t * hc + s) * dim + d, y[d]);
    }
    __syncthreads();  // the stream's values in LDS
    if (wave < hc) {
      const float* wr = winj + wave * hd;
      const uint32_t c0 = s * cpd, first = c0 + (lane + kLanes - c0 % kLanes) % kLanes;  // this lane's first in it
      for (uint32_t c = first; c < c0 + cpd; c += kLanes) {
        const float* vv = wr + (uint64_t) c * kChunk;
        const float* a = ys + (c - c0) * kChunk;
        acc = fmaf(vv[0], a[0], acc), acc = fmaf(vv[1], a[1], acc), acc = fmaf(vv[2], a[2], acc);
        acc = fmaf(vv[3], a[3], acc), acc = fmaf(vv[4], a[4], acc), acc = fmaf(vv[5], a[5], acc);
        acc = fmaf(vv[6], a[6], acc), acc = fmaf(vv[7], a[7], acc);
      }
    }
    __syncthreads();  // LDS free for the next stream; every thread past this stream's inject read
  }
  if (wave >= hc) return;
  for (uint32_t o = kLanes / 2; o > 0; o >>= 1) acc += __shfl_xor(acc, o, kLanes);
  if (lane == 0) inj[(uint64_t) t * hc + wave] = acc;
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

__global__ void tap_kernel(const float* h, uint32_t n_tok, uint32_t hc, uint32_t dim, float* taps, uint32_t stride) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * dim) return;
  const uint64_t t = i / dim, d = i % dim;
  float s = 0.0f;
  for (uint32_t q = 0; q < hc; q++) s += h[(t * hc + q) * dim + d];
  taps[t * stride + d] = s * (1.0f / (float) hc);
}

}  // namespace

extern "C" bool q4x_tap(const float* h, uint32_t n_tok, uint32_t hc, uint32_t dim, float* taps, uint32_t stride,
                        void* stream) {
  if (!shape(n_tok, hc, dim) || !h || !taps || stride < dim) return false;
  tap_kernel<<<grid((uint64_t) n_tok * dim), kThreads, 0, st(stream)>>>(h, n_tok, hc, dim, taps, stride);
  return ok();
}

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

extern "C" bool q4x_hc_combine_norm(float* h, const float* out, const float* inject, uint32_t n_tok, uint32_t hc,
                                    uint32_t dim, float inject_scale, const float* w, float eps, float* xn,
                                    const Q4xX16* side, void* stream) {
  if (!shape(n_tok, hc, dim) || !h || !out || !inject || !w || !xn) return false;
  combine_norm_kernel<<<dim3(hc, n_tok), kThreads, 0, st(stream)>>>(h, out, inject, hc, dim, inject_scale, w, eps,
                                                                    xn, side_of(side));
  return ok();
}

extern "C" bool q4x_hc_norm_inject(float* h, const float* out, const float* inject, uint32_t n_tok, uint32_t hc,
                                   uint32_t dim, float inject_scale, const float* w, float eps, float* xn,
                                   const Q4xX16* side, const float* winj, float* inj, bool combine, void* stream) {
  constexpr uint32_t kChunk = 8, kWaves = kThreads / 32, kMaxLds = 48 * 1024;
  if (!shape(n_tok, hc, dim) || hc > kWaves || dim % kChunk || dim * 4 > kMaxLds || !h || !w || !xn || !winj || !inj ||
      (combine && (!out || !inject)))
    return false;
  const size_t lds = (size_t) dim * 4;
  if (combine)
    hc_norm_inject_kernel<true><<<n_tok, kThreads, lds, st(stream)>>>(h, out, inject, hc, dim, inject_scale, w, eps, xn,
                                                                     side_of(side), winj, inj);
  else
    hc_norm_inject_kernel<false><<<n_tok, kThreads, lds, st(stream)>>>(h, out, inject, hc, dim, inject_scale, w, eps, xn,
                                                                      side_of(side), winj, inj);
  return ok();
}

extern "C" bool q4x_hc_combine(float* h, const float* out, const float* inject, uint32_t n_tok, uint32_t hc,
                               uint32_t dim, float inject_scale, void* stream) {
  if (!shape(n_tok, hc, dim) || !h || !out || !inject) return false;
  combine_kernel<<<grid((uint64_t) n_tok * hc * dim), kThreads, 0, st(stream)>>>(h, out, inject, n_tok, hc, dim,
                                                                                  inject_scale);
  return ok();
}
