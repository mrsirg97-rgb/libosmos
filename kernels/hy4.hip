// Hy4-flash's small steps (include/osmos/gpu/hy4_ops.h): iHC and Engram's ends, the reference's order in F32
#include <cmath>

#include "port.h"

#include "block.h"
#include "launch.h"
#include "round.h"

extern "C" {
#include "osmos/gpu/hy4_ops.h"
}

namespace {

using launches::ok;
using launches::st;

constexpr int kThreads = 256;
constexpr int kRowThreads = 128;               // an Engram row's values, a thread each (a row at most this)
constexpr float kFp8Max = 448.0f;              // torch.finfo(float8_e4m3fn).max
constexpr float kAmaxFloor = 1e-12f;           // _fp8_round_trip's clamp(min=1e-12)
constexpr float kGateFloor = 1e-6f;            // the gate's clamp_min(1e-6) before its square root

// token t's mixes over its flat streams (fn's `rows` rows and the sum of squares, a fixed tree each), pre and post,
// the pre-weighted sum of the streams, then the sublayer's RMSNorm. a block a token
template <uint32_t hc, uint32_t rows>
__global__ __launch_bounds__(kThreads) void hc_pre_kernel(const float* h, const float* fn, const float* scale,
                                                         const float* base, uint32_t dim, float eps, float hc_eps,
                                                         float magnitude, const float* norm_w, float* x,
                                                         float* post) {
  __shared__ float lds[kThreads];
  __shared__ float pre[hc], r_norm;
  const uint32_t t = blockIdx.x, len = hc * dim;
  const float* ht = h + (uint64_t) t * len;
  float dot[rows] = {}, sq = 0;
  for (uint32_t i = threadIdx.x; i < len; i += kThreads) {
    const float v = ht[i];
    sq = fmaf(v, v, sq);
#pragma unroll
    for (uint32_t r = 0; r < rows; r++) dot[r] = fmaf(fn[(uint64_t) r * len + i], v, dot[r]);
  }
  sq = block_sum<kThreads>(sq, lds);
#pragma unroll
  for (uint32_t r = 0; r < rows; r++) dot[r] = block_sum<kThreads>(dot[r], lds);
  if (threadIdx.x == 0) {
    const float inv = 1.0f / sqrtf(sq / (float) len + eps);
    for (uint32_t k = 0; k < hc; k++) pre[k] = sigmoid(dot[k] * inv * scale[0] + base[k]) + hc_eps;
    if constexpr (rows == 2 * hc)
      for (uint32_t k = 0; k < hc; k++)
        post[t * hc + k] = magnitude * sigmoid(dot[hc + k] * inv * scale[1] + base[hc + k]) + hc_eps;
  }
  __syncthreads();
  float* xt = x + (uint64_t) t * dim;
  float xs = 0;
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) {
    float v = pre[0] * ht[d];
    for (uint32_t k = 1; k < hc; k++) v += pre[k] * ht[(uint64_t) k * dim + d];
    xt[d] = v;
    xs = fmaf(v, v, xs);
  }
  xs = block_sum<kThreads>(xs, lds);
  if (threadIdx.x == 0) r_norm = 1.0f / sqrtf(xs / (float) dim + eps);
  __syncthreads();
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) xt[d] = norm_w[d] * (xt[d] * r_norm);
}

__global__ void hc_post_kernel(float* h, const float* y, const float* post, uint32_t hc, uint32_t dim, uint64_t n) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= n) return;
  const uint32_t d = i % dim;
  const uint64_t tk = i / dim, t = tk / hc;
  h[i] = post[tk] * y[t * dim + d] + h[i];
}

// a row a block: the fp8 round trip (its amax a fixed tree), then its head's RMSNorm
__global__ __launch_bounds__(kRowThreads) void engram_rows_kernel(const float* rows, const float* norm_w,
                                                                  uint32_t cols, uint32_t row_dim, float* emb) {
  __shared__ float lds[kRowThreads];
  const uint64_t r = blockIdx.x;
  const uint32_t c = r % cols, d = threadIdx.x;
  const bool live = d < row_dim;
  const float x = live ? rows[r * row_dim + d] : 0.0f;
  const float amax = block_max<kRowThreads>(fabsf(x), lds);
  const float s = fmaxf(amax, kAmaxFloor) / kFp8Max;
  const float q = rnd::e4m3(fminf(fmaxf(x / s, -kFp8Max), kFp8Max)) * s;
  const float v = x + (q - x);
  const float ss = block_sum<kRowThreads>(live ? v * v : 0.0f, lds);
  const float inv = 1.0f / sqrtf(ss / (float) row_dim + HY4_ENGRAM_EPS);
  if (live) emb[r * row_dim + d] = (v * inv) * norm_w[(uint64_t) c * row_dim + d];
}

// a block a (stream, token): the key's and the stream's norms and their weighted dot, the gate; the value's norm;
// the stream updated
__global__ __launch_bounds__(kThreads) void engram_update_kernel(float* h, const float* kv, const float* kv_b,
                                                                 const float* gate, const float* hc_scale,
                                                                 const float* pre_gamma, uint32_t hc,
                                                                 uint32_t dim) {
  __shared__ float lds[kThreads];
  const uint32_t k = blockIdx.x, t = blockIdx.y;
  float* hk = h + ((uint64_t) t * hc + k) * dim;
  const float* key = kv + ((uint64_t) t * (hc + 1) + k) * dim;
  const float* value = kv + ((uint64_t) t * (hc + 1) + hc) * dim;
  const float* kb = kv_b + (uint64_t) k * dim;
  const float* vb = kv_b + (uint64_t) hc * dim;
  const float* g = gate + (uint64_t) k * dim;
  float sk = 0, sh = 0, sv = 0, dot = 0;
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) {
    const float kd = key[d] + kb[d], hd = hk[d], vd = value[d] + vb[d];
    sk = fmaf(kd, kd, sk), sh = fmaf(hd, hd, sh), sv = fmaf(vd, vd, sv);
    dot = fmaf(kd * hd, g[d], dot);
  }
  sk = block_sum<kThreads>(sk, lds), sh = block_sum<kThreads>(sh, lds);
  sv = block_sum<kThreads>(sv, lds), dot = block_sum<kThreads>(dot, lds);
  const float rk = 1.0f / sqrtf(sk / (float) dim + HY4_ENGRAM_EPS);
  const float rh = 1.0f / sqrtf(sh / (float) dim + HY4_ENGRAM_EPS);
  const float rv = 1.0f / sqrtf(sv / (float) dim + HY4_ENGRAM_EPS);
  const float x = dot * rk * rh / sqrtf((float) dim);
  const float gk = sigmoid(x == 0.0f ? 0.0f : copysignf(sqrtf(fmaxf(fabsf(x), kGateFloor)), x));
  const float* s = hc_scale + (uint64_t) k * dim;
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) {
    const float v = pre_gamma[d] * ((value[d] + vb[d]) * rv);
    hk[d] = hk[d] + (gk * v + s[d] * v);
  }
}

}  // namespace

extern "C" bool hy4_hc_pre(const float* h, const float* fn, const float* scale, const float* base, uint32_t rows,
                           uint32_t hc, uint32_t dim, float eps, float hc_eps, float magnitude, const float* norm_w,
                           uint32_t n_tok, float* x, float* post, void* stream) {
  if (!h || !fn || !scale || !base || !norm_w || !x || !dim || !n_tok || hc != HY4_HC_MAX) return false;
  if (rows == 2 * hc ? !post : rows != hc) return false;
  const hipStream_t s = st(stream);
  if (rows == 2 * hc)
    hc_pre_kernel<HY4_HC_MAX, 2 * HY4_HC_MAX><<<n_tok, kThreads, 0, s>>>(h, fn, scale, base, dim, eps, hc_eps,
                                                                         magnitude, norm_w, x, post);
  else
    hc_pre_kernel<HY4_HC_MAX, HY4_HC_MAX><<<n_tok, kThreads, 0, s>>>(h, fn, scale, base, dim, eps, hc_eps, magnitude,
                                                                     norm_w, x, nullptr);
  return ok();
}

extern "C" bool hy4_hc_post(float* h, const float* y, const float* post, uint32_t hc, uint32_t dim, uint32_t n_tok,
                            void* stream) {
  if (!h || !y || !post || !hc || hc > HY4_HC_MAX || !dim || !n_tok) return false;
  const uint64_t n = (uint64_t) n_tok * hc * dim;
  hc_post_kernel<<<(unsigned) ((n + kThreads - 1) / kThreads), kThreads, 0, st(stream)>>>(h, y, post, hc, dim, n);
  return ok();
}

extern "C" bool hy4_engram_rows(const float* rows, const float* norm_w, uint32_t cols, uint32_t row_dim,
                                uint32_t n_tok, float* emb, void* stream) {
  if (!rows || !norm_w || !emb || !cols || !row_dim || row_dim > (uint32_t) kRowThreads || !n_tok) return false;
  engram_rows_kernel<<<n_tok * cols, kRowThreads, 0, st(stream)>>>(rows, norm_w, cols, row_dim, emb);
  return ok();
}

extern "C" bool hy4_engram_update(float* h, const float* kv, const float* kv_b, const float* gate,
                                  const float* hc_scale, const float* pre_gamma, uint32_t hc, uint32_t dim,
                                  uint32_t n_tok, void* stream) {
  if (!h || !kv || !kv_b || !gate || !hc_scale || !pre_gamma || !hc || hc > HY4_HC_MAX || !dim || !n_tok)
    return false;
  engram_update_kernel<<<dim3(hc, n_tok), kThreads, 0, st(stream)>>>(h, kv, kv_b, gate, hc_scale, pre_gamma, hc,
                                                                     dim);
  return ok();
}
