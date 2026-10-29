// ops (include/substrate/gpu/ops.h): the forward's small steps, each as ref.py computes it, in F32
#include <cmath>

#include "quant.h"
#include "round.h"

extern "C" {
#include "substrate/gpu/ops.h"
}

namespace {

constexpr int kThreads = 256;
constexpr float kSoftplusLinear = 20.0f;  // F.softplus's threshold: above it, x
constexpr float kRouteEps = 1e-20f;
constexpr float kEngramDotFloor = 1e-6f; // the gate's |dot| floor before the square root

// the sum of each thread's v over the block, by a fixed tree (every thread gets it)
__device__ float block_sum(float v, float* lds) {
  lds[threadIdx.x] = v;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) lds[threadIdx.x] += lds[threadIdx.x + s];
    __syncthreads();
  }
  const float r = lds[0];
  __syncthreads();
  return r;
}

__device__ inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

template <quant::Type T>
__global__ void embed_kernel(const uint8_t* table, uint32_t dim, const int32_t* ids, uint32_t hc, float* h) {
  using B = quant::Block<T>;
  constexpr int per = B::values / quant::kChunk;
  const uint32_t t = blockIdx.x;
  const uint8_t* row = table + (uint64_t) ids[t] * (dim / B::values) * B::bytes;
  for (uint32_t c = threadIdx.x; c < dim / quant::kChunk; c += kThreads) {
    float v[quant::kChunk];
    quant::decode<T>(row + (c / per) * B::bytes, (int) (c % per), v);
    for (int j = 0; j < quant::kChunk; j++)
      for (uint32_t k = 0; k < hc; k++)
        h[((uint64_t) t * hc + k) * dim + c * quant::kChunk + j] = rnd::bf16_round(v[j]);
  }
}

__global__ void rms_kernel(const float* x, uint32_t n, uint32_t xs, const float* w, float eps, float* y, uint32_t ys) {
  __shared__ float lds[kThreads];
  const float* xr = x + (uint64_t) blockIdx.x * xs;
  float* yr = y + (uint64_t) blockIdx.x * ys;
  float ss = 0;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) ss += xr[i] * xr[i];
  const float r = 1.0f / sqrtf(block_sum(ss, lds) / (float) n + eps);
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) yr[i] = w ? (xr[i] * r) * w[i] : xr[i] * r;
}

__global__ void hc_mix_kernel(const float* h, const float* mix, const float* scale, const float* base, uint32_t hc,
                              uint32_t dim, float eps, float hc_eps, uint32_t iters, float* pre, float* post,
                              float* comb) {
  __shared__ float lds[kThreads];
  const uint32_t t = blockIdx.x, n = hc * dim, nm = (2 + hc) * hc;
  const float* ht = h + (uint64_t) t * n;
  float ss = 0;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) ss += ht[i] * ht[i];
  const float r = 1.0f / sqrtf(block_sum(ss, lds) / (float) n + eps);
  if (threadIdx.x != 0) return;
  float m[(2 + OPS_MAX_HC) * OPS_MAX_HC];
  for (uint32_t i = 0; i < nm; i++) m[i] = mix[(uint64_t) t * nm + i] * r;
  for (uint32_t i = 0; i < hc; i++) {
    pre[t * hc + i] = sigmoid(m[i] * scale[0] + base[i]) + hc_eps;
    post[t * hc + i] = 2.0f * sigmoid(m[hc + i] * scale[1] + base[hc + i]);
  }
  float c[OPS_MAX_HC][OPS_MAX_HC];
  for (uint32_t i = 0; i < hc; i++) {
    float mx = -INFINITY;
    for (uint32_t j = 0; j < hc; j++) {
      c[i][j] = m[2 * hc + i * hc + j] * scale[2] + base[2 * hc + i * hc + j];
      mx = fmaxf(mx, c[i][j]);
    }
    float s = 0;
    for (uint32_t j = 0; j < hc; j++) s += (c[i][j] = expf(c[i][j] - mx));
    for (uint32_t j = 0; j < hc; j++) c[i][j] = c[i][j] / s + hc_eps;
  }
  for (uint32_t it = 0; it < iters; it++) {
    if (it > 0)
      for (uint32_t i = 0; i < hc; i++) {
        float s = 0;
        for (uint32_t j = 0; j < hc; j++) s += c[i][j];
        for (uint32_t j = 0; j < hc; j++) c[i][j] = c[i][j] / (s + hc_eps);
      }
    for (uint32_t j = 0; j < hc; j++) {
      float s = 0;
      for (uint32_t i = 0; i < hc; i++) s += c[i][j];
      for (uint32_t i = 0; i < hc; i++) c[i][j] = c[i][j] / (s + hc_eps);
    }
  }
  for (uint32_t i = 0; i < hc; i++)
    for (uint32_t j = 0; j < hc; j++) comb[(t * hc + i) * hc + j] = c[i][j];
}

__global__ void hc_pre_rms_kernel(const float* h, const float* pre, uint32_t hc, uint32_t dim, const float* w,
                                  float eps, float* x) {
  __shared__ float lds[kThreads];
  const uint32_t t = blockIdx.x;
  const float* ht = h + (uint64_t) t * hc * dim;
  float* xt = x + (uint64_t) t * dim;
  float ss = 0;
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) {
    float v = pre[t * hc] * ht[d];
    for (uint32_t k = 1; k < hc; k++) v += pre[t * hc + k] * ht[(uint64_t) k * dim + d];
    xt[d] = v;
    ss += v * v;
  }
  const float r = 1.0f / sqrtf(block_sum(ss, lds) / (float) dim + eps);
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) xt[d] = (xt[d] * r) * w[d];
}

__global__ void hc_post_kernel(const float* y, const float* res, const float* post, const float* comb, uint32_t hc,
                               uint32_t dim, uint32_t n_tok, float* out) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * hc * dim) return;
  const uint32_t d = i % dim, j = (i / dim) % hc, t = (uint32_t) (i / dim / hc);
  const float* rt = res + (uint64_t) t * hc * dim;
  const float* ct = comb + t * hc * hc;
  float s = ct[j] * rt[d];
  for (uint32_t k = 1; k < hc; k++) s += ct[k * hc + j] * rt[(uint64_t) k * dim + d];
  out[i] = post[t * hc + j] * y[(uint64_t) t * dim + d] + s;
}

__global__ void rope_kernel(float* x, uint32_t heads, uint32_t head_dim, uint32_t rope_dim, uint32_t stride,
                            const float* freqs, const int32_t* pos, bool inverse, uint32_t n_tok) {
  const uint32_t half = rope_dim / 2;
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * heads * half) return;
  const uint32_t p = i % half, hd = (i / half) % heads, t = (uint32_t) (i / half / heads);
  float* v = x + (uint64_t) t * stride + (uint64_t) hd * head_dim + (head_dim - rope_dim) + 2 * p;
  const float ang = (float) pos[t] * freqs[p];
  const float c = cosf(ang), s = inverse ? -sinf(ang) : sinf(ang);
  const float a = v[0], b = v[1];
  v[0] = a * c - b * s;
  v[1] = a * s + b * c;
}

// (BF16); dot = sum((h * w) * key) * (rstd(h) * rstd(key)) * scale; h += sigmoid(signed sqrt(dot)) * value
__global__ void engram_gate_kernel(float* h, const float* kv, const uint16_t* qw, const uint16_t* kw, uint32_t hc,
                                   uint32_t dim, float eps, float scale) {
  __shared__ float lds[kThreads];
  __shared__ float gate;
  const uint32_t k = blockIdx.x, t = blockIdx.y;
  float* hk = h + ((uint64_t) t * hc + k) * dim;
  const float* key = kv + ((uint64_t) t * (hc + 1) + k) * dim;
  const float* value = kv + ((uint64_t) t * (hc + 1) + hc) * dim;
  float sh = 0, sk = 0, dot = 0;
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) {
    const float w = __uint_as_float((uint32_t) qw[(uint64_t) k * dim + d] << 16) *
                    __uint_as_float((uint32_t) kw[(uint64_t) k * dim + d] << 16);
    sh += hk[d] * hk[d], sk += key[d] * key[d], dot += (hk[d] * w) * key[d];
  }
  sh = block_sum(sh, lds), sk = block_sum(sk, lds), dot = block_sum(dot, lds);
  if (threadIdx.x == 0) {
    const float rstd = (1.0f / sqrtf(sh / (float) dim + eps)) * (1.0f / sqrtf(sk / (float) dim + eps));
    const float x = dot * rstd * scale;
    gate = sigmoid(copysignf(sqrtf(fmaxf(fabsf(x), kEngramDotFloor)), x));
  }
  __syncthreads();
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) hk[d] = hk[d] + gate * value[d];
}

__global__ void router_kernel(const float* logits, const float* bias, uint32_t n_exp, uint32_t k, float scale,
                              int32_t* idx, float* w) {
  __shared__ float sc[OPS_MAX_EXPERTS];
  const uint32_t t = blockIdx.x;
  for (uint32_t e = threadIdx.x; e < n_exp; e += kThreads) {
    const float l = logits[(uint64_t) t * n_exp + e];
    sc[e] = sqrtf(l > kSoftplusLinear ? l : log1pf(expf(l)));
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  int32_t pick[OPS_MAX_TOPK];
  float wt[OPS_MAX_TOPK], sum = 0;
  for (uint32_t s = 0; s < k; s++) {
    int32_t best = -1;
    for (uint32_t e = 0; e < n_exp; e++) {
      bool taken = false;
      for (uint32_t r = 0; r < s; r++) taken |= pick[r] == (int32_t) e;
      if (!taken && (best < 0 || sc[e] + bias[e] > sc[best] + bias[best])) best = (int32_t) e;
    }
    pick[s] = best;
    sum += (wt[s] = sc[best]);
  }
  for (uint32_t s = 0; s < k; s++) {
    idx[t * k + s] = pick[s];
    w[t * k + s] = wt[s] / (sum + kRouteEps) * scale;
  }
}

__global__ void swiglu_kernel(const float* g, const float* u, uint32_t n, float lim, const float* scale,
                              uint32_t n_tok, float* h) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n * n_tok) return;
  const float gv = fminf(g[i], lim), uv = fminf(fmaxf(u[i], -lim), lim);
  const float v = gv / (1.0f + expf(-gv)) * uv;
  h[i] = scale ? v * scale[i / n] : v;
}

__global__ void scale_kernel(float* x, uint64_t n, float s) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) x[i] = x[i] * s;
}

__global__ void add_kernel(float* y, const float* a, uint64_t n) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) y[i] += a[i];
}

unsigned grid(uint64_t n) { return (unsigned) ((n + kThreads - 1) / kThreads); }
bool ok() { return hipGetLastError() == hipSuccess; }
hipStream_t st(void* s) { return (hipStream_t) s; }

}  // namespace

extern "C" bool ops_embed(uint32_t type, const void* table, uint32_t dim, const int32_t* ids, uint32_t n_tok,
                          uint32_t hc, float* h, void* stream) {
  if (!n_tok || !hc || dim % quant::kQK) return false;
  const uint8_t* tb = (const uint8_t*) table;
  switch (type) {
    case quant::kQ5_K: embed_kernel<quant::kQ5_K><<<n_tok, kThreads, 0, st(stream)>>>(tb, dim, ids, hc, h); break;
    case quant::kQ6_K: embed_kernel<quant::kQ6_K><<<n_tok, kThreads, 0, st(stream)>>>(tb, dim, ids, hc, h); break;
    case quant::kQ8_0: embed_kernel<quant::kQ8_0><<<n_tok, kThreads, 0, st(stream)>>>(tb, dim, ids, hc, h); break;
    case quant::kBF16: embed_kernel<quant::kBF16><<<n_tok, kThreads, 0, st(stream)>>>(tb, dim, ids, hc, h); break;
    default: return false;
  }
  return ok();
}

extern "C" bool ops_rms(const float* x, uint32_t n, uint32_t xs, const float* w, float eps, uint32_t rows, float* y,
                        uint32_t ys, void* stream) {
  if (!n || !rows) return false;
  rms_kernel<<<rows, kThreads, 0, st(stream)>>>(x, n, xs, w, eps, y, ys);
  return ok();
}

extern "C" bool ops_hc_mix(const float* h, const float* mix, const float* scale, const float* base, uint32_t hc,
                           uint32_t dim, float eps, float hc_eps, uint32_t iters, uint32_t n_tok, float* pre,
                           float* post, float* comb, void* stream) {
  if (!hc || hc > OPS_MAX_HC || !dim || !iters || !n_tok) return false;
  hc_mix_kernel<<<n_tok, kThreads, 0, st(stream)>>>(h, mix, scale, base, hc, dim, eps, hc_eps, iters, pre, post, comb);
  return ok();
}

extern "C" bool ops_hc_pre_rms(const float* h, const float* pre, uint32_t hc, uint32_t dim, const float* w, float eps,
                               uint32_t n_tok, float* x, void* stream) {
  if (!hc || hc > OPS_MAX_HC || !dim || !n_tok) return false;
  hc_pre_rms_kernel<<<n_tok, kThreads, 0, st(stream)>>>(h, pre, hc, dim, w, eps, x);
  return ok();
}

extern "C" bool ops_hc_post(const float* y, const float* res, const float* post, const float* comb, uint32_t hc,
                            uint32_t dim, uint32_t n_tok, float* out, void* stream) {
  if (!hc || hc > OPS_MAX_HC || !dim || !n_tok) return false;
  hc_post_kernel<<<grid((uint64_t) n_tok * hc * dim), kThreads, 0, st(stream)>>>(y, res, post, comb, hc, dim, n_tok,
                                                                                  out);
  return ok();
}

extern "C" bool ops_rope(float* x, uint32_t heads, uint32_t head_dim, uint32_t rope_dim, uint32_t stride,
                         const float* freqs, const int32_t* pos, bool inverse, uint32_t n_tok, void* stream) {
  if (!heads || !rope_dim || rope_dim % 2 || rope_dim > head_dim || !n_tok) return false;
  rope_kernel<<<grid((uint64_t) n_tok * heads * (rope_dim / 2)), kThreads, 0, st(stream)>>>(
      x, heads, head_dim, rope_dim, stride, freqs, pos, inverse, n_tok);
  return ok();
}

extern "C" bool ops_engram_gate(float* h, const float* kv, const void* q_w, const void* k_w, uint32_t hc, uint32_t dim,
                                float eps, uint32_t n_tok, void* stream) {
  if (!hc || hc > OPS_MAX_HC || !dim || !n_tok) return false;
  engram_gate_kernel<<<dim3(hc, n_tok), kThreads, 0, st(stream)>>>(h, kv, (const uint16_t*) q_w, (const uint16_t*) k_w,
                                                                    hc, dim, eps, (float) std::pow((double) dim, -0.5));
  return ok();
}

extern "C" bool ops_router(const float* logits, const float* bias, uint32_t n_exp, uint32_t k, float scale,
                           uint32_t n_tok, int32_t* idx, float* w, void* stream) {
  if (!n_exp || n_exp > OPS_MAX_EXPERTS || !k || k > OPS_MAX_TOPK || k > n_exp || !n_tok) return false;
  router_kernel<<<n_tok, kThreads, 0, st(stream)>>>(logits, bias, n_exp, k, scale, idx, w);
  return ok();
}

extern "C" bool ops_swiglu(const float* g, const float* u, uint32_t n, float lim, const float* scale, uint32_t n_tok,
                           float* h, void* stream) {
  if (!n || !n_tok) return false;
  swiglu_kernel<<<grid((uint64_t) n * n_tok), kThreads, 0, st(stream)>>>(g, u, n, lim, scale, n_tok, h);
  return ok();
}

extern "C" bool ops_scale(float* x, uint64_t n, float s, void* stream) {
  if (!n) return false;
  scale_kernel<<<grid(n), kThreads, 0, st(stream)>>>(x, n, s);
  return ok();
}

extern "C" bool ops_add(float* y, const float* a, uint64_t n, void* stream) {
  if (!n) return false;
  add_kernel<<<grid(n), kThreads, 0, st(stream)>>>(y, a, n);
  return ok();
}
