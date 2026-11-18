// Hy4-flash's small steps (include/osmos/gpu/hy4_ops.h): iHC and Engram's ends, the reference's order in F32
#include <cmath>

#include <hip/hip_fp16.h>

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


// gated MLA, absorbed. a Q8_0 row's value i (ggml's block: an f16 scale, 32 int8 codes)
constexpr uint32_t kQ8Block = 32, kQ8Bytes = 34;
constexpr uint32_t kMaxRope = 128;            // the rope part's values at most (a cache row latent + this)
constexpr uint32_t kWaves = kThreads / 32;

__device__ inline float q8_value(const uint8_t* row, uint32_t i) {
  const uint8_t* b = row + (i / kQ8Block) * kQ8Bytes;
  const float d = __half2float(__ushort_as_half(*(const unsigned short*) b));
  return __fmul_rn((float) (int8_t) b[2 + i % kQ8Block], d);
}

// pair i of x (x[2i], x[2i + 1]) rotated by angle pos * freqs[i]: the reference's x cos + rotate_half(x) sin
__device__ inline void rope_pair(const float* in, float* out, uint32_t i, int32_t pos, const float* freqs) {
  const float a = (float) pos * freqs[i], c = cosf(a), s = sinf(a);
  const float x0 = in[2 * i], x1 = in[2 * i + 1];
  out[2 * i] = x0 * c + (-x1) * s;
  out[2 * i + 1] = x1 * c + x0 * s;
}

__global__ __launch_bounds__(kThreads) void mla_kv_kernel(const float* kva, const float* norm_w, uint32_t latent,
                                                          uint32_t rope_dim, float eps, const float* freqs,
                                                          const int32_t* pos, float* cache) {
  __shared__ float lds[kThreads];
  const uint32_t t = blockIdx.x, row = latent + rope_dim;
  const float* in = kva + (uint64_t) t * row;
  float* out = cache + (uint64_t) pos[t] * row;
  float ss = 0;
  for (uint32_t d = threadIdx.x; d < latent; d += kThreads) ss = fmaf(in[d], in[d], ss);
  ss = block_sum<kThreads>(ss, lds);
  const float r = 1.0f / sqrtf(ss / (float) latent + eps);
  for (uint32_t d = threadIdx.x; d < latent; d += kThreads) out[d] = norm_w[d] * (in[d] * r);
  for (uint32_t i = threadIdx.x; i < rope_dim / 2; i += kThreads) rope_pair(in + latent, out + latent, i, pos[t], freqs);
}

// a block a (head, token): the head's nope part (LDS) through each of its k_b rows (a thread a latent value), its
// rope part roped
__global__ __launch_bounds__(kThreads) void mla_q_kernel(const float* q, const uint8_t* k_b, uint32_t nope,
                                                         uint32_t latent, uint32_t rope_dim, const float* freqs,
                                                         const int32_t* pos, float* qa) {
  __shared__ float qn[HY4_MLA_MAX_LATENT];
  const uint32_t h = blockIdx.x, t = blockIdx.y, heads = gridDim.x;
  const float* qh = q + ((uint64_t) t * heads + h) * (nope + rope_dim);
  float* out = qa + ((uint64_t) t * heads + h) * (latent + rope_dim);
  for (uint32_t i = threadIdx.x; i < nope; i += kThreads) qn[i] = qh[i];
  __syncthreads();
  const uint64_t row_bytes = nope / kQ8Block * kQ8Bytes;
  for (uint32_t j = threadIdx.x; j < latent; j += kThreads) {
    const uint8_t* w = k_b + ((uint64_t) h * latent + j) * row_bytes;
    float acc = 0;
    for (uint32_t i = 0; i < nope; i++) acc = fmaf(q8_value(w, i), qn[i], acc);
    out[j] = acc;
  }
  for (uint32_t i = threadIdx.x; i < rope_dim / 2; i += kThreads) rope_pair(qh + nope, out + latent, i, pos[t], freqs);
}

// a block a (head, token): a wave a key at a time (its lanes a strided share of the row, an xor tree), the scores
// in LDS; the softmax with the sink (a fixed tree each); the latent's values summed over the keys in order
__global__ __launch_bounds__(kThreads) void mla_attn_kernel(const float* qa, const float* cache, const float* sinks,
                                                            uint32_t latent, uint32_t rope_dim, uint32_t window,
                                                            float scale, const int32_t* pos, float* o_lat) {
  __shared__ float sc[HY4_MLA_MAX_KEYS], qs[HY4_MLA_MAX_LATENT + kMaxRope], lds[kThreads];
  const uint32_t h = blockIdx.x, t = blockIdx.y, heads = gridDim.x, row = latent + rope_dim;
  const int32_t p = pos[t], lo = window && p >= (int32_t) window ? p - (int32_t) window + 1 : 0;
  const uint32_t n = (uint32_t) (p - lo + 1);
  const float* qh = qa + ((uint64_t) t * heads + h) * row;
  for (uint32_t i = threadIdx.x; i < row; i += kThreads) qs[i] = qh[i];
  __syncthreads();
  const uint32_t wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  for (uint32_t k = wave; k < n; k += kWaves) {
    const float* r = cache + (uint64_t) (lo + k) * row;
    float acc = 0;
    for (uint32_t i = lane; i < row; i += 32) acc = fmaf(qs[i], r[i], acc);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) sc[k] = acc * scale;
  }
  __syncthreads();
  const float sink = sinks[h];
  float m = sink;
  for (uint32_t k = threadIdx.x; k < n; k += kThreads) m = fmaxf(m, sc[k]);
  m = block_max<kThreads>(m, lds);
  float z = 0;
  for (uint32_t k = threadIdx.x; k < n; k += kThreads) z += expf(sc[k] - m);
  z = block_sum<kThreads>(z, lds) + expf(sink - m);
  for (uint32_t k = threadIdx.x; k < n; k += kThreads) sc[k] = expf(sc[k] - m) / z;
  __syncthreads();
  float* out = o_lat + ((uint64_t) t * heads + h) * latent;
  for (uint32_t d = threadIdx.x; d < latent; d += kThreads) {
    float acc = 0;
    for (uint32_t k = 0; k < n; k++) acc = fmaf(sc[k], cache[(uint64_t) (lo + k) * row + d], acc);
    out[d] = acc;
  }
}

// a block a (head, token): the head's latent (LDS) through each of its v_b rows (a thread a value), times the gate
__global__ __launch_bounds__(kThreads) void mla_out_kernel(const float* o_lat, const uint8_t* v_b, const float* gate,
                                                           uint32_t latent, uint32_t v_dim, float* out) {
  __shared__ float ol[HY4_MLA_MAX_LATENT];
  const uint32_t h = blockIdx.x, t = blockIdx.y, heads = gridDim.x;
  for (uint32_t i = threadIdx.x; i < latent; i += kThreads) ol[i] = o_lat[((uint64_t) t * heads + h) * latent + i];
  __syncthreads();
  const uint64_t row_bytes = latent / kQ8Block * kQ8Bytes;
  for (uint32_t i = threadIdx.x; i < v_dim; i += kThreads) {
    const uint8_t* w = v_b + ((uint64_t) h * v_dim + i) * row_bytes;
    float acc = 0;
    for (uint32_t j = 0; j < latent; j++) acc = fmaf(q8_value(w, j), ol[j], acc);
    const uint64_t o = (uint64_t) t * heads * v_dim + (uint64_t) h * v_dim + i;
    out[o] = acc * sigmoid(gate[o]);
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

extern "C" bool hy4_mla_kv(const float* kva, const float* norm_w, uint32_t latent, uint32_t rope_dim, float eps,
                           const float* freqs, const int32_t* pos, uint32_t n_tok, float* cache, void* stream) {
  if (!kva || !norm_w || !freqs || !pos || !cache || !latent || !rope_dim || rope_dim % 2 || !n_tok) return false;
  mla_kv_kernel<<<n_tok, kThreads, 0, st(stream)>>>(kva, norm_w, latent, rope_dim, eps, freqs, pos, cache);
  return ok();
}

extern "C" bool hy4_mla_q(const float* q, const void* k_b, uint32_t heads, uint32_t nope, uint32_t latent,
                          uint32_t rope_dim, const float* freqs, const int32_t* pos, uint32_t n_tok, float* qa,
                          void* stream) {
  if (!q || !k_b || !freqs || !pos || !qa || !heads || !nope || nope % kQ8Block || nope > HY4_MLA_MAX_LATENT ||
      !latent || !rope_dim || rope_dim % 2 || !n_tok)
    return false;
  mla_q_kernel<<<dim3(heads, n_tok), kThreads, 0, st(stream)>>>(q, (const uint8_t*) k_b, nope, latent, rope_dim,
                                                                freqs, pos, qa);
  return ok();
}

extern "C" bool hy4_mla_attn(const float* qa, const float* cache, const float* sinks, uint32_t heads,
                             uint32_t latent, uint32_t rope_dim, uint32_t window, float scale, const int32_t* pos,
                             uint32_t n_tok, float* o_lat, void* stream) {
  if (!qa || !cache || !sinks || !pos || !o_lat || !heads || !latent || latent > HY4_MLA_MAX_LATENT || !rope_dim ||
      rope_dim > kMaxRope || !n_tok || (window && window > HY4_MLA_MAX_KEYS))
    return false;
  mla_attn_kernel<<<dim3(heads, n_tok), kThreads, 0, st(stream)>>>(qa, cache, sinks, latent, rope_dim, window, scale,
                                                                   pos, o_lat);
  return ok();
}

extern "C" bool hy4_mla_out(const float* o_lat, const void* v_b, const float* gate, uint32_t heads, uint32_t latent,
                            uint32_t v_dim, uint32_t n_tok, float* out, void* stream) {
  if (!o_lat || !v_b || !gate || !out || !heads || !latent || latent % kQ8Block || latent > HY4_MLA_MAX_LATENT ||
      !v_dim || !n_tok)
    return false;
  mla_out_kernel<<<dim3(heads, n_tok), kThreads, 0, st(stream)>>>(o_lat, (const uint8_t*) v_b, gate, latent, v_dim,
                                                                  out);
  return ok();
}
