// ops (include/substrate/gpu/ops.h): the forward's small steps, each as ref.py computes it, in F32
#include <cmath>

#include "quant.h"
#include "round.h"

extern "C" {
#include "substrate/gpu/ops.h"
}

namespace {

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr float kSoftplusLinear = 20.0f;  // F.softplus's threshold: above it, x
constexpr float kRouteEps = 1e-20f;
constexpr float kEngramDotFloor = 1e-6f; // the gate's |dot| floor before the square root

constexpr int kWide = 1024;  // the hc pre-norm's block: a token's whole stream (4 x 5,120) in one block's reach
constexpr int kHcParts = OPS_HC_PARTS;  // the hc mixes' dots, each split in OPS_HC_PARTS fixed parts

// the sum of each thread's v over the block, by a fixed tree (every thread gets it)
template <int NT = kThreads>
__device__ float block_sum(float v, float* lds) {
  lds[threadIdx.x] = v;
  __syncthreads();
  for (int s = NT / 2; s > 0; s >>= 1) {
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

// the hc mixes' dots and the stream's sum of squares, a block a part of 64 tokens: block (p, tt) the part p of every
// row r's dot with each token's stream (r = nm: the stream with itself), into partial[t][r][p] - fn's part read once
// for the tokens, staged with theirs a step at a time; each a sum over the part in order (fmaf), whatever the tile
constexpr int kHcTok = 64, kHcStep = 32, kHcLanes = kThreads / kHcTok;  // a thread's token, its rows r = lane + 4 j
template <uint32_t hc>
__global__ __launch_bounds__(kThreads) void hc_dot_kernel(const float* fn, const float* h, uint32_t n_tok,
                                                         uint32_t len, float* partial) {
  constexpr uint32_t nm = (2 + hc) * hc, rows = nm + 1, per = (rows + kHcLanes - 1) / kHcLanes;
  __shared__ float hs[kHcTok][kHcStep + 1], fs[rows][kHcStep];
  const uint32_t p = blockIdx.x, t0 = blockIdx.y * kHcTok, seg = len / kHcParts;
  const uint32_t tok = threadIdx.x % kHcTok, lane = threadIdx.x / kHcTok;
  float acc[per] = {};
  for (uint32_t i0 = 0; i0 < seg; i0 += kHcStep) {
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < kHcTok * kHcStep; j += kThreads) {
      const uint32_t tt = j / kHcStep, ii = j % kHcStep;
      hs[tt][ii] = t0 + tt < n_tok && i0 + ii < seg ? h[(uint64_t) (t0 + tt) * len + p * seg + i0 + ii] : 0;
    }
    for (uint32_t j = threadIdx.x; j < nm * kHcStep; j += kThreads) {
      const uint32_t r = j / kHcStep, ii = j % kHcStep;
      fs[r][ii] = i0 + ii < seg ? fn[(uint64_t) r * len + p * seg + i0 + ii] : 0;
    }
    __syncthreads();
    for (int ii = 0; ii < kHcStep; ii++) {
      const float x = hs[tok][ii];
#pragma unroll
      for (uint32_t j = 0; j < per; j++) {
        const uint32_t r = lane + kHcLanes * j;
        if (r < nm) acc[j] = fmaf(fs[r][ii], x, acc[j]);
        else if (r == nm) acc[j] = fmaf(x, x, acc[j]);
      }
    }
  }
  if (t0 + tok >= n_tok) return;
  for (uint32_t j = 0; j < per; j++) {
    const uint32_t r = lane + kHcLanes * j;
    if (r < rows) partial[((uint64_t) (t0 + tok) * rows + r) * kHcParts + p] = acc[j];
  }
}

// the same sums, a thread a (row, part, token): fewer tokens than a tile (decode) - fn read from the cache, every sum
// the tiled kernel's, in its order, so the same bits
__global__ void hc_dot_few_kernel(const float* fn, const float* h, uint32_t nm, uint32_t len, float* partial) {
  const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y, seg = len / kHcParts;
  if (j >= (nm + 1) * kHcParts) return;
  const uint32_t r = j / kHcParts, p = j % kHcParts;
  const float* x = h + (uint64_t) t * len + p * seg;
  const float* w = fn + (uint64_t) r * len + p * seg;
  float acc = 0;
  if (r < nm)
    for (uint32_t i = 0; i < seg; i++) acc = fmaf(w[i], x[i], acc);
  else
    for (uint32_t i = 0; i < seg; i++) acc = fmaf(x[i], x[i], acc);
  partial[((uint64_t) t * (nm + 1) + r) * kHcParts + p] = acc;
}

// hc a template argument: the mixes and the Sinkhorn matrix in registers (indexed by a runtime hc they spill)
template <uint32_t hc>
__global__ void hc_mix_kernel(const float* partial, const float* scale, const float* base, uint32_t dim, float eps,
                              float hc_eps, uint32_t iters, float* pre, float* post, float* comb) {
  __shared__ float rows[(2 + OPS_MAX_HC) * OPS_MAX_HC + 1];
  const uint32_t t = blockIdx.x, n = hc * dim, nm = (2 + hc) * hc;
  const float* pt = partial + (uint64_t) t * (nm + 1) * kHcParts;
  for (uint32_t i = threadIdx.x; i <= nm; i += kWave) {  // a lane a row: its parts, in order
    float s = 0;
    for (int p = 0; p < kHcParts; p++) s += pt[i * kHcParts + p];
    rows[i] = s;
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  auto whole = [&](uint32_t r) { return rows[r]; };
  const float r = 1.0f / sqrtf(whole(nm) / (float) n + eps);
  float m[(2 + hc) * hc];
  for (uint32_t i = 0; i < nm; i++) m[i] = whole(i) * r;
  for (uint32_t i = 0; i < hc; i++) {
    pre[t * hc + i] = sigmoid(m[i] * scale[0] + base[i]) + hc_eps;
    post[t * hc + i] = 2.0f * sigmoid(m[hc + i] * scale[1] + base[hc + i]);
  }
  float c[hc][hc];
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

__global__ __launch_bounds__(kWide) void hc_pre_rms_kernel(const float* h, const float* pre, uint32_t hc, uint32_t dim,
                                                            const float* w, float eps, float* x) {
  __shared__ float lds[kWide];
  const uint32_t t = blockIdx.x;
  const float* ht = h + (uint64_t) t * hc * dim;
  float* xt = x + (uint64_t) t * dim;
  float ss = 0;
  for (uint32_t d = threadIdx.x; d < dim; d += kWide) {
    float v = pre[t * hc] * ht[d];
    for (uint32_t k = 1; k < hc; k++) v += pre[t * hc + k] * ht[(uint64_t) k * dim + d];
    xt[d] = v;
    ss += v * v;
  }
  const float r = 1.0f / sqrtf(block_sum<kWide>(ss, lds) / (float) dim + eps);
  for (uint32_t d = threadIdx.x; d < dim; d += kWide) xt[d] = (xt[d] * r) * w[d];
}

// a thread a (token, d): the hc streams' values at d read once for its hc outputs, each the sum of a thread an
// output, in its order - the same bits (hc a template argument: the streams' values in registers)
template <uint32_t hc>
__global__ void hc_post_kernel(const float* y, const float* res, const float* post, const float* comb, uint32_t dim,
                               uint32_t n_tok, float* out) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * dim) return;
  const uint32_t d = i % dim, t = (uint32_t) (i / dim);
  const float* rt = res + (uint64_t) t * hc * dim + d;
  const float* ct = comb + t * hc * hc;
  float r[hc];
  for (uint32_t k = 0; k < hc; k++) r[k] = rt[(uint64_t) k * dim];
  const float yv = y[(uint64_t) t * dim + d];
  float* ot = out + (uint64_t) t * hc * dim + d;
  for (uint32_t j = 0; j < hc; j++) {
    float s = ct[j] * r[0];
    for (uint32_t k = 1; k < hc; k++) s += ct[k * hc + j] * r[k];
    ot[(uint64_t) j * dim] = post[t * hc + j] * yv + s;
  }
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

// k passes of a block argmax over score + bias (the larger, ties to the lower expert: the serial scan's picks)
__global__ void router_kernel(const float* logits, const float* bias, uint32_t n_exp, uint32_t k, float scale,
                              int32_t* idx, float* w) {
  __shared__ float sc[OPS_MAX_EXPERTS], key[OPS_MAX_EXPERTS];
  __shared__ bool taken[OPS_MAX_EXPERTS];
  __shared__ float bk[kThreads];
  __shared__ uint32_t bi[kThreads];
  __shared__ int32_t pick[OPS_MAX_TOPK];
  const uint32_t t = blockIdx.x, none = n_exp;
  for (uint32_t e = threadIdx.x; e < n_exp; e += kThreads) {
    const float l = logits[(uint64_t) t * n_exp + e];
    sc[e] = sqrtf(l > kSoftplusLinear ? l : log1pf(expf(l)));
    key[e] = sc[e] + bias[e];
    taken[e] = false;
  }
  __syncthreads();
  for (uint32_t s = 0; s < k; s++) {
    float best = 0;
    uint32_t at = none;
    for (uint32_t e = threadIdx.x; e < n_exp; e += kThreads)
      if (!taken[e] && (at == none || key[e] > best)) best = key[e], at = e;
    bk[threadIdx.x] = best, bi[threadIdx.x] = at;
    __syncthreads();
    for (uint32_t h = kThreads / 2; h > 0; h >>= 1) {
      if (threadIdx.x < h) {
        const uint32_t o = threadIdx.x + h;
        const bool take = bi[o] != none && (bi[threadIdx.x] == none || bk[o] > bk[threadIdx.x] ||
                                            (bk[o] == bk[threadIdx.x] && bi[o] < bi[threadIdx.x]));
        if (take) bk[threadIdx.x] = bk[o], bi[threadIdx.x] = bi[o];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) pick[s] = (int32_t) bi[0], taken[bi[0]] = true;
    __syncthreads();
  }
  if (threadIdx.x != 0) return;
  float wt[OPS_MAX_TOPK], sum = 0;
  for (uint32_t s = 0; s < k; s++) sum += (wt[s] = sc[pick[s]]);
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

// dst[i][:] = src[idx[i]][:], rows of len
__global__ void gather_rows_kernel(float* dst, const float* src, const int32_t* idx, uint32_t n, uint32_t len) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < (uint64_t) n * len) dst[i] = src[(uint64_t) idx[i / len] * len + i % len];
}

// y[t][d] = sum over j < m of src[idx[t * m + j]][d], in j's order from 0 (the same sums as a zeroed y and m adds)
__global__ void gather_sum_kernel(float* y, const float* src, const int32_t* idx, uint32_t m, uint32_t n,
                                  uint32_t dim) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n * dim) return;
  const uint32_t t = (uint32_t) (i / dim), d = (uint32_t) (i % dim);
  float s = 0.0f;
  for (uint32_t j = 0; j < m; j++) s += src[(uint64_t) idx[t * m + j] * dim + d];
  y[i] = s;
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

extern "C" bool ops_hc_dot(const float* fn, const float* h, uint32_t hc, uint32_t dim, uint32_t n_tok, float* partial,
                           void* stream) {
  const uint32_t len = hc * dim;
  if (!hc || hc > OPS_MAX_HC || !dim || !n_tok || len % kHcParts) return false;
  const hipStream_t s = st(stream);
  if (n_tok < (uint32_t) kHcTok) {  // not a tile's worth of tokens: a thread a sum
    const uint32_t sums = ((2 + hc) * hc + 1) * kHcParts;
    hc_dot_few_kernel<<<dim3((sums + kWave - 1) / kWave, n_tok), kWave, 0, s>>>(fn, h, (2 + hc) * hc, len, partial);
    return ok();
  }
  const dim3 grid(kHcParts, (n_tok + kHcTok - 1) / kHcTok);
  switch (hc) {
#define HC_DOT(n) \
  case n:         \
    hc_dot_kernel<n><<<grid, kThreads, 0, s>>>(fn, h, n_tok, len, partial); \
    break;
    HC_DOT(1) HC_DOT(2) HC_DOT(3) HC_DOT(4) HC_DOT(5) HC_DOT(6) HC_DOT(7) HC_DOT(8)
#undef HC_DOT
  }
  return ok();
}

extern "C" bool ops_hc_mix(const float* partial, const float* scale, const float* base, uint32_t hc, uint32_t dim,
                           float eps, float hc_eps, uint32_t iters, uint32_t n_tok, float* pre, float* post,
                           float* comb, void* stream) {
  if (!hc || hc > OPS_MAX_HC || !dim || !iters || !n_tok) return false;
  static_assert(OPS_MAX_HC == 8, "hc_mix instantiates 1 .. 8");
  const hipStream_t s = st(stream);
  switch (hc) {
#define HC_MIX(n) \
  case n:                                                                                                    \
    hc_mix_kernel<n><<<n_tok, kWave, 0, s>>>(partial, scale, base, dim, eps, hc_eps, iters, pre, post, comb);    \
    break;
    HC_MIX(1) HC_MIX(2) HC_MIX(3) HC_MIX(4) HC_MIX(5) HC_MIX(6) HC_MIX(7) HC_MIX(8)
#undef HC_MIX
    default: return false;
  }
  return ok();
}

extern "C" bool ops_hc_pre_rms(const float* h, const float* pre, uint32_t hc, uint32_t dim, const float* w, float eps,
                               uint32_t n_tok, float* x, void* stream) {
  if (!hc || hc > OPS_MAX_HC || !dim || !n_tok) return false;
  hc_pre_rms_kernel<<<n_tok, kWide, 0, st(stream)>>>(h, pre, hc, dim, w, eps, x);
  return ok();
}

extern "C" bool ops_hc_post(const float* y, const float* res, const float* post, const float* comb, uint32_t hc,
                            uint32_t dim, uint32_t n_tok, float* out, void* stream) {
  if (!hc || hc > OPS_MAX_HC || !dim || !n_tok) return false;
  static_assert(OPS_MAX_HC == 8, "hc_post instantiates 1 .. 8");
  const unsigned g = grid((uint64_t) n_tok * dim);
  const hipStream_t s = st(stream);
  switch (hc) {
#define HC_POST(n) \
  case n:          \
    hc_post_kernel<n><<<g, kThreads, 0, s>>>(y, res, post, comb, dim, n_tok, out); \
    break;
    HC_POST(1) HC_POST(2) HC_POST(3) HC_POST(4) HC_POST(5) HC_POST(6) HC_POST(7) HC_POST(8)
#undef HC_POST
  }
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

extern "C" bool ops_gather_rows(float* dst, const float* src, const int32_t* idx, uint32_t n, uint32_t len,
                                void* stream) {
  if (!n || !len) return false;
  gather_rows_kernel<<<grid((uint64_t) n * len), kThreads, 0, st(stream)>>>(dst, src, idx, n, len);
  return ok();
}

extern "C" bool ops_gather_sum(float* y, const float* src, const int32_t* idx, uint32_t m, uint32_t n, uint32_t dim,
                               void* stream) {
  if (!m || !n || !dim) return false;
  gather_sum_kernel<<<grid((uint64_t) n * dim), kThreads, 0, st(stream)>>>(y, src, idx, m, n, dim);
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
