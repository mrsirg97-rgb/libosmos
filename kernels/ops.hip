// ops (include/substrate/gpu/ops.h): the forward's small steps, each as ref.py computes it, in F32
#include <cmath>

#include "quant.h"
#include "round.h"
#include "lane.h"
#include "mm.h"

extern "C" {
#include "substrate/gpu/mm.h"
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
constexpr int kHcAhead = 16;            // hc_dot_few's loads in flight a thread (DESIGN.md: 24 -> 10.5 us)

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

// x16 (when not NULL): y as mm_x16 would put it, rows past `rows` (a block each) zero to a whole tile
__global__ void rms_kernel(const float* x, uint32_t n, uint32_t xs, const float* w, float eps, float* y, uint32_t ys,
                           uint32_t rows, _Float16* x16, int32_t* overflow) {
  __shared__ float lds[kThreads];
  if (blockIdx.x >= rows) {  // a pad row of x16's last tile
    for (uint32_t i = threadIdx.x; i < n; i += kThreads) mm::x16_put(x16, blockIdx.x, i, n, 0.0f, overflow);
    return;
  }
  const float* xr = x + (uint64_t) blockIdx.x * xs;
  float* yr = y + (uint64_t) blockIdx.x * ys;
  float ss = 0;
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) ss += xr[i] * xr[i];
  const float r = 1.0f / sqrtf(block_sum(ss, lds) / (float) n + eps);
  for (uint32_t i = threadIdx.x; i < n; i += kThreads) {
    const float v = w ? (xr[i] * r) * w[i] : xr[i] * r;
    yr[i] = v;
    if (x16) mm::x16_put(x16, blockIdx.x, i, n, v, overflow);
  }
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
  const float* a = r < nm ? w : x;  // the norm's row: x with itself
  float acc = 0;
  uint32_t i = 0;
  for (; i + kHcAhead <= seg; i += kHcAhead) {  // kHcAhead loads in flight, then their FMAs in order
    float av[kHcAhead], xv[kHcAhead];
#pragma unroll
    for (int u = 0; u < kHcAhead; u++) av[u] = a[i + u], xv[u] = x[i + u];
#pragma unroll
    for (int u = 0; u < kHcAhead; u++) acc = fmaf(av[u], xv[u], acc);
  }
  for (; i < seg; i++) acc = fmaf(a[i], x[i], acc);
  partial[((uint64_t) t * (nm + 1) + r) * kHcParts + p] = acc;
}

// hc a template argument: the mixes and the Sinkhorn matrix in registers (indexed by a runtime hc they spill). token
// t's mixes by the whole block (every thread enters: a block barrier inside): its threads a row's parts each, in
// order, then wave 0 the rest - hc_mix_kernel's block is that wave, hc_mix_pre_rms_kernel's a wide one
template <uint32_t hc>
__device__ void hc_mix_token(uint32_t t, const float* partial, const float* scale, const float* base, uint32_t dim,
                             float eps, float hc_eps, uint32_t iters, float* pre, float* post, float* comb,
                             float* rows) {
  const uint32_t n = hc * dim, nm = (2 + hc) * hc, tid = threadIdx.x;
  const float* pt = partial + (uint64_t) t * (nm + 1) * kHcParts;
  for (uint32_t i = tid; i <= nm; i += blockDim.x) {  // a thread a row: its parts, in order
    float s = 0;
    for (int p = 0; p < kHcParts; p++) s += pt[i * kHcParts + p];
    rows[i] = s;
  }
  __syncthreads();
  if (tid >= (uint32_t) kWave) return;
  auto whole = [&](uint32_t r) { return rows[r]; };
  const float r = 1.0f / sqrtf(whole(nm) / (float) n + eps);
  if constexpr (hc * hc <= kWave) {  // a lane an entry: each sum the serial one's, in its order, through shuffles
    const uint32_t lane = tid, i = lane / hc, j = lane % hc;
    if (lane < hc) pre[t * hc + lane] = sigmoid(whole(lane) * r * scale[0] + base[lane]) + hc_eps;
    else if (lane < 2 * hc)
      post[t * hc + lane - hc] = 2.0f * sigmoid(whole(lane) * r * scale[1] + base[lane]);
    const bool live = lane < hc * hc;
    float c = live ? whole(2 * hc + lane) * r * scale[2] + base[2 * hc + lane] : 0.0f;
    auto at = [&](float v, uint32_t a) { return __shfl(v, (int) (a % kWave), kWave); };
    float mx = -INFINITY;
    for (uint32_t k = 0; k < hc; k++) mx = fmaxf(mx, at(c, i * hc + k));
    const float e = expf(c - mx);
    float sum = 0;
    for (uint32_t k = 0; k < hc; k++) sum += at(e, i * hc + k);
    c = e / sum + hc_eps;
    for (uint32_t it = 0; it < iters; it++) {
      if (it > 0) {
        float rs = 0;
        for (uint32_t k = 0; k < hc; k++) rs += at(c, i * hc + k);
        c = c / (rs + hc_eps);
      }
      float cs = 0;
      for (uint32_t k = 0; k < hc; k++) cs += at(c, k * hc + j);
      c = c / (cs + hc_eps);
    }
    if (live) comb[(t * hc + i) * hc + j] = c;
    return;
  }
  if (tid != 0) return;
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

template <uint32_t hc>
__global__ void hc_mix_kernel(const float* partial, const float* scale, const float* base, uint32_t dim, float eps,
                              float hc_eps, uint32_t iters, float* pre, float* post, float* comb) {
  __shared__ float rows[(2 + OPS_MAX_HC) * OPS_MAX_HC + 1];
  hc_mix_token<hc>(blockIdx.x, partial, scale, base, dim, eps, hc_eps, iters, pre, post, comb, rows);
}

// x16 (when not NULL): the result as mm_x16 would put it, rows n .. a whole tile zero (a block each, past n). block
// t's: token t's, a kWide block's
__device__ void hc_pre_rms_token(uint32_t t, const float* h, const float* pre, uint32_t hc, uint32_t dim,
                                 const float* w, float eps, float* x, uint32_t n_tok, _Float16* x16,
                                 int32_t* overflow, float* lds) {
  if (t >= n_tok) {  // a pad row of x16's last tile
    for (uint32_t d = threadIdx.x; d < dim; d += kWide) mm::x16_put(x16, t, d, dim, 0.0f, overflow);
    return;
  }
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
  for (uint32_t d = threadIdx.x; d < dim; d += kWide) {
    const float v = (xt[d] * r) * w[d];
    xt[d] = v;
    if (x16) mm::x16_put(x16, t, d, dim, v, overflow);
  }
}

__global__ __launch_bounds__(kWide) void hc_pre_rms_kernel(const float* h, const float* pre, uint32_t hc, uint32_t dim,
                                                            const float* w, float eps, float* x, uint32_t n_tok,
                                                            _Float16* x16, int32_t* overflow) {
  __shared__ float lds[kWide];
  hc_pre_rms_token(blockIdx.x, h, pre, hc, dim, w, eps, x, n_tok, x16, overflow, lds);
}

// the next mixes and the pre-norm in one launch: token t's hc_mix_token, then (the block past it) its
// hc_pre_rms_token - the two don't depend on each other (the norm's pre is a mix before), the barrier only for a
// caller whose pre is the mix's own
template <uint32_t hc>
__global__ __launch_bounds__(kWide) void hc_mix_pre_rms_kernel(const float* partial, const float* scale,
                                                                const float* base, float eps, float hc_eps,
                                                                uint32_t iters, float* mpre, float* post, float* comb,
                                                                const float* h, const float* pre, uint32_t dim,
                                                                const float* w, float* x, uint32_t n_tok,
                                                                _Float16* x16, int32_t* overflow) {
  __shared__ float rows[(2 + OPS_MAX_HC) * OPS_MAX_HC + 1];
  __shared__ float lds[kWide];
  const uint32_t t = blockIdx.x;
  if (t < n_tok) hc_mix_token<hc>(t, partial, scale, base, dim, eps, hc_eps, iters, mpre, post, comb, rows);
  __syncthreads();
  hc_pre_rms_token(t, h, pre, hc, dim, w, eps, x, n_tok, x16, overflow, lds);
}

__global__ void hc_pre_kernel(const float* h, const float* pre, uint32_t hc, uint32_t dim, uint32_t n_tok, float* x) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * dim) return;
  const uint32_t d = i % dim, t = (uint32_t) (i / dim);
  const float* ht = h + (uint64_t) t * hc * dim + d;
  float v = pre[t * hc] * ht[0];
  for (uint32_t k = 1; k < hc; k++) v += pre[t * hc + k] * ht[(uint64_t) k * dim];
  x[(uint64_t) t * dim + d] = v;
}

__global__ void hc_mean_kernel(const float* h, uint32_t hc, uint32_t dim, uint32_t n_tok, float* out,
                               uint32_t out_stride) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * dim) return;
  const uint32_t d = i % dim, t = (uint32_t) (i / dim);
  const float* ht = h + (uint64_t) t * hc * dim + d;
  float s = ht[0];
  for (uint32_t k = 1; k < hc; k++) s += ht[(uint64_t) k * dim];
  out[(uint64_t) t * out_stride + d] = s / (float) hc;
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

// hc_post then hc_dot_few on its output, a block a (part, token): the part's values as hc_post_kernel's thread
// computes each (its sums in its order), written and kept in LDS, then a thread a row's dot over them in
// hc_dot_few_kernel's order - both kernels' bits
template <uint32_t hc>
__global__ void hc_post_dot_kernel(const float* y, const float* res, const float* post, const float* comb,
                                   uint32_t dim, float* out, const float* fn, float* partial) {
  extern __shared__ float xs[];
  const uint32_t p = blockIdx.x, t = blockIdx.y, len = hc * dim, seg = len / kHcParts, nm = (2 + hc) * hc;
  const float* ct = comb + t * hc * hc;
  for (uint32_t i = threadIdx.x; i < seg; i += blockDim.x) {
    const uint32_t at = p * seg + i, j = at / dim, d = at % dim;
    const float* rt = res + (uint64_t) t * len + d;
    float s = ct[j] * rt[0];
    for (uint32_t k = 1; k < hc; k++) s += ct[k * hc + j] * rt[(uint64_t) k * dim];
    const float v = post[t * hc + j] * y[(uint64_t) t * dim + d] + s;
    out[(uint64_t) t * len + at] = v;
    xs[i] = v;
  }
  __syncthreads();
  for (uint32_t r = threadIdx.x; r <= nm; r += blockDim.x) {
    const float* a = r < nm ? fn + (uint64_t) r * len + p * seg : xs;  // the norm's row: x with itself
    float acc = 0;
    uint32_t i = 0;
    for (; i + kHcAhead <= seg; i += kHcAhead) {  // hc_dot_few_kernel's: loads in flight, then the FMAs in order
      float av[kHcAhead];
#pragma unroll
      for (int u = 0; u < kHcAhead; u++) av[u] = a[i + u];
#pragma unroll
      for (int u = 0; u < kHcAhead; u++) acc = fmaf(av[u], xs[i + u], acc);
    }
    for (; i < seg; i++) acc = fmaf(a[i], xs[i], acc);
    partial[((uint64_t) t * (nm + 1) + r) * kHcParts + p] = acc;
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

// a key's whole tail a block a token (ops_kv_finish): rms_kernel's norm (its threads, its block_sum), rope_kernel's
// rotation of the last rope_dim, round.h's FP8 a wave a block of 32, then the row to kv and to the ring at pos % slots
// - each step the separate kernel's arithmetic in its order, so its bits
__global__ void kv_finish_kernel(float* kv, uint32_t hd, const float* w, float eps, uint32_t rd, const float* freqs,
                                 const int32_t* pos, float* ring, uint32_t slots) {
  __shared__ float lds[kThreads];
  __shared__ float row[OPS_MAX_KV_DIM];
  const uint32_t t = blockIdx.x;
  float* x = kv + (uint64_t) t * hd;
  float ss = 0;
  for (uint32_t i = threadIdx.x; i < hd; i += kThreads) ss += x[i] * x[i];
  const float r = 1.0f / sqrtf(block_sum(ss, lds) / (float) hd + eps);
  for (uint32_t i = threadIdx.x; i < hd; i += kThreads) row[i] = (x[i] * r) * w[i];
  __syncthreads();
  for (uint32_t p = threadIdx.x; p < rd / 2; p += kThreads) {
    float* v = row + (hd - rd) + 2 * p;
    const float ang = (float) pos[t] * freqs[p];
    const float c = cosf(ang), sn = sinf(ang);
    const float a = v[0], b = v[1];
    v[0] = a * c - b * sn;
    v[1] = a * sn + b * c;
  }
  __syncthreads();
  const uint32_t lane = threadIdx.x % kWave, wave = threadIdx.x / kWave;
  float* out = ring + (uint64_t) ((uint32_t) pos[t] % slots) * hd;
  for (uint32_t b = wave; b < hd / rnd::kFp8Block; b += kThreads / kWave) {  // the whole wave a block
    const uint32_t i = b * rnd::kFp8Block + lane;
    const float v = rnd::fp8_block(row[i]);
    x[i] = v, out[i] = v;
  }
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

// x16 (when not NULL): h as mm_x16 would put it, the rows past n_tok zero to a whole tile (threads past h's)
__global__ void swiglu_kernel(const float* g, const float* u, uint32_t n, float lim, const float* scale,
                              uint32_t n_tok, float* h, _Float16* x16, int32_t* overflow) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  const uint32_t t = (uint32_t) (i / n), c = (uint32_t) (i % n);
  if (i >= (uint64_t) n * n_tok) {
    if (x16 && t < (n_tok + mm::kTile - 1) / mm::kTile * mm::kTile) mm::x16_put(x16, t, c, n, 0.0f, overflow);
    return;
  }
  const float gv = fminf(g[i], lim), uv = fminf(fmaxf(u[i], -lim), lim);
  const float v = gv / (1.0f + expf(-gv)) * uv, o = scale ? v * scale[i / n] : v;
  h[i] = o;
  if (x16) mm::x16_put(x16, t, c, n, o, overflow);
}

// the lane's (lane.h): the same activation in operations the CPU repeats bit for bit
__global__ void swiglu_lane_kernel(const float* g, const float* u, uint32_t n, float lim, const float* scale,
                                   uint32_t n_tok, float* h) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n * n_tok) return;
  h[i] = lane_swiglu(g[i], u[i], lim, scale[i / n]);
}

// dst[i][:] = src[idx[i]][:], rows of len
__global__ void gather_rows_kernel(float* dst, const float* src, const int32_t* idx, uint32_t n, uint32_t len) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < (uint64_t) n * len) dst[i] = src[(uint64_t) idx[i / len] * len + i % len];
}

// y[t][d] = sum over j < m of src[idx[t * m + j]][d], in j's order from 0 (the same sums as a zeroed y and m adds)
// add (when not NULL): its value added after the sum, as ops_add after it would
__global__ void gather_sum_kernel(float* y, const float* src, const int32_t* idx, uint32_t m, uint32_t n,
                                  uint32_t dim, const float* add) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n * dim) return;
  const uint32_t t = (uint32_t) (i / dim), d = (uint32_t) (i % dim);
  float s = 0.0f;
  for (uint32_t j = 0; j < m; j++) s += src[(uint64_t) idx[t * m + j] * dim + d];
  y[i] = add ? s + add[i] : s;
}

__global__ void scale_kernel(float* x, uint64_t n, float s) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) x[i] = x[i] * s;
}

// a copy: a 16-byte word a thread, then the last bytes (fewer than 16) a byte a thread
__global__ void copy_kernel(uint4* dst, const uint4* src, uint64_t words, uint8_t* dst_tail, const uint8_t* src_tail,
                            uint32_t tail) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < words)
    dst[i] = src[i];
  else if (i - words < tail)
    dst_tail[i - words] = src_tail[i - words];
}

// one thread until *flag (host memory the GPU maps) holds value, or value with its top bit (the poster's failure):
// a system-scope acquire, so what the host wrote before it is seen by the kernels after
__global__ void wait_kernel(const uint32_t* flag, uint32_t value) {
  constexpr uint32_t kFailed = 0x80000000u;
  constexpr int kNap = 2;  // s_sleep's units (64 clocks each) between looks
  for (;;) {
    const uint32_t v = __hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    if ((v & ~kFailed) == value) break;
    __builtin_amdgcn_s_sleep(kNap);
  }
}

// rows idx[0 .. n) of src copied to the same rows of dst (len floats a row)
__global__ void copy_rows_kernel(float* dst, const float* src, const int32_t* idx, uint32_t n, uint32_t len) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n * len) return;
  const uint64_t at = (uint64_t) idx[i / len] * len + i % len;
  dst[at] = src[at];
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
  rms_kernel<<<rows, kThreads, 0, st(stream)>>>(x, n, xs, w, eps, y, ys, rows, nullptr, nullptr);
  return ok();
}

static uint32_t tiles(uint32_t n) { return (n + mm::kTile - 1) / mm::kTile * mm::kTile; }  // n to a whole x16 tile

extern "C" bool ops_rms_x16(const float* x, uint32_t n, uint32_t xs, const float* w, float eps, uint32_t rows,
                            float* y, uint32_t ys, void* x16, int32_t* overflow, void* stream) {
  if (!n || n % mm::kTile || !rows || rows > MM_MAX_TOKENS || !x16 || !overflow) return false;
  rms_kernel<<<tiles(rows), kThreads, 0, st(stream)>>>(x, n, xs, w, eps, y, ys, rows, (_Float16*) x16, overflow);
  return ok();
}

extern "C" bool ops_kv_finish(float* kv, uint32_t head_dim, const float* w, float eps, uint32_t rope_dim,
                              const float* freqs, const int32_t* pos, float* ring, uint32_t slots, uint32_t n_tok,
                              void* stream) {
  if (!kv || !w || !ring || !slots || !n_tok || !head_dim || head_dim > OPS_MAX_KV_DIM ||
      head_dim % rnd::kFp8Block || rope_dim > head_dim || rope_dim % 2)
    return false;
  kv_finish_kernel<<<n_tok, kThreads, 0, st(stream)>>>(kv, head_dim, w, eps, rope_dim, freqs, pos, ring, slots);
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
  hc_pre_rms_kernel<<<n_tok, kWide, 0, st(stream)>>>(h, pre, hc, dim, w, eps, x, n_tok, nullptr, nullptr);
  return ok();
}

extern "C" bool ops_hc_pre_rms_x16(const float* h, const float* pre, uint32_t hc, uint32_t dim, const float* w,
                                   float eps, uint32_t n_tok, float* x, void* x16, int32_t* overflow, void* stream) {
  if (!hc || hc > OPS_MAX_HC || !dim || dim % mm::kTile || !n_tok || n_tok > MM_MAX_TOKENS || !x16 || !overflow)
    return false;
  hc_pre_rms_kernel<<<tiles(n_tok), kWide, 0, st(stream)>>>(h, pre, hc, dim, w, eps, x, n_tok, (_Float16*) x16,
                                                             overflow);
  return ok();
}

extern "C" bool ops_hc_pre(const float* h, const float* pre, uint32_t hc, uint32_t dim, uint32_t n_tok, float* x,
                           void* stream) {
  if (!hc || !dim || !n_tok) return false;
  hc_pre_kernel<<<grid((uint64_t) n_tok * dim), kThreads, 0, st(stream)>>>(h, pre, hc, dim, n_tok, x);
  return ok();
}

extern "C" bool ops_hc_mean(const float* h, uint32_t hc, uint32_t dim, uint32_t n_tok, float* out,
                            uint32_t out_stride, void* stream) {
  if (!hc || !dim || !n_tok || out_stride < dim) return false;
  hc_mean_kernel<<<grid((uint64_t) n_tok * dim), kThreads, 0, st(stream)>>>(h, hc, dim, n_tok, out, out_stride);
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

extern "C" bool ops_swiglu_lane(const float* g, const float* u, uint32_t n, float lim, const float* scale,
                                uint32_t n_tok, float* h, void* stream) {
  if (!n || !n_tok || !scale) return false;
  swiglu_lane_kernel<<<grid((uint64_t) n * n_tok), kThreads, 0, st(stream)>>>(g, u, n, lim, scale, n_tok, h);
  return ok();
}

extern "C" bool ops_swiglu(const float* g, const float* u, uint32_t n, float lim, const float* scale, uint32_t n_tok,
                           float* h, void* stream) {
  if (!n || !n_tok) return false;
  swiglu_kernel<<<grid((uint64_t) n * n_tok), kThreads, 0, st(stream)>>>(g, u, n, lim, scale, n_tok, h, nullptr,
                                                                         nullptr);
  return ok();
}

extern "C" bool ops_swiglu_x16(const float* g, const float* u, uint32_t n, float lim, const float* scale,
                               uint32_t n_tok, float* h, void* x16, int32_t* overflow, void* stream) {
  if (!n || n % mm::kTile || !n_tok || n_tok > MM_MAX_TOKENS || !x16 || !overflow) return false;
  swiglu_kernel<<<grid((uint64_t) n * tiles(n_tok)), kThreads, 0, st(stream)>>>(g, u, n, lim, scale, n_tok, h,
                                                                                (_Float16*) x16, overflow);
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
  gather_sum_kernel<<<grid((uint64_t) n * dim), kThreads, 0, st(stream)>>>(y, src, idx, m, n, dim, nullptr);
  return ok();
}

extern "C" bool ops_gather_sum_add(float* y, const float* src, const int32_t* idx, uint32_t m, uint32_t n,
                                   uint32_t dim, const float* add, void* stream) {
  if (!m || !n || !dim || !add) return false;
  gather_sum_kernel<<<grid((uint64_t) n * dim), kThreads, 0, st(stream)>>>(y, src, idx, m, n, dim, add);
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

extern "C" bool ops_copy(void* dst, const void* src, uint64_t bytes, void* stream) {
  constexpr uint64_t kWord = sizeof(uint4);
  if (!dst || !src || !bytes || (uintptr_t) dst % kWord || (uintptr_t) src % kWord) return false;
  const uint64_t words = bytes / kWord;
  const uint32_t tail = (uint32_t) (bytes % kWord);
  copy_kernel<<<grid(words + tail), kThreads, 0, st(stream)>>>((uint4*) dst, (const uint4*) src, words,
                                                                (uint8_t*) dst + words * kWord,
                                                                (const uint8_t*) src + words * kWord, tail);
  return ok();
}

extern "C" bool ops_wait(const uint32_t* flag, uint32_t value, void* stream) {
  if (!flag) return false;
  wait_kernel<<<1, 1, 0, st(stream)>>>(flag, value);
  return ok();
}

extern "C" bool ops_copy_rows(float* dst, const float* src, const int32_t* idx, uint32_t n, uint32_t len,
                              void* stream) {
  if (!dst || !src || !idx || !n || !len) return false;
  copy_rows_kernel<<<grid((uint64_t) n * len), kThreads, 0, st(stream)>>>(dst, src, idx, n, len);
  return ok();
}

extern "C" bool ops_hc_post_dot(const float* y, const float* res, const float* post, const float* comb, uint32_t hc,
                                uint32_t dim, uint32_t n_tok, float* out, const float* fn, float* partial,
                                void* stream) {
  const uint32_t len = hc * dim;
  if (!hc || hc > OPS_MAX_HC || !dim || !n_tok || len % kHcParts) return false;
  if (n_tok >= (uint32_t) kHcTok)  // a tile's worth of tokens: the two launches (the tiled dot)
    return ops_hc_post(y, res, post, comb, hc, dim, n_tok, out, stream) &&
           ops_hc_dot(fn, out, hc, dim, n_tok, partial, stream);
  const hipStream_t s = st(stream);
  const dim3 g(kHcParts, n_tok);
  const size_t lds = (size_t) (len / kHcParts) * sizeof(float);
  switch (hc) {
#define HC_POST_DOT(n) \
  case n: hc_post_dot_kernel<n><<<g, kThreads, lds, s>>>(y, res, post, comb, dim, out, fn, partial); break;
    HC_POST_DOT(1) HC_POST_DOT(2) HC_POST_DOT(3) HC_POST_DOT(4) HC_POST_DOT(5) HC_POST_DOT(6) HC_POST_DOT(7)
    HC_POST_DOT(8)
#undef HC_POST_DOT
  }
  return ok();
}

extern "C" bool ops_hc_mix_pre_rms(const float* partial, const float* scale, const float* base, uint32_t hc,
                                   uint32_t dim, float eps, float hc_eps, uint32_t iters, uint32_t n_tok, float* mpre,
                                   float* post, float* comb, const float* h, const float* pre, const float* w,
                                   float* x, void* x16, int32_t* overflow, void* stream) {
  if (!hc || hc > OPS_MAX_HC || !dim || !iters || !n_tok || !pre || !w || !x || (!x16) != (!overflow)) return false;
  if (x16 && (dim % mm::kTile || n_tok > MM_MAX_TOKENS)) return false;
  const hipStream_t s = st(stream);
  const uint32_t blocks = x16 ? tiles(n_tok) : n_tok;
  switch (hc) {
#define HC_MIX_PRE(n)                                                                                              \
  case n:                                                                                                          \
    hc_mix_pre_rms_kernel<n><<<blocks, kWide, 0, s>>>(partial, scale, base, eps, hc_eps, iters, mpre, post, comb, h, \
                                                      pre, dim, w, x, n_tok, (_Float16*) x16, overflow);           \
    break;
    HC_MIX_PRE(1) HC_MIX_PRE(2) HC_MIX_PRE(3) HC_MIX_PRE(4) HC_MIX_PRE(5) HC_MIX_PRE(6) HC_MIX_PRE(7) HC_MIX_PRE(8)
#undef HC_MIX_PRE
  }
  return ok();
}
