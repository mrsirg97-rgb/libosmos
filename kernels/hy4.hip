// Hy4-flash's small steps (include/osmos/gpu/hy4_ops.h): iHC and Engram's ends, the reference's order in F32
#include <cmath>

#include <hip/hip_fp16.h>

#include "port.h"

#include "block.h"
#include "launch.h"
#include "round.h"

extern "C" {
#include "osmos/gpu/hy4_ops.h"
#include "osmos/gpu/mv.h"
#include "osmos/model/qtype.h"
}

namespace {

using launches::ok;
using launches::st;

constexpr int kThreads = 256;
constexpr int kRowThreads = 128;               // an Engram row's values, a thread each (a row at most this)
constexpr float kFp8Max = 448.0f;              // torch.finfo(float8_e4m3fn).max
constexpr float kAmaxFloor = 1e-12f;           // _fp8_round_trip's clamp(min=1e-12)
constexpr float kGateFloor = 1e-6f;            // the gate's clamp_min(1e-6) before its square root

// iHC's mixes in two launches: each token's dots with fn's rows and its sum of squares over kHcParts fixed slices of
// its flat streams (a block a slice: fn read by every block at once, not one), then a block a token sums the slices
// in order - pre, post, the pre-weighted sum, the sublayer's RMSNorm. the slices fixed: a token's bits whatever the step
constexpr uint32_t kHcParts = 32;

template <uint32_t rows>
__global__ __launch_bounds__(kThreads) void hc_dot_kernel(const float* h, const float* fn, uint32_t len,
                                                          float* partial) {
  __shared__ float lds[kThreads];
  const uint32_t p = blockIdx.x, t = blockIdx.y, seg = len / kHcParts, i0 = p * seg;
  const float* ht = h + (uint64_t) t * len;
  float dot[rows] = {}, sq = 0;
  for (uint32_t i = i0 + threadIdx.x; i < i0 + seg; i += kThreads) {
    const float v = ht[i];
    sq = fmaf(v, v, sq);
#pragma unroll
    for (uint32_t r = 0; r < rows; r++) dot[r] = fmaf(fn[(uint64_t) r * len + i], v, dot[r]);
  }
  float* out = partial + (uint64_t) t * (rows + 1) * kHcParts;
#pragma unroll
  for (uint32_t r = 0; r < rows; r++) {
    const float d = block_sum<kThreads>(dot[r], lds);
    if (threadIdx.x == 0) out[r * kHcParts + p] = d;
  }
  sq = block_sum<kThreads>(sq, lds);
  if (threadIdx.x == 0) out[rows * kHcParts + p] = sq;
}

template <uint32_t hc, uint32_t rows>
__global__ __launch_bounds__(kThreads) void hc_pre_kernel(const float* h, const float* partial, const float* scale,
                                                         const float* base, uint32_t dim, float eps, float hc_eps,
                                                         float magnitude, const float* norm_w, float* x,
                                                         float* post) {
  __shared__ float lds[kThreads];
  __shared__ float pre[hc], r_norm;
  const uint32_t t = blockIdx.x, len = hc * dim;
  const float* ht = h + (uint64_t) t * len;
  if (threadIdx.x == 0) {
    const float* pt = partial + (uint64_t) t * (rows + 1) * kHcParts;
    float sum[rows + 1];
    for (uint32_t r = 0; r <= rows; r++) {
      float a = 0;
      for (uint32_t p = 0; p < kHcParts; p++) a += pt[r * kHcParts + p];
      sum[r] = a;
    }
    const float inv = 1.0f / sqrtf(sum[rows] / (float) len + eps);
    for (uint32_t k = 0; k < hc; k++) pre[k] = sigmoid(sum[k] * inv * scale[0] + base[k]) + hc_eps;
    if constexpr (rows == 2 * hc)
      for (uint32_t k = 0; k < hc; k++)
        post[t * hc + k] = magnitude * sigmoid(sum[hc + k] * inv * scale[1] + base[hc + k]) + hc_eps;
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
                                                          const int32_t* pos, float* cache, uint32_t slots) {
  __shared__ float lds[kThreads];
  const uint32_t t = blockIdx.x, row = latent + rope_dim;
  const float* in = kva + (uint64_t) t * row;
  float* out = cache + (uint64_t) (slots ? (uint32_t) pos[t] % slots : (uint32_t) pos[t]) * row;
  float ss = 0;
  for (uint32_t d = threadIdx.x; d < latent; d += kThreads) ss = fmaf(in[d], in[d], ss);
  ss = block_sum<kThreads>(ss, lds);
  const float r = 1.0f / sqrtf(ss / (float) latent + eps);
  for (uint32_t d = threadIdx.x; d < latent; d += kThreads) out[d] = norm_w[d] * (in[d] * r);
  for (uint32_t i = threadIdx.x; i < rope_dim / 2; i += kThreads) rope_pair(in + latent, out + latent, i, pos[t], freqs);
}

// the per-head matvecs' table (mv_group_tok's): entry e = token t, head h (e = t heads + h) - head h's matrix,
// one row of x and of y each
__global__ void head_table_kernel(const uint8_t* base, uint64_t head_bytes, uint32_t heads, uint32_t n,
                                  const void** w, int32_t* cnt, int32_t* xrow) {
  const uint32_t e = blockIdx.x * kThreads + threadIdx.x;
  if (e >= n) return;
  w[e] = base + (uint64_t) (e % heads) * head_bytes, cnt[e] = 1, xrow[e] = (int32_t) e;
}

// each head's rope part of q (its last rope_dim) roped at pos[t] into qa's (its last rope_dim)
__global__ void q_rope_kernel(const float* q, uint32_t heads, uint32_t qk, uint32_t row, uint32_t rope_dim,
                              const float* freqs, const int32_t* pos, float* qa) {
  const uint32_t h = blockIdx.x, t = blockIdx.y;
  const float* in = q + ((uint64_t) t * heads + h) * qk + qk - rope_dim;
  float* out = qa + ((uint64_t) t * heads + h) * row + row - rope_dim;
  for (uint32_t i = threadIdx.x; i < rope_dim / 2; i += blockDim.x) rope_pair(in, out, i, pos[t], freqs);
}

// out *= sigmoid(gate), elementwise
__global__ void gate_kernel(float* out, const float* gate, uint64_t n) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) out[i] = out[i] * sigmoid(gate[i]);
}

// MLA's attention, flash decoding: absorbed MLA is MQA (every head reads the same rows), so a block takes a group of
// kHeadGroup heads (a wave each) over a split of kSplitKeys of its token's keys: the rows loaded once into LDS a tile
// of kTileKeys at a time for every head, each wave its head's scores (a lane 18 of the 576 dims, an xor tree), an
// online softmax in the keys' order, its latent output (a lane 16 of the 512 dims). partials (max, sum, output) a
// (token, head, split); the merge their order with the sink. the splits a token's own keys': its bits whatever the step
constexpr uint32_t kHeadGroup = kWaves, kSplitKeys = 64, kTileKeys = 16, kLanes = 32;
constexpr uint32_t kRowMax = HY4_MLA_MAX_LATENT + kMaxRope, kTokChunk = 16; /* a launch's tokens (the partials' room) */
constexpr uint32_t kMaxSplits = HY4_MLA_MAX_KEYS / kSplitKeys;

struct Keys {  // a token's keys: positions lo .. lo + n - 1, or its selected groups' tokens
  int64_t lo;
  uint32_t n, pool;
  const int32_t* gs;
  uint32_t slots;
  __device__ int64_t row(uint32_t k) const {
    const int64_t p = gs ? (int64_t) gs[k / pool] * pool + k % pool : lo + (int64_t) k;
    return slots ? p % slots : p;
  }
};

__device__ inline Keys keys_of(uint32_t t, const int32_t* pos, uint32_t window, const int32_t* sel,
                               const int32_t* n_sel, uint32_t sel_cap, uint32_t pool, uint32_t slots) {
  Keys k;
  const int32_t p = pos[t];
  k.lo = window && p >= (int32_t) window ? p - (int32_t) window + 1 : 0;
  k.gs = sel ? sel + (uint64_t) t * sel_cap : nullptr, k.pool = pool, k.slots = slots;
  const uint32_t groups = sel ? (uint32_t) n_sel[t] : 0;
  k.n = sel ? (groups - 1) * pool + (uint32_t) (p - k.gs[groups - 1] * (int32_t) pool + 1) : (uint32_t) (p - k.lo + 1);
  return k;
}

__global__ __launch_bounds__(kThreads) void mla_split_kernel(const float* qa, const float* cache, uint32_t slots,
                                                             uint32_t heads, uint32_t latent, uint32_t rope_dim,
                                                             uint32_t window, const int32_t* sel, const int32_t* n_sel,
                                                             uint32_t sel_cap, uint32_t pool, float scale,
                                                             const int32_t* pos, uint32_t t0, float* part) {
  __shared__ float tile[kTileKeys][kRowMax];
  const uint32_t split = blockIdx.x, g = blockIdx.y, t = t0 + blockIdx.z, row = latent + rope_dim;
  const Keys keys = keys_of(t, pos, window, sel, n_sel, sel_cap, pool, slots);
  const uint32_t k0 = split * kSplitKeys;
  const uint32_t wave = threadIdx.x / kLanes, lane = threadIdx.x % kLanes, h = g * kHeadGroup + wave;
  const bool live = h < heads;
  float* out = part + (((uint64_t) blockIdx.z * heads + (live ? h : 0)) * kMaxSplits + split) * (HY4_MLA_MAX_LATENT + 2);
  if (k0 >= keys.n) {  // a split past the token's keys: empty
    if (live && lane == 0) out[0] = -INFINITY, out[1] = 0.0f;
    return;
  }
  const uint32_t k1 = k0 + kSplitKeys < keys.n ? k0 + kSplitKeys : keys.n;
  constexpr uint32_t kQPer = kRowMax / kLanes, kOPer = HY4_MLA_MAX_LATENT / kLanes;
  float q[kQPer], o[kOPer] = {}, m = -INFINITY, l = 0.0f;
  const float* qh = qa + ((uint64_t) t * heads + (live ? h : 0)) * row;
#pragma unroll
  for (uint32_t j = 0; j < kQPer; j++) q[j] = lane + kLanes * j < row ? qh[lane + kLanes * j] : 0.0f;
  for (uint32_t kt = k0; kt < k1; kt += kTileKeys) {
    const uint32_t nt = k1 - kt < kTileKeys ? k1 - kt : kTileKeys;
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < nt * row; i += kThreads) {
      const uint32_t kk = i / row, d = i % row;
      tile[kk][d] = cache[(uint64_t) keys.row(kt + kk) * row + d];
    }
    __syncthreads();
    if (!live) continue;
    float sc[kTileKeys];
    float mt = m;
    for (uint32_t kk = 0; kk < nt; kk++) {
      float a = 0.0f;
#pragma unroll
      for (uint32_t j = 0; j < kQPer; j++) a = fmaf(q[j], lane + kLanes * j < row ? tile[kk][lane + kLanes * j] : 0.0f, a);
      for (int off = kLanes / 2; off > 0; off >>= 1) a += __shfl_xor(a, off, kLanes);
      sc[kk] = a * scale;
      mt = fmaxf(mt, sc[kk]);
    }
    const float alpha = expf(m - mt);  // (m -inf at the start: 0)
    l *= alpha;
#pragma unroll
    for (uint32_t j = 0; j < kOPer; j++) o[j] *= alpha;
    for (uint32_t kk = 0; kk < nt; kk++) {
      const float p = expf(sc[kk] - mt);
      l += p;
#pragma unroll
      for (uint32_t j = 0; j < kOPer; j++)
        if (lane + kLanes * j < latent) o[j] = fmaf(p, tile[kk][lane + kLanes * j], o[j]);
    }
    m = mt;
  }
  if (!live) return;
  if (lane == 0) out[0] = m, out[1] = l;
#pragma unroll
  for (uint32_t j = 0; j < kOPer; j++)
    if (lane + kLanes * j < latent) out[2 + lane + kLanes * j] = o[j];
}

// a (token, head)'s splits merged in their order with the sink (a logit with no value)
__global__ __launch_bounds__(kThreads) void mla_merge_kernel(const float* part, const float* sinks, uint32_t heads,
                                                             uint32_t latent, uint32_t window, const int32_t* sel,
                                                             const int32_t* n_sel, uint32_t sel_cap, uint32_t pool,
                                                             const int32_t* pos, uint32_t t0, float* o_lat) {
  const uint32_t h = blockIdx.x, t = t0 + blockIdx.y;
  const Keys keys = keys_of(t, pos, window, sel, n_sel, sel_cap, pool, 0);
  const uint32_t splits = (keys.n + kSplitKeys - 1) / kSplitKeys;
  const float* pt = part + ((uint64_t) blockIdx.y * heads + h) * kMaxSplits * (HY4_MLA_MAX_LATENT + 2);
  float m = sinks[h];
  for (uint32_t s = 0; s < splits; s++) m = fmaxf(m, pt[s * (HY4_MLA_MAX_LATENT + 2)]);
  float l = expf(sinks[h] - m);
  for (uint32_t s = 0; s < splits; s++) l += pt[s * (HY4_MLA_MAX_LATENT + 2) + 1] * expf(pt[s * (HY4_MLA_MAX_LATENT + 2)] - m);
  float* out = o_lat + ((uint64_t) t * heads + h) * latent;
  for (uint32_t d = threadIdx.x; d < latent; d += kThreads) {
    float acc = 0.0f;
    for (uint32_t s = 0; s < splits; s++) {
      const float* ps = pt + s * (HY4_MLA_MAX_LATENT + 2);
      acc = fmaf(ps[2 + d], expf(ps[0] - m), acc);
    }
    out[d] = acc / l;
  }
}

// the compressed indexer. a group's keys pooled: a thread a dimension (dim at most kIxDim)
constexpr uint32_t kIxDim = 128, kIxPoolMax = 8;

__global__ __launch_bounds__(kIxDim) void ix_pool_kernel(const float* keys, const uint8_t* w_value,
                                                         const uint8_t* w_gate, const float* ape, const float* norm_w,
                                                         uint32_t dim, uint32_t pool, uint32_t rope_dim, float eps,
                                                         const float* freqs, uint32_t g0, float* groups) {
  __shared__ float ks[kIxPoolMax][kIxDim], out[kIxDim], lds[kIxDim];
  const uint32_t g = g0 + blockIdx.x, i = threadIdx.x;
  const bool live = i < dim;
  for (uint32_t s = 0; s < pool; s++) ks[s][i] = live ? keys[((uint64_t) g * pool + s) * dim + i] : 0.0f;
  __syncthreads();
  float v[kIxPoolMax], gt[kIxPoolMax], m = -INFINITY;
  const uint64_t row_bytes = dim / kQ8Block * kQ8Bytes;
  for (uint32_t s = 0; live && s < pool; s++) {
    float av = 0, ag = 0;
    for (uint32_t j = 0; j < dim; j++) {
      av = fmaf(q8_value(w_value + i * row_bytes, j), ks[s][j], av);
      ag = fmaf(q8_value(w_gate + i * row_bytes, j), ks[s][j], ag);
    }
    v[s] = av, gt[s] = ag + ape[(uint64_t) s * dim + i];
    m = fmaxf(m, gt[s]);
  }
  float z = 0, x = 0;
  for (uint32_t s = 0; live && s < pool; s++) z += (gt[s] = expf(gt[s] - m));
  for (uint32_t s = 0; live && s < pool; s++) x += v[s] * (gt[s] / z);
  const float ss = block_sum<kIxDim>(live ? x * x : 0.0f, lds);
  const float r = 1.0f / sqrtf(ss / (float) dim + eps);
  __shared__ float roped[kIxDim];
  if (live) out[i] = roped[i] = norm_w[i] * (x * r);
  __syncthreads();
  if (freqs && i < rope_dim / 2) rope_pair(out + dim - rope_dim, roped + dim - rope_dim, i, (int32_t) (g * pool), freqs);
  __syncthreads();
  if (live) groups[(uint64_t) g * dim + i] = roped[i];
}

__global__ __launch_bounds__(kThreads) void ix_query_kernel(float* q, float* w, uint32_t heads, uint32_t dim,
                                                            uint32_t rope_dim, const float* freqs, const int32_t* pos,
                                                            float scale) {
  const uint32_t t = blockIdx.x, half = rope_dim / 2;
  for (uint32_t j = threadIdx.x; j < heads * half; j += kThreads) {
    float* r = q + ((uint64_t) t * heads + j / half) * dim + dim - rope_dim;
    float pair[2] = {r[2 * (j % half)], r[2 * (j % half) + 1]}, rot[2];
    const float a = (float) pos[t] * freqs[j % half], c = cosf(a), s = sinf(a);
    rot[0] = pair[0] * c + (-pair[1]) * s, rot[1] = pair[1] * c + pair[0] * s;
    r[2 * (j % half)] = rot[0], r[2 * (j % half) + 1] = rot[1];
  }
  for (uint32_t h = threadIdx.x; h < heads; h += kThreads) w[(uint64_t) t * heads + h] *= scale;
}

// a float's order as an unsigned key (the greater float, the greater key)
__device__ inline uint32_t order_key(float x) {
  const uint32_t u = __float_as_uint(x);
  return u & 0x80000000u ? ~u : u | 0x80000000u;
}

// a block's exclusive prefix sum of each thread's v (a fixed Hillis-Steele scan), and the total
__device__ inline uint32_t block_scan(uint32_t v, uint32_t* lds, uint32_t* total) {
  lds[threadIdx.x] = v;
  __syncthreads();
  for (uint32_t off = 1; off < kThreads; off <<= 1) {
    const uint32_t add = threadIdx.x >= off ? lds[threadIdx.x - off] : 0;
    __syncthreads();
    lds[threadIdx.x] += add;
    __syncthreads();
  }
  const uint32_t incl = lds[threadIdx.x];
  *total = lds[kThreads - 1];
  __syncthreads();
  return incl - v;
}

// a block a token: its complete groups' scores (a thread a group, the heads in order), then the top - 1 greatest by
// the scores' bits (the k-th greatest key bit by bit, then those over it and its ties from the lowest group),
// written ascending, its own last
__global__ __launch_bounds__(kThreads) void ix_select_kernel(const float* q, const float* w, const float* groups,
                                                             uint32_t heads, uint32_t dim, uint32_t pool, uint32_t top,
                                                             const int32_t* pos, uint32_t max_groups, float* scores,
                                                             int32_t* sel, int32_t* n_sel) {
  __shared__ float qs[kIxDim * 16], ws[16];
  __shared__ uint32_t lds[kThreads];
  const uint32_t t = blockIdx.x, own = (uint32_t) pos[t] / pool;
  int32_t* out = sel + (uint64_t) t * top;
  if (own < top) {  // every complete group, and its own
    for (uint32_t g = threadIdx.x; g <= own; g += kThreads) out[g] = (int32_t) g;
    if (threadIdx.x == 0) n_sel[t] = (int32_t) own + 1;
    return;
  }
  for (uint32_t i = threadIdx.x; i < heads * dim; i += kThreads) qs[i] = q[(uint64_t) t * heads * dim + i];
  for (uint32_t h = threadIdx.x; h < heads; h += kThreads) ws[h] = w[(uint64_t) t * heads + h];
  __syncthreads();
  float* sc = scores + (uint64_t) t * max_groups;
  for (uint32_t g = threadIdx.x; g < own; g += kThreads) {
    const float* k = groups + (uint64_t) g * dim;
    float s = 0;
    for (uint32_t h = 0; h < heads; h++) {
      float d = 0;
      for (uint32_t i = 0; i < dim; i++) d = fmaf(qs[h * dim + i], k[i], d);
      s = fmaf(ws[h], fmaxf(d, 0.0f), s);
    }
    sc[g] = s;
  }
  __syncthreads();
  const uint32_t want = top - 1;
  uint32_t prefix = 0, total = 0;
  for (int bit = 31; bit >= 0; bit--) {  // the greatest key V with at least `want` keys >= V
    const uint32_t cand = prefix | (1u << bit);
    uint32_t c = 0;
    for (uint32_t g = threadIdx.x; g < own; g += kThreads) c += order_key(sc[g]) >= cand;
    block_scan(c, lds, &total);
    if (total >= want) prefix = cand;
  }
  uint32_t over = 0;
  {
    uint32_t c = 0;
    for (uint32_t g = threadIdx.x; g < own; g += kThreads) c += order_key(sc[g]) > prefix;
    block_scan(c, lds, &over);
  }
  const uint32_t ties = want - over;
  uint32_t written = 0, tied = 0;
  for (uint32_t g0 = 0; g0 < own; g0 += kThreads) {  // in group order: a chunk's picks placed by a scan
    const uint32_t g = g0 + threadIdx.x;
    const uint32_t key = g < own ? order_key(sc[g]) : 0;
    const uint32_t eq = g < own && key == prefix;
    uint32_t eq_total = 0, take_total = 0;
    const uint32_t eq_before = block_scan(eq, lds, &eq_total);
    const uint32_t take = g < own && (key > prefix || (eq && tied + eq_before < ties));
    const uint32_t at = block_scan(take, lds, &take_total);
    if (take) out[written + at] = (int32_t) g;
    written += take_total, tied += eq_total;
  }
  if (threadIdx.x == 0) out[want] = (int32_t) own, n_sel[t] = (int32_t) top;
}

}  // namespace

extern "C" uint64_t hy4_hc_scratch(uint32_t n_tok) { return (uint64_t) n_tok * (2 * HY4_HC_MAX + 1) * kHcParts; }

extern "C" bool hy4_hc_pre(const float* h, const float* fn, const float* scale, const float* base, uint32_t rows,
                           uint32_t hc, uint32_t dim, float eps, float hc_eps, float magnitude, const float* norm_w,
                           uint32_t n_tok, float* x, float* post, float* scratch, void* stream) {
  if (!h || !fn || !scale || !base || !norm_w || !x || !scratch || !dim || !n_tok || hc != HY4_HC_MAX ||
      (hc * dim) % kHcParts)
    return false;
  if (rows == 2 * hc ? !post : rows != hc) return false;
  const hipStream_t s = st(stream);
  const uint32_t len = hc * dim;
  if (rows == 2 * hc) {
    hc_dot_kernel<2 * HY4_HC_MAX><<<dim3(kHcParts, n_tok), kThreads, 0, s>>>(h, fn, len, scratch);
    hc_pre_kernel<HY4_HC_MAX, 2 * HY4_HC_MAX><<<n_tok, kThreads, 0, s>>>(h, scratch, scale, base, dim, eps, hc_eps,
                                                                         magnitude, norm_w, x, post);
  } else {
    hc_dot_kernel<HY4_HC_MAX><<<dim3(kHcParts, n_tok), kThreads, 0, s>>>(h, fn, len, scratch);
    hc_pre_kernel<HY4_HC_MAX, HY4_HC_MAX><<<n_tok, kThreads, 0, s>>>(h, scratch, scale, base, dim, eps, hc_eps,
                                                                     magnitude, norm_w, x, nullptr);
  }
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
                           const float* freqs, const int32_t* pos, uint32_t n_tok, float* cache, uint32_t slots,
                           void* stream) {
  if (!kva || !norm_w || !freqs || !pos || !cache || !latent || !rope_dim || rope_dim % 2 || !n_tok) return false;
  mla_kv_kernel<<<n_tok, kThreads, 0, st(stream)>>>(kva, norm_w, latent, rope_dim, eps, freqs, pos, cache, slots);
  return ok();
}

/* the table's entries, chunked under mv_group_tok's most */
static bool heads_mv(const void* base, uint64_t head_bytes, uint32_t heads, uint32_t n_tok, uint32_t rows,
                     uint32_t cols, const float* x, uint32_t xs, float* y, uint32_t ys, const void** w, int32_t* cnt,
                     int32_t* xrow, hipStream_t s) {
  const uint32_t n = heads * n_tok;
  head_table_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, s>>>((const uint8_t*) base, head_bytes, heads, n, w,
                                                                       cnt, xrow);
  if (!ok()) return false;
  for (uint32_t e = 0; e < n; e += MV_MAX_GROUP)
    if (!mv_group_tok(QTYPE_Q8_0, w + e, cnt + e, xrow + e, n - e < MV_MAX_GROUP ? n - e : MV_MAX_GROUP, 1, rows, cols,
                      x, xs, y, ys, s))
      return false;
  return true;
}

extern "C" uint64_t hy4_mla_table_bytes(uint32_t n_tok, uint32_t heads) {
  return (uint64_t) n_tok * heads * (sizeof(void*) + 2 * sizeof(int32_t));
}

static void table_of(void* table, uint32_t n_tok, uint32_t heads, const void*** w, int32_t** cnt, int32_t** xrow) {
  const uint64_t n = (uint64_t) n_tok * heads;
  *w = (const void**) table, *cnt = (int32_t*) ((char*) table + n * sizeof(void*)), *xrow = *cnt + n;
}

extern "C" bool hy4_mla_q(const float* q, const void* k_b, uint32_t heads, uint32_t nope, uint32_t latent,
                          uint32_t rope_dim, const float* freqs, const int32_t* pos, uint32_t n_tok, float* qa,
                          void* table, void* stream) {
  if (!q || !k_b || !freqs || !pos || !qa || !table || !heads || !nope || nope % kQ8Block || !latent || !rope_dim ||
      rope_dim % 2 || (nope + rope_dim) % 8 || !n_tok)
    return false;
  const hipStream_t s = st(stream);
  const void** w;
  int32_t *cnt, *xrow;
  table_of(table, n_tok, heads, &w, &cnt, &xrow);
  const uint32_t qk = nope + rope_dim, row = latent + rope_dim;
  if (!heads_mv(k_b, (uint64_t) latent * nope / kQ8Block * kQ8Bytes, heads, n_tok, latent, nope, q, qk, qa, row, w,
                cnt, xrow, s))
    return false;
  q_rope_kernel<<<dim3(heads, n_tok), 64, 0, s>>>(q, heads, qk, row, rope_dim, freqs, pos, qa);
  return ok();
}

extern "C" uint64_t hy4_mla_attn_scratch(uint32_t heads) {
  return (uint64_t) kTokChunk * heads * kMaxSplits * (HY4_MLA_MAX_LATENT + 2);
}

extern "C" bool hy4_mla_attn(const float* qa, const float* cache, uint32_t slots, const float* sinks, uint32_t heads,
                             uint32_t latent, uint32_t rope_dim, uint32_t window, const int32_t* sel,
                             const int32_t* n_sel, uint32_t sel_cap, uint32_t pool, float scale, const int32_t* pos,
                             uint32_t n_tok, float* o_lat, float* scratch, void* stream) {
  if (!qa || !cache || !sinks || !pos || !o_lat || !scratch || !heads || !latent || latent > HY4_MLA_MAX_LATENT ||
      !rope_dim || rope_dim > kMaxRope || !n_tok || (window && window > HY4_MLA_MAX_KEYS))
    return false;
  if (sel && (!n_sel || window || !pool || !sel_cap || (uint64_t) sel_cap * pool > HY4_MLA_MAX_KEYS)) return false;
  if (slots && slots < window) return false;
  const hipStream_t s = st(stream);
  const uint32_t splits = (window ? window : HY4_MLA_MAX_KEYS) / kSplitKeys + 1 < kMaxSplits
                              ? (window ? window : HY4_MLA_MAX_KEYS) / kSplitKeys + 1
                              : kMaxSplits;
  for (uint32_t t0 = 0; t0 < n_tok; t0 += kTokChunk) {
    const uint32_t m = n_tok - t0 < kTokChunk ? n_tok - t0 : kTokChunk;
    mla_split_kernel<<<dim3(splits, (heads + kHeadGroup - 1) / kHeadGroup, m), kThreads, 0, s>>>(
        qa, cache, slots, heads, latent, rope_dim, window, sel, n_sel, sel_cap, pool, scale, pos, t0, scratch);
    mla_merge_kernel<<<dim3(heads, m), kThreads, 0, s>>>(scratch, sinks, heads, latent, window, sel, n_sel, sel_cap,
                                                         pool, pos, t0, o_lat);
    if (!ok()) return false;
  }
  return true;
}

extern "C" bool hy4_mla_out(const float* o_lat, const void* v_b, const float* gate, uint32_t heads, uint32_t latent,
                            uint32_t v_dim, uint32_t n_tok, float* out, void* table, void* stream) {
  if (!o_lat || !v_b || !gate || !out || !table || !heads || !latent || latent % kQ8Block || latent % 8 || !v_dim ||
      v_dim % 8 || !n_tok)
    return false;
  const hipStream_t s = st(stream);
  const void** w;
  int32_t *cnt, *xrow;
  table_of(table, n_tok, heads, &w, &cnt, &xrow);
  if (!heads_mv(v_b, (uint64_t) v_dim * latent / kQ8Block * kQ8Bytes, heads, n_tok, v_dim, latent, o_lat, latent, out,
                v_dim, w, cnt, xrow, s))
    return false;
  const uint64_t n = (uint64_t) n_tok * heads * v_dim;
  gate_kernel<<<(unsigned) ((n + kThreads - 1) / kThreads), kThreads, 0, s>>>(out, gate, n);
  return ok();
}

extern "C" bool hy4_ix_pool(const float* keys, const void* w_value, const void* w_gate, const float* ape,
                            const float* norm_w, uint32_t dim, uint32_t pool, uint32_t rope_dim, float eps,
                            const float* freqs, uint32_t g0, uint32_t n_groups, float* groups, void* stream) {
  if (!keys || !w_value || !w_gate || !ape || !norm_w || !groups || !dim || dim > kIxDim || dim % kQ8Block || !pool ||
      pool > kIxPoolMax || rope_dim % 2 || rope_dim > dim || !n_groups)
    return false;
  ix_pool_kernel<<<n_groups, kIxDim, 0, st(stream)>>>(keys, (const uint8_t*) w_value, (const uint8_t*) w_gate, ape,
                                                      norm_w, dim, pool, rope_dim, eps, freqs, g0, groups);
  return ok();
}

extern "C" bool hy4_ix_query(float* q, float* w, uint32_t heads, uint32_t dim, uint32_t rope_dim, const float* freqs,
                             const int32_t* pos, uint32_t n_tok, void* stream) {
  if (!q || !w || !freqs || !pos || !heads || !dim || rope_dim % 2 || rope_dim > dim || !n_tok) return false;
  const float scale = (float) (1.0 / sqrt((double) heads) * (1.0 / sqrt((double) dim)));
  ix_query_kernel<<<n_tok, kThreads, 0, st(stream)>>>(q, w, heads, dim, rope_dim, freqs, pos, scale);
  return ok();
}

extern "C" bool hy4_ix_select(const float* q, const float* w, const float* groups, uint32_t heads, uint32_t dim,
                              uint32_t pool, uint32_t top, const int32_t* pos, uint32_t n_tok, uint32_t max_groups,
                              float* scores, int32_t* sel, int32_t* n_sel, void* stream) {
  if (!q || !w || !groups || !pos || !scores || !sel || !n_sel || !heads || heads > 16 || !dim || dim > kIxDim ||
      !pool || top < 2 || !n_tok || !max_groups)
    return false;
  ix_select_kernel<<<n_tok, kThreads, 0, st(stream)>>>(q, w, groups, heads, dim, pool, top, pos, max_groups, scores,
                                                       sel, n_sel);
  return ok();
}
