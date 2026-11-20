// Hy4-flash's small steps (include/osmos/gpu/hy4_ops.h): iHC and Engram's ends, the reference's order in F32
#include <cmath>
#include <cstdlib>

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

// a block a slice and kHcTok tokens: the slice's fn rows in LDS once (fn read a block, not a token), each token's
// dots and sum of squares the same thread's terms in the same order, the same block sums - a token's bits its own
constexpr uint32_t kHcTok = 8;
template <uint32_t rows>
__global__ __launch_bounds__(kThreads) void hc_dot_kernel(const float* h, const float* fn, uint32_t len, uint32_t n_tok,
                                                          float* partial) {
  extern __shared__ float fs[];  // [rows][seg]
  __shared__ float lds[kThreads];
  const uint32_t p = blockIdx.x, t0 = blockIdx.y * kHcTok, seg = len / kHcParts, i0 = p * seg;
  for (uint32_t k = threadIdx.x; k < rows * seg; k += kThreads) fs[k] = fn[(uint64_t) (k / seg) * len + i0 + k % seg];
  __syncthreads();
  for (uint32_t t = t0; t < t0 + kHcTok && t < n_tok; t++) {
    const float* ht = h + (uint64_t) t * len;
    float dot[rows] = {}, sq = 0;
    for (uint32_t i = threadIdx.x; i < seg; i += kThreads) {
      const float v = ht[i0 + i];
      sq = fmaf(v, v, sq);
#pragma unroll
      for (uint32_t r = 0; r < rows; r++) dot[r] = fmaf(fs[r * seg + i], v, dot[r]);
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

// a cache row: F32 [latent | rope], or q8: the latent Q8_0 (ggml's blocks: an f16 d = amax / 127, 32 codes
// round(x / d)) then the rope part F32 (vLLM's fp8_ds_mla keeps it wide too) - kQ8Bytes a 32 latent values
__host__ __device__ inline uint64_t mla_row_bytes(uint32_t latent, uint32_t rope_dim, bool q8) {
  return q8 ? (uint64_t) latent / 32 * 34 + (uint64_t) rope_dim * 4 : (uint64_t) (latent + rope_dim) * 4;
}

__global__ __launch_bounds__(kThreads) void mla_kv_kernel(const float* kva, const float* norm_w, uint32_t latent,
                                                          uint32_t rope_dim, float eps, const float* freqs,
                                                          const int32_t* pos, uint8_t* cache, uint32_t slots, bool q8) {
  __shared__ float lds[kThreads];
  __shared__ float lat[HY4_MLA_MAX_LATENT];
  const uint32_t t = blockIdx.x, row = latent + rope_dim;
  const float* in = kva + (uint64_t) t * row;
  uint8_t* out = cache + (uint64_t) (slots ? (uint32_t) pos[t] % slots : (uint32_t) pos[t]) * mla_row_bytes(latent, rope_dim, q8);
  float ss = 0;
  for (uint32_t d = threadIdx.x; d < latent; d += kThreads) ss = fmaf(in[d], in[d], ss);
  ss = block_sum<kThreads>(ss, lds);
  const float r = 1.0f / sqrtf(ss / (float) latent + eps);
  float* rope_out = (float*) (out + (q8 ? (uint64_t) latent / 32 * 34 : (uint64_t) latent * 4));
  if (!q8) {
    for (uint32_t d = threadIdx.x; d < latent; d += kThreads) ((float*) out)[d] = norm_w[d] * (in[d] * r);
  } else {
    for (uint32_t d = threadIdx.x; d < latent; d += kThreads) lat[d] = norm_w[d] * (in[d] * r);
    __syncthreads();
    for (uint32_t b = threadIdx.x; b < latent / 32; b += kThreads) {  // a thread a block, in its values' order
      float amax = 0;
      for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(lat[b * 32 + i]));
      const float dd = amax / 127.0f, id = dd != 0.0f ? 1.0f / dd : 0.0f;
      uint8_t* blk = out + (uint64_t) b * 34;
      const __half h = __float2half(dd);
      *(__half*) blk = h;
      for (int i = 0; i < 32; i++) blk[2 + i] = (uint8_t) (int8_t) roundf(lat[b * 32 + i] * id);
    }
  }
  for (uint32_t i = threadIdx.x; i < rope_dim / 2; i += kThreads) rope_pair(in + latent, rope_out, i, pos[t], freqs);
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
// kSplitKeys, kTileKeys: tools/bench/mla_bench's sweep (2026-11-18, low-power, a decode's 512-window and 1,024-key
// layers): 32 and 8 the fastest of 8..64 and 8..16 (64 us vs 64/16's 81, 16/8's 79)
constexpr uint32_t kHeadGroup = kWaves, kSplitKeys = 32, kTileKeys = 8, kLanes = 32;
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

__global__ __launch_bounds__(kThreads) void mla_split_kernel(const float* qa, const uint8_t* cache, uint32_t slots, bool q8,
                                                             uint32_t heads, uint32_t latent, uint32_t rope_dim,
                                                             uint32_t window, const int32_t* sel, const int32_t* n_sel,
                                                             uint32_t sel_cap, uint32_t pool, float scale,
                                                             const int32_t* pos, uint32_t t0, float* part) {
  __shared__ __attribute__((aligned(16))) float tile[kTileKeys][kRowMax];
  __shared__ int64_t rows[kTileKeys];
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
  // a lane's dims in chunks of 4 (128-bit reads of the tile): the score's 4 x kQ4 of the row, the output's 4 x kO4
  constexpr uint32_t kQ4 = (kRowMax + 4 * kLanes - 1) / (4 * kLanes), kO4 = HY4_MLA_MAX_LATENT / (4 * kLanes);
  float4 q[kQ4], o[kO4];
  float m = -INFINITY, l = 0.0f;
  const float* qh = qa + ((uint64_t) t * heads + (live ? h : 0)) * row;
#pragma unroll
  for (uint32_t j = 0; j < kQ4; j++) {
    const uint32_t d = 4 * (lane + kLanes * j);
    q[j] = d < row ? *(const float4*) (qh + d) : make_float4(0, 0, 0, 0);
  }
#pragma unroll
  for (uint32_t j = 0; j < kO4; j++) o[j] = make_float4(0, 0, 0, 0);
  for (uint32_t kt = k0; kt < k1; kt += kTileKeys) {
    const uint32_t nt = k1 - kt < kTileKeys ? k1 - kt : kTileKeys;
    __syncthreads();
    if (threadIdx.x < nt) rows[threadIdx.x] = keys.row(kt + threadIdx.x);  // a key's cache row once, not a value's
    __syncthreads();
    const uint32_t row4 = row / 4;  // (row a multiple of 4: hy4_mla_attn's check)
    const uint64_t rb = mla_row_bytes(latent, rope_dim, q8);
    if (!q8)
      for (uint32_t i = threadIdx.x; i < nt * row4; i += kThreads) {
        const uint32_t kk = i / row4, d = i % row4;
        *(float4*) &tile[kk][4 * d] = *(const float4*) (cache + (uint64_t) rows[kk] * rb + 16 * d);
      }
    else {  // a row's latent decoded (d times a code: Q8_0's values, exact in F32), its rope part as stored
      const uint32_t lb = latent / 32;
      for (uint32_t i = threadIdx.x; i < nt * lb * 8; i += kThreads) {  // a thread 4 codes of a block
        const uint32_t kk = i / (lb * 8), b = (i / 8) % lb, c = i % 8;
        const uint8_t* blk = cache + (uint64_t) rows[kk] * rb + (uint64_t) b * 34;
        const float d = __half2float(*(const __half*) blk);
        const int8_t* q = (const int8_t*) (blk + 2 + 4 * c);  // (a block 34 bytes: its codes 2-aligned - byte loads)
        *(float4*) &tile[kk][b * 32 + 4 * c] = make_float4(d * q[0], d * q[1], d * q[2], d * q[3]);
      }
      const uint32_t r4 = rope_dim / 4;
      for (uint32_t i = threadIdx.x; i < nt * r4; i += kThreads) {
        const uint32_t kk = i / r4, d = i % r4;
        *(float4*) &tile[kk][latent + 4 * d] = *(const float4*) (cache + (uint64_t) rows[kk] * rb + (uint64_t) lb * 34 + 16 * d);
      }
    }
    __syncthreads();
    if (!live) continue;
    float sc[kTileKeys];
    float mt = m;
    {
      float a[kTileKeys];
#pragma unroll
      for (uint32_t kk = 0; kk < kTileKeys; kk++) {  // each lane's part of every key's dot
        a[kk] = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < kQ4; j++) {
          const uint32_t d = 4 * (lane + kLanes * j);
          if (d < row) {
            const float4 k4 = *(const float4*) &tile[kk][d];
            a[kk] = fmaf(q[j].x, k4.x, a[kk]), a[kk] = fmaf(q[j].y, k4.y, a[kk]);
            a[kk] = fmaf(q[j].z, k4.z, a[kk]), a[kk] = fmaf(q[j].w, k4.w, a[kk]);
          }
        }
      }
      // the lanes' parts summed transposed: each step a lane keeps half its keys (its lane bit's half) and adds its
      // partner's parts of them - kTileKeys - 1 shuffles in all, then the last lanes' sum; lane L holds key
      // (L / (kLanes / kTileKeys)) % kTileKeys's whole dot
      uint32_t live_n = kTileKeys;
#pragma unroll
      for (uint32_t off = kLanes / 2; off >= kLanes / kTileKeys; off >>= 1) {
        const bool upper = lane & off;
        live_n >>= 1;
#pragma unroll
        for (uint32_t i = 0; i < kTileKeys / 2; i++) {
          if (i >= live_n) break;
          const float send = upper ? a[i] : a[i + live_n], keep = upper ? a[i + live_n] : a[i];
          a[i] = keep + __shfl_xor(send, off, kLanes);
        }
      }
#pragma unroll
      for (uint32_t off = kLanes / kTileKeys / 2; off > 0; off >>= 1) a[0] += __shfl_xor(a[0], off, kLanes);
      constexpr uint32_t kSpan = kLanes / kTileKeys;  // lanes a key
      __shared__ float wsc[kWaves][kTileKeys];
      if (lane % kSpan == 0) {
        // lane L's key: its bits from the top step down (the halves kept), lowest step the key's lowest bit
        uint32_t key = 0;
#pragma unroll
        for (uint32_t off = kLanes / 2, bit = kTileKeys / 2; off >= kSpan; off >>= 1, bit >>= 1) key |= lane & off ? bit : 0;
        wsc[wave][key] = a[0];
      }
      __builtin_amdgcn_wave_barrier();
#pragma unroll
      for (uint32_t kk = 0; kk < kTileKeys; kk++) {
        sc[kk] = kk < nt ? wsc[wave][kk] * scale : -INFINITY;
        mt = fmaxf(mt, sc[kk]);
      }
      __builtin_amdgcn_wave_barrier();
    }
    const float alpha = expf(m - mt);  // (m -inf at the start: 0)
    l *= alpha;
#pragma unroll
    for (uint32_t j = 0; j < kO4; j++) o[j].x *= alpha, o[j].y *= alpha, o[j].z *= alpha, o[j].w *= alpha;
    for (uint32_t kk = 0; kk < nt; kk++) {
      const float p = expf(sc[kk] - mt);
      l += p;
#pragma unroll
      for (uint32_t j = 0; j < kO4; j++) {
        const uint32_t d = 4 * (lane + kLanes * j);
        if (d < latent) {
          const float4 v4 = *(const float4*) &tile[kk][d];
          o[j].x = fmaf(p, v4.x, o[j].x), o[j].y = fmaf(p, v4.y, o[j].y);
          o[j].z = fmaf(p, v4.z, o[j].z), o[j].w = fmaf(p, v4.w, o[j].w);
        }
      }
    }
    m = mt;
  }
  if (!live) return;
  if (lane == 0) out[0] = m, out[1] = l;
#pragma unroll
  for (uint32_t j = 0; j < kO4; j++) {
    const uint32_t d = 4 * (lane + kLanes * j);
    if (d < latent) out[2 + d] = o[j].x, out[3 + d] = o[j].y, out[4 + d] = o[j].z, out[5 + d] = o[j].w;
  }
}

// a prompt's MLA on the matrix cores: a block a token and 16 of its heads (absorbed MLA is MQA: one key list), its
// keys 16 at a time f16 in LDS (K [key][dim], and the latent's V^T [dim][key]); a wave 9 of the score's 36 16-dim
// chunks (its Q's A fragments in registers), the four partial tiles summed in LDS in the waves' order (every wave the
// same S), the online softmax a wave's own copy (the same P), then a wave a quarter of the latent's output (8 tiles).
// f16 in, F32 sums - the read's math, not the F32 path's bits; the sink folded at the end (the merge's arithmetic).
// a decode's and a verify's steps (16 tokens or fewer) stay on mla_split_kernel (their bits the reference-checked)
constexpr uint32_t kMlaRow = 576;  // a row at most (Hy4-flash's 512 + 64: the LDS a 2-group block fits)
constexpr uint32_t kMlaQc = (kMlaRow / 16 + 3) / 4, kMlaKeys = 16, kMlaKRow = kMlaRow + 8, kMlaVRow = kMlaKeys;
template <uint32_t G>  // head groups a block (each 16 heads, 4 waves): the key tile loaded once for all
__global__ __launch_bounds__(128 * G) void mla_mm_kernel(const float* qa, const uint8_t* cache, uint32_t slots, bool q8,
                                                     const float* sinks, uint32_t heads, uint32_t latent,
                                                     uint32_t rope_dim, uint32_t window, const int32_t* sel,
                                                     const int32_t* n_sel, uint32_t sel_cap, uint32_t pool, float scale,
                                                     const int32_t* pos, float* o_lat) {
  __shared__ __attribute__((aligned(32))) _Float16 ks[kMlaKeys * kMlaKRow], vt[HY4_MLA_MAX_LATENT * kMlaVRow];
  __shared__ __attribute__((aligned(32))) _Float16 ps[4 * G][16 * kMlaVRow];
  __shared__ float sp[G][4][8][32];
  constexpr uint32_t kT = 128 * G;
  __shared__ int64_t rows[kMlaKeys];
  const uint32_t row = latent + rope_dim, nch = row / 16, t = blockIdx.y;
  const uint32_t wid = threadIdx.x / 32, grp = wid / 4, wave = wid % 4, lane = threadIdx.x % 32;
  const uint32_t h0 = (blockIdx.x * G + grp) * 16;
  const Keys keys = keys_of(t, pos, window, sel, n_sel, sel_cap, pool, slots);
  const uint64_t rb = mla_row_bytes(latent, rope_dim, q8);
  osmos_fa qf[kMlaQc];
#pragma unroll
  for (uint32_t i = 0; i < kMlaQc; i++) {
    const uint32_t c = wave + 4 * i;
    osmos_h16 a = {};
    if (c < nch) {
      const float* qp = qa + ((uint64_t) t * heads + h0 + lane % 16) * row + c * 16;
#pragma unroll
      for (int e = 0; e < 16; e++) a[e] = (_Float16) qp[e];
    }
    qf[i] = a;
  }
  float m[osmos_slots], l[osmos_slots];
#pragma unroll
  for (int r = 0; r < osmos_slots; r++) m[r] = -INFINITY, l[r] = 0.0f;
  const uint32_t ot = latent / 64;  // a wave's output tiles (latent / 4 / 16)
  osmos_f8 o[HY4_MLA_MAX_LATENT / 64] = {};
  // the Q8_0 cache's next tile in registers while this one's computed: a thread its key's (threadIdx.x % 16: the
  // items key-fastest - V^T's writes consecutive halves, K's rows on distinct banks) 4 codes and their block's d an
  // item, and its rope part's float4s
  constexpr uint32_t kQ8Items = kMlaKeys * (HY4_MLA_MAX_LATENT / 32) * 8 / kT, kRopeItems = (kMlaKeys * 16 + kT - 1) / kT;
  const uint32_t lb = latent / 32, r4 = rope_dim / 4, kk = threadIdx.x % kMlaKeys;
  uint32_t cq[kQ8Items];
  _Float16 dq[kQ8Items];
  float4 rp[kRopeItems];
  auto fetch = [&](uint32_t kb) {
    const uint8_t* base = kb + kk < keys.n ? cache + (uint64_t) keys.row(kb + kk) * rb : nullptr;
#pragma unroll
    for (uint32_t j = 0; j < kQ8Items; j++) {
      const uint32_t i = threadIdx.x + kT * j, b = i / kMlaKeys / 8, c = i / kMlaKeys % 8;
      cq[j] = 0, dq[j] = (_Float16) 0.0f;
      if (base && b < lb) {
        const uint8_t* blk = base + (uint64_t) b * 34;
        const uint16_t* w = (const uint16_t*) (blk + 2 + 4 * c);  // (a block 34 bytes: its codes 2-aligned)
        dq[j] = *(const _Float16*) blk, cq[j] = (uint32_t) w[0] | (uint32_t) w[1] << 16;
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < kRopeItems; j++) {
      const uint32_t d = (threadIdx.x + kT * j) / kMlaKeys;
      rp[j] = base && d < r4 ? *(const float4*) (base + (uint64_t) lb * 34 + 16 * d) : make_float4(0, 0, 0, 0);
    }
  };
  if (q8) fetch(0);
  for (uint32_t kb = 0; kb < keys.n; kb += kMlaKeys) {
    const uint32_t nt = keys.n - kb < kMlaKeys ? keys.n - kb : kMlaKeys;
    __syncthreads();
    if (q8) {  // the fetched tile to LDS: a code times its d (exact in F32) to f16
#pragma unroll
      for (uint32_t j = 0; j < kQ8Items; j++) {
        const uint32_t i = threadIdx.x + kT * j, b = i / kMlaKeys / 8, c = i / kMlaKeys % 8, d0 = b * 32 + 4 * c;
        if (b >= lb) continue;
        const float d = (float) dq[j];
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const _Float16 v = (_Float16) (d * (float) (int8_t) (cq[j] >> (8 * e)));
          ks[kk * kMlaKRow + d0 + e] = v, vt[(d0 + e) * kMlaVRow + kk] = v;
        }
      }
#pragma unroll
      for (uint32_t j = 0; j < kRopeItems; j++) {
        const uint32_t d = (threadIdx.x + kT * j) / kMlaKeys;
        if (d >= r4) continue;
        _Float16* kp = ks + kk * kMlaKRow + latent + 4 * d;
        kp[0] = (_Float16) rp[j].x, kp[1] = (_Float16) rp[j].y, kp[2] = (_Float16) rp[j].z, kp[3] = (_Float16) rp[j].w;
      }
      __syncthreads();
      if (kb + kMlaKeys < keys.n) fetch(kb + kMlaKeys);  // the next tile's loads in flight under this one's math
    } else {
      if (threadIdx.x < kMlaKeys) rows[threadIdx.x] = threadIdx.x < nt ? keys.row(kb + threadIdx.x) : -1;
      __syncthreads();
      const uint32_t row4 = row / 4;
      for (uint32_t i = threadIdx.x; i < kMlaKeys * row4; i += kT) {
        const uint32_t k2 = i / row4, d = 4 * (i % row4);
        float4 x = make_float4(0, 0, 0, 0);
        if (rows[k2] >= 0) x = *(const float4*) (cache + (uint64_t) rows[k2] * rb + 4 * d);
        const _Float16 v[4] = {(_Float16) x.x, (_Float16) x.y, (_Float16) x.z, (_Float16) x.w};
#pragma unroll
        for (int e = 0; e < 4; e++) {
          ks[k2 * kMlaKRow + d + e] = v[e];
          if (d + e < latent) vt[(d + e) * kMlaVRow + k2] = v[e];
        }
      }
      __syncthreads();
    }
    osmos_f8 sacc = {};
#pragma unroll
    for (uint32_t i = 0; i < kMlaQc; i++) {
      const uint32_t c = wave + 4 * i;
      if (c < nch) sacc = osmos_mma(qf[i], osmos_load_b_s(ks + c * 16, kMlaKRow), sacc);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) sp[grp][wave][j][lane] = sacc[j];
    __syncthreads();
    float sv[8], alpha[osmos_slots];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const float s4 = ((sp[grp][0][j][lane] + sp[grp][1][j][lane]) + sp[grp][2][j][lane]) + sp[grp][3][j][lane];
      sv[j] = (uint32_t) osmos_acc_col(j) < nt ? s4 * scale : -INFINITY;
    }
#pragma unroll
    for (int r = 0; r < osmos_slots; r++) {
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < 8; j++)
        if (osmos_slot(j) == r) mx = fmaxf(mx, sv[j]);
      const float mn = fmaxf(m[r], osmos_row_max(mx));
      float sum = 0.0f;
#pragma unroll
      for (int j = 0; j < 8; j++)
        if (osmos_slot(j) == r) {
          const float pv = mn == -INFINITY ? 0.0f : expf(sv[j] - mn);
          sum += pv;
          ps[wid][osmos_acc_row(j) * kMlaVRow + osmos_acc_col(j)] = (_Float16) pv;
        }
      alpha[r] = mn == -INFINITY ? 1.0f : expf(m[r] - mn);
      l[r] = l[r] * alpha[r] + osmos_row_sum(sum), m[r] = mn;
    }
    osmos_wave_barrier();
    const osmos_fa pa = osmos_load_a(ps[wid], kMlaVRow);
#pragma unroll
    for (uint32_t n = 0; n < HY4_MLA_MAX_LATENT / 64; n++) {
      if (n >= ot) break;
#pragma unroll
      for (int j = 0; j < 8; j++) o[n][j] *= alpha[osmos_slot(j)];
      o[n] = osmos_mma(pa, osmos_load_b_s(vt + (wave * ot + n) * 16 * kMlaVRow, kMlaVRow), o[n]);
    }
    osmos_wave_barrier();
  }
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const uint32_t r = (uint32_t) osmos_acc_row(j), h = h0 + r;
    const float mm = m[osmos_slot(j)], big = fmaxf(mm, sinks[h]), f = expf(mm - big);
    const float inv = 1.0f / (l[osmos_slot(j)] * f + expf(sinks[h] - big));
    float* out = o_lat + ((uint64_t) t * heads + h) * latent;
#pragma unroll
    for (uint32_t n = 0; n < HY4_MLA_MAX_LATENT / 64; n++) {
      if (n >= ot) break;
      out[(wave * ot + n) * 16 + osmos_acc_col(j)] = o[n][j] * f * inv;
    }
  }
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
                                                         const float* freqs, uint32_t g0, const int32_t* pos,
                                                         float* groups, uint32_t slots) {
  __shared__ float ks[kIxPoolMax][kIxDim], out[kIxDim], lds[kIxDim];
  // pos given: the group this block's token completes, or none (a whole block's return: no barrier passed)
  if (pos && ((uint32_t) pos[blockIdx.x] + 1) % pool) return;
  const uint32_t g = pos ? (uint32_t) pos[blockIdx.x] / pool : g0 + blockIdx.x, i = threadIdx.x;
  const bool live = i < dim;
  for (uint32_t s = 0; s < pool; s++) {  // (slots: the keys a ring, a position's row its place mod slots)
    const uint64_t p = (uint64_t) g * pool + s;
    ks[s][i] = live ? keys[(slots ? p % slots : p) * dim + i] : 0.0f;
  }
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
constexpr uint32_t kIxSpreadTokens = 16;  // a step's tokens at most whose scores ix_score_kernel spreads

// each token's groups' scores, kIxTile groups a block: their keys loaded once into LDS (coalesced - a thread a group
// reading its own key once a head read 512-byte strides 16 times: 113 ms a decode step at 128K, 2026-11-19), a
// thread a (group, head)'s dot over the dims in their order, then a group's sum over the heads in theirs: score_g =
// sum_h w_h relu(q_h . g) - the bits the select kernel's own loop gives
constexpr uint32_t kIxTile = 32;
constexpr uint32_t kIxRow = kIxDim + 4;  // a key's and a query's row in LDS: whole float4s, the 16 heads' rows 4 banks apart
__global__ __launch_bounds__(kThreads) void ix_score_kernel(const float* q, const float* w, const float* groups,
                                                            uint32_t heads, uint32_t dim, uint32_t pool, uint32_t top,
                                                            const int32_t* pos, uint32_t n_tok, uint32_t max_groups,
                                                            float* scores) {
  // a tile's keys once for every token of the step (a verify's 6 read them once, not 6 times), each token's query in
  // turn
  // (qs's rows padded too: a wave's 32 threads are 2 groups x 16 heads, so unpadded its heads' rows 512 bytes apart
  // read one bank - a 16-way conflict every term; 1M, a token: 5.45 ms a layer before)
  __shared__ __attribute__((aligned(16))) float qs[kIxRow * 16], ks[kIxTile][kIxRow];
  __shared__ float ws[16], ds[kIxTile][16];
  const uint32_t g0 = blockIdx.x * kIxTile;
  uint32_t most = 0;
  for (uint32_t t = 0; t < n_tok; t++) {
    const uint32_t own = (uint32_t) pos[t] / pool;
    if (own >= top && own > most) most = own;
  }
  if (g0 >= most) return;
  const uint32_t nk = most - g0 < kIxTile ? most - g0 : kIxTile;
  for (uint32_t i = threadIdx.x; i < nk * dim; i += kThreads) ks[i / dim][i % dim] = groups[(uint64_t) (g0 + i / dim) * dim + i % dim];
  for (uint32_t t = 0; t < n_tok; t++) {
    const uint32_t own = (uint32_t) pos[t] / pool;
    if (own < top || g0 >= own) continue;
    const uint32_t n = own - g0 < kIxTile ? own - g0 : kIxTile;
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < heads * dim; i += kThreads) qs[i / dim * kIxRow + i % dim] = q[(uint64_t) t * heads * dim + i];
    for (uint32_t h = threadIdx.x; h < heads; h += kThreads) ws[h] = w[(uint64_t) t * heads + h];
    __syncthreads();
    for (uint32_t e = threadIdx.x; e < n * heads; e += kThreads) {
      const uint32_t gl = e / heads, h = e % heads;
      // 4 sums a dot, the terms i % 4 apart, added (0 + 1) + (2 + 3): a dependent chain a quarter long (the scores
      // latency-bound, not the keys' bytes: at 1M a token 1.6 ms a layer against ~0.6) - every step's the same
      float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
      const float4 *qh = (const float4*) (qs + h * kIxRow), *kg = (const float4*) ks[gl];
      for (uint32_t i = 0; i < dim / 4; i++) {
        const float4 a = qh[i], b = kg[i];
        d0 = fmaf(a.x, b.x, d0), d1 = fmaf(a.y, b.y, d1), d2 = fmaf(a.z, b.z, d2), d3 = fmaf(a.w, b.w, d3);
      }
      ds[gl][h] = (d0 + d1) + (d2 + d3);
    }
    __syncthreads();
    if (threadIdx.x < n) {
      float sc = 0;
      for (uint32_t h = 0; h < heads; h++) sc = fmaf(ws[h], fmaxf(ds[threadIdx.x][h], 0.0f), sc);
      scores[(uint64_t) t * max_groups + g0 + threadIdx.x] = sc;
    }
  }
}

// the same scores (each dot's 4 sums, the heads' fmaf in order: ix_score_kernel's bits) from a few blocks each over
// many tiles - the step's queries in LDS once a block (4 tokens a pass), the next tile's keys loaded into registers
// under this one's math. ix_score_kernel's a block a tile: its keys' load, then the queries' (8 KB a token, read by
// every one of 8,192 blocks at 1M), then the math, serial - latency-bound at ~1.6 ms a layer a token against ~0.6 of
// bytes
constexpr uint32_t kIxPTok = 4, kIxPBlocks = 160, kIxPLoads = kIxTile * kIxDim / 4 / kThreads;
__global__ __launch_bounds__(kThreads) void ix_score_persist_kernel(const float* q, const float* w, const float* groups,
                                                                    uint32_t pool, uint32_t top, const int32_t* pos,
                                                                    uint32_t n_tok, uint32_t max_groups,
                                                                    float* scores) {
  __shared__ __attribute__((aligned(16))) float qs[kIxPTok][16 * kIxRow], ks[kIxTile][kIxRow];
  __shared__ float ws[kIxPTok][16], ds[kIxTile][16];
  uint32_t most = 0;
  for (uint32_t t = 0; t < n_tok; t++) {
    const uint32_t own = (uint32_t) pos[t] / pool;
    if (own >= top && own > most) most = own;
  }
  const uint32_t tiles = (most + kIxTile - 1) / kIxTile;
  if (blockIdx.x >= tiles) return;
  auto fetch = [&](float4* r, uint32_t tile) {
#pragma unroll
    for (uint32_t j = 0; j < kIxPLoads; j++) {
      const uint32_t i = threadIdx.x + kThreads * j, g = tile * kIxTile + i / (kIxDim / 4);
      r[j] = g < most ? ((const float4*) (groups + (uint64_t) g * kIxDim))[i % (kIxDim / 4)] : make_float4(0, 0, 0, 0);
    }
  };
  for (uint32_t t0 = 0; t0 < n_tok; t0 += kIxPTok) {
    const uint32_t nt = n_tok - t0 < kIxPTok ? n_tok - t0 : kIxPTok;
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < nt * 16 * kIxDim; i += kThreads) {
      const uint32_t tt = i / (16 * kIxDim), r = i % (16 * kIxDim);
      qs[tt][r / kIxDim * kIxRow + r % kIxDim] = q[(uint64_t) (t0 + tt) * 16 * kIxDim + r];
    }
    for (uint32_t i = threadIdx.x; i < nt * 16; i += kThreads) ws[i / 16][i % 16] = w[(uint64_t) t0 * 16 + i];
    float4 pre[kIxPLoads];
    fetch(pre, blockIdx.x);
    for (uint32_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
      __syncthreads();
#pragma unroll
      for (uint32_t j = 0; j < kIxPLoads; j++) {
        const uint32_t i = threadIdx.x + kThreads * j;
        *(float4*) &ks[i / (kIxDim / 4)][4 * (i % (kIxDim / 4))] = pre[j];
      }
      __syncthreads();
      if (tile + gridDim.x < tiles) fetch(pre, tile + gridDim.x);
      const uint32_t g0 = tile * kIxTile;
      for (uint32_t tt = 0; tt < nt; tt++) {
        const uint32_t own = (uint32_t) pos[t0 + tt] / pool;
        if (own < top || g0 >= own) continue;  // (block-uniform)
        const uint32_t n = own - g0 < kIxTile ? own - g0 : kIxTile;
        for (uint32_t e = threadIdx.x; e < n * 16; e += kThreads) {
          const uint32_t gl = e / 16, h = e % 16;
          float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
          const float4 *qh = (const float4*) (qs[tt] + h * kIxRow), *kg = (const float4*) ks[gl];
          for (uint32_t i = 0; i < kIxDim / 4; i++) {
            const float4 a = qh[i], b = kg[i];
            d0 = fmaf(a.x, b.x, d0), d1 = fmaf(a.y, b.y, d1), d2 = fmaf(a.z, b.z, d2), d3 = fmaf(a.w, b.w, d3);
          }
          ds[gl][h] = (d0 + d1) + (d2 + d3);
        }
        __syncthreads();
        if (threadIdx.x < n) {
          float sc = 0;
          for (uint32_t h = 0; h < 16; h++) sc = fmaf(ws[tt][h], fmaxf(ds[threadIdx.x][h], 0.0f), sc);
          scores[(uint64_t) (t0 + tt) * max_groups + g0 + threadIdx.x] = sc;
        }
        __syncthreads();
      }
    }
  }
}

// a prompt's scores on the matrix cores: a block 16 tokens x 64 groups, a wave 16 of the groups; each head's 16 x 16
// tile 8 WMMAs over the 128 dims (f16 in, F32 sums - the prompt's other matrices' read math), its weighted relu
// summed into the score in registers, the heads in order. the keys f16 in LDS once a block, a head's queries in turn.
// not the F32 loop's bits (f16 inputs): a prompt's selections against the reference's measured (test_hy4_indexer_gpu)
constexpr uint32_t kMmTok = 16, kMmGroups = 64, kMmPad = 8, kMmRow = kIxDim + kMmPad;
__global__ __launch_bounds__(128) void ix_score_mm_kernel(const float* q, const float* w, const float* groups,
                                                          uint32_t heads, uint32_t pool, uint32_t top,
                                                          const int32_t* pos, uint32_t n_tok, uint32_t max_groups,
                                                          float* scores) {
  __shared__ _Float16 gk[kMmGroups * kMmRow], qh[kMmTok * kMmRow];
  __shared__ float ws[kMmTok][16];
  const uint32_t t0 = blockIdx.y * kMmTok, g0 = blockIdx.x * kMmGroups, nt = n_tok - t0 < kMmTok ? n_tok - t0 : kMmTok;
  uint32_t most = 0;
  for (uint32_t t = 0; t < nt; t++) {
    const uint32_t own = (uint32_t) pos[t0 + t] / pool;
    if (own >= top && own > most) most = own;
  }
  if (g0 >= most) return;
  const uint32_t ng = most - g0 < kMmGroups ? most - g0 : kMmGroups;
  for (uint32_t i = threadIdx.x; i < kMmGroups * kIxDim; i += 128) {
    const uint32_t r = i / kIxDim, c = i % kIxDim;
    gk[r * kMmRow + c] = (_Float16) (r < ng ? groups[(uint64_t) (g0 + r) * kIxDim + c] : 0.0f);
  }
  for (uint32_t i = threadIdx.x; i < kMmTok * 16; i += 128) {
    const uint32_t t = i / 16, h = i % 16;
    ws[t][h] = t < nt && h < heads ? w[(uint64_t) (t0 + t) * heads + h] : 0.0f;
  }
  const uint32_t wave = threadIdx.x / 32;
  osmos_f8 sc = {};
  for (uint32_t h = 0; h < heads; h++) {
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < kMmTok * kIxDim; i += 128) {
      const uint32_t t = i / kIxDim, c = i % kIxDim;
      qh[t * kMmRow + c] = (_Float16) (t < nt ? q[((uint64_t) (t0 + t) * heads + h) * kIxDim + c] : 0.0f);
    }
    __syncthreads();
    osmos_f8 acc = {};
#pragma unroll
    for (uint32_t kc = 0; kc < kIxDim / 16; kc++)
      acc = osmos_mma(osmos_load_a(qh + kc * 16, kMmRow), osmos_load_b_s(gk + wave * 16 * kMmRow + kc * 16, kMmRow), acc);
#pragma unroll
    for (int j = 0; j < 8; j++) sc[j] = fmaf(ws[osmos_acc_row(j)][h], fmaxf(acc[j], 0.0f), sc[j]);
  }
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const uint32_t t = (uint32_t) osmos_acc_row(j), g = g0 + wave * 16 + (uint32_t) osmos_acc_col(j);
    if (t < nt && g < (uint32_t) pos[t0 + t] / pool) scores[(uint64_t) (t0 + t) * max_groups + g] = sc[j];
  }
}

// a prompt's scores, a wave a token: its 16 heads the 16 rows of a WMMA tile (its queries f16 in registers once, 8
// fragments over the 128 dims), over a block's 128 groups in LDS (f16, 8 tiles of 16) - 64 WMMAs a token, not a
// head's 8 between two syncs; w_h relu(.) over the heads a column sum in the tile (a lane's 8 rows, then its half's
// partner). f16 in, F32 sums: the read's math (ix_score_mm_kernel's inputs; its heads' order another)
constexpr uint32_t kM2Tok = 16, kM2Groups = 128, kM2Row = kIxDim + 8;
__global__ __launch_bounds__(128) void ix_score_mm2_kernel(const float* q, const float* w, const float* groups,
                                                           uint32_t pool, uint32_t top, const int32_t* pos,
                                                           uint32_t n_tok, uint32_t max_groups, float* scores) {
  __shared__ __attribute__((aligned(32))) _Float16 gk[kM2Groups * kM2Row];
  const uint32_t t0 = blockIdx.y * kM2Tok, g0 = blockIdx.x * kM2Groups;
  const uint32_t nt = n_tok - t0 < kM2Tok ? n_tok - t0 : kM2Tok, wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  uint32_t most = 0;
  for (uint32_t t = 0; t < nt; t++) {
    const uint32_t own = (uint32_t) pos[t0 + t] / pool;
    if (own >= top && own > most) most = own;
  }
  if (g0 >= most) return;
  const uint32_t ng = most - g0 < kM2Groups ? most - g0 : kM2Groups;
  for (uint32_t i = threadIdx.x; i < kM2Groups * kIxDim / 4; i += 128) {
    const uint32_t r = i / (kIxDim / 4), c = 4 * (i % (kIxDim / 4));
    const float4 v = r < ng ? *(const float4*) (groups + (uint64_t) (g0 + r) * kIxDim + c) : make_float4(0, 0, 0, 0);
    _Float16* o = gk + r * kM2Row + c;
    o[0] = (_Float16) v.x, o[1] = (_Float16) v.y, o[2] = (_Float16) v.z, o[3] = (_Float16) v.w;
  }
  __syncthreads();
  for (uint32_t tt = wave; tt < nt; tt += 4) {
    const uint32_t t = t0 + tt, own = (uint32_t) pos[t] / pool;
    if (own < top || g0 >= own) continue;  // (wave-uniform)
    osmos_fa a[kIxDim / 16];
    const float* qr = q + ((uint64_t) t * 16 + lane % 16) * kIxDim;
#pragma unroll
    for (uint32_t kc = 0; kc < kIxDim / 16; kc++) {
      osmos_h16 v;
#pragma unroll
      for (int e = 0; e < 16; e += 4) {
        const float4 x = *(const float4*) (qr + kc * 16 + e);
        v[e] = (_Float16) x.x, v[e + 1] = (_Float16) x.y, v[e + 2] = (_Float16) x.z, v[e + 3] = (_Float16) x.w;
      }
      a[kc] = v;
    }
    float wr[8];
#pragma unroll
    for (int j = 0; j < 8; j++) wr[j] = w[(uint64_t) t * 16 + osmos_acc_row(j)];
    const uint32_t mine = own - g0 < kM2Groups ? own - g0 : kM2Groups;
    for (uint32_t gt = 0; gt * 16 < mine; gt++) {
      osmos_f8 acc = {};
#pragma unroll
      for (uint32_t kc = 0; kc < kIxDim / 16; kc++)
        acc = osmos_mma(a[kc], osmos_load_b_s(gk + gt * 16 * kM2Row + kc * 16, kM2Row), acc);
      float sc = 0;
#pragma unroll
      for (int j = 0; j < 8; j++) sc = fmaf(wr[j], fmaxf(acc[j], 0.0f), sc);
      sc += __shfl_xor(sc, 16, 32);
      const uint32_t g = gt * 16 + (uint32_t) osmos_acc_col(0);
      if (lane < 16 && g < mine) scores[(uint64_t) t * max_groups + g0 + g] = sc;
    }
  }
}

// the selection: a block a token; scored, its scores ix_score_kernel's - else computed here, a thread a group (a
// prompt's many tokens: one block each the better shape; the same bits either way)
__global__ __launch_bounds__(kThreads) void ix_select_kernel(const float* q, const float* w, const float* groups,
                                                             uint32_t heads, uint32_t dim, uint32_t pool, uint32_t top,
                                                             const int32_t* pos, uint32_t max_groups, float* scores,
                                                             int32_t* sel, int32_t* n_sel, bool scored,
                                                             bool dense_only) {
  __shared__ float qs[kIxDim * 16], ws[16];
  __shared__ uint32_t lds[kThreads];
  const uint32_t t = blockIdx.x, own = (uint32_t) pos[t] / pool;
  int32_t* out = sel + (uint64_t) t * top;
  if (own < top) {  // every complete group, and its own
    for (uint32_t g = threadIdx.x; g <= own; g += kThreads) out[g] = (int32_t) g;
    if (threadIdx.x == 0) n_sel[t] = (int32_t) own + 1;
    return;
  }
  if (dense_only) return;  // (the spread selection's tokens: ix_wide_*)
  float* sc = scores + (uint64_t) t * max_groups;
  if (!scored) {
    for (uint32_t i = threadIdx.x; i < heads * dim; i += kThreads) qs[i / dim * kIxRow + i % dim] = q[(uint64_t) t * heads * dim + i];
    for (uint32_t h = threadIdx.x; h < heads; h += kThreads) ws[h] = w[(uint64_t) t * heads + h];
    __syncthreads();
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
  }
  __syncthreads();
  const uint32_t want = top - 1;
  __shared__ uint32_t hist[256], st_prefix, st_need;
  if (threadIdx.x == 0) st_prefix = 0, st_need = want;
  for (int shift = 24; shift >= 0; shift -= 8) {  // the greatest key V with at least `want` keys >= V: four 8-bit
    for (uint32_t b = threadIdx.x; b < 256; b += kThreads) hist[b] = 0;  // digits, a histogram of the keys under
    __syncthreads();                                                     // the prefix so far each (ix_wide_*'s)
    const uint32_t pre = st_prefix, mask = shift == 24 ? 0u : ~0u << (shift + 8);
    for (uint32_t g = threadIdx.x; g < own; g += kThreads) {
      const uint32_t k = order_key(sc[g]);
      if ((k & mask) == (pre & mask)) atomicAdd(&hist[(k >> shift) & 255], 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {  // from the top digit down: the first whose keys reach the ones still wanted
      uint32_t acc = 0, b = 255;
      for (;; b--) {
        if (acc + hist[b] >= st_need || b == 0) break;
        acc += hist[b];
      }
      st_prefix = pre | b << shift, st_need -= acc;
    }
    __syncthreads();
  }
  const uint32_t prefix = st_prefix;
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


// the selection spread over the GPU for a step's few tokens (a decode's, a verify's - 2026-11-19: a block a token's
// radix over 32K groups took 113 ms a step at 128K, 0.9 s at 1M): the same picks as ix_select_kernel's - the key V
// with count(> V) < want <= count(>= V) by four 8-bit radix passes (each a many-block histogram of the keys under the
// prefix so far, integer atomics: exact in any order; then a block a token picks the digit), then in group order
// every key > V and the first want - count(> V) keys == V (block counts, their offsets, the writes), then its own
struct IxWide {   // a token's state (ix_wide_scratch's): the prefix so far, the keys still wanted, a pass's histogram
  uint32_t prefix, need, hist[256];
};
constexpr uint32_t kWideBlocks = 256;  // a token's blocks over its groups at most (each a chunk of them)

__device__ inline uint32_t wide_chunk(uint32_t own) { return ((own + kWideBlocks - 1) / kWideBlocks + kThreads - 1) / kThreads * kThreads; }

__global__ void ix_wide_init_kernel(const int32_t* pos, uint32_t pool, uint32_t top, IxWide* st) {
  const uint32_t t = blockIdx.x;
  for (uint32_t b = threadIdx.x; b < 256; b += blockDim.x) st[t].hist[b] = 0;
  if (threadIdx.x == 0) st[t].prefix = 0, st[t].need = top - 1;
}

__global__ __launch_bounds__(kThreads) void ix_wide_hist_kernel(const float* scores, const int32_t* pos, uint32_t pool,
                                                                uint32_t top, uint32_t max_groups, uint32_t shift,
                                                                IxWide* st) {
  __shared__ uint32_t h[256];
  const uint32_t t = blockIdx.y, own = (uint32_t) pos[t] / pool, chunk = wide_chunk(own), g0 = blockIdx.x * chunk;
  if (own < top || g0 >= own) return;
  for (uint32_t b = threadIdx.x; b < 256; b += kThreads) h[b] = 0;
  __syncthreads();
  const uint32_t prefix = st[t].prefix, mask = shift == 24 ? 0u : ~0u << (shift + 8);
  const float* sc = scores + (uint64_t) t * max_groups;
  const uint32_t g1 = g0 + chunk < own ? g0 + chunk : own;
  for (uint32_t g = g0 + threadIdx.x; g < g1; g += kThreads) {
    const uint32_t k = order_key(sc[g]);
    if ((k & mask) == (prefix & mask)) atomicAdd(&h[(k >> shift) & 255], 1u);
  }
  __syncthreads();
  for (uint32_t b = threadIdx.x; b < 256; b += kThreads)
    if (h[b]) atomicAdd(&st[t].hist[b], h[b]);
}

__global__ void ix_wide_pick_kernel(const int32_t* pos, uint32_t pool, uint32_t top, uint32_t shift, IxWide* st) {
  const uint32_t t = blockIdx.x, own = (uint32_t) pos[t] / pool;
  if (own < top || threadIdx.x) return;
  uint32_t acc = 0, b = 255;  // from the top digit down: the first whose keys reach the ones still wanted
  for (;; b--) {
    if (acc + st[t].hist[b] >= st[t].need || b == 0) break;
    acc += st[t].hist[b];
  }
  st[t].prefix |= b << shift, st[t].need -= acc;
  for (uint32_t i = 0; i < 256; i++) st[t].hist[i] = 0;
}

__global__ __launch_bounds__(kThreads) void ix_wide_count_kernel(const float* scores, const int32_t* pos, uint32_t pool,
                                                                 uint32_t top, uint32_t max_groups, const IxWide* st,
                                                                 uint32_t* cnt) {
  __shared__ uint32_t lds[kThreads];
  const uint32_t t = blockIdx.y, own = (uint32_t) pos[t] / pool, chunk = wide_chunk(own), g0 = blockIdx.x * chunk;
  if (own < top) return;
  uint32_t gt = 0, eq = 0;
  const uint32_t v = st[t].prefix, g1 = g0 + chunk < own ? g0 + chunk : own;
  const float* sc = scores + (uint64_t) t * max_groups;
  for (uint32_t g = g0 + threadIdx.x; g < g1; g += kThreads) {
    const uint32_t k = order_key(sc[g]);
    gt += k > v, eq += k == v;
  }
  uint32_t tg = 0, te = 0;
  block_scan(gt, lds, &tg);
  block_scan(eq, lds, &te);
  if (threadIdx.x == 0) cnt[((uint64_t) t * kWideBlocks + blockIdx.x) * 2] = tg, cnt[((uint64_t) t * kWideBlocks + blockIdx.x) * 2 + 1] = te;
}

__global__ __launch_bounds__(kThreads) void ix_wide_write_kernel(const float* scores, const int32_t* pos, uint32_t pool,
                                                                 uint32_t top, uint32_t max_groups, const IxWide* st,
                                                                 const uint32_t* cnt, int32_t* sel, int32_t* n_sel) {
  __shared__ uint32_t lds[kThreads];
  const uint32_t t = blockIdx.y, own = (uint32_t) pos[t] / pool, chunk = wide_chunk(own), g0 = blockIdx.x * chunk;
  if (own < top || g0 >= own) return;
  const uint32_t v = st[t].prefix, ties = st[t].need, want = top - 1;
  uint32_t taken = 0, eq_seen = 0;  // the blocks before this one's (their counts in order: a fixed sum)
  for (uint32_t b = 0; b < blockIdx.x; b++) {
    const uint32_t gt = cnt[((uint64_t) t * kWideBlocks + b) * 2], eq = cnt[((uint64_t) t * kWideBlocks + b) * 2 + 1];
    taken += gt + (eq_seen < ties ? (ties - eq_seen < eq ? ties - eq_seen : eq) : 0), eq_seen += eq;
  }
  int32_t* out = sel + (uint64_t) t * top;
  const float* sc = scores + (uint64_t) t * max_groups;
  const uint32_t g1 = g0 + chunk < own ? g0 + chunk : own;
  for (uint32_t c0 = g0; c0 < g1; c0 += kThreads) {  // in group order: a chunk's picks placed by a scan
    const uint32_t g = c0 + threadIdx.x;
    const uint32_t key = g < g1 ? order_key(sc[g]) : 0;
    const uint32_t eq = g < g1 && key == v;
    uint32_t eq_total = 0, take_total = 0;
    const uint32_t eq_before = block_scan(eq, lds, &eq_total);
    const uint32_t take = g < g1 && (key > v || (eq && eq_seen + eq_before < ties));
    const uint32_t at = block_scan(take, lds, &take_total);
    if (take) out[taken + at] = (int32_t) g;
    taken += take_total, eq_seen += eq_total;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) out[want] = (int32_t) own, n_sel[t] = (int32_t) top;
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
  if ((uint64_t) 2 * HY4_HC_MAX * (hc * dim / kHcParts) * 4 > 48 * 1024) return false;  // (a slice's fn rows in LDS)
  const hipStream_t s = st(stream);
  const uint32_t len = hc * dim;
  if (rows == 2 * hc) {
    hc_dot_kernel<2 * HY4_HC_MAX><<<dim3(kHcParts, (n_tok + kHcTok - 1) / kHcTok), kThreads,
                                     2 * HY4_HC_MAX * (len / kHcParts) * 4, s>>>(h, fn, len, n_tok, scratch);
    hc_pre_kernel<HY4_HC_MAX, 2 * HY4_HC_MAX><<<n_tok, kThreads, 0, s>>>(h, scratch, scale, base, dim, eps, hc_eps,
                                                                         magnitude, norm_w, x, post);
  } else {
    hc_dot_kernel<HY4_HC_MAX><<<dim3(kHcParts, (n_tok + kHcTok - 1) / kHcTok), kThreads,
                                 HY4_HC_MAX * (len / kHcParts) * 4, s>>>(h, fn, len, n_tok, scratch);
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

extern "C" uint64_t hy4_mla_row_bytes(uint32_t latent, uint32_t rope_dim, bool q8) {
  return mla_row_bytes(latent, rope_dim, q8);
}

extern "C" bool hy4_mla_kv(const float* kva, const float* norm_w, uint32_t latent, uint32_t rope_dim, float eps,
                           const float* freqs, const int32_t* pos, uint32_t n_tok, void* cache, uint32_t slots,
                           bool q8, void* stream) {
  if (!kva || !norm_w || !freqs || !pos || !cache || !latent || latent > HY4_MLA_MAX_LATENT || !rope_dim ||
      rope_dim % 2 || !n_tok || (q8 && latent % 32))
    return false;
  mla_kv_kernel<<<n_tok, kThreads, 0, st(stream)>>>(kva, norm_w, latent, rope_dim, eps, freqs, pos, (uint8_t*) cache,
                                                    slots, q8);
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

extern "C" bool hy4_mla_attn(const float* qa, const void* cache, uint32_t slots, bool q8, const float* sinks, uint32_t heads,
                             uint32_t latent, uint32_t rope_dim, uint32_t window, const int32_t* sel,
                             const int32_t* n_sel, uint32_t sel_cap, uint32_t pool, float scale, const int32_t* pos,
                             uint32_t n_tok, float* o_lat, float* scratch, bool read, void* stream) {
  if (!qa || !cache || !sinks || !pos || !o_lat || !scratch || !heads || !latent || latent > HY4_MLA_MAX_LATENT ||
      !rope_dim || rope_dim > kMaxRope || !n_tok || (window && window > HY4_MLA_MAX_KEYS) || latent % 4 || rope_dim % 4 ||
      (uintptr_t) cache % 16 || (uintptr_t) qa % 16 || (q8 && (latent % 32 || mla_row_bytes(latent, rope_dim, true) % 16)))
    return false;
  if (sel && (!n_sel || window || !pool || !sel_cap || (uint64_t) sel_cap * pool > HY4_MLA_MAX_KEYS)) return false;
  if (slots && slots < window) return false;
  const hipStream_t s = st(stream);
  const uint32_t splits = (window ? window : HY4_MLA_MAX_KEYS) / kSplitKeys + 1 < kMaxSplits
                              ? (window ? window : HY4_MLA_MAX_KEYS) / kSplitKeys + 1
                              : kMaxSplits;
  static const bool exact = getenv("HY4_MLA_F32") != nullptr;  // (a read's attention the F32 path's: A/B)
  if (read && !exact && heads % 16 == 0 && latent % 64 == 0 && (latent + rope_dim) % 16 == 0 &&
      latent + rope_dim <= kMlaRow && rope_dim <= 64) {
    if (heads % 64 == 0 && !getenv("MLA_G2"))
      mla_mm_kernel<4><<<dim3(heads / 64, n_tok), 512, 0, s>>>(qa, (const uint8_t*) cache, slots, q8, sinks, heads,
                                                               latent, rope_dim, window, sel, n_sel, sel_cap, pool, scale,
                                                               pos, o_lat);
    else if (heads % 32 == 0)
      mla_mm_kernel<2><<<dim3(heads / 32, n_tok), 256, 0, s>>>(qa, (const uint8_t*) cache, slots, q8, sinks, heads,
                                                               latent, rope_dim, window, sel, n_sel, sel_cap, pool, scale,
                                                               pos, o_lat);
    else
      mla_mm_kernel<1><<<dim3(heads / 16, n_tok), 128, 0, s>>>(qa, (const uint8_t*) cache, slots, q8, sinks, heads,
                                                               latent, rope_dim, window, sel, n_sel, sel_cap, pool, scale,
                                                               pos, o_lat);
    return ok();
  }
  for (uint32_t t0 = 0; t0 < n_tok; t0 += kTokChunk) {
    const uint32_t m = n_tok - t0 < kTokChunk ? n_tok - t0 : kTokChunk;
    mla_split_kernel<<<dim3(splits, (heads + kHeadGroup - 1) / kHeadGroup, m), kThreads, 0, s>>>(
        qa, (const uint8_t*) cache, slots, q8, heads, latent, rope_dim, window, sel, n_sel, sel_cap, pool, scale, pos, t0,
        scratch);
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
                                                      norm_w, dim, pool, rope_dim, eps, freqs, g0, nullptr, groups, 0);
  return ok();
}

extern "C" bool hy4_ix_pool_at(const float* keys, uint32_t slots, const void* w_value, const void* w_gate,
                               const float* ape, const float* norm_w, uint32_t dim, uint32_t pool, uint32_t rope_dim,
                               float eps, const float* freqs, const int32_t* pos, uint32_t n_tok, float* groups,
                               void* stream) {
  if (!keys || !w_value || !w_gate || !ape || !norm_w || !groups || !pos || !dim || dim > kIxDim || dim % kQ8Block ||
      !pool || pool > kIxPoolMax || rope_dim % 2 || rope_dim > dim || !n_tok || (slots && (slots < pool || slots % pool)))
    return false;
  ix_pool_kernel<<<n_tok, kIxDim, 0, st(stream)>>>(keys, (const uint8_t*) w_value, (const uint8_t*) w_gate, ape, norm_w,
                                                   dim, pool, rope_dim, eps, freqs, 0, pos, groups, slots);
  return ok();
}

__global__ void rows_at_kernel(float* dst, const float* src, const int32_t* pos, uint32_t len, uint32_t slots) {
  const uint32_t t = blockIdx.y;
  const uint64_t r = slots ? (uint32_t) pos[t] % slots : (uint32_t) pos[t];
  for (uint32_t i = blockIdx.x * kThreads + threadIdx.x; i < len; i += gridDim.x * kThreads)
    dst[r * len + i] = src[(uint64_t) t * len + i];
}

extern "C" bool hy4_rows_at(float* dst, const float* src, const int32_t* pos, uint32_t n, uint32_t len, uint32_t slots,
                            void* stream) {
  if (!dst || !src || !pos || !n || !len) return false;
  const uint32_t bx = (len + kThreads - 1) / kThreads;
  rows_at_kernel<<<dim3(bx < 64 ? bx : 64, n), kThreads, 0, st(stream)>>>(dst, src, pos, len, slots);
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
                              float* scores, int32_t* sel, int32_t* n_sel, bool read, void* stream) {
  if (!q || !w || !groups || !pos || !scores || !sel || !n_sel || !heads || heads > 16 || !dim || dim > kIxDim ||
      !pool || top < 2 || !n_tok || !max_groups)
    return false;
  // the scores tiled for every step (a prompt's too: the same bits as the select kernel's own loop); the selection over
  // many blocks for a decode's or a verify's few tokens, a block a token for a prompt's many
  const bool spread = n_tok <= kIxSpreadTokens;
  static const bool exact = getenv("HY4_IX_F32") != nullptr;  // (a read's scores the F32 loop's: A/B)
  if (read && !spread && !exact && dim == kIxDim && heads == 16 && !getenv("HY4_IX_MM1"))
    ix_score_mm2_kernel<<<dim3((max_groups + kM2Groups - 1) / kM2Groups, (n_tok + kM2Tok - 1) / kM2Tok), 128, 0,
                          st(stream)>>>(q, w, groups, pool, top, pos, n_tok, max_groups, scores);
  else if (read && !spread && !exact && dim == kIxDim && heads <= 16)
    ix_score_mm_kernel<<<dim3((max_groups + kMmGroups - 1) / kMmGroups, (n_tok + kMmTok - 1) / kMmTok), 128, 0,
                         st(stream)>>>(q, w, groups, heads, pool, top, pos, n_tok, max_groups, scores);
  else if (!read && heads == 16 && dim == kIxDim && !getenv("HY4_IX_TILE"))
    ix_score_persist_kernel<<<kIxPBlocks, kThreads, 0, st(stream)>>>(q, w, groups, pool, top, pos, n_tok, max_groups,
                                                                     scores);
  else
    ix_score_kernel<<<(max_groups + kIxTile - 1) / kIxTile, kThreads, 0, st(stream)>>>(q, w, groups, heads, dim, pool,
                                                                                         top, pos, n_tok, max_groups,
                                                                                         scores);
  ix_select_kernel<<<n_tok, kThreads, 0, st(stream)>>>(q, w, groups, heads, dim, pool, top, pos, max_groups, scores,
                                                       sel, n_sel, true, spread);
  if (!spread) return ok();
  static IxWide* wst = nullptr;  // (the spread selection's scratch: kIxSpreadTokens tokens' - made once)
  static uint32_t* wcnt = nullptr;
  if (!wst && (hipMalloc(&wst, kIxSpreadTokens * sizeof(IxWide)) != hipSuccess ||
               hipMalloc(&wcnt, (size_t) kIxSpreadTokens * kWideBlocks * 2 * 4) != hipSuccess))
    return false;
  const hipStream_t s = st(stream);
  const dim3 grid(kWideBlocks, n_tok);
  ix_wide_init_kernel<<<n_tok, 64, 0, s>>>(pos, pool, top, wst);
  for (int shift = 24; shift >= 0; shift -= 8) {
    ix_wide_hist_kernel<<<grid, kThreads, 0, s>>>(scores, pos, pool, top, max_groups, (uint32_t) shift, wst);
    ix_wide_pick_kernel<<<n_tok, 64, 0, s>>>(pos, pool, top, (uint32_t) shift, wst);
  }
  ix_wide_count_kernel<<<grid, kThreads, 0, s>>>(scores, pos, pool, top, max_groups, wst, wcnt);
  ix_wide_write_kernel<<<grid, kThreads, 0, s>>>(scores, pos, pool, top, max_groups, wst, wcnt, sel, n_sel);
  return ok();
}

extern "C" bool hy4_mla_qrope(const float* q, uint32_t heads, uint32_t qk, uint32_t row, uint32_t rope_dim,
                              const float* freqs, const int32_t* pos, uint32_t n_tok, float* qa, void* stream) {
  if (!q || !freqs || !pos || !qa || !heads || !n_tok || !rope_dim || rope_dim % 2 || rope_dim > qk || rope_dim > row)
    return false;
  q_rope_kernel<<<dim3(heads, n_tok), 64, 0, st(stream)>>>(q, heads, qk, row, rope_dim, freqs, pos, qa);
  return ok();
}

extern "C" bool hy4_gate_mul(float* out, const float* gate, uint64_t n, void* stream) {
  if (!out || !gate || !n) return false;
  gate_kernel<<<(unsigned) ((n + kThreads - 1) / kThreads), kThreads, 0, st(stream)>>>(out, gate, n);
  return ok();
}

// hy4_route: one block. the picks counted by expert, the counts' and entries' prefix sums, then the tokens a group at
// a time (each thread a pick): a pick's row its expert's start, the group's earlier tokens that picked it, the rows
// placed before the group - experts ascending, an expert's tokens ascending, the host's order
constexpr int kRouteThreads = 1024;
static_assert(kRouteThreads == HY4_ROUTE_MAX_EXPERTS, "the prefix sums: a thread an expert");

struct RouteBase {
  const uint8_t* p[3];
  uint64_t bytes[3];
};

__global__ __launch_bounds__(kRouteThreads) void route_kernel(const int32_t* idx, const float* wtok, uint32_t n,
                                                              uint32_t k, uint32_t E, uint32_t chunk, RouteBase base,
                                                              int32_t* gidx, int32_t* sidx, float* wsel, int32_t* cnt,
                                                              int32_t* xrow, const void** wptr, int32_t* fault) {
  __shared__ int32_t count[HY4_ROUTE_MAX_EXPERTS], start[HY4_ROUTE_MAX_EXPERTS], ent[HY4_ROUTE_MAX_EXPERTS];
  __shared__ int32_t run[HY4_ROUTE_MAX_EXPERTS], group[kRouteThreads];
  __shared__ int32_t bad, total;
  const uint32_t tid = threadIdx.x, nk = n * k;
  for (uint32_t e = tid; e < HY4_ROUTE_MAX_EXPERTS; e += kRouteThreads) count[e] = 0, run[e] = 0;
  if (tid == 0) bad = 0;
  __syncthreads();
  for (uint32_t m = tid; m < nk; m += kRouteThreads) {
    const int32_t e = idx[m];
    bool b = e < 0 || (uint32_t) e >= E;
    for (uint32_t s = m - m % k; !b && s < m; s++) b = idx[s] == e;
    if (b)
      bad = 1;
    else
      atomicAdd(&count[e], 1);
  }
  __syncthreads();
  if (bad) {
    if (tid == 0) *fault = 1;
    return;
  }
  // exclusive prefix sums of the rows and the entries (each expert's rows in chunks), Hillis-Steele over the experts
  const int32_t c_own = tid < E ? count[tid] : 0, e_own = (c_own + (int32_t) chunk - 1) / (int32_t) chunk;
  start[tid] = c_own, ent[tid] = e_own;
  __syncthreads();
  for (uint32_t d = 1; d < kRouteThreads; d <<= 1) {
    const int32_t a = tid >= d ? start[tid - d] : 0, b = tid >= d ? ent[tid - d] : 0;
    __syncthreads();
    start[tid] += a, ent[tid] += b;
    __syncthreads();
  }
  start[tid] -= c_own, ent[tid] -= e_own;
  if (tid == kRouteThreads - 1) total = ent[tid] + e_own;
  __syncthreads();
  const uint32_t G = kRouteThreads / k;  // tokens a group
  for (uint32_t t0 = 0; t0 < n; t0 += G) {
    const uint32_t m = t0 * k + tid, t = t0 + tid / k;
    const bool mine = tid < G * k && t < n;
    group[tid] = mine ? idx[m] : -1;
    __syncthreads();
    if (mine) {
      const int32_t e = group[tid];
      int32_t rank = 0;
      for (uint32_t q = 0; q < (tid / k) * k; q++) rank += group[q] == e;
      const int32_t r = start[e] + run[e] + rank;
      gidx[r] = (int32_t) t, sidx[m] = r, wsel[r] = wtok[m];
    }
    __syncthreads();
    if (mine) atomicAdd(&run[group[tid]], 1);
    __syncthreads();
  }
  for (uint32_t e = tid; e < E; e += kRouteThreads)
    for (int32_t j = 0, c = count[e]; j * (int32_t) chunk < c; j++) {
      const uint32_t i = (uint32_t) (ent[e] + j);
      cnt[i] = min((int32_t) chunk, c - j * (int32_t) chunk), xrow[i] = start[e] + j * (int32_t) chunk;
      for (int q = 0; q < 3; q++) wptr[(uint64_t) q * nk + i] = base.p[q] + (uint64_t) e * base.bytes[q];
    }
  for (uint32_t i = (uint32_t) total + tid; i < nk; i += kRouteThreads) cnt[i] = 0;
}

extern "C" bool hy4_route(const int32_t* idx, const float* wtok, uint32_t n_tok, uint32_t k, uint32_t n_exp,
                          uint32_t chunk, const void* const base[3], const uint64_t bytes[3], int32_t* gidx,
                          int32_t* sidx, float* wsel, int32_t* cnt, int32_t* xrow, const void** wptr, int32_t* fault,
                          void* stream) {
  if (!k || k > HY4_ROUTE_MAX_K || !n_exp || n_exp > HY4_ROUTE_MAX_EXPERTS || !chunk || !n_tok || !base || !bytes ||
      !idx || !wtok || !gidx || !sidx || !wsel || !cnt || !xrow || !wptr || !fault)
    return false;
  RouteBase b;
  for (int q = 0; q < 3; q++) b.p[q] = (const uint8_t*) base[q], b.bytes[q] = bytes[q];
  route_kernel<<<1, kRouteThreads, 0, st(stream)>>>(idx, wtok, n_tok, k, n_exp, chunk, b, gidx, sidx, wsel, cnt, xrow,
                                                    wptr, fault);
  return ok();
}

// DFly2's draft block (0.11.0): each draft row's groups - the target layer's picks for the last accepted token (its
// snapshot: snap_n groups ascending, the last the token's own) and every group after that one through the row's own
// position (the block's rows, written at anchor ..): a thread a row
__global__ void dfly_sel_kernel(const int32_t* snap, const int32_t* snap_n, int32_t anchor, uint32_t pool,
                                uint32_t rows, uint32_t cap, int32_t* sel, int32_t* n_sel) {
  const uint32_t j = threadIdx.x;
  if (j >= rows) return;
  const int32_t n = *snap_n, own = (anchor - 1) / (int32_t) pool, last = (anchor + (int32_t) j) / (int32_t) pool;
  int32_t* out = sel + (uint64_t) j * cap;
  for (int32_t i = 0; i < n; i++) out[i] = snap[i];
  int32_t k = n;
  for (int32_t g = own + 1; g <= last; g++) out[k++] = g;
  n_sel[j] = k;
}

extern "C" bool hy4_dfly_sel(const int32_t* snap, const int32_t* snap_n, int32_t anchor, uint32_t pool, uint32_t top,
                             uint32_t rows, uint32_t cap, int32_t* sel, int32_t* n_sel, void* stream) {
  if (!snap || !snap_n || anchor < 1 || !pool || !rows || rows > HY4_DFLY_MAX_ROWS ||
      cap < top + (rows + pool - 1) / pool + 1 || !sel || !n_sel)
    return false;
  dfly_sel_kernel<<<1, 64, 0, st(stream)>>>(snap, snap_n, anchor, pool, rows, cap, sel, n_sel);
  return ok();
}

// a row's greatest logit (the lowest index among equal ones) and its softmax probability: max, then sum exp(x - max)
// by a fixed tree, p = 1 / sum
constexpr int kPickThreads = 1024;
__global__ __launch_bounds__(kPickThreads) void dfly_pick_kernel(const float* x, uint32_t n, int32_t* id, float* p) {
  __shared__ float lds[kPickThreads];
  __shared__ int ldi[kPickThreads];
  float m = -INFINITY;
  int mi = 0;
  for (uint32_t i = threadIdx.x; i < n; i += kPickThreads)
    if (x[i] > m) m = x[i], mi = (int) i;
  const int best = block_argmax<kPickThreads>(&m, mi, lds, ldi);
  float s = 0;
  for (uint32_t i = threadIdx.x; i < n; i += kPickThreads) s += expf(x[i] - m);
  s = block_sum<kPickThreads>(s, lds);
  if (threadIdx.x == 0) *id = best, *p = 1.0f / s;
}

extern "C" bool hy4_dfly_pick(const float* logits, uint32_t n, int32_t* id, float* p, void* stream) {
  if (!logits || !n || !id || !p) return false;
  dfly_pick_kernel<<<1, kPickThreads, 0, st(stream)>>>(logits, n, id, p);
  return ok();
}
