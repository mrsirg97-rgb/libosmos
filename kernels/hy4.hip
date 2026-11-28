// Hy4-flash's small steps (include/osmos/gpu/hy4_ops.h): iHC and Engram's ends, the reference's order in F32
#include <cmath>
#include <cstdlib>

#include <hip/hip_fp16.h>

#include "port.h"

#include "block.h"
#include "launch.h"
#include "round.h"
#include "mla_row.h"

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

// gated MLA, absorbed: a cache row's layout mla_row.h's
using namespace mla;

// pair i of x (x[2i], x[2i + 1]) rotated by angle pos * freqs[i]: the reference's x cos + rotate_half(x) sin
__device__ inline void rope_pair(const float* in, float* out, uint32_t i, int32_t pos, const float* freqs) {
  const float a = (float) pos * freqs[i], c = cosf(a), s = sinf(a);
  const float x0 = in[2 * i], x1 = in[2 * i + 1];
  out[2 * i] = x0 * c + (-x1) * s;
  out[2 * i + 1] = x1 * c + x0 * s;
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
  if (isnan(x)) return 0u;
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
// reading its own key once a head would read 512-byte strides 16 times), a
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
  // read one bank - a 16-way conflict every term)
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
      // latency-bound, not the keys' bytes) - every step's the same
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
// every one of 8,192 blocks at 1M), then the math, serial - latency-bound, not the bytes
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
// not the F32 loop's bits (f16 inputs): a prompt's selections checked against the reference's (test_hy4_indexer_gpu)
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

// the selection: a block a token, its scores the scorer's (ix_score_*)
__global__ __launch_bounds__(kThreads) void ix_select_kernel(const int32_t* pos, uint32_t pool, uint32_t top,
                                                             uint32_t max_groups, float* scores, int32_t* sel,
                                                             int32_t* n_sel, bool dense_only) {
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


// the selection spread over the GPU for a step's few tokens (a decode's, a verify's: a block a token's radix over
// 32K groups leaves the GPU idle): the same picks as ix_select_kernel's - the key V
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
  if (own < top || g0 >= own) return;
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

// the spread selection's scratch: kIxSpreadTokens tokens' state, then their blocks' counts (whole IxWide's first)
static constexpr size_t kSpreadState = (kIxSpreadTokens * sizeof(IxWide) + 255) / 256 * 256;
extern "C" size_t hy4_ix_spread_bytes(void) { return kSpreadState + (size_t) kIxSpreadTokens * kWideBlocks * 2 * 4; }

extern "C" bool hy4_ix_select(const float* q, const float* w, const float* groups, uint32_t heads, uint32_t dim,
                              uint32_t pool, uint32_t top, const int32_t* pos, uint32_t n_tok, uint32_t max_groups,
                              float* scores, int32_t* sel, int32_t* n_sel, bool read, void* spread_scratch,
                              void* stream) {
  if (!q || !w || !groups || !pos || !scores || !sel || !n_sel || !heads || heads > 16 || !dim || dim > kIxDim ||
      !pool || top < 2 || !n_tok || !max_groups)
    return false;
  // every token's own < top (guaranteed when max_groups < top): dense prefix only, skip scoring and wide radix
  if (max_groups < top) {
    ix_select_kernel<<<n_tok, kThreads, 0, st(stream)>>>(pos, pool, top, max_groups, scores, sel, n_sel, false);
    return ok();
  }
  // a read's scores the matrix cores' at any step size (its bits a prompt's however it is split); a decode's or a
  // verify's F32. the selection over many blocks for a step's few tokens, a block a token for many (the
  // same picks either way: integer counts)
  const bool spread = n_tok <= kIxSpreadTokens;
  if (read && dim == kIxDim && heads == 16)
    ix_score_mm2_kernel<<<dim3((max_groups + kM2Groups - 1) / kM2Groups, (n_tok + kM2Tok - 1) / kM2Tok), 128, 0,
                          st(stream)>>>(q, w, groups, pool, top, pos, n_tok, max_groups, scores);
  else if (read && dim == kIxDim && heads <= 16)
    ix_score_mm_kernel<<<dim3((max_groups + kMmGroups - 1) / kMmGroups, (n_tok + kMmTok - 1) / kMmTok), 128, 0,
                         st(stream)>>>(q, w, groups, heads, pool, top, pos, n_tok, max_groups, scores);
  else if (!read && heads == 16 && dim == kIxDim)
    ix_score_persist_kernel<<<kIxPBlocks, kThreads, 0, st(stream)>>>(q, w, groups, pool, top, pos, n_tok, max_groups,
                                                                     scores);
  else
    ix_score_kernel<<<(max_groups + kIxTile - 1) / kIxTile, kThreads, 0, st(stream)>>>(q, w, groups, heads, dim, pool,
                                                                                         top, pos, n_tok, max_groups,
                                                                                         scores);
  ix_select_kernel<<<n_tok, kThreads, 0, st(stream)>>>(pos, pool, top, max_groups, scores, sel, n_sel, spread);
  if (!spread) return ok();
  if (!spread_scratch) return false;  // (the caller's: hy4_ix_spread_bytes, made with its forward)
  IxWide* wst = (IxWide*) spread_scratch;
  uint32_t* wcnt = (uint32_t*) ((char*) spread_scratch + kSpreadState);
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

// DFly2's draft block: each draft row's groups - the target layer's picks for the last accepted token (its
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
