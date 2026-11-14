// Inkling-2-Small's own steps (include/osmos/gpu/ink2_ops.h), F32
#include <cstdlib>

#include "block.h"
#include "launch.h"

extern "C" {
#include "osmos/gpu/ink2_ops.h"
}

namespace {

using launches::kThreads;
using launches::grid;
using launches::st;
using launches::ok;

__global__ void softcap_kernel(float* x, uint64_t n, float cap) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

__global__ void swiglu_kernel(const float* g, const float* u, uint64_t n, float gcap, float ucap, float* h) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= n) return;
  const float gv = g[i];
  h[i] = gcap * tanhf(gv / gcap) * sigmoid(gv) * (ucap * tanhf(u[i] / ucap));
}

// the same with each row's gate and up scaled first (an NVFP4 expert's scale: ink2_scale then ink2_swiglu's bits)
__global__ void swiglu_rows_kernel(const float* g, const float* u, const float* sg, const float* su, uint32_t n,
                                   uint64_t all, float gcap, float ucap, float* h) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= all) return;
  const uint64_t r = i / n;
  const float gv = g[i] * sg[r], uv = u[i] * su[r];
  h[i] = gcap * tanhf(gv / gcap) * sigmoid(gv) * (ucap * tanhf(uv / ucap));
}

// a token's experts summed in its slot order from zero: ink2_axpy's sum, one after another
__global__ void moe_sum_kernel(float* y, const float* x, const float* a, const int32_t* row, uint32_t k, uint64_t all,
                               uint32_t dim) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= all) return;
  const uint64_t t = i / dim, c = i % dim;
  float v = 0.0f;
  for (uint32_t s = 0; s < k; s++) v += a[t * k + s] * x[(uint64_t) row[t * k + s] * dim + c];
  y[i] = v;
}

// h[i][j] += sum over t of x[t][i] x[t][j] (a calibration's X^T X, F32): a 16 x 16 tile of h a block, the tokens in
// chunks of 16 through LDS
constexpr int kGram = 16;
__global__ void gram_kernel(const float* x, uint32_t n_tok, uint32_t dim, float* h) {
  __shared__ float a[kGram][kGram + 1], b[kGram][kGram + 1];
  const uint32_t i = blockIdx.y * kGram + threadIdx.y, j = blockIdx.x * kGram + threadIdx.x;
  float acc = 0.0f;
  for (uint32_t t0 = 0; t0 < n_tok; t0 += kGram) {
    const uint32_t t = t0 + threadIdx.y, ci = blockIdx.y * kGram + threadIdx.x, cj = blockIdx.x * kGram + threadIdx.x;
    a[threadIdx.y][threadIdx.x] = t < n_tok && ci < dim ? x[(uint64_t) t * dim + ci] : 0.0f;
    b[threadIdx.y][threadIdx.x] = t < n_tok && cj < dim ? x[(uint64_t) t * dim + cj] : 0.0f;
    __syncthreads();
    for (int k = 0; k < kGram; k++) acc = fmaf(a[k][threadIdx.y], b[k][threadIdx.x], acc);
    __syncthreads();
  }
  if (i < dim && j < dim) h[(uint64_t) i * dim + j] += acc;
}

// x' = hist (taps - 1 rows) then x: token t's taps x'[t .. t + taps - 1], the current last
__device__ inline float tap(const float* x, const float* hist, uint32_t at, uint32_t h, uint32_t ch, uint32_t c) {
  return at < h ? (hist ? hist[(uint64_t) at * ch + c] : 0.0f) : x[(uint64_t) (at - h) * ch + c];
}

__global__ void conv_kernel(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, const float* w,
                            const float* hist, float res, float scale, float* y) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * ch) return;
  const uint32_t c = (uint32_t) (i % ch), t = (uint32_t) (i / ch), h = taps - 1;
  float sum = 0;
  for (uint32_t j = 0; j < taps; j++) sum += w[(uint64_t) c * taps + j] * tap(x, hist, t + j, h, ch, c);
  y[i] = (sum + res * x[i]) * scale;
}

__global__ void conv_dyn_kernel(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, uint32_t heads,
                                const float* w, const float* dyn, const float* hist, float res, float scale, float* y) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * ch) return;
  const uint32_t c = (uint32_t) (i % ch), t = (uint32_t) (i / ch), h = taps - 1, head = c / (ch / heads);
  const float* d = dyn + (uint64_t) t * heads * taps + (uint64_t) head * taps;
  float sum = 0;
  for (uint32_t j = 0; j < taps; j++)  // tap j: j tokens back
    sum += (w[(uint64_t) j * ch + c] + d[j]) * tap(x, hist, t + h - j, h, ch, c);
  y[i] = (sum + res * x[i]) * scale;
}

// a block a token: its states' inverse RMS, each query's scores and softmax, then the weighted sum
__global__ void attn_res_kernel(const float* query, uint32_t queries, const float* blocks, uint32_t nb,
                                uint32_t bs, const float* partial, uint32_t dim, float eps, float scale,
                                uint32_t n_tok, float* out) {
  __shared__ float lds[kThreads];
  __shared__ float inv[INK2_MAX_RES], p[INK2_MAX_QUERIES][INK2_MAX_RES];
  const uint32_t t = blockIdx.x, ns = nb + 1;
  auto state = [&](uint32_t s) {
    return s < nb ? blocks + ((uint64_t) t * bs + s) * dim : partial + (uint64_t) t * dim;
  };
  for (uint32_t s = 0; s < ns; s++) {
    const float* v = state(s);
    float sq = 0;
    for (uint32_t c = threadIdx.x; c < dim; c += kThreads) sq += v[c] * v[c];
    sq = block_sum<kThreads>(sq, lds);
    if (!threadIdx.x) inv[s] = rsqrtf(sq / (float) dim + eps);
  }
  __syncthreads();
  for (uint32_t q = 0; q < queries; q++)
    for (uint32_t s = 0; s < ns; s++) {
      const float* v = state(s);
      float d = 0;
      for (uint32_t c = threadIdx.x; c < dim; c += kThreads) d += query[(uint64_t) q * dim + c] * v[c];
      d = block_sum<kThreads>(d, lds);
      if (!threadIdx.x) p[q][s] = d * inv[s] * scale;
    }
  __syncthreads();
  if (threadIdx.x < queries) {
    float* r = p[threadIdx.x];
    float m = r[0], z = 0;
    for (uint32_t s = 1; s < ns; s++) m = fmaxf(m, r[s]);
    for (uint32_t s = 0; s < ns; s++) z += (r[s] = expf(r[s] - m));
    for (uint32_t s = 0; s < ns; s++) r[s] = r[s] / z * inv[s];  // (the state's own normalization folded in)
  }
  __syncthreads();
  for (uint32_t c = threadIdx.x; c < dim; c += kThreads)
    for (uint32_t q = 0; q < queries; q++) {
      float o = 0;
      for (uint32_t s = 0; s < ns; s++) o += p[q][s] * state(s)[c];
      out[((uint64_t) q * n_tok + t) * dim + c] = o;
    }
}

// a block a row: times w, the L2 clip, tau
__global__ void clip_rows_kernel(float* x, uint32_t per_tok, uint32_t dim, const float* w, float clip, Ink2Tau tau) {
  __shared__ float lds[kThreads];
  const uint32_t r = blockIdx.x, j = r % per_tok, t = r / per_tok;
  float* row = x + (uint64_t) r * dim;
  float sq = 0;
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) {
    const float v = w ? row[d] * w[(uint64_t) j * dim + d] : row[d];
    row[d] = v, sq += v * v;
  }
  float f = 1.0f;
  if (clip > 0) f = fminf(1.0f, clip / (sqrtf(block_sum<kThreads>(sq, lds)) + 1e-6f));
  if (tau.alpha > 0) f *= 1.0f + tau.alpha * logf(fmaxf(1.0f, (float) (tau.pos0 + t + 1) / tau.floor));
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) row[d] *= f;
}

constexpr uint32_t kChunk = INK2_MAX_HEAD;  // keys a pass: a thread a key, then a thread a value dim

// a block a (token, head), head_dim threads: the relative bias's row (rel . rel_proj, extent values), then the keys
// in chunks - a thread a key's logit, the chunk's max and sum folded into the running ones (online softmax), a
// thread a dim's weighted sum of the chunk's values
__global__ void attn_kernel(Ink2Attn a, const float* q, const float* rel, const float* sink, const float* gate,
                            uint32_t n_tok, const float* k, const float* v, uint32_t n_keys, const float* rel_proj,
                            float* out) {
  __shared__ float qs[INK2_MAX_HEAD], rl[INK2_MAX_EXTENT], p[kChunk], red[kChunk];
  const uint32_t t = blockIdx.x / a.heads, h = blockIdx.x % a.heads, kvh = h / (a.heads / a.kv_heads);
  const uint32_t D = a.head_dim, i = threadIdx.x, qi = n_keys - n_tok + t;
  qs[i] = q[((uint64_t) t * a.heads + h) * D + i];
  const float* rr = rel + ((uint64_t) t * a.heads + h) * a.d_rel;
  for (uint32_t e = i; e < a.extent; e += D) {
    float b = 0;
    for (uint32_t r = 0; r < a.d_rel; r++) b += rr[r] * rel_proj[(uint64_t) r * a.extent + e];
    rl[e] = b;
  }
  __syncthreads();
  const uint32_t lo = a.window && qi + 1 > a.window ? qi + 1 - a.window : 0;
  float m = sink[(uint64_t) t * a.heads + h], z = 1.0f, acc = 0;  // the sink: its logit, weight 1 at its max, value 0
  for (uint32_t s0 = lo; s0 <= qi; s0 += kChunk) {
    const uint32_t s = s0 + i, n = qi + 1 - s0 < kChunk ? qi + 1 - s0 : kChunk;
    float l = -INFINITY;
    if (i < n) {
      const float* kr = k + ((uint64_t) s * a.kv_heads + kvh) * D;
      float d = 0;
      for (uint32_t c = 0; c < D; c++) d += qs[c] * kr[c];
      const uint32_t dist = qi - s;
      l = d * a.scaling + (dist < a.extent ? rl[dist] : 0.0f);
    }
    const float cm = block_max<kChunk>(l, red), nm = fmaxf(m, cm), f = expf(m - nm);
    const float e = i < n ? expf(l - nm) : 0.0f;
    p[i] = e;
    const float cs = block_sum<kChunk>(e, red);
    z = z * f + cs, acc *= f, m = nm;
    for (uint32_t j = 0; j < n; j++) acc += p[j] * v[((uint64_t) (s0 + j) * a.kv_heads + kvh) * D + i];
    __syncthreads();
  }
  const uint64_t o = (uint64_t) t * a.heads * D + (uint64_t) h * D + i;
  out[o] = acc / z * (gate ? sigmoid(gate[o]) : 1.0f);
}

// the prompt's attention on WMMA (16 x 16 x 16, f16 in, f32 sums): a block a 16-token tile and KV head, a wave a
// query head of its group; each 16-key tile's K and V^T in LDS once for the group. a lane holds rows 2 j + half
// (j < 8) at column lane % 16 of a tile's scores and of each 16-dim slice of O (q4x_attn.hip's layout)
constexpr int kFaTile = INK2_FA_TILE;
constexpr int kFaHead = INK2_MAX_HEAD;
constexpr int kFaRow = kFaHead + 8;  // f16s a row in LDS, then a pad
constexpr int kFaKey = kFaTile + 8;
constexpr int kFaRowTiles = 4;
constexpr uint32_t kFaUnit = INK2_FA_UNIT;  // a block's row tiles (INK2_FA_RT 1 / 2 / 4 for a sweep; 16K full: 756 / 716 / 638 ms)
typedef _Float16 fa_h16 __attribute__((ext_vector_type(16)));
typedef float fa_f8 __attribute__((ext_vector_type(8)));

__device__ inline float fa_row_max(float v) {
  for (int o = 1; o < kFaTile; o <<= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  return v;
}
__device__ inline float fa_row_sum(float v) {
  for (int o = 1; o < kFaTile; o <<= 1) v += __shfl_xor(v, o, 32);
  return v;
}

// a unit's online softmax folded into the running one (the sink's start): M the max, each side scaled to it - the
// prompt's kernel and a split decode's fold the same units in the same order with this, so the same bits
__device__ inline void fa_fold_coef(float& gm, float& gl, float m, float l, float& a, float& b) {
  const float M = fmaxf(gm, m);
  a = expf(gm - M), b = m == -INFINITY ? 0.0f : expf(m - M);
  gl = gl * a + l * b, gm = M;
}

// G waves; Q's A fragments in registers (a lane its row's 8 x 16 halves), the next tile's K and V prefetched into
// registers while this one's WMMAs run - LDS ~16 KB a block (K, V^T, P), several blocks a CU
// KV: the keys' and values' type in memory - F32 (rounded to f16 as loaded) or f16 (a cache's: the same values).
// a query's keys in units of kFaUnit at absolute positions (pos0 key 0's): each unit's online softmax from nothing,
// folded into the running one in order. SPLIT: a block a unit (blockIdx.z, from unit u0), its partial (m, l, o)
// written to part for ink2_fa_fold - a decode's units in parallel, the prompt's fold's bits
template <int G, int RT, class KV, bool SPLIT>
__global__ __launch_bounds__(G * RT * 32) void fa_kernel(Ink2Attn a, const float* q, const float* rel, const float* sink,
                                                     const float* gate, uint32_t n_tok, const KV* k,
                                                     const KV* v, uint32_t n_keys, uint32_t pos0,
                                                     const float* rel_proj, float* out, uint32_t u0, float* part) {
  constexpr int kNth = G * RT * 32, kItems = (kFaTile * kFaHead + kNth - 1) / kNth;
  __shared__ __attribute__((aligned(32))) _Float16 ks[kFaTile * kFaRow], vt[kFaHead * kFaKey];
  __shared__ __attribute__((aligned(32))) _Float16 ps[G * RT][kFaTile * kFaKey];
  // a block RT row tiles of 16 tokens; a wave (row tile, query head) - every K / V tile shared by the block's waves
  const uint32_t wave = threadIdx.x / 32, lane = threadIdx.x % 32, half = lane >> 4, col = lane % kFaTile;
  const uint32_t tb = blockIdx.x * kFaTile * RT, t0 = tb + wave / G * kFaTile, kvh = blockIdx.y, H = a.heads;
  const uint32_t h = kvh * G + wave % G, base = n_keys - n_tok;  // token t is key base + t
  fa_h16 qa[kFaHead / kFaTile];
  {
    const uint32_t t = t0 + col;
    const float* qr = q + ((uint64_t) t * H + h) * kFaHead;
    for (int g = 0; g < kFaHead / kFaTile; g++)
      for (int e = 0; e < kFaTile; e++) qa[g][e] = (_Float16) (t < n_tok ? qr[g * kFaTile + e] : 0.0f);
  }
  // the running softmax: the sink a logit of its own, its value zero (gm, gl, go); the unit's from nothing (m, l, o)
  float m[8], l[8], gm[8], gl[8];
  fa_f8 o[kFaHead / kFaTile] = {}, go[SPLIT ? 1 : kFaHead / kFaTile] = {};
  for (int j = 0; j < 8; j++) {
    const uint32_t t = t0 + 2 * j + half;
    gm[j] = t < n_tok ? sink[(uint64_t) t * H + h] : 0.0f, gl[j] = 1.0f, m[j] = -INFINITY, l[j] = 0.0f;
  }
  // the block's key range (its first row tile's lowest to its last's highest); a wave's own rows masked within it
  const uint32_t last = tb + kFaTile * RT - 1 < n_tok - 1 ? tb + kFaTile * RT - 1 : n_tok - 1;
  const uint32_t qlo = base + tb, qhi = base + last;
  // the first key tile at a multiple of kFaTile: a query's keys cut into the same tiles whatever block it is in
  // (the tiles before its window wholly masked: no-ops), so its bits are its own - given keys whose index is their
  // position mod kFaTile (INK2_FA_TILE: the engine's windows start so)
  uint32_t klo = a.window && qlo + 1 > a.window ? (qlo + 1 - a.window) / kFaTile * kFaTile : 0, khi = qhi;
  if constexpr (SPLIT) {  // this block's unit alone
    const uint32_t u = u0 + blockIdx.z, ua = u * kFaUnit;  // the unit's keys as indices (it may start before key 0)
    const uint32_t us = ua > pos0 ? ua - pos0 : 0, ue = ua + kFaUnit - 1 - pos0;
    klo = klo > us ? klo : us, khi = khi < ue ? khi : ue;
  }
  // a tile's K and V into registers (keys past the last zero): F32 a value a thread at a time; f16 (a cache's) 8
  // halves a 16-byte load - the same values either way
  constexpr bool kHalf = sizeof(KV) == 2;
  constexpr int kVec = 8, kVecs = kFaTile * kFaHead / kVec, kVItems = (kVecs + kNth - 1) / kNth;
  typedef _Float16 fa_h8 __attribute__((ext_vector_type(8)));
  float kr[kHalf ? 1 : kItems], vr[kHalf ? 1 : kItems];
  fa_h8 kh[kHalf ? kVItems : 1], vh[kHalf ? kVItems : 1];
  auto fetch = [&](uint32_t kb) {
    if constexpr (kHalf)
      for (int it = 0; it < kVItems; it++) {
        const uint32_t i = threadIdx.x + it * kNth, r = i / (kFaHead / kVec), d = i % (kFaHead / kVec) * kVec;
        const bool in = i < (uint32_t) kVecs && kb + r < n_keys;
        const uint64_t at = ((uint64_t) (kb + r) * a.kv_heads + kvh) * kFaHead + d;
        kh[it] = in ? *(const fa_h8*) (k + at) : fa_h8{}, vh[it] = in ? *(const fa_h8*) (v + at) : fa_h8{};
      }
    else
      for (int it = 0; it < kItems; it++) {
        const uint32_t i = threadIdx.x + it * kNth, r = i / kFaHead, d = i % kFaHead, key = kb + r;
        const bool in = i < kFaTile * kFaHead && key < n_keys;
        const uint64_t at = ((uint64_t) key * a.kv_heads + kvh) * kFaHead + d;
        kr[it] = in ? (float) k[at] : 0.0f, vr[it] = in ? (float) v[at] : 0.0f;
      }
  };
  if (klo <= khi) fetch(klo);
  for (uint32_t kb = klo; kb <= khi; kb += kFaTile) {
    __syncthreads();  // the last tile's K, V read by every wave
    if constexpr (kHalf)
      for (int it = 0; it < kVItems; it++) {
        const uint32_t i = threadIdx.x + it * kNth, r = i / (kFaHead / kVec), d = i % (kFaHead / kVec) * kVec;
        if (i >= (uint32_t) kVecs) continue;
        *(fa_h8*) (ks + r * kFaRow + d) = kh[it];
        for (int e = 0; e < kVec; e++) vt[(d + e) * kFaKey + r] = vh[it][e];
      }
    else
      for (int it = 0; it < kItems; it++) {
        const uint32_t i = threadIdx.x + it * kNth, r = i / kFaHead, d = i % kFaHead;
        if (i < kFaTile * kFaHead) ks[r * kFaRow + d] = (_Float16) kr[it], vt[d * kFaKey + r] = (_Float16) vr[it];
      }
    __syncthreads();
    if (kb + kFaTile <= khi) fetch(kb + kFaTile);  // the next tile's, in flight under this one's WMMAs
    fa_f8 sacc = {};
    for (int g = 0; g < kFaHead / kFaTile; g++) {
      const fa_h16 kb16 = *(const fa_h16*) (ks + col * kFaRow + g * kFaTile);
      sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[g], kb16, sacc);
    }
    const uint32_t key = kb + col;
    float alpha[8];
    for (int j = 0; j < 8; j++) {
      const uint32_t t = t0 + 2 * j + half, qi = base + t;
      const bool valid = t < n_tok && key <= qi && (!a.window || qi - key < a.window);
      float s = -INFINITY;
      if (valid) {
        const uint32_t dist = qi - key;
        s = sacc[j] * a.scaling;
        if (dist < a.extent) {
          const float* rr = rel + ((uint64_t) t * H + h) * a.d_rel;
          float b = 0;
          for (uint32_t r = 0; r < a.d_rel; r++) b += rr[r] * rel_proj[(uint64_t) r * a.extent + dist];
          s += b;
        }
      }
      const float mn = fmaxf(m[j], fa_row_max(s));
      const float pv = s == -INFINITY ? 0.0f : expf(s - mn);
      alpha[j] = m[j] == mn ? 1.0f : expf(m[j] - mn);  // (a unit's start: -inf both, nothing held)
      l[j] = l[j] * alpha[j] + fa_row_sum(pv), m[j] = mn;
      ps[wave][(2 * j + half) * kFaKey + col] = (_Float16) pv;
    }
    __builtin_amdgcn_wave_barrier();
    const fa_h16 pa = *(const fa_h16*) (ps[wave] + col * kFaKey);
    // a row whose own key is in this tile (not its last): its p . v over the tile's values up to its own alone - the
    // WMMA's sum moves with a value whose p is 0, and the keys past a query are the prompt's next tokens in one
    // step, none in a step that ends at it (a decode's). cut c: the rows at key kb + c (the wave's rows the valid)
    const int64_t c_lo = (int64_t) (base + t0) - kb, c_last = c_lo + (int64_t) (n_tok - 1 - t0 < 15 ? n_tok - 1 - t0 : 15);
    const int cmin = c_lo > 0 ? (int) c_lo : 0, cmax = c_last < kFaTile - 2 ? (int) c_last : kFaTile - 2;
    for (int nt = 0; nt < kFaHead / kFaTile; nt++) {
      for (int j = 0; j < 8; j++) o[nt][j] *= alpha[j];
      const fa_h16 bv = *(const fa_h16*) (vt + (nt * kFaTile + col) * kFaKey);
      const fa_f8 prev = o[nt];
      o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, bv, prev);
      for (int c = cmin; c <= cmax; c++) {
        fa_h16 bc = bv;
        for (int e = c + 1; e < kFaTile; e++) bc[e] = (_Float16) 0.0f;
        const fa_f8 d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, bc, prev);
        for (int j = 0; j < 8; j++)
          if ((int64_t) (base + t0 + 2 * j + half) == (int64_t) kb + c) o[nt][j] = d[j];
      }
    }
    __builtin_amdgcn_wave_barrier();
    if constexpr (!SPLIT)
      if ((pos0 + kb + kFaTile) % kFaUnit == 0 || kb + kFaTile > khi) {  // a unit's end: folded, the next from nothing
        for (int j = 0; j < 8; j++) {
          float ca, cb;
          fa_fold_coef(gm[j], gl[j], m[j], l[j], ca, cb);
          for (int nt = 0; nt < kFaHead / kFaTile; nt++) go[nt][j] = go[nt][j] * ca + o[nt][j] * cb, o[nt][j] = 0.0f;
          m[j] = -INFINITY, l[j] = 0.0f;
        }
      }
  }
  for (int j = 0; j < 8; j++) {
    const uint32_t t = t0 + 2 * j + half;
    if (t >= n_tok) continue;
    if constexpr (SPLIT) {  // the unit's partial: m, l, then o (ink2_fa_fold's)
      float* pp = part + (((uint64_t) blockIdx.z * n_tok + t) * H + h) * (kFaHead + 2);
      if (col == 0) pp[0] = m[j], pp[1] = l[j];
      for (int nt = 0; nt < kFaHead / kFaTile; nt++) pp[2 + nt * kFaTile + col] = o[nt][j];
    } else
      for (int nt = 0; nt < kFaHead / kFaTile; nt++) {
        const uint64_t i = ((uint64_t) t * H + h) * kFaHead + nt * kFaTile + col;
        out[i] = go[nt][j] / gl[j] * (gate ? sigmoid(gate[i]) : 1.0f);
      }
  }
}

// a split decode's units folded in order into the sink's start (fa_kernel's fold): a block a (token, head), a
// thread a value dim
__global__ void fa_fold_kernel(const float* part, uint32_t units, uint32_t n_tok, uint32_t H, const float* sink,
                               const float* gate, float* out) {
  const uint32_t t = blockIdx.x / H, h = blockIdx.x % H, d = threadIdx.x;
  float gm = sink[(uint64_t) t * H + h], gl = 1.0f, go = 0.0f;
  for (uint32_t u = 0; u < units; u++) {
    const float* pp = part + (((uint64_t) u * n_tok + t) * H + h) * (kFaHead + 2);
    float ca, cb;
    fa_fold_coef(gm, gl, pp[0], pp[1], ca, cb);
    go = go * ca + pp[2 + d] * cb;
  }
  const uint64_t i = ((uint64_t) t * H + h) * kFaHead + d;
  out[i] = go / gl * (gate ? sigmoid(gate[i]) : 1.0f);
}

}  // namespace

template <class KV>
static bool fa_launch(const Ink2Attn* a, const float* q, const float* rel, const float* sink, const float* gate,
                      uint32_t n_tok, const KV* k, const KV* v, uint32_t n_keys, uint32_t pos0, const float* rel_proj,
                      float* out, float* scratch, void* stream) {
  if (!a || !q || !rel || !sink || !k || !v || !rel_proj || !out || !n_tok || n_tok > n_keys ||
      a->head_dim != INK2_MAX_HEAD || !a->kv_heads || a->heads % a->kv_heads ||
      a->heads / a->kv_heads > INK2_FA_MAX_GROUP || !a->d_rel || a->d_rel > INK2_MAX_REL || !a->extent ||
      a->extent > INK2_MAX_EXTENT || pos0 % kFaTile)
    return false;
  const uint32_t G = a->heads / a->kv_heads;
  if (scratch && n_tok <= (uint32_t) kFaTile) {  // a decode's: a block a unit and KV head, then the fold
    const uint32_t qlo = n_keys - n_tok, qhi = n_keys - 1;
    const uint32_t klo = a->window && qlo + 1 > a->window ? (qlo + 1 - a->window) / kFaTile * kFaTile : 0;
    const uint32_t u0 = (pos0 + klo) / kFaUnit, units = (pos0 + qhi) / kFaUnit - u0 + 1;
    const dim3 grid(1, a->kv_heads, units);
#define FS(g)                                                                                                        \
  if (G == g) {                                                                                                      \
    fa_kernel<g, 1, KV, true><<<grid, g * 32, 0, st(stream)>>>(*a, q, rel, sink, gate, n_tok, k, v, n_keys, pos0,     \
                                                               rel_proj, out, u0, scratch);                          \
    fa_fold_kernel<<<n_tok * a->heads, kFaHead, 0, st(stream)>>>(scratch, units, n_tok, a->heads, sink, gate, out);   \
    return ok();                                                                                                     \
  }
    FS(1) FS(2) FS(4) FS(6) FS(8)
#undef FS
    return false;
  }
  static const int rt = std::getenv("INK2_FA_RT") ? std::atoi(std::getenv("INK2_FA_RT")) : kFaRowTiles;
  const uint32_t R = rt == 1 || rt == 2 || rt == 4 ? (uint32_t) rt : kFaRowTiles;
  const dim3 grid_fa((n_tok + kFaTile * R - 1) / (kFaTile * R), a->kv_heads);
#define FA(g, r)                                                                                                    \
  if (G == g && R == r) {                                                                                           \
    fa_kernel<g, r, KV, false><<<grid_fa, g * r * 32, 0, st(stream)>>>(*a, q, rel, sink, gate, n_tok, k, v, n_keys,  \
                                                                       pos0, rel_proj, out, 0, nullptr);            \
    return ok();                                                                                                    \
  }
  FA(1, 1) FA(2, 1) FA(4, 1) FA(6, 1) FA(8, 1) FA(1, 2) FA(2, 2) FA(4, 2) FA(6, 2) FA(8, 2) FA(1, 4) FA(2, 4) FA(4, 4)
  FA(6, 4)
#undef FA
  return false;
}

extern "C" uint64_t ink2_fa_scratch(uint32_t heads, uint32_t max_keys) {
  return ((uint64_t) max_keys / kFaUnit + 2) * kFaTile * heads * (kFaHead + 2) * sizeof(float);
}

extern "C" bool ink2_fa(const Ink2Attn* a, const float* q, const float* rel, const float* sink, const float* gate,
                        uint32_t n_tok, const float* k, const float* v, uint32_t n_keys, uint32_t pos0,
                        const float* rel_proj, float* out, float* scratch, void* stream) {
  return fa_launch<float>(a, q, rel, sink, gate, n_tok, k, v, n_keys, pos0, rel_proj, out, scratch, stream);
}

extern "C" bool ink2_fa16(const Ink2Attn* a, const float* q, const float* rel, const float* sink, const float* gate,
                          uint32_t n_tok, const uint16_t* k, const uint16_t* v, uint32_t n_keys, uint32_t pos0,
                          const float* rel_proj, float* out, float* scratch, void* stream) {
  return fa_launch<_Float16>(a, q, rel, sink, gate, n_tok, (const _Float16*) k, (const _Float16*) v, n_keys, pos0,
                             rel_proj, out, scratch, stream);
}

__global__ void to_f16_kernel(const float* x, uint64_t n, _Float16* y) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) y[i] = (_Float16) x[i];
}

extern "C" bool ink2_to_f16(const float* x, uint64_t n, uint16_t* y, void* stream) {
  if (!x || !y || !n) return false;
  to_f16_kernel<<<grid(n), kThreads, 0, st(stream)>>>(x, n, (_Float16*) y);
  return ok();
}

extern "C" bool ink2_clip_rows(float* x, uint32_t n_tok, uint32_t per_tok, uint32_t dim, const float* w, float clip,
                               Ink2Tau tau, void* stream) {
  if (!x || !n_tok || !per_tok || !dim || (tau.alpha > 0 && !(tau.floor > 0))) return false;
  clip_rows_kernel<<<n_tok * per_tok, kThreads, 0, st(stream)>>>(x, per_tok, dim, w, clip, tau);
  return ok();
}

extern "C" bool ink2_attn(const Ink2Attn* a, const float* q, const float* rel, const float* sink, const float* gate,
                          uint32_t n_tok, const float* k, const float* v, uint32_t n_keys, const float* rel_proj,
                          float* out, void* stream) {
  if (!a || !q || !rel || !sink || !k || !v || !rel_proj || !out || !n_tok || n_tok > n_keys ||
      a->head_dim != INK2_MAX_HEAD || !a->kv_heads || a->heads % a->kv_heads || !a->d_rel ||
      a->d_rel > INK2_MAX_REL || !a->extent || a->extent > INK2_MAX_EXTENT)
    return false;
  attn_kernel<<<n_tok * a->heads, a->head_dim, 0, st(stream)>>>(*a, q, rel, sink, gate, n_tok, k, v, n_keys, rel_proj,
                                                                out);
  return ok();
}

extern "C" bool ink2_softcap(float* x, uint64_t n, float cap, void* stream) {
  if (!x || !n || !(cap > 0)) return false;
  softcap_kernel<<<grid(n), kThreads, 0, st(stream)>>>(x, n, cap);
  return ok();
}

extern "C" bool ink2_swiglu(const float* g, const float* u, uint32_t n, float gcap, float ucap, uint32_t n_tok,
                            float* h, void* stream) {
  if (!g || !u || !h || !n || !n_tok || !(gcap > 0) || !(ucap > 0)) return false;
  const uint64_t all = (uint64_t) n * n_tok;
  swiglu_kernel<<<grid(all), kThreads, 0, st(stream)>>>(g, u, all, gcap, ucap, h);
  return ok();
}

extern "C" bool ink2_swiglu_rows(const float* g, const float* u, const float* sg, const float* su, uint32_t n,
                                 float gcap, float ucap, uint32_t rows, float* h, void* stream) {
  if (!g || !u || !sg || !su || !h || !n || !rows || !(gcap > 0) || !(ucap > 0)) return false;
  const uint64_t all = (uint64_t) n * rows;
  swiglu_rows_kernel<<<grid(all), kThreads, 0, st(stream)>>>(g, u, sg, su, n, all, gcap, ucap, h);
  return ok();
}

extern "C" bool ink2_moe_sum(float* y, const float* x, const float* a, const int32_t* row, uint32_t k, uint32_t n_tok,
                             uint32_t dim, void* stream) {
  if (!y || !x || !a || !row || !k || !n_tok || !dim) return false;
  const uint64_t all = (uint64_t) n_tok * dim;
  moe_sum_kernel<<<grid(all), kThreads, 0, st(stream)>>>(y, x, a, row, k, all, dim);
  return ok();
}

extern "C" bool ink2_gram(const float* x, uint32_t n_tok, uint32_t dim, float* h, void* stream) {
  if (!x || !h || !n_tok || !dim) return false;
  const dim3 g((dim + kGram - 1) / kGram, (dim + kGram - 1) / kGram), b(kGram, kGram);
  gram_kernel<<<g, b, 0, st(stream)>>>(x, n_tok, dim, h);
  return ok();
}

extern "C" bool ink2_conv(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, const float* w,
                          const float* hist, float res, float scale, float* y, void* stream) {
  if (!x || !w || !y || !n_tok || !ch || taps < 2 || taps > INK2_MAX_TAPS) return false;
  conv_kernel<<<grid((uint64_t) n_tok * ch), kThreads, 0, st(stream)>>>(x, n_tok, ch, taps, w, hist, res, scale, y);
  return ok();
}

extern "C" bool ink2_conv_dyn(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, uint32_t heads,
                              const float* w, const float* dyn, const float* hist, float res, float scale, float* y,
                              void* stream) {
  if (!x || !w || !dyn || !y || !n_tok || !ch || taps < 2 || taps > INK2_MAX_TAPS || !heads || ch % heads)
    return false;
  conv_dyn_kernel<<<grid((uint64_t) n_tok * ch), kThreads, 0, st(stream)>>>(x, n_tok, ch, taps, heads, w, dyn, hist,
                                                                             res, scale, y);
  return ok();
}

extern "C" bool ink2_attn_res(const float* query, uint32_t queries, const float* blocks, uint32_t n_blocks,
                              uint32_t b_stride, const float* partial, uint32_t n_tok, uint32_t dim, float eps,
                              float scale, float* out, void* stream) {
  if (!query || !partial || !out || (n_blocks && !blocks) || !queries || queries > INK2_MAX_QUERIES ||
      n_blocks + 1 > INK2_MAX_RES || n_blocks > b_stride || !n_tok || !dim)
    return false;
  attn_res_kernel<<<n_tok, kThreads, 0, st(stream)>>>(query, queries, blocks, n_blocks, b_stride, partial, dim, eps,
                                                      scale, n_tok, out);
  return ok();
}

__global__ void ink2_scale_kernel(float* x, uint64_t n, float s) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) x[i] *= s;
}

__global__ void ink2_axpy_kernel(float* y, const float* x, uint64_t n, float a) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) y[i] += a * x[i];
}

extern "C" bool ink2_scale(float* x, uint64_t n, float s, void* stream) {
  if (!x || !n) return false;
  ink2_scale_kernel<<<grid(n), kThreads, 0, st(stream)>>>(x, n, s);
  return ok();
}

extern "C" bool ink2_axpy(float* y, const float* x, uint64_t n, float a, void* stream) {
  if (!y || !x || !n) return false;
  ink2_axpy_kernel<<<grid(n), kThreads, 0, st(stream)>>>(y, x, n, a);
  return ok();
}
