// Hy4's MLA attention (the attention core's: attn_core.hip's MLA rows, docs/attention-core.md C3) - hy4_mla_attn's
// code moved whole, its bits: a prompt's read by mla_mm_kernel (f16 operands on the matrix cores, F32 sums), else
// the splits (F32: a split of a token's keys a block, merged in order with the sink)
#pragma once
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "launch.h"
#include "mla_row.h"

extern "C" {
#include "osmos/gpu/hy4_ops.h"
}

namespace attn_hy4 {

using launches::ok;
using launches::st;
using namespace mla;

constexpr int kThreads = 256;
constexpr uint32_t kWaves = kThreads / 32;

// MLA's attention, flash decoding: absorbed MLA is MQA (every head reads the same rows), so a block takes a group of
// kHeadGroup heads (a wave each) over a split of kSplitKeys of its token's keys: the rows loaded once into LDS a tile
// of kTileKeys at a time for every head, each wave its head's scores (a lane 18 of the 576 dims, an xor tree), an
// online softmax in the keys' order, its latent output (a lane 16 of the 512 dims). partials (max, sum, output) a
// (token, head, split); the merge their order with the sink. the splits a token's own keys': its bits whatever the step
// kSplitKeys, kTileKeys: tools/bench/mla_bench's sweep
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
// a decode's and a verify's steps (not a read) stay on mla_split_kernel (their bits the reference-checked); a read's
// at any size on this one (its bits a prompt's however it is split)
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


inline uint64_t scratch_floats(uint32_t heads) {
  return (uint64_t) kTokChunk * heads * kMaxSplits * (HY4_MLA_MAX_LATENT + 2);
}

inline bool run(const float* qa, const void* cache, uint32_t slots, bool q8, const float* sinks, uint32_t heads,
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
  if (read && heads % 16 == 0 && latent % 64 == 0 && (latent + rope_dim) % 16 == 0 &&
      latent + rope_dim <= kMlaRow && rope_dim <= 64) {
    if (heads % 64 == 0)
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

}  // namespace attn_hy4
