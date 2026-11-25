// the attention core's kernels (osmos/gpu/attn_core.h): Inkling's prompt attention and its split decode, the core's
// skeleton - its units, tiles and fold the sums' order every family takes (docs/attention-core.md)
#pragma once
#include <type_traits>

#include "block.h"
#include "launch.h"
#include "mm.h"

extern "C" {
#include "osmos/gpu/attn_core.h"
}

namespace attn {

// a launch's view of its core's desc: the fields the kernels read
struct AttnArgs {
  uint32_t heads, kv_heads, d_rel, extent, window;
  float scaling;
};

// the prompt's attention on WMMA (16 x 16 x 16, f16 in, f32 sums): a block a 16-token tile and KV head, a wave a
// query head of its group; each 16-key tile's K and V^T in LDS once for the group. a lane holds rows 2 j + half
// (j < 8) at column lane % 16 of a tile's scores and of each 16-dim slice of O (q4x_attn.hip's layout)
constexpr int kFaTile = ATTN_TILE;
constexpr int kFaHead = ATTN_HEAD;
constexpr int kFaRow = kFaHead + 8;  // f16s a row in LDS, then a pad
constexpr int kFaKey = kFaTile + 8;
constexpr int kFaRowTiles = 4;
// a split decode's waves past which a call's blocks keep one wave (more waves then slower: the GPU full, and only a
// one-wave block reads Q8 keys straight into registers), tuned on gfx1151 at Mini's shape
constexpr uint32_t kFaDecodeWaves = 256;
constexpr uint32_t kFaUnit = ATTN_UNIT;  // a unit's keys (absolute positions): each its own softmax, folded in order
typedef _Float16 fa_h16 __attribute__((ext_vector_type(16)));
typedef float fa_f8 __attribute__((ext_vector_type(8)));

// a row's 16 lanes' max and sum, each a butterfly over lane ^ 1, 2, 4, 8 - by DPP's row_xmask (within a row of 16:
// the same partners as __shfl_xor's, so the same values in the same order; not ds_bpermute's trip through LDS)
template <int O>
__device__ inline float fa_xor(float v) {
  return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v), 0x160 + O, 0xF, 0xF, false));
}
// the max a v_max_f32 with its DPP operand: one instruction a step, not fmaxf's move, canonicalize and max (the
// scores never NaN - a finite value or -inf - so the canonicalize kept nothing: the same values)
template <int O>
__device__ inline float fa_max_xor(float v) {
  float r;
  asm volatile("v_max_f32_dpp %0, %1, %1 row_xmask:%2 row_mask:0xf bank_mask:0xf bound_ctrl:1" : "=v"(r) : "v"(v), "i"(O));
  return r;
}
__device__ inline float fa_row_max(float v) {
  v = fa_max_xor<1>(v), v = fa_max_xor<2>(v), v = fa_max_xor<4>(v);
  return fa_max_xor<8>(v);
}
__device__ inline float fa_row_sum(float v) {
  v += fa_xor<1>(v), v += fa_xor<2>(v), v += fa_xor<4>(v);
  return v + fa_xor<8>(v);
}

// a unit's online softmax folded into the running one (the sink's start): M the max, each side scaled to it - the
// prompt's kernel and a split decode's fold the same units in the same order with this, so the same bits
__device__ inline void fa_fold_coef(float& gm, float& gl, float m, float l, float& a, float& b) {
  const float M = fmaxf(gm, m);
  a = expf(gm - M), b = m == -INFINITY ? 0.0f : expf(m - M);
  gl = gl * a + l * b, gm = M;
}

// a full layer's cache as Q8 records (ink2_ops.h's ATTN_KV8_RECORD): the kernel's KV a byte of them
struct Kv8 {
  uint8_t b;
};
typedef _Float16 fa_h8 __attribute__((ext_vector_type(8)));

// a record's 8 values from d as loaded (its int8s, its block's f16 scale), then as values: q times the scale, rounded
// to f16 (Q8 records' values) - loaded and turned into values apart, so a tile's loads are in flight under the last's work
struct Kv8Raw {
  uint2 q;
  _Float16 s;
};
__device__ inline Kv8Raw kv8_raw(const uint8_t* rec, uint32_t d) {
  return {*(const uint2*) (rec + d), *(const _Float16*) (rec + ATTN_HEAD + d / ATTN_KV8_BLOCK * 2)};
}
// (the product in f16: an int8 and an f16 exact in it, their F32 product exact - 19 significant bits - so the f16
// multiply's one rounding is the F32 product's to f16: the same bits, two values a packed multiply)
typedef _Float16 fa_h2 __attribute__((ext_vector_type(2)));
__device__ inline fa_h8 kv8_vals(const Kv8Raw& r) {
  const fa_h2 s2 = {r.s, r.s};
  const int8_t* q = (const int8_t*) &r.q;
  fa_h8 out;
  for (int e = 0; e < 8; e += 2) {
    const fa_h2 p = fa_h2{(_Float16) q[e], (_Float16) q[e + 1]} * s2;
    out[e] = p[0], out[e + 1] = p[1];
  }
  return out;
}

// keys Q8 records, values Q5: the kernel's KV a byte of them
struct Kv85 {
  uint8_t b;
};

// a Q5 record's 8 values from d (a multiple of 8) as loaded (nibbles, fifth bits, scale), then as values: (q - 16)
// times its block's f16 scale, rounded to f16
struct Kv5Raw {
  uint32_t nib, hi;
  _Float16 s;
};
__device__ inline Kv5Raw kv5_raw(const uint8_t* rec, uint32_t d) {
  constexpr uint32_t H = ATTN_HEAD;
  return {*(const uint32_t*) (rec + d / 2), rec[H / 2 + d / 8],
          *(const _Float16*) (rec + H / 2 + H / 8 + d / ATTN_KV8_BLOCK * 2)};
}
__device__ inline fa_h8 kv5_vals(const Kv5Raw& r) {  // (kv8_vals' f16 product: q - 16 exact in f16 too)
  const fa_h2 s2 = {r.s, r.s};
  fa_h8 out;
  for (int e = 0; e < 8; e += 2) {
    const int q0 = (int) ((r.nib >> (4 * e) & 15u) | (r.hi >> e & 1u) << 4),
              q1 = (int) ((r.nib >> (4 * e + 4) & 15u) | (r.hi >> (e + 1) & 1u) << 4);
    const fa_h2 p = fa_h2{(_Float16) (q0 - 16), (_Float16) (q1 - 16)} * s2;
    out[e] = p[0], out[e + 1] = p[1];
  }
  return out;
}

// G waves; Q's A fragments in registers (a lane its row's 8 x 16 halves), the next tile's K and V prefetched into
// registers while this one's WMMAs run - LDS ~16 KB a block (K, V^T, P), several blocks a CU
// KV: the keys' and values' type in memory - F32 (rounded to f16 as loaded) or f16 (a cache's: the same values).
// a query's keys in units of kFaUnit at absolute positions (pos0 key 0's): each unit's online softmax from nothing,
// folded into the running one in order. SPLIT: a block a unit (blockIdx.z, from unit u0), its partial (m, l, o)
// written to part for the fold - a decode's units in parallel, the prompt's fold's bits
template <int G, int RT, class KV, bool SPLIT, int W = G * RT>
__global__ __launch_bounds__(W * 32) void fa_kernel(AttnArgs a, const float* q, const float* rel, const float* sink,
                                                     const float* gate, uint32_t n_tok, const KV* k,
                                                     const KV* v, uint32_t n_keys, uint32_t pos0,
                                                     const float* rel_proj, float* out, uint32_t u0, float* part) {
  // W waves: the prompt's G RT (a row tile's heads); a split decode's as many as its (token, head) rows fill - a wave
  // without rows still holds its registers, so a decode's token runs one wave, not G (more blocks resident)
  static_assert(SPLIT || W == G * RT, "the prompt's waves its row tiles' heads");
  // ST 16-key tiles a stage (a prompt's 4: one pair of barriers and one prefetch for 64 keys; a split decode's 1) - each
  // tile's scores, softmax, P V and fold as before, in the same order: the stage changes no bit
  constexpr int ST = SPLIT ? 1 : 4, kStage = kFaTile * ST, kVKey = kStage + 8;
  constexpr int kNth = W * 32, kItems = (kStage * kFaHead + kNth - 1) / kNth;
  // a one-wave decode over a full layer's Q8 records: its keys straight from each lane's own record into the scores'
  // B fragments (a lane its key's, as LDS would hand them) - no K tile in LDS, so more blocks resident
  constexpr bool kKD = sizeof(KV) == 1 && SPLIT && W == 1;
  __shared__ __attribute__((aligned(32))) _Float16 ks[kKD ? 8 : kStage * kFaRow], vt[kFaHead * kVKey];
  __shared__ __attribute__((aligned(32))) _Float16 ps[W][kFaTile * kFaKey];
  // a block RT row tiles of 16 tokens; a wave (row tile, query head) - every K / V tile shared by the block's waves
  const uint32_t wave = threadIdx.x / 32, lane = threadIdx.x % 32, half = lane >> 4, col = lane % kFaTile;
  const uint32_t tb = blockIdx.x * kFaTile * RT, t0 = tb + wave / G * kFaTile, kvh = blockIdx.y, H = a.heads;
  const uint32_t h = kvh * G + wave % G, base = n_keys - n_tok;  // token t is key base + t
  // a row's token and query head: the prompt's a wave's head, its rows 16 tokens; a split decode's rows (token, head)
  // pairs, wave w's the w * 16th on (a token's G heads in G rows: one wave for a decode's token, not G of one row each)
  // - each row's sums its own, so the same bits either way; a wave past the pairs only loads the tiles
  auto rtok = [&](uint32_t r) -> uint32_t { return SPLIT ? (wave * kFaTile + r) / G : t0 + r; };
  auto rhead = [&](uint32_t r) -> uint32_t { return SPLIT ? kvh * G + (wave * kFaTile + r) % G : h; };
  const bool active = !SPLIT || wave * kFaTile < n_tok * G;
  fa_h16 qa[kFaHead / kFaTile];
  {
    const uint32_t t = rtok(col);
    const float* qr = q + ((uint64_t) t * H + rhead(col)) * kFaHead;
    for (int g = 0; g < kFaHead / kFaTile; g++)
      for (int e = 0; e < kFaTile; e++) qa[g][e] = (_Float16) (t < n_tok ? qr[g * kFaTile + e] : 0.0f);
  }
  // the running softmax: the sink a logit of its own, its value zero (gm, gl, go); the unit's from nothing (m, l, o)
  float m[8], l[8];
  // the running fold's (gm, gl) a row's in LDS: read at a unit's end alone (every 16 tiles), not 16 registers a lane
  // (a row's 16 lanes the same values; its lane 0 writes) - fewer spills
  __shared__ float gml[W][kFaTile][2];
  auto gm = [&](int j) -> float& { return gml[wave][2 * j + half][0]; };
  auto gl = [&](int j) -> float& { return gml[wave][2 * j + half][1]; };
  // the unit's (m, l, o) in registers; the prompt's running fold's o in its out rows (each lane its own elements: zero
  // first, a unit's end its read and write, the end its division) - not 64 more registers a wave, so fewer spills
  fa_f8 o[kFaHead / kFaTile] = {};
  auto out_at = [&](uint32_t t, int nt) -> uint64_t { return ((uint64_t) t * H + h) * kFaHead + nt * kFaTile + col; };
  if constexpr (!SPLIT)
    for (int j = 0; j < 8; j++)
      if (rtok(2 * j + half) < n_tok)
        for (int nt = 0; nt < kFaHead / kFaTile; nt++) out[out_at(rtok(2 * j + half), nt)] = 0.0f;
  for (int j = 0; j < 8; j++) {
    const uint32_t t = rtok(2 * j + half);
    gm(j) = t < n_tok ? sink[(uint64_t) t * H + rhead(2 * j + half)] : 0.0f, gl(j) = 1.0f, m[j] = -INFINITY, l[j] = 0.0f;
  }
  // the block's key range (its first row tile's lowest to its last's highest); a wave's own rows masked within it
  const uint32_t last = tb + kFaTile * RT - 1 < n_tok - 1 ? tb + kFaTile * RT - 1 : n_tok - 1;
  const uint32_t qlo = base + tb, qhi = base + last;
  // the first key tile at a multiple of kFaTile: a query's keys cut into the same tiles whatever block it is in
  // (the tiles before its window wholly masked: no-ops), so its bits are its own - given keys whose index is their
  // position mod kFaTile (ATTN_TILE: the engine's windows start so)
  uint32_t klo = a.window && qlo + 1 > a.window ? (qlo + 1 - a.window) / kFaTile * kFaTile : 0, khi = qhi;
  if constexpr (SPLIT) {  // this block's unit alone
    const uint32_t u = u0 + blockIdx.z, ua = u * kFaUnit;  // the unit's keys as indices (it may start before key 0)
    const uint32_t us = ua > pos0 ? ua - pos0 : 0, ue = ua + kFaUnit - 1 - pos0;
    klo = klo > us ? klo : us, khi = khi < ue ? khi : ue;
  }
  // a tile's K and V into registers (keys past the last zero): F32 a value a thread at a time; f16 (a cache's) 8
  // halves a 16-byte load - the same values either way
  constexpr bool kQ8 = sizeof(KV) == 1, kHalf = sizeof(KV) <= 2;  // (kHalf: 8 f16s a load - a Q8 record's too)
  constexpr bool kV5 = std::is_same<KV, Kv85>::value;             // (the values Q5 records)
  constexpr uint32_t kVRec = kV5 ? ATTN_KV5_RECORD : ATTN_KV8_RECORD;
  constexpr int kVec = 8, kVecs = kStage * kFaHead / kVec, kVItems = (kVecs + kNth - 1) / kNth;
  float kr[kHalf ? 1 : kItems], vr[kHalf ? 1 : kItems];
  fa_h8 kh[kHalf && !kQ8 ? kVItems : 1], vh[kHalf && !kQ8 ? kVItems : 1];
  Kv8Raw k8[kQ8 ? kVItems : 1], v8[kQ8 && !kV5 ? kVItems : 1];  // (a Q8 / Q5 tile as loaded: its values at the store)
  Kv5Raw v5[kV5 ? kVItems : 1];
  uint2 kq[kKD ? kFaHead / kVec : 1];  // (K-direct: the lane's key's record - its int8s 8 a piece, a scale a block)
  _Float16 ksc[kKD ? kFaHead / ATTN_KV8_BLOCK : 1];
  // K-direct: the lane's key's record, loaded after this tile's scores (its only reader), in flight under the rest
  auto fetch_k = [&](uint32_t kb) {
    const bool in = kb + col < n_keys;
    const uint8_t* rec = (const uint8_t*) k + ((uint64_t) (kb + col) * a.kv_heads + kvh) * ATTN_KV8_RECORD;
    for (int i = 0; i < kFaHead / kVec; i++) kq[i] = in ? *(const uint2*) (rec + i * kVec) : uint2{0, 0};
    for (int b = 0; b < kFaHead / ATTN_KV8_BLOCK; b++)
      ksc[b] = in ? *(const _Float16*) (rec + ATTN_HEAD + 2 * b) : (_Float16) 0.0f;
  };
  auto fetch = [&](uint32_t kb) {
    if constexpr (kQ8)
      for (int it = 0; it < kVItems; it++) {
        const uint32_t i = threadIdx.x + it * kNth, r = i % kStage, d = i / kStage * kVec;  // (a wave's threads its keys: V^T's stores apart)
        const bool in = i < (uint32_t) kVecs && kb + r < n_keys;
        const uint64_t at = (uint64_t) (kb + r) * a.kv_heads + kvh;
        if constexpr (!kKD) k8[it] = in ? kv8_raw((const uint8_t*) k + at * ATTN_KV8_RECORD, d) : Kv8Raw{};
        if constexpr (kV5)
          v5[it] = in ? kv5_raw((const uint8_t*) v + at * kVRec, d) : Kv5Raw{};
        else
          v8[it] = in ? kv8_raw((const uint8_t*) v + at * kVRec, d) : Kv8Raw{};
      }
    else if constexpr (kHalf)
      for (int it = 0; it < kVItems; it++) {
        const uint32_t i = threadIdx.x + it * kNth, r = i % kStage, d = i / kStage * kVec;  // (a wave's threads its keys: V^T's stores apart)
        const bool in = i < (uint32_t) kVecs && kb + r < n_keys;
        const uint64_t at = ((uint64_t) (kb + r) * a.kv_heads + kvh) * kFaHead + d;
        kh[it] = in ? *(const fa_h8*) (k + at) : fa_h8{}, vh[it] = in ? *(const fa_h8*) (v + at) : fa_h8{};
      }
    else
      for (int it = 0; it < kItems; it++) {
        const uint32_t i = threadIdx.x + it * kNth, r = i / kFaHead, d = i % kFaHead, key = kb + r;
        const bool in = i < kStage * kFaHead && key < n_keys;
        const uint64_t at = ((uint64_t) key * a.kv_heads + kvh) * kFaHead + d;
        kr[it] = in ? (float) k[at] : 0.0f, vr[it] = in ? (float) v[at] : 0.0f;
      }
  };
  if (klo <= khi) {
    fetch(klo);
    if constexpr (kKD) fetch_k(klo);
  }
  for (uint32_t kb0 = klo; kb0 <= khi; kb0 += kStage) {
    __syncthreads();  // the last stage's K, V read by every wave
    if constexpr (kHalf)
      for (int it = 0; it < kVItems; it++) {
        const uint32_t i = threadIdx.x + it * kNth, r = i % kStage, d = i / kStage * kVec;  // (a wave's threads its keys: V^T's stores apart)
        if (i >= (uint32_t) kVecs) continue;
        fa_h8 kv, vv;  // a Q8 / Q5 tile's values now (a missing key's raw zeros: 0 x 0, f16's +0)
        if constexpr (kQ8) {
          if constexpr (!kKD) kv = kv8_vals(k8[it]);
          if constexpr (kV5)
            vv = kv5_vals(v5[it]);
          else
            vv = kv8_vals(v8[it]);
        } else
          kv = kh[it], vv = vh[it];
        if constexpr (!kKD) *(fa_h8*) (ks + r * kFaRow + d) = kv;
        for (int e = 0; e < kVec; e++) vt[(d + e) * kVKey + r] = vv[e];
      }

    else
      for (int it = 0; it < kItems; it++) {
        const uint32_t i = threadIdx.x + it * kNth, r = i / kFaHead, d = i % kFaHead;
        if (i < kStage * kFaHead) ks[r * kFaRow + d] = (_Float16) kr[it], vt[d * kVKey + r] = (_Float16) vr[it];
      }
    __syncthreads();
    if (kb0 + kStage <= khi) fetch(kb0 + kStage);  // the next stage's, in flight under this one's WMMAs
    if (!active) continue;                         // (its loads done: no row of its own)
    for (int sub = 0; sub < ST; sub++) {
    const uint32_t kb = kb0 + (uint32_t) sub * kFaTile;  // this tile's first key
    if (kb > khi) break;
    fa_f8 sacc = {};
    for (int g = 0; g < kFaHead / kFaTile; g++) {
      fa_h16 kb16;
      if constexpr (kKD) {  // the lane's key's B fragment: dims g 16 .. g 16 + 15, kv8_vals' values
        constexpr int kPer = ATTN_KV8_BLOCK / kVec;  // (pieces a scale)
        const fa_h8 lo = kv8_vals({kq[2 * g], ksc[2 * g / kPer]}), hi = kv8_vals({kq[2 * g + 1], ksc[(2 * g + 1) / kPer]});
        for (int e = 0; e < kVec; e++) kb16[e] = lo[e], kb16[kVec + e] = hi[e];
      } else
        kb16 = *(const fa_h16*) (ks + (sub * kFaTile + col) * kFaRow + g * kFaTile);
      sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[g], kb16, sacc);
    }
    if constexpr (kKD)
      if (kb + kFaTile <= khi) fetch_k(kb + kFaTile);
    const uint32_t key = kb + col;
    float alpha[8];
#pragma unroll  // (the rows' registers named: not m0's indirect moves and a branch a row)
    for (int j = 0; j < 8; j++) {
      const uint32_t t = rtok(2 * j + half), qi = base + t;
      const bool valid = t < n_tok && key <= qi && (!a.window || qi - key < a.window);
      float s = -INFINITY;
      if (valid) {
        const uint32_t dist = qi - key;
        s = sacc[j] * a.scaling;
        if (dist < a.extent) {
          const float* rr = rel + ((uint64_t) t * H + rhead(2 * j + half)) * a.d_rel;
          float b = 0;
#pragma unroll  // (attn_core_create's ATTN_REL: the 16 terms' loads in flight together, the sum in their order)
          for (uint32_t r = 0; r < ATTN_REL; r++) b += rr[r] * rel_proj[(uint64_t) r * a.extent + dist];
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
    const uint32_t tlo = rtok(0), thi = rtok(kFaTile - 1) < n_tok - 1 ? rtok(kFaTile - 1) : n_tok - 1;
    const int64_t c_lo = (int64_t) (base + tlo) - kb, c_last = (int64_t) (base + thi) - kb;
    const int cmin = c_lo > 0 ? (int) c_lo : 0, cmax = c_last < kFaTile - 2 ? (int) c_last : kFaTile - 2;
#pragma unroll
    for (int nt = 0; nt < kFaHead / kFaTile; nt++) {
      for (int j = 0; j < 8; j++) o[nt][j] *= alpha[j];
      const fa_h16 bv = *(const fa_h16*) (vt + (nt * kFaTile + col) * kVKey + sub * kFaTile);
      const fa_f8 prev = o[nt];
      o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, bv, prev);
      for (int c = cmin; c <= cmax; c++) {
        fa_h16 bc = bv;
        for (int e = c + 1; e < kFaTile; e++) bc[e] = (_Float16) 0.0f;
        const fa_f8 d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, bc, prev);
        for (int j = 0; j < 8; j++)
          if ((int64_t) (base + rtok(2 * j + half)) == (int64_t) kb + c) o[nt][j] = d[j];
      }
    }
    __builtin_amdgcn_wave_barrier();
    if constexpr (!SPLIT)
      if ((pos0 + kb + kFaTile) % kFaUnit == 0 || kb + kFaTile > khi) {  // a unit's end: folded, the next from nothing
        for (int j = 0; j < 8; j++) {
          float ca, cb, fgm = gm(j), fgl = gl(j);
          fa_fold_coef(fgm, fgl, m[j], l[j], ca, cb);
          __builtin_amdgcn_wave_barrier();
          if (col == 0) gm(j) = fgm, gl(j) = fgl;
          const uint32_t t = rtok(2 * j + half);
          for (int nt = 0; nt < kFaHead / kFaTile; nt++) {
            if (t < n_tok) out[out_at(t, nt)] = out[out_at(t, nt)] * ca + o[nt][j] * cb;
            o[nt][j] = 0.0f;
          }
          m[j] = -INFINITY, l[j] = 0.0f;
        }
      }
    }  // (the stage's tiles)
  }
  for (int j = 0; j < 8 && active; j++) {
    const uint32_t t = rtok(2 * j + half);
    if (t >= n_tok) continue;
    if constexpr (SPLIT) {  // the unit's partial: m, l, then o (the fold's)
      float* pp = part + (((uint64_t) blockIdx.z * n_tok + t) * H + rhead(2 * j + half)) * (kFaHead + 2);
      if (col == 0) pp[0] = m[j], pp[1] = l[j];
      for (int nt = 0; nt < kFaHead / kFaTile; nt++) pp[2 + nt * kFaTile + col] = o[nt][j];
    } else
      for (int nt = 0; nt < kFaHead / kFaTile; nt++) {
        const uint64_t i = out_at(t, nt);
        out[i] = out[i] / gl(j) * (gate ? sigmoid(gate[i]) : 1.0f);
      }
  }
}

// a split decode's units folded in order into the sink's start (fa_kernel's fold): a block a (token, head), a
// thread a value dim
__global__ void fa_fold_kernel(const float* part, uint32_t units, uint32_t n_tok, uint32_t H, const float* sink,
                               const float* gate, float* out, _Float16* x16, int32_t* overflow) {
  const uint32_t t = blockIdx.x / H, h = blockIdx.x % H, d = threadIdx.x, cols = H * kFaHead;
  if (t >= n_tok) return mm::x16_put(x16, t, h * kFaHead + d, cols, 0.0f, overflow);  // a pad row of x16's tile
  float gm = sink[(uint64_t) t * H + h], gl = 1.0f, go = 0.0f;
  for (uint32_t u = 0; u < units; u++) {
    const float* pp = part + (((uint64_t) u * n_tok + t) * H + h) * (kFaHead + 2);
    float ca, cb;
    fa_fold_coef(gm, gl, pp[0], pp[1], ca, cb);
    go = go * ca + pp[2 + d] * cb;
  }
  const uint64_t i = ((uint64_t) t * H + h) * kFaHead + d;
  const float o = go / gl * (gate ? sigmoid(gate[i]) : 1.0f);
  out[i] = o;
  if (x16) mm::x16_put(x16, t, h * kFaHead + d, cols, o, overflow);
}

}  // namespace attn
