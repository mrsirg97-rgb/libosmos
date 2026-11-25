// the attention core's heads block (ATTN_BLOCK_HEADS, attn_core.h): a block one token and one KV head, its rows the
// token's query heads (up to 16) - the rows read the same keys, a token's picks its own. its sums the skeleton's:
// 256-key units at absolute positions, each from nothing, folded in order (fa_fold_coef); 16-key tiles ascending;
// one WMMA chain over the head's dims. a key not in the token's list (unpicked, or past it) loaded as zeros and
// masked - the matrix cores not blind to a key its weight zeroes - so a token's bits are its list's alone, and every
// block picked reads exactly the dense tiles. a head of D (256: two waves, each the tile's scores, each half of O's
// columns); rows Q8_0 (ggml's blocks: an f16 d, then 32 int8s: a value q d, its product exact, rounded to f16)
#pragma once
#include "attn_core_kernel.h"

namespace attn {

constexpr int kQ8Block = 32, kQ8Bytes = 34;  // (q4x_kv_put's rows: Q4X_KV_BLOCK, Q4X_KV_BLOCK_BYTES)

struct HeadsArgs {
  uint32_t heads, kv_heads, ratio, stride, unit;  // (unit: a unit's keys - the desc's, ATTN_UNIT unless set)
  float scaling;
  uint32_t key0, pos0;  // the cache's row 0's position; the step's first token's
  uint32_t x16_row0, x16_cols;
};

// a token's list (picks given and not all: its picked blocks' positions, then its tail - its visible blocks' end to
// itself; else every position to itself): its length, and entry i's position. tiles and units are its entries',
// so every block picked tiles exactly as dense, and a token's bits are its list's alone
struct List {
  const int32_t* sel;
  uint32_t np, nb, ratio, p;
  bool all;
  __device__ uint32_t len() const { return all ? p + 1 : np * ratio + (p + 1 - nb * ratio); }
  __device__ uint32_t at(uint32_t i) const {
    if (all) return i;
    const uint32_t picked = np * ratio;
    return i < picked ? (uint32_t) sel[i / ratio] * ratio + i % ratio : nb * ratio + (i - picked);
  }
};

// a Q8_0 row's 8 values from dim d as loaded (its int8s by halves - 2-byte aligned in a 34-byte block - and the
// block's scale), then as f16 values: q d (the product exact, then rounded)
struct Q8Raw {
  uint32_t lo, hi;  // the 8 int8s, packed
  _Float16 s;
};
__device__ inline Q8Raw q8_raw(const uint8_t* row, uint32_t d) {
  const uint8_t* blk = row + d / kQ8Block * kQ8Bytes;
  const uint16_t* qb = (const uint16_t*) (blk + 2 + d % kQ8Block);
  return {(uint32_t) qb[0] | (uint32_t) qb[1] << 16, (uint32_t) qb[2] | (uint32_t) qb[3] << 16, *(const _Float16*) blk};
}
__device__ inline fa_h8 q8_vals(const Q8Raw& r) {
  const fa_h2 s2 = {r.s, r.s};
  const uint32_t w[2] = {r.lo, r.hi};
  const int8_t* q = (const int8_t*) w;
  fa_h8 out;
  for (int e = 0; e < 8; e += 2) {
    const fa_h2 v = fa_h2{(_Float16) q[e], (_Float16) q[e + 1]} * s2;
    out[e] = v[0], out[e + 1] = v[1];
  }
  return out;
}

constexpr int kHeadsWaves = 4;  // a block's waves: each the tile's scores, each a quarter of O's columns
// D the head; G the query heads a KV head (<= 16: a tile's rows); SPLIT: a block a unit (blockIdx.z, from u0), its
// partial (m, l, o) a row to part (the fold's) - a decode's units in parallel, the prompt's in-kernel fold's bits
template <int D, bool SPLIT>
__global__ __launch_bounds__(kHeadsWaves * 32) void heads_kernel(HeadsArgs a, const float* q, const float* gate, uint32_t n_tok,
                                                   const uint8_t* kc, const uint8_t* vc, const int32_t* sel,
                                                   const int32_t* npick, float* out, _Float16* x16, int32_t* overflow,
                                                   uint32_t u0, float* part) {
  constexpr int kW = kHeadsWaves, kThreads = kW * 32;
  constexpr int kRow = D + 8, kVKey = kFaTile + 8, kFrags = D / kFaTile, kHalf = kFrags / kW;  // (kHalf: a wave's)
  __shared__ __attribute__((aligned(32))) _Float16 qs[kFaTile * kRow], ks[kFaTile * kRow], vt[D * kVKey];
  __shared__ __attribute__((aligned(32))) _Float16 ps[kW][kFaTile * kVKey];
  __shared__ float gml[kW][kFaTile][2];  // a row's running (max, sum): a wave's own copy (its lane 0 writes)
  const uint32_t t = blockIdx.x, kvh = blockIdx.y, wave = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint32_t half = lane >> 4, col = lane % kFaTile, H = a.heads, G = H / a.kv_heads;
  const uint32_t p = a.pos0 + t;  // the token's position
  const uint64_t rowb = (uint64_t) a.kv_heads * D / kQ8Block * kQ8Bytes;  // a cache row's bytes
  // Q's rows (the KV head's G query heads, zeros past them) into LDS as f16
  for (uint32_t i = threadIdx.x; i < (uint32_t) (kFaTile * D); i += kThreads) {
    const uint32_t r = i / D, d = i % D;
    qs[r * kRow + d] = (_Float16) (r < G ? q[((uint64_t) t * H + kvh * G + r) * D + d] : 0.0f);
  }
  // the token's list; the entries walked: the unit's (SPLIT), or all of them
  const bool picks = sel != nullptr;
  const uint32_t nb = a.ratio ? (p + 1) / a.ratio : 0, np = picks ? (uint32_t) npick[t] : 0;
  const List L = {picks ? sel + (uint64_t) t * a.stride : nullptr, np, nb, a.ratio, p, !picks || np == nb};
  const uint32_t len = L.len();
  uint32_t elo = 0, ehi = len;  // [elo, ehi)
  if constexpr (SPLIT) {
    elo = (u0 + blockIdx.z) * a.unit, ehi = ehi < elo + a.unit ? ehi : elo + a.unit;
  }
  // a row's running fold (no sink: from nothing; its o in its out row, Inkling's way - not 64 more registers) and its
  // unit's (m, l, o): o this wave's half of the columns
  float m[8], l[8];
  auto gm = [&](int j) -> float& { return gml[wave][2 * j + half][0]; };
  auto gl = [&](int j) -> float& { return gml[wave][2 * j + half][1]; };
  fa_f8 o[kHalf] = {};
  auto out_at = [&](uint32_t r, int nt) -> uint64_t {
    return ((uint64_t) t * H + kvh * G + r) * D + (wave * kHalf + nt) * kFaTile + col;
  };
  for (int j = 0; j < 8; j++) {
    if (col == 0) gm(j) = -INFINITY, gl(j) = 0.0f;
    m[j] = -INFINITY, l[j] = 0.0f;
    if constexpr (!SPLIT)
      if (2 * j + half < G)
        for (int nt = 0; nt < kHalf; nt++) out[out_at(2 * j + half, nt)] = 0.0f;
  }
  // a tile's K and V raw into registers (the next tile's while this one's WMMAs run): a thread kItems pieces of 8
  // values, entries past the list zeros
  constexpr int kItems = kFaTile * D / 8 / kThreads;
  Q8Raw kr[kItems], vr[kItems];
  auto fetch = [&](uint32_t e0) {
    for (int it = 0; it < kItems; it++) {
      const uint32_t i = threadIdx.x + it * kThreads, r = i / (D / 8), d = i % (D / 8) * 8, e = e0 + r;
      const bool in = e < ehi;
      const uint64_t row = in ? (uint64_t) (L.at(e) - a.key0) * rowb : 0;
      kr[it] = in ? q8_raw(kc + row, kvh * D + d) : Q8Raw{};
      vr[it] = in ? q8_raw(vc + row, kvh * D + d) : Q8Raw{};
    }
  };
  __syncthreads();  // (Q's rows in LDS)
  if (elo < ehi) fetch(elo);
  for (uint32_t e0 = elo; e0 < ehi; e0 += kFaTile) {
    __syncthreads();  // the last tile's K, V read by both waves
    for (int it = 0; it < kItems; it++) {
      const uint32_t i = threadIdx.x + it * kThreads, r = i / (D / 8), d = i % (D / 8) * 8;
      const fa_h8 kv = q8_vals(kr[it]), vv = q8_vals(vr[it]);
      *(fa_h8*) (ks + r * kRow + d) = kv;
      for (int e = 0; e < 8; e++) vt[(d + e) * kVKey + r] = vv[e];
    }
    __syncthreads();
    if (e0 + kFaTile < ehi) fetch(e0 + kFaTile);  // the next tile's, in flight under this one's
    fa_f8 sacc = {};
    for (int g = 0; g < kFrags; g++) {  // the scores: one chain over the head's dims, in order
      const fa_h16 qa = *(const fa_h16*) (qs + col * kRow + g * kFaTile);
      const fa_h16 kb16 = *(const fa_h16*) (ks + col * kRow + g * kFaTile);
      sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa, kb16, sacc);
    }
    const bool key_in = e0 + col < ehi;
    float alpha[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint32_t r = 2 * j + half;
      const float sc = r < G && key_in ? sacc[j] * a.scaling : -INFINITY;
      const float mn = fmaxf(m[j], fa_row_max(sc));
      const float pv = sc == -INFINITY ? 0.0f : expf(sc - mn);
      alpha[j] = m[j] == mn ? 1.0f : expf(m[j] - mn);
      l[j] = l[j] * alpha[j] + fa_row_sum(pv), m[j] = mn;
      ps[wave][r * kVKey + col] = (_Float16) pv;
    }
    __builtin_amdgcn_wave_barrier();
    const fa_h16 pa = *(const fa_h16*) (ps[wave] + col * kVKey);
#pragma unroll
    for (int nt = 0; nt < kHalf; nt++) {
      const uint32_t c0 = (wave * kHalf + nt) * kFaTile;
      for (int j = 0; j < 8; j++) o[nt][j] *= alpha[j];
      const fa_h16 bv = *(const fa_h16*) (vt + (c0 + col) * kVKey);
      o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, bv, o[nt]);
    }
    __builtin_amdgcn_wave_barrier();
    if constexpr (!SPLIT)
      if ((e0 + kFaTile) % a.unit == 0 || e0 + kFaTile >= ehi) {  // a unit's end: folded, the next from nothing
        for (int j = 0; j < 8; j++) {
          const uint32_t r = 2 * j + half;
          if (m[j] != -INFINITY && r < G) {
            float ca, cb, fgm = gm(j), fgl = gl(j);
            fa_fold_coef(fgm, fgl, m[j], l[j], ca, cb);
            __builtin_amdgcn_wave_barrier();
            if (col == 0) gm(j) = fgm, gl(j) = fgl;
            for (int nt = 0; nt < kHalf; nt++) out[out_at(r, nt)] = out[out_at(r, nt)] * ca + o[nt][j] * cb;
          }
          for (int nt = 0; nt < kHalf; nt++) o[nt][j] = 0.0f;
          m[j] = -INFINITY, l[j] = 0.0f;
        }
      }
  }
  for (int j = 0; j < 8; j++) {
    const uint32_t r = 2 * j + half;
    if (r >= G) continue;
    const uint32_t h = kvh * G + r;
    if constexpr (SPLIT) {  // the unit's partial: m, l, then o
      float* pp = part + (((uint64_t) blockIdx.z * n_tok + t) * H + h) * (D + 2);
      if (col == 0 && wave == 0) pp[0] = m[j], pp[1] = l[j];
      for (int nt = 0; nt < kHalf; nt++) pp[2 + (wave * kHalf + nt) * kFaTile + col] = o[nt][j];
    } else
      for (int nt = 0; nt < kHalf; nt++) {
        const uint32_t d = (wave * kHalf + nt) * kFaTile + col;
        const uint64_t i = out_at(r, nt);
        const float v = out[i] / gl(j) * (gate ? sigmoid(gate[i]) : 1.0f);
        out[i] = v;
        if (x16) mm::x16_put(x16, a.x16_row0 + t, h * D + d, a.x16_cols, v, overflow);
      }
  }
}

// a step's units merged max-first (Qwen's, q4x_attn's fa_merge_kernel: the row's units to its list's last - their
// greatest max, then each unit's sum and o weighted by exp(m - it), in order - its bits), the gate a value's: a
// block a (token, head), a thread a value dim
template <int D>
__global__ __launch_bounds__(D) void heads_merge_kernel(const float* part, uint32_t n_tok, HeadsArgs a,
                                                        const int32_t* sel, const int32_t* npick, const float* gate,
                                                        float* out, _Float16* x16, int32_t* overflow) {
  const uint32_t H = a.heads, t = blockIdx.x / H, h = blockIdx.x % H, d = threadIdx.x, p = a.pos0 + t;
  const bool picks = sel != nullptr;
  const uint32_t nb = a.ratio ? (p + 1) / a.ratio : 0, np = picks ? (uint32_t) npick[t] : 0;
  const List L = {nullptr, np, nb, a.ratio, p, !picks || np == nb};
  const uint32_t last = (L.len() - 1) / a.unit;
  auto at = [&](uint32_t u) { return part + (((uint64_t) u * n_tok + t) * H + h) * (D + 2); };
  float mx = -INFINITY;
  for (uint32_t u = 0; u <= last; u++) mx = fmaxf(mx, at(u)[0]);
  float sum = 0.0f, acc = 0.0f;
  for (uint32_t u = 0; u <= last; u++) {
    const float w = expf(at(u)[0] - mx);
    sum += at(u)[1] * w, acc += at(u)[2 + d] * w;
  }
  const uint64_t i = ((uint64_t) t * H + h) * D + d;
  out[i] = acc / sum * (gate ? sigmoid(gate[i]) : 1.0f);
  if (x16) mm::x16_put(x16, a.x16_row0 + t, h * D + d, a.x16_cols, out[i], overflow);
}

}  // namespace attn
