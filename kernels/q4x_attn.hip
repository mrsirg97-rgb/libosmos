// Qwen3.8-Flash-Next's attention (include/osmos/gpu/q4x_ops.h): q's and k's norm and rotation, and the gated
// attention over the cache - each token's keys its key list: every position, or QSA's picks (q4x_qsa.hip)
#include "block.h"
#include "port.h"
#include "launch.h"
#include "rope.h"
#include "x16side.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

namespace {

using launches::st;
using launches::ok;

constexpr int kHead = Q4X_HEAD_DIM;  // a head's dim: a block's threads, a value a thread


// a block a (head, token): blockIdx.x < heads a q head, past them a KV head - a head H wide, a thread a dim. q's row
// with its gate beside each head (Qwen's: [q h, gate h], the gate copied out) or plain (gate NULL: LFM3's, its gate
// its own matrix). the norm, then the first rope.dim rotated in pairs (i, i + rope.dim / 2) for i < rope.pairs -
// theta_i a running product as ggml's rope cache builds it - the pairs past them (LFM3's zero tail) unrotated, times
// the rotation's magnitude (cos mscale, sin 0: Liquid's)
template <int H>
__global__ __launch_bounds__(H) void qk_kernel(const float* qfull, const float* k, uint32_t heads, uint32_t kv_heads,
                                               Q4xRope rope, const float* q_norm, const float* k_norm, float eps,
                                               int32_t pos0, float* q, float* gate, float* k_out) {
  __shared__ float lds[H], y[H];
  const uint32_t h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
  const bool is_q = h < heads;
  const uint32_t g = is_q ? h : h - heads;
  float x;
  if (is_q && gate) {
    const float* row = qfull + (uint64_t) t * 2 * heads * H + 2ull * h * H;
    x = row[d];
    gate[((uint64_t) t * heads + h) * H + d] = row[H + d];
  } else if (is_q) {
    x = qfull[((uint64_t) t * heads + h) * H + d];
  } else {
    x = k[((uint64_t) t * kv_heads + g) * H + d];
  }
  const float r = 1.0f / sqrtf(block_sum<H>(x * x, lds) / (float) H + eps);
  y[d] = x * r * (is_q ? q_norm : k_norm)[d];
  __syncthreads();
  float v = y[d];
  const uint32_t half = rope.dim / 2;
  if (d < rope.dim) {
    const uint32_t i = d % half;
    if (i < rope.pairs) {
      float c, s;
      rope_cs(rope, (float) (pos0 + (int32_t) t), i, &c, &s);
      const float x0 = y[i], x1 = y[i + half];
      v = d < half ? x0 * c - x1 * s : x0 * s + x1 * c;
    } else {
      v = y[d] * rope.mscale;
    }
  }
  if (is_q)
    q[((uint64_t) t * heads + h) * H + d] = v;
  else
    k_out[((uint64_t) t * kv_heads + g) * H + d] = v;
}

// the attention on the matrix cores (WMMA f32 <- f16, 16 x 16 x 16; mm.h's lane layout: an operand's lane l its row
// or column l % 16, a result's lane l rows 2 j + l / 16 of column l % 16). a block: one token's q heads of a KV head
// (G <= 16 rows) over one chunk of kChunk keys, a block of 16 keys at a time: S = Q K^T (f16 q and k), each row's
// running max and sum (fixed shuffle trees in its lane half), P = exp(S - max) to f16, O += P V (f16 v); its rows'
// (max, sum, O) into the chunk's scratch. a tile one token's: its keys past the token's position loaded as zeros in
// every batch - the WMMA's sum is not blind to a value its weight zeroes (a masked key's v moves the last bit), so a
// row's keys must be the same whatever it is batched with. then a row's bits are its own: its q, its keys,
// its position; and whichever chunks its batch computed past its own, it merges only its own
constexpr int kTile = Q4X_GQA_MAX; /* a key head's query heads at most (and the tile's rows) */
constexpr int kWaves = 4;                   // a wave a quarter of O's columns
constexpr int kChunk = 128;                 // a chunk's keys: absolute positions [c kChunk, (c + 1) kChunk)
constexpr int kStepTokens = 64;             // a piece's most tokens
constexpr uint32_t kPieceChunks = 8192;     // the scratch's (token, chunk) pairs: a piece's tokens x its chunks
template <int HD> constexpr int kRowLds = HD + 8;  // f16s a row in LDS, past it a pad (rows 4 banks apart)
constexpr int kKeyLds = kTile + 8;
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));

// a token's key list (q4x_ops.h's Q4xPick; none: every position to its own): its length, and entry i's position -
// its picked blocks' positions in block order, then its tail. every block picked, entry i is position i
// a window's (AttnShape's): dense keys from the window's first position (lo), each position's cache row pos % slots
struct Keys {
  const int32_t* sel;  // the token's picked blocks, or NULL: entry i position lo + i
  uint32_t ratio, np, nv, len, lo, slots;
  __device__ uint32_t at(uint32_t i) const {
    if (!sel) return lo + i;
    return i < ratio * np ? ratio * (uint32_t) sel[i / ratio] + i % ratio : ratio * nv + (i - ratio * np);
  }
  __device__ uint64_t row_of(uint32_t i) const { return slots ? at(i) % slots : at(i); }
};

__device__ inline Keys keys_of(const Q4xPick& pk, uint32_t t, uint32_t top, uint32_t window = 0, uint32_t slots = 0) {
  if (!pk.npick) {
    const uint32_t lo = window && top + 1 > window ? top + 1 - window : 0;
    return Keys{nullptr, 1, 0, 0, top + 1 - lo, lo, slots};
  }
  const uint32_t nv = (top + 1) / pk.ratio, np = (uint32_t) pk.npick[t];
  return Keys{np == nv ? nullptr : pk.sel + (uint64_t) t * pk.blocks, pk.ratio, np, nv,
              pk.ratio * np + (top + 1) % pk.ratio, 0, 0};
}

constexpr int kLoad = 8;  // an item's values: a load within one Q8_0 block
static_assert(Q4X_KV_BLOCK % kLoad == 0, "a load within one block");

// a thread's share of a block of kTile keys' K and V, raw: each item kLoad int8s and their block's scale - loaded a
// block ahead (the next block's while this one's WMMAs run); a value its scale times its int8, the product exact in
// F32, then rounded to f16
template <int HD> constexpr int kItems = kTile * HD / kLoad / (kWaves * 32);
static_assert(kItems<Q4X_HEAD_DIM> * kWaves * 32 * kLoad == kTile * Q4X_HEAD_DIM &&
                  kItems<Q4X_QK_HEAD_NARROW> * kWaves * 32 * kLoad == kTile * Q4X_QK_HEAD_NARROW,
              "a block's values the threads' whole items");
template <int HD>
struct Raw {
  uint64_t kq[kItems<HD>], vq[kItems<HD>];
  _Float16 ksc[kItems<HD>], vsc[kItems<HD>];
};

// an item's key and first dim: K's lanes along a key's dims (its LDS rows written whole), V's along the keys (V^T's
// columns: a wave's stores across the banks - along the dims they would be 16-way on one bank pair)
template <int HD>
__device__ inline void item_k(uint32_t it, uint32_t* r, uint32_t* d) {
  const uint32_t i = threadIdx.x + it * kWaves * 32;
  *r = i / (HD / kLoad), *d = i % (HD / kLoad) * kLoad;
}
__device__ inline void item_v(uint32_t it, uint32_t* r, uint32_t* d) {
  const uint32_t i = threadIdx.x + it * kWaves * 32;
  *r = i % kTile, *d = i / kTile * kLoad;
}

// an item's kLoad int8s and their block's scale: zeros past k1
template <int HD>
__device__ inline void raw_one(const uint8_t* cache, uint64_t row, uint32_t kvh, const Keys& keys, uint32_t e,
                               uint32_t k1, uint32_t d, uint64_t* q, _Float16* sc) {
  if (e < k1) {
    const uint32_t c = kvh * HD + d;
    const uint8_t* b8 = cache + keys.row_of(e) * row + (c / Q4X_KV_BLOCK) * Q4X_KV_BLOCK_BYTES;
    *sc = *(const _Float16*) b8;
    __builtin_memcpy(q, b8 + 2 + c % Q4X_KV_BLOCK, kLoad);
  } else {
    *sc = (_Float16) 0.0f, *q = 0;
  }
}

template <int HD>
__device__ inline void raw_fetch(Raw<HD>& w, const uint8_t* kc, const uint8_t* vc, uint64_t row, uint32_t kvh,
                                 const Keys& keys, uint32_t kb, uint32_t k1) {
  for (int it = 0; it < kItems<HD>; it++) {
    uint32_t r, d;
    item_k<HD>(it, &r, &d);
    raw_one<HD>(kc, row, kvh, keys, kb + r, k1, d, &w.kq[it], &w.ksc[it]);
    item_v(it, &r, &d);
    raw_one<HD>(vc, row, kvh, keys, kb + r, k1, d, &w.vq[it], &w.vsc[it]);
  }
}

template <int HD>
__device__ inline void raw_store(const Raw<HD>& w, _Float16* ks, _Float16* vt) {
  for (int it = 0; it < kItems<HD>; it++) {
    uint32_t r, d;
    item_k<HD>(it, &r, &d);
    const float sk = (float) w.ksc[it], sv = (float) w.vsc[it];
    for (int j = 0; j < kLoad; j++) ks[r * kRowLds<HD> + d + j] = (_Float16) (sk * (float) (int8_t) (w.kq[it] >> (8 * j)));
    item_v(it, &r, &d);
    for (int j = 0; j < kLoad; j++) vt[(d + j) * kKeyLds + r] = (_Float16) (sv * (float) (int8_t) (w.vq[it] >> (8 * j)));
  }
}

// a thread a block of a row: its scale, then its 32 values rounded to nearest (ggml's quantize_row_q8_0)
__global__ void kv_put_kernel(const float* x, uint32_t n_blocks, uint8_t* dst) {
  const uint32_t i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n_blocks) return;
  uint8_t* b = dst + (uint64_t) i * Q4X_KV_BLOCK_BYTES;
  *(_Float16*) b = (_Float16) q8_0_block<Q4X_KV_BLOCK>(x + (uint64_t) i * Q4X_KV_BLOCK, 1, b + 2);
}

// whole-chunk groups (fa_whole_kernel): a group of tt consecutive tokens shares the chunks whole for its first - those
// fa_whole_kernel's, the rest a token's own here (tt 0: every chunk here)
__device__ inline uint32_t whole_chunks(uint32_t pos0, uint32_t t, uint32_t tt) {
  return tt ? (pos0 + t / tt * tt + 1) / kChunk : 0;
}

template <int HD>
__global__ __launch_bounds__(kWaves * 32) void fa_kernel(const float* q, const uint8_t* kc, const uint8_t* vc,
                                                         uint32_t n_tok, uint32_t heads, uint32_t kv_heads,
                                                         uint32_t pos0, Q4xPick pk, uint32_t tt, uint32_t window,
                                                         uint32_t slots, float scale, float* ml, float* po) {
  constexpr int kRowLds = ::kRowLds<HD>, kHead = HD;
  __shared__ __attribute__((aligned(32))) _Float16 qs[kTile * kRowLds], ks[kTile * kRowLds], vt[kHead * kKeyLds];
  __shared__ __attribute__((aligned(32))) _Float16 ps[kWaves][kTile * kKeyLds];
  const uint32_t G = heads / kv_heads, t = blockIdx.x, kvh = blockIdx.y, chunk = blockIdx.z;
  const uint32_t wave = threadIdx.x / 32;
  const uint64_t row = (uint64_t) kv_heads * kHead / Q4X_KV_BLOCK * Q4X_KV_BLOCK_BYTES;  // a position's bytes
  const Keys keys = keys_of(pk, t, pos0 + t, window, slots);  // the token's key list: its entries in chunks
  const uint32_t k0 = chunk * kChunk;
  if (k0 >= keys.len) return;  // a chunk past the token's list: it reads no scratch of it
  if (chunk < whole_chunks(pos0, t, tt)) return;  // its group's, fa_whole_kernel's
  const uint32_t k1 = k0 + kChunk < keys.len ? k0 + kChunk : keys.len;
  for (uint32_t i = threadIdx.x; i < kTile * kHead; i += kWaves * 32) {  // Q, f16 (rows past G zero)
    const uint32_t r = i / kHead, d = i % kHead;
    qs[r * kRowLds + d] = (_Float16) (r < G ? q[((uint64_t) t * heads + kvh * G + r) * kHead + d] : 0.0f);
  }
  // this lane's rows, a slot each (port.h: its D elements' rows): the token's last entry, or none past G; their
  // running max and sum
  int32_t lim[osmos_slots];
  float m[osmos_slots], l[osmos_slots];
  for (int r = 0; r < osmos_slots; r++)
    lim[r] = (uint32_t) osmos_slot_row(r) < G ? (int32_t) keys.len - 1 : -1, m[r] = -INFINITY, l[r] = 0.0f;
  f8 o[kHead / kTile / kWaves] = {};
  Raw<HD> raw;
  raw_fetch<HD>(raw, kc, vc, row, kvh, keys, k0, k1);
  for (uint32_t kb = k0; kb < k1; kb += kTile) {
    __syncthreads();  // the last block's K, V read by every wave
    raw_store<HD>(raw, ks, vt);  // K and V^T, f16 (keys past k1 zero)
    __syncthreads();
    if (kb + kTile < k1) raw_fetch<HD>(raw, kc, vc, row, kvh, keys, kb + kTile, k1);  // the next block's, in flight
    f8 sacc = {};
    for (int g = 0; g < kHead / kTile; g++)
      sacc = osmos_mma(osmos_load_a(qs + g * kTile, kRowLds), osmos_load_b_s(ks + g * kTile, kRowLds), sacc);
    // each row: its scores scaled and masked, the running max and sum, P to f16 at (its row, its key)
    float sv[8], alpha[osmos_slots];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int32_t key = (int32_t) (kb + osmos_acc_col(j));
      sv[j] = key <= lim[osmos_slot(j)] ? sacc[j] * scale : -INFINITY;  // 1 / sqrt(HD): Qwen's 1 / 16 exactly
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
          ps[wave][osmos_acc_row(j) * kKeyLds + osmos_acc_col(j)] = (_Float16) pv;
        }
      alpha[r] = mn == -INFINITY ? 1.0f : expf(m[r] - mn);
      l[r] = l[r] * alpha[r] + osmos_row_sum(sum), m[r] = mn;
    }
    osmos_wave_barrier();
    const osmos_fa pa = osmos_load_a(ps[wave], kKeyLds);
#pragma unroll
    for (int nt = 0; nt < kHead / kTile / kWaves; nt++) {
#pragma unroll
      for (int j = 0; j < 8; j++) o[nt][j] *= alpha[osmos_slot(j)];
      const uint32_t col0 = (wave * (kHead / kTile / kWaves) + nt) * kTile;
      o[nt] = osmos_mma(pa, osmos_load_b_s(vt + col0 * kKeyLds, kKeyLds), o[nt]);
    }
    osmos_wave_barrier();
  }
  // the chunk's partials: (max, sum) a row, O a row's dims - row R = t G + r of KV head kvh, at [chunk][kvh][R]
  const uint64_t base = ((uint64_t) chunk * kv_heads + kvh) * n_tok * G;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const uint32_t r = osmos_acc_row(j), R = t * G + r;
    if (r >= G) continue;
    if (wave == 0 && osmos_acc_col(j) == 0)  // (a row's one element at column 0: its state written once)
      ml[(base + R) * 2] = m[osmos_slot(j)], ml[(base + R) * 2 + 1] = l[osmos_slot(j)];
#pragma unroll
    for (int nt = 0; nt < kHead / kTile / kWaves; nt++)
      po[(base + R) * kHead + (wave * (kHead / kTile / kWaves) + nt) * kTile + osmos_acc_col(j)] = o[nt][j];
  }
}

// a chunk whole for every token of a group (dense keys: no picks) - tt consecutive tokens' rows (t G + r: up to
// kWholeRows), the chunk's K and V read once for the group, not a token at a time. a wave a 16-row tile and half of
// O's columns (its pair the other half: both the tile's scores, as fa_kernel's waves - 128 accumulators a lane spilled).
// each row's sums fa_kernel's: its keys all the chunk's (none masked, none zeroed), the same WMMAs in the same order,
// the same softmax (a WMMA's row its own: the tile's other rows another token's, or zeros, change none of its bits),
// its partials at the same (chunk, kvh, row). a block kWholeChunks
// chunks (its q read once), each chunk's partials its own
constexpr int kWholeTiles = 4, kWholeWaves = 2 * kWholeTiles, kWholeRows = kWholeTiles * kTile;
#ifndef WHOLE_CHUNKS
#define WHOLE_CHUNKS 4
#endif
constexpr uint32_t kWholeChunks = WHOLE_CHUNKS;
template <int HD>
__global__ __launch_bounds__(kWholeWaves * 32) void fa_whole_kernel(const float* q, const uint8_t* kc,
                                                                    const uint8_t* vc, uint32_t n_tok, uint32_t heads,
                                                                    uint32_t kv_heads, uint32_t pos0, uint32_t tt,
                                                                    float scale, float* ml, float* po) {
  constexpr int kRowLds = ::kRowLds<HD>, kHead = HD, kHalfCols = HD / kTile / 2;  // a wave's O tiles
  __shared__ __attribute__((aligned(32))) _Float16 qs[kWholeRows * kRowLds], ks[kTile * kRowLds], vt[kHead * kKeyLds];
  __shared__ __attribute__((aligned(32))) _Float16 ps[kWholeWaves][kTile * kKeyLds];
  const uint32_t G = heads / kv_heads, t0 = blockIdx.x * tt, kvh = blockIdx.y, c0 = blockIdx.z * kWholeChunks;
  const uint32_t wave = threadIdx.x / 32, rt = wave / 2, half = wave % 2;
  const uint32_t nt = n_tok - t0 < tt ? n_tok - t0 : tt, whole = whole_chunks(pos0, t0, tt);
  if (c0 >= whole) return;  // not whole for the group: its tokens' own (fa_kernel)
  const uint32_t c1 = c0 + kWholeChunks < whole ? c0 + kWholeChunks : whole;
  const uint64_t row = (uint64_t) kv_heads * kHead / Q4X_KV_BLOCK * Q4X_KV_BLOCK_BYTES;
  const Keys keys{nullptr, 1, 0, 0, pos0 + t0 + 1, 0, 0};
  for (uint32_t i = threadIdx.x; i < kWholeRows * kHead; i += kWholeWaves * 32) {
    const uint32_t r = i / kHead, d = i % kHead, tl = r / G;
    qs[r * kRowLds + d] = (_Float16) (tl < nt ? q[((uint64_t) (t0 + tl) * heads + kvh * G + r % G) * kHead + d] : 0.0f);
  }
  const bool fetcher = threadIdx.x < kWaves * 32;  // raw's items fa_kernel's threads' (the first four waves)
  const _Float16* qw = qs + rt * kTile * kRowLds;
  Raw<HD> raw;
  if (fetcher) raw_fetch<HD>(raw, kc, vc, row, kvh, keys, c0 * kChunk, c0 * kChunk + kChunk);
  for (uint32_t chunk = c0; chunk < c1; chunk++) {
    const uint32_t k0 = chunk * kChunk, k1 = k0 + kChunk;
    float m[osmos_slots], l[osmos_slots];
    for (int r = 0; r < osmos_slots; r++) m[r] = -INFINITY, l[r] = 0.0f;
    f8 o[kHalfCols] = {};
    for (uint32_t kb = k0; kb < k1; kb += kTile) {
      __syncthreads();
      if (fetcher) {
        raw_store<HD>(raw, ks, vt);
      }
      __syncthreads();
      if (fetcher) {
        if (kb + kTile < k1) raw_fetch<HD>(raw, kc, vc, row, kvh, keys, kb + kTile, k1);
        else if (chunk + 1 < c1) raw_fetch<HD>(raw, kc, vc, row, kvh, keys, k1, k1 + kChunk);  // the next chunk's first
      }
      f8 sacc = {};
      for (int g = 0; g < kHead / kTile; g++)
        sacc = osmos_mma(osmos_load_a(qw + g * kTile, kRowLds), osmos_load_b_s(ks + g * kTile, kRowLds), sacc);
      float sv[8], alpha[osmos_slots];
#pragma unroll
      for (int j = 0; j < 8; j++) sv[j] = sacc[j] * scale;  // every key the rows' (fa_kernel's mask all true)
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
            ps[wave][osmos_acc_row(j) * kKeyLds + osmos_acc_col(j)] = (_Float16) pv;
          }
        alpha[r] = mn == -INFINITY ? 1.0f : expf(m[r] - mn);
        l[r] = l[r] * alpha[r] + osmos_row_sum(sum), m[r] = mn;
      }
      osmos_wave_barrier();
      const osmos_fa pa = osmos_load_a(ps[wave], kKeyLds);
#pragma unroll
      for (int c = 0; c < kHalfCols; c++) {
#pragma unroll
        for (int j = 0; j < 8; j++) o[c][j] *= alpha[osmos_slot(j)];
        o[c] = osmos_mma(pa, osmos_load_b_s(vt + (half * kHalfCols + c) * kTile * kKeyLds, kKeyLds), o[c]);
      }
      osmos_wave_barrier();
    }
    const uint64_t base = ((uint64_t) chunk * kv_heads + kvh) * n_tok * G;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const uint32_t r = rt * kTile + osmos_acc_row(j), R = t0 * G + r;
      if (r >= nt * G) continue;
      if (half == 0 && osmos_acc_col(j) == 0)
        ml[(base + R) * 2] = m[osmos_slot(j)], ml[(base + R) * 2 + 1] = l[osmos_slot(j)];
#pragma unroll
      for (int c = 0; c < kHalfCols; c++)
        po[(base + R) * kHead + (half * kHalfCols + c) * kTile + osmos_acc_col(j)] = o[c][j];
    }
  }
}

// a block a (token, q head), a thread a dim: its row's chunks merged in order, the gate - out[t][h][d]. the chunks a
// row reads: those up to its own position, whatever its batch computed
// a gate a value (span 1) or a head (span HD)
template <int HD>
__global__ __launch_bounds__(HD) void fa_merge_kernel(const float* ml, const float* po, const float* gate,
                                                      uint32_t n_tok, uint32_t heads, uint32_t kv_heads,
                                                      uint32_t pos0, Q4xPick pk, uint32_t t0, uint32_t window,
                                                      uint32_t gate_span, float* out, Q4xX16 side) {
  constexpr int kHead = HD;
  const uint32_t t = blockIdx.x / heads, h = blockIdx.x % heads, d = threadIdx.x, G = heads / kv_heads;
  const uint32_t rows = n_tok * G, R = t * G + h % G, kvh = h / G;
  const uint32_t last = (keys_of(pk, t, pos0 + t, window).len - 1) / kChunk;
  float mx = -INFINITY;
  for (uint32_t c = 0; c <= last; c++) mx = fmaxf(mx, ml[(((uint64_t) c * kv_heads + kvh) * rows + R) * 2]);
  float sum = 0.0f, acc = 0.0f;
  for (uint32_t c = 0; c <= last; c++) {
    const uint64_t at = ((uint64_t) c * kv_heads + kvh) * rows + R;
    const float w = expf(ml[at * 2] - mx);
    sum += ml[at * 2 + 1] * w, acc += po[at * kHead + d] * w;
  }
  const uint64_t i = ((uint64_t) t * heads + h) * kHead + d;
  out[i] = acc / sum * sigmoid(gate[i / gate_span]);
  x16_side(side, i + (uint64_t) t0 * heads * kHead, out[i]);  // the step's row: the piece's from its first, t0
}

}  // namespace

extern "C" bool q4x_rope(uint32_t dim, uint32_t pairs, float base, uint32_t native_ctx, uint32_t ctx, Q4xTheta theta,
                         Q4xRope* rope) {
  if (!rope || !dim || dim % 2 || !pairs || pairs > dim / 2 || !native_ctx || !ctx ||
      (theta != Q4X_THETA_GGML && theta != Q4X_THETA_HF) ||
      (uint64_t) ctx > 64ull * native_ctx || !(base > 1.0f))
    return false;
  if (dim / 2 > Q4X_ROPE_PAIRS_MAX) return false;
  *rope = Q4xRope{dim, pairs, powf(base, -2.0f / (float) dim), 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, theta, {}};
  for (uint32_t i = 0; theta == Q4X_THETA_HF && i < dim / 2; i++)
    rope->inv_freq[i] = 1.0f / powf(base, (float) (2 * i) / (float) dim);
  if (ctx <= native_ctx) return true;  // the plain rotation
  // the model card's static YaRN, llama.cpp's for it: ext 1, beta_fast 32 / beta_slow 1 (ggml_rope_yarn_corr_dims)
  const float factor = (float) ctx / (float) native_ctx;
  const auto corr = [&](float beta) {
    return (float) dim * logf((float) native_ctx / (beta * 2.0f * (float) M_PI)) / (2.0f * logf(base));
  };
  rope->freq_scale = 1.0f / factor, rope->ext_factor = 1.0f;
  rope->corr_lo = fmaxf(0.0f, floorf(corr(32.0f))), rope->corr_hi = fminf((float) (dim - 1), ceilf(corr(1.0f)));
  rope->mscale = 1.0f + 0.1f * logf(1.0f / rope->freq_scale);
  return true;
}

extern "C" bool q4x_qk(const float* qfull, const float* k, uint32_t n_tok, uint32_t heads, uint32_t kv_heads,
                       uint32_t dim, const Q4xRope* rope, const float* q_norm, const float* k_norm, float eps,
                       int32_t pos0, float* q, float* gate, float* k_out, void* stream) {
  if ((heads && (!qfull || !q)) || !k || !rope || !q_norm || !k_norm || !k_out || !n_tok || !kv_heads ||
      (dim != kHead && dim != Q4X_QK_HEAD_NARROW) || !rope->dim || rope->dim > dim || rope->dim % 2 ||
      !rope->pairs || rope->pairs > rope->dim / 2 || pos0 < 0)
    return false;
  const dim3 grid(heads + kv_heads, n_tok);
  if (dim == kHead)
    qk_kernel<kHead><<<grid, kHead, 0, st(stream)>>>(qfull, k, heads, kv_heads, *rope, q_norm, k_norm, eps, pos0, q,
                                                     gate, k_out);
  else
    qk_kernel<Q4X_QK_HEAD_NARROW><<<grid, Q4X_QK_HEAD_NARROW, 0, st(stream)>>>(qfull, k, heads, kv_heads, *rope, q_norm,
                                                                               k_norm, eps, pos0, q, gate, k_out);
  return ok();
}

extern "C" uint64_t q4x_kv_row_bytes(uint32_t values) {
  return (uint64_t) values / Q4X_KV_BLOCK * Q4X_KV_BLOCK_BYTES;
}

// LFM3's dense prompt (a head of 128, no window, no picks): a block a group of tt consecutive tokens (kWholeRows
// rows), one KV head, one span of kSpan keys - absolute positions [s kSpan, (s + 1) kSpan) - its tiles in order
// with one running (max, sum, O) a row across the span, one partial a (span, row): the merge's reads a span's, not a
// chunk's (a 2M context's 1,024, not 16,384). a row's keys past its position masked (-inf: weight 0); the tile's keys
// past the group's last position loaded as zeros. a row's bits its own: its tiles' values its position's (the
// masked keys' weights 0 exactly), whichever group it is in
#ifndef SPAN_WAVES
#define SPAN_WAVES 8
#endif
constexpr uint32_t kSpanChunks = 16, kSpan = kSpanChunks * kChunk;
constexpr uint32_t kSpanStepTokens = 256;  // a piece's most tokens
constexpr int kSpanWaves = SPAN_WAVES, kSpanRows = kSpanWaves * kTile;  // a wave a row tile, all of O's columns

// a row's update by one tile (fa_span_kernel's): its scores masked past lim, the running max and sum, O += P V -
// only the rows upd selects (the rest's m, l, O untouched: the WMMA's other rows' results dropped)
__device__ inline int32_t span_lim(int r, uint32_t rt, uint32_t G, uint32_t nt, uint32_t first) {
  const uint32_t tl = (rt * kTile + (uint32_t) osmos_slot_row(r)) / G;  // a slot's row's token, its position (-1 none)
  return tl < nt ? (int32_t) (first + tl) : -1;
}

constexpr int kSpanKeys = 2 * kTile;          // a step of the span's: 32 keys (two WMMA tiles)
constexpr int kSpanKeyLds = kSpanKeys + 8;    // V^T's row (a dim's keys) in LDS, past it a pad
constexpr int kSpanFetch = kWaves * 32;       // a half's fetchers (raw's items: 16 keys a kWaves-wave half)

// a thread's half of a step's 32 keys (threads 0 .. 127 the first 16, the rest the next): raw's items (Raw<HD>'s
// layout, its thread index within the half)
template <int HD>
__device__ inline void span_fetch(Raw<HD>& w, const uint8_t* kc, const uint8_t* vc, uint64_t row, uint32_t kvh,
                                  const Keys& keys, uint32_t kb, uint32_t k1) {
  const uint32_t tid = threadIdx.x % kSpanFetch, base = kb + threadIdx.x / kSpanFetch * kTile;
  for (int it = 0; it < kItems<HD>; it++) {
    const uint32_t i = tid + it * kSpanFetch;
    raw_one<HD>(kc, row, kvh, keys, base + i / (HD / kLoad), k1, i % (HD / kLoad) * kLoad, &w.kq[it], &w.ksc[it]);
    raw_one<HD>(vc, row, kvh, keys, base + i % kTile, k1, i / kTile * kLoad, &w.vq[it], &w.vsc[it]);
  }
}

template <int HD>
__device__ inline void span_store(const Raw<HD>& w, _Float16* ks, _Float16* vt) {
  const uint32_t tid = threadIdx.x % kSpanFetch, h = threadIdx.x / kSpanFetch * kTile;
  for (int it = 0; it < kItems<HD>; it++) {
    const uint32_t i = tid + it * kSpanFetch;
    const uint32_t kr = h + i / (HD / kLoad), kd = i % (HD / kLoad) * kLoad, vr = h + i % kTile, vd = i / kTile * kLoad;
    const float sk = (float) w.ksc[it], sv = (float) w.vsc[it];
    for (int j = 0; j < kLoad; j++) ks[kr * kRowLds<HD> + kd + j] = (_Float16) (sk * (float) (int8_t) (w.kq[it] >> (8 * j)));
    for (int j = 0; j < kLoad; j++) vt[(vd + j) * kSpanKeyLds + vr] = (_Float16) (sv * (float) (int8_t) (w.vq[it] >> (8 * j)));
  }
}

// a row's update by one step's 32 keys (fa_span_kernel's): its scores (s0 the first 16 keys', s1 the next) masked
// past its position, the running max and sum, O += P V - only the rows upd selects (the rest's m, l, O untouched:
// the WMMA's other rows' results dropped)
template <int HD>
__device__ inline void span_tile(const f8& s0, const f8& s1, uint32_t rt, uint32_t G, uint32_t nt, uint32_t first,
                                 const bool* upd, uint32_t kb, float scale, const _Float16* vt, _Float16* pw, float* m,
                                 float* l, f8* o) {
  constexpr int kCols = HD / kTile;
  float sv[16], alpha[osmos_slots];
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const int32_t lim = span_lim(osmos_slot(j), rt, G, nt, first), key = (int32_t) (kb + osmos_acc_col(j));
    sv[j] = key <= lim ? s0[j] * scale : -INFINITY;
    sv[8 + j] = key + kTile <= lim ? s1[j] * scale : -INFINITY;
  }
#pragma unroll
  for (int r = 0; r < osmos_slots; r++) {
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < 8; j++)
      if (osmos_slot(j) == r) mx = fmaxf(mx, fmaxf(sv[j], sv[8 + j]));
    const float mn = fmaxf(m[r], osmos_row_max(mx));
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; j++)
      if (osmos_slot(j) == r) {
        const float p0 = mn == -INFINITY ? 0.0f : expf(sv[j] - mn), p1 = mn == -INFINITY ? 0.0f : expf(sv[8 + j] - mn);
        sum += p0, sum += p1;
        pw[osmos_acc_row(j) * kSpanKeyLds + osmos_acc_col(j)] = (_Float16) p0;
        pw[osmos_acc_row(j) * kSpanKeyLds + kTile + osmos_acc_col(j)] = (_Float16) p1;
      }
    alpha[r] = mn == -INFINITY ? 1.0f : expf(m[r] - mn);
    const float ln = l[r] * alpha[r] + osmos_row_sum(sum);
    if (upd[r]) m[r] = mn, l[r] = ln;
  }
  osmos_wave_barrier();
  const osmos_fa pa0 = osmos_load_a(pw, kSpanKeyLds), pa1 = osmos_load_a(pw + kTile, kSpanKeyLds);
#pragma unroll
  for (int c = 0; c < kCols; c++) {
    const f8 prev = o[c];
    f8 x = prev;
#pragma unroll
    for (int j = 0; j < 8; j++) x[j] *= alpha[osmos_slot(j)];
    const _Float16* vb = vt + c * kTile * kSpanKeyLds;
    x = osmos_mma(pa0, osmos_load_b_s(vb, kSpanKeyLds), x);
    x = osmos_mma(pa1, osmos_load_b_s(vb + kTile, kSpanKeyLds), x);
#pragma unroll
    for (int j = 0; j < 8; j++) o[c][j] = upd[osmos_slot(j)] ? x[j] : prev[j];
  }
  osmos_wave_barrier();
}

template <int HD>
__global__ __launch_bounds__(kSpanWaves * 32) void fa_span_kernel(const float* q, const uint8_t* kc,
                                                                   const uint8_t* vc, uint32_t n_tok, uint32_t heads,
                                                                   uint32_t kv_heads, uint32_t pos0, uint32_t tt,
                                                                   float scale, float* ml, float* po) {
  constexpr int kRowLds = ::kRowLds<HD>, kHead = HD, kCols = HD / kTile;  // a wave a row tile, all of O's columns
  static_assert(kSpanWaves * 32 == 2 * kSpanFetch, "a step's two halves the block's threads");
  __shared__ __attribute__((aligned(32))) _Float16 ks[kSpanKeys * kRowLds], vt[kHead * kSpanKeyLds];
  __shared__ __attribute__((aligned(32))) _Float16 ps[kSpanWaves][kTile * kSpanKeyLds];
  const uint32_t G = heads / kv_heads, t0 = blockIdx.x * tt, kvh = blockIdx.y, sp = blockIdx.z;
  const uint32_t wave = threadIdx.x / 32, rt = wave;
  const uint32_t nt = n_tok - t0 < tt ? n_tok - t0 : tt, first = pos0 + t0, top = first + nt - 1;
  const uint32_t k0 = sp * kSpan;
  if (k0 > top) return;  // a span past the group's every token
  const uint32_t k1 = k0 + kSpan < top + 1 ? k0 + kSpan : top + 1;
  const uint64_t row = (uint64_t) kv_heads * kHead / Q4X_KV_BLOCK * Q4X_KV_BLOCK_BYTES;
  const Keys keys{nullptr, 1, 0, 0, top + 1, 0, 0};
  // the wave's Q tile in registers, f16: an A operand a lane's row's 16 dims (port.h), a k-step's (rows past nt G zero)
  osmos_fa qa[kHead / kTile];
  {
    const uint32_t r = rt * kTile + threadIdx.x % 32 % 16, tl = r / G;
    const float* qr = q + ((uint64_t) (t0 + tl) * heads + kvh * G + r % G) * kHead;
#pragma unroll
    for (int g = 0; g < kHead / kTile; g++)
#pragma unroll
      for (int e = 0; e < kTile; e++) qa[g][e] = (_Float16) (tl < nt ? qr[g * kTile + e] : 0.0f);
  }
  float m[osmos_slots], l[osmos_slots];
#pragma unroll
  for (int r = 0; r < osmos_slots; r++) m[r] = -INFINITY, l[r] = 0.0f;
  Raw<HD> raw;
  span_fetch<HD>(raw, kc, vc, row, kvh, keys, k0, k1);
  f8 o[kCols] = {};
  for (uint32_t kb = k0; kb < k1; kb += kSpanKeys) {
    __syncthreads();
    span_store<HD>(raw, ks, vt);
    __syncthreads();
    if (kb + kSpanKeys < k1) span_fetch<HD>(raw, kc, vc, row, kvh, keys, kb + kSpanKeys, k1);
    f8 s0 = {}, s1 = {};
    for (int g = 0; g < kHead / kTile; g++) {
      s0 = osmos_mma(qa[g], osmos_load_b_s(ks + g * kTile, kRowLds), s0);
      s1 = osmos_mma(qa[g], osmos_load_b_s(ks + kTile * kRowLds + g * kTile, kRowLds), s1);
    }
    // pass 0 the rows the step is whole for; then each token whose position is in it (last first), V's keys past it
    // zeroed - its rows' step its position's whatever the group
    const uint32_t last = kb + kSpanKeys - 1;
    const bool diag = last > first;
    int32_t p = (int32_t) (top < last ? top : last - 1) + 1;
    for (bool pass0 = true;; pass0 = false) {
      bool upd[osmos_slots], mine = pass0;
      if (!pass0) {
        if (--p < (int32_t) kb || p < (int32_t) first) break;
        __syncthreads();  // the last pass's reads of vt
        for (uint32_t i = threadIdx.x; i < (uint32_t) kHead * kSpanKeys; i += kSpanWaves * 32) {
          const uint32_t d = i / kSpanKeys, k = i % kSpanKeys;
          if (kb + k > (uint32_t) p) vt[d * kSpanKeyLds + k] = (_Float16) 0.0f;
        }
        __syncthreads();
      }
#pragma unroll
      for (int r = 0; r < osmos_slots; r++) {
        const int32_t at = span_lim(r, rt, G, nt, first);
        upd[r] = pass0 ? at >= (int32_t) last : at == p;
        mine |= upd[r];
      }
      if (__any(mine)) span_tile<HD>(s0, s1, rt, G, nt, first, upd, kb, scale, vt, ps[wave], m, l, o);
      if (!diag) break;
    }
  }
  const uint64_t base = ((uint64_t) sp * kv_heads + kvh) * n_tok * G;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const uint32_t r = rt * kTile + osmos_acc_row(j), R = t0 * G + r;
    if (r >= nt * G) continue;
    if (osmos_acc_col(j) == 0)
      ml[(base + R) * 2] = m[osmos_slot(j)], ml[(base + R) * 2 + 1] = l[osmos_slot(j)];
#pragma unroll
    for (int c = 0; c < kCols; c++)
      po[(base + R) * kHead + c * kTile + osmos_acc_col(j)] = o[c][j];
  }
}

// a (token, head)'s spans merged in order (fa_merge_kernel's, by span)
template <int HD>
__global__ __launch_bounds__(HD) void span_merge_kernel(const float* ml, const float* po, const float* gate,
                                                        uint32_t n_tok, uint32_t heads, uint32_t kv_heads,
                                                        uint32_t pos0, uint32_t t0, uint32_t gate_span, float* out,
                                                        Q4xX16 side) {
  const uint32_t t = blockIdx.x / heads, h = blockIdx.x % heads, d = threadIdx.x, G = heads / kv_heads;
  const uint32_t rows = n_tok * G, R = t * G + h % G, kvh = h / G, last = (pos0 + t) / kSpan;
  float mx = -INFINITY;
  for (uint32_t c = 0; c <= last; c++) mx = fmaxf(mx, ml[(((uint64_t) c * kv_heads + kvh) * rows + R) * 2]);
  float sum = 0.0f, acc = 0.0f;
  for (uint32_t c = 0; c <= last; c++) {
    const uint64_t at = ((uint64_t) c * kv_heads + kvh) * rows + R;
    const float w = expf(ml[at * 2] - mx);
    sum += ml[at * 2 + 1] * w, acc += po[at * HD + d] * w;
  }
  const uint64_t i = ((uint64_t) t * heads + h) * HD + d;
  out[i] = acc / sum * sigmoid(gate[i / gate_span]);
  x16_side(side, i + (uint64_t) t0 * heads * HD, out[i]);
}

// a prompt's sparse reads (q4x_attn_lists): a block one wave, a token's KV head, its list's entries 16 a tile in order
// with one running (max, sum, O) - K's B operands straight from the cache (a lane a key's 16 dims a k-step: one load),
// V^T through LDS; a tile's next K and V fetched while its softmax and P V run. entries past the list loaded as
// zeros: a token's bits its list's alone
constexpr int kListGqa = 4;  // a KV head's query heads at most (LFM3's 4)
constexpr int kListVBlocks = Q4X_QK_HEAD_NARROW / Q4X_KV_BLOCK / 2;  // a lane's V blocks: its key's half of the dims

struct ListRaw {
  uint64_t kq[Q4X_QK_HEAD_NARROW / kTile][2];  // a k-step's 16 int8s of the lane's key
  _Float16 ksc[Q4X_QK_HEAD_NARROW / Q4X_KV_BLOCK];
  uint64_t vq[kListVBlocks][Q4X_KV_BLOCK / 8];  // the lane's key's V, its half (lane / 16) of the dims
  _Float16 vsc[kListVBlocks];
};

__device__ inline void list_fetch(ListRaw& w, const uint8_t* kc, const uint8_t* vc, uint64_t row, uint32_t kvh,
                                  const uint32_t* list, uint32_t kb, uint32_t len) {
  constexpr int HD = Q4X_QK_HEAD_NARROW, kBlocks = HD / Q4X_KV_BLOCK;
  const uint32_t lane = threadIdx.x % 32, e = kb + lane % 16;
  const uint8_t* kr = kc + (uint64_t) (e < len ? list[e] : 0) * row + kvh * kBlocks * Q4X_KV_BLOCK_BYTES;
#pragma unroll
  for (int b = 0; b < kBlocks; b++) {
    w.ksc[b] = e < len ? *(const _Float16*) (kr + b * Q4X_KV_BLOCK_BYTES) : (_Float16) 0.0f;
#pragma unroll
    for (int h = 0; h < 2; h++) {
      uint64_t x[2] = {0, 0};
      if (e < len) __builtin_memcpy(x, kr + b * Q4X_KV_BLOCK_BYTES + 2 + h * kTile, kTile);
      w.kq[2 * b + h][0] = x[0], w.kq[2 * b + h][1] = x[1];
    }
  }
  const uint8_t* vr = vc + (uint64_t) (e < len ? list[e] : 0) * row +
                      (kvh * kBlocks + lane / 16 * kListVBlocks) * Q4X_KV_BLOCK_BYTES;
#pragma unroll
  for (int b = 0; b < kListVBlocks; b++) {
    w.vsc[b] = e < len ? *(const _Float16*) (vr + b * Q4X_KV_BLOCK_BYTES) : (_Float16) 0.0f;
    uint64_t x[Q4X_KV_BLOCK / 8] = {};
    if (e < len) __builtin_memcpy(x, vr + b * Q4X_KV_BLOCK_BYTES + 2, Q4X_KV_BLOCK);
#pragma unroll
    for (int i = 0; i < Q4X_KV_BLOCK / 8; i++) w.vq[b][i] = x[i];
  }
}

__global__ __launch_bounds__(32) void fa_list_kernel(const float* q, const uint8_t* kc, const uint8_t* vc,
                                                     uint32_t heads, uint32_t kv_heads, const uint32_t* lists,
                                                     const int32_t* counts, uint32_t stride, const float* gate,
                                                     uint32_t gate_span, float scale, float* out) {
  constexpr int HD = Q4X_QK_HEAD_NARROW, kLds = kRowLds<HD>, kCols = HD / kTile;
  __shared__ __attribute__((aligned(32))) _Float16 qs[kTile * kLds], vt[HD * kKeyLds], ps[kTile * kKeyLds];
  const uint32_t t = blockIdx.x, kvh = blockIdx.y, G = heads / kv_heads, lane = threadIdx.x % 32;
  const uint32_t len = (uint32_t) counts[t];
  const uint32_t* list = lists + (uint64_t) t * stride;
  const uint64_t row = (uint64_t) kv_heads * HD / Q4X_KV_BLOCK * Q4X_KV_BLOCK_BYTES;
  for (uint32_t i = lane; i < kTile * HD; i += 32) {  // the KV head's query rows, f16 (past G zero)
    const uint32_t r = i / HD, d = i % HD;
    qs[r * kLds + d] = (_Float16) (r < G ? q[((uint64_t) t * heads + kvh * G + r) * HD + d] : 0.0f);
  }
  float m[osmos_slots], l[osmos_slots];
#pragma unroll
  for (int r = 0; r < osmos_slots; r++) m[r] = -INFINITY, l[r] = 0.0f;
  f8 o[kCols] = {};
  ListRaw raw;
  list_fetch(raw, kc, vc, row, kvh, list, 0, len);
  osmos_wave_barrier();
  for (uint32_t kb = 0; kb < len; kb += kTile) {
    f8 sacc = {};
#pragma unroll
    for (int g = 0; g < HD / kTile; g++) {  // K's operand: its int8s times their block's scale, exact in F32, to f16
      osmos_fb kf;
      const float sk = (float) raw.ksc[g / 2];
#pragma unroll
      for (int e = 0; e < kTile; e++) kf[e] = (_Float16) (sk * (float) (int8_t) (raw.kq[g][e / 8] >> (8 * (e % 8))));
      sacc = osmos_mma(osmos_load_a(qs + g * kTile, kLds), kf, sacc);
    }
#pragma unroll
    for (int b = 0; b < kListVBlocks; b++) {  // V^T, f16: the lane's key's column, its half of the dims
      const float sv = (float) raw.vsc[b];
      const uint32_t d0 = (lane / 16 * kListVBlocks + b) * Q4X_KV_BLOCK;
#pragma unroll
      for (int j = 0; j < Q4X_KV_BLOCK; j++)
        vt[(d0 + j) * kKeyLds + lane % 16] = (_Float16) (sv * (float) (int8_t) (raw.vq[b][j / 8] >> (8 * (j % 8))));
    }
    if (kb + kTile < len) list_fetch(raw, kc, vc, row, kvh, list, kb + kTile, len);  // the next tile's, in flight
    float sv[8], alpha[osmos_slots];
#pragma unroll
    for (int j = 0; j < 8; j++) sv[j] = kb + osmos_acc_col(j) < len ? sacc[j] * scale : -INFINITY;
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
          ps[osmos_acc_row(j) * kKeyLds + osmos_acc_col(j)] = (_Float16) pv;
        }
      alpha[r] = mn == -INFINITY ? 1.0f : expf(m[r] - mn);
      l[r] = l[r] * alpha[r] + osmos_row_sum(sum), m[r] = mn;
    }
    osmos_wave_barrier();
    const osmos_fa pa = osmos_load_a(ps, kKeyLds);
#pragma unroll
    for (int c = 0; c < kCols; c++) {
#pragma unroll
      for (int j = 0; j < 8; j++) o[c][j] *= alpha[osmos_slot(j)];
      o[c] = osmos_mma(pa, osmos_load_b_s(vt + c * kTile * kKeyLds, kKeyLds), o[c]);
    }
    osmos_wave_barrier();
  }
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const uint32_t r = osmos_acc_row(j);
    if (r >= G) continue;
#pragma unroll
    for (int c = 0; c < kCols; c++) {
      const uint64_t at = ((uint64_t) t * heads + kvh * G + r) * HD + c * kTile + osmos_acc_col(j);
      out[at] = o[c][j] / l[osmos_slot(j)] * sigmoid(gate[at / gate_span]);
    }
  }
}

extern "C" bool q4x_attn_lists(const float* q, const void* kc, const void* vc, uint32_t n_tok, uint32_t heads,
                               uint32_t kv_heads, uint32_t dim, const uint32_t* lists, const int32_t* counts,
                               uint32_t list_stride, const float* gate, const AttnShape* shape, float* out,
                               void* stream) {
  const AttnShape sh = shape ? *shape : AttnShape{1, 0, 0};
  if (!q || !kc || !vc || !lists || !counts || !gate || !out || !n_tok || !kv_heads || heads % kv_heads ||
      heads / kv_heads > (uint32_t) kListGqa || dim != Q4X_QK_HEAD_NARROW || !list_stride || sh.window || sh.slots ||
      (sh.gate_span != 1 && sh.gate_span != dim))
    return false;
  fa_list_kernel<<<dim3(n_tok, kv_heads), 32, 0, st(stream)>>>(
      q, (const uint8_t*) kc, (const uint8_t*) vc, heads, kv_heads, lists, counts, list_stride, gate, sh.gate_span,
      1.0f / sqrtf((float) dim), out);
  return ok();
}

extern "C" bool q4x_kv_put(const float* k, const float* v, uint32_t n_tok, uint32_t values, void* kc, void* vc,
                           void* stream) {
  if (!k || !v || !kc || !vc || !n_tok || !values || values % Q4X_KV_BLOCK) return false;
  const uint32_t blocks = n_tok * values / Q4X_KV_BLOCK;
  kv_put_kernel<<<(blocks + 255) / 256, 256, 0, st(stream)>>>(k, blocks, (uint8_t*) kc);
  if (!ok()) return false;
  kv_put_kernel<<<(blocks + 255) / 256, 256, 0, st(stream)>>>(v, blocks, (uint8_t*) vc);
  return ok();
}

extern "C" uint64_t q4x_attn_scratch(uint32_t heads, uint32_t max_ctx) {
  const uint64_t chunks = (max_ctx + kChunk - 1) / kChunk, pairs = chunks < kPieceChunks ? kPieceChunks : chunks;
  return pairs * heads * (kHead + 2);  // a piece of at least one token at the context's end
}

// the span path's pairs: a piece of a group's tokens (its least) at the context's end, or the chunk path's
static uint64_t span_pairs(uint32_t heads, uint32_t kv_heads, uint64_t end) {
  const uint64_t chunks = (end + kChunk - 1) / kChunk, spans = (end + kSpan - 1) / kSpan;
  const uint64_t tt = (uint64_t) kSpanRows / (heads / kv_heads), least = tt * spans;
  const uint64_t pairs = chunks < kPieceChunks ? kPieceChunks : chunks;
  return pairs > least ? pairs : least;
}

extern "C" uint64_t q4x_attn_scratch_dim(uint32_t heads, uint32_t kv_heads, uint32_t dim, uint32_t max_ctx) {
  if (dim != Q4X_QK_HEAD_NARROW || !kv_heads || heads % kv_heads) return q4x_attn_scratch(heads, max_ctx);
  const uint64_t a = span_pairs(heads, kv_heads, max_ctx) * heads * (dim + 2), b = q4x_attn_scratch(heads, max_ctx);
  return a > b ? a : b;
}

template <int HD>
static bool attn(const float* q, const void* kc, const void* vc, uint32_t n_tok, uint32_t heads, uint32_t kv_heads,
                 uint32_t pos0, const float* gate, const Q4xPick* pick, const AttnShape& sh, float* scratch,
                 float* out, const Q4xX16* side, void* stream) {
  if constexpr (HD == Q4X_QK_HEAD_NARROW)
  if (!pick && !sh.window && !sh.slots) {  // LFM3's dense prompt: by spans
    const float scale = 1.0f / sqrtf((float) HD);
    const uint32_t G = heads / kv_heads, tt = kSpanRows / G;
    // the scratch's pairs (q4x_attn_scratch's for a context at least this step's end): a piece's tokens x its spans
    // (q4x_attn_scratch_dim's: a piece of at least a group at the step's end)
    const uint64_t end = (uint64_t) pos0 + n_tok, pairs = span_pairs(heads, kv_heads, end), spans = (end - 1) / kSpan + 1;
    uint32_t piece = (uint32_t) (pairs / spans / tt * tt);
    piece = piece > kSpanStepTokens ? kSpanStepTokens : piece;
    for (uint32_t t0 = 0; t0 < n_tok; t0 += piece) {
      const uint32_t n = n_tok - t0 < piece ? n_tok - t0 : piece, p0 = pos0 + t0, ps = (p0 + n - 1) / kSpan + 1;
      float *ml = scratch, *po = scratch + (uint64_t) 2 * ps * kv_heads * n * G;
      const uint64_t at = (uint64_t) t0 * heads * HD;
      fa_span_kernel<HD><<<dim3((n + tt - 1) / tt, kv_heads, ps), kSpanWaves * 32, 0, st(stream)>>>(
          q + at, (const uint8_t*) kc, (const uint8_t*) vc, n, heads, kv_heads, p0, tt, scale, ml, po);
      if (!ok()) return false;
      span_merge_kernel<HD><<<n * heads, HD, 0, st(stream)>>>(ml, po, gate + at / sh.gate_span, n, heads, kv_heads,
                                                              p0, t0, sh.gate_span, out + at, side_of(side));
      if (!ok()) return false;
    }
    return true;
  }
  const Q4xPick pk = pick ? *pick : Q4xPick{nullptr, nullptr, 1, 0};
  const float scale = 1.0f / sqrtf((float) HD);
  // a step's token's most entries: its position's (a window's at most), or with picks at most its blocks' and a tail
  uint64_t dense = (uint64_t) pos0 + n_tok;
  dense = sh.window && dense > sh.window ? sh.window : dense;
  const uint64_t picked = (uint64_t) pk.ratio * pk.blocks + pk.ratio - 1;
  const uint64_t longest = pick && picked < dense ? picked : dense;
  const uint32_t most = (uint32_t) ((longest - 1) / kChunk + 1), fit = kPieceChunks / most;
  const uint32_t piece = fit < 1 ? 1 : fit < (uint32_t) kStepTokens ? fit : (uint32_t) kStepTokens;
  for (uint32_t t0 = 0; t0 < n_tok; t0 += piece) {  // a piece of the step at a time: the scratch's tokens
    const uint32_t n = n_tok - t0 < piece ? n_tok - t0 : piece;
    const uint32_t rows = n * (heads / kv_heads), p0 = pos0 + t0;
    uint64_t piece_dense = (uint64_t) p0 + n;
    piece_dense = sh.window && piece_dense > sh.window ? sh.window : piece_dense;
    const uint64_t piece_long = pick && picked < piece_dense ? picked : piece_dense;
    const uint32_t chunks = (uint32_t) ((piece_long - 1) / kChunk + 1);
    Q4xPick ppk = pk;  // the piece's tokens' rows of the picks
    if (pick) ppk.sel = pk.sel + (uint64_t) t0 * pk.blocks, ppk.npick = pk.npick + t0;
    float *ml = scratch, *po = scratch + (uint64_t) 2 * chunks * kv_heads * rows;
    const uint64_t at = (uint64_t) t0 * heads * HD;
    // dense keys from 0: the chunks whole for a group's first token the group's (fa_whole_kernel), its tokens' rest
    // their own - not under a window (a token's keys start at its own place) or a ring
    const uint32_t tt = pick || sh.window ? 0 : kWholeRows / (heads / kv_heads);
    if (tt && (p0 + n) / kChunk) {  // some group's first token past a whole chunk (fa_kernel skips them all the same)
      fa_whole_kernel<HD><<<dim3((n + tt - 1) / tt, kv_heads, ((p0 + n) / kChunk + kWholeChunks - 1) / kWholeChunks),
                            kWholeWaves * 32, 0, st(stream)>>>(q + at, (const uint8_t*) kc, (const uint8_t*) vc, n,
                                                               heads, kv_heads, p0, tt, scale, ml, po);
      if (!ok()) return false;
    }
    fa_kernel<HD><<<dim3(n, kv_heads, chunks), kWaves * 32, 0, st(stream)>>>(
        q + at, (const uint8_t*) kc, (const uint8_t*) vc, n, heads, kv_heads, p0, ppk, tt, sh.window, sh.slots, scale,
        ml, po);
    if (!ok()) return false;
    fa_merge_kernel<HD><<<n * heads, HD, 0, st(stream)>>>(ml, po, gate + (uint64_t) t0 * heads * HD / sh.gate_span, n,
                                                          heads, kv_heads, p0, ppk, t0, sh.window, sh.gate_span,
                                                          out + at, side_of(side));
    if (!ok()) return false;
  }
  return true;
}

extern "C" bool q4x_attn(const float* q, const void* kc, const void* vc, uint32_t n_tok, uint32_t heads,
                         uint32_t kv_heads, uint32_t dim, uint32_t pos0, const float* gate, const Q4xPick* pick,
                         const AttnShape* shape, float* scratch, float* out, const Q4xX16* side, void* stream) {
  const AttnShape sh = shape ? *shape : AttnShape{1, 0, 0};
  if (!n_tok || !heads || !kv_heads || heads % kv_heads || heads / kv_heads > kTile ||
      (dim != (uint32_t) kHead && dim != Q4X_QK_HEAD_NARROW) || (sh.gate_span != 1 && sh.gate_span != dim) ||
      (sh.slots && (!sh.window || sh.slots < sh.window + n_tok - 1)) || (uint64_t) pos0 + n_tok > OPS_ATTN_MAX_KEYS)
    return false;
  if (!q || !kc || !vc || !gate || !scratch || !out ||
      (pick && (!pick->npick || !pick->sel || pick->ratio < 2 || !pick->blocks || sh.window)))
    return false;
  return dim == (uint32_t) kHead
             ? attn<Q4X_HEAD_DIM>(q, kc, vc, n_tok, heads, kv_heads, pos0, gate, pick, sh, scratch, out, side, stream)
             : attn<Q4X_QK_HEAD_NARROW>(q, kc, vc, n_tok, heads, kv_heads, pos0, gate, pick, sh, scratch, out, side,
                                        stream);
}
