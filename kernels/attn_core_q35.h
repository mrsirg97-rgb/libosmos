// the attention core's decode and LSA read for the reads-apart families (osmos1.7b, q35: attn_core.h's apart), as
// q35_attn.hip had it, moved whole. a decode step's attention (q35_attn_decode): flash decoding over the Q8_0 cache (q4x_kv_put's rows), every key a
// token's own position or before. a warp a run of keys of one KV head for one token - its lanes a key's 256 dims, 8
// each (a load's 8 int8 within one Q8_0 block, its f16 scale), the KV head's G query heads' scores summed across the
// warp, an online softmax a head and O's 8 dims a lane a head; each warp's (max, sum, O) to scratch, then a merge
// block a (token, query head): the warps in order, the gate's sigmoid on the result (q4x_attn's). F32 throughout
// (fa_kernel's f16 Q, K, V and P not taken): q35's own bits. q4x_attn's prefill unchanged
#pragma once
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "block.h"
#include "launch.h"
extern "C" {
#include "osmos/gpu/q35_ops.h"
#include "osmos/gpu/q4x_ops.h"
}

namespace attn_q35 {


constexpr int kGMax = Q35_ATTN_GQA_MAX, kWarps = 4;
constexpr int kHeadWide = Q4X_HEAD_DIM, kHeadNarrow = Q4X_QK_HEAD_NARROW;  // Qwen's 256, LFM3's 128

// a lane's L values of a cache row's head kvh (c = kvh dim + L lane): the block's scale times its int8s (L 8 or 4:
// one load within one Q8_0 block)
template <int L>
__device__ inline void row_vals(const uint8_t* cache, uint64_t row, uint64_t at, uint32_t c, float* v) {
  static_assert(L % 2 == 0 && Q4X_KV_BLOCK % L == 0, "a lane's dims one load within one Q8_0 block");
  const uint8_t* b = cache + at * row + (uint64_t) (c / Q4X_KV_BLOCK) * Q4X_KV_BLOCK_BYTES;
  const float d = (float) *(const _Float16*) b;
  const uint16_t* q = (const uint16_t*) (b + 2 + c % Q4X_KV_BLOCK);  // a block 34 bytes: its int8s 2-byte aligned
#pragma unroll
  for (int j = 0; j < L / 2; j++) {
    const uint16_t u = q[j];
    v[2 * j] = d * (float) (int8_t) (u & 0xFF), v[2 * j + 1] = d * (float) (int8_t) (u >> 8);
  }
}

// a token's keys: positions lo .. its own (lo past 0 under a window), or (list: one token) the read's, entry k list[k]
struct Keys {
  uint32_t lo, n;
};
__device__ inline Keys keys_of(uint32_t pos, uint32_t window, const uint32_t* list_len) {
  if (list_len) return {0, *list_len};
  const uint32_t lo = window && pos + 1 > window ? pos + 1 - window : 0;
  return {lo, pos + 1 - lo};
}

// a token's keys a part (at least 16) and its parts (at most Q35_ATTN_PARTS) from its own key count - not the call's
// last token's: a token's split its own, its bits whatever the step. len_t: min(its position + 1, the window), or a
// list's length (a one-token call)
__device__ inline uint32_t part_keys(uint32_t len_t) {
  const uint32_t per = (len_t + Q35_ATTN_PARTS - 1) / Q35_ATTN_PARTS;
  return per < 16 ? 16 : per;
}
__device__ inline uint32_t token_len(uint32_t pos, uint32_t window, uint32_t list_len) {
  if (list_len) return list_len;
  return window && pos + 1 > window ? window : pos + 1;
}

template <int HD, int G>
__global__ __launch_bounds__(32 * kWarps) void decode_kernel(const float* q, const uint8_t* kc, const uint8_t* vc,
                                                             uint32_t heads, uint32_t kv_heads, uint32_t pos0,
                                                             uint32_t window, uint32_t slots, float scale,
                                                             uint32_t list_max, uint32_t parts, const uint32_t* list,
                                                             const uint32_t* list_len, float* ml, float* po) {
  constexpr int L = HD / 32;
  const uint32_t lane = threadIdx.x % 32, kvh = blockIdx.x, t = blockIdx.z;
  const uint32_t part = blockIdx.y * kWarps + threadIdx.x / 32;
  const Keys keys = keys_of(pos0 + t, window, list ? list_len : nullptr);
  const uint32_t len_t = token_len(pos0 + t, window, list ? list_max : 0), per = part_keys(len_t);
  const uint32_t parts_t = (len_t + per - 1) / per;  // (parts: the stride, the call's most)
  const uint32_t k0 = part * per, k1 = k0 + per < keys.n ? k0 + per : keys.n;
  const uint64_t row = (uint64_t) kv_heads * HD / Q4X_KV_BLOCK * Q4X_KV_BLOCK_BYTES;
  const uint32_t c = kvh * HD + lane * L;
  float qv[G][L], o[G][L], m[G], l[G];
#pragma unroll
  for (int h = 0; h < G; h++) {
    const float* qh = q + ((uint64_t) t * heads + kvh * G + h) * HD + lane * L;
#pragma unroll
    for (int j = 0; j < L; j++) qv[h][j] = qh[j] * scale, o[h][j] = 0.0f;
    m[h] = -INFINITY, l[h] = 0.0f;
  }
  for (uint32_t k = k0; k < k1; k++) {
    float kv[L], vv[L];
    const uint64_t p = list ? list[k] : keys.lo + k, at = slots ? p % slots : p;
    row_vals<L>(kc, row, at, c, kv);
    row_vals<L>(vc, row, at, c, vv);
#pragma unroll
    for (int h = 0; h < G; h++) {
      float s = 0.0f;
#pragma unroll
      for (int j = 0; j < L; j++) s = __fmaf_rn(qv[h][j], kv[j], s);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) s += __shfl_xor(s, off);
      const float mn = fmaxf(m[h], s), a = expf(m[h] - mn), p = expf(s - mn);
      l[h] = l[h] * a + p, m[h] = mn;
#pragma unroll
      for (int j = 0; j < L; j++) o[h][j] = __fmaf_rn(p, vv[j], o[h][j] * a);
    }
  }
  if (part >= parts_t) return;
  // a part past the keys (k0 >= its count) writes (-inf, 0): the merge weighs it 0
  const uint64_t base = ((uint64_t) t * heads + kvh * G) * parts + part;
#pragma unroll
  for (int h = 0; h < G; h++) {
    const uint64_t at = base + (uint64_t) h * parts;
    if (lane == 0) ml[at * 2] = m[h], ml[at * 2 + 1] = l[h];
#pragma unroll
    for (int j = 0; j < L; j++) po[at * HD + lane * L + j] = o[h][j];
  }
}

// a block a (token, query head), a thread a dim: the parts in order, the gate's sigmoid on the result - a gate a
// value (span 1) or a head (span HD)
template <int HD>
__global__ __launch_bounds__(HD) void merge_kernel(const float* ml, const float* po, const float* gate, uint32_t heads,
                                                   uint32_t parts, uint32_t gate_span, uint32_t pos0, uint32_t window,
                                                   uint32_t list_max, float* out) {
  const uint32_t th = blockIdx.x, d = threadIdx.x;  // th: token t's query head h, t heads + h
  const uint32_t len_t = token_len(pos0 + th / heads, window, list_max), per = part_keys(len_t);
  const uint32_t parts_t = (len_t + per - 1) / per;  // its own parts, at the call's stride
  const uint64_t base = (uint64_t) th * parts;
  float mx = -INFINITY;
  for (uint32_t p = 0; p < parts_t; p++) mx = fmaxf(mx, ml[(base + p) * 2]);
  float sum = 0.0f, acc = 0.0f;
  for (uint32_t p = 0; p < parts_t; p++) {
    const float mp = ml[(base + p) * 2];
    if (mp == -INFINITY) continue;
    const float w = expf(mp - mx);
    sum += ml[(base + p) * 2 + 1] * w, acc += po[(base + p) * HD + d] * w;
  }
  const uint64_t i = (uint64_t) th * HD + d;
  out[i] = acc / sum * sigmoid(gate[i / gate_span]);
}


template <int HD>
inline bool launch(const float* q, const uint8_t* k8, const uint8_t* v8, uint32_t n_tok, uint32_t heads,
                   uint32_t kv_heads, uint32_t pos0, uint32_t window, uint32_t slots, uint32_t gate_span,
                   const uint32_t* list, const uint32_t* list_len, uint32_t len, const float* gate, float* scratch,
                   float* out, hipStream_t s) {
  // keys a warp: at least 16, the parts at most Q35_ATTN_PARTS a (token, head) - each token's split its own
  // (part_keys), the grid the most parts any token of the call wants
  // (a shorter token may want more parts than the longest - 8,703 keys 512 of 17, 8,706 484 of 18: the most of any)
  const uint32_t list_max = list ? len : 0;
  uint32_t parts = 0;
  for (uint32_t t = 0; t < n_tok; t++) {
    const uint32_t len_t = list_max ? list_max : window && pos0 + t + 1 > window ? window : pos0 + t + 1;
    uint32_t per = (len_t + Q35_ATTN_PARTS - 1) / Q35_ATTN_PARTS;
    per = per < 16 ? 16 : per;
    parts = (len_t + per - 1) / per > parts ? (len_t + per - 1) / per : parts;
  }
  const uint32_t blocks = (parts + kWarps - 1) / kWarps;
  float* ml = scratch;
  float* po = scratch + (uint64_t) n_tok * heads * parts * 2;
  const dim3 grid(kv_heads, blocks, n_tok);
  const float scale = 1.0f / sqrtf((float) HD);
#define Q35_DECODE(G)                                                                                              \
  decode_kernel<HD, G><<<grid, 32 * kWarps, 0, s>>>(q, k8, v8, heads, kv_heads, pos0, window, slots, scale,        \
                                                     list_max, parts, list, list_len, ml, po)
  switch (heads / kv_heads) {
    case 1: Q35_DECODE(1); break;
    case 2: Q35_DECODE(2); break;
    case 4: Q35_DECODE(4); break;
    case 6: Q35_DECODE(6); break;
    case 8: Q35_DECODE(8); break;
    default: return false;
  }
#undef Q35_DECODE
  if (!launches::ok()) return false;
  merge_kernel<HD><<<n_tok * heads, HD, 0, s>>>(ml, po, gate, heads, parts, gate_span, pos0, window, list_max, out);
  return launches::ok();
}

inline bool run(const float* q, const void* kc, const void* vc, uint32_t n_tok, uint32_t heads, uint32_t kv_heads,
                uint32_t dim, const AttnShape* shape, uint32_t pos0, const uint32_t* list, const uint32_t* list_len,
                uint32_t list_max, const float* gate, float* scratch, float* out, void* stream) {
  const AttnShape sh = shape ? *shape : AttnShape{1, 0, 0};
  if (!q || !kc || !vc || !gate || !scratch || !out || !n_tok || n_tok > Q35_ATTN_DECODE_MAX || !kv_heads ||
      heads % kv_heads || heads / kv_heads > kGMax || heads / kv_heads == 0 ||
      (list && (n_tok != 1 || !list_len || !list_max || sh.window || sh.slots)) ||
      (dim != (uint32_t) kHeadWide && dim != (uint32_t) kHeadNarrow) || (sh.gate_span != 1 && sh.gate_span != dim) ||
      (sh.slots && (!sh.window || sh.slots < sh.window + n_tok - 1)))
    return false;
  // the longest key list: the last token's (a window's at most), or a read's bound
  uint32_t len = list ? list_max : pos0 + n_tok;
  len = sh.window && len > sh.window ? sh.window : len;
  const hipStream_t s = launches::st(stream);
  const uint8_t *k8 = (const uint8_t*) kc, *v8 = (const uint8_t*) vc;
  return dim == (uint32_t) kHeadWide
             ? launch<kHeadWide>(q, k8, v8, n_tok, heads, kv_heads, pos0, sh.window, sh.slots, sh.gate_span, list,
                                 list_len, len, gate, scratch, out, s)
             : launch<kHeadNarrow>(q, k8, v8, n_tok, heads, kv_heads, pos0, sh.window, sh.slots, sh.gate_span, list,
                                   list_len, len, gate, scratch, out, s);
}


}  // namespace attn_q35
