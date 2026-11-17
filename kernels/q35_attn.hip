// a decode step's attention (q35_attn_decode): flash decoding over the Q8_0 cache (q4x_kv_put's rows), every key a
// token's own position or before. a warp a run of keys of one KV head for one token - its lanes a key's 256 dims, 8
// each (a load's 8 int8 within one Q8_0 block, its f16 scale), the KV head's G query heads' scores summed across the
// warp, an online softmax a head and O's 8 dims a lane a head; each warp's (max, sum, O) to scratch, then a merge
// block a (token, query head): the warps in order, the gate's sigmoid on the result (q4x_attn's). F32 throughout
// (fa_kernel's f16 Q, K, V and P not taken): q35's own bits. q4x_attn's prefill unchanged
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "block.h"
#include "launch.h"
#include "osmos/gpu/q35_ops.h"
#include "osmos/gpu/q4x_ops.h"

namespace {

constexpr int kHead = Q4X_HEAD_DIM, kLane = kHead / 32, kGMax = Q35_ATTN_GQA_MAX, kWarps = 4;
static_assert(kLane == 8 && Q4X_KV_BLOCK % kLane == 0, "a lane's dims one load within one Q8_0 block");

// a lane's 8 values of a cache row's head kvh (c = kvh 256 + 8 lane): the block's scale times its int8s
__device__ inline void row8(const uint8_t* cache, uint64_t row, uint64_t pos, uint32_t c, float* v) {
  const uint8_t* b = cache + pos * row + (uint64_t) (c / Q4X_KV_BLOCK) * Q4X_KV_BLOCK_BYTES;
  const float d = (float) *(const _Float16*) b;
  const uint16_t* q = (const uint16_t*) (b + 2 + c % Q4X_KV_BLOCK);  // a block 34 bytes: its int8s 2-byte aligned
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const uint16_t u = q[j];
    v[2 * j] = d * (float) (int8_t) (u & 0xFF), v[2 * j + 1] = d * (float) (int8_t) (u >> 8);
  }
}

template <int G>
__global__ __launch_bounds__(32 * kWarps) void decode_kernel(const float* q, const uint8_t* kc, const uint8_t* vc,
                                                             uint32_t heads, uint32_t kv_heads, uint32_t pos0,
                                                             uint32_t per, uint32_t parts, const uint32_t* list,
                                                             const uint32_t* list_len, float* ml, float* po) {
  const uint32_t lane = threadIdx.x % 32, kvh = blockIdx.x, t = blockIdx.z;
  // a token's keys: positions 0 .. its own, or (list: one token) the read's positions, entry k list[k], their count
  // the device's (*list_len: the read's own; the launch sized for the longest a read can be)
  const uint32_t part = blockIdx.y * kWarps + threadIdx.x / 32, len = list ? *list_len : pos0 + t + 1;
  const uint32_t k0 = part * per, k1 = k0 + per < len ? k0 + per : len;
  const uint64_t row = (uint64_t) kv_heads * kHead / Q4X_KV_BLOCK * Q4X_KV_BLOCK_BYTES;
  const uint32_t c = kvh * kHead + lane * kLane;
  float qv[G][kLane], o[G][kLane], m[G], l[G];
#pragma unroll
  for (int h = 0; h < G; h++) {
    const float* qh = q + ((uint64_t) t * heads + kvh * G + h) * kHead + lane * kLane;
#pragma unroll
    for (int j = 0; j < kLane; j++) qv[h][j] = qh[j] * (1.0f / 16.0f), o[h][j] = 0.0f;  // 1 / sqrt(256)
    m[h] = -INFINITY, l[h] = 0.0f;
  }
  for (uint32_t k = k0; k < k1; k++) {
    float kv[kLane], vv[kLane];
    const uint64_t at = list ? list[k] : k;
    row8(kc, row, at, c, kv);
    row8(vc, row, at, c, vv);
#pragma unroll
    for (int h = 0; h < G; h++) {
      float s = 0.0f;
#pragma unroll
      for (int j = 0; j < kLane; j++) s = __fmaf_rn(qv[h][j], kv[j], s);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) s += __shfl_xor(s, off);
      const float mn = fmaxf(m[h], s), a = expf(m[h] - mn), p = expf(s - mn);
      l[h] = l[h] * a + p, m[h] = mn;
#pragma unroll
      for (int j = 0; j < kLane; j++) o[h][j] = __fmaf_rn(p, vv[j], o[h][j] * a);
    }
  }
  if (part >= parts) return;
  // a part past the keys (k0 >= len) writes (-inf, 0): the merge weighs it 0
  const uint64_t base = ((uint64_t) t * heads + kvh * G) * parts + part;
#pragma unroll
  for (int h = 0; h < G; h++) {
    const uint64_t at = base + (uint64_t) h * parts;
    if (lane == 0) ml[at * 2] = m[h], ml[at * 2 + 1] = l[h];
#pragma unroll
    for (int j = 0; j < kLane; j++) po[at * kHead + lane * kLane + j] = o[h][j];
  }
}

__global__ __launch_bounds__(kHead) void merge_kernel(const float* ml, const float* po, const float* gate,
                                                      uint32_t heads, uint32_t parts, float* out) {
  const uint32_t th = blockIdx.x, d = threadIdx.x;  // th: token t's query head h, t heads + h
  const uint64_t base = (uint64_t) th * parts;
  float mx = -INFINITY;
  for (uint32_t p = 0; p < parts; p++) mx = fmaxf(mx, ml[(base + p) * 2]);
  float sum = 0.0f, acc = 0.0f;
  for (uint32_t p = 0; p < parts; p++) {
    const float mp = ml[(base + p) * 2];
    if (mp == -INFINITY) continue;
    const float w = expf(mp - mx);
    sum += ml[(base + p) * 2 + 1] * w, acc += po[(base + p) * kHead + d] * w;
  }
  const uint64_t i = (uint64_t) th * kHead + d;
  out[i] = acc / sum * sigmoid(gate[i]);
}

}  // namespace

extern "C" uint64_t q35_attn_decode_scratch(uint32_t heads, uint32_t max_tokens) {
  return (uint64_t) max_tokens * heads * Q35_ATTN_PARTS * (kHead + 2);
}

static bool run(const float* q, const void* kc, const void* vc, uint32_t n_tok, uint32_t heads, uint32_t kv_heads,
                uint32_t pos0, const uint32_t* list, const uint32_t* list_len, uint32_t list_max, const float* gate,
                float* scratch, float* out, void* stream) {
  if (!q || !kc || !vc || !gate || !scratch || !out || !n_tok || n_tok > Q35_ATTN_DECODE_MAX || !kv_heads ||
      heads % kv_heads || heads / kv_heads > kGMax || heads / kv_heads == 0 ||
      (list && (n_tok != 1 || !list_len || !list_max)))
    return false;
  const uint32_t len = list ? list_max : pos0 + n_tok;  // the longest list: the last token's, or a read's bound
  // keys a warp: at least 16, the parts at most Q35_ATTN_PARTS a (token, head)
  uint32_t per = (len + Q35_ATTN_PARTS - 1) / Q35_ATTN_PARTS;
  per = per < 16 ? 16 : per;
  const uint32_t parts = (len + per - 1) / per, blocks = (parts + kWarps - 1) / kWarps;
  float* ml = scratch;
  float* po = scratch + (uint64_t) n_tok * heads * parts * 2;
  const dim3 grid(kv_heads, blocks, n_tok);
  const hipStream_t s = launches::st(stream);
  const uint8_t *k8 = (const uint8_t*) kc, *v8 = (const uint8_t*) vc;
#define Q35_DECODE(G) decode_kernel<G><<<grid, 32 * kWarps, 0, s>>>(q, k8, v8, heads, kv_heads, pos0, per, parts, list, list_len, ml, po)
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
  merge_kernel<<<n_tok * heads, kHead, 0, s>>>(ml, po, gate, heads, parts, out);
  return launches::ok();
}

static bool qwen(uint32_t dim, const AttnShape* shape) {  // the one shape run today: LFM3's is the kernel's next
  return dim == kHead && (!shape || (shape->gate_span == 1 && !shape->window));
}

extern "C" bool q35_attn_decode(const float* q, const void* kc, const void* vc, uint32_t n_tok, uint32_t heads,
                                uint32_t kv_heads, uint32_t dim, const AttnShape* shape, uint32_t pos0,
                                const float* gate, float* scratch, float* out, void* stream) {
  return qwen(dim, shape) && run(q, kc, vc, n_tok, heads, kv_heads, pos0, nullptr, nullptr, 0, gate, scratch, out, stream);
}

extern "C" bool q35_attn_read(const float* q, const void* kc, const void* vc, uint32_t heads, uint32_t kv_heads,
                              uint32_t dim, const AttnShape* shape, const uint32_t* list, const uint32_t* list_len,
                              uint32_t list_max, const float* gate, float* scratch, float* out, void* stream) {
  return qwen(dim, shape) && run(q, kc, vc, 1, heads, kv_heads, 0, list, list_len, list_max, gate, scratch, out, stream);
}
