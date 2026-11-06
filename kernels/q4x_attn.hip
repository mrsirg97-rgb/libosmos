// Qwen3.8-Flash-Next's attention (include/osmos/gpu/q4x_ops.h): q's and k's norm and rotation, and the gated
// attention over the cache - dense, its keys' scores in a block's LDS
#include "block.h"
#include "x16side.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

namespace {

constexpr int kHead = 256;  // a head's dim: a block's threads, a value a thread

inline hipStream_t st(void* s) { return (hipStream_t) s; }
inline bool ok() { return hipGetLastError() == hipSuccess; }

// a block a (head, token): blockIdx.x < heads a q head (its gate copied out), past them a KV head. the norm, then
// the first rope_dim rotated in pairs (i, i + rope_dim / 2), theta_i a running product as ggml's rope cache builds it
__global__ __launch_bounds__(kHead) void qk_kernel(const float* qfull, const float* k, uint32_t heads,
                                                   uint32_t kv_heads, uint32_t rope_dim, const float* q_norm,
                                                   const float* k_norm, float eps, int32_t pos0, float theta_scale,
                                                   float* q, float* gate, float* k_out) {
  __shared__ float lds[kHead], y[kHead];
  const uint32_t h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
  const bool is_q = h < heads;
  const uint32_t g = is_q ? h : h - heads;
  float x;
  if (is_q) {
    const float* row = qfull + (uint64_t) t * 2 * heads * kHead + 2ull * h * kHead;
    x = row[d];
    gate[((uint64_t) t * heads + h) * kHead + d] = row[kHead + d];
  } else {
    x = k[((uint64_t) t * kv_heads + g) * kHead + d];
  }
  const float r = 1.0f / sqrtf(block_sum<kHead>(x * x, lds) / (float) kHead + eps);
  y[d] = x * r * (is_q ? q_norm : k_norm)[d];
  __syncthreads();
  float v = y[d];
  const uint32_t half = rope_dim / 2;
  if (d < rope_dim) {
    const uint32_t i = d % half;
    float theta = (float) (pos0 + (int32_t) t);
    for (uint32_t j = 0; j < i; j++) theta *= theta_scale;
    const float c = cosf(theta), s = sinf(theta), x0 = y[i], x1 = y[i + half];
    v = d < half ? x0 * c - x1 * s : x0 * s + x1 * c;
  }
  if (is_q)
    q[((uint64_t) t * heads + h) * kHead + d] = v;
  else
    k_out[((uint64_t) t * kv_heads + g) * kHead + d] = v;
}

// the attention on the matrix cores (WMMA f32 <- f16, 16 x 16 x 16; mm.h's lane layout: an operand's lane l its row
// or column l % 16, a result's lane l rows 2 j + l / 16 of column l % 16). a block: one token's q heads of a KV head
// (G <= 16 rows) over one chunk of kChunk keys, a block of 16 keys at a time: S = Q K^T (f16 q and k), each row's
// running max and sum (fixed shuffle trees in its lane half), P = exp(S - max) to f16, O += P V (f16 v); its rows'
// (max, sum, O) into the chunk's scratch. a tile one token's: its keys past the token's position loaded as zeros in
// every batch - the WMMA's sum is not blind to a value its weight zeroes (measured: a masked key's v moved the last
// bit), so a row's keys must be the same whatever it is batched with. then a row's bits are its own: its q, its keys,
// its position; and whichever chunks its batch computed past its own, it merges only its own
constexpr int kTile = 16;
constexpr int kWaves = 4;                   // a wave a quarter of O's columns
constexpr int kChunk = 128;                 // a chunk's keys: absolute positions [c kChunk, (c + 1) kChunk)
constexpr int kStepTokens = 64;             // a launch's tokens (the scratch's): a long step in pieces
constexpr int kRowLds = kHead + 8;          // f16s a row in LDS, past it a pad (rows 4 banks apart)
constexpr int kKeyLds = kTile + 8;
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));

__device__ inline float row_max(float v) {  // the 16 lanes of a half: one row's 16 columns
  for (int o = 1; o < kTile; o <<= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  return v;
}
__device__ inline float row_sum(float v) {
  for (int o = 1; o < kTile; o <<= 1) v += __shfl_xor(v, o, 32);
  return v;
}

__global__ __launch_bounds__(kWaves * 32) void fa_kernel(const float* q, const float* kc, const float* vc,
                                                         uint32_t n_tok, uint32_t heads, uint32_t kv_heads,
                                                         uint32_t pos0, float* ml, float* po) {
  __shared__ __attribute__((aligned(32))) _Float16 qs[kTile * kRowLds], ks[kTile * kRowLds], vt[kHead * kKeyLds];
  __shared__ __attribute__((aligned(32))) _Float16 ps[kWaves][kTile * kKeyLds];
  const uint32_t G = heads / kv_heads, t = blockIdx.x, kvh = blockIdx.y, chunk = blockIdx.z;
  const uint32_t wave = threadIdx.x / 32, lane = threadIdx.x % 32, half = lane >> 4;
  const uint32_t top = pos0 + t;  // the token's last key
  const uint32_t k0 = chunk * kChunk;
  if (k0 > top) return;  // a chunk past the token: it reads no scratch of it
  const uint32_t k1 = (k0 + kChunk - 1 < top ? k0 + kChunk - 1 : top) + 1;
  for (uint32_t i = threadIdx.x; i < kTile * kHead; i += kWaves * 32) {  // Q, f16 (rows past G zero)
    const uint32_t r = i / kHead, d = i % kHead;
    qs[r * kRowLds + d] = (_Float16) (r < G ? q[((uint64_t) t * heads + kvh * G + r) * kHead + d] : 0.0f);
  }
  int32_t lim[8];  // this lane's rows (2 j + half): the token's last key, or none past G
  for (int j = 0; j < 8; j++) lim[j] = (uint32_t) (2 * j + half) < G ? (int32_t) top : -1;
  float m[8], l[8];
  f8 o[kHead / kTile / kWaves] = {};
  for (int j = 0; j < 8; j++) m[j] = -INFINITY, l[j] = 0.0f;
  for (uint32_t kb = k0; kb < k1; kb += kTile) {
    __syncthreads();  // the last block's K, V read by every wave
    for (uint32_t i = threadIdx.x; i < kTile * kHead; i += kWaves * 32) {  // K and V^T, f16 (keys past k1 zero)
      const uint32_t r = i / kHead, d = i % kHead, p = kb + r;
      const uint64_t at = ((uint64_t) p * kv_heads + kvh) * kHead + d;
      ks[r * kRowLds + d] = (_Float16) (p < k1 ? kc[at] : 0.0f);
      vt[d * kKeyLds + r] = (_Float16) (p < k1 ? vc[at] : 0.0f);
    }
    __syncthreads();
    f8 sacc = {};
    for (int g = 0; g < kHead / kTile; g++) {
      const h16 a = *(const h16*) (qs + (lane % kTile) * kRowLds + g * kTile);
      const h16 bk = *(const h16*) (ks + (lane % kTile) * kRowLds + g * kTile);
      sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bk, sacc);
    }
    // each row: its scores scaled and masked, the running max and sum, P to f16
    const int32_t key = (int32_t) (kb + lane % kTile);
    float alpha[8];
    for (int j = 0; j < 8; j++) {
      const float sv = key <= lim[j] ? sacc[j] * (1.0f / 16.0f) : -INFINITY;  // 1 / sqrt(256)
      const float mn = fmaxf(m[j], row_max(sv));
      const float pv = mn == -INFINITY ? 0.0f : expf(sv - mn);
      alpha[j] = mn == -INFINITY ? 1.0f : expf(m[j] - mn);
      l[j] = l[j] * alpha[j] + row_sum(pv), m[j] = mn;
      ps[wave][(2 * j + half) * kKeyLds + lane % kTile] = (_Float16) pv;
    }
    __builtin_amdgcn_wave_barrier();
    const h16 pa = *(const h16*) (ps[wave] + (lane % kTile) * kKeyLds);
    for (int nt = 0; nt < kHead / kTile / kWaves; nt++) {
      for (int j = 0; j < 8; j++) o[nt][j] *= alpha[j];
      const uint32_t col = (wave * (kHead / kTile / kWaves) + nt) * kTile + lane % kTile;
      const h16 bv = *(const h16*) (vt + col * kKeyLds);
      o[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, bv, o[nt]);
    }
    __builtin_amdgcn_wave_barrier();
  }
  // the chunk's partials: (max, sum) a row, O a row's dims - row R = t G + r of KV head kvh, at [chunk][kvh][R]
  const uint64_t base = ((uint64_t) chunk * kv_heads + kvh) * n_tok * G;
  for (int j = 0; j < 8; j++) {
    const uint32_t r = 2 * j + half, R = t * G + r;
    if (r >= G) continue;
    if (wave == 0 && lane % kTile == 0) ml[(base + R) * 2] = m[j], ml[(base + R) * 2 + 1] = l[j];
    for (int nt = 0; nt < kHead / kTile / kWaves; nt++)
      po[(base + R) * kHead + (wave * (kHead / kTile / kWaves) + nt) * kTile + lane % kTile] = o[nt][j];
  }
}

// a block a (token, q head), a thread a dim: its row's chunks merged in order, the gate - out[t][h][d]. the chunks a
// row reads: those up to its own position, whatever its batch computed
__global__ __launch_bounds__(kHead) void fa_merge_kernel(const float* ml, const float* po, const float* gate,
                                                         uint32_t n_tok, uint32_t heads, uint32_t kv_heads,
                                                         uint32_t pos0, float* out, Q4xX16 side) {
  const uint32_t t = blockIdx.x / heads, h = blockIdx.x % heads, d = threadIdx.x, G = heads / kv_heads;
  const uint32_t rows = n_tok * G, R = t * G + h % G, kvh = h / G, last = (pos0 + t) / kChunk;
  float mx = -INFINITY;
  for (uint32_t c = 0; c <= last; c++) mx = fmaxf(mx, ml[(((uint64_t) c * kv_heads + kvh) * rows + R) * 2]);
  float sum = 0.0f, acc = 0.0f;
  for (uint32_t c = 0; c <= last; c++) {
    const uint64_t at = ((uint64_t) c * kv_heads + kvh) * rows + R;
    const float w = expf(ml[at * 2] - mx);
    sum += ml[at * 2 + 1] * w, acc += po[at * kHead + d] * w;
  }
  const uint64_t i = ((uint64_t) t * heads + h) * kHead + d;
  out[i] = acc / sum * sigmoid(gate[i]);
  x16_side(side, i, out[i]);
}

}  // namespace

extern "C" bool q4x_qk(const float* qfull, const float* k, uint32_t n_tok, uint32_t heads, uint32_t kv_heads,
                       uint32_t dim, uint32_t rope_dim, const float* q_norm, const float* k_norm, float eps,
                       int32_t pos0, float rope_base, float* q, float* gate, float* k_out, void* stream) {
  if ((heads && (!qfull || !q || !gate)) || !k || !q_norm || !k_norm || !k_out || !n_tok || !kv_heads || dim != kHead ||
      !rope_dim || rope_dim > dim || rope_dim % 2 || pos0 < 0)
    return false;
  const float theta_scale = powf(rope_base, -2.0f / (float) rope_dim);
  qk_kernel<<<dim3(heads + kv_heads, n_tok), kHead, 0, st(stream)>>>(qfull, k, heads, kv_heads, rope_dim, q_norm,
                                                                     k_norm, eps, pos0, theta_scale, q, gate, k_out);
  return ok();
}

extern "C" uint64_t q4x_attn_scratch(uint32_t heads, uint32_t max_ctx) {
  return (uint64_t) kStepTokens * heads * ((max_ctx + kChunk - 1) / kChunk) * (kHead + 2);
}

extern "C" bool q4x_attn(const float* q, const float* kc, const float* vc, uint32_t n_tok, uint32_t heads,
                         uint32_t kv_heads, uint32_t dim, uint32_t pos0, const float* gate, float* scratch,
                         float* out, const Q4xX16* side, void* stream) {
  if (!n_tok || !heads || !kv_heads || heads % kv_heads || heads / kv_heads > kTile || dim != kHead ||
      pos0 + n_tok > OPS_Q4X_MAX_KEYS)
    return false;
  if (!q || !kc || !vc || !gate || !scratch || !out) return false;
  for (uint32_t t0 = 0; t0 < n_tok; t0 += kStepTokens) {  // a piece of the step at a time: the scratch's tokens
    const uint32_t n = n_tok - t0 < (uint32_t) kStepTokens ? n_tok - t0 : (uint32_t) kStepTokens;
    const uint32_t rows = n * (heads / kv_heads), p0 = pos0 + t0, chunks = (p0 + n - 1) / kChunk + 1;
    float *ml = scratch, *po = scratch + (uint64_t) 2 * chunks * kv_heads * rows;
    const uint64_t at = (uint64_t) t0 * heads * kHead;
    fa_kernel<<<dim3(n, kv_heads, chunks), kWaves * 32, 0, st(stream)>>>(
        q + at, kc, vc, n, heads, kv_heads, p0, ml, po);
    if (!ok()) return false;
    Q4xX16 sd = side_of(side);  // this piece's rows: from its first token's
    if (sd.x16) sd.x16 = (_Float16*) sd.x16 + mm::x16_at(t0, 0, sd.cols);
    fa_merge_kernel<<<n * heads, kHead, 0, st(stream)>>>(ml, po, gate + at, n, heads, kv_heads, p0, out + at, sd);
    if (!ok()) return false;
  }
  return true;
}
