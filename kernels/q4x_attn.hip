// Qwen3.8-Flash-Next's attention (include/osmos/gpu/q4x_ops.h): q's and k's norm and rotation, and the gated
// attention over the cache - dense, its keys' scores in a block's LDS
#include "block.h"

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

// a block a (head, token): its keys' scores in LDS (a thread a key), their softmax, then a dim a thread
__global__ __launch_bounds__(kHead) void attn_kernel(const float* q, const float* kc, const float* vc, uint32_t heads,
                                                     uint32_t kv_heads, uint32_t pos0, const float* gate, float* out) {
  extern __shared__ float s[];
  __shared__ float qs[kHead], lds[kHead];
  const uint32_t h = blockIdx.x, t = blockIdx.y, g = h / (heads / kv_heads), n = pos0 + t + 1;
  const uint64_t at = ((uint64_t) t * heads + h) * kHead + threadIdx.x;
  qs[threadIdx.x] = q[at];
  __syncthreads();
  const float scale = 1.0f / sqrtf((float) kHead);
  float m = -INFINITY;
  for (uint32_t p = threadIdx.x; p < n; p += kHead) {
    const float* kp = kc + ((uint64_t) p * kv_heads + g) * kHead;
    float dot = 0;
    for (int i = 0; i < kHead; i++) dot += qs[i] * kp[i];
    s[p] = dot * scale, m = fmaxf(m, s[p]);
  }
  m = block_max<kHead>(m, lds);
  float sum = 0;
  for (uint32_t p = threadIdx.x; p < n; p += kHead) sum += (s[p] = expf(s[p] - m));
  sum = block_sum<kHead>(sum, lds);
  float o = 0;
  for (uint32_t p = 0; p < n; p++) o += s[p] * vc[((uint64_t) p * kv_heads + g) * kHead + threadIdx.x];
  out[at] = o / sum * sigmoid(gate[at]);
}

}  // namespace

extern "C" bool q4x_qk(const float* qfull, const float* k, uint32_t n_tok, uint32_t heads, uint32_t kv_heads,
                       uint32_t dim, uint32_t rope_dim, const float* q_norm, const float* k_norm, float eps,
                       int32_t pos0, float rope_base, float* q, float* gate, float* k_out, void* stream) {
  if (!qfull || !k || !q_norm || !k_norm || !q || !gate || !k_out || !n_tok || !heads || !kv_heads || dim != kHead ||
      !rope_dim || rope_dim > dim || rope_dim % 2 || pos0 < 0)
    return false;
  const float theta_scale = powf(rope_base, -2.0f / (float) rope_dim);
  qk_kernel<<<dim3(heads + kv_heads, n_tok), kHead, 0, st(stream)>>>(qfull, k, heads, kv_heads, rope_dim, q_norm,
                                                                     k_norm, eps, pos0, theta_scale, q, gate, k_out);
  return ok();
}

extern "C" bool q4x_attn(const float* q, const float* kc, const float* vc, uint32_t n_tok, uint32_t heads,
                         uint32_t kv_heads, uint32_t dim, uint32_t pos0, const float* gate, float* out, void* stream) {
  if (!n_tok || !heads || !kv_heads || heads % kv_heads || dim != kHead || pos0 + n_tok > OPS_Q4X_MAX_KEYS)
    return false;
  if (!q || !kc || !vc || !gate || !out) return false;
  const size_t lds = (size_t) (pos0 + n_tok) * sizeof(float);
  attn_kernel<<<dim3(heads, n_tok), kHead, lds, st(stream)>>>(q, kc, vc, heads, kv_heads, pos0, gate, out);
  return ok();
}
