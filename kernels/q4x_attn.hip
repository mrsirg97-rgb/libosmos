// Qwen3.8-Flash-Next's attention inputs (include/osmos/gpu/q4x_ops.h): q's and k's norm and rotation, the cache's
// Q8_0 rows - the attention itself the core's (attn_core_heads.h: q4x_attn and q4x_attn_lists there)
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

// a thread a block of a row: its scale, then its 32 values rounded to nearest (ggml's quantize_row_q8_0)
__global__ void kv_put_kernel(const float* x, uint32_t n_blocks, uint8_t* dst) {
  const uint32_t i = blockIdx.x * 256 + threadIdx.x;
  if (i >= n_blocks) return;
  uint8_t* b = dst + (uint64_t) i * Q4X_KV_BLOCK_BYTES;
  *(_Float16*) b = (_Float16) q8_0_block<Q4X_KV_BLOCK>(x + (uint64_t) i * Q4X_KV_BLOCK, 1, b + 2);
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

extern "C" bool q4x_kv_put(const float* k, const float* v, uint32_t n_tok, uint32_t values, void* kc, void* vc,
                           void* stream) {
  if (!k || !kc || (v != nullptr) != (vc != nullptr) || !n_tok || !values || values % Q4X_KV_BLOCK) return false;
  const uint32_t blocks = n_tok * values / Q4X_KV_BLOCK;
  kv_put_kernel<<<(blocks + 255) / 256, 256, 0, st(stream)>>>(k, blocks, (uint8_t*) kc);
  if (!ok()) return false;
  if (v) {
    kv_put_kernel<<<(blocks + 255) / 256, 256, 0, st(stream)>>>(v, blocks, (uint8_t*) vc);
    return ok();
  }
  return true;
}

