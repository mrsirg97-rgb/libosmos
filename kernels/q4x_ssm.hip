// Qwen3.8-Flash-Next's Gated DeltaNet (include/osmos/gpu/q4x_ops.h): its conv, gates, delta rule and gated norm, F32
#include "block.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

namespace {

constexpr int kThreads = 256;
constexpr int kHead = 128;                // a head's dim: the delta rule's block, a column of its state a thread
constexpr uint32_t kMaxTaps = 8;
constexpr float kSoftplusLinear = 20.0f;  // ggml's softplus: above it, x

inline unsigned grid(uint64_t n) { return (unsigned) ((n + kThreads - 1) / kThreads); }
inline hipStream_t st(void* s) { return (hipStream_t) s; }
inline bool ok() { return hipGetLastError() == hipSuccess; }

__device__ inline float silu(float x) { return x / (1.0f + expf(-x)); }

// x' = hist then x: token t's taps x'[t .. t + taps - 1], the oldest first
__global__ void conv_kernel(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, const float* w,
                            const float* hist, float* y) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * ch) return;
  const uint32_t c = (uint32_t) (i % ch), t = (uint32_t) (i / ch), h = taps - 1;
  float sum = 0;
  for (uint32_t k = 0; k < taps; k++) {
    const uint32_t at = t + k;  // in x'
    const float v = at < h ? hist[(uint64_t) at * ch + c] : x[(uint64_t) (at - h) * ch + c];
    sum += v * w[(uint64_t) c * taps + k];
  }
  y[i] = silu(sum);
}

// the last taps - 1 of x' into hist, a channel a thread, oldest first (a slot read before it is written)
__global__ void conv_hist_kernel(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, float* hist) {
  const uint32_t c = blockIdx.x * kThreads + threadIdx.x;
  if (c >= ch) return;
  const uint32_t h = taps - 1;
  for (uint32_t k = 0; k < h; k++) {
    const uint32_t at = n_tok + k;  // in x'
    hist[(uint64_t) k * ch + c] = at < h ? hist[(uint64_t) at * ch + c] : x[(uint64_t) (at - h) * ch + c];
  }
}

__global__ void gates_kernel(const float* alpha, const float* beta, const float* dt, const float* a, uint32_t n_tok,
                             uint32_t heads, float* g, float* b) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * heads) return;
  const uint32_t h = (uint32_t) (i % heads);
  const float z = alpha[i] + dt[h];
  g[i] = (z > kSoftplusLinear ? z : log1pf(expf(z))) * a[h];
  b[i] = sigmoid(beta[i]);
}

// value head blockIdx.x, its state's column j a thread (state[h][j][i] = S[i][j], in registers): a token at a time,
// q and k of its key group normed in LDS
__global__ __launch_bounds__(kHead) void delta_kernel(const float* qkv, const float* g, const float* b,
                                                      uint32_t n_tok, uint32_t k_heads, uint32_t v_heads, float eps,
                                                      float* state, float* out) {
  __shared__ float qs[kHead], ks[kHead], lds[kHead];
  const uint32_t h = blockIdx.x, j = threadIdx.x, kg = h % k_heads;
  const uint64_t ch = 2ull * k_heads * kHead + (uint64_t) v_heads * kHead;
  float* sj = state + ((uint64_t) h * kHead + j) * kHead;
  float s[kHead];
#pragma unroll
  for (int i = 0; i < kHead; i++) s[i] = sj[i];
  const float inv_n = 1.0f / sqrtf((float) kHead), scale = 1.0f / sqrtf((float) kHead);
  for (uint32_t t = 0; t < n_tok; t++) {
    const float* row = qkv + t * ch;
    const float q = row[(uint64_t) kg * kHead + j], k = row[(uint64_t) (k_heads + kg) * kHead + j];
    const float v = row[2ull * k_heads * kHead + (uint64_t) h * kHead + j];
    const float rq = 1.0f / sqrtf(block_sum<kHead>(q * q, lds) / (float) kHead + eps / (float) kHead);
    const float rk = 1.0f / sqrtf(block_sum<kHead>(k * k, lds) / (float) kHead + eps / (float) kHead);
    qs[j] = q * rq * inv_n, ks[j] = k * rk * inv_n;
    __syncthreads();
    const float decay = expf(g[(uint64_t) t * v_heads + h]), beta = b[(uint64_t) t * v_heads + h];
    float dot = 0;
#pragma unroll
    for (int i = 0; i < kHead; i++) s[i] *= decay, dot += s[i] * ks[i];
    const float delta = (v - dot) * beta;
    float o = 0;
#pragma unroll
    for (int i = 0; i < kHead; i++) s[i] += ks[i] * delta, o += s[i] * qs[i];
    out[((uint64_t) t * v_heads + h) * kHead + j] = o * scale;
    __syncthreads();  // qs, ks free for the next token
  }
#pragma unroll
  for (int i = 0; i < kHead; i++) sj[i] = s[i];
}

// a block a (head, token): x normed by its own sum of squares, weighted, gated
__global__ __launch_bounds__(kHead) void gated_norm_kernel(const float* x, const float* w, const float* z,
                                                           float eps, float* y) {
  __shared__ float lds[kHead];
  const uint64_t at = ((uint64_t) blockIdx.y * gridDim.x + blockIdx.x) * kHead + threadIdx.x;
  const float v = x[at];
  const float r = 1.0f / sqrtf(block_sum<kHead>(v * v, lds) / (float) kHead + eps);
  y[at] = v * r * w[threadIdx.x] * sigmoid(z[at]);
}

}  // namespace

extern "C" bool q4x_conv(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, const float* w, float* hist,
                         float* y, void* stream) {
  if (!x || !w || !hist || !y || !n_tok || !ch || taps < 2 || taps > kMaxTaps) return false;
  conv_kernel<<<grid((uint64_t) n_tok * ch), kThreads, 0, st(stream)>>>(x, n_tok, ch, taps, w, hist, y);
  if (!ok()) return false;
  conv_hist_kernel<<<grid(ch), kThreads, 0, st(stream)>>>(x, n_tok, ch, taps, hist);
  return ok();
}

extern "C" bool q4x_ssm_gates(const float* alpha, const float* beta, const float* dt, const float* a, uint32_t n_tok,
                              uint32_t heads, float* g, float* b, void* stream) {
  if (!alpha || !beta || !dt || !a || !g || !b || !n_tok || !heads) return false;
  gates_kernel<<<grid((uint64_t) n_tok * heads), kThreads, 0, st(stream)>>>(alpha, beta, dt, a, n_tok, heads, g, b);
  return ok();
}

extern "C" bool q4x_delta(const float* qkv, const float* g, const float* b, uint32_t n_tok, uint32_t k_heads,
                          uint32_t v_heads, uint32_t dim, float eps, float* state, float* out, void* stream) {
  if (!qkv || !g || !b || !state || !out || !n_tok || !k_heads || !v_heads || v_heads % k_heads || dim != kHead)
    return false;
  delta_kernel<<<v_heads, kHead, 0, st(stream)>>>(qkv, g, b, n_tok, k_heads, v_heads, eps, state, out);
  return ok();
}

extern "C" bool q4x_gated_norm(const float* x, const float* w, const float* z, uint32_t n_tok, uint32_t heads,
                               uint32_t dim, float eps, float* y, void* stream) {
  if (!x || !w || !z || !y || !n_tok || !heads || dim != kHead) return false;
  gated_norm_kernel<<<dim3(heads, n_tok), kHead, 0, st(stream)>>>(x, w, z, eps, y);
  return ok();
}
