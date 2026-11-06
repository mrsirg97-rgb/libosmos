// Qwen3.8-Flash-Next's Gated DeltaNet (include/osmos/gpu/q4x_ops.h): its conv, gates, delta rule and gated norm, F32
#include "block.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

namespace {

constexpr int kThreads = 256;
constexpr int kHead = 128;                // a head's dim: the delta rule's block, a column of its state a thread
constexpr uint32_t kMaxTaps = 8; /* a conv's taps, and its dilation, at most */
constexpr uint32_t kMaxHist = (kMaxTaps - 1) * kMaxTaps; /* a conv state's rows at most */
constexpr float kSoftplusLinear = 20.0f;  // ggml's softplus: above it, x

inline unsigned grid(uint64_t n) { return (unsigned) ((n + kThreads - 1) / kThreads); }
inline hipStream_t st(void* s) { return (hipStream_t) s; }
inline bool ok() { return hipGetLastError() == hipSuccess; }

__device__ inline float silu(float x) { return x / (1.0f + expf(-x)); }

// x' = hist (h = (taps - 1) dil rows) then x: token t's taps x'[t + k dil], the oldest first
__global__ void conv_kernel(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, uint32_t dil, const float* w,
                            const float* hist, float* y) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * ch) return;
  const uint32_t c = (uint32_t) (i % ch), t = (uint32_t) (i / ch), h = (taps - 1) * dil;
  float sum = 0;
  for (uint32_t k = 0; k < taps; k++) {
    const uint32_t at = t + k * dil;  // in x'
    const float v = at < h ? hist[(uint64_t) at * ch + c] : x[(uint64_t) (at - h) * ch + c];
    sum += v * w[(uint64_t) c * taps + k];
  }
  y[i] = silu(sum);
}

// after each of the step's last `keep` tokens, the last h of x' up to it into its ring slot, a channel a thread: the
// slot read (cur) first, whole, into registers - a later token's slot may be it
__global__ void conv_hist_kernel(const float* x, uint32_t n_tok, uint32_t ch, uint32_t h, float* ring, uint32_t slots,
                                 uint32_t cur, uint32_t keep) {
  const uint32_t c = blockIdx.x * kThreads + threadIdx.x;
  if (c >= ch) return;
  const uint64_t slot = (uint64_t) h * ch;
  float old[kMaxHist];
  for (uint32_t k = 0; k < h; k++) old[k] = ring[cur * slot + (uint64_t) k * ch + c];
  for (uint32_t t = n_tok - keep; t < n_tok; t++)
    for (uint32_t k = 0; k < h; k++) {
      const uint32_t at = t + 1 + k;  // in x', the token's last h
      ring[(cur + 1 + t) % slots * slot + (uint64_t) k * ch + c] = at < h ? old[at] : x[(uint64_t) (at - h) * ch + c];
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
// q and k of its key group normed in LDS; S read from ring slot cur, after each of the last `keep` tokens into its slot
__global__ __launch_bounds__(kHead) void delta_kernel(const float* qkv, const float* g, const float* b,
                                                      uint32_t n_tok, uint32_t k_heads, uint32_t v_heads, float eps,
                                                      float* ring, uint32_t slots, uint32_t cur, uint32_t keep,
                                                      float* out) {
  __shared__ float qs[kHead], ks[kHead], lds[kHead];
  const uint32_t h = blockIdx.x, j = threadIdx.x, kg = h % k_heads;
  const uint64_t ch = 2ull * k_heads * kHead + (uint64_t) v_heads * kHead;
  const uint64_t slot = (uint64_t) v_heads * kHead * kHead, col = ((uint64_t) h * kHead + j) * kHead;
  float s[kHead];
#pragma unroll
  for (int i = 0; i < kHead; i++) s[i] = ring[cur * slot + col + i];
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
    if (t >= n_tok - keep) {
      float* to = ring + (cur + 1 + t) % slots * slot + col;
#pragma unroll
      for (int i = 0; i < kHead; i++) to[i] = s[i];
    }
    __syncthreads();  // qs, ks free for the next token
  }
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

// the step's tokens whose states a ring keeps: its last slots - 1 (one, in place, when slots is 1)
inline uint32_t kept(uint32_t n_tok, uint32_t slots) {
  const uint32_t back = slots > 1 ? slots - 1 : 1;
  return n_tok < back ? n_tok : back;
}

extern "C" bool q4x_conv(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, uint32_t dilation,
                         const float* w, float* ring, uint32_t slots, uint32_t cur, float* y, void* stream) {
  if (!x || !w || !ring || !y || !n_tok || !ch || taps < 2 || taps > kMaxTaps || !dilation || dilation > kMaxTaps ||
      !slots || cur >= slots)
    return false;
  const uint32_t h = (taps - 1) * dilation;
  conv_kernel<<<grid((uint64_t) n_tok * ch), kThreads, 0, st(stream)>>>(x, n_tok, ch, taps, dilation, w,
                                                                        ring + (uint64_t) cur * h * ch, y);
  if (!ok()) return false;
  conv_hist_kernel<<<grid(ch), kThreads, 0, st(stream)>>>(x, n_tok, ch, h, ring, slots, cur, kept(n_tok, slots));
  return ok();
}

extern "C" bool q4x_ssm_gates(const float* alpha, const float* beta, const float* dt, const float* a, uint32_t n_tok,
                              uint32_t heads, float* g, float* b, void* stream) {
  if (!alpha || !beta || !dt || !a || !g || !b || !n_tok || !heads) return false;
  gates_kernel<<<grid((uint64_t) n_tok * heads), kThreads, 0, st(stream)>>>(alpha, beta, dt, a, n_tok, heads, g, b);
  return ok();
}

extern "C" bool q4x_delta(const float* qkv, const float* g, const float* b, uint32_t n_tok, uint32_t k_heads,
                          uint32_t v_heads, uint32_t dim, float eps, float* ring, uint32_t slots, uint32_t cur,
                          float* out, void* stream) {
  if (!qkv || !g || !b || !ring || !out || !n_tok || !k_heads || !v_heads || v_heads % k_heads || dim != kHead ||
      !slots || cur >= slots)
    return false;
  delta_kernel<<<v_heads, kHead, 0, st(stream)>>>(qkv, g, b, n_tok, k_heads, v_heads, eps, ring, slots, cur,
                                                  kept(n_tok, slots), out);
  return ok();
}

extern "C" bool q4x_gated_norm(const float* x, const float* w, const float* z, uint32_t n_tok, uint32_t heads,
                               uint32_t dim, float eps, float* y, void* stream) {
  if (!x || !w || !z || !y || !n_tok || !heads || dim != kHead) return false;
  gated_norm_kernel<<<dim3(heads, n_tok), kHead, 0, st(stream)>>>(x, w, z, eps, y);
  return ok();
}
