// Qwen3.8-Flash-Next's Gated DeltaNet (include/osmos/gpu/q4x_ops.h): its conv, gates, delta rule and gated norm, F32
#include "block.h"
#include "x16side.h"

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

// a block a (q or k head, token): its l2 norm (block_sum's tree), then x r / sqrt(dim) into qk - the expressions the
// delta rule used inline, in their order: the same bits, out of its serial loop
__global__ __launch_bounds__(kHead) void qk_norm_kernel(const float* qkv, uint32_t k_heads, uint32_t v_heads, float eps,
                                                        float* qk) {
  __shared__ float lds[kHead];
  const uint32_t hh = blockIdx.x, t = blockIdx.y, j = threadIdx.x;  // hh: q heads 0 .. k_heads - 1, then k heads
  const uint64_t ch = 2ull * k_heads * kHead + (uint64_t) v_heads * kHead;
  const float x = qkv[t * ch + (uint64_t) hh * kHead + j];
  const float r = 1.0f / sqrtf(block_sum<kHead>(x * x, lds) / (float) kHead + eps / (float) kHead);
  qk[((uint64_t) t * 2 * k_heads + hh) * kHead + j] = x * r * (1.0f / sqrtf((float) kHead));
}

// value head blockIdx.x, its state's column j two threads, each half its rows (state[h][j][i] = S[i][j], in
// registers: 64 a thread - a whole column of 128 spilled), a token at a time: each half's sum, then low + high, the
// same in both; its key group's normed q and k (qk_norm_kernel's) through LDS, two buffers - the next token's loads
// issued before this one's sums, one barrier a token; S read from ring slot cur, after each of the last `keep` tokens
// into its slot
constexpr int kHalfRows = kHead / 2;
__global__ __launch_bounds__(2 * kHead) void delta_kernel(const float* qkv, const float* qk, const float* g,
                                                          const float* b, uint32_t n_tok, uint32_t k_heads,
                                                          uint32_t v_heads, float* ring, uint32_t slots, uint32_t cur,
                                                          uint32_t keep, float* out) {
  __shared__ float qs[2][kHead], ks[2][kHead];
  const uint32_t h = blockIdx.x, j = threadIdx.x >> 1, half = threadIdx.x & 1, kg = h % k_heads;
  const uint32_t r0 = half * kHalfRows;
  const uint64_t ch = 2ull * k_heads * kHead + (uint64_t) v_heads * kHead, qkr = 2ull * k_heads * kHead;
  const uint64_t slot = (uint64_t) v_heads * kHead * kHead, col = ((uint64_t) h * kHead + j) * kHead + r0;
  float s[kHalfRows];
#pragma unroll
  for (int i = 0; i < kHalfRows; i++) s[i] = ring[cur * slot + col + i];
  const float scale = 1.0f / sqrtf((float) kHead);
  // a thread's share of the token's loads: its column's q (the low half's thread) or k (the high's); v both
  auto load = [&](uint32_t t, float* x, float* v, float* gg, float* bb) {
    *x = qk[t * qkr + (uint64_t) (half ? k_heads + kg : kg) * kHead + j];
    *v = qkv[t * ch + 2ull * k_heads * kHead + (uint64_t) h * kHead + j];
    *gg = g[(uint64_t) t * v_heads + h], *bb = b[(uint64_t) t * v_heads + h];
  };
  float nx, nv, ng, nb;
  load(0, &nx, &nv, &ng, &nb);
  for (uint32_t t = 0; t < n_tok; t++) {
    const int buf = t & 1;
    (half ? ks : qs)[buf][j] = nx;
    const float v = nv, decay = expf(ng), beta = nb;
    if (t + 1 < n_tok) load(t + 1, &nx, &nv, &ng, &nb);
    __syncthreads();  // this token's q, k in; the buffer before it read by every thread (it was, before the last)
    const float *qb = qs[buf] + r0, *kb = ks[buf] + r0;
    float part = 0;
#pragma unroll
    for (int i = 0; i < kHalfRows; i++) s[i] *= decay, part += s[i] * kb[i];
    const float other = __shfl_xor(part, 1);
    const float dot = half ? other + part : part + other;  // low + high in both threads
    const float delta = (v - dot) * beta;
    float po = 0;
#pragma unroll
    for (int i = 0; i < kHalfRows; i++) s[i] += kb[i] * delta, po += s[i] * qb[i];
    const float oo = __shfl_xor(po, 1);
    if (!half) out[((uint64_t) t * v_heads + h) * kHead + j] = (po + oo) * scale;
    if (t >= n_tok - keep) {
      float* to = ring + (cur + 1 + t) % slots * slot + col;
#pragma unroll
      for (int i = 0; i < kHalfRows; i++) to[i] = s[i];
    }
  }
}

// a block a (head, token): x normed by its own sum of squares, weighted, gated
__global__ __launch_bounds__(kHead) void gated_norm_kernel(const float* x, const float* w, const float* z,
                                                           float eps, float* y, Q4xX16 side) {
  __shared__ float lds[kHead];
  const uint64_t at = ((uint64_t) blockIdx.y * gridDim.x + blockIdx.x) * kHead + threadIdx.x;
  const float v = x[at];
  const float r = 1.0f / sqrtf(block_sum<kHead>(v * v, lds) / (float) kHead + eps);
  y[at] = v * r * w[threadIdx.x] * sigmoid(z[at]);
  x16_side(side, at, y[at]);
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
                          uint32_t v_heads, uint32_t dim, float eps, float* qk, float* ring, uint32_t slots, uint32_t cur,
                          float* out, void* stream) {
  if (!qkv || !g || !b || !qk || !ring || !out || !n_tok || !k_heads || !v_heads || v_heads % k_heads ||
      dim != kHead || !slots || cur >= slots)
    return false;
  qk_norm_kernel<<<dim3(2 * k_heads, n_tok), kHead, 0, st(stream)>>>(qkv, k_heads, v_heads, eps, qk);
  if (!ok()) return false;
  delta_kernel<<<v_heads, 2 * kHead, 0, st(stream)>>>(qkv, qk, g, b, n_tok, k_heads, v_heads, ring, slots, cur,
                                                  kept(n_tok, slots), out);
  return ok();
}

extern "C" bool q4x_gated_norm(const float* x, const float* w, const float* z, uint32_t n_tok, uint32_t heads,
                               uint32_t dim, float eps, float* y, const Q4xX16* side, void* stream) {
  if (!x || !w || !z || !y || !n_tok || !heads || dim != kHead) return false;
  gated_norm_kernel<<<dim3(heads, n_tok), kHead, 0, st(stream)>>>(x, w, z, eps, y, side_of(side));
  return ok();
}
