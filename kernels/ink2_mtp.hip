// Inkling-2's drafter (DSpark, the RNN Markov head; src/ink2/hip_mtp.c, tools/convert/ink2_mtp_oracle.py): its taps'
// mixtures, its context caches (f16), its attention over a context and a block, its Markov steps. F32 math.
#include <hip/hip_runtime.h>

#include <cmath>

extern "C" {
#include "osmos/gpu/ink2_ops.h"
}

namespace {

constexpr uint32_t kThreads = 256;
inline uint32_t grid(uint64_t n) { return (uint32_t) ((n + kThreads - 1) / kThreads); }
inline hipStream_t st(void* s) { return (hipStream_t) s; }
inline bool ok() { return hipGetLastError() == hipSuccess; }

__global__ void mix_kernel(const float* taps, uint32_t n_taps, const float* alpha, uint64_t n, uint32_t dim, float* out) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= n * dim) return;
  const uint64_t t = i / dim, c = i % dim;
  float v = 0.0f;
  for (uint32_t k = 0; k < n_taps; k++) v += alpha[k] * taps[(t * n_taps + k) * dim + c];
  out[i] = v;
}

__global__ void kv_put_kernel(_Float16* ring, uint32_t ring_rows, const float* x, uint32_t n, uint32_t row, int32_t pos0) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n * row) return;
  const uint32_t t = (uint32_t) (i / row), c = (uint32_t) (i % row);
  ring[(uint64_t) ((uint32_t) (pos0 + (int32_t) t) % ring_rows) * row + c] = (_Float16) x[i];
}

// a (query, head, chunk of keys): its keys' scores, their max, sum and weighted values - the chunk's partial
constexpr uint32_t kChunk = 256, kHead = 128;
__global__ void attn_part_kernel(const float* q, uint32_t n_q, int32_t pos0, const float* kb, const float* vb,
                                 const _Float16* kr, const _Float16* vr, uint32_t ring_rows, int32_t ctx_from,
                                 uint32_t heads, uint32_t kv_heads, uint32_t window, float* part, uint32_t chunks) {
  __shared__ float qv[kHead], sc[kChunk], red[kThreads];
  const uint32_t qi = blockIdx.x / heads, h = blockIdx.x % heads, g = h / (heads / kv_heads), row = kv_heads * kHead;
  const int32_t qp = pos0 + (int32_t) qi;
  int32_t lo = ctx_from;
  if (window && qp - (int32_t) window + 1 > lo) lo = qp - (int32_t) window + 1;
  const uint32_t nctx = pos0 > lo ? (uint32_t) (pos0 - lo) : 0, nk = nctx + n_q;
  const uint32_t j0 = blockIdx.y * kChunk, j = j0 + threadIdx.x;
  for (uint32_t d = threadIdx.x; d < kHead; d += kThreads) qv[d] = q[((uint64_t) qi * heads + h) * kHead + d];
  __syncthreads();
  float s = -INFINITY;
  if (j < nk) {
    float acc = 0.0f;
    if (j < nctx) {
      const _Float16* k = kr + (uint64_t) ((uint32_t) (lo + (int32_t) j) % ring_rows) * row + g * kHead;
      for (uint32_t d = 0; d < kHead; d++) acc += qv[d] * (float) k[d];
    } else {
      const float* k = kb + (uint64_t) (j - nctx) * row + g * kHead;
      for (uint32_t d = 0; d < kHead; d++) acc += qv[d] * k[d];
    }
    s = acc / sqrtf((float) kHead);
  }
  sc[threadIdx.x] = s;
  red[threadIdx.x] = s;
  __syncthreads();
  for (uint32_t o = kThreads / 2; o; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
    __syncthreads();
  }
  const float mx = red[0];
  __syncthreads();
  const float p = s == -INFINITY ? 0.0f : expf(s - mx);
  sc[threadIdx.x] = p;
  red[threadIdx.x] = p;
  __syncthreads();
  for (uint32_t o = kThreads / 2; o; o >>= 1) {
    if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
    __syncthreads();
  }
  float* pp = part + ((uint64_t) blockIdx.x * chunks + blockIdx.y) * (kHead + 2);
  if (threadIdx.x == 0) pp[0] = mx, pp[1] = red[0];
  for (uint32_t d = threadIdx.x; d < kHead; d += kThreads) {
    float o = 0.0f;
    for (uint32_t t = 0; t < kChunk && j0 + t < nk; t++) {
      const uint32_t jj = j0 + t;
      const float v = jj < nctx ? (float) vr[(uint64_t) ((uint32_t) (lo + (int32_t) jj) % ring_rows) * row + g * kHead + d]
                                : vb[(uint64_t) (jj - nctx) * row + g * kHead + d];
      o += sc[t] * v;
    }
    pp[2 + d] = o;
  }
}

// a (query, head)'s chunks merged in order: out = sum o_c e^(m_c - M) / sum l_c e^(m_c - M)
__global__ void attn_merge_kernel(const float* part, uint32_t chunks, float* out) {
  const uint32_t d = threadIdx.x;
  const float* pp = part + (uint64_t) blockIdx.x * chunks * (kHead + 2);
  float M = -INFINITY;
  for (uint32_t c = 0; c < chunks; c++) M = fmaxf(M, pp[c * (kHead + 2)]);
  float L = 0.0f, O = 0.0f;
  for (uint32_t c = 0; c < chunks; c++) {
    const float m = pp[c * (kHead + 2)];
    if (m == -INFINITY) continue;
    const float w = expf(m - M);
    L += pp[c * (kHead + 2) + 1] * w, O += pp[c * (kHead + 2) + 2 + d] * w;
  }
  out[(uint64_t) blockIdx.x * kHead + d] = O / L;
}

// [state | markov_w1[prev]]: the Markov step's own input (prev on the device: the last step's pick, or the anchor)
__global__ void markov_in_kernel(const float* state, const uint16_t* w1, const int32_t* prev, uint32_t rank, float* vec) {
  const uint32_t i = threadIdx.x + blockIdx.x * kThreads;
  if (i >= 2 * rank) return;
  if (i < rank) {
    vec[i] = state[i];
  } else {
    const uint32_t b = (uint32_t) w1[(uint64_t) (*prev) * rank + (i - rank)] << 16;
    float f;
    __builtin_memcpy(&f, &b, 4);
    vec[i] = f;
  }
}

// the GRU-like step: g, c, o from the step's two projections and the bias; state <- g state + (1 - g) tanh(c)
__global__ void markov_gru_kernel(const float* p, const float* inv, const float* bias, uint32_t rank, float* state,
                                  float* latent) {
  const uint32_t i = threadIdx.x + blockIdx.x * kThreads;
  if (i >= rank) return;
  const float g = 1.0f / (1.0f + expf(-(p[i] + inv[i] + bias[i])));
  const float c = p[rank + i] + inv[rank + i] + bias[rank + i], o = p[2 * rank + i] + inv[2 * rank + i] + bias[2 * rank + i];
  state[i] = g * state[i] + (1.0f - g) * tanhf(c);
  latent[i] = tanhf(o);
}

__global__ void markov_logits_kernel(const float* base, const float* corr, const float* fold_b, float cap, uint32_t n,
                                     float* out) {
  const uint32_t i = threadIdx.x + blockIdx.x * kThreads;
  if (i >= n) return;
  const float x = base[i] + corr[i] + fold_b[i];
  out[i] = cap * tanhf(x / cap);
}

__global__ void transpose_bf16_kernel(const float* in, uint32_t rows, uint32_t cols, uint16_t* out) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) rows * cols) return;
  const uint32_t r = (uint32_t) (i / cols), c = (uint32_t) (i % cols);
  uint32_t u;
  __builtin_memcpy(&u, &in[i], 4);
  const uint32_t rounded = u + 0x7FFFu + ((u >> 16) & 1u);  // nearest, ties to even
  out[(uint64_t) c * rows + r] = (uint16_t) (rounded >> 16);
}

}  // namespace

extern "C" bool ink2_mtp_mix(const float* taps, uint32_t n_taps, const float* alpha, uint32_t n, uint32_t dim, float* out,
                             void* stream) {
  if (!taps || !alpha || !out || !n || !dim || !n_taps) return false;
  mix_kernel<<<grid((uint64_t) n * dim), kThreads, 0, st(stream)>>>(taps, n_taps, alpha, n, dim, out);
  return ok();
}

extern "C" bool ink2_mtp_kv_put(uint16_t* ring, uint32_t ring_rows, const float* x, uint32_t n, uint32_t row, int32_t pos0,
                                void* stream) {
  if (!ring || !x || !n || !row || !ring_rows) return false;
  kv_put_kernel<<<grid((uint64_t) n * row), kThreads, 0, st(stream)>>>((_Float16*) ring, ring_rows, x, n, row, pos0);
  return ok();
}

extern "C" uint64_t ink2_mtp_attn_scratch(uint32_t n_q, uint32_t heads, uint32_t max_keys) {
  return (uint64_t) n_q * heads * ((max_keys + n_q + kChunk - 1) / kChunk) * (kHead + 2) * sizeof(float);
}

extern "C" bool ink2_mtp_attn(const float* q, uint32_t n_q, int32_t pos0, const float* kb, const float* vb,
                              const uint16_t* kr, const uint16_t* vr, uint32_t ring_rows, int32_t ctx_from,
                              uint32_t heads, uint32_t kv_heads, uint32_t window, float* scratch, float* out,
                              void* stream) {
  if (!q || !kb || !vb || !kr || !vr || !scratch || !out || !n_q || !heads || !kv_heads || heads % kv_heads ||
      ctx_from < 0 || ctx_from > pos0 || (uint64_t) (pos0 - ctx_from) > ring_rows)
    return false;
  const uint32_t nk_most = (uint32_t) (pos0 - ctx_from) + n_q, chunks = (nk_most + kChunk - 1) / kChunk;
  attn_part_kernel<<<dim3(n_q * heads, chunks), kThreads, 0, st(stream)>>>(
      q, n_q, pos0, kb, vb, (const _Float16*) kr, (const _Float16*) vr, ring_rows, ctx_from, heads, kv_heads, window,
      scratch, chunks);
  attn_merge_kernel<<<n_q * heads, kHead, 0, st(stream)>>>(scratch, chunks, out);
  return ok();
}

extern "C" bool ink2_mtp_markov_in(const float* state, const uint16_t* w1, const int32_t* prev, uint32_t rank, float* vec,
                                   void* stream) {
  markov_in_kernel<<<grid(2 * rank), kThreads, 0, st(stream)>>>(state, w1, prev, rank, vec);
  return ok();
}

extern "C" bool ink2_mtp_markov_gru(const float* p, const float* inv, const float* bias, uint32_t rank, float* state,
                                    float* latent, void* stream) {
  markov_gru_kernel<<<grid(rank), kThreads, 0, st(stream)>>>(p, inv, bias, rank, state, latent);
  return ok();
}

extern "C" bool ink2_mtp_markov_logits(const float* base, const float* corr, const float* fold_b, float cap, uint32_t n,
                                       float* out, void* stream) {
  markov_logits_kernel<<<grid(n), kThreads, 0, st(stream)>>>(base, corr, fold_b, cap, n, out);
  return ok();
}

extern "C" bool ink2_transpose_bf16(const float* in, uint32_t rows, uint32_t cols, uint16_t* out, void* stream) {
  transpose_bf16_kernel<<<grid((uint64_t) rows * cols), kThreads, 0, st(stream)>>>(in, rows, cols, out);
  return ok();
}
