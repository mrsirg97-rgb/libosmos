// Inkling-2-Small's own steps (include/osmos/gpu/ink2_ops.h), F32
#include "block.h"
#include "launch.h"

extern "C" {
#include "osmos/gpu/ink2_ops.h"
}

namespace {

using launches::kThreads;
using launches::grid;
using launches::st;
using launches::ok;

__global__ void softcap_kernel(float* x, uint64_t n, float cap) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

__global__ void swiglu_kernel(const float* g, const float* u, uint64_t n, float gcap, float ucap, float* h) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= n) return;
  const float gv = g[i];
  h[i] = gcap * tanhf(gv / gcap) * sigmoid(gv) * (ucap * tanhf(u[i] / ucap));
}

// x' = hist (taps - 1 rows) then x: token t's taps x'[t .. t + taps - 1], the current last
__device__ inline float tap(const float* x, const float* hist, uint32_t at, uint32_t h, uint32_t ch, uint32_t c) {
  return at < h ? (hist ? hist[(uint64_t) at * ch + c] : 0.0f) : x[(uint64_t) (at - h) * ch + c];
}

__global__ void conv_kernel(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, const float* w,
                            const float* hist, float scale, float* y) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * ch) return;
  const uint32_t c = (uint32_t) (i % ch), t = (uint32_t) (i / ch), h = taps - 1;
  float sum = 0;
  for (uint32_t j = 0; j < taps; j++) sum += w[(uint64_t) c * taps + j] * tap(x, hist, t + j, h, ch, c);
  y[i] = (sum + x[i]) * scale;
}

__global__ void conv_dyn_kernel(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, uint32_t heads,
                                const float* w, const float* dyn, const float* hist, float scale, float* y) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * ch) return;
  const uint32_t c = (uint32_t) (i % ch), t = (uint32_t) (i / ch), h = taps - 1, head = c / (ch / heads);
  const float* d = dyn + (uint64_t) t * heads * taps + (uint64_t) head * taps;
  float sum = 0;
  for (uint32_t j = 0; j < taps; j++)  // tap j: j tokens back
    sum += (w[(uint64_t) j * ch + c] + d[j]) * tap(x, hist, t + h - j, h, ch, c);
  y[i] = (sum + x[i]) * scale;
}

// a block a token: its states' inverse RMS, each query's scores and softmax, then the weighted sum
__global__ void attn_res_kernel(const float* query, uint32_t queries, const float* blocks, uint32_t nb,
                                const float* partial, uint32_t dim, float eps, float scale, uint32_t n_tok,
                                float* out) {
  __shared__ float lds[kThreads];
  __shared__ float inv[INK2_MAX_RES], p[INK2_MAX_QUERIES][INK2_MAX_RES];
  const uint32_t t = blockIdx.x, ns = nb + 1;
  auto state = [&](uint32_t s) {
    return s < nb ? blocks + ((uint64_t) t * nb + s) * dim : partial + (uint64_t) t * dim;
  };
  for (uint32_t s = 0; s < ns; s++) {
    const float* v = state(s);
    float sq = 0;
    for (uint32_t c = threadIdx.x; c < dim; c += kThreads) sq += v[c] * v[c];
    sq = block_sum<kThreads>(sq, lds);
    if (!threadIdx.x) inv[s] = rsqrtf(sq / (float) dim + eps);
  }
  __syncthreads();
  for (uint32_t q = 0; q < queries; q++)
    for (uint32_t s = 0; s < ns; s++) {
      const float* v = state(s);
      float d = 0;
      for (uint32_t c = threadIdx.x; c < dim; c += kThreads) d += query[(uint64_t) q * dim + c] * v[c];
      d = block_sum<kThreads>(d, lds);
      if (!threadIdx.x) p[q][s] = d * inv[s] * scale;
    }
  __syncthreads();
  if (threadIdx.x < queries) {
    float* r = p[threadIdx.x];
    float m = r[0], z = 0;
    for (uint32_t s = 1; s < ns; s++) m = fmaxf(m, r[s]);
    for (uint32_t s = 0; s < ns; s++) z += (r[s] = expf(r[s] - m));
    for (uint32_t s = 0; s < ns; s++) r[s] = r[s] / z * inv[s];  // (the state's own normalization folded in)
  }
  __syncthreads();
  for (uint32_t c = threadIdx.x; c < dim; c += kThreads)
    for (uint32_t q = 0; q < queries; q++) {
      float o = 0;
      for (uint32_t s = 0; s < ns; s++) o += p[q][s] * state(s)[c];
      out[((uint64_t) q * n_tok + t) * dim + c] = o;
    }
}

// a block a row: times w, the L2 clip, tau
__global__ void clip_rows_kernel(float* x, uint32_t per_tok, uint32_t dim, const float* w, float clip, Ink2Tau tau) {
  __shared__ float lds[kThreads];
  const uint32_t r = blockIdx.x, j = r % per_tok, t = r / per_tok;
  float* row = x + (uint64_t) r * dim;
  float sq = 0;
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) {
    const float v = w ? row[d] * w[(uint64_t) j * dim + d] : row[d];
    row[d] = v, sq += v * v;
  }
  float f = 1.0f;
  if (clip > 0) f = fminf(1.0f, clip / (sqrtf(block_sum<kThreads>(sq, lds)) + 1e-6f));
  if (tau.alpha > 0) f *= 1.0f + tau.alpha * logf(fmaxf(1.0f, (float) (tau.pos0 + t + 1) / tau.floor));
  for (uint32_t d = threadIdx.x; d < dim; d += kThreads) row[d] *= f;
}

constexpr uint32_t kChunk = INK2_MAX_HEAD;  // keys a pass: a thread a key, then a thread a value dim

// a block a (token, head), head_dim threads: the relative bias's row (rel . rel_proj, extent values), then the keys
// in chunks - a thread a key's logit, the chunk's max and sum folded into the running ones (online softmax), a
// thread a dim's weighted sum of the chunk's values
__global__ void attn_kernel(Ink2Attn a, const float* q, const float* rel, const float* sink, const float* gate,
                            uint32_t n_tok, const float* k, const float* v, uint32_t n_keys, const float* rel_proj,
                            float* out) {
  __shared__ float qs[INK2_MAX_HEAD], rl[INK2_MAX_EXTENT], p[kChunk], red[kChunk];
  const uint32_t t = blockIdx.x / a.heads, h = blockIdx.x % a.heads, kvh = h / (a.heads / a.kv_heads);
  const uint32_t D = a.head_dim, i = threadIdx.x, qi = n_keys - n_tok + t;
  qs[i] = q[((uint64_t) t * a.heads + h) * D + i];
  const float* rr = rel + ((uint64_t) t * a.heads + h) * a.d_rel;
  for (uint32_t e = i; e < a.extent; e += D) {
    float b = 0;
    for (uint32_t r = 0; r < a.d_rel; r++) b += rr[r] * rel_proj[(uint64_t) r * a.extent + e];
    rl[e] = b;
  }
  __syncthreads();
  const uint32_t lo = a.window && qi + 1 > a.window ? qi + 1 - a.window : 0;
  float m = sink[(uint64_t) t * a.heads + h], z = 1.0f, acc = 0;  // the sink: its logit, weight 1 at its max, value 0
  for (uint32_t s0 = lo; s0 <= qi; s0 += kChunk) {
    const uint32_t s = s0 + i, n = qi + 1 - s0 < kChunk ? qi + 1 - s0 : kChunk;
    float l = -INFINITY;
    if (i < n) {
      const float* kr = k + ((uint64_t) s * a.kv_heads + kvh) * D;
      float d = 0;
      for (uint32_t c = 0; c < D; c++) d += qs[c] * kr[c];
      const uint32_t dist = qi - s;
      l = d * a.scaling + (dist < a.extent ? rl[dist] : 0.0f);
    }
    const float cm = block_max<kChunk>(l, red), nm = fmaxf(m, cm), f = expf(m - nm);
    const float e = i < n ? expf(l - nm) : 0.0f;
    p[i] = e;
    const float cs = block_sum<kChunk>(e, red);
    z = z * f + cs, acc *= f, m = nm;
    for (uint32_t j = 0; j < n; j++) acc += p[j] * v[((uint64_t) (s0 + j) * a.kv_heads + kvh) * D + i];
    __syncthreads();
  }
  const uint64_t o = (uint64_t) t * a.heads * D + (uint64_t) h * D + i;
  out[o] = acc / z * (gate ? sigmoid(gate[o]) : 1.0f);
}

}  // namespace

extern "C" bool ink2_clip_rows(float* x, uint32_t n_tok, uint32_t per_tok, uint32_t dim, const float* w, float clip,
                               Ink2Tau tau, void* stream) {
  if (!x || !n_tok || !per_tok || !dim || (tau.alpha > 0 && !(tau.floor > 0))) return false;
  clip_rows_kernel<<<n_tok * per_tok, kThreads, 0, st(stream)>>>(x, per_tok, dim, w, clip, tau);
  return ok();
}

extern "C" bool ink2_attn(const Ink2Attn* a, const float* q, const float* rel, const float* sink, const float* gate,
                          uint32_t n_tok, const float* k, const float* v, uint32_t n_keys, const float* rel_proj,
                          float* out, void* stream) {
  if (!a || !q || !rel || !sink || !k || !v || !rel_proj || !out || !n_tok || n_tok > n_keys ||
      a->head_dim != INK2_MAX_HEAD || !a->kv_heads || a->heads % a->kv_heads || !a->d_rel ||
      a->d_rel > INK2_MAX_REL || !a->extent || a->extent > INK2_MAX_EXTENT)
    return false;
  attn_kernel<<<n_tok * a->heads, a->head_dim, 0, st(stream)>>>(*a, q, rel, sink, gate, n_tok, k, v, n_keys, rel_proj,
                                                                out);
  return ok();
}

extern "C" bool ink2_softcap(float* x, uint64_t n, float cap, void* stream) {
  if (!x || !n || !(cap > 0)) return false;
  softcap_kernel<<<grid(n), kThreads, 0, st(stream)>>>(x, n, cap);
  return ok();
}

extern "C" bool ink2_swiglu(const float* g, const float* u, uint32_t n, float gcap, float ucap, uint32_t n_tok,
                            float* h, void* stream) {
  if (!g || !u || !h || !n || !n_tok || !(gcap > 0) || !(ucap > 0)) return false;
  const uint64_t all = (uint64_t) n * n_tok;
  swiglu_kernel<<<grid(all), kThreads, 0, st(stream)>>>(g, u, all, gcap, ucap, h);
  return ok();
}

extern "C" bool ink2_conv(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, const float* w,
                          const float* hist, float scale, float* y, void* stream) {
  if (!x || !w || !y || !n_tok || !ch || taps < 2 || taps > INK2_MAX_TAPS) return false;
  conv_kernel<<<grid((uint64_t) n_tok * ch), kThreads, 0, st(stream)>>>(x, n_tok, ch, taps, w, hist, scale, y);
  return ok();
}

extern "C" bool ink2_conv_dyn(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, uint32_t heads,
                              const float* w, const float* dyn, const float* hist, float scale, float* y,
                              void* stream) {
  if (!x || !w || !dyn || !y || !n_tok || !ch || taps < 2 || taps > INK2_MAX_TAPS || !heads || ch % heads)
    return false;
  conv_dyn_kernel<<<grid((uint64_t) n_tok * ch), kThreads, 0, st(stream)>>>(x, n_tok, ch, taps, heads, w, dyn, hist,
                                                                             scale, y);
  return ok();
}

extern "C" bool ink2_attn_res(const float* query, uint32_t queries, const float* blocks, uint32_t n_blocks,
                              const float* partial, uint32_t n_tok, uint32_t dim, float eps, float scale, float* out,
                              void* stream) {
  if (!query || !partial || !out || (n_blocks && !blocks) || !queries || queries > INK2_MAX_QUERIES ||
      n_blocks + 1 > INK2_MAX_RES || !n_tok || !dim)
    return false;
  attn_res_kernel<<<n_tok, kThreads, 0, st(stream)>>>(query, queries, blocks, n_blocks, partial, dim, eps, scale,
                                                      n_tok, out);
  return ok();
}
