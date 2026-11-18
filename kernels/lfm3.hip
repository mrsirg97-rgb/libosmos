// LFM3's own steps (include/osmos/gpu/lfm3_ops.h)
#include <hip/hip_runtime.h>

#include <cstdint>

#include "launch.h"

extern "C" {
#include "osmos/gpu/lfm3_ops.h"
}

namespace {

using launches::ok;
using launches::st;

constexpr uint32_t kThreads = 256, kRun = 8;  // a thread a channel over kRun consecutive tokens

// u = B * x of token t (its in_proj row: B, C, x), or of the state's row i (t < 0: hist[H + t])
__device__ inline float u_at(const float* bcx, const float* hist, uint32_t dim, int32_t t, uint32_t h, uint32_t c) {
  if (t < 0) return hist[(uint64_t) (h + t) * dim + c];
  const float* r = bcx + (uint64_t) t * 3 * dim;
  return r[c] * r[2 * dim + c];
}

// a thread a channel over kRun tokens: u' read once into a window of registers, the channel's taps once; each
// token's sum its taps in order (the oldest first), then C times it
template <int TAPS>
__global__ void conv_kernel(const float* bcx, uint32_t n_tok, uint32_t dim, const float* w, const float* hist,
                            float* out) {
  const uint32_t c = blockIdx.x * kThreads + threadIdx.x, t0 = blockIdx.y * kRun;
  if (c >= dim) return;
  constexpr uint32_t h = TAPS - 1;
  float wk[TAPS], win[kRun + h];
#pragma unroll
  for (int k = 0; k < TAPS; k++) wk[k] = w[(uint64_t) c * TAPS + k];
#pragma unroll
  for (uint32_t i = 0; i < kRun + h; i++) {  // u' rows t0 .. t0 + kRun + h - 1: token t0 + i - h
    const int32_t t = (int32_t) (t0 + i) - (int32_t) h;
    win[i] = t < (int32_t) n_tok ? u_at(bcx, hist, dim, t, h, c) : 0.0f;
  }
#pragma unroll
  for (uint32_t r = 0; r < kRun; r++) {
    if (t0 + r >= n_tok) break;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < TAPS; k++) s += wk[k] * win[r + k];
    out[(uint64_t) (t0 + r) * dim + c] = bcx[(uint64_t) (t0 + r) * 3 * dim + dim + c] * s;
  }
}

// after each of the step's last `keep` tokens, u''s last h rows up to it into its ring slot, a channel a thread: the
// slot read (cur) whole into registers first - a later token's slot may be it
__global__ void state_kernel(const float* bcx, uint32_t n_tok, uint32_t dim, uint32_t h, float* ring, uint32_t slots,
                             uint32_t cur, uint32_t keep) {
  const uint32_t c = blockIdx.x * kThreads + threadIdx.x;
  if (c >= dim) return;
  const uint64_t slot = (uint64_t) h * dim;
  float old[LFM3_CONV_TAPS_MAX - 1];
  for (uint32_t k = 0; k < h; k++) old[k] = ring[cur * slot + (uint64_t) k * dim + c];
  for (uint32_t t = n_tok - keep; t < n_tok; t++)
    for (uint32_t k = 0; k < h; k++) {
      const int32_t at = (int32_t) (t + 1 + k) - (int32_t) h;  // token at's u (at < 0: the state's)
      ring[(cur + 1 + t) % slots * slot + (uint64_t) k * dim + c] = at < 0 ? old[h + at] : u_at(bcx, nullptr, dim, at, h, c);
    }
}

}  // namespace

extern "C" bool lfm3_shortconv(const float* bcx, uint32_t n_tok, uint32_t dim, uint32_t taps, const float* w,
                               float* ring, uint32_t slots, uint32_t cur, float* out, void* stream) {
  if (!bcx || !w || !ring || !out || !n_tok || !dim || taps < 2 || taps > LFM3_CONV_TAPS_MAX || !slots || cur >= slots)
    return false;
  const uint32_t h = taps - 1;
  const float* hist = ring + (uint64_t) cur * h * dim;
  const dim3 grid((dim + kThreads - 1) / kThreads, (n_tok + kRun - 1) / kRun);
  const hipStream_t s = st(stream);
  switch (taps) {
#define TAPS(n) \
  case n: conv_kernel<n><<<grid, kThreads, 0, s>>>(bcx, n_tok, dim, w, hist, out); break;
    TAPS(2) TAPS(3) TAPS(4) TAPS(5) TAPS(6) TAPS(7) TAPS(8)
#undef TAPS
    default: return false;
  }
  if (!ok()) return false;
  const uint32_t keep = n_tok < slots ? n_tok : slots;
  state_kernel<<<(dim + kThreads - 1) / kThreads, kThreads, 0, s>>>(bcx, n_tok, dim, h, ring, slots, cur, keep);
  return ok();
}
