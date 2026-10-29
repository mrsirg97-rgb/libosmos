// ops_sparse_attn (include/substrate/gpu/ops.h): the model has one latent - key and value - for all its heads, so a
// token's attention is two products, scores = q (heads x hd) . keys^T and o = p . keys (p the scores' softmax), each a
// register-tiled F32 matmul over the token's own keys. every output a sum in one order - over d ascending, over the
// keys in theirs - whatever the tile, the batch, or the token's place in it
#include <cmath>
#include <cstdint>

#include <hip/hip_runtime.h>

extern "C" {
#include "substrate/gpu/ops.h"
}

namespace {

constexpr int kWave = 32;
constexpr int kTile = 64;               // a block's tile: 64 heads x 64 keys (scores), 64 heads x 64 dims (values)
constexpr int kPer = 4;                 // a thread's 4 x 4 of it
constexpr int kSide = kTile / kPer;     // 16 threads a side
constexpr int kThreads = kSide * kSide; // 256
constexpr int kStep = 16;               // the sum's values staged at a time (d for scores, keys for values)
constexpr int kLoads = kTile * kStep / (kThreads * kPer);  // a thread's float4s of a tile's stage: 1
constexpr float kNegHuge = -1e30f;
static_assert(kLoads == 1, "a stage a float4 a thread");

typedef float f4 __attribute__((ext_vector_type(4)));

hipStream_t st(void* s) { return (hipStream_t) s; }
bool ok() { return hipGetLastError() == hipSuccess; }

// a token's keys: the window's (positions pos - window + 1 .. pos, ring rows) then its selected compressed rows
struct Keys {
  const float* ring;
  uint32_t slots;
  const int32_t* pos;
  uint32_t window;
  int32_t floor;  // no window key below it (a sequence's first position with the layer's state)
  const float* ckv;
  const int32_t* sel;
  const int32_t* n_sel;
  uint32_t sel_cap, hd, stride;  // stride: a score row's floats (the keys' bound, to a float4)

  __device__ uint32_t j0(uint32_t t) const {
    const int32_t j = pos[t] - (int32_t) window + 1;
    return j > floor ? (uint32_t) j : (uint32_t) floor;
  }
  __device__ uint32_t count(uint32_t t, uint32_t& nw) const {
    nw = (uint32_t) pos[t] - j0(t) + 1;
    return nw + (ckv ? (uint32_t) n_sel[t] : 0);
  }
  __device__ const float* row(uint32_t t, uint32_t k, uint32_t nw) const {
    return k < nw ? ring + (uint64_t) ((j0(t) + k) % slots) * hd
                  : ckv + (uint64_t) sel[(uint64_t) t * sel_cap + k - nw] * hd;
  }
};

// s[t][h][k] = (sum over d ascending of q[t][h][d] key_k[d]) * scale: a block a token's 64 heads x 64 keys
__global__ __launch_bounds__(kThreads) void scores_kernel(Keys K, const float* __restrict__ q, float scale,
                                                         uint32_t heads, float* __restrict__ s) {
  __shared__ __attribute__((aligned(16))) float qs[kStep][kTile + kPer], ks[kStep][kTile + kPer];
  const uint32_t t = blockIdx.z, h0 = blockIdx.y * kTile, k0 = blockIdx.x * kTile, hd = K.hd;
  uint32_t nw;
  const uint32_t n = K.count(t, nw);
  if (k0 >= n) return;  // the whole block
  const int tx = threadIdx.x % kSide, ty = threadIdx.x / kSide;
  const int li = threadIdx.x / (kStep / kPer), ld = threadIdx.x % (kStep / kPer) * kPer;  // a stage's row, d
  const float* qr = q + ((uint64_t) t * heads + h0 + li) * hd + ld;
  const float* kr = k0 + li < n ? K.row(t, k0 + li, nw) + ld : nullptr;
  f4 acc[kPer] = {};
  for (uint32_t d0 = 0; d0 < hd; d0 += kStep) {
    const f4 qv = *(const f4*) (qr + d0), kv = kr ? *(const f4*) (kr + d0) : f4{};
    __syncthreads();
    for (int j = 0; j < kPer; j++) qs[ld + j][li] = qv[j], ks[ld + j][li] = kv[j];
    __syncthreads();
#pragma unroll
    for (int d = 0; d < kStep; d++) {
      const f4 a = *(const f4*) &qs[d][ty * kPer], b = *(const f4*) &ks[d][tx * kPer];
      for (int i = 0; i < kPer; i++) acc[i] = __builtin_elementwise_fma(f4{a[i], a[i], a[i], a[i]}, b, acc[i]);
    }
  }
  for (int i = 0; i < kPer; i++) {
    float* sr = s + ((uint64_t) t * heads + h0 + ty * kPer + i) * K.stride + k0 + tx * kPer;
    if (k0 + tx * kPer < n) *(f4*) sr = acc[i] * scale;  // past n: the row's padding, never read as a key
  }
}

// a row's softmax with the head's sink, a wave a (token, head): p = exp(s - m) in place, den[t][h] its sum + the
// sink's - lane-strided then a fixed tree (the max exact whatever the order)
__global__ __launch_bounds__(kThreads) void softmax_kernel(Keys K, const float* __restrict__ sink, uint32_t heads,
                                                          float* __restrict__ s, float* __restrict__ den) {
  const uint32_t t = blockIdx.y, h = blockIdx.x * (kThreads / kWave) + threadIdx.x / kWave, lane = threadIdx.x % kWave;
  if (h >= heads) return;
  uint32_t nw;
  const uint32_t n = K.count(t, nw);
  float* r = s + ((uint64_t) t * heads + h) * K.stride;
  const float sk = sink[h];
  float m = sk;
  for (uint32_t k = lane; k < n; k += kWave) m = fmaxf(m, r[k]);
  for (int off = kWave / 2; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor(m, off, kWave));
  m = fmaxf(m, kNegHuge);
  float sum = 0;
  for (uint32_t k = lane; k < n; k += kWave) sum += (r[k] = expf(r[k] - m));
  for (int off = kWave / 2; off > 0; off >>= 1) sum += __shfl_xor(sum, off, kWave);
  if (lane == 0) den[(uint64_t) t * heads + h] = sum + expf(sk - m);
}

// o[t][h][d] = (sum over the keys in order of p[t][h][k] key_k[d]) / den[t][h]: a block a token's 64 heads x 64 dims
__global__ __launch_bounds__(kThreads) void values_kernel(Keys K, const float* __restrict__ p,
                                                         const float* __restrict__ den, uint32_t heads,
                                                         float* __restrict__ o) {
  __shared__ __attribute__((aligned(16))) float ps[kStep][kTile + kPer], vs[kStep][kTile + kPer];
  const uint32_t t = blockIdx.z, h0 = blockIdx.y * kTile, d0 = blockIdx.x * kTile, hd = K.hd;
  uint32_t nw;
  const uint32_t n = K.count(t, nw);
  const int tx = threadIdx.x % kSide, ty = threadIdx.x / kSide;
  const int ph = threadIdx.x / (kStep / kPer), pk = threadIdx.x % (kStep / kPer) * kPer;  // p's stage: head, key
  const int vk = threadIdx.x / (kTile / kPer), vd = threadIdx.x % (kTile / kPer) * kPer;  // the values': key, d
  const float* pr = p + ((uint64_t) t * heads + h0 + ph) * K.stride;
  f4 acc[kPer] = {};
  for (uint32_t k0 = 0; k0 < n; k0 += kStep) {
    f4 pv = {}, vv = {};  // past n: 0 x 0, the sums' bits unchanged
    for (int j = 0; j < kPer; j++)
      if (k0 + pk + j < n) pv[j] = pr[k0 + pk + j];
    if (k0 + vk < n) vv = *(const f4*) (K.row(t, k0 + vk, nw) + d0 + vd);
    __syncthreads();
    for (int j = 0; j < kPer; j++) ps[pk + j][ph] = pv[j];
    *(f4*) &vs[vk][vd] = vv;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kStep; k++) {
      const f4 a = *(const f4*) &ps[k][ty * kPer], b = *(const f4*) &vs[k][tx * kPer];
      for (int i = 0; i < kPer; i++) acc[i] = __builtin_elementwise_fma(f4{a[i], a[i], a[i], a[i]}, b, acc[i]);
    }
  }
  for (int i = 0; i < kPer; i++) {
    const uint32_t h = h0 + ty * kPer + i;
    *(f4*) (o + ((uint64_t) t * heads + h) * hd + d0 + tx * kPer) = acc[i] / den[(uint64_t) t * heads + h];
  }
}

}  // namespace

extern "C" bool ops_sparse_attn(const float* q, const float* ring, uint32_t slots, const int32_t* pos,
                                uint32_t window, int32_t floor, const float* ckv, const int32_t* sel, const int32_t* n_sel,
                                uint32_t sel_cap, uint32_t max_sel, const float* sink, float scale, uint32_t heads,
                                uint32_t head_dim, uint32_t n_tok, float* scores, float* o, void* stream) {
  if (!window || window > OPS_MAX_WINDOW || slots < window || sel_cap > OPS_MAX_SELECT || max_sel > sel_cap ||
      !heads || heads % kTile || !head_dim || head_dim % kTile || !n_tok || !scores || (ckv && (!sel || !n_sel)) ||
      floor < 0)
    return false;
  const uint32_t keys = window + (ckv ? max_sel : 0);
  const Keys K{ring, slots, pos, window, floor, ckv, sel, n_sel, sel_cap, head_dim, OPS_ATTN_ROW(keys)};
  float* den = scores + (uint64_t) n_tok * heads * K.stride;
  const uint32_t ht = heads / kTile;
  scores_kernel<<<dim3((keys + kTile - 1) / kTile, ht, n_tok), kThreads, 0, st(stream)>>>(K, q, scale, heads, scores);
  softmax_kernel<<<dim3((heads + kThreads / kWave - 1) / (kThreads / kWave), n_tok), kThreads, 0, st(stream)>>>(
      K, sink, heads, scores, den);
  values_kernel<<<dim3(head_dim / kTile, ht, n_tok), kThreads, 0, st(stream)>>>(K, scores, den, heads, o);
  return ok();
}
