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
constexpr int kHeads = 64;  // a block's heads (both products)
constexpr int kStep = 16;   // the sum's values staged at a time (d for scores, keys for values)
// the register tiles, the best measured on gfx1151 at 2,048 tokens x 640 keys (scratch sab.hip / sav.hip; every one
// the same bits): scores a thread 16 heads x 8 keys of a block's 64 x 128 (8.9 ms; the 4 x 4 of 64 x 64 before it
// 13.0, 8 x 8 9.4); values 8 heads x 16 dims of 64 x 256 (9.8 ms; 4 x 4 13.9, 8 x 8 of 128 dims 10.0)
constexpr int kSr = 16, kSc = 8, kSkeys = 128;
constexpr int kVr = 8, kVc = 16, kVdims = 256;
constexpr int kSmThreads = 256;  // the softmax's: a wave a (token, head)
constexpr float kNegHuge = -1e30f;

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

// a thread's R x C of a block's tile: R heads (a float4 group every 64 / (R / 4)) by C columns (a group every
// TW / (C / 4)) - acc[i][g] += a(head i) * b(group g), over the staged values in order
template <int R, int C, int TW>
__device__ inline void tile_sums(const float (*as)[kHeads + 4], const float (*bs)[TW + 4], int ty, int tx,
                                 f4 (&acc)[R][C / 4]) {
#pragma unroll
  for (int d = 0; d < kStep; d++) {
    f4 a[R / 4], b[C / 4];
    for (int i = 0; i < R / 4; i++) a[i] = *(const f4*) &as[d][ty * 4 + i * (kHeads / (R / 4))];
    for (int g = 0; g < C / 4; g++) b[g] = *(const f4*) &bs[d][tx * 4 + g * (TW / (C / 4))];
    for (int i = 0; i < R; i++) {
      const float av = a[i / 4][i % 4];
      for (int g = 0; g < C / 4; g++)
        acc[i][g] = __builtin_elementwise_fma(f4{av, av, av, av}, b[g], acc[i][g]);
    }
  }
}
__device__ inline int tile_head(int R, int ty, int i) { return ty * 4 + (i / 4) * (kHeads / (R / 4)) + i % 4; }

// s[t][h][k] = (sum over d ascending of q[t][h][d] key_k[d]) * scale: a block a token's 64 heads x kSkeys keys, a
// thread kSr x kSc; each thread's rows' pointers once, the next stage's loads issued before this one's sums
constexpr int kSthreads = (kHeads / kSr) * (kSkeys / kSc);
__global__ __launch_bounds__(kSthreads) void scores_kernel(Keys K, const float* __restrict__ q, float scale,
                                                          uint32_t heads, float* __restrict__ s) {
  constexpr int TX = kSkeys / kSc, G = kSc / 4;
  constexpr int QL = kHeads * kStep / 4 / kSthreads, KL = kSkeys * kStep / 4 / kSthreads;  // a thread's float4s
  static_assert(QL >= 1 && KL >= 1 && QL * kSthreads * 4 == kHeads * kStep, "whole float4s a thread");
  __shared__ __attribute__((aligned(16))) float qs[kStep][kHeads + 4], ks[kStep][kSkeys + 4];
  const uint32_t t = blockIdx.z, h0 = blockIdx.y * kHeads, k0 = blockIdx.x * kSkeys, hd = K.hd;
  uint32_t nw;
  const uint32_t n = K.count(t, nw);
  if (k0 >= n) return;  // the whole block
  const int tx = threadIdx.x % TX, ty = threadIdx.x / TX;
  const float* qp[QL];
  const float* kp[KL];
  int qi[QL], qd[QL], ki[KL], kd[KL];  // a load's row and d in the stage
  for (int j = 0; j < QL; j++) {
    const int i = threadIdx.x + j * kSthreads;
    qi[j] = i / (kStep / 4), qd[j] = i % (kStep / 4) * 4;
    qp[j] = q + ((uint64_t) t * heads + h0 + qi[j]) * hd + qd[j];
  }
  for (int j = 0; j < KL; j++) {
    const int i = threadIdx.x + j * kSthreads;
    ki[j] = i / (kStep / 4), kd[j] = i % (kStep / 4) * 4;
    kp[j] = k0 + ki[j] < n ? K.row(t, k0 + ki[j], nw) + kd[j] : nullptr;
  }
  f4 qn[QL], kn[KL];
  for (int j = 0; j < QL; j++) qn[j] = *(const f4*) qp[j];
  for (int j = 0; j < KL; j++) kn[j] = kp[j] ? *(const f4*) kp[j] : f4{};
  f4 acc[kSr][G] = {};
  for (uint32_t d0 = 0; d0 < hd; d0 += kStep) {
    __syncthreads();
    for (int j = 0; j < QL; j++)
      for (int e = 0; e < 4; e++) qs[qd[j] + e][qi[j]] = qn[j][e];
    for (int j = 0; j < KL; j++)
      for (int e = 0; e < 4; e++) ks[kd[j] + e][ki[j]] = kn[j][e];
    __syncthreads();
    if (d0 + kStep < hd) {
      for (int j = 0; j < QL; j++) qn[j] = *(const f4*) (qp[j] + d0 + kStep);
      for (int j = 0; j < KL; j++) kn[j] = kp[j] ? *(const f4*) (kp[j] + d0 + kStep) : f4{};
    }
    tile_sums<kSr, kSc, kSkeys>(qs, ks, ty, tx, acc);
  }
  for (int i = 0; i < kSr; i++)
    for (int g = 0; g < G; g++) {
      const uint32_t h = h0 + tile_head(kSr, ty, i), k = k0 + tx * 4 + g * (kSkeys / G);
      if (k < n) *(f4*) (s + ((uint64_t) t * heads + h) * K.stride + k) = acc[i][g] * scale;  // past n: padding
    }
}

// a row's softmax with the head's sink, a wave a (token, head): p = exp(s - m) in place, den[t][h] its sum + the
// sink's - lane-strided then a fixed tree (the max exact whatever the order)
__global__ __launch_bounds__(kSmThreads) void softmax_kernel(Keys K, const float* __restrict__ sink, uint32_t heads,
                                                            float* __restrict__ s, float* __restrict__ den) {
  const uint32_t t = blockIdx.y, h = blockIdx.x * (kSmThreads / kWave) + threadIdx.x / kWave;
  const uint32_t lane = threadIdx.x % kWave;
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

// o[t][h][d] = (sum over the keys in order of p[t][h][k] key_k[d]) / den[t][h]: a block a token's 64 heads x kVdims
// dims, a thread kVr x kVc, the keys kStep a stage (past n zeros: 0 x 0, the sums' bits unchanged); the next stage's
// loads issued before this one's sums
constexpr int kVthreads = (kHeads / kVr) * (kVdims / kVc);
__global__ __launch_bounds__(kVthreads) void values_kernel(Keys K, const float* __restrict__ p,
                                                          const float* __restrict__ den, uint32_t heads,
                                                          float* __restrict__ o) {
  constexpr int TX = kVdims / kVc, G = kVc / 4;
  constexpr int PL = kHeads * kStep / 4 / kVthreads, VL = kStep * kVdims / 4 / kVthreads;  // a thread's float4s
  static_assert(PL >= 1 && VL >= 1 && PL * kVthreads * 4 == kHeads * kStep, "whole float4s a thread");
  __shared__ __attribute__((aligned(16))) float ps[kStep][kHeads + 4], vs[kStep][kVdims + 4];
  const uint32_t t = blockIdx.z, h0 = blockIdx.y * kHeads, d0 = blockIdx.x * kVdims, hd = K.hd;
  uint32_t nw;
  const uint32_t n = K.count(t, nw);
  const int tx = threadIdx.x % TX, ty = threadIdx.x / TX;
  int ph[PL], pk[PL], vk[VL], vd[VL];  // a load's head and key (p's), key and d (the values')
  for (int j = 0; j < PL; j++) {
    const int i = threadIdx.x + j * kVthreads;
    ph[j] = i / (kStep / 4), pk[j] = i % (kStep / 4) * 4;
  }
  for (int j = 0; j < VL; j++) {
    const int i = threadIdx.x + j * kVthreads;
    vk[j] = i / (kVdims / 4), vd[j] = i % (kVdims / 4) * 4;
  }
  const float* pr = p + ((uint64_t) t * heads + h0) * K.stride;
  f4 pn[PL], vn[VL];
  const auto load = [&](uint32_t k0) {
    for (int j = 0; j < PL; j++) {
      f4 x = {};
      for (int e = 0; e < 4; e++)
        if (k0 + pk[j] + e < n) x[e] = pr[(uint64_t) ph[j] * K.stride + k0 + pk[j] + e];
      pn[j] = x;
    }
    for (int j = 0; j < VL; j++)
      vn[j] = k0 + vk[j] < n ? *(const f4*) (K.row(t, k0 + vk[j], nw) + d0 + vd[j]) : f4{};
  };
  load(0);
  f4 acc[kVr][G] = {};
  for (uint32_t k0 = 0; k0 < n; k0 += kStep) {
    __syncthreads();
    for (int j = 0; j < PL; j++)
      for (int e = 0; e < 4; e++) ps[pk[j] + e][ph[j]] = pn[j][e];
    for (int j = 0; j < VL; j++) *(f4*) &vs[vk[j]][vd[j]] = vn[j];
    __syncthreads();
    if (k0 + kStep < n) load(k0 + kStep);
    tile_sums<kVr, kVc, kVdims>(ps, vs, ty, tx, acc);
  }
  for (int i = 0; i < kVr; i++) {
    const uint32_t h = h0 + tile_head(kVr, ty, i);
    const float dn = den[(uint64_t) t * heads + h];
    float* orow = o + ((uint64_t) t * heads + h) * hd + d0 + tx * 4;
    for (int g = 0; g < G; g++) *(f4*) (orow + g * (kVdims / G)) = acc[i][g] / dn;
  }
}

}  // namespace

extern "C" bool ops_sparse_attn(const float* q, const float* ring, uint32_t slots, const int32_t* pos,
                                uint32_t window, int32_t floor, const float* ckv, const int32_t* sel,
                                const int32_t* n_sel, uint32_t sel_cap, uint32_t max_sel, const float* sink,
                                float scale, uint32_t heads, uint32_t head_dim, uint32_t n_tok, float* scores, float* o,
                                void* stream) {
  if (!window || window > OPS_MAX_WINDOW || slots < window || sel_cap > OPS_MAX_SELECT || max_sel > sel_cap ||
      !heads || heads % kHeads || !head_dim || head_dim % kVdims || head_dim % kStep || !n_tok || !scores ||
      (ckv && (!sel || !n_sel)) || floor < 0)
    return false;
  const uint32_t keys = window + (ckv ? max_sel : 0);
  const Keys K{ring, slots, pos, window, floor, ckv, sel, n_sel, sel_cap, head_dim, OPS_ATTN_ROW(keys)};
  float* den = scores + (uint64_t) n_tok * heads * K.stride;
  const uint32_t ht = heads / kHeads;
  scores_kernel<<<dim3((keys + kSkeys - 1) / kSkeys, ht, n_tok), kSthreads, 0, st(stream)>>>(K, q, scale, heads,
                                                                                               scores);
  softmax_kernel<<<dim3((heads + kSmThreads / kWave - 1) / (kSmThreads / kWave), n_tok), kSmThreads, 0, st(stream)>>>(
      K, sink, heads, scores, den);
  values_kernel<<<dim3(head_dim / kVdims, ht, n_tok), kVthreads, 0, st(stream)>>>(K, scores, den, heads, o);
  return ok();
}
