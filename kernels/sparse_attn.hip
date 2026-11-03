// ops_sparse_attn (include/substrate/gpu/ops.h): the model has one latent - key and value - for all its heads, so a
// token's attention is two products, scores = q (heads x hd) . keys^T and o = p . keys (p the scores' softmax), each a
// register-tiled F32 matmul over the token's own keys. every output a sum in one order - over d ascending, over the
// keys in theirs - whatever the tile, the batch, or the token's place in it
#include <cmath>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "kvload.h"

extern "C" {
#include "substrate/gpu/kvfmt.h"
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
// a few tokens (a decode's, a verify's: fewer than kSmallTok) take small tiles - all 64 heads x 64 keys, 64 heads x 64
// dims, 4 x 4 a thread - for blocks enough to fill the GPU (the large ones gave one token ~5 blocks of scores, 2 of
// values); a tile changes no sum's order: the same bits. 64 heads, not the 16 it was with F32 keys: a block decodes
// its keys once for all heads (kvload.h) - one token x 640 keys 136 -> 70 us a call (16 / 32 / 64 heads; F32 keys at
// 16: 59)
constexpr int kSmallHeads = 64, kSmallR = 4, kSmallC = 4, kSmallKeys = 64, kSmallDims = 64;
constexpr uint32_t kSmallTok = 16;
constexpr int kSmThreads = 256;  // the softmax's: a wave a (token, head)
constexpr float kNegHuge = -1e30f;

typedef float f4 __attribute__((ext_vector_type(4)));

hipStream_t st(void* s) { return (hipStream_t) s; }
bool ok() { return hipGetLastError() == hipSuccess; }

// a token's keys: the window's (positions pos - window + 1 .. pos, ring rows: KV_FP8) then its selected rows (ckv's,
// in ckv_fmt) - each read four values at a time as kvload.h reads them, the F32 their rounding gave
struct Keys {
  const uint8_t* ring;
  uint32_t slots;
  const int32_t* pos;
  uint32_t window;
  int32_t floor;  // no window key below it (a sequence's first position with the layer's state)
  const uint8_t* ckv;
  uint32_t ckv_fmt;
  const int32_t* sel;
  const int32_t* n_sel;
  uint32_t sel_cap, hd, stride;  // stride: a score row's floats (the keys' bound, to a float4)
  uint64_t ring_bytes, ckv_bytes;  // a row's (kvfmt.h)

  __device__ uint32_t j0(uint32_t t) const {
    const int32_t j = pos[t] - (int32_t) window + 1;
    return j > floor ? (uint32_t) j : (uint32_t) floor;
  }
  __device__ uint32_t count(uint32_t t, uint32_t& nw) const {
    nw = (uint32_t) pos[t] - j0(t) + 1;
    return nw + (ckv ? (uint32_t) n_sel[t] : 0);
  }
  // key k's row and its format
  __device__ const uint8_t* row(uint32_t t, uint32_t k, uint32_t nw, uint32_t& fmt) const {
    fmt = k < nw ? (uint32_t) KV_FP8 : ckv_fmt;
    return k < nw ? ring + (uint64_t) ((j0(t) + k) % slots) * ring_bytes
                  : ckv + (uint64_t) sel[(uint64_t) t * sel_cap + k - nw] * ckv_bytes;
  }
  __device__ kv::Raw fetch(const uint8_t* r, uint32_t fmt, uint32_t d) const { return kv::fetch4(fmt, r, hd, d); }
};

// a thread's R x C of a block's tile: R heads (a float4 group every 64 / (R / 4)) by C columns (a group every
// TW / (C / 4)) - acc[i][g] += a(head i) * b(group g), over the staged values in order
template <int HT, int R, int C, int TW>
__device__ inline void tile_sums(const float (*as)[HT + 4], const float (*bs)[TW + 4], int ty, int tx,
                                 f4 (&acc)[R][C / 4]) {
#pragma unroll
  for (int d = 0; d < kStep; d++) {
    f4 a[R / 4], b[C / 4];
    for (int i = 0; i < R / 4; i++) a[i] = *(const f4*) &as[d][ty * 4 + i * (HT / (R / 4))];
    for (int g = 0; g < C / 4; g++) b[g] = *(const f4*) &bs[d][tx * 4 + g * (TW / (C / 4))];
    for (int i = 0; i < R; i++) {
      const float av = a[i / 4][i % 4];
      for (int g = 0; g < C / 4; g++)
        acc[i][g] = __builtin_elementwise_fma(f4{av, av, av, av}, b[g], acc[i][g]);
    }
  }
}
__device__ inline int tile_head(int HT, int R, int ty, int i) { return ty * 4 + (i / 4) * (HT / (R / 4)) + i % 4; }

// s[t][h][k] = (sum over d ascending of q[t][h][d] key_k[d]) * scale: a block a token's 64 heads x SK keys, a
// thread SR x SC; each thread's rows' pointers once, the next stage's loads issued before this one's sums
template <int HT, int SR, int SC, int SK>
__global__ __launch_bounds__((HT / SR) * (SK / SC)) void scores_kernel(Keys K, const float* __restrict__ q,
                                                                      float scale, uint32_t heads,
                                                                      float* __restrict__ s) {
  constexpr int kSthreads = (HT / SR) * (SK / SC);
  constexpr int TX = SK / SC, G = SC / 4;
  constexpr int QL = HT * kStep / 4 / kSthreads, KL = SK * kStep / 4 / kSthreads;  // a thread's float4s
  static_assert(QL >= 1 && KL >= 1 && QL * kSthreads * 4 == HT * kStep, "whole float4s a thread");
  __shared__ __attribute__((aligned(16))) float qs[kStep][HT + 4], ks[kStep][SK + 4];
  const uint32_t t = blockIdx.z, h0 = blockIdx.y * HT, k0 = blockIdx.x * SK, hd = K.hd;
  uint32_t nw;
  const uint32_t n = K.count(t, nw);
  if (k0 >= n) return;  // the whole block
  const int tx = threadIdx.x % TX, ty = threadIdx.x / TX;
  const float* qp[QL];
  const uint8_t* kp[KL];
  uint32_t kf[KL];
  int qi[QL], qd[QL], ki[KL], kd[KL];  // a load's row and d in the stage
  for (int j = 0; j < QL; j++) {
    const int i = threadIdx.x + j * kSthreads;
    qi[j] = i / (kStep / 4), qd[j] = i % (kStep / 4) * 4;
    qp[j] = q + ((uint64_t) t * heads + h0 + qi[j]) * hd + qd[j];
  }
  for (int j = 0; j < KL; j++) {
    const int i = threadIdx.x + j * kSthreads;
    ki[j] = i / (kStep / 4), kd[j] = i % (kStep / 4) * 4;
    kp[j] = k0 + ki[j] < n ? K.row(t, k0 + ki[j], nw, kf[j]) : nullptr;
  }
  f4 qn[QL];
  kv::Raw kn[KL];  // the keys' bytes in flight, decoded when staged
  for (int j = 0; j < QL; j++) qn[j] = *(const f4*) qp[j];
  for (int j = 0; j < KL; j++) kn[j] = kp[j] ? K.fetch(kp[j], kf[j], kd[j]) : kv::Raw{};
  f4 acc[SR][G] = {};
  for (uint32_t d0 = 0; d0 < hd; d0 += kStep) {
    __syncthreads();
    for (int j = 0; j < QL; j++)
      for (int e = 0; e < 4; e++) qs[qd[j] + e][qi[j]] = qn[j][e];
    for (int j = 0; j < KL; j++) {
      const f4 kj = kp[j] ? kv::decode4(kf[j], kn[j]) : f4{};
      for (int e = 0; e < 4; e++) ks[kd[j] + e][ki[j]] = kj[e];
    }
    __syncthreads();
    if (d0 + kStep < hd) {
      for (int j = 0; j < QL; j++) qn[j] = *(const f4*) (qp[j] + d0 + kStep);
      for (int j = 0; j < KL; j++) kn[j] = kp[j] ? K.fetch(kp[j], kf[j], kd[j] + d0 + kStep) : kv::Raw{};
    }
    tile_sums<HT, SR, SC, SK>(qs, ks, ty, tx, acc);
  }
  for (int i = 0; i < SR; i++)
    for (int g = 0; g < G; g++) {
      const uint32_t h = h0 + tile_head(HT, SR, ty, i), k = k0 + tx * 4 + g * (SK / G);
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

// o[t][h][d] = (sum over the keys in order of p[t][h][k] key_k[d]) / den[t][h]: a block a token's 64 heads x VD
// dims, a thread VR x VC, the keys kStep a stage (past n zeros: 0 x 0, the sums' bits unchanged); the next stage's
// loads issued before this one's sums
template <int HT, int VR, int VC, int VD>
__global__ __launch_bounds__((HT / VR) * (VD / VC)) void values_kernel(Keys K, const float* __restrict__ p,
                                                                      const float* __restrict__ den, uint32_t heads,
                                                                      float* __restrict__ o) {
  constexpr int kVthreads = (HT / VR) * (VD / VC);
  constexpr int TX = VD / VC, G = VC / 4;
  constexpr int PL = HT * kStep / 4 / kVthreads, VL = kStep * VD / 4 / kVthreads;  // a thread's float4s
  static_assert(PL >= 1 && VL >= 1 && PL * kVthreads * 4 == HT * kStep, "whole float4s a thread");
  __shared__ __attribute__((aligned(16))) float ps[kStep][HT + 4], vs[kStep][VD + 4];
  const uint32_t t = blockIdx.z, h0 = blockIdx.y * HT, d0 = blockIdx.x * VD, hd = K.hd;
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
    vk[j] = i / (VD / 4), vd[j] = i % (VD / 4) * 4;
  }
  const float* pr = p + ((uint64_t) t * heads + h0) * K.stride;
  f4 pn[PL];
  kv::Raw vn[VL];  // the values' bytes in flight, decoded when staged
  uint32_t vf[VL];
  const auto load = [&](uint32_t k0) {
    for (int j = 0; j < PL; j++) {
      f4 x = {};
      for (int e = 0; e < 4; e++)
        if (k0 + pk[j] + e < n) x[e] = pr[(uint64_t) ph[j] * K.stride + k0 + pk[j] + e];
      pn[j] = x;
    }
    for (int j = 0; j < VL; j++) {
      vf[j] = KV_F32;  // (past n: zeros)
      vn[j] = k0 + vk[j] < n ? K.fetch(K.row(t, k0 + vk[j], nw, vf[j]), vf[j], d0 + vd[j]) : kv::Raw{};
    }
  };
  load(0);
  f4 acc[VR][G] = {};
  for (uint32_t k0 = 0; k0 < n; k0 += kStep) {
    __syncthreads();
    for (int j = 0; j < PL; j++)
      for (int e = 0; e < 4; e++) ps[pk[j] + e][ph[j]] = pn[j][e];
    for (int j = 0; j < VL; j++) *(f4*) &vs[vk[j]][vd[j]] = kv::decode4(vf[j], vn[j]);
    __syncthreads();
    if (k0 + kStep < n) load(k0 + kStep);
    tile_sums<HT, VR, VC, VD>(ps, vs, ty, tx, acc);
  }
  for (int i = 0; i < VR; i++) {
    const uint32_t h = h0 + tile_head(HT, VR, ty, i);
    const float dn = den[(uint64_t) t * heads + h];
    float* orow = o + ((uint64_t) t * heads + h) * hd + d0 + tx * 4;
    for (int g = 0; g < G; g++) *(f4*) (orow + g * (VD / G)) = acc[i][g] / dn;
  }
}

}  // namespace

extern "C" bool ops_sparse_attn(const float* q, const void* ring, uint32_t slots, const int32_t* pos,
                                uint32_t window, int32_t floor, const void* ckv, uint32_t ckv_fmt, const int32_t* sel,
                                const int32_t* n_sel, uint32_t sel_cap, uint32_t max_sel, const float* sink,
                                float scale, uint32_t heads, uint32_t head_dim, uint32_t n_tok, float* scores, float* o,
                                void* stream) {
  if (!window || window > OPS_MAX_WINDOW || slots < window || sel_cap > OPS_MAX_SELECT || max_sel > sel_cap ||
      !heads || heads % kHeads || !head_dim || head_dim % kVdims || head_dim % kStep || !n_tok || !scores ||
      (ckv && (!sel || !n_sel)) || floor < 0 || ckv_fmt > KV_FP4_POW2 || head_dim % kv_block((KvFmt) ckv_fmt))
    return false;
  const uint32_t keys = window + (ckv ? max_sel : 0);
  const Keys K{(const uint8_t*) ring, slots, pos, window, floor, (const uint8_t*) ckv, ckv_fmt, sel, n_sel, sel_cap,
               head_dim, OPS_ATTN_ROW(keys), kv_row_bytes(KV_FP8, head_dim), kv_row_bytes((KvFmt) ckv_fmt, head_dim)};
  float* den = scores + (uint64_t) n_tok * heads * K.stride;
  const hipStream_t ss = st(stream);
  if (n_tok < kSmallTok) {
    constexpr int ST = (kSmallHeads / kSmallR) * (kSmallKeys / kSmallC);
    constexpr int VT = (kSmallHeads / kSmallR) * (kSmallDims / kSmallC);
    scores_kernel<kSmallHeads, kSmallR, kSmallC, kSmallKeys>
        <<<dim3((keys + kSmallKeys - 1) / kSmallKeys, heads / kSmallHeads, n_tok), ST, 0, ss>>>(K, q, scale, heads,
                                                                                                scores);
    softmax_kernel<<<dim3((heads + kSmThreads / kWave - 1) / (kSmThreads / kWave), n_tok), kSmThreads, 0, ss>>>(
        K, sink, heads, scores, den);
    values_kernel<kSmallHeads, kSmallR, kSmallC, kSmallDims>
        <<<dim3(head_dim / kSmallDims, heads / kSmallHeads, n_tok), VT, 0, ss>>>(K, scores, den, heads, o);
    return ok();
  }
  constexpr int ST = (kHeads / kSr) * (kSkeys / kSc), VT = (kHeads / kVr) * (kVdims / kVc);
  const uint32_t ht = heads / kHeads;
  scores_kernel<kHeads, kSr, kSc, kSkeys><<<dim3((keys + kSkeys - 1) / kSkeys, ht, n_tok), ST, 0, ss>>>(K, q, scale,
                                                                                                    heads, scores);
  softmax_kernel<<<dim3((heads + kSmThreads / kWave - 1) / (kSmThreads / kWave), n_tok), kSmThreads, 0, ss>>>(
      K, sink, heads, scores, den);
  values_kernel<kHeads, kVr, kVc, kVdims><<<dim3(head_dim / kVdims, ht, n_tok), VT, 0, ss>>>(K, scores, den, heads, o);
  return ok();
}
