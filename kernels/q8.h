// the lane's experts over int8 inputs (kernels/lane.h's contract; include/substrate/gpu/q8.h): a 32-value block of a
// weight row decoded to int8 codes, two words a chunk of 8, against the input's int8 words by v_dot4 - exact integer
// sums - then each half's lane_q8_part and the block's scale, in the contract's order
#pragma once
#include "lane.h"
#include "quant.h"

namespace q8 {

constexpr int kWave = 32;
constexpr int kWavesPerBlock = 8;
constexpr int kWords = LANE_Q8_VALUES / 4;  // a block's int8 x as 32-bit words
constexpr int kBlocksPer = quant::kQK / LANE_Q8_VALUES;

__device__ inline int dot4(int a, int b, int c) { return __builtin_amdgcn_sudot4(true, a, true, b, c, false); }

// a block's codes: w its 8 words (two a chunk); IQ1_M's chunk c a sign of its own (its grid + delta as 8 grid +- 1:
// 8 (grid . x) +- the chunk's x summed); a half h's 2s + 1; the 256-value block's scale d
struct Codes {
  int w[kWords];
  int delta[4];  // 0: none (the IQ2s), else +-1 (IQ1_M)
  int s21[2];
  float d;
};

// 8 unsigned grid bytes signed by 8 sign bits, as two words of int8: (v ^ m) + (m & 1) a byte, m 0xFF where negative
__device__ inline void signed8(uint64_t grid, uint32_t signs, int& lo, int& hi) {
  uint32_t m0 = 0, m1 = 0;
  for (int i = 0; i < 4; i++) m0 |= ((signs >> i) & 1u) * (0xFFu << (8 * i)), m1 |= ((signs >> (4 + i)) & 1u) * (0xFFu << (8 * i));
  lo = (int) (((uint32_t) grid ^ m0) + (m0 & 0x01010101u)), hi = (int) (((uint32_t) (grid >> 32) ^ m1) + (m1 & 0x01010101u));
}

template <quant::Type T> __device__ void codes(const uint8_t* b, int j, Codes& c);

template <> __device__ inline void codes<quant::kIQ2_XXS>(const uint8_t* b, int j, Codes& c) {
  const uint8_t* q = b + 2 + 8 * j;
  const uint32_t aux = quant::ld32(q + 4);
  for (int k = 0; k < 4; k++) {
    signed8(kIq2xxsGrid[q[k]], kKsigns[(aux >> (7 * k)) & 0x7F], c.w[2 * k], c.w[2 * k + 1]);
    c.delta[k] = 0;
  }
  c.s21[0] = c.s21[1] = 2 * (int) (aux >> 28) + 1;
  c.d = quant::f16(quant::ld16(b));
}

template <> __device__ inline void codes<quant::kIQ2_XS>(const uint8_t* b, int j, Codes& c) {
  for (int k = 0; k < 4; k++) {
    const uint32_t q = quant::ld16(b + 2 + 2 * (4 * j + k));
    signed8(kIq2xsGrid[q & 511], kKsigns[q >> 9], c.w[2 * k], c.w[2 * k + 1]);
    c.delta[k] = 0;
  }
  const uint32_t s = b[66 + j];
  c.s21[0] = 2 * (int) (s & 0x0F) + 1, c.s21[1] = 2 * (int) (s >> 4) + 1;
  c.d = quant::f16(quant::ld16(b));
}

template <> __device__ inline void codes<quant::kIQ1_M>(const uint8_t* b, int j, Codes& c) {
  const uint64_t sc = *(const uint64_t*) (b + 48);  // (a block is 56 bytes, every one 8-aligned)
  const uint32_t s0 = (uint32_t) sc & 0xFFFF, s1 = (uint32_t) (sc >> 16) & 0xFFFF;
  const uint32_t s2 = (uint32_t) (sc >> 32) & 0xFFFF, s3 = (uint32_t) (sc >> 48);
  for (int k = 0; k < 4; k++) {
    const int kk = 4 * j + k;
    const uint32_t h = (b[32 + kk / 2] >> (4 * (kk % 2))) & 0x0F;
    const uint64_t g = kIq1sGrid[b[kk] | (h & 0x07) << 8];
    c.w[2 * k] = (int) (uint32_t) g, c.w[2 * k + 1] = (int) (uint32_t) (g >> 32);
    c.delta[k] = h & 0x08 ? -1 : 1;
  }
  for (int h = 0; h < 2; h++) {
    const int kk = 4 * j + 2 * h;
    c.s21[h] = 2 * (int) ((sc >> (16 * (kk / 8) + 3 * ((kk / 2) % 4))) & 0x07) + 1;
  }
  c.d = quant::f16((s0 >> 12) | ((s1 >> 8) & 0x00F0) | ((s2 >> 4) & 0x0F00) | (s3 & 0xF000));
}

// the block's sum against x's 8 words, before x's scale: each half's integer dot, then lane_q8_part's, added
template <quant::Type T>
__device__ inline float block_sum(const Codes& c, const int* x) {
  int isum[2];
  for (int h = 0; h < 2; h++) {
    int s = 0;
    for (int k = 2 * h; k < 2 * h + 2; k++) {
      const int gx = dot4(c.w[2 * k + 1], x[2 * k + 1], dot4(c.w[2 * k], x[2 * k], 0));
      if constexpr (T == quant::kIQ1_M) {
        const int sx = dot4(0x01010101, x[2 * k + 1], dot4(0x01010101, x[2 * k], 0));
        s += 8 * gx + c.delta[k] * sx;
      } else {
        s += gx;
      }
    }
    isum[h] = s;
  }
  return lane_q8_part(c.d, c.s21[0], isum[0]) + lane_q8_part(c.d, c.s21[1], isum[1]);
}

// matrix blockIdx.y: G = 32 / L rows a wave, L = LANE_Q8_LANES(cols) lanes a row; lane l of a row its blocks l,
// l + L, ... in order for each of its NT tokens, then the lanes' tree. rows past cnt[m] summed, not stored
template <quant::Type T, int NT, int L>
__global__ __launch_bounds__(kWave* kWavesPerBlock) void group_tok_kernel(const uint8_t* const* __restrict__ w,
                                                                           const int32_t* __restrict__ cnt,
                                                                           const int32_t* __restrict__ xrow,
                                                                           uint32_t rows, uint32_t cols,
                                                                           const int8_t* __restrict__ q,
                                                                           const float* __restrict__ d,
                                                                           float* __restrict__ y, uint32_t ys) {
  constexpr int G = kWave / L;
  const int lane = threadIdx.x % kWave, ll = lane % L;
  const uint32_t m = blockIdx.y, row = (blockIdx.x * kWavesPerBlock + threadIdx.x / kWave) * G + lane / L;
  const uint32_t nblk = cols / LANE_Q8_VALUES, r0 = (uint32_t) xrow[m];
  const uint8_t* rb = w[m] + (uint64_t) (row < rows ? row : 0) * (cols / quant::kQK) * quant::Block<T>::bytes;
  float acc[NT];
  for (int t = 0; t < NT; t++) acc[t] = 0.0f;
  for (uint32_t s = ll; s < nblk; s += L) {
    Codes c;
    codes<T>(rb + (s / kBlocksPer) * quant::Block<T>::bytes, (int) (s % kBlocksPer), c);
    for (int t = 0; t < NT; t++) {
      const uint64_t xr = r0 + t;
      const int4* xp = (const int4*) (q + xr * cols + (uint64_t) s * LANE_Q8_VALUES);
      const int4 a = xp[0], b = xp[1];
      const int x[kWords] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      acc[t] = fmaf(d[xr * nblk + s], block_sum<T>(c, x), acc[t]);
    }
  }
  for (int t = 0; t < NT; t++) {
    float v = acc[t];
    for (int o = L / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, kWave);
    if (ll == 0 && row < rows && t < cnt[m]) y[(uint64_t) (r0 + t) * ys + row] = v;
  }
}

// gate and up at once, then the activation: matrix pair blockIdx.y's gate rows (wg) and up rows (wu) of one type,
// each lane's blocks decoded from both and summed against x read once, each sum in group_tok_kernel's order and tree;
// then lane 0 of a row h = lane_swiglu(gate, up, lim, scale[x row]) - swiglu_lane_kernel's operation on the sums
// q8_group_tok gives, its two outputs never written
template <quant::Type T, int NT, int L>
__global__ __launch_bounds__(kWave* kWavesPerBlock) void gate_up_kernel(const uint8_t* const* __restrict__ wg,
                                                                         const uint8_t* const* __restrict__ wu,
                                                                         const int32_t* __restrict__ cnt,
                                                                         const int32_t* __restrict__ xrow,
                                                                         uint32_t rows, uint32_t cols,
                                                                         const int8_t* __restrict__ q,
                                                                         const float* __restrict__ d,
                                                                         const float* __restrict__ scale, float lim,
                                                                         float* __restrict__ h, uint32_t hs) {
  constexpr int G = kWave / L;
  const int lane = threadIdx.x % kWave, ll = lane % L;
  const uint32_t m = blockIdx.y, row = (blockIdx.x * kWavesPerBlock + threadIdx.x / kWave) * G + lane / L;
  const uint32_t nblk = cols / LANE_Q8_VALUES, r0 = (uint32_t) xrow[m];
  const uint64_t off = (uint64_t) (row < rows ? row : 0) * (cols / quant::kQK) * quant::Block<T>::bytes;
  const uint8_t *rg = wg[m] + off, *ru = wu[m] + off;
  float ag[NT], au[NT];
  for (int t = 0; t < NT; t++) ag[t] = au[t] = 0.0f;
  for (uint32_t s = ll; s < nblk; s += L) {
    Codes cg, cu;
    const uint64_t at = (s / kBlocksPer) * quant::Block<T>::bytes;
    codes<T>(rg + at, (int) (s % kBlocksPer), cg);
    codes<T>(ru + at, (int) (s % kBlocksPer), cu);
    for (int t = 0; t < NT; t++) {
      const uint64_t xr = r0 + t;
      const int4* xp = (const int4*) (q + xr * cols + (uint64_t) s * LANE_Q8_VALUES);
      const int4 a = xp[0], b = xp[1];
      const int x[kWords] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      const float xd = d[xr * nblk + s];
      ag[t] = fmaf(xd, block_sum<T>(cg, x), ag[t]);
      au[t] = fmaf(xd, block_sum<T>(cu, x), au[t]);
    }
  }
  for (int t = 0; t < NT; t++) {
    float vg = ag[t], vu = au[t];
    for (int o = L / 2; o > 0; o >>= 1) vg += __shfl_xor(vg, o, kWave), vu += __shfl_xor(vu, o, kWave);
    if (ll == 0 && row < rows && t < cnt[m])
      h[(uint64_t) (r0 + t) * hs + row] = lane_swiglu(vg, vu, lim, scale[r0 + t]);
  }
}

// rows of x to int8 a block at a time: a wave a block (lane i its value i), the block's max magnitude by the wave's
// tree, each lane's code, four lanes' bytes a word; d the block's scale
__global__ __launch_bounds__(kWave* kWavesPerBlock) void round_kernel(const float* __restrict__ x, uint32_t rows,
                                                                       uint32_t cols, uint32_t xs,
                                                                       int8_t* __restrict__ q, float* __restrict__ d) {
  const uint32_t nblk = cols / LANE_Q8_VALUES, blk = blockIdx.x * kWavesPerBlock + threadIdx.x / kWave;
  const int lane = threadIdx.x % kWave;
  if (blk >= rows * nblk) return;  // the whole wave
  const uint32_t r = blk / nblk, s = blk % nblk;
  const float v = x[(uint64_t) r * xs + (uint64_t) s * LANE_Q8_VALUES + lane];
  float a = fabsf(v);
  for (int o = kWave / 2; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, kWave));
  q[(uint64_t) r * cols + (uint64_t) s * LANE_Q8_VALUES + lane] = (int8_t) lane_q8_round(v, lane_q8_inv(a));
  if (lane == 0) d[(uint64_t) r * nblk + s] = lane_q8_scale(a);
}

}  // namespace q8
