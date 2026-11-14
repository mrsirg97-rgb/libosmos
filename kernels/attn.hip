// ops (include/osmos/gpu/ops.h): the attention's keys - the compressor, the indexer's scores and its top k
// (sparse_attn.hip the attention itself) - each as ref.py computes it, in F32, reductions in a fixed order
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include "launch.h"
#include <cstdint>

#include <hip/hip_runtime.h>

#include "kvload.h"

extern "C" {
#include "osmos/gpu/kvfmt.h"
#include "osmos/gpu/ops.h"
}

namespace {

using launches::st;
using launches::ok;

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kSelectThreads = 1024;
constexpr int kRadixBits = 8;
constexpr int kBins = 1 << kRadixBits;


// a token that completes a compressed row (its position + 1 a multiple of ratio): the row from the ratio positions
// ending at it (ring row position % slots of kvc and sc), out[t] = sum over them of kvc * softmax(sc), the softmax a
// value at a time; any other token's out[t] zeros
__global__ void compress_kernel(const float* kvc, const float* sc, uint32_t slots, const int32_t* pos, uint32_t ratio,
                                uint32_t dim, float* out) {
  const uint32_t d = blockIdx.x * kThreads + threadIdx.x, t = blockIdx.y;
  if (d >= dim) return;
  const int32_t p = pos[t];
  float* o = out + (uint64_t) t * dim + d;
  if ((uint32_t) (p + 1) % ratio) {
    *o = 0;
    return;
  }
  auto at = [&](const float* x, uint32_t j) {
    return x[(uint64_t) ((uint32_t) (p + 1 - ratio + j) % slots) * dim + d];
  };
  float m = -INFINITY;
  for (uint32_t j = 0; j < ratio; j++) m = fmaxf(m, at(sc, j));
  float e[OPS_MAX_RATIO], s = 0;
  for (uint32_t j = 0; j < ratio; j++) s += (e[j] = expf(at(sc, j) - m));
  float v = 0;
  for (uint32_t j = 0; j < ratio; j++) v += at(kvc, j) * (e[j] / s);
  *o = v;
}

// token t's row into dst: every 1, at row pos % slots; every r, only a token completing a row of r, at row
// ((pos + 1) / r - 1) % slots
// a row's 4-byte words (F32 values, or a packed row's bytes: kvfmt.h's rows are whole words)
__global__ void put_rows_kernel(uint32_t* dst, uint32_t slots, const uint32_t* src, const int32_t* pos, uint32_t every,
                                uint32_t dim) {
  const uint32_t t = blockIdx.y, p1 = (uint32_t) pos[t] + 1;
  if (p1 % every) return;
  const uint64_t row = (p1 / every - 1) % slots;
  for (uint32_t d = blockIdx.x * kThreads + threadIdx.x; d < dim; d += gridDim.x * kThreads)
    dst[row * dim + d] = src[(uint64_t) t * dim + d];
}

// a token's visible compressed rows: those whose ratio positions are all at or before its own
__device__ inline uint32_t visible(const int32_t* pos, uint32_t t, uint32_t ratio) {
  return (uint32_t) (pos[t] + 1) / ratio;
}

// token t's score[c] = sum over heads h of relu(q_h . k_c) * w_h, for its visible rows c - a product over a run's
// tokens and its rows (the run's tokens share their keys): a block kIxTok tokens x kIxHeads head slots by kIxRows
// rows, a thread kIxR of the slots by kIxC rows, d staged kIxStep at a time, the next stage's loads issued before this
// one's sums. each (token, head, row)'s dot s += q d * k d over d ascending (a multiply, then an add), relu * w, then
// the heads summed in the order a 32-lane xor tree sums them (lane i + lane i ^ off, off 16 .. 1: lane 0's) - the
// bits of a wave a row, a lane a head. the best of 1-4 tokens, 64-128 rows, 4-16 x 4-8 measured on gfx1151 (2,048
// tokens x 4,096 rows, 32 heads x 128: 47 ms a wave a row, 10.1 here; scratch ixb.hip)
constexpr int kIxHeads = 32, kIxTok = 2, kIxRows = 64, kIxR = 8, kIxC = 4, kIxStep = 16;
constexpr int kIxRcFp4 = 84;  // its thread's block (slots x rows): 8 x 4 (8 x 8 and 16 x 4: 15x slower, spilled)
constexpr int kIxTokFp4 = 4;  // ops_index_score_fp4's tile's tokens: 4 the fastest of 1, 2, 4 (ixbench)
typedef float f4 __attribute__((ext_vector_type(4)));
// GATHER (0.7.3, the candidate layers past 16K: ops_index_score_rows): a tile's rows a token's kept blocks' (list:
// block ids, n_list of them), not its first max_n - a token a tile (TOK 1), each (token, head, row) the same order of
// sums as the plain tile's, so a kept row's score the same bits; the rest unwritten (ops_topk masks them unread)
// FMA (0.7.3, ops_index_score_fp4): the sums' a + q x k one fused step - q and the keys FP4 values (E2M1 times a
// power of two: 2 bits of mantissa each), so q x k is exact and fma(q, k, a) rounds as round(q x k) + a does: the
// same bits in half the instructions. exact but where a product leaves the normal range: each (slot, row)'s
// smallest nonzero |q| x |k| under 2^-126 (a product under- or subnormal-rounded) or its largest at 2^127 (an fma's
// finite past a product's inf), its sum taken again the multiply-then-add way (counted into fallbacks)
template <int TOK, bool GATHER, bool FMA = false, int RR = kIxR, int CC = kIxC>
__global__ __launch_bounds__((kIxHeads * TOK / RR) * (kIxRows / CC)) void index_score_kernel(
    const float* __restrict__ q, const uint8_t* __restrict__ keys, uint32_t kfmt, const float* __restrict__ w,
    const int32_t* __restrict__ pos, uint32_t ratio, uint32_t heads, uint32_t hd, uint32_t n_tok,
    float* __restrict__ score, uint32_t stride, const uint32_t* __restrict__ list, const uint32_t* __restrict__ n_list,
    uint32_t list_stride, uint32_t block, uint32_t* __restrict__ fallbacks = nullptr) {
  constexpr int kIxM = kIxHeads * TOK, kIxThreads = (kIxM / RR) * (kIxRows / CC);
  constexpr int TX = kIxRows / CC, G = CC / 4, KS = kIxStep;
  constexpr int QL = kIxM * KS / 4 / kIxThreads, KL = kIxRows * KS / 4 / kIxThreads;  // a thread's float4s
  static_assert(QL >= 1 && KL >= 1 && RR % 4 == 0 && kIxHeads == kWave, "whole float4s; a lane a head");
  static_assert(!GATHER || TOK == 1, "a gathered tile one token's rows");
  constexpr int kStage = KS * (kIxM + 4) + KS * (kIxRows + 4), kSums = kIxM * (kIxRows + 1);
  __shared__ __attribute__((aligned(16))) float lds[kStage > kSums ? kStage : kSums];
  float(*qs)[kIxM + 4] = (float(*)[kIxM + 4]) lds;
  float(*ks)[kIxRows + 4] = (float(*)[kIxRows + 4]) (lds + KS * (kIxM + 4));
  const uint32_t t0 = blockIdx.y * TOK, c0 = blockIdx.x * kIxRows;
  uint32_t n_hi = 0;  // the most rows a token of the tile sees
  for (uint32_t t = t0; t < t0 + TOK && t < n_tok; t++) n_hi = max(n_hi, visible(pos, t, ratio));
  // a tile's row i: c0 + i, or (GATHER) the row of its token's kept list's entry; UINT32_MAX past them
  const uint32_t* lt = GATHER ? list + (uint64_t) t0 * list_stride : nullptr;
  const uint32_t nl = GATHER && t0 < n_tok ? n_list[t0] : 0;
  const auto row_of = [&](uint32_t i) -> uint32_t {
    if constexpr (GATHER) {
      const uint32_t e = (c0 + i) / block;
      const uint32_t r = e < nl ? lt[e] * block + (c0 + i) % block : UINT32_MAX;
      return r < n_hi ? r : UINT32_MAX;
    } else {
      return c0 + i < n_hi ? c0 + i : UINT32_MAX;
    }
  };
  if (GATHER ? c0 >= nl * block : c0 >= n_hi) return;  // the whole block
  const int tx = threadIdx.x % TX, ty = threadIdx.x / TX;
  const float* qp[QL];
  const uint8_t* kp[KL];  // a key row (kvfmt.h's, kfmt), read as kvload.h reads it
  int qi[QL], qd[QL], ki[KL], kd[KL];  // a load's slot (or row) and d in the stage
  for (int j = 0; j < QL; j++) {
    const int i = threadIdx.x + j * kIxThreads;
    qi[j] = i / (KS / 4), qd[j] = i % (KS / 4) * 4;
    const uint32_t t = t0 + qi[j] / kIxHeads, h = qi[j] % kIxHeads;
    qp[j] = t < n_tok && h < heads ? q + ((uint64_t) t * heads + h) * hd + qd[j] : nullptr;
  }
  for (int j = 0; j < KL; j++) {
    const int i = threadIdx.x + j * kIxThreads;
    ki[j] = i / (KS / 4), kd[j] = i % (KS / 4) * 4;
    const uint32_t r = row_of((uint32_t) ki[j]);
    kp[j] = r != UINT32_MAX ? keys + (uint64_t) r * kv_row_bytes((KvFmt) kfmt, hd) : nullptr;
  }
  f4 qn[QL];
  kv::Raw kn[KL];  // the keys' bytes in flight, decoded when staged
  for (int j = 0; j < QL; j++) qn[j] = qp[j] ? *(const f4*) qp[j] : f4{};
  for (int j = 0; j < KL; j++) kn[j] = kp[j] ? kv::fetch4(kfmt, kp[j], hd, kd[j]) : kv::Raw{};
  f4 acc[RR][G] = {};
  // (FMA) the guard's bounds: each slot's and row's smallest nonzero |value| and largest, as uint bits (positive
  // floats order as their bits)
  __shared__ uint32_t q_lo[FMA ? kIxM : 1], q_hi[FMA ? kIxM : 1], k_lo[FMA ? kIxRows : 1], k_hi[FMA ? kIxRows : 1];
  float qlo[QL], qhi[QL], klo[KL], khi[KL];
  if constexpr (FMA) {
    for (uint32_t i = threadIdx.x; i < (uint32_t) kIxM; i += kIxThreads) q_lo[i] = 0x7F800000u, q_hi[i] = 0;
    for (uint32_t i = threadIdx.x; i < (uint32_t) kIxRows; i += kIxThreads) k_lo[i] = 0x7F800000u, k_hi[i] = 0;
    for (int j = 0; j < QL; j++) qlo[j] = INFINITY, qhi[j] = 0.0f;
    for (int j = 0; j < KL; j++) klo[j] = INFINITY, khi[j] = 0.0f;
  }
  const auto bound = [](float& lo, float& hi, float v) {
    const float a = fabsf(v);
    lo = a > 0.0f ? fminf(lo, a) : lo, hi = fmaxf(hi, a);  // (a NaN: fmaxf keeps hi; its sum NaN either way)
  };
  for (uint32_t d0 = 0; d0 < hd; d0 += KS) {
    __syncthreads();
    for (int j = 0; j < QL; j++)
      for (int e = 0; e < 4; e++) {
        qs[qd[j] + e][qi[j]] = qn[j][e];
        if constexpr (FMA) bound(qlo[j], qhi[j], qn[j][e]);
      }
    for (int j = 0; j < KL; j++) {
      const f4 kj = kp[j] ? kv::decode4(kfmt, kn[j]) : f4{};
      for (int e = 0; e < 4; e++) {
        ks[kd[j] + e][ki[j]] = kj[e];
        if constexpr (FMA) bound(klo[j], khi[j], kj[e]);
      }
    }
    __syncthreads();
    if (d0 + KS < hd) {
      for (int j = 0; j < QL; j++) qn[j] = qp[j] ? *(const f4*) (qp[j] + d0 + KS) : f4{};
      for (int j = 0; j < KL; j++) kn[j] = kp[j] ? kv::fetch4(kfmt, kp[j], hd, kd[j] + d0 + KS) : kv::Raw{};
    }
#pragma unroll
    for (int d = 0; d < KS; d++) {
      f4 a[RR / 4], b[G];
      for (int i = 0; i < RR / 4; i++) a[i] = *(const f4*) &qs[d][ty * 4 + i * (kIxM / (RR / 4))];
      for (int g = 0; g < G; g++) b[g] = *(const f4*) &ks[d][tx * 4 + g * (kIxRows / G)];
      for (int i = 0; i < RR; i++) {
        const float av = a[i / 4][i % 4];
        if constexpr (FMA) {
          for (int g = 0; g < G; g++)
            for (int e = 0; e < 4; e++) acc[i][g][e] = __fmaf_rn(av, b[g][e], acc[i][g][e]);
        } else {
          for (int g = 0; g < G; g++) acc[i][g] = acc[i][g] + f4{av, av, av, av} * b[g];
        }
      }
    }
  }
  if constexpr (FMA) {
    for (int j = 0; j < QL; j++)
      atomicMin(&q_lo[qi[j]], __float_as_uint(qlo[j])), atomicMax(&q_hi[qi[j]], __float_as_uint(qhi[j]));
    for (int j = 0; j < KL; j++)
      atomicMin(&k_lo[ki[j]], __float_as_uint(klo[j])), atomicMax(&k_hi[ki[j]], __float_as_uint(khi[j]));
  }
  __syncthreads();
  float(*vs)[kIxRows + 1] = (float(*)[kIxRows + 1]) lds;  // [slot][row]: relu(s) * w, in the staging's room
  for (int i = 0; i < RR; i++) {
    const int m = ty * 4 + (i / 4) * (kIxM / (RR / 4)) + i % 4;
    const uint32_t t = t0 + m / kIxHeads, h = m % kIxHeads;
    const bool real = t < n_tok && h < heads;  // an empty slot: 0, as a lane past the heads
    const float wv = real ? w[(uint64_t) t * heads + h] : 0.0f;
    for (int g = 0; g < G; g++)
      for (int e = 0; e < 4; e++) {
        const int c = tx * 4 + g * (kIxRows / G) + e;
        float sdot = acc[i][g][e];
        if constexpr (FMA) {
          const float lo = __uint_as_float(q_lo[m]) * __uint_as_float(k_lo[c]);
          const float hi = __uint_as_float(q_hi[m]) * __uint_as_float(k_hi[c]);
          const uint32_t r = row_of((uint32_t) c);
          if (real && r != UINT32_MAX && (lo < 0x1p-126f || hi >= 0x1p127f)) {  // the multiply-then-add sum
            const float* qr = q + ((uint64_t) t * heads + h) * hd;
            const uint8_t* kr = keys + (uint64_t) r * kv_row_bytes((KvFmt) kfmt, hd);
            float sum = 0.0f;
            for (uint32_t d = 0; d < hd; d += 4) {
              const f4 kv4 = kv::load4(kfmt, kr, hd, d);
              for (int x = 0; x < 4; x++) sum = __fadd_rn(sum, __fmul_rn(qr[d + x], kv4[x]));
            }
            sdot = sum;
            if (fallbacks) atomicAdd(fallbacks, 1u);
          }
        }
        vs[m][c] = real ? fmaxf(sdot, 0.0f) * wv : 0.0f;
      }
  }
  __syncthreads();
  for (int o = threadIdx.x; o < TOK * kIxRows; o += kIxThreads) {
    const int tt = o / kIxRows, c = o % kIxRows;
    const uint32_t t = t0 + tt, r = GATHER ? row_of((uint32_t) c) : c0 + c;
    if (t >= n_tok || r == UINT32_MAX || r >= visible(pos, t, ratio)) continue;
    float x[kIxHeads];
    for (int h = 0; h < kIxHeads; h++) x[h] = vs[tt * kIxHeads + h][c];
    for (int off = kIxHeads / 2; off > 0; off >>= 1)
      for (int h = 0; h < off; h++) x[h] = x[h] + x[h + off];
    score[(uint64_t) t * stride + r] = x[0];
  }
}

// 0.7.4: the indexer's scores on the packed int8 dots (ops_index_score_dot4), ops_index_score's bits. q and the keys
// FP4 (a key E2M1 codes under a UE8M0 power of two a 32; q ops_fp4_round's F32 on that grid): a block's 32 values
// each I 2^E, I an integer of -12 .. 12 (a key's E its scale's - 1; q's the block's largest |value|'s exponent - 3,
// every value checked to be I 2^E), so a head's block dot an integer (|sum| <= 32 x 144: 8 v_dot4_i32_iu8 of 4 int8
// codes a word) times 2^(Eq + Ek) - a quarter of the fused kernel's LDS reads a product. index_score_kernel's own
// sum (d ascending, a multiply then an add in F32) is exact when the blocks whose q and key both hold a nonzero code
// have Eq + Ek within 9 of each other (a partial sum under 2^24 of the smallest one's unit) and every E within +-55
// (no product or sum leaves the normal range): then this sum, the blocks' integers times their powers added in F32
// (exact too), is its bits. an element outside those (a q block off the grid, a NaN scale, the spread past 9: rare,
// the keys RMS-normed) is summed index_score_kernel's way (counted into fallbacks). then relu x w and the heads'
// 32-lane xor tree, as there. a block 4 tokens x 32 heads by NT tiles of 64 rows, q staged once; a thread 2 tokens x
// heads j, j + 8, j + 16, j + 24 by 4 rows - the tree's first two levels its own adds, the last three lane shuffles
// (j a lane's low 3 bits)
constexpr int kDHd = 128, kDB = kDHd / 32, kDW = kDHd / 4, kDTok = 4, kDM = kDTok * kIxHeads, kDRows = 64;
constexpr int kDThreads = 256, kDSpread = 9, kDEmin = -55, kDEmax = 55;
constexpr int kDTiles = 8;  // a block's row tiles: 8 (512 rows) 66 ms at 1M, 4 71, 2 82, 1 94, 16 66 (scratch d4b)
typedef int i4 __attribute__((ext_vector_type(4)));
// 4 E2M1 codes (a byte each, low nibbles) as int8 2 x their values (0 1 2 3 4 6 8 12, signed): two byte permutes, a
// table each sign
__device__ inline uint32_t e2m1_int4(uint32_t x) {
  const uint32_t sel = x & 0x07070707u, neg = ((x >> 3) & 0x01010101u) * 0xFFu;
  const uint32_t p = __builtin_amdgcn_perm(0x0C080604u, 0x03020100u, sel);
  const uint32_t n = __builtin_amdgcn_perm(0xF4F8FAFCu, 0xFDFEFF00u, sel);
  return (p & ~neg) | (n & neg);
}
__device__ inline float pow2(int e) { return __uint_as_float((uint32_t) (e + 127) << 23); }  // e within +-126
// an element's exact-sum test: no bad scale, its both-coded blocks' Eq + Ek within kDSpread (g: 4 int8 E's a word,
// INT8_MIN a block with no nonzero code)
__device__ inline bool dot4_exact(uint32_t gq, uint32_t gk) {
  int lo = INT32_MAX, hi = INT32_MIN;
  for (int b = 0; b < kDB; b++) {
    const int eq = (int8_t) (gq >> (8 * b)), ek = (int8_t) (gk >> (8 * b));
    if (eq != INT8_MIN && ek != INT8_MIN) lo = min(lo, eq + ek), hi = max(hi, eq + ek);
  }
  return lo > hi || hi - lo <= kDSpread;
}
__device__ inline int dot4_span(uint32_t g) {  // a slot's (row's) coded blocks' E's spread
  int lo = INT32_MAX, hi = INT32_MIN;
  for (int b = 0; b < kDB; b++) {
    const int e = (int8_t) (g >> (8 * b));
    if (e != INT8_MIN) lo = min(lo, e), hi = max(hi, e);
  }
  return lo > hi ? 0 : hi - lo;
}
template <int NT>
__global__ __launch_bounds__(kDThreads) void index_score_dot4_kernel(
    const float* __restrict__ q, const uint8_t* __restrict__ keys, const float* __restrict__ w,
    const int32_t* __restrict__ pos, uint32_t ratio, uint32_t heads, uint32_t n_tok, float* __restrict__ score,
    uint32_t stride, uint32_t* __restrict__ fallbacks) {
  constexpr int kRowBytes = kDHd / 2 + kDB;  // 68
  // the codes 4 a word, word 2i the values 8i, 8i + 2, 8i + 4, 8i + 6 and word 2i + 1 the odd ones (a key byte's
  // low nibbles, its high: any order of d the same integer): q's word x of slot (token tt, head h) at
  // [x][(tt * 8 + h % 8) * 4 + h / 8] - a thread's 4 heads of a token one int4; the keys' at [x][row]
  __shared__ __attribute__((aligned(16))) int32_t qs[kDW][kDM];
  __shared__ __attribute__((aligned(16))) int32_t ks[kDW][kDRows];
  __shared__ __attribute__((aligned(16))) float sq[kDB][kDM], sk[kDB][kDRows];  // 2^E (a block with no code: 1)
  __shared__ uint32_t gq[kDM];                                                  // the E's (dot4_exact's)
  __shared__ __attribute__((aligned(4))) int8_t gk[kDRows][kDB];
  __shared__ uint8_t bq[kDM];
  __shared__ __attribute__((aligned(4))) uint8_t bk[kDRows][kDB];  // a scale outside the bounds (NaN's, any block)
  const uint32_t tid = threadIdx.x, t0 = blockIdx.y * kDTok;
  uint32_t n_hi = 0;  // the most rows a token of the tile sees
  for (uint32_t t = t0; t < t0 + kDTok && t < n_tok; t++) n_hi = max(n_hi, visible(pos, t, ratio));
  const uint32_t r00 = blockIdx.x * NT * kDRows;
  if (r00 >= n_hi) return;
  // the keys' staging: thread tid row tid % 64's block tid / 64 - its 16 bytes of codes and its scale
  const uint32_t sr = tid % kDRows, sb = tid / kDRows;
  uint32_t kraw[4], kscale = 0;
  const auto fetch = [&](uint32_t r0) {
    const uint32_t r = r0 + sr;
    if (r < n_hi) {
      const uint32_t* p = (const uint32_t*) (keys + (uint64_t) r * kRowBytes + sb * 16);
      for (int i = 0; i < 4; i++) kraw[i] = p[i];
      kscale = keys[(uint64_t) r * kRowBytes + kDHd / 2 + sb];
    } else {
      for (int i = 0; i < 4; i++) kraw[i] = 0;
      kscale = 127;
    }
  };
  fetch(r00);
  if (tid < (uint32_t) kDM) bq[tid] = 0;
  __syncthreads();
  // q's: two (slot, block)s a thread - 32 values, their largest |value| the exponent, each value checked I 2^E
  for (uint32_t p = tid; p < (uint32_t) (kDM * kDB); p += kDThreads) {
    const uint32_t s = p % kDM, b = p / kDM, tt = s / kIxHeads, h = s % kIxHeads, t = t0 + tt;
    const uint32_t at = (tt * 8 + h % 8) * 4 + h / 8;
    f4 v[8];
    if (t < n_tok && h < heads) {
      const f4* src = (const f4*) (q + ((uint64_t) t * heads + h) * kDHd + b * 32);
      for (int i = 0; i < 8; i++) v[i] = src[i];
    } else {
      for (int i = 0; i < 8; i++) v[i] = f4{};
    }
    float m = 0.0f;
    for (int i = 0; i < 8; i++)
      for (int e = 0; e < 4; e++) m = fmaxf(m, fabsf(v[i][e]));  // (a NaN: fmaxf drops it; the check below finds it)
    const uint32_t mx = (__float_as_uint(m) >> 23) & 0xFF;
    const int E = (int) mx - 127 - 3;
    bool bad = m != 0.0f && (mx == 0 || mx == 0xFF || E < kDEmin || E > kDEmax);
    const bool coded = m != 0.0f && !bad;
    const float inv = coded ? pow2(-E) : 0.0f, sc = coded ? pow2(E) : 1.0f;
    int32_t word[8] = {};
    for (int d = 0; d < 32; d++) {
      const float x = v[d / 4][d % 4], y = x * inv;
      const int I = (int) y;
      bad |= coded && ((float) I * sc != x || I > 12 || I < -12);
      bad |= !coded && x != x;  // (a NaN past an all-zero m)
      word[d / 8 * 2 + d % 2] |= (int32_t) ((uint32_t) (I & 0xFF) << (8 * (d % 8 / 2)));
    }
    for (int i = 0; i < 8; i++) qs[b * 8 + i][at] = bad ? 0 : word[i];
    sq[b][at] = bad ? 1.0f : sc;
    reinterpret_cast<int8_t*>(&gq[at])[b] = coded && !bad ? (int8_t) E : (int8_t) INT8_MIN;
    if (bad) bq[at] = 1;
  }
  const int lane = (int) (tid % kWave), wave = (int) (tid / kWave), j = lane % 8;
  const int tp = wave / 4, tx = (wave % 4) * 4 + lane / 8;  // tokens 2 tp, 2 tp + 1; rows tx * 4 ..
  const int qa = (2 * tp * 8 + j) * 4, qb = ((2 * tp + 1) * 8 + j) * 4;  // the thread's two int4s of q
  int qsp = 0;  // the thread's slots' spread (and a bad scale), the tiles' own
  bool q_bad = false;
  __syncthreads();
  for (int tt = 0; tt < 2; tt++)
    for (int a = 0; a < 4; a++) {
      const int at = (tt ? qb : qa) + a;
      qsp = max(qsp, dot4_span(gq[at])), q_bad |= bq[at] != 0;
    }
  for (int n = 0; n < NT; n++) {
    const uint32_t r0 = r00 + n * kDRows;
    if (r0 >= n_hi) break;
    if (n) __syncthreads();  // the last tile's reads done
    {  // the keys' codes and E
      bool any = false;
      for (int i = 0; i < 4; i++) any |= (kraw[i] & 0x77777777u) != 0;
      const int E = (int) kscale - 127 - 1;
      for (int i = 0; i < 4; i++) {
        ks[sb * 8 + 2 * i][sr] = (int32_t) e2m1_int4(kraw[i] & 0x0F0F0F0Fu);
        ks[sb * 8 + 2 * i + 1][sr] = (int32_t) e2m1_int4((kraw[i] >> 4) & 0x0F0F0F0Fu);
      }
      const bool bad = kscale == 0xFF || (any && (E < kDEmin || E > kDEmax));
      sk[sb][sr] = any && !bad ? pow2(E) : 1.0f;
      gk[sr][sb] = any && !bad ? (int8_t) E : (int8_t) INT8_MIN;
      bk[sr][sb] = bad;
    }
    __syncthreads();
    if (n + 1 < NT) fetch(r0 + kDRows);  // the next tile's keys in flight
    float acc[2][4][4] = {};  // [token][head j + 8a][row]
#pragma unroll
    for (int b = 0; b < kDB; b++) {
      int ia[2][4][4] = {};
#pragma unroll
      for (int x = 0; x < 8; x++) {
        const int wd = b * 8 + x;
        const i4 a0 = *(const i4*) &qs[wd][qa], a1 = *(const i4*) &qs[wd][qb], kb = *(const i4*) &ks[wd][tx * 4];
        for (int a = 0; a < 4; a++)
          for (int c = 0; c < 4; c++) {
            ia[0][a][c] = __builtin_amdgcn_sudot4(true, a0[a], true, kb[c], ia[0][a][c], false);
            ia[1][a][c] = __builtin_amdgcn_sudot4(true, a1[a], true, kb[c], ia[1][a][c], false);
          }
      }
      const f4 s0 = *(const f4*) &sq[b][qa], s1 = *(const f4*) &sq[b][qb], sc = *(const f4*) &sk[b][tx * 4];
      for (int a = 0; a < 4; a++)
        for (int c = 0; c < 4; c++) {  // an exact product, so the fma the multiply-then-add's bits
          acc[0][a][c] = __fmaf_rn((float) ia[0][a][c], s0[a] * sc[c], acc[0][a][c]);
          acc[1][a][c] = __fmaf_rn((float) ia[1][a][c], s1[a] * sc[c], acc[1][a][c]);
        }
    }
    // the guard: the thread's slots' and rows' spreads summed under kDSpread, else each element's own test
    int ksp = 0;
    bool any_bad = q_bad;
    for (int c = 0; c < 4; c++)
      ksp = max(ksp, dot4_span(*(const uint32_t*) gk[tx * 4 + c])), any_bad |= *(const uint32_t*) bk[tx * 4 + c] != 0;
    if (any_bad || qsp + ksp > kDSpread) {
      for (int tt = 0; tt < 2; tt++)
        for (int a = 0; a < 4; a++)
          for (int c = 0; c < 4; c++) {
            const int at = (tt ? qb : qa) + a, rc = tx * 4 + c;
            const uint32_t t = t0 + 2 * tp + tt, h = j + 8 * a, r = r0 + rc;
            if (t >= n_tok || h >= heads || r >= n_hi) continue;
            if (!bq[at] && !*(const uint32_t*) bk[rc] && dot4_exact(gq[at], *(const uint32_t*) gk[rc])) continue;
            const float* qr = q + ((uint64_t) t * heads + h) * kDHd;  // index_score_kernel's sum
            const uint8_t* kr = keys + (uint64_t) r * kRowBytes;
            float sum = 0.0f;
            for (uint32_t d = 0; d < (uint32_t) kDHd; d += 4) {
              const f4 kv4 = kv::load4(KV_FP4_POW2, kr, kDHd, d);
              for (int x = 0; x < 4; x++) sum = __fadd_rn(sum, __fmul_rn(qr[d + x], kv4[x]));
            }
            acc[tt][a][c] = sum;
            if (fallbacks) atomicAdd(fallbacks, 1u);
          }
    }
    // relu x w, then the heads' tree: + 16 and + 8 here (heads j + 8a), + 4, + 2, + 1 across the lanes
    for (int tt = 0; tt < 2; tt++) {
      const uint32_t t = t0 + 2 * tp + tt;
      float wv[4];
      for (int a = 0; a < 4; a++) {
        const uint32_t h = j + 8 * a;
        wv[a] = t < n_tok && h < heads ? w[(uint64_t) t * heads + h] : 0.0f;
      }
      for (int c = 0; c < 4; c++) {
        float v[4];
        for (int a = 0; a < 4; a++) {
          const uint32_t h = j + 8 * a;
          v[a] = t < n_tok && h < heads ? fmaxf(acc[tt][a][c], 0.0f) * wv[a] : 0.0f;
        }
        float z = (v[0] + v[2]) + (v[1] + v[3]);
        for (int off = 4; off > 0; off >>= 1) z = z + __shfl_xor(z, off, kWave);
        const uint32_t r = r0 + tx * 4 + c;
        if (j == 0 && t < n_tok && r < visible(pos, t, ratio)) score[(uint64_t) t * stride + r] = z;
      }
    }
  }
}

// a float's order as an unsigned's
__device__ inline uint32_t order(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : u | 0x80000000u;
}

// what a selection ranks: a token's rows by score (a row outside the kept blocks -inf: ops_topk under the candidate
// mask), or its blocks of rows by their highest score, its last block highest of all (ops_cand_blocks)
struct Pick {
  const float* score;
  uint32_t n;             // the rows visible
  const uint32_t* keep;   // the kept blocks' bits (rows; NULL: none masked)
  uint32_t block;
  const uint32_t* list;   // (0.7.3, ops_topk_rows) the kept blocks listed, ascending: the items their visible rows
  uint32_t n_list;
  const float* bmax;      // (0.7.3, ops_cand_blocks) each block's highest score, ops_block_max's (NULL: folded here)
  // the list's items: its blocks' rows below n - ascending rows, so only its last block can be partial
  __device__ uint32_t listed() const {
    if (!n_list) return 0;
    const uint32_t last = list[n_list - 1] * block;
    return (n_list - 1) * block + (n > last ? min(block, n - last) : 0);
  }
  __device__ uint32_t row(uint32_t i) const { return list ? list[i / block] * block + i % block : i; }
  template <bool BLOCKS> __device__ uint32_t count() const {
    return BLOCKS ? (n + block - 1) / block : list ? listed() : n;
  }
  template <bool BLOCKS> __device__ uint32_t key(uint32_t i) const {
    if (!BLOCKS && list) return order(score[row(i)]);  // a listed row: kept
    if constexpr (BLOCKS) {
      if (i + 1 == count<true>()) return order(INFINITY);
      if (bmax) return order(bmax[i]);  // (0.7.3) ops_block_max's: the same fold, once
      float m = -INFINITY;
      for (uint32_t c = i * block, e = c + block < n ? c + block : n; c < e; c++) m = fmaxf(m, score[c]);
      return order(m);
    } else {
      const uint32_t j = i / block;
      return keep && !((keep[j / 32] >> (j % 32)) & 1u) ? order(-INFINITY) : order(score[i]);
    }
  }
};

// a token's k highest of a Pick's items (ties to the lower index): rows ascending into sel and their count into n_sel,
// or blocks as bits into bits. a block a token: radix select of the k-th key (8 bits a pass, integer counts), then a
// compaction in index order
template <bool BLOCKS>
__global__ void select_kernel(const float* score, uint32_t stride, const int32_t* pos, uint32_t ratio, uint32_t k,
                              const uint32_t* keep, uint32_t block, uint32_t keep_words, int32_t* sel,
                              uint32_t sel_stride, int32_t* n_sel, uint32_t* bits, const uint32_t* list = nullptr,
                              const uint32_t* n_list = nullptr, uint32_t list_stride = 0, const float* bmax = nullptr,
                              uint32_t bstride = 0) {
  __shared__ uint32_t hist[kBins];
  __shared__ uint32_t prefix, mask, need;
  const uint32_t tid = threadIdx.x, t = blockIdx.x;
  const Pick p = {score + (uint64_t) t * stride, visible(pos, t, ratio), keep ? keep + (uint64_t) t * keep_words : nullptr,
                  block, list ? list + (uint64_t) t * list_stride : nullptr, list ? n_list[t] : 0,
                  bmax ? bmax + (uint64_t) t * bstride : nullptr};
  const uint32_t n = p.count<BLOCKS>();
  if constexpr (BLOCKS) {
    bits += (uint64_t) t * keep_words;
    for (uint32_t w = tid; w < keep_words; w += kSelectThreads) bits[w] = 0;
    __syncthreads();
  } else {
    sel += (uint64_t) t * sel_stride, n_sel += t;
  }
  const auto take = [&](uint32_t i) {
    if constexpr (BLOCKS) atomicOr(&bits[i / 32], 1u << (i % 32));
    else sel[i] = (int32_t) p.row(i);
  };
  if (n <= k) {  // every one
    for (uint32_t c = tid; c < n; c += kSelectThreads) take(c);
    if (!BLOCKS && tid == 0) *n_sel = (int32_t) n;
    return;
  }
  if (tid == 0) prefix = 0, mask = 0, need = k;
  __syncthreads();
  // 0.7.3: after the first pass, the keys of its bin (the k-th's: every later pass's candidates) gathered into LDS,
  // the later passes over them alone - their histograms the same counts - when they fit (past kCandCap: the full
  // scans, as before)
  constexpr uint32_t kCandCap = 8192;
  __shared__ uint32_t cand[kCandCap];
  __shared__ uint32_t n_cand;
  bool listed = false;
  for (int shift = 32 - kRadixBits; shift >= 0; shift -= kRadixBits) {
    for (uint32_t b = tid; b < kBins; b += kSelectThreads) hist[b] = 0;
    __syncthreads();
    if (listed) {
      for (uint32_t i = tid; i < n_cand; i += kSelectThreads) {
        const uint32_t u = cand[i];
        if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & (kBins - 1)], 1u);
      }
    } else {
      for (uint32_t c = tid; c < n; c += kSelectThreads) {
        const uint32_t u = p.key<BLOCKS>(c);
        if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & (kBins - 1)], 1u);
      }
    }
    __syncthreads();
    if (tid == 0) {  // the bin the need-th highest of the candidates falls in, from the top
      uint32_t acc = 0;
      for (int b = kBins - 1; b >= 0; b--) {
        if (acc + hist[b] >= need) {
          prefix |= (uint32_t) b << shift;
          mask |= (uint32_t) (kBins - 1) << shift;
          need -= acc;
          break;
        }
        acc += hist[b];
      }
    }
    __syncthreads();
    if (shift == 32 - kRadixBits && shift > 0) {  // the first pass done: its bin's keys gathered
      if (tid == 0) n_cand = 0;
      __syncthreads();
      for (uint32_t c = tid; c < n; c += kSelectThreads) {
        const uint32_t u = p.key<BLOCKS>(c);
        if ((u & mask) == prefix) {
          const uint32_t at = atomicAdd(&n_cand, 1u);
          if (at < kCandCap) cand[at] = u;
        }
      }
      __syncthreads();
      listed = n_cand <= kCandCap;
    }
  }
  const uint32_t kth = prefix;  // the k-th highest key; `need` of the ones equal to it to take
  if (!BLOCKS && tid == 0) *n_sel = (int32_t) k;
  // 0.7.3: one pass in index order, a tile of kSelectThreads items at a time (item base + tid: the loads coalesced) -
  // an item's higher and equal ones before it counted by its wave's ballots and the waves' prefix, the tiles' running
  // totals carried: taken when higher, or equal with fewer than `need` equal before it, into slot (higher before) +
  // min(equal before, need) - the rows (and ties) the per-thread runs took, in the same order
  constexpr int kSelWaves = kSelectThreads / kWave;
  __shared__ uint32_t w_hi[kSelWaves], w_eq[kSelWaves];
  const uint32_t lane = tid % kWave, wv = tid / kWave, below = (1u << lane) - 1u;
  uint32_t base_hi = 0, base_eq = 0;
  for (uint32_t base = 0; base < n; base += kSelectThreads) {
    const uint32_t c = base + tid;
    const uint32_t u = c < n ? p.key<BLOCKS>(c) : 0;
    const bool hi_ = c < n && u > kth, eq_ = c < n && u == kth;
    const uint32_t bh = (uint32_t) __ballot(hi_), be = (uint32_t) __ballot(eq_);
    if (lane == 0) w_hi[wv] = __popc(bh), w_eq[wv] = __popc(be);
    __syncthreads();
    uint32_t ph = base_hi + __popc(bh & below), pe = base_eq + __popc(be & below), th = 0, te = 0;
    for (uint32_t x = 0; x < (uint32_t) kSelWaves; x++) {
      if (x < wv) ph += w_hi[x], pe += w_eq[x];
      th += w_hi[x], te += w_eq[x];
    }
    if (hi_ || (eq_ && pe < need)) {
      if constexpr (BLOCKS) atomicOr(&bits[c / 32], 1u << (c % 32));
      else sel[ph + (pe < need ? pe : need)] = (int32_t) p.row(c);
    }
    base_hi += th, base_eq += te;
    __syncthreads();  // the waves' counts read before the next tile's
  }
}

}  // namespace

extern "C" bool ops_compress(const float* kvc, const float* sc, uint32_t slots, const int32_t* pos, uint32_t ratio,
                             uint32_t dim, uint32_t n_tok, float* out, void* stream) {
  if (!ratio || ratio > OPS_MAX_RATIO || slots < ratio || !dim || !n_tok) return false;
  compress_kernel<<<dim3((dim + kThreads - 1) / kThreads, n_tok), kThreads, 0, st(stream)>>>(kvc, sc, slots, pos,
                                                                                              ratio, dim, out);
  return ok();
}

extern "C" bool ops_put_rows(void* dst, uint32_t slots, const void* src, const int32_t* pos, uint32_t every,
                             uint32_t fmt, uint32_t vals, uint32_t n_tok, void* stream) {
  if (!slots || !every || !vals || !n_tok || fmt > KV_FP4_POW2 || vals % kv_block((KvFmt) fmt)) return false;
  const uint64_t row_bytes = kv_row_bytes((KvFmt) fmt, vals);
  if (row_bytes % 4) return false;
  const uint32_t dim = (uint32_t) (row_bytes / 4);
  put_rows_kernel<<<dim3((dim + kThreads - 1) / kThreads, n_tok), kThreads, 0, st(stream)>>>(
      (uint32_t*) dst, slots, (const uint32_t*) src, pos, every, dim);
  return ok();
}

extern "C" bool ops_index_score(const float* q, const void* keys, uint32_t keys_fmt, const float* w, const int32_t* pos,
                                uint32_t ratio, uint32_t max_n, uint32_t heads, uint32_t head_dim, uint32_t n_tok,
                                float* score, uint32_t stride, void* stream) {
  if (!ratio || !max_n || max_n > stride || !heads || heads > (uint32_t) kWave || !head_dim || !n_tok) return false;
  if (head_dim % kIxStep || keys_fmt > KV_FP4_POW2 || head_dim % kv_block((KvFmt) keys_fmt)) return false;
  constexpr int threads = (kIxHeads * kIxTok / kIxR) * (kIxRows / kIxC);
  index_score_kernel<kIxTok, false><<<dim3((max_n + kIxRows - 1) / kIxRows, (n_tok + kIxTok - 1) / kIxTok), threads, 0,
                                      st(stream)>>>(q, (const uint8_t*) keys, keys_fmt, w, pos, ratio, heads, head_dim,
                                                    n_tok, score, stride, nullptr, nullptr, 0, 1);
  return ok();
}

extern "C" bool ops_index_score_fp4(const float* q, const void* keys, const float* w, const int32_t* pos,
                                    uint32_t ratio, uint32_t max_n, uint32_t heads, uint32_t head_dim, uint32_t n_tok,
                                    float* score, uint32_t stride, uint32_t* fallbacks, void* stream) {
  if (!ratio || !max_n || max_n > stride || !heads || heads > (uint32_t) kWave || !head_dim || !n_tok) return false;
  if (head_dim % kIxStep || head_dim % kv_block(KV_FP4_POW2)) return false;
  constexpr int tok = kIxTokFp4;  // the tile's tokens: 4 the fastest of 1, 2, 4 (ixbench)
  // the thread's block: 8 slots x 4 rows (8 x 8 and 16 x 4 measured 15x slower - their accumulators spilled; ixbench)
  constexpr int rc = kIxRcFp4;
  const auto launch = [&](auto T, auto R, auto C) {
    constexpr int TK = decltype(T)::value, RK = decltype(R)::value, CK = decltype(C)::value;
    constexpr int threads = (kIxHeads * TK / RK) * (kIxRows / CK);
    index_score_kernel<TK, false, true, RK, CK><<<dim3((max_n + kIxRows - 1) / kIxRows, (n_tok + TK - 1) / TK),
                                                  threads, 0, st(stream)>>>(q, (const uint8_t*) keys, KV_FP4_POW2, w,
                                                                            pos, ratio, heads, head_dim, n_tok, score,
                                                                            stride, nullptr, nullptr, 0, 1, fallbacks);
  };
  using std::integral_constant;
  static_assert(rc == 84, "the thread's block: 8 x 4");
  const auto with_rc = [&](auto T) { launch(T, integral_constant<int, 8>{}, integral_constant<int, 4>{}); };
  with_rc(integral_constant<int, tok>{});
  return ok();
}

extern "C" bool ops_index_score_dot4(const float* q, const void* keys, const float* w, const int32_t* pos,
                                     uint32_t ratio, uint32_t max_n, uint32_t heads, uint32_t head_dim, uint32_t n_tok,
                                     float* score, uint32_t stride, uint32_t* fallbacks, void* stream) {
  if (!ratio || !max_n || max_n > stride || !heads || heads > (uint32_t) kWave || head_dim != (uint32_t) kDHd ||
      !n_tok)
    return false;
  constexpr int span = kDTiles * kDRows;
  index_score_dot4_kernel<kDTiles><<<dim3((max_n + span - 1) / span, (n_tok + kDTok - 1) / kDTok), kDThreads, 0,
                                     st(stream)>>>(q, (const uint8_t*) keys, w, pos, ratio, heads, n_tok, score,
                                                   stride, fallbacks);
  return ok();
}

extern "C" bool ops_index_score_rows(const float* q, const void* keys, uint32_t keys_fmt, const float* w,
                                     const int32_t* pos, uint32_t ratio, const uint32_t* list, const uint32_t* n_list,
                                     uint32_t list_stride, uint32_t block, uint32_t heads, uint32_t head_dim,
                                     uint32_t n_tok, float* score, uint32_t stride, void* stream) {
  if (!ratio || !list || !n_list || !list_stride || !block || kIxRows % block || !heads || heads > (uint32_t) kWave ||
      !head_dim || !n_tok)
    return false;
  if (head_dim % kIxStep || keys_fmt > KV_FP4_POW2 || head_dim % kv_block((KvFmt) keys_fmt)) return false;
  constexpr int threads = (kIxHeads * 1 / kIxR) * (kIxRows / kIxC);
  const uint32_t tiles = (uint32_t) (((uint64_t) list_stride * block + kIxRows - 1) / kIxRows);
  index_score_kernel<1, true, true><<<dim3(tiles, n_tok), threads, 0, st(stream)>>>(
      q, (const uint8_t*) keys, keys_fmt, w, pos, ratio, heads, head_dim, n_tok, score, stride, list, n_list,
      list_stride, block);
  return ok();
}

// a token's set bits (its candidate blocks) as their indices, ascending: a block a token, each thread a run of words,
// their counts' prefix in thread order
constexpr int kListThreads = 256;
__global__ __launch_bounds__(kListThreads) void cand_list_kernel(const uint32_t* bits, uint32_t words, uint32_t* list,
                                                                  uint32_t list_stride, uint32_t* n_list) {
  __shared__ uint32_t at[kListThreads];
  const uint32_t t = blockIdx.x, tid = threadIdx.x;
  const uint32_t* b = bits + (uint64_t) t * words;
  const uint32_t per = (words + kListThreads - 1) / kListThreads, lo = tid * per, hi = min(words, lo + per);
  uint32_t n = 0;
  for (uint32_t i = lo; i < hi; i++) n += __popc(b[i]);
  at[tid] = n;
  __syncthreads();
  if (tid == 0) {
    uint32_t a = 0;
    for (int i = 0; i < kListThreads; i++) {
      const uint32_t x = at[i];
      at[i] = a, a += x;
    }
    n_list[t] = min(a, list_stride);
  }
  __syncthreads();
  uint32_t o = at[tid];
  uint32_t* l = list + (uint64_t) t * list_stride;
  for (uint32_t i = lo; i < hi; i++)
    for (uint32_t x = b[i]; x; x &= x - 1)
      if (o < list_stride) l[o++] = i * 32 + (uint32_t) __ffs(x) - 1;
}

extern "C" bool ops_cand_list(const uint32_t* bits, uint32_t words, uint32_t n_tok, uint32_t* list,
                              uint32_t list_stride, uint32_t* n_list, void* stream) {
  if (!bits || !words || !n_tok || !list || !list_stride || !n_list) return false;
  cand_list_kernel<<<n_tok, kListThreads, 0, st(stream)>>>(bits, words, list, list_stride, n_list);
  return ok();
}

extern "C" bool ops_topk(const float* score, uint32_t stride, const int32_t* pos, uint32_t ratio, uint32_t k,
                         const uint32_t* keep, uint32_t block, uint32_t keep_words, uint32_t n_tok, int32_t* sel,
                         uint32_t sel_stride, int32_t* n_sel, void* stream) {
  if (!ratio || !k || k > OPS_MAX_SELECT || k > sel_stride || !n_tok ||
      (keep && (!block || keep_words < (stride / block + 31) / 32)))
    return false;
  select_kernel<false><<<n_tok, kSelectThreads, 0, st(stream)>>>(score, stride, pos, ratio, k, keep, block ? block : 1,
                                                                  keep_words, sel, sel_stride, n_sel, nullptr);
  return ok();
}

extern "C" bool ops_topk_rows(const float* score, uint32_t stride, const int32_t* pos, uint32_t ratio, uint32_t k,
                              const uint32_t* list, const uint32_t* n_list, uint32_t list_stride, uint32_t block,
                              uint32_t n_tok, int32_t* sel, uint32_t sel_stride, int32_t* n_sel, void* stream) {
  if (!ratio || !k || k > OPS_MAX_SELECT || k > sel_stride || !n_tok || !list || !n_list || !list_stride || !block)
    return false;
  select_kernel<false><<<n_tok, kSelectThreads, 0, st(stream)>>>(score, stride, pos, ratio, k, nullptr, block, 0, sel,
                                                                  sel_stride, n_sel, nullptr, list, n_list,
                                                                  list_stride);
  return ok();
}

extern "C" bool ops_cand_blocks(const float* score, uint32_t stride, const int32_t* pos, uint32_t ratio, uint32_t block,
                                uint32_t topk_blocks, uint32_t n_tok, uint32_t* keep, uint32_t keep_words, void* stream) {
  return ops_cand_blocks_max(score, stride, pos, ratio, block, topk_blocks, n_tok, keep, keep_words, nullptr, 0,
                             stream);
}

// (0.7.3) a token's blocks' highest scores: each block's visible rows folded as Pick's key folds them (from -inf,
// fmaxf in row order: a +0 and a -0 met in the same order), a thread a block
__global__ void block_max_kernel(const float* score, uint32_t stride, const int32_t* pos, uint32_t ratio,
                                 uint32_t block, float* bmax, uint32_t bstride) {
  const uint32_t t = blockIdx.y, n = visible(pos, t, ratio), nb = (n + block - 1) / block;
  const float* sc = score + (uint64_t) t * stride;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nb; i += gridDim.x * blockDim.x) {
    float m = -INFINITY;
    for (uint32_t c = i * block, e = c + block < n ? c + block : n; c < e; c++) m = fmaxf(m, sc[c]);
    bmax[(uint64_t) t * bstride + i] = m;
  }
}

extern "C" bool ops_cand_blocks_max(const float* score, uint32_t stride, const int32_t* pos, uint32_t ratio,
                                    uint32_t block, uint32_t topk_blocks, uint32_t n_tok, uint32_t* keep,
                                    uint32_t keep_words, float* bmax, uint32_t bstride, void* stream) {
  if (!ratio || !block || !topk_blocks || !n_tok || !keep || keep_words < ((stride + block - 1) / block + 31) / 32)
    return false;
  if (bmax && bstride < (stride + block - 1) / block) return false;
  if (bmax)
    block_max_kernel<<<dim3(64, n_tok), kThreads, 0, st(stream)>>>(score, stride, pos, ratio, block, bmax, bstride);
  select_kernel<true><<<n_tok, kSelectThreads, 0, st(stream)>>>(score, stride, pos, ratio, topk_blocks, nullptr, block,
                                                                 keep_words, nullptr, 0, nullptr, keep, nullptr,
                                                                 nullptr, 0, bmax, bstride);
  return ok();
}
