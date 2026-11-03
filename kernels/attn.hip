// ops (include/substrate/gpu/ops.h): the attention's keys - the compressor, the indexer's scores and its top k
// (sparse_attn.hip the attention itself) - each as ref.py computes it, in F32, reductions in a fixed order
#include <cmath>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "kvload.h"

extern "C" {
#include "substrate/gpu/kvfmt.h"
#include "substrate/gpu/ops.h"
}

namespace {

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kSelectThreads = 1024;
constexpr int kRadixBits = 8;
constexpr int kBins = 1 << kRadixBits;

hipStream_t st(void* s) { return (hipStream_t) s; }
bool ok() { return hipGetLastError() == hipSuccess; }

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
constexpr int kIxM = kIxHeads * kIxTok, kIxThreads = (kIxM / kIxR) * (kIxRows / kIxC);
typedef float f4 __attribute__((ext_vector_type(4)));
__global__ __launch_bounds__(kIxThreads) void index_score_kernel(const float* __restrict__ q,
                                                                  const uint8_t* __restrict__ keys, uint32_t kfmt,
                                                                  const float* __restrict__ w,
                                                                  const int32_t* __restrict__ pos, uint32_t ratio,
                                                                  uint32_t heads, uint32_t hd, uint32_t n_tok,
                                                                  float* __restrict__ score, uint32_t stride) {
  constexpr int TX = kIxRows / kIxC, G = kIxC / 4, KS = kIxStep;
  constexpr int QL = kIxM * KS / 4 / kIxThreads, KL = kIxRows * KS / 4 / kIxThreads;  // a thread's float4s
  static_assert(QL >= 1 && KL >= 1 && kIxR % 4 == 0 && kIxHeads == kWave, "whole float4s; a lane a head");
  constexpr int kStage = KS * (kIxM + 4) + KS * (kIxRows + 4), kSums = kIxM * (kIxRows + 1);
  __shared__ __attribute__((aligned(16))) float lds[kStage > kSums ? kStage : kSums];
  float(*qs)[kIxM + 4] = (float(*)[kIxM + 4]) lds;
  float(*ks)[kIxRows + 4] = (float(*)[kIxRows + 4]) (lds + KS * (kIxM + 4));
  const uint32_t t0 = blockIdx.y * kIxTok, c0 = blockIdx.x * kIxRows;
  uint32_t n_hi = 0;  // the most rows a token of the tile sees
  for (uint32_t t = t0; t < t0 + kIxTok && t < n_tok; t++) n_hi = max(n_hi, visible(pos, t, ratio));
  if (c0 >= n_hi) return;  // the whole block
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
    kp[j] = c0 + ki[j] < n_hi ? keys + (uint64_t) (c0 + ki[j]) * kv_row_bytes((KvFmt) kfmt, hd) : nullptr;
  }
  f4 qn[QL];
  kv::Raw kn[KL];  // the keys' bytes in flight, decoded when staged
  for (int j = 0; j < QL; j++) qn[j] = qp[j] ? *(const f4*) qp[j] : f4{};
  for (int j = 0; j < KL; j++) kn[j] = kp[j] ? kv::fetch4(kfmt, kp[j], hd, kd[j]) : kv::Raw{};
  f4 acc[kIxR][G] = {};
  for (uint32_t d0 = 0; d0 < hd; d0 += KS) {
    __syncthreads();
    for (int j = 0; j < QL; j++)
      for (int e = 0; e < 4; e++) qs[qd[j] + e][qi[j]] = qn[j][e];
    for (int j = 0; j < KL; j++) {
      const f4 kj = kp[j] ? kv::decode4(kfmt, kn[j]) : f4{};
      for (int e = 0; e < 4; e++) ks[kd[j] + e][ki[j]] = kj[e];
    }
    __syncthreads();
    if (d0 + KS < hd) {
      for (int j = 0; j < QL; j++) qn[j] = qp[j] ? *(const f4*) (qp[j] + d0 + KS) : f4{};
      for (int j = 0; j < KL; j++) kn[j] = kp[j] ? kv::fetch4(kfmt, kp[j], hd, kd[j] + d0 + KS) : kv::Raw{};
    }
#pragma unroll
    for (int d = 0; d < KS; d++) {
      f4 a[kIxR / 4], b[G];
      for (int i = 0; i < kIxR / 4; i++) a[i] = *(const f4*) &qs[d][ty * 4 + i * (kIxM / (kIxR / 4))];
      for (int g = 0; g < G; g++) b[g] = *(const f4*) &ks[d][tx * 4 + g * (kIxRows / G)];
      for (int i = 0; i < kIxR; i++) {
        const float av = a[i / 4][i % 4];
        for (int g = 0; g < G; g++) acc[i][g] = acc[i][g] + f4{av, av, av, av} * b[g];
      }
    }
  }
  __syncthreads();
  float(*vs)[kIxRows + 1] = (float(*)[kIxRows + 1]) lds;  // [slot][row]: relu(s) * w, in the staging's room
  for (int i = 0; i < kIxR; i++) {
    const int m = ty * 4 + (i / 4) * (kIxM / (kIxR / 4)) + i % 4;
    const uint32_t t = t0 + m / kIxHeads, h = m % kIxHeads;
    const bool real = t < n_tok && h < heads;  // an empty slot: 0, as a lane past the heads
    const float wv = real ? w[(uint64_t) t * heads + h] : 0.0f;
    for (int g = 0; g < G; g++)
      for (int e = 0; e < 4; e++) vs[m][tx * 4 + g * (kIxRows / G) + e] = real ? fmaxf(acc[i][g][e], 0.0f) * wv : 0.0f;
  }
  __syncthreads();
  for (int o = threadIdx.x; o < kIxTok * kIxRows; o += kIxThreads) {
    const int tt = o / kIxRows, c = o % kIxRows;
    const uint32_t t = t0 + tt;
    if (t >= n_tok || c0 + c >= visible(pos, t, ratio)) continue;
    float x[kIxHeads];
    for (int h = 0; h < kIxHeads; h++) x[h] = vs[tt * kIxHeads + h][c];
    for (int off = kIxHeads / 2; off > 0; off >>= 1)
      for (int h = 0; h < off; h++) x[h] = x[h] + x[h + off];
    score[(uint64_t) t * stride + c0 + c] = x[0];
  }
}

// a float's order as an unsigned's
__device__ inline uint32_t order(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : u | 0x80000000u;
}

// a token's k highest of score[0 .. n) (ties to the lower row), their rows ascending into sel, how many into n_sel. a
// block a token: radix select of the k-th key (8 bits a pass, integer counts), then a compaction in row order
__global__ void topk_kernel(const float* score, uint32_t stride, const int32_t* pos, uint32_t ratio, uint32_t k,
                            int32_t* sel, uint32_t sel_stride, int32_t* n_sel) {
  __shared__ uint32_t hist[kBins];
  __shared__ uint32_t prefix, mask, need;
  __shared__ uint32_t cnt_hi[kSelectThreads], cnt_eq[kSelectThreads];
  const uint32_t tid = threadIdx.x, t = blockIdx.x, n = visible(pos, t, ratio);
  score += (uint64_t) t * stride, sel += (uint64_t) t * sel_stride, n_sel += t;
  if (n <= k) {  // every row
    for (uint32_t c = tid; c < n; c += kSelectThreads) sel[c] = (int32_t) c;
    if (tid == 0) *n_sel = (int32_t) n;
    return;
  }
  if (tid == 0) prefix = 0, mask = 0, need = k;
  __syncthreads();
  for (int shift = 32 - kRadixBits; shift >= 0; shift -= kRadixBits) {
    for (uint32_t b = tid; b < kBins; b += kSelectThreads) hist[b] = 0;
    __syncthreads();
    for (uint32_t c = tid; c < n; c += kSelectThreads) {
      const uint32_t u = order(score[c]);
      if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & (kBins - 1)], 1u);
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
  }
  const uint32_t kth = prefix;  // the k-th highest key; `need` of the rows equal to it to take
  const uint32_t per = (n + kSelectThreads - 1) / kSelectThreads, lo = tid * per, hi = lo + per < n ? lo + per : n;
  uint32_t h_ = 0, e_ = 0;
  for (uint32_t c = lo; c < hi; c++) {
    const uint32_t u = order(score[c]);
    h_ += u > kth, e_ += u == kth;
  }
  cnt_hi[tid] = h_, cnt_eq[tid] = e_;
  __syncthreads();
  if (tid == 0) {  // exclusive prefixes, in row order: the equal ones taken lowest rows first
    uint32_t ah = 0, ae = 0;
    for (uint32_t i = 0; i < kSelectThreads; i++) {
      const uint32_t th = cnt_hi[i], te = cnt_eq[i];
      cnt_hi[i] = ah, cnt_eq[i] = ae;
      ah += th, ae += te;
    }
    *n_sel = (int32_t) k;
  }
  __syncthreads();
  uint32_t out = cnt_hi[tid] + (cnt_eq[tid] < need ? cnt_eq[tid] : need), eq = cnt_eq[tid];
  for (uint32_t c = lo; c < hi; c++) {
    const uint32_t u = order(score[c]);
    if (u > kth || (u == kth && eq++ < need)) sel[out++] = (int32_t) c;
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
  index_score_kernel<<<dim3((max_n + kIxRows - 1) / kIxRows, (n_tok + kIxTok - 1) / kIxTok), kIxThreads, 0,
                       st(stream)>>>(q, (const uint8_t*) keys, keys_fmt, w, pos, ratio, heads, head_dim, n_tok, score,
                                     stride);
  return ok();
}

extern "C" bool ops_topk(const float* score, uint32_t stride, const int32_t* pos, uint32_t ratio, uint32_t k,
                         uint32_t n_tok, int32_t* sel, uint32_t sel_stride, int32_t* n_sel, void* stream) {
  if (!ratio || !k || k > OPS_MAX_SELECT || k > sel_stride || !n_tok) return false;
  topk_kernel<<<n_tok, kSelectThreads, 0, st(stream)>>>(score, stride, pos, ratio, k, sel, sel_stride, n_sel);
  return ok();
}
