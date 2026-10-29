// ops (include/substrate/gpu/ops.h): the attention's keys - the compressor, the indexer's scores and its top k
// (sparse_attn.hip the attention itself) - each as ref.py computes it, in F32, reductions in a fixed order
#include <cmath>
#include <cstdint>

#include <hip/hip_runtime.h>

extern "C" {
#include "substrate/gpu/ops.h"
}

namespace {

constexpr int kThreads = 256;
constexpr int kWave = 32;
constexpr int kWaves = kThreads / kWave;
constexpr int kSelectThreads = 1024;
constexpr int kRadixBits = 8;
constexpr int kBins = 1 << kRadixBits;

hipStream_t st(void* s) { return (hipStream_t) s; }
bool ok() { return hipGetLastError() == hipSuccess; }

// a compressed row: sum over the ratio's rows of kvc * softmax(sc) (the softmax per value, over the rows)
__global__ void compress_kernel(const float* kvc, const float* sc, uint32_t ratio, uint32_t dim, float* out) {
  const uint32_t d = blockIdx.x * kThreads + threadIdx.x;
  if (d >= dim) return;
  float m = -INFINITY;
  for (uint32_t j = 0; j < ratio; j++) m = fmaxf(m, sc[(uint64_t) j * dim + d]);
  float e[OPS_MAX_RATIO], s = 0;
  for (uint32_t j = 0; j < ratio; j++) s += (e[j] = expf(sc[(uint64_t) j * dim + d] - m));
  float v = 0;
  for (uint32_t j = 0; j < ratio; j++) v += kvc[(uint64_t) j * dim + d] * (e[j] / s);
  out[d] = v;
}

// score[c] = sum over heads h of relu(q_h . k_c) * w_h, for the rows c < n: a wave a row, a lane a head
__global__ void index_score_kernel(const float* q, const float* keys, const float* w, uint32_t n, uint32_t heads,
                                   uint32_t hd, float* score) {
  extern __shared__ float qs[];
  for (uint32_t i = threadIdx.x; i < heads * hd; i += kThreads) qs[i] = q[i];
  __syncthreads();
  const int lane = threadIdx.x % kWave;
  const uint32_t stride = gridDim.x * kWaves;
  for (uint32_t c = blockIdx.x * kWaves + threadIdx.x / kWave; c < n; c += stride) {
    const float* k = keys + (uint64_t) c * hd;
    float v = 0;
    if ((uint32_t) lane < heads) {
      float s = 0;
      for (uint32_t d = 0; d < hd; d++) s += qs[lane * hd + d] * k[d];
      v = fmaxf(s, 0.0f) * w[lane];
    }
    for (int off = kWave / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, kWave);
    if (lane == 0) score[c] = v;
  }
}

// a float's order as an unsigned's
__device__ inline uint32_t order(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : u | 0x80000000u;
}

// the k highest of score[0 .. n) (ties to the lower row), their rows ascending into sel, how many into n_sel. one
// block: radix select of the k-th key (8 bits a pass, integer counts), then a compaction in row order
__global__ void topk_kernel(const float* score, uint32_t n, uint32_t k, int32_t* sel, int32_t* n_sel) {
  __shared__ uint32_t hist[kBins];
  __shared__ uint32_t prefix, mask, need;
  __shared__ uint32_t cnt_hi[kSelectThreads], cnt_eq[kSelectThreads];
  const uint32_t tid = threadIdx.x;
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

extern "C" bool ops_compress(const float* kvc, const float* sc, uint32_t ratio, uint32_t dim, float* out,
                             void* stream) {
  if (!ratio || ratio > OPS_MAX_RATIO || !dim) return false;
  compress_kernel<<<(dim + kThreads - 1) / kThreads, kThreads, 0, st(stream)>>>(kvc, sc, ratio, dim, out);
  return ok();
}

extern "C" bool ops_index_score(const float* q, const float* keys, const float* w, uint32_t n, uint32_t heads,
                                uint32_t head_dim, float* score, void* stream) {
  if (!n || !heads || heads > (uint32_t) kWave || !head_dim) return false;
  const uint32_t blocks = (n + kWaves - 1) / kWaves < 1024 ? (n + kWaves - 1) / kWaves : 1024;
  index_score_kernel<<<blocks, kThreads, heads * head_dim * sizeof(float), st(stream)>>>(q, keys, w, n, heads,
                                                                                          head_dim, score);
  return ok();
}

extern "C" bool ops_topk(const float* score, uint32_t n, uint32_t k, int32_t* sel, int32_t* n_sel, void* stream) {
  if (!n || !k || k > OPS_MAX_SELECT) return false;
  topk_kernel<<<1, kSelectThreads, 0, st(stream)>>>(score, n, k, sel, n_sel);
  return ok();
}
