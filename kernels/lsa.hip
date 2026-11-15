// latent sparse attention's read on the GPU (osmos/gpu/lsa_ops.h): src/lsa's lsa_read, list for list. a bucket's
// score its 64 dims in order in double, each product and sum rounded on its own (no fused step: the host's x86 sums
// are unfused) - the host's bits. the top LSA_TOP_P: a radix select over the scores' ordered bits (8 passes of 8, a
// histogram each, its bin chosen on the device: the read never waits for the host), the threshold's ties the
// lowest buckets (one block, in bucket order), the picks sorted
// by score then bucket (a bitonic sort in LDS); their keys gathered in pick order (a block-wide scan), cut at
// LSA_TOP_K; the window and the sinks after
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

#include "launch.h"
#include "osmos/gpu/lsa_ops.h"
#include "osmos/lsa.h"

namespace {

constexpr int kThreads = 1024;
static_assert(LSA_TOP_P == kThreads, "a pick a thread: the sort and the gather one block");

// a double's bits in an order whose ascending is the value's
__device__ inline uint64_t ordered(double v) {
  const uint64_t u = (uint64_t) __double_as_longlong(v);
  return u >> 63 ? ~u : u | (1ull << 63);
}

// a block 64 buckets, a thread one: their rows read coalesced into LDS (a thread a dim of each row), then each its own
// row's dot, its dims in order
__global__ __launch_bounds__(LSA_DIM) void scores_kernel(const float* __restrict__ cent, const float* __restrict__ iq,
                                                         uint32_t nb, uint64_t* __restrict__ key) {
  __shared__ float tile[LSA_DIM][LSA_DIM + 1];
  const uint32_t t = threadIdx.x, b0 = blockIdx.x * LSA_DIM;
  for (uint32_t i = 0; i < LSA_DIM; i++)
    tile[i][t] = b0 + i < nb ? cent[(uint64_t) (b0 + i) * LSA_DIM + t] : 0.0f;
  __syncthreads();
  if (b0 + t >= nb) return;
  double s = 0.0;
  for (int d = 0; d < LSA_DIM; d++) s = __dadd_rn(s, __dmul_rn((double) tile[t][d], (double) iq[d]));
  key[b0 + t] = ordered(s);
}

// the histogram of byte `shift` of the keys under the prefix (the bits above it chosen so far)
// the select's state: the bits chosen so far (prefix under mask), the keys still wanted among those under it
struct Sel {
  uint64_t prefix, mask;
  uint32_t want;
};

__global__ void sel_init_kernel(Sel* sel) { *sel = Sel{0, 0, LSA_TOP_P}; }

// a pass's bin: the highest whose keys, with those above it, reach the keys wanted; the histogram cleared for the next
__global__ void choose_kernel(uint32_t* __restrict__ hist, Sel* __restrict__ sel, int shift) {
  __shared__ uint32_t h[256];
  const uint32_t t = threadIdx.x;
  h[t] = hist[t];
  hist[t] = 0;
  __syncthreads();
  if (t) return;
  uint32_t above = 0;
  int bin = 255;
  for (; bin > 0 && above + h[bin] < sel->want; bin--) above += h[bin];
  sel->want -= above;
  sel->prefix |= (uint64_t) bin << shift, sel->mask |= 255ull << shift;
}

__global__ void hist_kernel(const uint64_t* __restrict__ key, uint32_t nb, const Sel* __restrict__ sel, int shift,
                            uint32_t* __restrict__ hist) {
  const uint64_t prefix = sel->prefix, mask = sel->mask;
  __shared__ uint32_t h[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) h[i] = 0;
  __syncthreads();
  for (uint32_t b = blockIdx.x * blockDim.x + threadIdx.x; b < nb; b += gridDim.x * blockDim.x) {
    const uint64_t k = key[b];
    if ((k & mask) == prefix) atomicAdd(&h[(k >> shift) & 255u], 1u);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 256; i += blockDim.x)
    if (h[i]) atomicAdd(&hist[i], h[i]);
}

// an exclusive scan of v over the block (kThreads), the total into *total
__device__ uint32_t block_scan(uint32_t v, uint32_t* lds, uint32_t* total) {
  const uint32_t t = threadIdx.x;
  lds[t] = v;
  __syncthreads();
  for (uint32_t o = 1; o < kThreads; o <<= 1) {
    const uint32_t x = t >= o ? lds[t - o] : 0;
    __syncthreads();
    lds[t] += x;
    __syncthreads();
  }
  const uint32_t incl = lds[t];
  *total = lds[kThreads - 1];
  __syncthreads();
  return incl - v;
}

// the picks, in three passes over the scores and a sort: a chunk a block (kChunk keys, kPer a thread, contiguous),
// its keys above the threshold and its ties counted; the chunks' offsets (one block's scan); each chunk's picks
// written at its offsets in bucket order (a tie taken while the ties before it, every chunk's, are fewer than the
// select's) - every key above the threshold and the lowest `ties` equal to it, LSA_TOP_P in all; then sorted
constexpr int kPer = 16, kChunkThreads = 256, kChunk = kPer * kChunkThreads, kMaxChunks = kThreads;

__global__ __launch_bounds__(kChunkThreads) void count_kernel(const uint64_t* __restrict__ key, uint32_t nb,
                                                              const Sel* __restrict__ sel, uint32_t* __restrict__ gt,
                                                              uint32_t* __restrict__ eq) {
  __shared__ uint32_t ng, ne;
  const uint64_t thr = sel->prefix;
  if (threadIdx.x == 0) ng = 0, ne = 0;
  __syncthreads();
  uint32_t g = 0, e = 0;
  for (int i = 0; i < kPer; i++) {
    const uint32_t b = blockIdx.x * kChunk + threadIdx.x * kPer + i;
    if (b < nb) g += key[b] > thr, e += key[b] == thr;
  }
  atomicAdd(&ng, g), atomicAdd(&ne, e);
  __syncthreads();
  if (threadIdx.x == 0) gt[blockIdx.x] = ng, eq[blockIdx.x] = ne;
}

__global__ __launch_bounds__(kThreads) void offsets_kernel(uint32_t* __restrict__ gt, uint32_t* __restrict__ eq,
                                                           uint32_t chunks) {
  __shared__ uint32_t lds[kThreads];
  uint32_t total;
  const uint32_t t = threadIdx.x;
  const uint32_t g = t < chunks ? gt[t] : 0, e = t < chunks ? eq[t] : 0;
  const uint32_t og = block_scan(g, lds, &total), oe = block_scan(e, lds, &total);
  if (t < chunks) gt[t] = og, eq[t] = oe;
}

__global__ __launch_bounds__(kChunkThreads) void write_kernel(const uint64_t* __restrict__ key, uint32_t nb,
                                                              const Sel* __restrict__ sel,
                                                              const uint32_t* __restrict__ gt0,
                                                              const uint32_t* __restrict__ eq0,
                                                              uint32_t* __restrict__ picks) {
  __shared__ uint32_t sg[kChunkThreads], se[kChunkThreads];
  const uint64_t thr = sel->prefix;
  const uint32_t ties = sel->want, above = LSA_TOP_P - ties, t = threadIdx.x;
  uint32_t g = 0, e = 0;
  for (int i = 0; i < kPer; i++) {
    const uint32_t b = blockIdx.x * kChunk + t * kPer + i;
    if (b < nb) g += key[b] > thr, e += key[b] == thr;
  }
  sg[t] = g, se[t] = e;  // the block's inclusive scans of both (a thread's keys contiguous: bucket order)
  __syncthreads();
  for (uint32_t o = 1; o < kChunkThreads; o <<= 1) {
    const uint32_t xg = t >= o ? sg[t - o] : 0, xe = t >= o ? se[t - o] : 0;
    __syncthreads();
    sg[t] += xg, se[t] += xe;
    __syncthreads();
  }
  uint32_t ag = gt0[blockIdx.x] + sg[t] - g, ae = eq0[blockIdx.x] + se[t] - e;
  for (int i = 0; i < kPer; i++) {
    const uint32_t b = blockIdx.x * kChunk + t * kPer + i;
    if (b >= nb) break;
    if (key[b] > thr) picks[ag++] = b;
    else if (key[b] == thr && ae < ties) picks[above + ae++] = b;
    else if (key[b] == thr) ae++;
  }
}

// the picks (LSA_TOP_P, or all `nb` when no more) by key descending, bucket ascending (bitonic in LDS, empties last)
__global__ __launch_bounds__(kThreads) void sort_kernel(const uint64_t* __restrict__ key, uint32_t nb, bool all,
                                                        uint32_t* __restrict__ picks) {
  __shared__ uint64_t sk[kThreads];
  __shared__ uint32_t sb[kThreads];
  const uint32_t t = threadIdx.x;
  const uint32_t b = all ? (t < nb ? t : UINT32_MAX) : picks[t];
  sb[t] = b, sk[t] = b != UINT32_MAX ? key[b] : 0;
  __syncthreads();
  for (uint32_t size = 2; size <= kThreads; size <<= 1)
    for (uint32_t j = size >> 1; j > 0; j >>= 1) {
      const uint32_t o = t ^ j;
      if (o > t) {
        const bool up = (t & size) == 0;
        const bool empty_t = sb[t] == UINT32_MAX, empty_o = sb[o] == UINT32_MAX;
        const bool first = !empty_t && (empty_o || sk[t] > sk[o] || (sk[t] == sk[o] && sb[t] < sb[o]));
        if (first != up) {
          const uint64_t k = sk[t];
          const uint32_t x = sb[t];
          sk[t] = sk[o], sb[t] = sb[o], sk[o] = k, sb[o] = x;
        }
      }
      __syncthreads();
    }
  picks[t] = sb[t];
}

// pick t's keys (from LSA_SINK on, before the window) at its place in pick order, the first LSA_TOP_K; then the window
// and the sinks; the count into *count
__global__ __launch_bounds__(kThreads) void gather_kernel(const uint32_t* __restrict__ picks,
                                                          const int32_t* __restrict__ members, uint32_t p, uint32_t win0,
                                                          uint32_t* __restrict__ out, uint32_t* __restrict__ count) {
  __shared__ uint32_t lds[kThreads];
  const uint32_t t = threadIdx.x, b = picks[t];
  uint32_t mine = 0;
  if (b != UINT32_MAX)
    for (int k = 0; k < LSA_CAP; k++) {
      const int32_t m = members[(uint64_t) b * LSA_CAP + k];
      mine += m >= LSA_SINK && (uint32_t) m < win0;
    }
  uint32_t total;
  uint32_t at = block_scan(mine, lds, &total);
  if (b != UINT32_MAX)
    for (int k = 0; k < LSA_CAP && at < LSA_TOP_K; k++) {
      const int32_t m = members[(uint64_t) b * LSA_CAP + k];
      if (m >= LSA_SINK && (uint32_t) m < win0) out[at++] = (uint32_t) m;
    }
  const uint32_t n = total < LSA_TOP_K ? total : LSA_TOP_K;
  const uint32_t win = p + 1 - win0, sinks = LSA_SINK < win0 ? LSA_SINK : win0 < p + 1 ? win0 : p + 1;
  for (uint32_t i = t; i < win; i += kThreads) out[n + i] = win0 + i;
  for (uint32_t i = t; i < sinks; i += kThreads) out[n + win + i] = i;
  if (t == 0) *count = n + win + sinks;
}

__device__ inline float bf16_round(float v) {  // to nearest, ties to even (src/lsa's)
  uint32_t u = __float_as_uint(v);
  if ((u & 0x7F800000u) != 0x7F800000u) u += 0x7FFFu + ((u >> 16) & 1u);
  return __uint_as_float(u & 0xFFFF0000u);
}

// a block a token, a thread a dim of the index: the input gathered (and a query's heads summed) into LDS, the dot in
// double over the input in order, the norm thread 0's (the dims in order), as lsa_index
__global__ __launch_bounds__(LSA_DIM) void index_kernel(const float* __restrict__ src, uint32_t row_stride,
                                                        uint32_t kv_heads, uint32_t group, uint32_t head_dim,
                                                        uint32_t rot, const float* __restrict__ W, bool bf16,
                                                        float* __restrict__ out) {
  __shared__ float x[LSA_IN];
  __shared__ double y[LSA_DIM];
  __shared__ double norm;
  const uint32_t t = blockIdx.x, d = threadIdx.x, free = head_dim - rot;
  const float* row = src + (uint64_t) t * row_stride;
  for (uint32_t i = d; i < LSA_IN; i += LSA_DIM) {
    const uint32_t h = i / free, j = i % free;
    float v = 0.0f;
    for (uint32_t g = 0; g < group; g++) v += row[(h * group + g) * head_dim + rot + j];
    x[i] = v;
  }
  __syncthreads();
  double s = 0.0;
  for (int i = 0; i < LSA_IN; i++) s = __dadd_rn(s, __dmul_rn((double) x[i], (double) W[(uint64_t) i * LSA_DIM + d]));
  y[d] = s;
  __syncthreads();
  if (d == 0) {
    double n = 0.0;
    for (int k = 0; k < LSA_DIM; k++) n = __dadd_rn(n, __dmul_rn(y[k], y[k]));
    n = sqrt(n);
    norm = n > 1e-12 ? n : 1e-12;
  }
  __syncthreads();
  const float v = (float) (y[d] / norm);
  out[(uint64_t) t * LSA_DIM + d] = bf16 ? bf16_round(v) : v;
}

}  // namespace

extern "C" bool lsa_index_gpu(const float* src, uint32_t n, uint32_t row_stride, uint32_t kv_heads, uint32_t group,
                              uint32_t head_dim, uint32_t rot, const float* W, bool bf16, float* out, void* stream) {
  if (!src || !W || !out || !n || !group || rot >= head_dim || kv_heads * (head_dim - rot) != LSA_IN) return false;
  index_kernel<<<n, LSA_DIM, 0, launches::st(stream)>>>(src, row_stride, kv_heads, group, head_dim, rot, W, bf16, out);
  return launches::ok();
}

struct LsaScratch {
  uint32_t max;
  uint64_t* key;
  uint32_t *hist, *picks, *gt, *eq; /* gt, eq: a chunk's counts, then its offsets */
  Sel* sel;
};

extern "C" LsaScratch* lsa_scratch_create(uint32_t max_buckets) {
  LsaScratch* s = (LsaScratch*) calloc(1, sizeof *s);
  if (!s) return nullptr;
  s->max = max_buckets;
  if (hipMalloc((void**) &s->key, (max_buckets ? max_buckets : 1) * 8ull) != hipSuccess ||
      hipMalloc((void**) &s->hist, 256 * 4) != hipSuccess || hipMalloc((void**) &s->picks, kThreads * 4) != hipSuccess ||
      hipMalloc((void**) &s->sel, sizeof(Sel)) != hipSuccess || hipMemset(s->hist, 0, 256 * 4) != hipSuccess ||
      (max_buckets + kChunk - 1) / kChunk > kMaxChunks || hipMalloc((void**) &s->gt, kMaxChunks * 4) != hipSuccess ||
      hipMalloc((void**) &s->eq, kMaxChunks * 4) != hipSuccess) {
    lsa_scratch_destroy(s);
    return nullptr;
  }
  return s;
}

extern "C" void lsa_scratch_destroy(LsaScratch* s) {
  if (!s) return;
  hipFree(s->key), hipFree(s->hist), hipFree(s->picks), hipFree(s->sel), hipFree(s->gt), hipFree(s->eq);
  free(s);
}

extern "C" bool lsa_read_gpu(const float* iq, uint32_t p, const float* cent, const int32_t* members, uint32_t blocks,
                             LsaScratch* s, uint32_t* out, uint32_t* n, void* stream) {
  const hipStream_t st = launches::st(stream);
  const uint32_t win0 = p + 1 > LSA_WINDOW ? p + 1 - LSA_WINDOW : 0;
  uint32_t nb = p / LSA_WINDOW * LSA_BUCKETS;
  nb = nb < blocks * LSA_BUCKETS ? nb : blocks * LSA_BUCKETS;
  if (!s || nb > s->max || !out || !n) return false;
  const bool all = nb <= LSA_TOP_P;
  if (nb) {
    scores_kernel<<<(nb + LSA_DIM - 1) / LSA_DIM, LSA_DIM, 0, st>>>(cent, iq, nb, s->key);
    if (!launches::ok()) return false;
  }
  if (nb && !all) {  // the K-th largest key, a byte at a time from the top, every step on the device
    const uint32_t grid = (nb + 255) / 256 < 512 ? (nb + 255) / 256 : 512;
    sel_init_kernel<<<1, 1, 0, st>>>(s->sel);
    for (int shift = 56; shift >= 0; shift -= 8) {
      hist_kernel<<<grid, 256, 0, st>>>(s->key, nb, s->sel, shift, s->hist);
      choose_kernel<<<1, 256, 0, st>>>(s->hist, s->sel, shift);
    }
    if (!launches::ok()) return false;
  }
  if (nb && !all) {
    const uint32_t chunks = (nb + kChunk - 1) / kChunk;
    count_kernel<<<chunks, kChunkThreads, 0, st>>>(s->key, nb, s->sel, s->gt, s->eq);
    offsets_kernel<<<1, kThreads, 0, st>>>(s->gt, s->eq, chunks);
    write_kernel<<<chunks, kChunkThreads, 0, st>>>(s->key, nb, s->sel, s->gt, s->eq, s->picks);
  }
  if (nb) {
    sort_kernel<<<1, kThreads, 0, st>>>(s->key, nb, all, s->picks);
  } else if (hipMemsetAsync(s->picks, 0xFF, kThreads * 4, st) != hipSuccess) {
    return false;
  }
  if (!launches::ok()) return false;
  gather_kernel<<<1, kThreads, 0, st>>>(s->picks, members, p, win0, out, n);
  return launches::ok();
}
