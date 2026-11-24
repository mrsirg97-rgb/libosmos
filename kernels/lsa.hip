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
#include "osmos/mem/ledger_hip.h"

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
                                                        LsaDims dims, uint32_t in, const float* __restrict__ W,
                                                        bool bf16, float* __restrict__ out) {
  __shared__ float x[LSA_IN_MAX];
  __shared__ double y[LSA_DIM];
  __shared__ double norm;
  const uint32_t t = blockIdx.x, d = threadIdx.x, n0 = dims.hi0 - dims.lo0, free = n0 + (dims.hi1 - dims.lo1);
  const float* row = src + (uint64_t) t * row_stride;
  for (uint32_t i = d; i < in; i += LSA_DIM) {
    const uint32_t h = i / free, j = i % free, dim = j < n0 ? dims.lo0 + j : dims.lo1 + (j - n0);
    float v = 0.0f;
    for (uint32_t g = 0; g < group; g++) v += row[(h * group + g) * head_dim + dim];
    x[i] = v;
  }
  __syncthreads();
  double s = 0.0;
  for (uint32_t i = 0; i < in; i++) s = __dadd_rn(s, __dmul_rn((double) x[i], (double) W[(uint64_t) i * LSA_DIM + d]));
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
                              uint32_t head_dim, const LsaDims* dims, const float* W, bool bf16, float* out,
                              void* stream) {
  if (!src || !W || !out || !n || !group || !dims || dims->lo0 > dims->hi0 || dims->lo1 > dims->hi1 ||
      dims->hi0 > head_dim || dims->hi1 > head_dim || (dims->hi1 > dims->lo1 && dims->lo1 < dims->hi0))
    return false;
  const uint32_t in = kv_heads * ((dims->hi0 - dims->lo0) + (dims->hi1 - dims->lo1));
  if (!in || in > LSA_IN_MAX) return false;
  index_kernel<<<n, LSA_DIM, 0, launches::st(stream)>>>(src, row_stride, kv_heads, group, head_dim, *dims, in, W, bf16,
                                                        out);
  return launches::ok();
}

struct LsaScratch {
  OsmosLedger* ledger;  // borrowed: its buffers counted in it
  uint32_t max;
  uint64_t* key;
  uint32_t *hist, *picks, *gt, *eq; /* gt, eq: a chunk's counts, then its offsets */
  Sel* sel;
};

extern "C" LsaScratch* lsa_scratch_create(OsmosLedger* l, uint32_t max_buckets) {
  LsaScratch* s = (LsaScratch*) calloc(1, sizeof *s);
  if (!s) return nullptr;
  s->max = max_buckets, s->ledger = l;
  if (ledger_malloc(l, (void**) &s->key, (max_buckets ? max_buckets : 1) * 8ull) != hipSuccess ||
      ledger_malloc(l, (void**) &s->hist, 256 * 4) != hipSuccess ||
      ledger_malloc(l, (void**) &s->picks, kThreads * 4) != hipSuccess ||
      ledger_malloc(l, (void**) &s->sel, sizeof(Sel)) != hipSuccess || hipMemset(s->hist, 0, 256 * 4) != hipSuccess ||
      (max_buckets + kChunk - 1) / kChunk > kMaxChunks || ledger_malloc(l, (void**) &s->gt, kMaxChunks * 4) != hipSuccess ||
      ledger_malloc(l, (void**) &s->eq, kMaxChunks * 4) != hipSuccess) {
    lsa_scratch_destroy(s);
    return nullptr;
  }
  return s;
}

extern "C" void lsa_scratch_destroy(LsaScratch* s) {
  if (!s) return;
  void* b[] = {s->key, s->hist, s->picks, s->sel, s->gt, s->eq};
  for (void* p : b)
    if (p) (void) ledger_free(s->ledger, p);
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

// a prompt's reads in a batch (lsa_read_batch_gpu): the scores of a tile of tokens x buckets (F32, dims in order, a
// product and a sum each rounded: the file's -ffp-contract=off), then a block a token - its top LSA_TOP_P by a radix
// select over the scores' ordered bits (4 passes of 8, histograms in LDS), the ties at the threshold the lowest
// buckets, the picks sorted by (score, then the lower bucket), their keys gathered as gather_kernel's
namespace {

constexpr int kBt = 32, kBb = 64;  // a scores tile: tokens x buckets (256 threads: a thread 2 tokens x 4 buckets)

__global__ __launch_bounds__(256) void batch_scores_kernel(const float* __restrict__ iq, uint32_t n, uint32_t p0,
                                                           const float* __restrict__ cent, uint32_t blocks,
                                                           uint32_t stride, float* __restrict__ out) {
  __shared__ float qs[kBt][LSA_DIM + 1], cs[kBb][LSA_DIM + 1];
  const uint32_t t0 = blockIdx.y * kBt, b0 = blockIdx.x * kBb, tid = threadIdx.x;
  const uint32_t last = t0 + kBt < n ? t0 + kBt : n;  // the tile's tokens: the visible buckets of its last the most
  uint32_t vis = (p0 + last - 1) / LSA_WINDOW;
  vis = (vis < blocks ? vis : blocks) * LSA_BUCKETS;
  if (b0 >= vis) return;
  for (uint32_t i = tid; i < kBt * LSA_DIM; i += 256) {
    const uint32_t t = i / LSA_DIM, d = i % LSA_DIM;
    qs[t][d] = t0 + t < n ? iq[(uint64_t) (t0 + t) * LSA_DIM + d] : 0.0f;
  }
  for (uint32_t i = tid; i < kBb * LSA_DIM; i += 256) {
    const uint32_t b = i / LSA_DIM, d = i % LSA_DIM;
    cs[b][d] = b0 + b < vis ? cent[(uint64_t) (b0 + b) * LSA_DIM + d] : 0.0f;
  }
  __syncthreads();
  const uint32_t tx = tid % 16, ty = tid / 16;
  float acc[2][4] = {};
  for (int d = 0; d < LSA_DIM; d++)
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
      for (int j = 0; j < 4; j++) acc[i][j] = acc[i][j] + cs[tx * 4 + j][d] * qs[ty * 2 + i][d];
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const uint32_t t = t0 + ty * 2 + i;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const uint32_t b = b0 + tx * 4 + j;
      if (t < n && b < vis) out[(uint64_t) t * stride + b] = acc[i][j];
    }
  }
}

// a float's bits in an order whose ascending is the value's
__device__ inline uint32_t ordered32(float v) {
  const uint32_t u = __float_as_uint(v);
  return u & 0x80000000u ? ~u : u | 0x80000000u;
}

__global__ __launch_bounds__(kThreads) void batch_select_kernel(const float* __restrict__ sc, uint32_t stride,
                                                                uint32_t n, uint32_t p0, uint32_t blocks,
                                                                const int32_t* __restrict__ members,
                                                                uint32_t* __restrict__ lists,
                                                                int32_t* __restrict__ counts) {
  __shared__ uint32_t hist[256], lds[kThreads];
  __shared__ uint64_t key[kThreads];
  __shared__ uint32_t s_prefix, s_want, s_above, s_ties;
  const uint32_t t = blockIdx.x, tid = threadIdx.x, p = p0 + t;
  const float* row = sc + (uint64_t) t * stride;
  const uint32_t win0 = p + 1 > LSA_WINDOW ? p + 1 - LSA_WINDOW : 0;
  uint32_t nb = p / LSA_WINDOW;
  nb = (nb < blocks ? nb : blocks) * LSA_BUCKETS;
  // the threshold: the LSA_TOP_P-th largest key, a byte at a time from the top
  if (tid == 0) s_prefix = 0, s_want = LSA_TOP_P;
  __syncthreads();
  const bool all = nb <= LSA_TOP_P;
  uint32_t mask = 0;
  for (int shift = 24; !all && shift >= 0; shift -= 8) {
    if (tid < 256) hist[tid] = 0;
    __syncthreads();
    const uint32_t prefix = s_prefix;
    for (uint32_t b = tid; b < nb; b += kThreads) {
      const uint32_t k = ordered32(row[b]);
      if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255], 1u);
    }
    __syncthreads();
    if (tid == 0) {
      uint32_t want = s_want, d = 255;
      for (;; d--) {
        if (hist[d] >= want || d == 0) break;
        want -= hist[d];
      }
      s_prefix = prefix | (d << shift), s_want = want;
    }
    __syncthreads();
    mask |= 255u << shift;
  }
  const uint32_t thr = s_prefix, ties = all ? 0 : s_want;  // ties: the threshold's keys taken, the lowest buckets
  // the picks: every key above the threshold, then the ties in bucket order (a block's scan a round of kThreads)
  if (tid == 0) s_above = 0, s_ties = 0;
  key[tid] = ~0ull;
  __syncthreads();
  const uint32_t above_n = all ? nb : LSA_TOP_P - ties;
  for (uint32_t base = 0; base < nb; base += kThreads) {
    const uint32_t b = base + tid;
    const uint32_t k = b < nb ? ordered32(row[b]) : 0;
    const bool gt = b < nb && (all || k > thr), eq = b < nb && !all && k == thr;
    if (gt) {
      const uint32_t at = atomicAdd(&s_above, 1u);
      key[at] = ((uint64_t) (~k) << 32) | b;  // ascending: score descending, then the lower bucket
    }
    uint32_t total;
    const uint32_t r = block_scan(eq ? 1u : 0u, lds, &total);
    const uint32_t taken = s_ties;
    if (eq && taken + r < ties) key[above_n + taken + r] = ((uint64_t) (~k) << 32) | b;
    __syncthreads();
    if (tid == 0) s_ties = taken + total;
    __syncthreads();
  }
  // sort ascending (bitonic, kThreads in LDS; the empties ~0 last)
  for (uint32_t size = 2; size <= kThreads; size <<= 1)
    for (uint32_t j = size >> 1; j > 0; j >>= 1) {
      const uint32_t o = tid ^ j;
      if (o > tid) {
        const bool up = (tid & size) == 0;
        if ((key[tid] > key[o]) == up) {
          const uint64_t x = key[tid];
          key[tid] = key[o], key[o] = x;
        }
      }
      __syncthreads();
    }
  // gather: a pick's keys (from LSA_SINK on, before the window) at its place in pick order, the first LSA_TOP_K
  const uint32_t pb = key[tid] == ~0ull ? UINT32_MAX : (uint32_t) key[tid];
  uint32_t* out = lists + (uint64_t) t * LSA_READ_MAX;
  uint32_t mine = 0;
  if (pb != UINT32_MAX)
    for (int k = 0; k < LSA_CAP; k++) {
      const int32_t m = members[(uint64_t) pb * LSA_CAP + k];
      mine += m >= LSA_SINK && (uint32_t) m < win0;
    }
  uint32_t total;
  uint32_t at = block_scan(mine, lds, &total);
  if (pb != UINT32_MAX)
    for (int k = 0; k < LSA_CAP && at < LSA_TOP_K; k++) {
      const int32_t m = members[(uint64_t) pb * LSA_CAP + k];
      if (m >= LSA_SINK && (uint32_t) m < win0) out[at++] = (uint32_t) m;
    }
  const uint32_t nk = total < LSA_TOP_K ? total : LSA_TOP_K;
  const uint32_t win = p + 1 - win0, sinks = LSA_SINK < win0 ? LSA_SINK : win0 < p + 1 ? win0 : p + 1;
  for (uint32_t i = tid; i < win; i += kThreads) out[nk + i] = win0 + i;
  for (uint32_t i = tid; i < sinks; i += kThreads) out[nk + win + i] = i;
  if (tid == 0) counts[t] = (int32_t) (nk + win + sinks);
}

}  // namespace

extern "C" uint64_t lsa_batch_scratch(uint32_t n_sub, uint32_t max_buckets) {
  return (uint64_t) n_sub * (max_buckets ? max_buckets : 1);
}

extern "C" bool lsa_read_batch_gpu(const float* iq, uint32_t n, uint32_t p0, const float* cent, const int32_t* members,
                                   uint32_t blocks, float* scratch, uint32_t n_sub, uint32_t max_buckets,
                                   uint32_t* lists, int32_t* counts, void* stream) {
  if (!iq || !n || !scratch || !n_sub || !lists || !counts || (blocks && (!cent || !members)) ||
      (uint64_t) blocks * LSA_BUCKETS > max_buckets)
    return false;
  const hipStream_t st = (hipStream_t) stream;
  const uint32_t stride = max_buckets ? max_buckets : 1;
  for (uint32_t t0 = 0; t0 < n; t0 += n_sub) {
    const uint32_t m = n - t0 < n_sub ? n - t0 : n_sub, p = p0 + t0;
    uint32_t vis = (p + m - 1) / LSA_WINDOW;
    vis = (vis < blocks ? vis : blocks) * LSA_BUCKETS;
    if (vis) {
      batch_scores_kernel<<<dim3((vis + kBb - 1) / kBb, (m + kBt - 1) / kBt), 256, 0, st>>>(
          iq + (uint64_t) t0 * LSA_DIM, m, p, cent, blocks, stride, scratch);
      if (!launches::ok()) return false;
    }
    batch_select_kernel<<<m, kThreads, 0, st>>>(scratch, stride, m, p, blocks, members,
                                                lists + (uint64_t) t0 * LSA_READ_MAX, counts + t0);
    if (!launches::ok()) return false;
  }
  return true;
}
