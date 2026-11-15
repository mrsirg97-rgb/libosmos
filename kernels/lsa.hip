// latent sparse attention's read on the GPU (osmos/gpu/lsa_ops.h): src/lsa's lsa_read, list for list. a bucket's
// score its 64 dims in order in double, each product and sum rounded on its own (no fused step: the host's x86 sums
// are unfused) - the host's bits. the top LSA_TOP_P: a radix select over the scores' ordered bits (8 passes of 8, a
// histogram each, read back), the threshold's ties the lowest buckets (one block, in bucket order), the picks sorted
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

__global__ void scores_kernel(const float* __restrict__ cent, const float* __restrict__ iq, uint32_t nb,
                              uint64_t* __restrict__ key) {
  const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nb) return;
  const float* c = cent + (uint64_t) b * LSA_DIM;
  double s = 0.0;
  for (int d = 0; d < LSA_DIM; d++) s = __dadd_rn(s, __dmul_rn((double) c[d], (double) iq[d]));
  key[b] = ordered(s);
}

// the histogram of byte `shift` of the keys under the prefix (the bits above it chosen so far)
__global__ void hist_kernel(const uint64_t* __restrict__ key, uint32_t nb, uint64_t prefix, uint64_t mask, int shift,
                            uint32_t* __restrict__ hist) {
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

// the picks in bucket order: every key above the threshold, and the first `ties` equal to it; then sorted by key
// descending, then bucket ascending (bitonic, kThreads slots, an empty slot last)
__global__ __launch_bounds__(kThreads) void pick_kernel(const uint64_t* __restrict__ key, uint32_t nb, uint64_t thr,
                                                        uint32_t ties, bool all, uint32_t* __restrict__ picks) {
  __shared__ uint32_t lds[kThreads];
  __shared__ uint64_t sk[kThreads];
  __shared__ uint32_t sb[kThreads];
  const uint32_t t = threadIdx.x;
  sk[t] = 0, sb[t] = UINT32_MAX;
  uint32_t base = 0, eq_base = 0;
  __syncthreads();
  for (uint32_t c0 = 0; c0 < nb; c0 += kThreads) {
    const uint32_t b = c0 + t;
    const uint64_t k = b < nb ? key[b] : 0;
    const bool eq = b < nb && !all && k == thr;
    uint32_t n_eq;
    const uint32_t before_eq = block_scan(eq ? 1u : 0u, lds, &n_eq);
    const bool take = b < nb && (all || k > thr || (eq && eq_base + before_eq < ties));
    uint32_t n_take;
    const uint32_t at = block_scan(take ? 1u : 0u, lds, &n_take);
    if (take && base + at < kThreads) sk[base + at] = k, sb[base + at] = b;
    base += n_take, eq_base += n_eq;
    __syncthreads();
  }
  for (uint32_t size = 2; size <= kThreads; size <<= 1)  // bitonic: key descending, bucket ascending, empties last
    for (uint32_t j = size >> 1; j > 0; j >>= 1) {
      const uint32_t o = t ^ j;
      if (o > t) {
        const bool up = (t & size) == 0;
        const bool empty_t = sb[t] == UINT32_MAX, empty_o = sb[o] == UINT32_MAX;
        // "t before o" in the final order
        const bool first = !empty_t && (empty_o || sk[t] > sk[o] || (sk[t] == sk[o] && sb[t] < sb[o]));
        if (first != up) {
          const uint64_t k = sk[t];
          const uint32_t b = sb[t];
          sk[t] = sk[o], sb[t] = sb[o], sk[o] = k, sb[o] = b;
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

}  // namespace

struct LsaScratch {
  uint32_t max;
  uint64_t* key;
  uint32_t *hist, *picks, *count;
  uint32_t* pin;  // the histogram and the count read back
};

extern "C" LsaScratch* lsa_scratch_create(uint32_t max_buckets) {
  LsaScratch* s = (LsaScratch*) calloc(1, sizeof *s);
  if (!s) return nullptr;
  s->max = max_buckets;
  if (hipMalloc((void**) &s->key, (max_buckets ? max_buckets : 1) * 8ull) != hipSuccess ||
      hipMalloc((void**) &s->hist, 256 * 4) != hipSuccess || hipMalloc((void**) &s->picks, kThreads * 4) != hipSuccess ||
      hipMalloc((void**) &s->count, 4) != hipSuccess || hipHostMalloc((void**) &s->pin, 257 * 4, 0) != hipSuccess) {
    lsa_scratch_destroy(s);
    return nullptr;
  }
  return s;
}

extern "C" void lsa_scratch_destroy(LsaScratch* s) {
  if (!s) return;
  hipFree(s->key), hipFree(s->hist), hipFree(s->picks), hipFree(s->count), hipHostFree(s->pin);
  free(s);
}

extern "C" bool lsa_read_gpu(const float* iq, uint32_t p, const float* cent, const int32_t* members, uint32_t blocks,
                             LsaScratch* s, uint32_t* out, uint32_t* n, void* stream) {
  const hipStream_t st = launches::st(stream);
  const uint32_t win0 = p + 1 > LSA_WINDOW ? p + 1 - LSA_WINDOW : 0;
  uint32_t nb = p / LSA_WINDOW * LSA_BUCKETS;
  nb = nb < blocks * LSA_BUCKETS ? nb : blocks * LSA_BUCKETS;
  if (!s || nb > s->max || !out || !n) return false;
  uint64_t thr = 0;
  uint32_t ties = 0;
  const bool all = nb <= LSA_TOP_P;
  if (nb) {
    scores_kernel<<<(nb + 255) / 256, 256, 0, st>>>(cent, iq, nb, s->key);
    if (!launches::ok()) return false;
  }
  if (nb && !all) {  // the K-th largest key, a byte at a time from the top
    uint64_t prefix = 0, mask = 0;
    uint32_t want = LSA_TOP_P;
    const uint32_t grid = (nb + 255) / 256 < 512 ? (nb + 255) / 256 : 512;
    for (int shift = 56; shift >= 0; shift -= 8) {
      if (hipMemsetAsync(s->hist, 0, 256 * 4, st) != hipSuccess) return false;
      hist_kernel<<<grid, 256, 0, st>>>(s->key, nb, prefix, mask, shift, s->hist);
      if (!launches::ok() || hipMemcpyAsync(s->pin, s->hist, 256 * 4, hipMemcpyDeviceToHost, st) != hipSuccess ||
          hipStreamSynchronize(st) != hipSuccess)
        return false;
      uint32_t above = 0;
      int bin = 255;
      for (; bin > 0 && above + s->pin[bin] < want; bin--) above += s->pin[bin];
      want -= above;  // of the bin's keys, the ones still wanted
      prefix |= (uint64_t) bin << shift, mask |= 255ull << shift;
    }
    thr = prefix, ties = want;
  }
  if (nb) {
    pick_kernel<<<1, kThreads, 0, st>>>(s->key, nb, thr, ties, all, s->picks);
  } else if (hipMemsetAsync(s->picks, 0xFF, kThreads * 4, st) != hipSuccess) {
    return false;
  }
  if (!launches::ok()) return false;
  gather_kernel<<<1, kThreads, 0, st>>>(s->picks, members, p, win0, out, s->count);
  return launches::ok() && hipMemcpyAsync(s->pin + 256, s->count, 4, hipMemcpyDeviceToHost, st) == hipSuccess &&
         hipStreamSynchronize(st) == hipSuccess && (*n = s->pin[256], true);
}
