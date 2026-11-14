// the indexers' selection (0.7.4: DeepSeek's and Qwen's one shape): a radix select of the k-th key, then the taken
// emitted in item order. select_core's callers: a block a token, or (a small step) kChunks blocks a token and a merge
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

extern "C" {
#include "osmos/gpu/ops.h"
}

namespace sel {

constexpr int kWave = 32;
constexpr int kSelectThreads = 1024;
constexpr int kRadixBits = 8;
constexpr int kBins = 1 << kRadixBits;

// a selection's body (0.7.4: one shape for every caller): the k highest of n items by key(i) - ties to the lower item -
// each taken item emitted as emit(i, slot), slot its place among the taken in item order; returns how many (n <= k:
// all). a block's kSelectThreads threads: the radix passes (8 bits each, integer counts; after the first, its bin's
// keys gathered into LDS when they fit), then one coalesced pass in item order (the waves' ballots, their prefix)
template <typename Key, typename Emit>
__device__ inline uint32_t select_core(uint32_t n, uint32_t k, const Key& key, const Emit& emit) {
  __shared__ uint32_t hist[kBins];
  __shared__ uint32_t prefix, mask, need;
  const uint32_t tid = threadIdx.x;
  if (n <= k) {  // every one
    for (uint32_t c = tid; c < n; c += kSelectThreads) emit(c, c);
    return n;
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
        const uint32_t u = key(c);
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
        const uint32_t u = key(c);
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
  // 0.7.3: one pass in item order, a tile of kSelectThreads items at a time (item base + tid: the loads coalesced) -
  // an item's higher and equal ones before it counted by its wave's ballots and the waves' prefix, the tiles' running
  // totals carried: taken when higher, or equal with fewer than `need` equal before it, into slot (higher before) +
  // min(equal before, need)
  constexpr int kSelWaves = kSelectThreads / kWave;
  __shared__ uint32_t w_hi[kSelWaves], w_eq[kSelWaves];
  const uint32_t lane = tid % kWave, wv = tid / kWave, below = (1u << lane) - 1u;
  uint32_t base_hi = 0, base_eq = 0;
  for (uint32_t base = 0; base < n; base += kSelectThreads) {
    const uint32_t c = base + tid;
    const uint32_t u = c < n ? key(c) : 0;
    const bool hi_ = c < n && u > kth, eq_ = c < n && u == kth;
    const uint32_t bh = (uint32_t) __ballot(hi_), be = (uint32_t) __ballot(eq_);
    if (lane == 0) w_hi[wv] = __popc(bh), w_eq[wv] = __popc(be);
    __syncthreads();
    uint32_t ph = base_hi + __popc(bh & below), pe = base_eq + __popc(be & below), th = 0, te = 0;
    for (uint32_t x = 0; x < (uint32_t) kSelWaves; x++) {
      if (x < wv) ph += w_hi[x], pe += w_eq[x];
      th += w_hi[x], te += w_eq[x];
    }
    if (hi_ || (eq_ && pe < need)) emit(c, ph + (pe < need ? pe : need));
    base_hi += th, base_eq += te;
    __syncthreads();  // the waves' counts read before the next tile's
  }
  return k;
}

// 0.7.4: a small step's selection on many blocks - a token's items cut into kChunks chunks, each chunk's k highest
// (select_core: the same order - key, then the lower item) written with their keys in item order (chunk_kernel), then
// the k highest of those (merge_kernel). every item of the whole's k highest is among its chunk's k highest, and the
// candidates are in item order, so the same items, emitted as the one-block selection emits them. the chunks' scratch
// one per library (a TU's): a chunked selection's two kernels on one stream, never two at once (DeepSeek's selections
// in a layer's order on its stream; Qwen's pick on the side stream joined before the next layer's)
constexpr uint32_t kChunks = 32, kChunkTok = 16, kChunkK = OPS_MAX_SELECT > 2048 ? OPS_MAX_SELECT : 2048;
static __device__ uint32_t g_ckey[kChunkTok * kChunks * kChunkK], g_citem[kChunkTok * kChunks * kChunkK];
static __device__ uint32_t g_ccount[kChunkTok * kChunks];
// (host) the chunked selection: a small step (decode's, a draft's verify) over items enough that a block a token idles the
// GPU - kChunks chunks each well above k. OSMOS_SELECT_ONE=1 keeps the one-block selection (A/B)
inline bool chunked(uint32_t n_tok, uint32_t items, uint32_t k) {
  static const int one = [] { const char* e = getenv("OSMOS_SELECT_ONE"); return e && *e == '1'; }();
  return !one && n_tok <= kChunkTok && k <= kChunkK && items >= 4u * kChunks * k;
}

// candidate i's slot in a token's chunk scratch, from its chunks' offsets (off[m]: the candidates before chunk m)
__device__ inline uint32_t chunk_slot(const uint32_t* off, uint32_t i) {
  uint32_t lo = 0, hi = kChunks;  // the chunk m with off[m] <= i < off[m + 1]
  while (hi - lo > 1) {
    const uint32_t mid = (lo + hi) / 2;
    if (off[mid] <= i) lo = mid;
    else hi = mid;
  }
  return lo * kChunkK + (i - off[lo]);
}

}  // namespace sel
