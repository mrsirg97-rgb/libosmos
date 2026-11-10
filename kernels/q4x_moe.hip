// Qwen3.8-Flash-Next's MoE (include/osmos/gpu/q4x_ops.h): its routing, the experts' tables and inputs, their
// activation and their sum with the shared expert's, F32
#include <climits>

#include "block.h"
#include "x16side.h"

extern "C" {
#include "osmos/gpu/mm.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/q4x_ops.h"
}

static_assert(Q4X_FF_TILE == mm::kTile, "the loader's expert-width rule (q4x_ops.h) is the matmul's tile");

namespace {

constexpr int kThreads = 256;
constexpr int kGroupThreads = 512; /* a step's grouping: a thread an expert (512 of them) */
constexpr uint32_t kMaxExperts = 1024; /* a token's probabilities in LDS */
constexpr uint32_t kMaxK = 32;
constexpr float kSumFloor = 6.103515625e-5f; /* llama.cpp's clamp of the picked weights' sum: f16's least normal */

inline unsigned grid(uint64_t n) { return (unsigned) ((n + kThreads - 1) / kThreads); }
inline hipStream_t st(void* s) { return (hipStream_t) s; }
inline bool ok() { return hipGetLastError() == hipSuccess; }

__device__ inline float silu(float x) { return x / (1.0f + expf(-x)); }

// a block a token: its softmax (ggml's: exp(x - max), times 1 / their sum, the block's trees), then the k greatest by
// its first wave alone - each lane its kRouteLane probabilities in registers (expert lane + 32 i), a pick the wave's
// argmax of those not yet taken by shuffles (the lowest id among equals), no barrier between picks: the same picks in
// the same order as a block-wide argmax a pick (0.5.11's, 8 barriers each), the same sum and weights. Qwen3.8's 512
// experts: 19.3 -> 13.9 us median a decode token (rocprofv3, low-power, 2026-11-09); every ids hash the same
constexpr uint32_t kRouteLane = kMaxExperts / 32; /* a lane's experts at most (n_exp <= kMaxExperts) */
__global__ void route_kernel(const float* logits, uint32_t n_exp, uint32_t k, int32_t* ids, float* w, int32_t* rows) {
  __shared__ float p[kMaxExperts], lds[kThreads];
  const uint32_t t = blockIdx.x;
  const float* l = logits + (uint64_t) t * n_exp;
  float m = -INFINITY;
  for (uint32_t i = threadIdx.x; i < n_exp; i += kThreads) m = fmaxf(m, l[i]);
  m = block_max<kThreads>(m, lds);
  float sum = 0;
  for (uint32_t i = threadIdx.x; i < n_exp; i += kThreads) sum += (p[i] = expf(l[i] - m));
  const float inv = 1.0f / block_sum<kThreads>(sum, lds);
  for (uint32_t i = threadIdx.x; i < n_exp; i += kThreads) p[i] *= inv;
  __syncthreads();
  if (threadIdx.x >= 32) return;
  const uint32_t lane = threadIdx.x;
  float v[kRouteLane];
  for (uint32_t r = 0; r < kRouteLane; r++) v[r] = lane + 32 * r < n_exp ? p[lane + 32 * r] : -INFINITY;
  uint32_t taken = 0; /* a lane's experts taken, bit r: expert lane + 32 r */
  float pick[kMaxK], total = 0;
  for (uint32_t j = 0; j < k; j++) {
    float bv = -INFINITY;
    int bi = INT_MAX;
    for (uint32_t r = 0; r < kRouteLane; r++) /* ascending ids: a strict > keeps the lowest among equals */
      if (!(taken >> r & 1) && lane + 32 * r < n_exp && v[r] > bv) bv = v[r], bi = (int) (lane + 32 * r);
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor(bv, o, 32);
      const int oi = __shfl_xor(bi, o, 32);
      if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    if ((uint32_t) bi % 32 == lane) taken |= 1u << ((uint32_t) bi / 32);
    pick[j] = bv, total += bv;
    if (!lane) ids[t * k + j] = bi, rows[t * k + j] = (int32_t) t;
  }
  if (lane) return;
  const float div = fmaxf(total, kSumFloor);
  for (uint32_t j = 0; j < k; j++) w[t * k + j] = pick[j] / div;
}

// a step's picks as mm_group's tables (q4x_expert_tiles): each expert's count, then scans of its places, its slots
// and its entries in each run, then a thread an expert lays its picks out in pick order and writes its parts - the
// same layout however it runs
constexpr int kPart = 128;  // an entry's most tokens: 8 tiles
// a step's picks at most this many: each expert's thread places its own picks itself (a scan of the step's picks, cheap
// when they are few: a decode token's 10, a draft's tens), not place_kernel - one launch, not two (decode 23.5 -> 23.8
// tok/s). the same tables either way (test_q4x_moe_gpu: a step of 30 picks and one of 2,560 against a stable sort)
constexpr uint32_t kPlaceInline = 256;
__device__ inline int run_of(int c) { return c <= 16 ? 0 : c <= 32 ? 1 : c <= 64 ? 2 : 3; }

__global__ __launch_bounds__(kGroupThreads) void tiles_kernel(const int32_t* ids, uint32_t n, uint32_t k,
                                                             uint32_t n_exp, uint32_t dim, uint32_t ff,
                                                             Q4xExpertTiles t, Q4xExpertBounds bd) {
  constexpr int T = MM_TILE, A = 2 + Q4X_TILE_RUNS;  // scanned: places, slots, each run's entries
  __shared__ int count[kMaxExperts], sc[A][kMaxExperts], live[A];
  for (uint32_t e = threadIdx.x; e < n_exp; e += kGroupThreads) count[e] = 0;
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < n; i += kGroupThreads) atomicAdd(&count[ids[i]], 1);
  __syncthreads();
  auto own = [&](uint32_t e, int a) {  // expert e's own share of scan a
    const int c = count[e];
    if (a == 0) return c;
    if (a == 1) return (c + T - 1) / T * T;
    const int r = a - 2, rem = c % kPart;
    return (r == Q4X_TILE_RUNS - 1 ? c / kPart : 0) + (rem && run_of(rem) == r);
  };
  for (uint32_t e = threadIdx.x; e < n_exp; e += kGroupThreads)
    for (int a = 0; a < A; a++) sc[a][e] = own(e, a);
  __syncthreads();
  for (uint32_t d = 1; d < n_exp; d <<= 1) {  // inclusive scans (Hillis-Steele), then each less its own
    int v[A][kMaxExperts / kGroupThreads];
    for (uint32_t e = threadIdx.x, j = 0; e < n_exp; e += kGroupThreads, j++)
      for (int a = 0; a < A; a++) v[a][j] = e >= d ? sc[a][e - d] : 0;
    __syncthreads();
    for (uint32_t e = threadIdx.x, j = 0; e < n_exp; e += kGroupThreads, j++)
      for (int a = 0; a < A; a++) sc[a][e] += v[a][j];
    __syncthreads();
  }
  if (threadIdx.x < A) live[threadIdx.x] = sc[threadIdx.x][n_exp - 1];
  __syncthreads();  // the totals read before the last expert's thread makes its own exclusive below
  for (uint32_t e = threadIdx.x; e < n_exp; e += kGroupThreads)
    for (int a = 0; a < A; a++) sc[a][e] -= own(e, a);
  __syncthreads();
  for (uint32_t e = threadIdx.x; e < n_exp; e += kGroupThreads) {
    const int c = count[e];
    t.off[e] = sc[0][e], t.off[n_exp + e] = sc[1][e], t.off[2 * n_exp + e] = c;  // place_kernel's
    if (!c) continue;
    const int p0 = sc[0][e], s0 = sc[1][e];
    if (n <= kPlaceInline) {  // the picks in pick order, this thread's expert's
      int q = p0;
      for (uint32_t i = 0; i < n && q < p0 + c; i++)
        if (ids[i] == (int32_t) e) t.where[i] = q, t.gidx[s0 + q - p0] = (int32_t) (i / k), t.hidx[s0 + q - p0] = q, q++;
    }
    for (int s = s0 + c; s < s0 + own(e, 1); s++) t.gidx[s] = -1, t.hidx[s] = -1;
    int at[Q4X_TILE_RUNS];
    for (int r = 0; r < Q4X_TILE_RUNS; r++) at[r] = (int) bd.first[r] + sc[2 + r][e];
    for (int part = 0; part * kPart < c; part++) {
      const int m = c - part * kPart < kPart ? c - part * kPart : kPart, en = at[run_of(m)]++;
      const int slot = s0 + part * kPart, place = p0 + part * kPart;
      t.cnt[en] = m;
      t.xg[en] = (uint32_t) slot * dim, t.xd[en] = (uint32_t) slot * ff;
      t.yg[en] = (uint32_t) place * ff, t.yd[en] = (uint32_t) place * dim;
      for (int j = 0; j < Q4X_GROUP_TENSORS; j++) t.ptr[j][en] = (const uint8_t*) t.base[j] + (uint64_t) e * t.bytes[j];
    }
  }
  for (int r = 0; r < Q4X_TILE_RUNS; r++)
    for (uint32_t i = (uint32_t) live[2 + r] + threadIdx.x; i < bd.count[r]; i += kGroupThreads) t.cnt[bd.first[r] + i] = 0;
  for (uint32_t s = (uint32_t) live[1] + threadIdx.x; s < bd.slots; s += kGroupThreads) t.gidx[s] = -1, t.hidx[s] = -1;
}

// a block an expert: its picks placed in pick order (the stable sort's) - the picks a block-wide tile at a time, each
// one's place its expert's first plus the expert's picks before it (an exclusive scan of the tile's flags)
__global__ __launch_bounds__(kThreads) void place_kernel(const int32_t* ids, uint32_t n, uint32_t k, uint32_t n_exp,
                                                         Q4xExpertTiles t) {
  __shared__ uint32_t scan[kThreads];
  const uint32_t e = blockIdx.x;
  const int c = t.off[2 * n_exp + e];
  if (!c) return;
  const int p0 = t.off[e], s0 = t.off[n_exp + e];
  int done = 0;
  for (uint32_t i0 = 0; i0 < n && done < c; i0 += kThreads) {
    const uint32_t i = i0 + threadIdx.x, f = i < n && ids[i] == (int32_t) e;
    scan[threadIdx.x] = f;
    __syncthreads();
    for (uint32_t o = 1; o < kThreads; o <<= 1) {  // inclusive scan (Hillis-Steele)
      const uint32_t v = threadIdx.x >= o ? scan[threadIdx.x - o] : 0;
      __syncthreads();
      scan[threadIdx.x] += v;
      __syncthreads();
    }
    if (f) {
      const int q = p0 + done + (int) scan[threadIdx.x] - 1;
      t.where[i] = q, t.gidx[s0 + q - p0] = (int32_t) (i / k), t.hidx[s0 + q - p0] = q;
    }
    done += (int) scan[kThreads - 1];
    __syncthreads();  // the tile's scan read by every thread before the next one's
  }
}

__global__ void gather_kernel(const float* src, const int32_t* rows, uint32_t n, uint32_t len, float* dst) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n * len) return;
  dst[i] = src[(uint64_t) rows[i / len] * len + i % len];
}

__global__ void swiglu_kernel(const float* g, const float* u, uint64_t n, float* y, Q4xX16 side) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i < n) y[i] = silu(g[i]) * u[i], x16_side(side, i, y[i]);
}

__global__ void moe_sum_kernel(const float* y, const int32_t* where, const float* w, uint32_t n_tok, uint32_t k,
                               uint32_t dim, const float* shared, const float* shared_logit, float* out) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * dim) return;
  const uint64_t t = i / dim, d = i % dim;
  float sum = 0;
  for (uint32_t j = 0; j < k; j++) sum += y[(uint64_t) where[t * k + j] * dim + d] * w[t * k + j];
  out[i] = sum + shared[i] * sigmoid(shared_logit[t]);
}

// moe_sum_kernel's, 4 adjacent values a thread: each of a token's k rows one 16-byte load, not 4 - each value the
// same products and sums in the same order, so its bits (a 2048-token step: ~140 GB/s, 10 dependent 4-byte loads a
// value)
constexpr uint32_t kSumVec = 4;
__global__ void moe_sum4_kernel(const float* y, const int32_t* where, const float* w, uint32_t n_tok, uint32_t k,
                                uint32_t dim, const float* shared, const float* shared_logit, float* out) {
  const uint64_t i = ((uint64_t) blockIdx.x * kThreads + threadIdx.x) * kSumVec;
  if (i >= (uint64_t) n_tok * dim) return;
  const uint64_t t = i / dim, d = i % dim;
  float4 sum = {0, 0, 0, 0};
  for (uint32_t j = 0; j < k; j++) {
    const float4 v = *(const float4*) (y + (uint64_t) where[t * k + j] * dim + d);
    const float wj = w[t * k + j];
    sum.x += v.x * wj, sum.y += v.y * wj, sum.z += v.z * wj, sum.w += v.w * wj;
  }
  const float g = sigmoid(shared_logit[t]);
  const float4 sh = *(const float4*) (shared + i);
  *(float4*) (out + i) = float4{sum.x + sh.x * g, sum.y + sh.y * g, sum.z + sh.z * g, sum.w + sh.w * g};
}

}  // namespace

extern "C" bool q4x_route(const float* logits, uint32_t n_tok, uint32_t n_exp, uint32_t k, int32_t* ids, float* w,
                          int32_t* rows, void* stream) {
  if (!logits || !ids || !w || !rows || !n_tok || !n_exp || n_exp > kMaxExperts || !k || k > kMaxK || k > n_exp)
    return false;
  route_kernel<<<n_tok, kThreads, 0, st(stream)>>>(logits, n_exp, k, ids, w, rows);
  return ok();
}

extern "C" void q4x_expert_bounds(uint32_t n, uint32_t n_exp, Q4xExpertBounds* b) {
  const uint32_t most = n < n_exp ? n : n_exp;
  const uint32_t cap[Q4X_TILE_RUNS] = {most, n / 17 < most ? n / 17 : most, n / 33 < most ? n / 33 : most, n / 65};
  b->entries = 0;
  for (int r = 0; r < Q4X_TILE_RUNS; r++) b->first[r] = b->entries, b->count[r] = cap[r], b->entries += cap[r];
  b->slots = n + (MM_TILE - 1) * most;
}

extern "C" bool q4x_expert_tiles(const int32_t* ids, uint32_t n, uint32_t k, uint32_t n_exp, uint32_t dim, uint32_t ff,
                                 const Q4xExpertTiles* t, void* stream) {
  if (!ids || !n || !k || n % k || !n_exp || n_exp > kMaxExperts || !dim || !ff || !t || !t->cnt || !t->gidx ||
      !t->hidx || !t->xg || !t->yg || !t->xd || !t->yd || !t->where || !t->off)
    return false;
  for (int j = 0; j < Q4X_GROUP_TENSORS; j++)
    if (!t->base[j] || !t->bytes[j] || !t->ptr[j]) return false;
  Q4xExpertBounds bd;
  q4x_expert_bounds(n, n_exp, &bd);
  tiles_kernel<<<1, kGroupThreads, 0, st(stream)>>>(ids, n, k, n_exp, dim, ff, *t, bd);
  if (!ok()) return false;
  if (n > kPlaceInline) place_kernel<<<n_exp, kThreads, 0, st(stream)>>>(ids, n, k, n_exp, *t);  // an expert a block
  return ok();
}

extern "C" bool q4x_gather(const float* src, const int32_t* rows, uint32_t n, uint32_t len, float* dst, void* stream) {
  if (!src || !rows || !dst || !n || !len) return false;
  gather_kernel<<<grid((uint64_t) n * len), kThreads, 0, st(stream)>>>(src, rows, n, len, dst);
  return ok();
}

// a thread a slot's value: swiglu_kernel's expression, then x16_rows_kernel's rounding (mm::x16_put)
__global__ void swiglu_rows_kernel(const float* g, const float* u, uint32_t ff, const int32_t* idx, uint32_t slots,
                                   _Float16* x16, int32_t* overflow) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) slots * ff) return;
  const uint32_t s = (uint32_t) (i / ff), c = (uint32_t) (i % ff);
  const int32_t r = idx[s];
  const uint64_t at = (uint64_t) (r >= 0 ? r : 0) * ff + c;
  mm::x16_put(x16, s, c, ff, r >= 0 ? silu(g[at]) * u[at] : 0.0f, overflow);
}

extern "C" bool q4x_swiglu_rows(const float* g, const float* u, uint32_t ff, const int32_t* idx, uint32_t slots,
                                void* x16, int32_t* overflow, void* stream) {
  if (!g || !u || !idx || !x16 || !overflow || !slots || !ff || ff % mm::kTile) return false;
  const uint64_t work = (uint64_t) slots * ff;
  swiglu_rows_kernel<<<(unsigned) ((work + kThreads - 1) / kThreads), kThreads, 0, st(stream)>>>(
      g, u, ff, idx, slots, (_Float16*) x16, overflow);
  return ok();
}

extern "C" bool q4x_swiglu(const float* g, const float* u, uint64_t n, float* y, const Q4xX16* side, void* stream) {
  if (!g || !u || !y || !n) return false;
  swiglu_kernel<<<grid(n), kThreads, 0, st(stream)>>>(g, u, n, y, side_of(side));
  return ok();
}

__global__ void spin_kernel(uint64_t ticks) {  // wall_clock64: a constant rate (clock64 is the shader clock's)
  const uint64_t t0 = wall_clock64();
  while (wall_clock64() - t0 < ticks) __builtin_amdgcn_s_sleep(8);
}

extern "C" bool q4x_spin(uint32_t us, void* stream) {
  static int khz = 0;  // wall_clock64's rate (hipDeviceAttributeWallClockRate, kHz): 0.5.10's spin counted the shader
                       // clock at 100 MHz's rate, ~20x short of `us`
  if (!khz && (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0) != hipSuccess || khz <= 0)) return false;
  spin_kernel<<<1, 32, 0, st(stream)>>>((uint64_t) us * (uint64_t) khz / 1000);
  return ok();
}

extern "C" bool q4x_moe_sum(const float* y, const int32_t* where, const float* w, uint32_t n_tok, uint32_t k,
                            uint32_t dim, const float* shared, const float* shared_logit, float* out, void* stream) {
  if (!y || !where || !w || !shared || !shared_logit || !out || !n_tok || !k || !dim) return false;
  const bool vec = dim % kSumVec == 0 && ((uintptr_t) y | (uintptr_t) shared | (uintptr_t) out) % 16 == 0;
  if (vec)
    moe_sum4_kernel<<<grid((uint64_t) n_tok * dim / kSumVec), kThreads, 0, st(stream)>>>(y, where, w, n_tok, k, dim,
                                                                                         shared, shared_logit, out);
  else
    moe_sum_kernel<<<grid((uint64_t) n_tok * dim), kThreads, 0, st(stream)>>>(y, where, w, n_tok, k, dim, shared,
                                                                               shared_logit, out);
  return ok();
}
