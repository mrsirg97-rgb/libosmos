// QSA's indexer (include/osmos/gpu/q4x_ops.h, q4x_ix_*; DESIGN.md, "QSA"): its query's norm and rotation, the keys'
// blocks pooled, a token's scores against them and its top blocks - llama.cpp's build_qsa_sel in its order mode
#include "block.h"
#include "launch.h"
#include "rope.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

namespace {

using launches::st;
using launches::ok;

constexpr int kDim = Q4X_IX_DIM;  // a head's dim: a block's threads, a value a thread
constexpr int kMaxHeads = Q4X_IX_HEADS_MAX; // the score tile's query rows (attention.indexer.head_count)
constexpr int kTT = 16, kTB = 32; // the score tile: tokens x blocks
constexpr int kPick = 256;        // the pick's threads: a token's
constexpr uint64_t kScoreBudget = 4u << 20;  // the scratch's scores at least: a piece's tokens x their blocks


// a block of kDim threads, x its value d: x normed (its L2 over the block) times w, then its first rope.dim rotated in
// pairs (i, i + dim / 2) at position pos - q4x_qk's rotation (rope_cs: the running product, YaRN's where it has it)
__device__ float norm_rope(float x, const float* w, float eps, float pos, const Q4xRope& rope) {
  __shared__ float lds[kDim], y[kDim];
  const uint32_t d = threadIdx.x;
  const float r = 1.0f / sqrtf(block_sum<kDim>(x * x, lds) / (float) kDim + eps);
  y[d] = x * r * w[d];
  __syncthreads();
  float v = y[d];
  const uint32_t half = rope.dim / 2;
  if (d < rope.dim) {
    const uint32_t i = d % half;
    float c, s;
    rope_cs(rope, pos, i, &c, &s);
    const float x0 = y[i], x1 = y[i + half];
    v = d < half ? x0 * c - x1 * s : x0 * s + x1 * c;
  }
  return v;
}

// a block a (head, token)
__global__ __launch_bounds__(kDim) void ix_q_kernel(const float* x, uint32_t heads, Q4xRope rope, const float* w,
                                                    float eps, int32_t pos0, float* q) {
  const uint64_t i = ((uint64_t) blockIdx.y * heads + blockIdx.x) * kDim + threadIdx.x;
  q[i] = norm_rope(x[i], w, eps, (float) (pos0 + (int32_t) blockIdx.y), rope);
}

// a block a pooled block j0 + blockIdx.x: its members summed in order, times 1 / ratio, normed and rotated at its first
__global__ __launch_bounds__(kDim) void ix_pool_kernel(const float* raw, int32_t pos0, uint32_t ratio,
                                                       const float* ring, uint32_t ring_rows, uint32_t j0,
                                                       Q4xRope rope, const float* w, float eps, float* pooled) {
  const uint32_t j = j0 + blockIdx.x, d = threadIdx.x;
  float sum = 0.0f;
  for (uint32_t m = 0; m < ratio; m++) {
    const int32_t p = (int32_t) (ratio * j + m);
    sum += p < pos0 ? ring[(uint64_t) (p % ring_rows) * kDim + d] : raw[(uint64_t) (p - pos0) * kDim + d];
  }
  const float mean = sum * (1.0f / (float) ratio);
  pooled[(uint64_t) j * kDim + d] = norm_rope(mean, w, eps, (float) (ratio * j), rope);
}

// a block a step's token t0 + blockIdx.x: its raw key into the ring
__global__ __launch_bounds__(kDim) void ix_ring_kernel(const float* raw, int32_t pos0, uint32_t t0, float* ring,
                                                       uint32_t ring_rows) {
  const uint32_t t = t0 + blockIdx.x;
  ring[(uint64_t) ((uint32_t) (pos0 + (int32_t) t) % ring_rows) * kDim + threadIdx.x] = raw[(uint64_t) t * kDim + threadIdx.x];
}

// a tile of kTT tokens x kTB blocks: each (token, visible block) score - the heads in order, each its dot product's
// dims in order, relu, times 1 / sqrt(kDim) - the same sums whatever tile or piece holds it
__global__ __launch_bounds__(256) void ix_score_kernel(const float* q, const float* pooled, uint32_t m, uint32_t heads,
                                                       int32_t p0, uint32_t ratio, uint32_t stride, float* score) {
  __shared__ float qs[kTT][kMaxHeads * kDim + 1], ks[kTB][kDim];
  const uint32_t tt = blockIdx.y * kTT, tb = blockIdx.x * kTB;
  const uint32_t t_last = tt + kTT - 1 < m ? tt + kTT - 1 : m - 1;
  const uint32_t nv_last = (uint32_t) (p0 + (int32_t) t_last + 1) / ratio;
  if (tb >= nv_last) return;  // past every visible block of the tile's tokens
  const uint32_t qw = heads * kDim;
  for (uint32_t i = threadIdx.x; i < kTT * qw; i += 256) {
    const uint32_t r = i / qw, c = i % qw;
    qs[r][c] = tt + r < m ? q[(uint64_t) (tt + r) * qw + c] : 0.0f;
  }
  for (uint32_t i = threadIdx.x; i < kTB * kDim; i += 256) {
    const uint32_t r = i / kDim, d = i % kDim;
    ks[r][d] = tb + r < nv_last ? pooled[(uint64_t) (tb + r) * kDim + d] : 0.0f;
  }
  __syncthreads();
  const float scale = 1.0f / sqrtf((float) kDim);
  const uint32_t tl = threadIdx.x % kTT, t = tt + tl;
  if (t >= m) return;
  const uint32_t nv = (uint32_t) (p0 + (int32_t) t + 1) / ratio;
  for (uint32_t jl = threadIdx.x / kTT; jl < kTB; jl += 256 / kTT) {
    const uint32_t j = tb + jl;
    if (j >= nv) continue;
    float s = 0.0f;
    for (uint32_t h = 0; h < heads; h++) {
      float dot = 0.0f;
      for (int d = 0; d < kDim; d++) dot += qs[tl][h * kDim + d] * ks[jl][d];
      s += fmaxf(dot, 0.0f) * scale;
    }
    score[(uint64_t) t * stride + j] = s;
  }
}

// an exclusive prefix sum over the block's kPick threads, and its total
__device__ uint32_t scan_ex(uint32_t v, uint32_t* lds, uint32_t* total) {
  const uint32_t i = threadIdx.x;
  lds[i] = v;
  __syncthreads();
  for (uint32_t o = 1; o < kPick; o <<= 1) {
    const uint32_t a = i >= o ? lds[i - o] : 0;
    __syncthreads();
    lds[i] += a;
    __syncthreads();
  }
  const uint32_t inc = lds[i];
  *total = lds[kPick - 1];
  __syncthreads();
  return inc - v;
}

// a block a token: with more visible blocks than `blocks`, its top `blocks` by score (a radix select on the scores'
// bits, non-negative floats ordered as their bits; the lower block among equals) written in block order
__global__ __launch_bounds__(kPick) void ix_pick_kernel(const float* score, int32_t p0, uint32_t ratio,
                                                        uint32_t blocks, uint32_t stride, int32_t* sel,
                                                        int32_t* npick) {
  __shared__ uint32_t hist[256], scan[kPick], found[2];
  const uint32_t t = blockIdx.x, nv = (uint32_t) (p0 + (int32_t) t + 1) / ratio;
  if (nv <= blocks) {
    if (threadIdx.x == 0) npick[t] = (int32_t) nv;
    return;
  }
  const uint32_t* key = (const uint32_t*) (score + (uint64_t) t * stride);
  uint32_t prefix = 0, mask = 0, remaining = blocks;  // the keys above prefix taken; remaining still to take
  for (int shift = 24; shift >= 0; shift -= 8) {
    hist[threadIdx.x] = 0;
    __syncthreads();
    for (uint32_t j = threadIdx.x; j < nv; j += kPick)
      if ((key[j] & mask) == prefix) atomicAdd(&hist[(key[j] >> shift) & 255], 1u);
    __syncthreads();
    if (threadIdx.x == 0) {
      uint32_t above = 0;
      for (int b = 255; b >= 0; b--) {
        if (above + hist[b] >= remaining) {
          found[0] = prefix | ((uint32_t) b << shift), found[1] = remaining - above;
          break;
        }
        above += hist[b];
      }
    }
    __syncthreads();
    prefix = found[0], remaining = found[1], mask |= 255u << shift;
    __syncthreads();
  }
  // prefix: the blocks-th greatest key; every key above it taken, and the first `remaining` equal to it
  uint32_t out = 0, eq_seen = 0;
  int32_t* row = sel + (uint64_t) t * blocks;
  for (uint32_t base = 0; base < nv; base += kPick) {
    const uint32_t j = base + threadIdx.x, k = j < nv ? key[j] : 0;
    const uint32_t gt = j < nv && k > prefix, eq = j < nv && k == prefix;
    uint32_t n_eq, n_take;
    const uint32_t eq_before = scan_ex(eq, scan, &n_eq);
    const uint32_t take = gt || (eq && eq_seen + eq_before < remaining);
    const uint32_t at = scan_ex(take, scan, &n_take);
    if (take) row[out + at] = (int32_t) j;
    out += n_take, eq_seen += n_eq;
  }
  if (threadIdx.x == 0) npick[t] = (int32_t) blocks;
}

}  // namespace

// a rotation the indexer's head takes: its dims within the head, pairs whole
static bool rope_ok(const Q4xRope* r) { return r && r->dim && r->dim <= (uint32_t) kDim && r->dim % 2 == 0; }

extern "C" bool q4x_ix_q(const float* x, uint32_t n_tok, uint32_t heads, const Q4xRope* rope, const float* norm,
                         float eps, int32_t pos0, float* q, void* stream) {
  if (!x || !norm || !q || !n_tok || !heads || !rope_ok(rope) || pos0 < 0) return false;
  ix_q_kernel<<<dim3(heads, n_tok), kDim, 0, st(stream)>>>(x, heads, *rope, norm, eps, pos0, q);
  return ok();
}

extern "C" bool q4x_ix_pool(const float* raw, uint32_t n_tok, int32_t pos0, uint32_t ratio, float* ring,
                            uint32_t ring_rows, const Q4xRope* rope, const float* norm, float eps, float* pooled,
                            void* stream) {
  if (!raw || !ring || !norm || !pooled || !n_tok || ratio < 2 || ring_rows < ratio - 1 || !rope_ok(rope) || pos0 < 0)
    return false;
  const uint32_t j0 = (uint32_t) pos0 / ratio, j1 = ((uint32_t) pos0 + n_tok) / ratio;  // the blocks it completes
  if (j1 > j0) {
    ix_pool_kernel<<<j1 - j0, kDim, 0, st(stream)>>>(raw, pos0, ratio, ring, ring_rows, j0, *rope, norm, eps, pooled);
    if (!ok()) return false;
  }
  const uint32_t t0 = n_tok > ring_rows ? n_tok - ring_rows : 0;
  ix_ring_kernel<<<n_tok - t0, kDim, 0, st(stream)>>>(raw, pos0, t0, ring, ring_rows);
  return ok();
}

extern "C" uint64_t q4x_ix_scratch(uint32_t max_ctx, uint32_t ratio) {
  const uint64_t row = ratio ? max_ctx / ratio : 0;
  return row > kScoreBudget ? row : kScoreBudget;
}

extern "C" bool q4x_ix_pick(const float* q, const float* pooled, uint32_t n_tok, uint32_t heads, int32_t pos0,
                            uint32_t ratio, uint32_t blocks, float* scratch, uint64_t scratch_floats, int32_t* sel,
                            int32_t* npick, void* stream) {
  if (!q || !pooled || !scratch || !sel || !npick || !n_tok || !heads || heads > kMaxHeads || ratio < 2 || !blocks ||
      pos0 < 0)
    return false;
  const uint32_t nv_max = (uint32_t) (pos0 + (int32_t) n_tok) / ratio;  // the step's last token's
  if (nv_max <= blocks) {  // every token's every block: no score
    ix_pick_kernel<<<n_tok, kPick, 0, st(stream)>>>(scratch, pos0, ratio, blocks, 0, sel, npick);
    return ok();
  }
  if (scratch_floats < nv_max) return false;
  const uint32_t piece = scratch_floats / nv_max < n_tok ? (uint32_t) (scratch_floats / nv_max) : n_tok;
  for (uint32_t t0 = 0; t0 < n_tok; t0 += piece) {  // a piece of tokens at a time: their scores in the scratch
    const uint32_t n = n_tok - t0 < piece ? n_tok - t0 : piece;
    const int32_t p0 = pos0 + (int32_t) t0;
    const uint32_t nv = (uint32_t) (p0 + (int32_t) n) / ratio;
    if (nv > blocks) {
      ix_score_kernel<<<dim3((nv + kTB - 1) / kTB, (n + kTT - 1) / kTT), 256, 0, st(stream)>>>(
          q + (uint64_t) t0 * heads * kDim, pooled, n, heads, p0, ratio, nv, scratch);
      if (!ok()) return false;
    }
    ix_pick_kernel<<<n, kPick, 0, st(stream)>>>(scratch, p0, ratio, blocks, nv, sel + (uint64_t) t0 * blocks,
                                                npick + t0);
    if (!ok()) return false;
  }
  return true;
}
