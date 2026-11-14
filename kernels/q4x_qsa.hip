// QSA's indexer (include/osmos/gpu/q4x_ops.h, q4x_ix_*; DESIGN.md, "QSA"): its query's norm and rotation, the keys'
// blocks pooled, a token's scores against them and its top blocks - llama.cpp's build_qsa_sel in its order mode
#include "block.h"
#include "launch.h"
#include "rope.h"
#include "select.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

namespace {

using launches::st;
using launches::ok;

constexpr int kDim = Q4X_IX_DIM;  // a head's dim: a block's threads, a value a thread
constexpr int kMaxHeads = Q4X_IX_HEADS_MAX; // the score tile's query rows (attention.indexer.head_count)
constexpr int kTT = 16, kTB = 32; // the score tile: tokens x blocks
constexpr uint64_t kScoreBudget = 4u << 20;  // the scratch's scores at least: a piece's tokens x their blocks
constexpr uint64_t kScoreTokens = 128;       // (0.7.3) and room for this many tokens' rows (q4x_ix_scratch)
// (0.7.3) the blocked scores from a step of this many tokens: below it its 64-token tile mostly empty - 1 token 0.8x
// the plain kernel's time at 511K, 16 0.9x, 32 1.4x, 64 2.1x (qxbench QX_N); the same bits either way
constexpr uint32_t kBlockedFrom = 32;


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

// 0.7.3: the scores register-blocked - a tile of 64 tokens x 64 blocks, 256 threads, a thread 4 tokens x 4 blocks x
// the 4 heads' dot products (64 sums in registers), d staged 16 at a time: 20 LDS reads a thread for 64 products
// (ix_score_kernel's 2 for 1). each (token, head, block) dot over d ascending, a multiply then an add; the heads
// folded in order, s += relu(dot) / sqrt(kDim) (a head past `heads` 0: s + 0) - ix_score_kernel's bits
constexpr int kBT = 64, kBB = 64, kBS = 16, kBThreads = 256;
__global__ __launch_bounds__(kBThreads) void ix_score_blocked_kernel(const float* __restrict__ q,
                                                                     const float* __restrict__ pooled, uint32_t m,
                                                                     uint32_t heads, int32_t p0, uint32_t ratio,
                                                                     uint32_t stride, float* __restrict__ score) {
  static_assert(kMaxHeads == 4 && kDim % kBS == 0, "4 head slots; whole stages");
  __shared__ float qs[kBS][kBT * kMaxHeads + 4], ks[kBS][kBB + 4];
  const uint32_t tt = blockIdx.y * kBT, tb = blockIdx.x * kBB;
  const uint32_t t_last = tt + kBT - 1 < m ? tt + kBT - 1 : m - 1;
  const uint32_t nv_last = (uint32_t) (p0 + (int32_t) t_last + 1) / ratio;
  if (tb >= nv_last) return;
  const uint32_t tid = threadIdx.x, tx = tid % 16, ty = tid / 16;  // blocks tx + 16 j, tokens ty + 16 i
  float acc[4][4][kMaxHeads] = {};
  for (uint32_t d0 = 0; d0 < (uint32_t) kDim; d0 += kBS) {
    for (uint32_t i = tid; i < (uint32_t) (kBT * kMaxHeads * kBS); i += kBThreads) {
      const uint32_t r = i / kBS, dd = i % kBS, t = tt + r / kMaxHeads, h = r % kMaxHeads;
      qs[dd][r] = t < m && h < heads ? q[((uint64_t) t * heads + h) * kDim + d0 + dd] : 0.0f;
    }
    for (uint32_t i = tid; i < (uint32_t) (kBB * kBS); i += kBThreads) {
      const uint32_t r = i / kBS, dd = i % kBS;
      ks[dd][r] = tb + r < nv_last ? pooled[(uint64_t) (tb + r) * kDim + d0 + dd] : 0.0f;
    }
    __syncthreads();
#pragma unroll 4
    for (int dd = 0; dd < kBS; dd++) {
      float kv[4];
      for (int j = 0; j < 4; j++) kv[j] = ks[dd][tx + 16 * j];
      for (int i = 0; i < 4; i++)
        for (int h = 0; h < kMaxHeads; h++) {
          const float qv = qs[dd][(ty + 16 * i) * kMaxHeads + h];
          for (int j = 0; j < 4; j++) acc[i][j][h] = acc[i][j][h] + qv * kv[j];
        }
    }
    __syncthreads();
  }
  const float scale = 1.0f / sqrtf((float) kDim);
  for (int i = 0; i < 4; i++) {
    const uint32_t t = tt + ty + 16 * i;
    if (t >= m) continue;
    const uint32_t nv = (uint32_t) (p0 + (int32_t) t + 1) / ratio;
    for (int j = 0; j < 4; j++) {
      const uint32_t b = tb + tx + 16 * j;
      if (b >= nv) continue;
      float s = 0.0f;
      for (uint32_t h = 0; h < heads; h++) s += fmaxf(acc[i][j][h], 0.0f) * scale;
      score[(uint64_t) t * stride + b] = s;
    }
  }
}
bool g_score_naive = false;  // (tests, qxbench: ix_score_kernel itself - q4x_ix_score_naive)

// (0.7.4) a small step's scores (decode's, a verify's: up to kOneTok tokens): a block kOB pooled keys, a thread a
// (head, key) pair - every token's dot of it, d ascending, a multiply then an add (ix_score_kernel's) - then a thread
// a (token, key) folding the heads in order, s += relu(dot) / sqrt(kDim): ix_score_kernel's bits, all 256 threads busy
// (that tile's at one token 16 of them)
constexpr uint32_t kOneTok = 8, kOB = 256 / kMaxHeads;
__global__ __launch_bounds__(256) void ix_score_small_kernel(const float* q, const float* pooled, uint32_t m,
                                                             uint32_t heads, int32_t p0, uint32_t ratio,
                                                             uint32_t stride, float* score) {
  __shared__ float qs[kOneTok * kMaxHeads * kDim], ks[kOB][kDim + 1], dd[kOneTok][kMaxHeads][kOB];
  const uint32_t tb = blockIdx.x * kOB, nv_last = (uint32_t) (p0 + (int32_t) m) / ratio;
  if (tb >= nv_last) return;
  const uint32_t qw = heads * kDim;
  for (uint32_t i = threadIdx.x; i < m * qw; i += 256) qs[i] = q[i];
  for (uint32_t i = threadIdx.x; i < kOB * kDim; i += 256) {
    const uint32_t r = i / kDim, d = i % kDim;
    ks[r][d] = tb + r < nv_last ? pooled[(uint64_t) (tb + r) * kDim + d] : 0.0f;
  }
  __syncthreads();
  const uint32_t jl = threadIdx.x % kOB, h = threadIdx.x / kOB;
  if (h < heads)
    for (uint32_t t = 0; t < m; t++) {
      float dot = 0.0f;
      for (int d = 0; d < kDim; d++) dot += qs[t * qw + h * kDim + d] * ks[jl][d];
      dd[t][h][jl] = dot;
    }
  __syncthreads();
  const float scale = 1.0f / sqrtf((float) kDim);
  for (uint32_t i = threadIdx.x; i < m * kOB; i += 256) {
    const uint32_t t = i / kOB, l = i % kOB, j = tb + l;
    if (j >= (uint32_t) (p0 + (int32_t) t + 1) / ratio) continue;
    float s = 0.0f;
    for (uint32_t x = 0; x < heads; x++) s += fmaxf(dd[t][x][l], 0.0f) * scale;
    score[(uint64_t) t * stride + j] = s;
  }
}

// a block a token: with more visible blocks than `blocks`, its top `blocks` by score (non-negative floats ordered as
// their bits; the lower block among equals) written in block order - (0.7.4) sel::select_core, DeepSeek's selection
__global__ __launch_bounds__(sel::kSelectThreads) void ix_pick_kernel(const float* score, int32_t p0, uint32_t ratio,
                                                                      uint32_t blocks, uint32_t stride, int32_t* out,
                                                                      int32_t* npick) {
  const uint32_t t = blockIdx.x, nv = (uint32_t) (p0 + (int32_t) t + 1) / ratio;
  if (nv <= blocks) {
    if (threadIdx.x == 0) npick[t] = (int32_t) nv;
    return;
  }
  const uint32_t* key = (const uint32_t*) (score + (uint64_t) t * stride);
  int32_t* row = out + (uint64_t) t * blocks;
  sel::select_core(nv, blocks, [&](uint32_t j) { return key[j]; }, [&](uint32_t j, uint32_t slot) { row[slot] = (int32_t) j; });
  if (threadIdx.x == 0) npick[t] = (int32_t) blocks;
}

// (0.7.4) a small step's pick on many blocks: each of sel::kChunks chunks' top `blocks` (ix_chunk_kernel), then the
// top `blocks` of those (ix_merge_kernel) - the same blocks, in the same order (attn.hip's chunk_kernel)
__global__ __launch_bounds__(sel::kSelectThreads) void ix_chunk_kernel(const float* score, int32_t p0, uint32_t ratio,
                                                                       uint32_t blocks, uint32_t stride) {
  const uint32_t m = blockIdx.x, t = blockIdx.y, nv = (uint32_t) (p0 + (int32_t) t + 1) / ratio;
  if (nv <= blocks) return;  // ix_merge_kernel's: every block
  const uint32_t per = (nv + sel::kChunks - 1) / sel::kChunks, lo = min(nv, m * per), hi = min(nv, lo + per);
  const uint32_t* key = (const uint32_t*) (score + (uint64_t) t * stride);
  uint32_t* ck = sel::g_ckey + ((uint64_t) t * sel::kChunks + m) * sel::kChunkK;
  uint32_t* ci = sel::g_citem + ((uint64_t) t * sel::kChunks + m) * sel::kChunkK;
  const uint32_t got = sel::select_core(hi - lo, blocks, [&](uint32_t c) { return key[lo + c]; },
                                        [&](uint32_t c, uint32_t slot) { ck[slot] = key[lo + c], ci[slot] = lo + c; });
  if (threadIdx.x == 0) sel::g_ccount[t * sel::kChunks + m] = got;
}

__global__ __launch_bounds__(sel::kSelectThreads) void ix_merge_kernel(int32_t p0, uint32_t ratio, uint32_t blocks,
                                                                       int32_t* out, int32_t* npick) {
  __shared__ uint32_t off[sel::kChunks + 1];
  const uint32_t t = blockIdx.x, nv = (uint32_t) (p0 + (int32_t) t + 1) / ratio;
  if (nv <= blocks) {
    if (threadIdx.x == 0) npick[t] = (int32_t) nv;
    return;
  }
  if (threadIdx.x == 0) {
    off[0] = 0;
    for (uint32_t m = 0; m < sel::kChunks; m++) off[m + 1] = off[m] + sel::g_ccount[t * sel::kChunks + m];
  }
  __syncthreads();
  const uint32_t* ck = sel::g_ckey + (uint64_t) t * sel::kChunks * sel::kChunkK;
  const uint32_t* ci = sel::g_citem + (uint64_t) t * sel::kChunks * sel::kChunkK;
  int32_t* row = out + (uint64_t) t * blocks;
  sel::select_core(off[sel::kChunks], blocks, [&](uint32_t i) { return ck[sel::chunk_slot(off, i)]; },
                   [&](uint32_t i, uint32_t slot) { row[slot] = (int32_t) ci[sel::chunk_slot(off, i)]; });
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
  // 0.7.3: room for kScoreTokens tokens' rows - a piece the blocked scores' tiles whole (32 tokens at 512K, from the
  // old floor, left its 64-token tile half empty: 31.6 ms a 256-token step at 511K; 128 tokens 15.9, 67 MB there)
  const uint64_t row = ratio ? max_ctx / ratio : 0, want = row * kScoreTokens;
  return want > kScoreBudget ? want : kScoreBudget;
}

extern "C" void q4x_ix_score_naive(bool on) { g_score_naive = on; }

extern "C" bool q4x_ix_pick(const float* q, const float* pooled, uint32_t n_tok, uint32_t heads, int32_t pos0,
                            uint32_t ratio, uint32_t blocks, float* scratch, uint64_t scratch_floats, int32_t* sel,
                            int32_t* npick, void* stream) {
  if (!q || !pooled || !scratch || !sel || !npick || !n_tok || !heads || heads > kMaxHeads || ratio < 2 || !blocks ||
      pos0 < 0)
    return false;
  const uint32_t nv_max = (uint32_t) (pos0 + (int32_t) n_tok) / ratio;  // the step's last token's
  if (nv_max <= blocks) {  // every token's every block: no score
    ix_pick_kernel<<<n_tok, sel::kSelectThreads, 0, st(stream)>>>(scratch, pos0, ratio, blocks, 0, sel, npick);
    return ok();
  }
  if (scratch_floats < nv_max) return false;
  const uint32_t piece = scratch_floats / nv_max < n_tok ? (uint32_t) (scratch_floats / nv_max) : n_tok;
  for (uint32_t t0 = 0; t0 < n_tok; t0 += piece) {  // a piece of tokens at a time: their scores in the scratch
    const uint32_t n = n_tok - t0 < piece ? n_tok - t0 : piece;
    const int32_t p0 = pos0 + (int32_t) t0;
    const uint32_t nv = (uint32_t) (p0 + (int32_t) n) / ratio;
    if (nv > blocks) {
      if (!g_score_naive && n <= kOneTok)  // (0.7.4) decode's, a verify's: the small step's kernel
        ix_score_small_kernel<<<(nv + kOB - 1) / kOB, 256, 0, st(stream)>>>(q + (uint64_t) t0 * heads * kDim, pooled,
                                                                          n, heads, p0, ratio, nv, scratch);
      else if (g_score_naive || n < kBlockedFrom)  // a small step the plain kernel's: faster there than the blocked
        ix_score_kernel<<<dim3((nv + kTB - 1) / kTB, (n + kTT - 1) / kTT), 256, 0, st(stream)>>>(
            q + (uint64_t) t0 * heads * kDim, pooled, n, heads, p0, ratio, nv, scratch);
      else
        ix_score_blocked_kernel<<<dim3((nv + kBB - 1) / kBB, (n + kBT - 1) / kBT), kBThreads, 0, st(stream)>>>(
            q + (uint64_t) t0 * heads * kDim, pooled, n, heads, p0, ratio, nv, scratch);
      if (!ok()) return false;
    }
    if (sel::chunked(n, nv, blocks)) {
      ix_chunk_kernel<<<dim3(sel::kChunks, n), sel::kSelectThreads, 0, st(stream)>>>(scratch, p0, ratio, blocks, nv);
      if (!ok()) return false;
      ix_merge_kernel<<<n, sel::kSelectThreads, 0, st(stream)>>>(p0, ratio, blocks, sel + (uint64_t) t0 * blocks,
                                                                  npick + t0);
    } else {
      ix_pick_kernel<<<n, sel::kSelectThreads, 0, st(stream)>>>(scratch, p0, ratio, blocks, nv,
                                                                 sel + (uint64_t) t0 * blocks, npick + t0);
    }
    if (!ok()) return false;
  }
  return true;
}
