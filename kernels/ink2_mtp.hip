// Inkling-2's drafter (DSpark, the RNN Markov head; src/ink2/hip_ink2_draft.c, tools/convert/ink2_mtp_oracle.py): its taps'
// mixtures, its context caches (f16), its attention over a context and a block, its Markov steps. F32 math.
#include <hip/hip_runtime.h>

#include <climits>
#include <cmath>

#include "block.h"
#include "launch.h"

extern "C" {
#include "osmos/gpu/ink2_ops.h"
#include "osmos/ink2/mtp.h"
}

namespace {

using launches::grid;
using launches::kThreads;
using launches::ok;
using launches::st;

__global__ void mix_kernel(const float* taps, uint32_t n_taps, const float* alpha, uint64_t n, uint32_t dim, float* out) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= n * dim) return;
  const uint64_t t = i / dim, c = i % dim;
  float v = 0.0f;
  for (uint32_t k = 0; k < n_taps; k++) v += alpha[k] * taps[(t * n_taps + k) * dim + c];
  out[i] = v;
}

__global__ void kv_put_kernel(_Float16* ring, uint32_t ring_rows, const float* x, uint32_t n, uint32_t row, int32_t pos0) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n * row) return;
  const uint32_t t = (uint32_t) (i / row), c = (uint32_t) (i % row);
  ring[(uint64_t) ((uint32_t) (pos0 + (int32_t) t) % ring_rows) * row + c] = (_Float16) x[i];
}

// a (KV head, span of keys, queries): the queries' heads of the KV head at once - each key read once for their rows
// (qpb queries times the heads a KV head serves) - a tile of kTile keys at a time, its keys then its values staged in
// LDS by every thread's independent loads: the rows' scores, then a running softmax over the span (each row's max,
// sum and weighted values rescaled as its max grows) - the span's partial a row. keys by position from the earliest
// any query sees (its window's start, or ctx_from), then the block's own; a context key past a query's window -inf
// for it
constexpr uint32_t kTile = 32, kHead = INK2_MTP_HEAD, kRows = INK2_MTP_ATTN_ROWS;
constexpr uint32_t kSpans = 80;  // gfx1151's 40 CUs, two blocks each, over the KV heads and the queries
constexpr uint32_t kKeyGroups = kThreads / kTile, kValHalves = kThreads / kHead;
__global__ __launch_bounds__(kThreads) void attn_part_kernel(const float* q, uint32_t n_q, int32_t pos0,
                                                             const float* kb, const float* vb, const _Float16* kr,
                                                             const _Float16* vr, uint32_t ring_rows, int32_t lo,
                                                             int32_t ctx_from, uint32_t heads, uint32_t kv_heads,
                                                             uint32_t window, uint32_t span, uint32_t qpb,
                                                             float* part, uint32_t spans) {
  __shared__ float qv[kRows][kHead], tile[kTile * kHead], sc[kRows][kTile], scale[kRows];
  const uint32_t g = blockIdx.x, grp = heads / kv_heads, q0 = blockIdx.z * qpb, row = kv_heads * kHead,
                 rows = (q0 + qpb < n_q ? qpb : n_q - q0) * grp;
  const uint32_t nctx = (uint32_t) (pos0 - lo), nk = nctx + n_q, j0 = blockIdx.y * span,
                 j1 = j0 + span < nk ? j0 + span : nk;
  for (uint32_t i = threadIdx.x; i < rows * kHead; i += kThreads) {
    const uint32_t r = i / kHead, qi = q0 + r / grp, h = g * grp + r % grp;
    qv[r][i % kHead] = q[((uint64_t) qi * heads + h) * kHead + i % kHead];
  }
  // a tile's key (or value) j's dims d .. d + kPer: the ring's f16, or the block's F32; 0 past the span
  constexpr uint32_t kPer = kTile * kHead / kThreads;
  const uint32_t lk = threadIdx.x / (kHead / kPer), ld = threadIdx.x % (kHead / kPer) * kPer;
  auto stage = [&](const _Float16* ring, const float* blk, uint32_t t0, bool transpose) {
    const uint32_t j = t0 + lk;
    float x[kPer];
    if (j >= j1) {
      for (uint32_t i = 0; i < kPer; i++) x[i] = 0.0f;
    } else if (j < nctx) {
      const _Float16* p = ring + (uint64_t) ((uint32_t) (lo + (int32_t) j) % ring_rows) * row + g * kHead + ld;
      for (uint32_t i = 0; i < kPer; i++) x[i] = (float) p[i];
    } else {
      const float* p = blk + (uint64_t) (j - nctx) * row + g * kHead + ld;
      for (uint32_t i = 0; i < kPer; i++) x[i] = p[i];
    }
    for (uint32_t i = 0; i < kPer; i++) tile[transpose ? (ld + i) * kTile + lk : lk * kHead + ld + i] = x[i];
  };
  const uint32_t kl = threadIdx.x % kTile, kg = threadIdx.x / kTile;  // scores: a key, rows kg, kg + kKeyGroups, ..
  const uint32_t dl = threadIdx.x % kHead, dh = threadIdx.x / kHead;  // values: a dim, rows dh, dh + kValHalves, ..
  float m = -INFINITY, l = 0.0f, acc[kRows / kValHalves];
  for (uint32_t k = 0; k < kRows / kValHalves; k++) acc[k] = 0.0f;
  for (uint32_t t0 = j0; t0 < j1; t0 += kTile) {
    __syncthreads();  // the last tile's values read
    stage(kr, kb, t0, true);
    __syncthreads();
    const uint32_t j = t0 + kl;
    for (uint32_t r = kg; r < rows; r += kKeyGroups) {
      float dot = 0.0f;
      for (uint32_t d = 0; d < kHead; d++) dot += qv[r][d] * tile[d * kTile + kl];
      const int32_t qp = pos0 + (int32_t) (q0 + r / grp),
                    from = window && qp - (int32_t) window + 1 > ctx_from ? qp - (int32_t) window + 1 : ctx_from;
      sc[r][kl] = j >= j1 || (j < nctx && lo + (int32_t) j < from) ? -INFINITY : dot / sqrtf((float) kHead);
    }
    __syncthreads();  // the scores written, the keys read
    stage(vr, vb, t0, false);
    if (threadIdx.x < rows) {  // a row's running max and sum, its weights in place
      const uint32_t r = threadIdx.x;
      float tm = m;
      for (uint32_t i = 0; i < kTile; i++) tm = fmaxf(tm, sc[r][i]);
      const float a = tm == -INFINITY ? 1.0f : expf(m - tm);
      float sum = 0.0f;
      for (uint32_t i = 0; i < kTile; i++) {
        const float w = sc[r][i] == -INFINITY ? 0.0f : expf(sc[r][i] - tm);
        sc[r][i] = w, sum += w;
      }
      l = l * a + sum, m = tm, scale[r] = a;
    }
    __syncthreads();
    for (uint32_t k = 0, r = dh; r < rows; k++, r += kValHalves) {
      float o = acc[k] * scale[r];
      for (uint32_t i = 0; i < kTile; i++) o += sc[r][i] * tile[i * kHead + dl];
      acc[k] = o;
    }
  }
  for (uint32_t k = 0, r = dh; r < rows; k++, r += kValHalves) {
    const uint32_t qi = q0 + r / grp, h = g * grp + r % grp;
    part[(((uint64_t) qi * heads + h) * spans + blockIdx.y) * (kHead + 2) + 2 + dl] = acc[k];
  }
  if (threadIdx.x < rows) {
    const uint32_t r = threadIdx.x, qi = q0 + r / grp, h = g * grp + r % grp;
    float* pp = part + (((uint64_t) qi * heads + h) * spans + blockIdx.y) * (kHead + 2);
    pp[0] = m, pp[1] = l;
  }
}

// a (query, head)'s chunks merged in order: out = sum o_c e^(m_c - M) / sum l_c e^(m_c - M)
__global__ void attn_merge_kernel(const float* part, uint32_t chunks, float* out) {
  const uint32_t d = threadIdx.x;
  const float* pp = part + (uint64_t) blockIdx.x * chunks * (kHead + 2);
  float M = -INFINITY;
  for (uint32_t c = 0; c < chunks; c++) M = fmaxf(M, pp[c * (kHead + 2)]);
  float L = 0.0f, O = 0.0f;
  for (uint32_t c = 0; c < chunks; c++) {
    const float m = pp[c * (kHead + 2)];
    if (m == -INFINITY) continue;
    const float w = expf(m - M);
    L += pp[c * (kHead + 2) + 1] * w, O += pp[c * (kHead + 2) + 2 + d] * w;
  }
  out[(uint64_t) blockIdx.x * kHead + d] = O / L;
}

// [state | markov_w1[prev]]: the Markov step's own input (prev on the device: the last step's pick, or the anchor)
__global__ void markov_in_kernel(const float* state, const uint16_t* w1, const int32_t* prev, uint32_t rank, float* vec) {
  const uint32_t i = threadIdx.x + blockIdx.x * kThreads;
  if (i >= 2 * rank) return;
  if (i < rank) {
    vec[i] = state[i];
  } else {
    const uint32_t b = (uint32_t) w1[(uint64_t) (*prev) * rank + (i - rank)] << 16;
    float f;
    __builtin_memcpy(&f, &b, 4);
    vec[i] = f;
  }
}

// the GRU-like step: g, c, o from the step's two projections and the bias; state <- g state + (1 - g) tanh(c)
__global__ void markov_gru_kernel(const float* p, const float* inv, const float* bias, uint32_t rank, float* state,
                                  float* latent) {
  const uint32_t i = threadIdx.x + blockIdx.x * kThreads;
  if (i >= rank) return;
  const float g = sigmoid(p[i] + inv[i] + bias[i]);
  const float c = p[rank + i] + inv[rank + i] + bias[rank + i], o = p[2 * rank + i] + inv[2 * rank + i] + bias[2 * rank + i];
  state[i] = g * state[i] + (1.0f - g) * tanhf(c);
  latent[i] = tanhf(o);
}

// a block's slice of the step's logits (cap tanh((base + corr + fold_b) / cap)): its greatest (the lowest id among
// equals), and the sum of exp(x - that) - each thread's running (its max, its sum rescaled as the max grows), then the
// block's - into part[block] = {max, id's bits, sum}
__global__ void pick_part_kernel(const float* base, const float* corr, const float* fold_b, float cap, uint32_t n,
                                 float* part) {
  __shared__ float lds[kThreads];
  __shared__ int ldi[kThreads];
  const uint32_t per = (n + gridDim.x - 1) / gridDim.x, lo = blockIdx.x * per, hi = lo + per < n ? lo + per : n;
  float m = -INFINITY, sum = 0.0f;
  int id = INT_MAX;
  for (uint32_t i = lo + threadIdx.x; i < hi; i += kThreads) {
    const float x = cap * tanhf((base[i] + corr[i] + fold_b[i]) / cap);
    if (x > m)
      sum = sum * expf(m - x) + 1.0f, m = x, id = (int) i;
    else
      sum += expf(x - m);
  }
  float bm = m;
  const int bid = block_argmax<kThreads>(&bm, id, lds, ldi);
  sum = block_sum<kThreads>(m == -INFINITY ? 0.0f : sum * expf(m - bm), lds);
  if (!threadIdx.x) {
    part[3 * blockIdx.x] = bm, part[3 * blockIdx.x + 2] = sum;
    __builtin_memcpy(&part[3 * blockIdx.x + 1], &bid, 4);
  }
}

// the blocks' parts merged: the greatest (the lowest id among equals) and its probability, 1 / sum exp(x - max)
constexpr uint32_t kPickBlocks = 80;  // gfx1151's 40 CUs, two blocks each: every CU's bandwidth on the logits
__global__ void pick_merge_kernel(const float* part, uint32_t blocks, int32_t* id, float* p) {
  __shared__ float lds[kThreads];
  __shared__ int ldi[kThreads];
  const uint32_t b = threadIdx.x;
  float m = b < blocks ? part[3 * b] : -INFINITY, s = b < blocks ? part[3 * b + 2] : 0.0f;
  int i = INT_MAX;
  if (b < blocks) __builtin_memcpy(&i, &part[3 * b + 1], 4);
  float bm = m;
  const int best = block_argmax<kThreads>(&bm, i, lds, ldi);
  const float sum = block_sum<kThreads>(m == -INFINITY ? 0.0f : s * expf(m - bm), lds);
  if (!threadIdx.x) *id = best, *p = 1.0f / sum;
}
static_assert(kPickBlocks <= kThreads, "a merge thread a block");
static_assert(kThreads % kTile == 0 && kThreads % kHead == 0 && kTile * kHead % kThreads == 0 && kRows <= kThreads,
              "the attention's threads: whole groups a key and a dim, a tile's staging even, a thread a row");

// an output row's block of 32 (row c, values r0 .. r0 + 31 of in's column c) as ggml's Q8_0 reference quantizes it:
// d = amax / 127 (f16), q = round(x / d) (half away from zero)
constexpr uint32_t kQ8 = 32, kQ8Bytes = 34;
__global__ void transpose_q8_0_kernel(const float* in, uint32_t rows, uint32_t cols, uint8_t* out) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x, blocks = rows / kQ8;
  if (i >= blocks * cols) return;
  const uint32_t c = (uint32_t) (i / blocks), r0 = (uint32_t) (i % blocks) * kQ8;
  uint8_t* b = out + i * kQ8Bytes;
  const _Float16 dh = (_Float16) q8_0_block<kQ8>(in + (uint64_t) r0 * cols + c, cols, b + 2);
  __builtin_memcpy(b, &dh, 2);
}

}  // namespace

extern "C" bool ink2_mtp_mix(const float* taps, uint32_t n_taps, const float* alpha, uint32_t n, uint32_t dim, float* out,
                             void* stream) {
  if (!taps || !alpha || !out || !n || !dim || !n_taps) return false;
  mix_kernel<<<grid((uint64_t) n * dim), kThreads, 0, st(stream)>>>(taps, n_taps, alpha, n, dim, out);
  return ok();
}

extern "C" bool ink2_mtp_kv_put(uint16_t* ring, uint32_t ring_rows, const float* x, uint32_t n, uint32_t row, int32_t pos0,
                                void* stream) {
  if (!ring || !x || !n || !row || !ring_rows) return false;
  kv_put_kernel<<<grid((uint64_t) n * row), kThreads, 0, st(stream)>>>((_Float16*) ring, ring_rows, x, n, row, pos0);
  return ok();
}

extern "C" uint64_t ink2_mtp_attn_scratch(uint32_t n_q, uint32_t heads, uint32_t kv_heads) {
  return kv_heads ? (uint64_t) n_q * heads * ((kSpans + kv_heads - 1) / kv_heads) * (kHead + 2) * sizeof(float) : 0;
}

extern "C" bool ink2_mtp_attn(const float* q, uint32_t n_q, int32_t pos0, const float* kb, const float* vb,
                              const uint16_t* kr, const uint16_t* vr, uint32_t ring_rows, int32_t ctx_from,
                              uint32_t heads, uint32_t kv_heads, uint32_t window, float* scratch, float* out,
                              void* stream) {
  if (!q || !kb || !vb || !kr || !vr || !scratch || !out || !n_q || !heads || !kv_heads || heads % kv_heads ||
      n_q * (heads / kv_heads) > kRows || ctx_from < 0 || ctx_from > pos0 || (uint64_t) (pos0 - ctx_from) > ring_rows)
    return false;
  const int32_t lo = window && pos0 - (int32_t) window + 1 > ctx_from ? pos0 - (int32_t) window + 1 : ctx_from;
  const uint32_t nk = (uint32_t) (pos0 - lo) + n_q, tiles = (nk + kTile - 1) / kTile,
                 most = (kSpans + kv_heads - 1) / kv_heads, per = (tiles + most - 1) / most,
                 spans = (tiles + per - 1) / per;
  /* a short context's few spans: the queries apart too, to fill the GPU's blocks (each apart reads the keys again) */
  const uint32_t want = (kSpans + kv_heads * spans - 1) / (kv_heads * spans), split = want < n_q ? want : n_q,
                 qpb = (n_q + split - 1) / split;
  attn_part_kernel<<<dim3(kv_heads, spans, (n_q + qpb - 1) / qpb), kThreads, 0, st(stream)>>>(
      q, n_q, pos0, kb, vb, (const _Float16*) kr, (const _Float16*) vr, ring_rows, lo, ctx_from, heads, kv_heads,
      window, per * kTile, qpb, scratch, spans);
  attn_merge_kernel<<<n_q * heads, kHead, 0, st(stream)>>>(scratch, spans, out);
  return ok();
}

extern "C" bool ink2_mtp_markov_in(const float* state, const uint16_t* w1, const int32_t* prev, uint32_t rank, float* vec,
                                   void* stream) {
  markov_in_kernel<<<grid(2 * rank), kThreads, 0, st(stream)>>>(state, w1, prev, rank, vec);
  return ok();
}

extern "C" bool ink2_mtp_markov_gru(const float* p, const float* inv, const float* bias, uint32_t rank, float* state,
                                    float* latent, void* stream) {
  markov_gru_kernel<<<grid(rank), kThreads, 0, st(stream)>>>(p, inv, bias, rank, state, latent);
  return ok();
}

extern "C" uint64_t ink2_mtp_pick_scratch(void) { return 3ull * kPickBlocks * sizeof(float); }

extern "C" bool ink2_mtp_markov_pick(const float* base, const float* corr, const float* fold_b, float cap, uint32_t n,
                                     float* scratch, int32_t* id, float* p, void* stream) {
  if (!base || !corr || !fold_b || !scratch || !id || !p || !n || n > (uint32_t) INT_MAX || !(cap > 0)) return false;
  pick_part_kernel<<<kPickBlocks, kThreads, 0, st(stream)>>>(base, corr, fold_b, cap, n, scratch);
  pick_merge_kernel<<<1, kThreads, 0, st(stream)>>>(scratch, kPickBlocks, id, p);
  return ok();
}

extern "C" bool ink2_transpose_q8_0(const float* in, uint32_t rows, uint32_t cols, void* out, void* stream) {
  if (!in || !out || !rows || rows % kQ8 || !cols) return false;
  transpose_q8_0_kernel<<<grid((uint64_t) rows / kQ8 * cols), kThreads, 0, st(stream)>>>(in, rows, cols, (uint8_t*) out);
  return ok();
}
