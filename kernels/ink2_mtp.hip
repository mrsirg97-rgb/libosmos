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
