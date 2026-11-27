// the drafters' attention (the attention core's ATTN_BLOCK_DRAFT, docs/attention-core.md C3): a block's queries against
// the ring's context and the block's own keys, not causal - each family's kernel moved whole, its bits: Inkling's MTP
// (ink2_mtp_attn's: an f16 ring, spans merged in order) and Qwen's DFlash (q4x_df_attn's: an F32 ring, one softmax a
// (query, head))
#pragma once
#include <hip/hip_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

#include "block.h"
#include "launch.h"

extern "C" {
#include "osmos/gpu/ink2_ops.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/ink2/mtp.h"
}

namespace attn_mtp {

using launches::kThreads;
using launches::ok;
using launches::st;

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

static_assert(kThreads % kTile == 0 && kThreads % kHead == 0 && kTile * kHead % kThreads == 0 && kRows <= kThreads,
              "the attention's threads: whole groups a key and a dim, a tile's staging even, a thread a row");

inline uint64_t scratch_bytes(uint32_t n_q, uint32_t heads, uint32_t kv_heads) {
  return kv_heads ? (uint64_t) n_q * heads * ((kSpans + kv_heads - 1) / kv_heads) * (kHead + 2) * sizeof(float) : 0;
}

inline bool run(const float* q, uint32_t n_q, int32_t pos0, const float* kb, const float* vb,
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

}  // namespace attn_mtp

namespace attn_df {

using launches::kThreads;
using launches::ok;
using launches::st;

// a block a (query, head): its visible keys' scores into the LDS (the ring's then the block's), their max and sum by
// block_sum's tree order, then each of its dims the weighted sum over the keys in order
__global__ void attn_kernel(const float* q, uint32_t n_q, int32_t pos0, const float* kb, const float* vb, const float* kr,
                            const float* vr, uint32_t ring_rows, int32_t ctx_from, uint32_t heads, uint32_t kv_heads,
                            uint32_t window, float* out) {
  extern __shared__ float lds[];
  __shared__ float qv[Q4X_DF_HEAD], red[kThreads];
  const uint32_t qi = blockIdx.x, h = blockIdx.y, g = h / (heads / kv_heads), row = kv_heads * Q4X_DF_HEAD;
  const int32_t qp = pos0 + (int32_t) qi;
  const int32_t lo = ctx_from > qp - (int32_t) window + 1 ? ctx_from : qp - (int32_t) window + 1;
  const uint32_t nctx = pos0 > lo ? (uint32_t) (pos0 - lo) : 0, nk = nctx + n_q;
  const float* qh = q + ((uint64_t) qi * heads + h) * Q4X_DF_HEAD;
  for (uint32_t d = threadIdx.x; d < Q4X_DF_HEAD; d += kThreads) qv[d] = qh[d];
  __syncthreads();
  const float scale = 1.0f / sqrtf((float) Q4X_DF_HEAD);
  float mx = -INFINITY;
  for (uint32_t j = threadIdx.x; j < nk; j += kThreads) {
    const float* k = j < nctx ? kr + (uint64_t) ((uint64_t) (lo + (int32_t) j) % ring_rows) * row + g * Q4X_DF_HEAD
                              : kb + (uint64_t) (j - nctx) * row + g * Q4X_DF_HEAD;
    float s = 0.0f;
    for (uint32_t d = 0; d < Q4X_DF_HEAD; d++) s = s + qv[d] * k[d];
    lds[j] = s * scale;
    mx = fmaxf(mx, lds[j]);
  }
  mx = block_max<kThreads>(mx, red);
  float sum = 0.0f;
  for (uint32_t j = threadIdx.x; j < nk; j += kThreads) {
    lds[j] = expf(lds[j] - mx);
    sum = sum + lds[j];
  }
  sum = block_sum<kThreads>(sum, red);
  for (uint32_t d = threadIdx.x; d < Q4X_DF_HEAD; d += kThreads) {
    float o = 0.0f;
    for (uint32_t j = 0; j < nk; j++) {
      const float* v = j < nctx ? vr + (uint64_t) ((uint64_t) (lo + (int32_t) j) % ring_rows) * row + g * Q4X_DF_HEAD
                                : vb + (uint64_t) (j - nctx) * row + g * Q4X_DF_HEAD;
      o = o + lds[j] / sum * v[d];
    }
    out[((uint64_t) qi * heads + h) * Q4X_DF_HEAD + d] = o;
  }
}

inline bool run(const float* q, uint32_t n_q, int32_t pos0, const float* kb, const float* vb, const float* kr,
                            const float* vr, uint32_t ring_rows, int32_t ctx_from, uint32_t heads, uint32_t kv_heads,
                            uint32_t window, float* out, void* stream) {
  if (!q || !kb || !vb || !kr || !vr || !out || !n_q || !heads || !kv_heads || heads % kv_heads || !window ||
      window < n_q || ctx_from < 0 || ctx_from > pos0 || ctx_from < pos0 - (int32_t) ring_rows)
    return false;
  const size_t lds = ((size_t) ring_rows + n_q) * sizeof(float);
  attn_kernel<<<dim3(n_q, heads), kThreads, lds, st(stream)>>>(q, n_q, pos0, kb, vb, kr, vr, ring_rows, ctx_from, heads,
                                                                kv_heads, window, out);
  return ok();
}

}  // namespace attn_df
