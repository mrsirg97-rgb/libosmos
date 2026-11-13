// the NPU's routed-expert kernel's input and output on the GPU's side (include/osmos/gpu/npu_pack.h)
#include <cstdint>

#include <hip/hip_runtime.h>

#include "launch.h"

extern "C" {
#include "osmos/gpu/npu_pack.h"
}

namespace {

using launches::grid;
using launches::kThreads;
using launches::ok;
using launches::st;

constexpr uint32_t kPiece = 64 * 128 * 9 / 8; // a 64 x 128 piece's bytes

// a thread a block of 8 values: part p, token row m, block kb along K. x f32 or f16 rows, rows[p * NPU_TOKENS + m] the
// part's token m's (-1 a zero row); halves: the pieces [K half][token half] (Qwen's down: K no whole 256), else
// [K-step][token half][K half]
template <typename T>
__global__ void pack_kernel(const T* x, uint32_t K, const int32_t* rows, uint32_t parts, bool halves, uint8_t* dst) {
  const uint32_t nb = K / 8;
  const uint64_t id = blockIdx.x * (uint64_t) kThreads + threadIdx.x;
  if (id >= (uint64_t) parts * NPU_TOKENS * nb) return;
  const uint32_t kb = (uint32_t) (id % nb), m = (uint32_t) (id / nb % NPU_TOKENS), p = (uint32_t) (id / nb / NPU_TOKENS);
  const int32_t row = rows[p * NPU_TOKENS + m];
  float v[8];
  uint32_t e = 0;
  for (int j = 0; j < 8; j++) {
    v[j] = row < 0 ? 0.0f : (float) x[(uint64_t) row * K + kb * 8 + j];
    const uint32_t ej = (__float_as_uint(v[j]) >> 23) & 0xFF;
    e = ej > e ? ej : e;
  }
  // mantissa m_j = rint(v_j / 2^(e - 133)); a carry past 127 raises e (the block requantized)
  int32_t q[8];
  for (int pass = 0; pass < 2; pass++) {
    bool carry = false;
    for (int j = 0; j < 8; j++) {
      q[j] = (int32_t) rintf(ldexpf(v[j], 133 - (int) e));
      carry |= q[j] > 127;
    }
    if (!carry) break;
    e++;
  }
  const uint32_t s = kb / 32, kq = kb % 32 / 16, sx = kb % 16, h = m / 64, sy = m % 64 / 8, i = m % 8;
  const uint32_t piece = halves ? (kb / 16) * 2 + h : (s * 2 + h) * 2 + kq;
  uint8_t* o = dst + (uint64_t) p * NPU_TOKENS * K * 9 / 8 + piece * (uint64_t) kPiece + ((sy * 16 + sx) * 8 + i) * 9;
  o[0] = (uint8_t) e;
  for (int j = 0; j < 8; j++) o[1 + j] = (uint8_t) (int8_t) (q[j] < -128 ? -128 : q[j] > 127 ? 127 : q[j]);
}

// Qwen's parts: the first `want` entries of the 128-token run (its tables' last: [first, first + count)) that are
// whole (cnt 128), in table order - one thread scans, the block fills; each taken entry's count zeroed (mm_group then
// skips it), its weights and offsets written to the host's sel, its tokens' rows (gidx) and its slots for the down
__global__ void select_kernel(int32_t* cnt, uint32_t first, uint32_t count, uint32_t want, const void* const* ptr,
                              uint32_t entries, const uint32_t* xg, const uint32_t* yg, const uint32_t* xd,
                              const uint32_t* yd, const int32_t* gidx, uint32_t D, uint32_t ff, NpuPart* sel,
                              uint32_t* n_sel, int32_t* tok, int32_t* slot) {
  __shared__ uint32_t take[NPU_SELECT_MAX], n;
  if (threadIdx.x == 0) {
    n = 0;
    for (uint32_t e = first; e < first + count && n < want && n < NPU_SELECT_MAX; e++)
      if (cnt[e] == (int32_t) NPU_TOKENS) take[n++] = e;
    *n_sel = n;
  }
  __syncthreads();
  for (uint32_t p = 0; p < n; p++) {
    const uint32_t e = take[p], s0 = xg[e] / D, d0 = xd[e] / ff;
    for (uint32_t t = threadIdx.x; t < NPU_TOKENS; t += blockDim.x)
      tok[p * NPU_TOKENS + t] = gidx[s0 + t], slot[p * NPU_TOKENS + t] = (int32_t) (d0 + t);
    if (threadIdx.x == 0)
      sel[p] = NpuPart{ptr[e], ptr[entries + e], ptr[2 * entries + e], yg[e], yd[e]};
  }
  __syncthreads();
  if (threadIdx.x == 0)
    for (uint32_t p = 0; p < n; p++) cnt[take[p]] = 0;
}

}  // namespace

extern "C" bool npu_pack_a(const float* x, uint32_t K, const int32_t* tok, uint32_t parts, uint8_t* dst, void* stream) {
  if (!parts || !K || K % 256) return false;
  pack_kernel<float><<<grid((uint64_t) parts * NPU_TOKENS * (K / 8)), kThreads, 0, st(stream)>>>(x, K, tok, parts, false, dst);
  return ok();
}

extern "C" bool npu_pack_a_x16(const void* x16, uint32_t K, const int32_t* rows, uint32_t parts, bool halves,
                               uint8_t* dst, void* stream) {
  if (!parts || !K || K % (halves ? 128 : 256)) return false;
  pack_kernel<_Float16><<<grid((uint64_t) parts * NPU_TOKENS * (K / 8)), kThreads, 0, st(stream)>>>(
      (const _Float16*) x16, K, rows, parts, halves, dst);
  return ok();
}

extern "C" bool npu_select_q4x(int32_t* cnt, uint32_t first, uint32_t count, uint32_t want, const void* const* ptr,
                               uint32_t entries, const uint32_t* xg, const uint32_t* yg, const uint32_t* xd,
                               const uint32_t* yd, const int32_t* gidx, uint32_t D, uint32_t ff, NpuPart* sel,
                               uint32_t* n_sel, int32_t* tok, int32_t* slot, void* stream) {
  if (want > NPU_SELECT_MAX || !D || !ff) return false;
  select_kernel<<<1, kThreads, 0, st(stream)>>>(cnt, first, count, want, ptr, entries, xg, yg, xd, yd, gidx, D, ff, sel,
                                                n_sel, tok, slot);
  return ok();
}
