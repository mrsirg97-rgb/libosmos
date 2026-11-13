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

// a thread a block of 8 values: part p, token row m, block kb along K
__global__ void pack_kernel(const float* x, uint32_t K, const int32_t* tok, uint32_t parts, uint8_t* dst) {
  const uint32_t nb = K / 8;
  const uint64_t id = blockIdx.x * (uint64_t) kThreads + threadIdx.x;
  if (id >= (uint64_t) parts * NPU_TOKENS * nb) return;
  const uint32_t kb = (uint32_t) (id % nb), m = (uint32_t) (id / nb % NPU_TOKENS), p = (uint32_t) (id / nb / NPU_TOKENS);
  const int32_t row = tok[p * NPU_TOKENS + m];
  float v[8];
  uint32_t e = 0;
  for (int j = 0; j < 8; j++) {
    v[j] = row < 0 ? 0.0f : x[(uint64_t) row * K + kb * 8 + j];
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
  uint8_t* o = dst + (uint64_t) p * NPU_TOKENS * K * 9 / 8 + ((s * 2 + h) * 2 + kq) * (uint64_t) kPiece +
               ((sy * 16 + sx) * 8 + i) * 9;
  o[0] = (uint8_t) e;
  for (int j = 0; j < 8; j++) o[1 + j] = (uint8_t) (int8_t) (q[j] < -128 ? -128 : q[j] > 127 ? 127 : q[j]);
}

// a thread a value: part p, matrix mat, token m, column n
__global__ void unpack_kernel(const float* c, uint32_t N, uint32_t nr, uint32_t mats, uint32_t parts,
                              const uint32_t* row0, float* y0, float* y1) {
  const uint64_t id = blockIdx.x * (uint64_t) kThreads + threadIdx.x;
  if (id >= (uint64_t) parts * mats * NPU_TOKENS * N) return;
  const uint32_t n = (uint32_t) (id % N), m = (uint32_t) (id / N % NPU_TOKENS),
                 mat = (uint32_t) (id / N / NPU_TOKENS % mats), p = (uint32_t) (id / N / NPU_TOKENS / mats);
  const uint32_t P = N / (NPU_CORES * nr), ps = n / (NPU_CORES * nr), core = n % (NPU_CORES * nr) / nr, nn = n % nr,
                 half = m / 64, z = m % 64 / 8;
  const float v = c[(uint64_t) p * mats * NPU_TOKENS * N +  /* npu_c_floats(N, mats) a part */
                    ((((((uint64_t) mat * P + ps) * NPU_CORES + core) * 2 + half) * 8 + z) * (nr / 8) + nn / 8) * 64 +
                    m % 8 * 8 + nn % 8];
  (mat ? y1 : y0)[((uint64_t) row0[p] + m) * N + n] = v;
}

}  // namespace

extern "C" bool npu_pack_a(const float* x, uint32_t K, const int32_t* tok, uint32_t parts, uint8_t* dst, void* stream) {
  if (!parts || !K || K % 256) return false;
  pack_kernel<<<grid((uint64_t) parts * NPU_TOKENS * (K / 8)), kThreads, 0, st(stream)>>>(x, K, tok, parts, dst);
  return ok();
}

extern "C" bool npu_unpack_c(const float* c, uint32_t N, uint32_t nr, uint32_t mats, uint32_t parts,
                             const uint32_t* row0, float* y0, float* y1, void* stream) {
  if (!parts || !nr || nr % 8 || !N || N % (NPU_CORES * nr) || (mats != 1 && mats != 2) || (mats == 2 && !y1))
    return false;
  unpack_kernel<<<grid((uint64_t) parts * mats * NPU_TOKENS * N), kThreads, 0, st(stream)>>>(c, N, nr, mats, parts,
                                                                                               row0, y0, y1);
  return ok();
}
