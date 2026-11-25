// an MLA cache row's layout (Hy4's: hy4.hip's writer, the attention core's reader - attn_core_mla.h): the latent
// F32 or ggml's Q8_0 blocks, then the rope part F32
#pragma once
#include <cstdint>

#include <hip/hip_fp16.h>

namespace mla {

// a Q8_0 row's value i (ggml's block: an f16 scale, 32 int8 codes)
constexpr uint32_t kQ8Block = 32, kQ8Bytes = 34;
constexpr uint32_t kMaxRope = 128;  // the rope part's values at most (a cache row latent + this)

__device__ inline float q8_value(const uint8_t* row, uint32_t i) {
  const uint8_t* b = row + (i / kQ8Block) * kQ8Bytes;
  const float d = __half2float(__ushort_as_half(*(const unsigned short*) b));
  return __fmul_rn((float) (int8_t) b[2 + i % kQ8Block], d);
}

// a cache row: F32 [latent | rope], or q8: the latent Q8_0 (ggml's blocks: an f16 d = amax / 127, 32 codes
// round(x / d)) then the rope part F32 (vLLM's fp8_ds_mla keeps it wide too) - kQ8Bytes a 32 latent values
__host__ __device__ inline uint64_t mla_row_bytes(uint32_t latent, uint32_t rope_dim, bool q8) {
  return q8 ? (uint64_t) latent / 32 * 34 + (uint64_t) rope_dim * 4 : (uint64_t) (latent + rope_dim) * 4;
}

}  // namespace mla
