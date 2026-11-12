// 0.7.0 spike: one AIE2P core's bf16 matmul rate - 64x64x64 tiles in the core's own memory, REPS of them back to back
// (mlir-aie's matmul_vectorized kernels: 4x8x8 native bf16, or with AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 the
// 8x8x8 bfp16-emulated one). the fifo argument only triggers a run.
#include "linalg/mm_aie2p.h"

#ifndef REPS
#define REPS 64
#endif

alignas(64) static bfloat16 A[64 * 64], B[64 * 64], C[64 * 64];

extern "C" {
void mmbench(int32_t *__restrict in, int32_t *__restrict out) {
  for (int i = 0; i < 64 * 64; i++) A[i] = (bfloat16)(float)((i * 7) % 13 - 6), B[i] = (bfloat16)(float)((i * 5) % 11 - 5);
  for (int i = 0; i < 64 * 64; i++) C[i] = (bfloat16)0.0f;
  event0();
  for (int r = 0; r < REPS; r++) {
#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
    matmul_vectorized_8x8x8_bf16_bf16<64, 64, 64>(A, B, C);
#else
    matmul_vectorized_4x8x8_bf16_bf16<64, 64, 64>(A, B, C);
#endif
  }
  event1();
  out[0] = (int32_t)(float)C[0] + in[0];
}
}
