// 0.7.0 spike: one AIE2P core's native bfp16 matmul rate - mlir-aie's matmul_vectorized_bfp16 (8x8x8 block MACs on
// bfp16ebs8: 8 mantissa bytes and a shared exponent byte per 8 values), 64x64x64 tiles in the core's memory, REPS back
// to back. the tiles' bytes are zero (the rate is not the data's). the fifo argument only triggers a run.
#define MATMUL_ONLY
#ifndef DIM_K
#define DIM_K 64
#endif
#include "linalg/mm_bfp.cc"

#ifndef REPS
#define REPS 64
#endif

alignas(64) static uint8_t A[64 * DIM_K * 9 / 8], B[DIM_K * 64 * 9 / 8], C[64 * 64 * 9 / 8];

extern "C" {
void mmbfp(int32_t *__restrict in, int32_t *__restrict out) {
  for (int r = 0; r < REPS; r++)
    matmul_vectorized_bfp16((bfp16ebs8 *)A, (bfp16ebs8 *)B, (bfp16ebs8 *)C);
  out[0] = C[0] + in[0];
}
}
