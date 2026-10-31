#ifndef CPU_EXPERT_H
#define CPU_EXPERT_H

#include <stdbool.h>
#include <stdint.h>

#define CPU_LANES 32 /* the GPU matvec's wave (kernels/mv.h): a row's sums split over 32 lanes */
#define CPU_CHUNK 8  /* a decode's values (kernels/quant.h kChunk) */

bool cpu_mv_known(uint32_t type);
bool cpu_mv_rows(uint32_t type, const void *w, uint32_t cols, uint32_t r0, uint32_t r1, const float *x, uint32_t xs,
                 uint32_t n, float *y, uint32_t ys);

#endif

/*
cpu expert
  the CPU lane's matrix-vector products (forward_lane): an expert's matrices on the CPU with the bits the GPU's
  matvec gives them (kernels/mv.h's rows_of) - each value decoded as kernels/quant.h decodes it (the same products,
  each rounded; built -ffp-contract=off), lane l of 32 summing its chunks l, l + 32, ... in order, 8 fmaf a chunk,
  then the lanes' sums by the same xor tree (16, 8, 4, 2, 1). which side computes an expert changes no bit.

cpu_mv_known
  whether `type` (a GGUF type id) has a CPU decode: the experts' IQ1_M, IQ2_XXS, IQ2_XS.

cpu_mv_rows
  rows r0 .. r1 of the matrix at `w` (a GGUF tensor's layout: rows of cols / 256 blocks) times n tokens' x (token t's
  cols values at x + t * xs) into y (token t's row r at y + t * ys + r). cols a multiple of 256 and of 8 x 32;
  false, nothing written, for a type without a decode or a shape it cannot take. one thread's: callers split the
  rows.
*/
