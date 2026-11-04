#ifndef CPU_Q8_H
#define CPU_Q8_H

#include <stdbool.h>
#include <stdint.h>

#define CPU_Q8_TOKENS 8 /* a row's tokens decoded once for (the GPU's widest q8_group_tok) */

bool cpu_q8_ready(void);
bool cpu_q8_round(const float *x, uint32_t rows, uint32_t cols, uint32_t xs, int8_t *q, float *d);
bool cpu_q8_rows(uint32_t type, const void *w, uint32_t cols, uint32_t r0, uint32_t r1, const int8_t *q, const float *d,
                 uint32_t n, float *y, uint32_t ys);

#endif

/*
cpu q8
  the CPU lane's experts over int8 inputs: kernels/lane.h's contract (q8_round, q8_group_tok in
  include/osmos/gpu/q8.h) on AVX-512 - the same rounding, the same exact integer sum for each half-block, the same
  float steps in the same order, the blocks to the same lanes and the lanes' sums by the same xor tree. which side
  computes an expert changes no bit.

cpu_q8_ready
  whether this CPU runs the kernels (AVX-512 F, BW, VL, DQ and VNNI). false: cpu_q8_round and cpu_q8_rows refuse.

cpu_q8_round
  q8_round's bytes and scales: rows of x (row r at x + r * xs, cols values) a block of 32 at a time to int8 q (row r at
  q + r * cols) and its scales d (row r at d + r * cols / 32). cols a multiple of 32, xs >= cols; false, nothing
  written, otherwise or on a CPU without the kernels.

cpu_q8_rows
  rows r0 .. r1 of the expert matrix at `w` (a GGUF tensor's layout: rows of cols / 256 blocks; type IQ2_XXS, IQ2_XS
  or IQ1_M - lane_q8_type) times n tokens' int8 x (token t's at q + t * cols, its scales at d + t * cols / 32, as
  cpu_q8_round writes them) into y (token t's row r at y + t * ys + r): q8_group_tok's bits. each token's sums its
  own: rows split anywhere, tokens one at a time or together, the same bits. cols a multiple of 256, n >= 1; false,
  nothing written, for another type, a shape it cannot take, or a CPU without the kernels. one thread's: callers
  split the rows.
*/
