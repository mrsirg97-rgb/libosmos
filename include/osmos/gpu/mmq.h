#ifndef OSMOS_GPU_MMQ_H
#define OSMOS_GPU_MMQ_H

#include <stdbool.h>
#include <stdint.h>

#define MMQ_GROUP 32     /* an activation's int8 group: one scale a 32 values (Q4_K's sub-block) */
#define MMQ_MAX_TILES 4  /* mmq_group's column tiles a block at most (16 tokens each) */
#define MMQ_SLOTS 16     /* the layouts' slot tile: x8 / dx / sx hold whole tiles of slots */

#ifdef __HIPCC__
#define MMQ_HD __host__ __device__
#else
#define MMQ_HD
#endif
/* slot s's value c in x8: a tile of 16 slots a group at a time, each slot's 32 codes together (a WMMA lane's B) */
static inline MMQ_HD uint64_t mmq_x8_at(uint32_t s, uint32_t c, uint32_t cols)
{
  return (uint64_t)(s / MMQ_SLOTS) * MMQ_SLOTS * cols + (uint64_t)(c / MMQ_GROUP) * (MMQ_SLOTS * MMQ_GROUP) +
         (s % MMQ_SLOTS) * MMQ_GROUP + c % MMQ_GROUP;
}
/* slot s's group g in dx / sx: a tile's 16 slots side by side a group */
static inline MMQ_HD uint64_t mmq_dx_at(uint32_t s, uint32_t g, uint32_t groups)
{
  return (uint64_t)(s / MMQ_SLOTS) * MMQ_SLOTS * groups + (uint64_t)g * MMQ_SLOTS + s % MMQ_SLOTS;
}

bool mmq_x8_rows(const float *x, uint32_t xs, const int32_t *idx, uint32_t slots, uint32_t cols, int8_t *x8, float *dx,
                 float *sx, void *stream);
bool mmq_group(uint32_t type, const void *const *w, const int32_t *cnt, uint32_t n_mat, uint32_t max_cnt,
               uint32_t rows, uint32_t cols, const int8_t *x8, const float *dx, const float *sx, const uint32_t *soff,
               float *y, const uint32_t *yoff, uint32_t ys, void *stream);
bool mmq_known(uint32_t type);

#endif

/*
mmq (kernels/mmq.hip): the experts' matmul on the matrix cores' int8 (v_wmma_i32_16x16x16_iu8) - DESIGN.md, "0.6.0".

mmq_x8_rows
  slot s: row idx[s] of x (xs floats apart; -1: zeros) quantized a MMQ_GROUP columns at a time: dx[s][g] = max |x| /
  127 over the group (0 for an all-zero group), x8[s][c] = rint(x / dx) clamped to +-127 (0 where dx is), sx[s][g] =
  dx * the group's sum of x8 (F32, the int sum exact), at mmq_x8_at / mmq_dx_at: buffers of whole MMQ_SLOTS tiles
  (slots rounded up). cols a multiple of MMQ_GROUP. false for a shape it refuses.

mmq_group
  n_mat matrices of one type and shape (Q4_K alone, so far: mmq_known): matrix m's w[m] over its first cnt[m] (at most
  max_cnt) tokens - the slots soff[m] .. of x8 / dx / sx, soff[m] a multiple of MMQ_SLOTS (the slots past its count
  read, their outputs not written) - into y + yoff[m] + t * ys. a weight d sc q - dmin m (Q4_K's
  32-value group g), so a group's part of an output is (d sc) dx I - (dmin m) sx, I = sum of q x8 over the group (the
  WMMA's, exact): in F32, the products and the sums in a fixed order - K split over 4 waves by 256-value blocks (wave
  p's blocks p, p + 4, ...; each its groups in order: v = (d sc) dx, then acc = fma(v, I, acc), acc = fma(dmin m, -sx,
  acc)), then the waves' partial sums in wave order - so a token's bits are the same whatever its batch (test_mmq_gpu: a host
  reference of the same operations, and 1 / 16 / 37 / 64 tokens). rows a multiple of 16, cols of 256. w, cnt, soff,
  yoff on the device.
*/
