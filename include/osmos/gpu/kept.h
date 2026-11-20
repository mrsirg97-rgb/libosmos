#ifndef OSMOS_GPU_KEPT_H
#define OSMOS_GPU_KEPT_H

/* a step's kept rows' logits (src/mem/hip_kept.c): the families whose step keeps some rows' logits (slot[row] its
   row in logits, -1 not kept) - LFM3's, Qwen3.5's, Hy4's row lookup and greedy pick, one copy (0.11.2) */

#include <stdint.h>

/* row's logits (device, V floats), NULL for a row past rows or not kept */
const float *kept_row(const int32_t *slot, uint32_t rows, const float *logits, uint32_t V, uint32_t row);
/* the greatest token of each of rows row0 .. row0 + n - 1 (the lowest among equals; ops_argmax on stream, then the
   ids copied out on it and the stream synchronized) into ids: NULL, or why (rows past the step's, a row not kept or
   kept rows not one run of slots, a GPU call failed, a NaN in a row) */
const char *kept_greedy(const int32_t *slot, uint32_t rows, const float *logits, uint32_t V, uint32_t row0, uint32_t n,
                        int32_t *dev_ids, void *argmax, void *stream, int32_t *ids);

#endif
