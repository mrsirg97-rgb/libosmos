#ifndef OSMOS_GPU_KEPT_H
#define OSMOS_GPU_KEPT_H

/* a step's kept rows (src/mem/hip_kept.c): which rows of a family's last step kept logits, and the one way to read a
   row and pick its greatest token - every family's (the engine reads a family's view, never its copies) */

#include <stdbool.h>
#include <stdint.h>

typedef struct KeptView
{
  const int32_t *slot;  /* [rows]: a row's slot in logits, or -1 (not kept) */
  uint32_t rows;        /* the step's rows */
  const float *logits;  /* device, [slots][V] */
  uint32_t V;
  int32_t *dev_ids;     /* device scratch for a run's picks (a step's rows at most) */
  void *argmax;         /* ops_argmax's scratch */
  void *stream;         /* the stream the pick is queued on (NULL: the default) */
  int32_t *host_ids;    /* pinned host memory the picks are copied into, or NULL: straight into the caller's ids */
  bool (*settle)(void *ctx); /* the family's wait for the stream's work (DeepSeek's spins on a decode's), */
  void *settle_ctx;          /* or NULL: hipStreamSynchronize */
} KeptView;

/* row's logits (device, V floats), NULL for a row past rows or not kept */
const float *kept_row(const KeptView *k, uint32_t row);
/* the greatest token of each of rows row0 .. row0 + n - 1 (the lowest among equals; ops_argmax on the view's stream,
   the ids copied out, the family's wait) into ids: NULL, or why (rows past the step's, a row not kept or kept rows
   not one run of slots, a GPU call failed, a NaN in a row) */
const char *kept_greedy(const KeptView *k, uint32_t row0, uint32_t n, int32_t *ids);

#endif
