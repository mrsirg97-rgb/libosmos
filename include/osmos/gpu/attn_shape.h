#ifndef OSMOS_GPU_ATTN_SHAPE_H
#define OSMOS_GPU_ATTN_SHAPE_H

#include <stdint.h>

/* an attention's shape past its heads and dim (DESIGN.md, "vibes-flash: LFM3"): the values one gate scales - 1, a
   gate a value (Qwen's); the head's dim, a gate a head (LFM3's) - and its window: a token's keys at most, its own the
   last; 0 every key. with a window the cache is a ring of `window` rows, a position's row its position % window.
   a kernel taking one reads NULL as Qwen's: a gate a value, every key */
typedef struct AttnShape
{
  uint32_t gate_span, window;
} AttnShape;

#endif
