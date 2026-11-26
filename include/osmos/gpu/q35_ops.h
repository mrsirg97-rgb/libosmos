#ifndef OSMOS_GPU_Q35_OPS_H
#define OSMOS_GPU_Q35_OPS_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/gpu/attn_shape.h"

#ifdef __cplusplus
extern "C" {
#endif

/* the qwen35 family's decode attention's bounds (its kernels the attention core's now: kernels/attn_core_q35.h): a decode step's attention, flash decoding over q4x's
   Q8_0 cache - n_tok 1 .. Q35_ATTN_DECODE_MAX tokens at positions pos0 .., each its keys 0 .. its own; a warp a run of
   at least 16 keys, Q35_ATTN_PARTS runs at most a (token, head); out = softmax(q k / sqrt(dim)) v * sigmoid(gate), the
   gate's span and a window the shape's (attn_shape.h: NULL Qwen's) */
#define Q35_ATTN_DECODE_MAX 4
#define Q35_ATTN_PARTS 512
#define Q35_ATTN_GQA_MAX 8

#ifdef __cplusplus
}
#endif

#endif
