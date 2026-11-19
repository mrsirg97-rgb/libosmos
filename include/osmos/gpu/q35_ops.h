#ifndef OSMOS_GPU_Q35_OPS_H
#define OSMOS_GPU_Q35_OPS_H

#include <stdbool.h>
#include <stdint.h>

#include "osmos/gpu/attn_shape.h"

#ifdef __cplusplus
extern "C" {
#endif

/* the qwen35 family's own kernels (kernels/q35_attn.hip): a decode step's attention, flash decoding over q4x's
   Q8_0 cache - n_tok 1 .. Q35_ATTN_DECODE_MAX tokens at positions pos0 .., each its keys 0 .. its own; a warp a run of
   at least 16 keys, Q35_ATTN_PARTS runs at most a (token, head); out = softmax(q k / sqrt(dim)) v * sigmoid(gate), the
   gate's span and a window the shape's (attn_shape.h: NULL Qwen's) */
#define Q35_ATTN_DECODE_MAX 4
#define Q35_ATTN_PARTS 512
#define Q35_ATTN_GQA_MAX 8
uint64_t q35_attn_decode_scratch(uint32_t heads, uint32_t max_tokens); /* floats */
bool q35_attn_decode(const float *q, const void *kc, const void *vc, uint32_t n_tok, uint32_t heads, uint32_t kv_heads,
                     uint32_t dim, const AttnShape *shape, uint32_t pos0, const float *gate, float *scratch, float *out,
                     void *stream);
/* one token's attention over its read alone (osmos/lsa.h's lsa_read: a device list, its count *list_len the device's,
   at most list_max): the same kernels, the keys the list's - nothing read back */
bool q35_attn_read(const float *q, const void *kc, const void *vc, uint32_t heads, uint32_t kv_heads,
                   uint32_t dim, const AttnShape *shape, const uint32_t *list, const uint32_t *list_len,
                   uint32_t list_max, const float *gate, float *scratch, float *out, void *stream);

#ifdef __cplusplus
}
#endif

#endif
