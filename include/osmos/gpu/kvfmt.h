#ifndef KVFMT_H
#define KVFMT_H

#include <stdint.h>

#if defined(__HIPCC__)
#define KVFMT_FN __host__ __device__ static inline /* the kernels read rows by these too */
#else
#define KVFMT_FN static inline
#endif

/* a row of the attention's state, as the model rounds it (ref.py): its values' codes, then a scale byte a block */
typedef enum KvFmt
{
  KV_F32 = 0,      /* 4 bytes a value: a row not kept (the drafter's block's own) */
  KV_FP8 = 1,      /* the window's rows (act_quant): E4M3 codes, a UE8M0 (power of two) scale a 32 */
  KV_FP4_E4M3 = 2, /* the compressed rows (fp4_act_quant, block 16): E2M1 codes two a byte, an E4M3 scale a 16 */
  KV_FP4_POW2 = 3, /* the indexer's keys (fp4_act_quant, block 32): E2M1 codes two a byte, a UE8M0 scale a 32 */
} KvFmt;

#define KV_FP8_BLOCK 32
#define KV_FP4_E4M3_BLOCK 16
#define KV_FP4_POW2_BLOCK 32
#define KV_UE8M0_BIAS 127 /* a power-of-two scale 2^e as the byte e + 127 */

/* a row's bytes: dim values (a multiple of the format's block) */
KVFMT_FN uint64_t kv_row_bytes(KvFmt f, uint32_t dim)
{
  switch (f)
  {
  case KV_F32: return (uint64_t)dim * 4;
  case KV_FP8: return (uint64_t)dim + dim / KV_FP8_BLOCK;
  case KV_FP4_E4M3: return (uint64_t)dim / 2 + dim / KV_FP4_E4M3_BLOCK;
  case KV_FP4_POW2: return (uint64_t)dim / 2 + dim / KV_FP4_POW2_BLOCK;
  }
  return 0;
}

KVFMT_FN uint32_t kv_block(KvFmt f)
{
  return f == KV_FP8 ? KV_FP8_BLOCK : f == KV_FP4_E4M3 ? KV_FP4_E4M3_BLOCK : f == KV_FP4_POW2 ? KV_FP4_POW2_BLOCK : 1;
}

#endif

/*
kvfmt
  the attention's state in the formats the model rounds it to - its window's keys FP8, its compressed rows and the
  indexer's keys FP4 (DESIGN.md, "the KV in the model's formats") - so a sequence keeps what DeepSeek's design keeps
  (~0.9 KB a token), not 4 bytes a value. a row: dim codes (a byte each, or two a byte), then dim / block scale bytes.
  a value is its code's value times its block's scale, one product: the bits the rounding gave (round.h's e4m3() * s,
  e2m1() * s), so a row kept and read back is the F32 it was.

kv_row_bytes
  a row's bytes in format f (dim a multiple of its block): FP8 528 for 512 values, FP4_E4M3 288, FP4_POW2 68 for 128.
*/
