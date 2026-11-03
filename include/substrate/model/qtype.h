#ifndef SUBSTRATE_QTYPE_H
#define SUBSTRATE_QTYPE_H

#include <stdint.h>

/* the paths that read a type: a role asks one of its tensor (DESIGN.md, "roles by capability") */
#define QTYPE_FILE (1u << 0)    /* a GGUF tensor may hold it (gguf_bytes) */
#define QTYPE_DECODE (1u << 1)  /* the GPU's decode to F32 / F16 (quant_dequant, quant_dequant_f16) */
#define QTYPE_MV (1u << 2)      /* the GPU's matvec (mv_group) */
#define QTYPE_MV_TOK (1u << 3)  /* the GPU's matvec, an expert's tokens (mv_group_tok) */
#define QTYPE_MM (1u << 4)      /* the GPU's prompt matmul (mm_run, mm_run2, mm_group, mm_col_tiles) */
#define QTYPE_Q8 (1u << 5)      /* the lane's int8 experts on the GPU (q8_group_tok, q8_gate_up) */
#define QTYPE_CPU_MV (1u << 6)  /* the CPU lane's matvec (cpu_mv_rows) */
#define QTYPE_CPU_Q8 (1u << 7)  /* the CPU lane's int8 experts (cpu/q8.c) */
#define QTYPE_EMBED (1u << 8)   /* a token's row of the embedding (ops_embed) */

#define QTYPE_IQ (QTYPE_FILE | QTYPE_DECODE | QTYPE_MV | QTYPE_MV_TOK | QTYPE_MM | QTYPE_Q8 | QTYPE_CPU_MV | QTYPE_CPU_Q8)

/* each type this build reads, once: X(name, ggml id, a block's values, a block's bytes, paths) */
#define QTYPES(X)                                                                                  \
  X(F32, 0, 1, 4, QTYPE_FILE | QTYPE_MV)                                                           \
  X(Q8_0, 8, 32, 34, QTYPE_FILE | QTYPE_DECODE | QTYPE_MV | QTYPE_MV_TOK | QTYPE_MM | QTYPE_EMBED) \
  X(Q5_K, 13, 256, 176, QTYPE_FILE | QTYPE_DECODE | QTYPE_MV | QTYPE_MM | QTYPE_EMBED)             \
  X(Q6_K, 14, 256, 210, QTYPE_FILE | QTYPE_DECODE | QTYPE_MV | QTYPE_MM | QTYPE_EMBED)             \
  X(IQ2_XXS, 16, 256, 66, QTYPE_IQ)                                                                \
  X(IQ2_XS, 17, 256, 74, QTYPE_IQ)                                                                 \
  X(IQ4_NL, 20, 32, 18, QTYPE_FILE | QTYPE_DECODE | QTYPE_MV)                                      \
  X(I32, 26, 1, 4, QTYPE_FILE)                                                                     \
  X(I64, 27, 1, 8, QTYPE_FILE)                                                                     \
  X(IQ1_M, 29, 256, 56, QTYPE_IQ)                                                                  \
  X(BF16, 30, 1, 2, QTYPE_FILE | QTYPE_DECODE | QTYPE_MV | QTYPE_EMBED)                            \
  X(MXFP4, 39, 32, 17, QTYPE_DECODE | QTYPE_MV | QTYPE_MV_TOK)

#define QTYPE_ID(name, id, values, bytes, paths) QTYPE_##name = id,
enum QType { QTYPES(QTYPE_ID) };
#undef QTYPE_ID

/* a row's numbers as constants, for code that switches at compile time (the kernels, the lane): QTYPE_<name>_VALUES,
   QTYPE_<name>_BYTES, QTYPE_<name>_PATHS */
#define QTYPE_SIZE(name, id, values, bytes, paths) \
  QTYPE_##name##_VALUES = values, QTYPE_##name##_BYTES = bytes, QTYPE_##name##_PATHS = paths,
enum QTypeSize { QTYPES(QTYPE_SIZE) };
#undef QTYPE_SIZE

typedef struct QTypeInfo
{
  uint32_t id;
  uint32_t values;
  uint32_t bytes;
  uint32_t paths;
  const char *name;
} QTypeInfo;

#ifdef __cplusplus
extern "C" {
#endif
const QTypeInfo *qtype_info(uint32_t id);
uint32_t qtype_paths(uint32_t id);
#ifdef __cplusplus
}
#endif

#endif

/*
qtype
  the one table of the quant types: what a GGUF's type id means here - a block's values and bytes - and which paths
  read it. every other list of types (gguf.c's sizes, weights.c's role types, kernels/quant.h's quant::Type, the
  lane's ids, the known-lists, the kernels' switches) is a projection of QTYPES or is checked against it
  (test_qtype, test_qtype_gpu). a type no row names is refused everywhere; a path a row names and a kernel does not
  take fails test_qtype_gpu.

qtype_info
  the row of ggml type `id`, or NULL for a type this build does not read.

qtype_paths
  the row's paths (QTYPE_*), 0 for a type this build does not read.
*/
