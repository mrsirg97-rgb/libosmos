#ifndef QUANT_H
#define QUANT_H

#include <stdbool.h>
#include <stdint.h>

bool quant_known(uint32_t type);
bool quant_dequant(uint32_t type, const void *src, uint64_t values, float *dst, void *stream);
bool quant_dequant_f16(uint32_t type, const void *src, uint64_t values, void *dst, void *stream);
bool quant_repack_q5k(const void *src, uint32_t rows, uint32_t cols, void *dst, void *stream);

#include "substrate/model/qtype.h"

#define QUANT_Q5_K QTYPE_Q5_K               /* the table's (qtype.h) */
#define QUANT_Q5_K_VALUES QTYPE_Q5_K_VALUES
#define QUANT_Q5_K_BYTES QTYPE_Q5_K_BYTES
#define QUANT_Q5_KR (0x100 | QUANT_Q5_K)    /* Q5_K repacked (quant_repack_q5k): mm's and mv's type */
#define QUANT_Q5_KR_BYTES_A_BLOCK 180       /* a row's share of a tile's block: +2.3% */
#define QUANT_Q5_KR_BYTES(rows, cols) ((uint64_t)(rows) * ((cols) / QUANT_Q5_K_VALUES) * QUANT_Q5_KR_BYTES_A_BLOCK)

#endif

/*
quant
  the weights' types on the GPU (kernels/quant.h, the decoders the kernels share): Q8_0, Q5_K, Q6_K, IQ2_XXS, IQ2_XS,
  IQ4_NL, IQ1_M, BF16, MXFP4 (DSpark's experts) and MXFP8 (unsloth's Engram tables), by their GGUF type ids. each value the bits gguf-py's dequantize
  gives - the oracle's weights.

quant_known
  whether `type` is one of them.

quant_dequant
  `values` of `type` at `src` (device-readable: device memory or a pinned registration) as F32 into `dst`, queued on
  `stream` (a hipStream_t; NULL the default stream) and not waited for. false - nothing queued - for a type not known,
  `values` not whole blocks, or a launch the runtime refused.

quant_dequant_f16
  as quant_dequant, each value as the f16 the matmul's WMMA reads (kernels/quant.h's decode_h: quant_dequant's value
  rounded to nearest, ties to even - IQ2_XXS and IQ2_XS by a path of their own, the same bits), into `dst` (2 bytes a
  value).

quant_repack_q5k
  a Q5_K matrix (rows of cols, row-major blocks: a GGUF's) at `src` into QUANT_Q5_KR at `dst` (QUANT_Q5_KR_BYTES of
  it; not `src`), queued on `stream`: a tile of 16 rows a block of 256 columns at a time, each lane's values of mm's
  one-token step side by side and the scales unpacked (kernels/quant.h, kQ5_KR) - the same values, so mm_run,
  mm_group and mv_strided give QUANT_Q5_KR the bits they give Q5_K. false, nothing queued, for rows not a multiple
  of 16, cols not of 256, or src and dst the same. QUANT_Q5_KR is not a quant_dequant type.
*/
