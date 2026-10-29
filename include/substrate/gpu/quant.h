#ifndef QUANT_H
#define QUANT_H

#include <stdbool.h>
#include <stdint.h>

bool quant_known(uint32_t type);
bool quant_dequant(uint32_t type, const void *src, uint64_t values, float *dst, void *stream);

#endif

/*
quant
  the weights' types on the GPU (kernels/quant.h, the decoders the kernels share): Q8_0, Q5_K, Q6_K, IQ2_XXS, IQ2_XS,
  IQ4_NL, IQ1_M and BF16, by their GGUF type ids. each value the bits gguf-py's dequantize gives - the oracle's weights.

quant_known
  whether `type` is one of them.

quant_dequant
  `values` of `type` at `src` (device-readable: device memory or a pinned registration) as F32 into `dst`, queued on
  `stream` (a hipStream_t; NULL the default stream) and not waited for. false - nothing queued - for a type not known,
  `values` not whole blocks, or a launch the runtime refused.
*/
