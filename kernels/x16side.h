// an op's output written beside as mm's f16 activations (q4x_ops.h's Q4xX16): x16_put's rounding and layout, so the
// bits mm_x16 would make of it
#pragma once
#include "mm.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

// element i of an output of rows side.cols long
__device__ inline void x16_side(const Q4xX16& side, uint64_t i, float v) {
  if (side.x16) mm::x16_put((_Float16*) side.x16, (uint32_t) (i / side.cols), (uint32_t) (i % side.cols), side.cols, v,
                            side.overflow);
}

inline Q4xX16 side_of(const Q4xX16* s) { return s ? *s : Q4xX16{nullptr, nullptr, 1}; }
