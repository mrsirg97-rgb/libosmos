// the int8 lane's launchers shared by q8.hip (the matvec) and q8_experts.hip (the driver)
#pragma once
#include <hip/hip_runtime.h>

#include <cstdint>

#include "q8.h"

// every type the lane takes: q8.h's codes<T> (the table's QTYPE_Q8 rows: test_qtype_gpu holds the two equal)
#define Q8_TYPES(X)                                                                                              \
  X(kIQ2_XXS) X(kIQ2_XS) X(kIQ1_M) X(kIQ1_S) X(kIQ2_S) X(kIQ3_XXS) X(kQ2_K) X(kQ3_K) X(kQ4_K) X(kQ4_0) X(kQ5_1) \
  X(kQ8_0) X(kQ5_K) X(kQ6_K) X(kIQ4_NL) X(kIQ4_XS) X(kMXFP4) X(kNVFP4) X(kQ2_0)

namespace q8 {

// a fused gate + up's tables and its activation (q8.h's activate)
struct GateUp {
  const void* const *wg, *const *wu;
  const int32_t *cnt, *xrow, *xin;
  uint32_t n, rows, cols;
  const int8_t* q;
  const float* d;
  int act;
  ActArgs args;
  float* h;
  uint32_t hs;
};

// the matvec: at most MV_MAX_TOKENS rows an entry - a launch at max_cnt, or exact (a launch a count, any order)
bool group(uint32_t type, const void* const* w, const int32_t* cnt, const int32_t* xrow, const int32_t* xin, uint32_t n,
           uint32_t max_cnt, uint32_t rows, uint32_t cols, const int8_t* q, const float* d, float* y, uint32_t ys,
           bool exact, hipStream_t s);
bool gate_up(uint32_t type, const GateUp& g, uint32_t max_cnt, bool exact, hipStream_t s);

}  // namespace q8
