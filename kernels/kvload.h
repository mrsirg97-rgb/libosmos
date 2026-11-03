// a packed row's values read back (substrate/gpu/kvfmt.h): four at a time, each its code's value times its block's
// scale - round.h's own product, so the F32 the rounding gave. the attention's keys and values, the indexer's keys
#pragma once
#include "round.h"

extern "C" {
#include "substrate/gpu/kvfmt.h"
}

namespace kv {

typedef float f4 __attribute__((ext_vector_type(4)));

// four values' bytes as fetched, decoded later (a kernel's next stage fetched before this one's sums, decoded when
// staged: the fetch's latency hidden as an F32 load's was): F32 its 16 bytes; packed, the codes in a and the scale's
// byte in b
struct Raw {
  uint32_t a, b, c, d;
};

__device__ inline Raw fetch4(uint32_t fmt, const uint8_t* row, uint32_t dim, uint32_t d) {
  if (fmt == KV_FP8) return Raw{*(const uint32_t*) (row + d), row[dim + d / KV_FP8_BLOCK], 0, 0};
  if (fmt == KV_FP4_E4M3) return Raw{*(const uint16_t*) (row + d / 2), row[dim / 2 + d / KV_FP4_E4M3_BLOCK], 0, 0};
  if (fmt == KV_FP4_POW2) return Raw{*(const uint16_t*) (row + d / 2), row[dim / 2 + d / KV_FP4_POW2_BLOCK], 0, 0};
  const uint4 v = *(const uint4*) ((const float*) row + d);
  return Raw{v.x, v.y, v.z, v.w};
}

__device__ inline f4 decode4(uint32_t fmt, const Raw& r) {
  using namespace rnd;
  if (fmt == KV_FP8) {
    const float s = ue8m0_value(r.b);
    return f4{e4m3_value_finite(r.a & 0xFF) * s, e4m3_value_finite((r.a >> 8) & 0xFF) * s,
              e4m3_value_finite((r.a >> 16) & 0xFF) * s, e4m3_value_finite(r.a >> 24) * s};
  }
  if (fmt == KV_FP4_E4M3 || fmt == KV_FP4_POW2) {
    const float s = fmt == KV_FP4_E4M3 ? e4m3_value(r.b) : ue8m0_value(r.b);  // a scale may be NaN's: kept
    return f4{e2m1_value(r.a & 0xF) * s, e2m1_value((r.a >> 4) & 0xF) * s, e2m1_value((r.a >> 8) & 0xF) * s,
              e2m1_value(r.a >> 12) * s};
  }
  return f4{__uint_as_float(r.a), __uint_as_float(r.b), __uint_as_float(r.c), __uint_as_float(r.d)};
}

// values d .. d + 3 (d a multiple of 4) of a row of dim in format fmt at row, at once
__device__ inline f4 load4(uint32_t fmt, const uint8_t* row, uint32_t dim, uint32_t d) {
  return decode4(fmt, fetch4(fmt, row, dim, d));
}

}  // namespace kv
