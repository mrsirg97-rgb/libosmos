// the weights' types, decoded on the GPU: 8 values at a time, chunk k of a block (values 8k .. 8k + 7), each value
// the bits gguf-py's dequantize gives (the oracle's weights; tests/gpu/test_quant_gpu.hip). most products here are
// exact in F32 (an f16 scale's 11 bits times small integers: Q5_K's d * sc * q is 22 bits), so their order cannot
// change a bit; Q6_K's (25 bits) and the last subtract are rounded once each, as numpy rounds them. the products are
// __fmul_rn / __fsub_rn all the same, so no contraction into an FMA changes that when a decoder changes.
#pragma once
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "quant_grids.h"

namespace quant {

// ggml's type ids, the GGUF's
enum Type : uint32_t {
  kF32 = 0, kQ8_0 = 8, kQ5_K = 13, kQ6_K = 14, kIQ2_XXS = 16, kIQ2_XS = 17, kIQ4_NL = 20, kIQ1_M = 29, kBF16 = 30,
};

constexpr int kChunk = 8;
constexpr int kQK = 256;  // a K-quant's or an IQ type's values a block
constexpr int kQ32 = 32;  // Q8_0's and IQ4_NL's
constexpr float kIq1Delta = 0.125f;

template <Type T> struct Block;
template <> struct Block<kQ8_0> { static constexpr int values = kQ32, bytes = 34; };
template <> struct Block<kQ5_K> { static constexpr int values = kQK, bytes = 176; };
template <> struct Block<kQ6_K> { static constexpr int values = kQK, bytes = 210; };
template <> struct Block<kIQ2_XXS> { static constexpr int values = kQK, bytes = 66; };
template <> struct Block<kIQ2_XS> { static constexpr int values = kQK, bytes = 74; };
template <> struct Block<kIQ4_NL> { static constexpr int values = kQ32, bytes = 18; };
template <> struct Block<kIQ1_M> { static constexpr int values = kQK, bytes = 56; };
template <> struct Block<kF32> { static constexpr int values = kChunk, bytes = 4 * kChunk; };  // no block: a chunk
template <> struct Block<kBF16> { static constexpr int values = kChunk, bytes = 2 * kChunk; };  // no block: a chunk

// every block's size is even (34, 18, 66, 74, 210 ... bytes) and every row starts even: 16-bit loads, never wider
__device__ inline uint32_t ld16(const uint8_t* p) { return *(const uint16_t*) p; }
__device__ inline uint32_t ld32(const uint8_t* p) { return ld16(p) | ld16(p + 2) << 16; }
__device__ inline float f16(uint32_t bits) { return __half2float(__ushort_as_half((unsigned short) bits)); }
__device__ inline float mul(float a, float b) { return __fmul_rn(a, b); }

template <Type T> __device__ void decode(const uint8_t* b, int k, float* v);

// d * x
template <> __device__ inline void decode<kQ8_0>(const uint8_t* b, int k, float* v) {
  const float d = f16(ld16(b));
  for (int j = 0; j < kChunk; j++) v[j] = mul((float) (int8_t) b[2 + k * kChunk + j], d);
}

// (d * sc) * q - (dmin * m); 8 groups of 32, q 4 low bits from qs and a fifth from qh. a block is 176 bytes and
// every one 16-aligned (a tensor's data 32-aligned, its rows whole blocks): wide loads
template <> __device__ inline void decode<kQ5_K>(const uint8_t* b, int k, float* v) {
  const int g = k / 4, l0 = (k % 4) * kChunk;
  const uint32_t dd = *(const uint32_t*) b;
  const uint32_t s0 = *(const uint32_t*) (b + 4), s1 = *(const uint32_t*) (b + 8), s2 = *(const uint32_t*) (b + 12);
  auto byte = [&](int i) { return (int) (((i < 4 ? s0 : i < 8 ? s1 : s2) >> (8 * (i % 4))) & 0xFF); };
  const int s = g < 4 ? byte(g) & 0x3F : (byte(g + 4) & 0x0F) | ((byte(g - 4) >> 2) & 0x30);
  const int m = g < 4 ? byte(g + 4) & 0x3F : (byte(g + 4) >> 4) | ((byte(g) >> 2) & 0x30);
  const float d = mul(f16(dd & 0xFFFF), (float) s), dm = mul(f16(dd >> 16), (float) m);
  const uint64_t qh = *(const uint64_t*) (b + 16 + l0);
  const uint64_t qs = *(const uint64_t*) (b + 48 + (g / 2) * 32 + l0);
  for (int j = 0; j < kChunk; j++) {
    const int q = (int) ((qs >> (8 * j + 4 * (g % 2))) & 0x0F) | (int) (((qh >> (8 * j + g)) & 1) << 4);
    v[j] = __fsub_rn(mul(d, (float) q), dm);
  }
}

// (d * sc) * (q - 32); 16 scales of 16, q 4 bits from ql and 2 from qh
template <> __device__ inline void decode<kQ6_K>(const uint8_t* b, int k, float* v) {
  const uint8_t* ql = b;
  const uint8_t* qh = b + 128;
  const int8_t* sc = (const int8_t*) (b + 192);
  const float d0 = f16(ld16(b + 208));
  const int g = k / 4, h = g / 4, r = g % 4, l0 = (k % 4) * kChunk;
  const float d = mul(d0, (float) sc[(g * 32 + l0) / 16]);
  for (int j = 0; j < kChunk; j++) {
    const int l = l0 + j;
    const int lo = (ql[h * 64 + (r % 2) * 32 + l] >> (4 * (r / 2))) & 0x0F;
    const int hi = (qh[h * 32 + l] >> (2 * r)) & 0x03;
    v[j] = mul(d, (float) ((lo | hi << 4) - 32));
  }
}

// (d * (0.5 + s) * 0.25) * grid * sign; 8 groups of 32, each two words: 4 grid bytes, then 4 7-bit signs and s
template <> __device__ inline void decode<kIQ2_XXS>(const uint8_t* b, int k, float* v) {
  const uint8_t* q = b + 2 + (k / 4) * 8;
  const uint32_t aux = ld32(q + 4);
  const float db = mul(mul(f16(ld16(b)), 0.5f + (float) (aux >> 28)), 0.25f);
  const uint64_t grid = kIq2xxsGrid[q[k % 4]];
  const uint8_t signs = kKsigns[(aux >> (7 * (k % 4))) & 0x7F];
  for (int j = 0; j < kChunk; j++) {
    const float x = mul(db, (float) (uint8_t) (grid >> (8 * j)));
    v[j] = (signs >> j) & 1 ? -x : x;
  }
}

// as IQ2_XXS, a 16-bit word a chunk: 9 grid bits, 7 sign bits; a 4-bit scale every two chunks
template <> __device__ inline void decode<kIQ2_XS>(const uint8_t* b, int k, float* v) {
  const uint32_t q = ld16(b + 2 + 2 * k);
  const int s = (b[66 + k / 4] >> (4 * ((k / 2) % 2))) & 0x0F;
  const float db = mul(mul(f16(ld16(b)), 0.5f + (float) s), 0.25f);
  const uint64_t grid = kIq2xsGrid[q & 511];
  const uint8_t signs = kKsigns[q >> 9];
  for (int j = 0; j < kChunk; j++) {
    const float x = mul(db, (float) (uint8_t) (grid >> (8 * j)));
    v[j] = (signs >> j) & 1 ? -x : x;
  }
}

// d * kvalues[q]; values 0-15 the low nibbles, 16-31 the high
template <> __device__ inline void decode<kIQ4_NL>(const uint8_t* b, int k, float* v) {
  const float d = f16(ld16(b));
  const uint8_t* qs = b + 2 + (k % 2) * kChunk;
  for (int j = 0; j < kChunk; j++) v[j] = mul(d, (float) kIq4nlValues[(qs[j] >> (4 * (k / 2))) & 0x0F]);
}

// (d * (2s + 1)) * (grid + delta); d an f16 in the top nibbles of the four scale words, a 3-bit s each 16 values
template <> __device__ inline void decode<kIQ1_M>(const uint8_t* b, int k, float* v) {
  const uint8_t* qs = b;
  const uint8_t* qh = b + 32;
  const uint64_t sc = *(const uint64_t*) (b + 48);  // a block is 56 bytes, every one 8-aligned
  const uint32_t s0 = (uint32_t) sc & 0xFFFF, s1 = (uint32_t) (sc >> 16) & 0xFFFF;
  const uint32_t s2 = (uint32_t) (sc >> 32) & 0xFFFF, s3 = (uint32_t) (sc >> 48);
  const float d = f16((s0 >> 12) | ((s1 >> 8) & 0x00F0) | ((s2 >> 4) & 0x0F00) | (s3 & 0xF000));
  const int s = (int) ((sc >> (16 * (k / 8) + 3 * ((k / 2) % 4))) & 0x07);
  const float dl = mul(d, (float) (2 * s + 1));
  const uint32_t h = (qh[k / 2] >> (4 * (k % 2))) & 0x0F;
  const float delta = h & 0x08 ? -kIq1Delta : kIq1Delta;
  const uint64_t grid = kIq1sGrid[qs[k] | (h & 0x07) << 8];
  for (int j = 0; j < kChunk; j++) v[j] = mul(dl, (float) (int8_t) (grid >> (8 * j)) + delta);
}

// the top half of an F32; 8 values a chunk like the others
template <> __device__ inline void decode<kBF16>(const uint8_t* b, int k, float* v) {
  for (int j = 0; j < kChunk; j++) v[j] = __uint_as_float(ld16(b + 2 * (k * kChunk + j)) << 16);
}

template <> __device__ inline void decode<kF32>(const uint8_t* b, int k, float* v) {
  for (int j = 0; j < kChunk; j++) v[j] = __uint_as_float(ld32(b + 4 * (k * kChunk + j)));
}

}  // namespace quant
