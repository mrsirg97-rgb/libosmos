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
#include "round.h"
#include "osmos/model/qtype.h"

namespace quant {

// the table's ids (osmos/model/qtype.h)
enum Type : uint32_t {
  kF32 = QTYPE_F32, kQ8_0 = QTYPE_Q8_0, kQ5_K = QTYPE_Q5_K, kQ6_K = QTYPE_Q6_K, kIQ2_XXS = QTYPE_IQ2_XXS,
  kIQ2_XS = QTYPE_IQ2_XS, kIQ1_S = QTYPE_IQ1_S, kIQ4_NL = QTYPE_IQ4_NL, kIQ2_S = QTYPE_IQ2_S, kIQ1_M = QTYPE_IQ1_M,
  kBF16 = QTYPE_BF16, kMXFP4 = QTYPE_MXFP4, kMXFP8 = QTYPE_MXFP8, kIQ3_XXS = QTYPE_IQ3_XXS, kIQ4_XS = QTYPE_IQ4_XS,
  kQ2_0 = QTYPE_Q2_0, kQ2_K = QTYPE_Q2_K,
  kQ5_KR = 0x100 | kQ5_K,  // Q5_K repacked a 16-row tile at a time (below; not a GGUF's: the forward's, at load)
};

constexpr int kChunk = 8;
constexpr int kQK = 256;  // a K-quant's or an IQ type's values a block
constexpr int kQ32 = 32;  // Q8_0's and IQ4_NL's
constexpr int kQ64 = 64;  // Q2_0's
constexpr float kIq1Delta = 0.125f;

template <Type T> struct Block;
#define QUANT_BLOCK(name) \
  template <> struct Block<k##name> { static constexpr int values = QTYPE_##name##_VALUES, bytes = QTYPE_##name##_BYTES; };
QUANT_BLOCK(Q8_0) QUANT_BLOCK(Q5_K) QUANT_BLOCK(Q6_K) QUANT_BLOCK(IQ2_XXS) QUANT_BLOCK(IQ2_XS) QUANT_BLOCK(IQ4_NL)
QUANT_BLOCK(IQ1_M) QUANT_BLOCK(MXFP4) QUANT_BLOCK(MXFP8) QUANT_BLOCK(IQ1_S) QUANT_BLOCK(IQ2_S) QUANT_BLOCK(IQ3_XXS)
QUANT_BLOCK(IQ4_XS) QUANT_BLOCK(Q2_0) QUANT_BLOCK(Q2_K)
#undef QUANT_BLOCK
template <> struct Block<kQ5_KR> { static constexpr int values = kQK, bytes = 180; };  // a row's share of a tile's
// no block: a chunk of the table's one-value "blocks"
template <> struct Block<kF32> { static constexpr int values = kChunk, bytes = QTYPE_F32_BYTES * kChunk; };
template <> struct Block<kBF16> { static constexpr int values = kChunk, bytes = QTYPE_BF16_BYTES * kChunk; };
static_assert(QTYPE_F32_VALUES == 1 && QTYPE_BF16_VALUES == 1, "F32 and BF16 decode a value at a time");
static_assert(Block<kQ8_0>::values == kQ32 && Block<kIQ4_NL>::values == kQ32 && Block<kMXFP4>::values == kQ32 &&
                  Block<kMXFP8>::values == kQ32 &&
                  Block<kQ5_K>::values == kQK && Block<kQ6_K>::values == kQK && Block<kIQ1_M>::values == kQK &&
                  Block<kIQ2_XXS>::values == kQK && Block<kIQ2_XS>::values == kQK && Block<kIQ1_S>::values == kQK &&
                  Block<kIQ2_S>::values == kQK && Block<kIQ3_XXS>::values == kQK &&
                  Block<kIQ4_XS>::values == kQK && Block<kQ2_0>::values == kQ64 && Block<kQ2_K>::values == kQK,
              "the decoders' block lengths");

// every block's size is even (34, 18, 66, 74, 210 ... bytes; MXFP4's and MXFP8's odd: byte loads only) and every row
// starts even: 16-bit loads, never wider
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

// Q5_K repacked (kQ5_KR): rows in tiles of 16 (the matmul's), a tile's block of 256 columns kQ5rTile bytes - the
// 16 rows' d and dmin (kQ5rHead: 4 B a row), then a segment a sub-block g of 32 columns (kQ5rSeg): the 32 slots of
// 16 values (slot half * 16 + r: row r's chunks 4g + half and 4g + 2 + half - a lane's two chunks of mm's one-token
// step) as 8 B of low nibbles (value i at nibble i) at slot * 8, their fifth bits (bit i) as 2 B at 256 + slot * 2,
// then each row's 6-bit (s, m) as 2 B at 320 + r * 2. 16 x 180 bytes a tile's block (+2.3%): a wave's loads of a step
// side by side, the scales unpacked once. the same (d * sc) * q - dmin * m: decode<kQ5_K>'s bits
constexpr int kQ5rRows = 16, kQ5rHead = kQ5rRows * 4, kQ5rSeg = 352, kQ5rTile = kQ5rRows * 180;
constexpr int kQ5rQh = 256, kQ5rSm = 320;
static_assert(kQ5rHead + 8 * kQ5rSeg == kQ5rTile, "a tile's block: its head and 8 segments");

// where a row's chunk c is - a block's bytes and the chunk in it (decode<T>'s b and k), in two halves: the row's base
// (a matvec keeps one a row) and its chunk's place from it. Q5_KR's base its tile's, the chunk its block's in it at
// (row % 16) * 32 + c's chunk in the block; any other type's the row's start and its block's
template <Type T>
__device__ inline const uint8_t* row_base(const uint8_t* w, uint32_t row, uint32_t cols) {
  using B = Block<T>;
  const uint64_t nb = cols / B::values;
  if constexpr (T == kQ5_KR) return w + (uint64_t) (row / kQ5rRows) * nb * kQ5rTile;
  else return w + (uint64_t) row * nb * B::bytes;
}

template <Type T>
__device__ inline const uint8_t* chunk_at(const uint8_t* base, uint32_t row, uint32_t c, int* k) {
  using B = Block<T>;
  constexpr int per = B::values / kChunk;
  if constexpr (T == kQ5_KR) {
    *k = (int) (row % kQ5rRows) * per + (int) (c % per);
    return base + (uint64_t) (c / per) * kQ5rTile;
  } else {
    (void) row;
    *k = (int) (c % per);
    return base + (uint64_t) (c / per) * B::bytes;
  }
}

template <Type T>
__device__ inline const uint8_t* at(const uint8_t* w, uint32_t row, uint32_t cols, uint32_t c, int* k) {
  return chunk_at<T>(row_base<T>(w, row, cols), row, c, k);
}

template <> __device__ inline void decode<kQ5_KR>(const uint8_t* b, int k, float* v) {
  const int r = k / 32, c = k % 32, g = c / 4, half = c % 2, gg = (c % 4) / 2, slot = half * kQ5rRows + r;
  const uint8_t* seg = b + kQ5rHead + g * kQ5rSeg;
  const uint32_t dd = *(const uint32_t*) (b + r * 4);
  const uint32_t qs = (uint32_t) (*(const uint64_t*) (seg + slot * 8) >> (32 * gg));
  const uint32_t qh = ld16(seg + kQ5rQh + slot * 2) >> (8 * gg), sm = ld16(seg + kQ5rSm + r * 2);
  const float d = mul(f16(dd & 0xFFFF), (float) (sm & 0xFF)), dm = mul(f16(dd >> 16), (float) (sm >> 8));
  for (int j = 0; j < kChunk; j++) {
    const int q = (int) ((qs >> (4 * j)) & 0x0F) | (int) (((qh >> j) & 1) << 4);
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

// (d * (0.5 + s) * 0.5) * grid * sign; 8 groups of 32: 8 grid bytes (4 values each), and after the 64 a word a
// group - 4 7-bit signs, then s
template <> __device__ inline void decode<kIQ3_XXS>(const uint8_t* b, int k, float* v) {
  const uint8_t* q = b + 2 + (k / 4) * 8 + (k % 4) * 2;
  const uint32_t aux = ld32(b + 2 + 64 + (k / 4) * 4);
  const float db = mul(mul(f16(ld16(b)), 0.5f + (float) (aux >> 28)), 0.5f);
  const uint8_t signs = kKsigns[(aux >> (7 * (k % 4))) & 0x7F];
  for (int j = 0; j < kChunk; j++) {
    const float x = mul(db, (float) (uint8_t) (kIq3xxsGrid[q[j / 4]] >> (8 * (j % 4))));
    v[j] = (signs >> j) & 1 ? -x : x;
  }
}

// (d * (ls - 32)) * IQ4_NL's value; 8 groups of 32, each a 6-bit ls (its low 4 bits a nibble of bytes 4 .. 7, its
// high 2 of the 16-bit word at 2), its values IQ4_NL's layout in 16 bytes
template <> __device__ inline void decode<kIQ4_XS>(const uint8_t* b, int k, float* v) {
  const int g = k / 4, j0 = (k % 4) * kChunk;
  const int ls = ((b[4 + g / 2] >> (4 * (g % 2))) & 0x0F) | ((ld16(b + 2) >> (2 * g)) & 0x03) << 4;
  const float dl = mul(f16(ld16(b)), (float) (ls - 32));
  const uint8_t* qs = b + 8 + 16 * g + j0 % 16;
  for (int j = 0; j < kChunk; j++) v[j] = mul(dl, (float) kIq4nlValues[(qs[j] >> (4 * (j0 / 16))) & 0x0F]);
}

// d * (code - 1): 64 values, an f16 then 2-bit codes, value i bits 2 (i % 4) of byte i / 4
template <> __device__ inline void decode<kQ2_0>(const uint8_t* b, int k, float* v) {
  const float d = f16(ld16(b));
  const uint8_t* qs = b + 2 + 2 * k;
  for (int j = 0; j < kChunk; j++) v[j] = mul(d, (float) (((qs[j / 4] >> (2 * (j % 4))) & 0x03) - 1));
}

// (d * sc) * q - (dmin * m); 16 groups of 16, each a byte of scales (sc its low nibble, m its high), q 2 bits:
// value i's of byte qs[32 (i / 128) + i % 32] at shift 2 ((i % 128) / 32). bytes: scales (16), qs (64), d, dmin.
// the products exact in F32 (an f16 times 4 bits, times 2): the subtract alone rounds, as numpy's
template <> __device__ inline void decode<kQ2_K>(const uint8_t* b, int k, float* v) {
  const int g = k / 2, l0 = (k % 4) * kChunk, shift = 2 * ((k % 16) / 4);
  const uint8_t* qs = b + 16 + 32 * (k / 16) + l0;
  const float dl = mul(f16(ld16(b + 80)), (float) (b[g] & 0x0F));
  const float ml = mul(f16(ld16(b + 82)), (float) (b[g] >> 4));
  for (int j = 0; j < kChunk; j++) v[j] = __fsub_rn(mul(dl, (float) ((qs[j] >> shift) & 0x03)), ml);
}

// DSpark's experts: an E8M0 scale e (2^(e - 128), exact at every e: the low ones subnormal), then E2M1 codes as
// IQ4_NL lays them out (value j < 16 byte j's low nibble, 16 + j its high), worth E2M1 doubled {0, 1, 2, 3, 4, 6, 8,
// 12} and their negatives
template <> __device__ inline void decode<kMXFP4>(const uint8_t* b, int k, float* v) {
  const float d = ldexpf(1.0f, (int) b[0] - 128);
  const uint8_t* qs = b + 1 + (k % 2) * kChunk;
  for (int j = 0; j < kChunk; j++) v[j] = mul(d, (float) kMxfp4Values[(qs[j] >> (4 * (k / 2))) & 0x0F]);
}

// unsloth's Engram tables: an E8M0 scale e (2^(e - 127)), then 32 E4M3 codes, value j byte 1 + j (a code 0x7F / 0xFF
// NaN, as torch's float8_e4m3fn); each product exact (4 significant bits, the smallest 2^-136: no rounding)
template <> __device__ inline void decode<kMXFP8>(const uint8_t* b, int k, float* v) {
  const float d = ldexpf(1.0f, (int) b[0] - 127);
  for (int j = 0; j < kChunk; j++) v[j] = mul(d, rnd::e4m3_value(b[1 + k * kChunk + j]));
}

// (d * (2s + 1)) * (grid + delta); d an f16 in the top nibbles of the four scale words, a 3-bit s each 16 values
// IQ2_XS's sums with a grid of 1024 and the signs stored: d, qs (32 index bytes), signs (32), qh (2 high index bits
// for each 8), scales (a nibble for each 16)
template <> __device__ inline void decode<kIQ2_S>(const uint8_t* b, int k, float* v) {
  const int s = (b[74 + k / 4] >> (4 * ((k / 2) % 2))) & 0x0F;
  const float db = mul(mul(f16(ld16(b)), 0.5f + (float) s), 0.25f);
  const uint64_t grid = kIq2sGrid[b[2 + k] | ((b[66 + k / 4] >> (2 * (k % 4))) & 0x03) << 8];
  const uint8_t signs = b[34 + k];
  for (int j = 0; j < kChunk; j++) {
    const float x = mul(db, (float) (uint8_t) (grid >> (8 * j)));
    v[j] = (signs >> j) & 1 ? -x : x;
  }
}
// IQ1_M's grid with one scale and delta for each 32: d, qs (32 low index bytes), qh (a u16 for each 32: 3 high index
// bits for each 8, the scale in bits 12-14, the delta's sign in bit 15)
template <> __device__ inline void decode<kIQ1_S>(const uint8_t* b, int k, float* v) {
  const uint32_t h = ld16(b + 34 + 2 * (k / 4));
  const float dl = mul(f16(ld16(b)), (float) (2 * ((h >> 12) & 0x07) + 1));
  const float delta = h & 0x8000 ? -kIq1Delta : kIq1Delta;
  const uint64_t grid = kIq1sGrid[b[2 + k] | ((h >> (3 * (k % 4))) & 0x07) << 8];
  for (int j = 0; j < kChunk; j++) v[j] = mul(dl, (float) (int8_t) (grid >> (8 * j)) + delta);
}
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

// a chunk as the f16s the matmul's WMMA reads: each value decode<T>'s, rounded to f16 (nearest, ties to even) - the
// same bits by either path below
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef uint32_t u4 __attribute__((ext_vector_type(4)));

// the F32 value first, then f16: the compiler would fuse decode's last product and the rounding into one
// v_fma_mixlo_f16 (a * b + 0, rounded once to f16), which is a value of its own - +0 for a -0, and for Q6_K's 25-bit
// products not round(round(x)) (measured: 40690 of test_quant_gpu's 3670016 Q6_K values). the empty asm keeps them
// two; not volatile, so the schedule around it stays free (a volatile one: a decode token 74 -> 85 ms)
template <Type T>
__device__ inline h8 decode_h(const uint8_t* b, int k) {
  float v[kChunk];
  decode<T>(b, k, v);
  h8 h;
  for (int j = 0; j < kChunk; j++) {
    asm("" : "+v"(v[j]));
    h[j] = (_Float16) v[j];
  }
  return h;
}

// kKsigns' 128 patterns as f16 sign bits: pattern i's value j negative - bit 15 of half j. kKsigns[i] is i with an
// eighth bit that makes its ones even (built here from that rule; test_quant_gpu checks every value against gguf-py)
struct SignMasks {
  uint32_t m[128][kChunk / 2];
};
constexpr SignMasks make_sign_masks() {
  SignMasks t{};
  for (int i = 0; i < 128; i++) {
    int ones = 0;
    for (int b = 0; b < 7; b++) ones += (i >> b) & 1;
    const int s = i | (ones & 1) << 7;
    for (int p = 0; p < kChunk / 2; p++)
      t.m[i][p] = (uint32_t) ((s >> (2 * p)) & 1) << 15 | (uint32_t) ((s >> (2 * p + 1)) & 1) << 31;
  }
  return t;
}
static __constant__ SignMasks kSignMasks = make_sign_masks();

// IQ2's values as f16: each |value| rounded (db * grid, exact in F32, then to f16), its sign then one xor a pair -
// rounding to nearest is symmetric, so round(-x) = -round(x): decode<T>'s bits, and per value a compare and a select
// fewer (the decode's VALU 268 -> 172 a slab, measured)
// the magnitude |db| grid rounded (never -0: the fused product and rounding the same bits), then the sign - db's
// and the pattern's - one xor: a -0 scale's values -0 as gguf-py's (UD-IQ2_XXS's IQ2_XS: a product fused with its
// rounding gave +0 for 256 of 3.67 M values; test_quant_gpu), with no barrier between them
__device__ inline h8 iq2_h(uint64_t grid, float db, uint32_t pattern) {
  h8 h;
  const float a = __builtin_fabsf(db);
  for (int j = 0; j < kChunk; j++) h[j] = (_Float16) mul(a, (float) (uint8_t) (grid >> (8 * j)));
  const uint32_t ds = __float_as_uint(db) >> 31 ? 0x80008000u : 0u;
  const u4 m = *(const u4*) kSignMasks.m[pattern] ^ (u4){ds, ds, ds, ds};
  return __builtin_bit_cast(h8, __builtin_bit_cast(u4, h) ^ m);
}

// any 8 sign bits as f16 sign masks: byte i's bit j negative - bit 15 of half j (IQ2_S's explicit signs)
struct SignMasks8 {
  uint32_t m[256][kChunk / 2];
};
constexpr SignMasks8 make_sign_masks8() {
  SignMasks8 t{};
  for (int i = 0; i < 256; i++)
    for (int p = 0; p < kChunk / 2; p++)
      t.m[i][p] = (uint32_t) ((i >> (2 * p)) & 1) << 15 | (uint32_t) ((i >> (2 * p + 1)) & 1) << 31;
  return t;
}
static __constant__ SignMasks8 kSignMasks8 = make_sign_masks8();

// the grid's |values| a chunk rounded to f16, then the signs one xor a pair (iq2_h's, any mask)
__device__ inline h8 grid_h(uint64_t grid, float db, const uint32_t* mask) {
  h8 h;
  for (int j = 0; j < kChunk; j++) {
    float x = mul(db, (float) (uint8_t) (grid >> (8 * j)));
    asm("" : "+v"(x));  // decode_h's: the product, then f16 - not one fused v_fma_mixlo_f16 (a -0's +0)
    h[j] = (_Float16) x;
  }
  return __builtin_bit_cast(h8, __builtin_bit_cast(u4, h) ^ *(const u4*) mask);
}

// Qwen3.8's IQ3_XXS and IQ2_S as f16 the same way: decode<T>'s bits (a sign's rounding symmetric), the selects gone
template <> __device__ inline h8 decode_h<kIQ3_XXS>(const uint8_t* b, int k) {
  const uint8_t* q = b + 2 + (k / 4) * 8 + (k % 4) * 2;
  const uint32_t aux = ld32(b + 2 + 64 + (k / 4) * 4);
  const float db = mul(mul(f16(ld16(b)), 0.5f + (float) (aux >> 28)), 0.5f);
  const uint64_t grid = (uint64_t) kIq3xxsGrid[q[0]] | (uint64_t) kIq3xxsGrid[q[1]] << 32;
  return grid_h(grid, db, kSignMasks.m[(aux >> (7 * (k % 4))) & 0x7F]);
}

template <> __device__ inline h8 decode_h<kIQ2_S>(const uint8_t* b, int k) {
  const int s = (b[74 + k / 4] >> (4 * ((k / 2) % 2))) & 0x0F;
  const float db = mul(mul(f16(ld16(b)), 0.5f + (float) s), 0.25f);
  const uint64_t grid = kIq2sGrid[b[2 + k] | ((b[66 + k / 4] >> (2 * (k % 4))) & 0x03) << 8];
  return grid_h(grid, db, kSignMasks8.m[b[34 + k]]);
}

template <> __device__ inline h8 decode_h<kIQ2_XXS>(const uint8_t* b, int k) {
  const uint8_t* q = b + 2 + (k / 4) * 8;
  const uint32_t aux = ld32(q + 4);
  const float db = mul(mul(f16(ld16(b)), 0.5f + (float) (aux >> 28)), 0.25f);
  return iq2_h(kIq2xxsGrid[q[k % 4]], db, (aux >> (7 * (k % 4))) & 0x7F);
}

template <> __device__ inline h8 decode_h<kIQ2_XS>(const uint8_t* b, int k) {
  const uint32_t q = ld16(b + 2 + 2 * k);
  const int s = (b[66 + k / 4] >> (4 * ((k / 2) % 2))) & 0x0F;
  const float db = mul(mul(f16(ld16(b)), 0.5f + (float) s), 0.25f);
  return iq2_h(kIq2xsGrid[q & 511], db, q >> 9);
}

}  // namespace quant
