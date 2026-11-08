// quant_dequant (include/osmos/gpu/quant.h): a thread an 8-value chunk
#include "quant.h"

extern "C" {
#include "osmos/gpu/quant.h"
}

namespace {

constexpr int kThreads = 256;
constexpr uint64_t kMaxBlocks = 1 << 16;

template <quant::Type T>
__global__ void dequant_kernel(const uint8_t* __restrict__ src, uint64_t chunks, float* __restrict__ dst) {
  constexpr int per = quant::Block<T>::values / quant::kChunk;
  const uint64_t stride = (uint64_t) gridDim.x * blockDim.x;
  for (uint64_t c = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; c < chunks; c += stride) {
    float v[quant::kChunk];
    quant::decode<T>(src + (c / per) * quant::Block<T>::bytes, (int) (c % per), v);
    for (int j = 0; j < quant::kChunk; j++) dst[c * quant::kChunk + j] = v[j];
  }
}

template <quant::Type T>
__global__ void dequant_h_kernel(const uint8_t* __restrict__ src, uint64_t chunks, quant::h8* __restrict__ dst) {
  constexpr int per = quant::Block<T>::values / quant::kChunk;
  const uint64_t stride = (uint64_t) gridDim.x * blockDim.x;
  for (uint64_t c = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; c < chunks; c += stride)
    dst[c] = quant::decode_h<T>(src + (c / per) * quant::Block<T>::bytes, (int) (c % per));
}

template <quant::Type T>
bool launch_h(const void* src, uint64_t values, void* dst, hipStream_t stream) {
  if (values % quant::Block<T>::values) return false;
  const uint64_t chunks = values / quant::kChunk;
  const uint64_t blocks = chunks / kThreads + 1 < kMaxBlocks ? chunks / kThreads + 1 : kMaxBlocks;
  dequant_h_kernel<T><<<(unsigned) blocks, kThreads, 0, stream>>>((const uint8_t*) src, chunks, (quant::h8*) dst);
  return hipGetLastError() == hipSuccess;
}

template <quant::Type T>
bool launch(const void* src, uint64_t values, float* dst, hipStream_t stream) {
  if (values % quant::Block<T>::values) return false;
  const uint64_t chunks = values / quant::kChunk;
  const uint64_t blocks = chunks / kThreads + 1 < kMaxBlocks ? chunks / kThreads + 1 : kMaxBlocks;
  dequant_kernel<T><<<(unsigned) blocks, kThreads, 0, stream>>>((const uint8_t*) src, chunks, dst);
  return hipGetLastError() == hipSuccess;
}

// a thread a row's block: its tile's block's head word, then per sub-block its (s, m) and its two slots - Q5_K's
// fields moved, none computed (kernels/quant.h: kQ5_KR)
__global__ void repack_q5k_kernel(const uint8_t* __restrict__ src, uint64_t items, uint32_t nb,
                                  uint8_t* __restrict__ dst) {
  using namespace quant;
  const uint64_t stride = (uint64_t) gridDim.x * blockDim.x;
  for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < items; i += stride) {
    const uint64_t tb = i / kQ5rRows, t = tb / nb, b = tb % nb;
    const int r = (int) (i % kQ5rRows);
    const uint8_t* blk = src + ((t * kQ5rRows + r) * nb + b) * Block<kQ5_K>::bytes;
    uint8_t* out = dst + tb * kQ5rTile;
    for (int j = 0; j < 4; j++) out[r * 4 + j] = blk[j];
    auto byte = [&](int x) { return (int) blk[4 + x]; };
    for (int g = 0; g < 8; g++) {
      uint8_t* seg = out + kQ5rHead + g * kQ5rSeg;
      const int sc = g < 4 ? byte(g) & 0x3F : (byte(g + 4) & 0x0F) | ((byte(g - 4) >> 2) & 0x30);
      const int m = g < 4 ? byte(g + 4) & 0x3F : (byte(g + 4) >> 4) | ((byte(g) >> 2) & 0x30);
      seg[kQ5rSm + r * 2] = (uint8_t) sc, seg[kQ5rSm + r * 2 + 1] = (uint8_t) m;
      for (int half = 0; half < 2; half++) {
        uint64_t qs = 0;
        uint32_t qh = 0;
        for (int v = 0; v < 16; v++) {  // value v: chunk 4g + 2 (v / 8) + half's value v % 8
          const int l0 = (2 * (v / 8) + half) * kChunk, j = v % 8;
          const int lo = (blk[48 + (g / 2) * 32 + l0 + j] >> (4 * (g % 2))) & 0x0F, hi = (blk[16 + l0 + j] >> g) & 1;
          const int q = lo | hi << 4;
          qs |= (uint64_t) (q & 0x0F) << (4 * v), qh |= (uint32_t) (q >> 4) << v;
        }
        const int slot = half * kQ5rRows + r;
        for (int j = 0; j < 8; j++) seg[slot * 8 + j] = (uint8_t) (qs >> (8 * j));
        seg[kQ5rQh + slot * 2] = (uint8_t) qh, seg[kQ5rQh + slot * 2 + 1] = (uint8_t) (qh >> 8);
      }
    }
  }
}

}  // namespace

extern "C" bool quant_repack_q5k(const void* src, uint32_t rows, uint32_t cols, void* dst, void* stream) {
  if (!src || !dst || src == dst || !rows || rows % quant::kQ5rRows || !cols || cols % quant::kQK) return false;
  const uint64_t items = (uint64_t) rows * (cols / quant::kQK);
  const uint64_t blocks = items / kThreads + 1 < kMaxBlocks ? items / kThreads + 1 : kMaxBlocks;
  repack_q5k_kernel<<<(unsigned) blocks, kThreads, 0, (hipStream_t) stream>>>((const uint8_t*) src, items,
                                                                              cols / quant::kQK, (uint8_t*) dst);
  return hipGetLastError() == hipSuccess;
}

extern "C" bool quant_known(uint32_t type) { return qtype_paths(type) & QTYPE_DECODE; }

extern "C" bool quant_dequant(uint32_t type, const void* src, uint64_t values, float* dst, void* stream) {
  const hipStream_t s = (hipStream_t) stream;
  switch (type) {
    case quant::kQ8_0: return launch<quant::kQ8_0>(src, values, dst, s);
    case quant::kQ5_K: return launch<quant::kQ5_K>(src, values, dst, s);
    case quant::kQ6_K: return launch<quant::kQ6_K>(src, values, dst, s);
    case quant::kIQ2_XXS: return launch<quant::kIQ2_XXS>(src, values, dst, s);
    case quant::kIQ2_XS: return launch<quant::kIQ2_XS>(src, values, dst, s);
    case quant::kIQ4_NL: return launch<quant::kIQ4_NL>(src, values, dst, s);
    case quant::kIQ1_M: return launch<quant::kIQ1_M>(src, values, dst, s);
    case quant::kIQ1_S: return launch<quant::kIQ1_S>(src, values, dst, s);
    case quant::kIQ2_S: return launch<quant::kIQ2_S>(src, values, dst, s);
    case quant::kBF16: return launch<quant::kBF16>(src, values, dst, s);
    case quant::kMXFP4: return launch<quant::kMXFP4>(src, values, dst, s);
    case quant::kMXFP8: return launch<quant::kMXFP8>(src, values, dst, s);
    case quant::kIQ3_XXS: return launch<quant::kIQ3_XXS>(src, values, dst, s);
    case quant::kIQ4_XS: return launch<quant::kIQ4_XS>(src, values, dst, s);
    case quant::kQ2_0: return launch<quant::kQ2_0>(src, values, dst, s);
    case quant::kQ2_K: return launch<quant::kQ2_K>(src, values, dst, s);
    default: return false;
  }
}

extern "C" bool quant_dequant_f16(uint32_t type, const void* src, uint64_t values, void* dst, void* stream) {
  const hipStream_t s = (hipStream_t) stream;
  switch (type) {
#define H(t) \
  case quant::t: return launch_h<quant::t>(src, values, dst, s);
    H(kQ8_0) H(kQ5_K) H(kQ6_K) H(kIQ2_XXS) H(kIQ2_XS) H(kIQ4_NL) H(kIQ1_M) H(kBF16) H(kMXFP4) H(kMXFP8) H(kIQ1_S) H(kIQ2_S)
    H(kIQ3_XXS) H(kIQ4_XS) H(kQ2_0) H(kQ2_K)
#undef H
    default: return false;
  }
}
