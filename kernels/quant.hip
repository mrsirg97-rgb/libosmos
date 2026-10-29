// quant_dequant (include/substrate/gpu/quant.h): a thread an 8-value chunk
#include "quant.h"

extern "C" {
#include "substrate/gpu/quant.h"
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
bool launch(const void* src, uint64_t values, float* dst, hipStream_t stream) {
  if (values % quant::Block<T>::values) return false;
  const uint64_t chunks = values / quant::kChunk;
  const uint64_t blocks = chunks / kThreads + 1 < kMaxBlocks ? chunks / kThreads + 1 : kMaxBlocks;
  dequant_kernel<T><<<(unsigned) blocks, kThreads, 0, stream>>>((const uint8_t*) src, chunks, dst);
  return hipGetLastError() == hipSuccess;
}

}  // namespace

extern "C" bool quant_known(uint32_t type) {
  switch (type) {
    case quant::kQ8_0: case quant::kQ5_K: case quant::kQ6_K: case quant::kIQ2_XXS: case quant::kIQ2_XS:
    case quant::kIQ4_NL: case quant::kIQ1_M: case quant::kBF16: return true;
    default: return false;
  }
}

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
    case quant::kBF16: return launch<quant::kBF16>(src, values, dst, s);
    default: return false;
  }
}
