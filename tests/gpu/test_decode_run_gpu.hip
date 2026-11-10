// the prompt kernels' Q4_K decodes (kernels/quant.h): decode_h<Q4_K> (one fma: (d sc) q exact in F32) the bits of
// decode<Q4_K> then f16, and decode_hn's runs of 2 and 4 chunks (a group's scales once) decode_h's chunk by chunk -
// on random blocks (their f16 scales any finite value, subnormals and the largest among them). needs a GPU; skipped
// (77) without one.
#pragma clang fp contract(off)
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "quant.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr int kBlocks = 1 << 20, kThreads = 256, kChunks = quant::kQK / quant::kChunk;
constexpr int kBytes = quant::Block<quant::kQ4_K>::bytes;

// a half's bits (an element copied out of its vector first)
__device__ inline uint16_t bits(_Float16 x) {
  uint16_t u;
  __builtin_memcpy(&u, &x, 2);
  return u;
}

// a thread a block: every chunk three ways, the mismatching halves counted
__global__ void check_kernel(const uint8_t* blocks, unsigned long long* bad) {
  const int i = blockIdx.x * kThreads + threadIdx.x;
  if (i >= kBlocks) return;
  const uint8_t* b = blocks + (uint64_t) i * kBytes;
  unsigned long long n = 0;
  for (int k = 0; k < kChunks; k++) {
    float v[quant::kChunk];
    quant::decode<quant::kQ4_K>(b, k, v);
    const quant::h8 h = quant::decode_h<quant::kQ4_K>(b, k);
    for (int j = 0; j < quant::kChunk; j++) {
      asm("" : "+v"(v[j]));
      const _Float16 ref = (_Float16) v[j];
      n += bits(ref) != bits(h[j]);
    }
  }
  for (int k = 0; k < kChunks; k += 2) {
    quant::h8 r[2];
    quant::decode_hn<quant::kQ4_K, 2>(b, k, r);
    for (int c = 0; c < 2; c++)
      for (int j = 0; j < quant::kChunk; j++)
        n += bits(r[c][j]) != bits(quant::decode_h<quant::kQ4_K>(b, k + c)[j]);
  }
  for (int k = 0; k < kChunks; k += 4) {
    quant::h8 r[4];
    quant::decode_hn<quant::kQ4_K, 4>(b, k, r);
    for (int c = 0; c < 4; c++)
      for (int j = 0; j < quant::kChunk; j++)
        n += bits(r[c][j]) != bits(quant::decode_h<quant::kQ4_K>(b, k + c)[j]);
  }
  if (n) atomicAdd(bad, n);
}

int main() {
  std::printf("[test_decode_run_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_decode_run_gpu] skipped: no GPU\n");
    return 77;
  }
  std::vector<uint8_t> h((size_t) kBlocks * kBytes);
  uint32_t s = 1017;
  for (uint8_t& x : h) s = s * 1664525u + 1013904223u, x = (uint8_t) (s >> 24);
  for (int i = 0; i < kBlocks; i++) {  // d and dmin finite f16s: an exponent of 31 made 30; some subnormal, some 65504
    uint8_t* b = &h[(size_t) i * kBytes];
    for (int o = 0; o < 4; o += 2) {
      uint16_t f = (uint16_t) (b[o] | b[o + 1] << 8);
      if (((f >> 10) & 0x1F) == 0x1F) f &= ~(1u << 10);
      if (i % 7 == 0) f &= 0x83FF;  // subnormal (or zero)
      if (i % 11 == 0) f = (uint16_t) ((f & 0x8000) | 0x7BFF);
      b[o] = (uint8_t) f, b[o + 1] = (uint8_t) (f >> 8);
    }
    if (i % 13 == 0)
      for (int o = 4; o < 16; o++) b[o] = 0xFF;  // every scale and min 63
  }
  uint8_t* d;
  unsigned long long* bad;
  CHECK(hipMalloc(&d, h.size()) == hipSuccess && hipMalloc(&bad, 8) == hipSuccess);
  CHECK(hipMemcpy(d, h.data(), h.size(), hipMemcpyHostToDevice) == hipSuccess && hipMemset(bad, 0, 8) == hipSuccess);
  check_kernel<<<kBlocks / kThreads, kThreads>>>(d, bad);
  unsigned long long n = 1;
  CHECK(hipGetLastError() == hipSuccess && hipMemcpy(&n, bad, 8, hipMemcpyDeviceToHost) == hipSuccess);
  if (n) std::fprintf(stderr, "  %llu halves differ\n", n);
  CHECK(n == 0);
  std::printf("  Q4_K: decode_h and decode_hn (runs of 2, 4) decode<Q4_K>'s bits over %d random blocks\n", kBlocks);
  CHECK(hipFree(d) == hipSuccess && hipFree(bad) == hipSuccess);
  std::printf("[test_decode_run_gpu] ok\n");
  return 0;
}
