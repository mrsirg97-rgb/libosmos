// the prompt kernels' f16 decodes (kernels/quant.h): each type's decode_h the bits of decode then f16 (Q4_K's one
// fma, (d sc) q exact in F32; IQ4_NL / IQ4_XS's values by v_perm_b32), and decode_hn's runs of 2 and 4 chunks (a
// group's scales once) decode_h's chunk by chunk - on random blocks (their f16 scales any finite value, subnormals
// and the largest among them). needs a GPU; skipped
// (77) without one.
#pragma clang fp contract(off)
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#include "quant.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr int kBlocks = 1 << 20, kThreads = 256;

// a half's bits (an element copied out of its vector first)
__device__ inline uint16_t bits(_Float16 x) {
  uint16_t u;
  __builtin_memcpy(&u, &x, 2);
  return u;
}

// a run of N chunks from k: within one block, and a group of 4 (decode_hn's)
template <quant::Type T, int N>
__device__ unsigned long long runs(const uint8_t* b) {
  constexpr int chunks = quant::Block<T>::values / quant::kChunk;
  unsigned long long n = 0;
  for (int k = 0; k + N <= chunks; k += N) {
    if (k % 4 + N > 4) continue;
    quant::h8 r[N];
    quant::decode_hn<T, N>(b, k, r);
    for (int c = 0; c < N; c++) {
      const quant::h8 h = quant::decode_h<T>(b, k + c);
      for (int j = 0; j < quant::kChunk; j++) n += bits(r[c][j]) != bits(h[j]);
    }
  }
  return n;
}

// a thread a block: decode_h against decode then f16 (decode_h's generic rounding), and decode_hn's runs of 2 and 4
// against decode_h, the mismatching halves counted
template <quant::Type T>
__global__ void check_kernel(const uint8_t* blocks, unsigned long long* bad) {
  const int i = blockIdx.x * kThreads + threadIdx.x;
  if (i >= kBlocks) return;
  const uint8_t* b = blocks + (uint64_t) i * quant::Block<T>::bytes;
  constexpr int chunks = quant::Block<T>::values / quant::kChunk;
  unsigned long long n = 0;
  for (int k = 0; k < chunks; k++) {
    float v[quant::kChunk];
    quant::decode<T>(b, k, v);
    const quant::h8 h = quant::decode_h<T>(b, k);
    for (int j = 0; j < quant::kChunk; j++) {
      asm("" : "+v"(v[j]));
      const _Float16 ref = (_Float16) v[j];
      n += bits(ref) != bits(h[j]);
    }
  }
  n += runs<T, 2>(b) + runs<T, 4>(b);
  if (n) atomicAdd(bad, n);
}

// random blocks of a type, its f16 scales (at the byte offsets given) finite: some subnormal, some the largest
template <quant::Type T>
static void check(const char* name, std::initializer_list<int> f16_at) {
  constexpr int bytes = quant::Block<T>::bytes;
  std::vector<uint8_t> h((size_t) kBlocks * bytes);
  uint32_t s = 1017;
  for (uint8_t& x : h) s = s * 1664525u + 1013904223u, x = (uint8_t) (s >> 24);
  for (int i = 0; i < kBlocks; i++) {
    uint8_t* b = &h[(size_t) i * bytes];
    for (int o : f16_at) {
      uint16_t f = (uint16_t) (b[o] | b[o + 1] << 8);
      if (((f >> 10) & 0x1F) == 0x1F) f &= ~(1u << 10);
      if (i % 7 == 0) f &= 0x83FF;  // subnormal (or zero)
      if (i % 11 == 0) f = (uint16_t) ((f & 0x8000) | 0x7BFF);
      b[o] = (uint8_t) f, b[o + 1] = (uint8_t) (f >> 8);
    }
    if (T == quant::kQ4_K && i % 13 == 0)
      for (int o = 4; o < 16; o++) b[o] = 0xFF;  // every scale and min 63
  }
  uint8_t* d;
  unsigned long long* bad;
  CHECK(hipMalloc(&d, h.size()) == hipSuccess && hipMalloc(&bad, 8) == hipSuccess);
  CHECK(hipMemcpy(d, h.data(), h.size(), hipMemcpyHostToDevice) == hipSuccess && hipMemset(bad, 0, 8) == hipSuccess);
  check_kernel<T><<<kBlocks / kThreads, kThreads>>>(d, bad);
  unsigned long long n = 1;
  CHECK(hipGetLastError() == hipSuccess && hipMemcpy(&n, bad, 8, hipMemcpyDeviceToHost) == hipSuccess);
  if (n) std::fprintf(stderr, "  %s: %llu halves differ\n", name, n);
  CHECK(n == 0);
  std::printf("  %s: decode_h decode's bits, decode_hn (runs of 2, 4) decode_h's, over %d random blocks\n", name, kBlocks);
  CHECK(hipFree(d) == hipSuccess && hipFree(bad) == hipSuccess);
}

int main() {
  std::printf("[test_decode_run_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_decode_run_gpu] skipped: no GPU\n");
    return 77;
  }
  check<quant::kQ4_K>("Q4_K", {0, 2});
  check<quant::kIQ4_NL>("IQ4_NL", {0});
  check<quant::kIQ4_XS>("IQ4_XS", {0});
  check<quant::kIQ3_XXS>("IQ3_XXS", {0});
  std::printf("[test_decode_run_gpu] ok\n");
  return 0;
}
