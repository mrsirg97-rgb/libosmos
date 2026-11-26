// the attention core's GPU tests' own helpers (test_attn_core_gpu, test_attn_core_qwen_gpu): checks counted, device
// buffers, byte comparisons, a hash, and the inputs' generator (an LCG: the same values on any library)
#pragma once
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_OK(x)                                                                \
  do {                                                                           \
    if ((x) != hipSuccess) {                                                     \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

static int g_fail = 0, g_pass = 0;
static void check(bool ok, const std::string& what) {
  if (ok)
    g_pass++;
  else
    g_fail++, std::printf("  FAIL %s\n", what.c_str());
}

template <class T>
static T* dev(size_t n) {
  T* p;
  HIP_OK(hipMalloc((void**) &p, n * sizeof(T) + 256));
  HIP_OK(hipMemset(p, 0, n * sizeof(T) + 256));
  return p;
}
static float* up(const std::vector<float>& v) {
  float* d = dev<float>(v.size());
  HIP_OK(hipMemcpy(d, v.data(), v.size() * 4, hipMemcpyHostToDevice));
  return d;
}
static std::vector<uint8_t> down(const void* p, size_t bytes) {
  std::vector<uint8_t> b(bytes);
  HIP_OK(hipDeviceSynchronize());
  HIP_OK(hipMemcpy(b.data(), p, bytes, hipMemcpyDeviceToHost));
  return b;
}
static bool same(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size());
}

// the inputs' own generator (an LCG, Box-Muller): the same values on any library, so the hashes below hold
static uint32_t g_rng = 1017;
static float uni() { return g_rng = g_rng * 1664525u + 1013904223u, ((float) (g_rng >> 8) + 0.5f) / 16777216.0f; }
static std::vector<float> rnd(size_t n, float s) {
  std::vector<float> v(n);
  for (auto& x : v) x = s * std::sqrt(-2.0f * std::log(uni())) * std::cos(6.2831853f * uni());
  return v;
}
// a buffer's FNV-1a (64-bit)
static uint64_t fnv(const std::vector<uint8_t>& b) {
  uint64_t h = 1469598103934665603ull;
  for (uint8_t c : b) h = (h ^ c) * 1099511628211ull;
  return h;
}

