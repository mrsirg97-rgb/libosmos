// every byte the library's own code takes from HIP and HSA, counted by the linker's --wrap (the Makefile's
// ALLOC_COUNT_WRAP on a test that includes this): hipMalloc / hipHostMalloc / hipHostRegister /
// hsa_amd_memory_pool_allocate and their frees. what the runtime allocates for itself is not seen - only the calls
// the engine makes, which is what its ledger must count (0.15.3: an engine's every allocation through its ledger)
#pragma once
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>
#include <mutex>
#include <unordered_map>

namespace alloc_count {
inline std::mutex mu;
inline std::unordered_map<void*, uint64_t>& live() {
  static std::unordered_map<void*, uint64_t> m;
  return m;
}
inline void add(void* p, uint64_t b) {
  if (!p) return;
  std::lock_guard<std::mutex> g(mu);
  live()[p] = b;
}
inline void drop(void* p) {
  if (!p) return;
  std::lock_guard<std::mutex> g(mu);
  live().erase(p);
}
// the bytes the library holds now
inline uint64_t held() {
  std::lock_guard<std::mutex> g(mu);
  uint64_t n = 0;
  for (auto& kv : live()) n += kv.second;
  return n;
}
// what is held, said (a leak's allocations: their sizes, the largest first)
inline void say(uint64_t from) {
  std::lock_guard<std::mutex> g(mu);
  std::vector<uint64_t> b;
  for (auto& kv : live()) b.push_back(kv.second);
  std::sort(b.begin(), b.end(), std::greater<uint64_t>());
  uint64_t n = 0;
  for (uint64_t x : b) n += x;
  std::fprintf(stderr, "  held: %zu allocations, %llu bytes past %llu:", b.size(), (unsigned long long) (n - from),
               (unsigned long long) from);
  for (size_t i = 0; i < b.size() && i < 12; i++) std::fprintf(stderr, " %llu", (unsigned long long) b[i]);
  std::fprintf(stderr, "\n");
}
}  // namespace alloc_count

extern "C" {
hipError_t __real_hipMalloc(void** p, size_t b);
hipError_t __real_hipFree(void* p);
hipError_t __real_hipHostMalloc(void** p, size_t b, unsigned int f);
hipError_t __real_hipHostFree(void* p);
hipError_t __real_hipHostRegister(void* p, size_t b, unsigned int f);
hipError_t __real_hipHostUnregister(void* p);
hsa_status_t __real_hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t pool, size_t b, uint32_t f, void** p);
hsa_status_t __real_hsa_amd_memory_pool_free(void* p);

hipError_t __wrap_hipMalloc(void** p, size_t b) {
  const hipError_t r = __real_hipMalloc(p, b);
  if (r == hipSuccess) alloc_count::add(*p, b);
  return r;
}
hipError_t __wrap_hipFree(void* p) {
  alloc_count::drop(p);
  return __real_hipFree(p);
}
hipError_t __wrap_hipHostMalloc(void** p, size_t b, unsigned int f) {
  const hipError_t r = __real_hipHostMalloc(p, b, f);
  if (r == hipSuccess) alloc_count::add(*p, b);
  return r;
}
hipError_t __wrap_hipHostFree(void* p) {
  alloc_count::drop(p);
  return __real_hipHostFree(p);
}
hipError_t __wrap_hipHostRegister(void* p, size_t b, unsigned int f) {
  const hipError_t r = __real_hipHostRegister(p, b, f);
  if (r == hipSuccess) alloc_count::add(p, b);
  return r;
}
hipError_t __wrap_hipHostUnregister(void* p) {
  alloc_count::drop(p);
  return __real_hipHostUnregister(p);
}
hsa_status_t __wrap_hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t pool, size_t b, uint32_t f, void** p) {
  const hsa_status_t r = __real_hsa_amd_memory_pool_allocate(pool, b, f, p);
  if (r == HSA_STATUS_SUCCESS) alloc_count::add(*p, b);
  return r;
}
hsa_status_t __wrap_hsa_amd_memory_pool_free(void* p) {
  alloc_count::drop(p);
  return __real_hsa_amd_memory_pool_free(p);
}
}
