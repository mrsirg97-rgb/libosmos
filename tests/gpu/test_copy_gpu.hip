// ops_copy between hip_slots' memory (the pool's and the resident set's, which the runtime did not allocate): an
// expert's size and sizes with a tail of 1 .. 15 bytes, every byte copied and none past the end touched;
// hipMemcpyAsync of the same timed beside it (printed: the copy engine's against the kernel's); refusals
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "substrate/gpu/ops.h"
#include "substrate/mem/hip_slots.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint64_t kExpert = 9123840;  // an IQ2_XXS expert's three matrices (gate, up, down)
constexpr uint64_t kGuard = 64;        // bytes past each copy's end, to be left as they were

static uint32_t rng = 1017;
static uint8_t byte() { return (uint8_t) ((rng = rng * 1664525u + 1013904223u) >> 24); }

template <class F>
static double ms_of(F fn) {
  const auto t0 = std::chrono::steady_clock::now();
  fn();
  CHECK(hipDeviceSynchronize() == hipSuccess);
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
  std::printf("[test_copy_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_copy_gpu] skipped: no GPU\n");
    return 77;
  }
  const Slots s = hip_slots();
  const uint64_t cap = kExpert + kGuard;
  uint8_t *src = (uint8_t*) s.alloc(cap, s.ctx), *dst = (uint8_t*) s.alloc(cap, s.ctx);
  CHECK(src && dst);
  std::vector<uint8_t> want(cap), got(cap);
  for (uint64_t n : {kExpert, (uint64_t) 1, (uint64_t) 15, (uint64_t) 16, (uint64_t) 17, (uint64_t) 4096 + 13,
                     kExpert - 9}) {
    for (uint64_t i = 0; i < cap; i++) src[i] = byte(), dst[i] = want[i] = byte();  // (the CPU writes them: hip_slots)
    std::memcpy(want.data(), src, n);
    CHECK(ops_copy(dst, src, n, nullptr) && hipDeviceSynchronize() == hipSuccess);
    std::memcpy(got.data(), dst, cap);
    CHECK(std::memcmp(got.data(), want.data(), cap) == 0);
  }
  // refusals: no bytes, a pointer not 16-byte aligned
  CHECK(!ops_copy(dst, src, 0, nullptr) && !ops_copy(dst + 1, src, 64, nullptr) &&
        !ops_copy(dst, src + 8, 64, nullptr));
  CHECK(!ops_copy(nullptr, src, 64, nullptr) && !ops_copy(dst, nullptr, 64, nullptr));
  // the expert's size each way, 20 times (after one of each)
  CHECK(ops_copy(dst, src, kExpert, nullptr));
  CHECK(hipMemcpyAsync(dst, src, kExpert, hipMemcpyDeviceToDevice, nullptr) == hipSuccess);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  constexpr int kReps = 20;
  const double k = ms_of([&] {
    for (int r = 0; r < kReps; r++) CHECK(ops_copy(dst, src, kExpert, nullptr));
  });
  const double m = ms_of([&] {
    for (int r = 0; r < kReps; r++)
      CHECK(hipMemcpyAsync(dst, src, kExpert, hipMemcpyDeviceToDevice, nullptr) == hipSuccess);
  });
  std::printf("  1 .. 15-byte tails and an expert (%llu B): every byte, none past; an expert's copy %.3f ms as a "
              "kernel, %.3f by hipMemcpyAsync\n", (unsigned long long) kExpert, k / kReps, m / kReps);
  s.free(src, s.ctx), s.free(dst, s.ctx);
  std::printf("[test_copy_gpu] ok\n");
  return 0;
}
