// ops_argmax against a scan in order (the first maximum): rows of a vocabulary's length (not a multiple of the
// block), random values; a planted tie (the lower index), -0 against +0, a row of -inf (index 0), a row with a NaN
// (-1); refusals
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
#include "substrate/gpu/ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kN = 129280, kStride = kN + 64, kRows = 9;  // DeepSeek-V4.1's vocabulary

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / (float) (1u << 23) - 1.0f;
}

static int32_t scan(const float* row) {  // the bench's: the first maximum; -1 for a NaN
  int32_t b = 0;
  for (uint32_t i = 0; i < kN; i++) {
    if (std::isnan(row[i])) return -1;
    if (row[i] > row[b]) b = (int32_t) i;
  }
  return b;
}

int main() {
  std::printf("[test_argmax_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_argmax_gpu] skipped: no GPU\n");
    return 77;
  }
  std::vector<float> x((size_t) kRows * kStride);
  for (float& v : x) v = uniform() * 20.0f;
  float* r = x.data();
  r[0 * kStride + 70000] = r[0 * kStride + 90000] = 50.0f;      // a tie: 70,000
  r[1 * kStride + kN - 1] = 51.0f;                              // the last
  for (uint32_t i = 0; i < kN; i++) r[2 * kStride + i] = -0.0f;  // -0 and +0 equal: index 0
  r[2 * kStride + 5] = 0.0f;
  for (uint32_t i = 0; i < kN; i++) r[3 * kStride + i] = -INFINITY;  // all -inf: index 0
  r[4 * kStride + 12345] = NAN;                                      // a NaN: -1
  r[5 * kStride + 3] = 60.0f, r[5 * kStride + 2000] = 60.0f;         // a tie across threads: 3
  float* dx;
  int32_t* dout;
  CHECK(hipMalloc(&dx, x.size() * 4) == hipSuccess && hipMalloc(&dout, kRows * 4) == hipSuccess);
  CHECK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(ops_argmax(dx, kN, kRows, kStride, dout, nullptr) && hipDeviceSynchronize() == hipSuccess);
  std::vector<int32_t> got(kRows);
  CHECK(hipMemcpy(got.data(), dout, kRows * 4, hipMemcpyDeviceToHost) == hipSuccess);
  for (uint32_t i = 0; i < kRows; i++) CHECK(got[i] == scan(&x[(size_t) i * kStride]));
  CHECK(got[0] == 70000 && got[1] == (int32_t) kN - 1 && got[2] == 0 && got[3] == 0 && got[4] == -1 && got[5] == 3);
  CHECK(!ops_argmax(dx, kN, kRows, kN - 1, dout, nullptr) && !ops_argmax(dx, 0, kRows, kStride, dout, nullptr) &&
        !ops_argmax(dx, kN, 0, kStride, dout, nullptr) && !ops_argmax(nullptr, kN, 1, kStride, dout, nullptr));
  CHECK(hipFree(dx) == hipSuccess && hipFree(dout) == hipSuccess);
  std::printf("  %u rows of %u: the first maximum of each (a tie, -0 and +0, all -inf, a NaN: -1); refusals\n", kRows,
              kN);
  std::printf("[test_argmax_gpu] ok\n");
  return 0;
}
