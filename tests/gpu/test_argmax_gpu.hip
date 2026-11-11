// ops_argmax - both families' greedy pick - against a scan in order (the first maximum; -1 for a NaN), at each
// vocabulary's length (DeepSeek-V4.1's, Qwen3.8's: not multiples of the slices or the block) and a row shorter than
// the slices: random rows; a tie far apart and one across a slice's seam (the lower index), the last index, -0
// against +0 and a row all one value (index 0), a row all -inf (index 0), a NaN alone and NaNs under a greater
// finite value (-1: never the greatest of what is left); a stride past the row; refusals
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
#include "osmos/gpu/ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kRows = 12, kSlices = 64;  // (the kernel's split: the seam's place)

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / (float) (1u << 23) - 1.0f;
}

static int32_t scan(const float* row, uint32_t n) {
  int32_t b = 0;
  for (uint32_t i = 0; i < n; i++) {
    if (std::isnan(row[i])) return -1;
    if (row[i] > row[b]) b = (int32_t) i;
  }
  return b;
}

static void rows_of(uint32_t n) {
  const uint32_t stride = n + 64, per = (n + kSlices - 1) / kSlices;
  std::vector<float> x((size_t) kRows * stride);
  for (float& v : x) v = uniform() * 20.0f;
  auto at = [&](uint32_t r, uint32_t i) -> float& { return x[(size_t) r * stride + i]; };
  at(0, n / 2) = at(0, n - n / 4) = 50.0f;                    // a tie far apart: the lower
  at(1, n - 1) = 51.0f;                                       // the last
  for (uint32_t i = 0; i < n; i++) at(2, i) = -0.0f;          // -0 and +0 equal: index 0
  at(2, n / 3) = 0.0f;
  for (uint32_t i = 0; i < n; i++) at(3, i) = -INFINITY;      // all -inf: index 0
  at(4, n / 5) = NAN;                                         // a NaN alone: -1
  at(5, 3) = 60.0f, at(5, n / 2 + 1) = 60.0f;                 // a tie across threads and slices: 3
  if (per < n) at(6, per - 1) = at(6, per) = 61.0f;           // a tie across a slice's seam: the lower
  for (uint32_t i = 0; i < n; i += 97) at(7, i) = NAN;        // NaNs under a greater finite value: -1
  at(7, n / 2 + 1) = 70.0f;
  for (uint32_t i = 0; i < n; i++) at(8, i) = 1.5f;           // all one value: index 0
  at(9, n - 1) = NAN;                                         // a NaN at the row's end
  at(10, 0) = 80.0f;                                          // the first
  for (uint32_t i = n; i < stride; i++) at(11, i) = 1e30f;    // past the row: not the row's
  float* dx;
  int32_t* dout;
  void* part;
  CHECK(hipMalloc(&dx, x.size() * 4) == hipSuccess && hipMalloc(&dout, kRows * 4) == hipSuccess &&
        hipMalloc(&part, ops_argmax_scratch(kRows)) == hipSuccess);
  CHECK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(ops_argmax(dx, n, kRows, stride, dout, part, nullptr) && hipDeviceSynchronize() == hipSuccess);
  std::vector<int32_t> got(kRows);
  CHECK(hipMemcpy(got.data(), dout, kRows * 4, hipMemcpyDeviceToHost) == hipSuccess);
  for (uint32_t r = 0; r < kRows; r++) {
    const int32_t want = scan(&x[(size_t) r * stride], n);
    if (got[r] != want) std::fprintf(stderr, "  %u values, row %u: %d, the scan %d\n", n, r, got[r], want);
    CHECK(got[r] == want);
  }
  CHECK(got[0] == (int32_t) (n / 2) && got[1] == (int32_t) n - 1 && got[2] == 0 && got[3] == 0 && got[4] == -1 &&
        got[5] == 3 && got[7] == -1 && got[8] == 0 && got[9] == -1 && got[10] == 0 && (uint32_t) got[11] < n);
  if (per < n) CHECK(got[6] == (int32_t) per - 1);
  // one row at a time: the rows' own
  for (uint32_t r = 0; r < kRows; r++) {
    int32_t one = -2;
    CHECK(ops_argmax(dx + (size_t) r * stride, n, 1, stride, dout, part, nullptr) &&
          hipMemcpy(&one, dout, 4, hipMemcpyDeviceToHost) == hipSuccess && one == got[r]);
  }
  CHECK(!ops_argmax(dx, n, kRows, n - 1, dout, part, nullptr) && !ops_argmax(dx, 0, kRows, stride, dout, part, nullptr) &&
        !ops_argmax(dx, n, 0, stride, dout, part, nullptr) && !ops_argmax(nullptr, n, 1, stride, dout, part, nullptr) &&
        !ops_argmax(dx, n, 1, stride, nullptr, part, nullptr) && !ops_argmax(dx, n, 1, stride, dout, nullptr, nullptr));
  CHECK(hipFree(dx) == hipSuccess && hipFree(dout) == hipSuccess && hipFree(part) == hipSuccess);
  std::printf("  %u rows of %u: the first maximum of each (ties, -0 and +0, all -inf, NaNs: -1); refusals\n", kRows, n);
}

int main() {
  std::printf("[test_argmax_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_argmax_gpu] skipped: no GPU\n");
    return 77;
  }
  rows_of(129280);  // DeepSeek-V4.1's vocabulary
  rows_of(248320);  // Qwen3.8's
  rows_of(40);      // fewer than the slices: some hold nothing
  std::printf("[test_argmax_gpu] ok\n");
  return 0;
}
