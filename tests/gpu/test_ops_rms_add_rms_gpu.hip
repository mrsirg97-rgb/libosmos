// ops_rms_add_rms (osmos/gpu/ops.h): a block's post-norm added to the residual and the next norm, one launch -
// against ops_rms, ops_add and ops_rms in turn: h, y and xn bit for bit, with a next norm and without, at vibes1.7b's
// width (2,048) and another (5,120), 1 and 7 rows; refusals
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" {
#include "osmos/gpu/ops.h"
}

#define CHECK(x)                                                           \
  do {                                                                     \
    if (!(x)) {                                                            \
      std::fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

static float* up(const std::vector<float>& v) {
  float* d = nullptr;
  CHECK(hipMalloc(&d, v.size() * 4) == hipSuccess && hipMemcpy(d, v.data(), v.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  return d;
}

static std::vector<float> down(const float* d, size_t n) {
  std::vector<float> v(n);
  CHECK(hipDeviceSynchronize() == hipSuccess && hipMemcpy(v.data(), d, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return v;
}

static void one(std::mt19937& rng, uint32_t dim, uint32_t rows, bool next) {
  std::normal_distribution<float> nd(0, 1);
  auto rnd = [&](size_t n, float s) {
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng) * s;
    return v;
  };
  const std::vector<float> h = rnd((size_t) rows * dim, 3), y = rnd((size_t) rows * dim, 2);
  const std::vector<float> w1 = rnd(dim, 1), w2 = rnd(dim, 1), xn0 = rnd((size_t) rows * dim, 1);
  float *h1 = up(h), *y1 = up(y), *x1 = up(xn0), *h2 = up(h), *y2 = up(y), *x2 = up(xn0), *dw1 = up(w1), *dw2 = up(w2);
  CHECK(ops_rms(y1, dim, dim, dw1, 1e-5f, rows, y1, dim, nullptr) && ops_add(h1, y1, (uint64_t) rows * dim, nullptr));
  if (next) CHECK(ops_rms(h1, dim, dim, dw2, 1e-5f, rows, x1, dim, nullptr));
  CHECK(ops_rms_add_rms(h2, y2, dw1, next ? dw2 : nullptr, 1e-5f, dim, rows, x2, nullptr));
  const size_t n = (size_t) rows * dim;
  CHECK(std::memcmp(down(h1, n).data(), down(h2, n).data(), n * 4) == 0);
  CHECK(std::memcmp(down(y1, n).data(), down(y2, n).data(), n * 4) == 0);
  CHECK(std::memcmp(down(x1, n).data(), down(x2, n).data(), n * 4) == 0);
  std::printf("  %u wide, %u rows%s: the three ops' h, y and xn, bit for bit\n", dim, rows, next ? ", a next norm" : "");
  for (float* p : {h1, y1, x1, h2, y2, x2, dw1, dw2}) CHECK(hipFree(p) == hipSuccess);
}

int main() {
  int devs = 0;
  if (hipGetDeviceCount(&devs) != hipSuccess || !devs) return std::printf("[test_ops_rms_add_rms_gpu] skipped: no GPU\n"), 77;
  std::printf("[test_ops_rms_add_rms_gpu] start\n");
  float* p = nullptr;
  CHECK(!ops_rms_add_rms(p, p, p, nullptr, 1e-5f, 0, 1, p, nullptr));
  CHECK(!ops_rms_add_rms(p, p, p, nullptr, 1e-5f, 2048, 0, p, nullptr));
  std::mt19937 rng(2048);
  for (uint32_t rows : {1u, 7u})
    for (bool next : {true, false}) one(rng, 2048, rows, next), one(rng, 5120, rows, next);
  std::printf("[test_ops_rms_add_rms_gpu] ok\n");
  return 0;
}
