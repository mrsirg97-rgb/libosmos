// mv_strided's prompt tiles for F32 weights (R 3 NW 12; at most 4 rows the narrow tile, a wave all the rows of its
// own tokens) against the wide tile they replace (mv.h's wide_kernel at R 4 NW 8): every output the same bits - on
// Qwen3.8's F32 shapes (the router 512 x 2560, the inject 4 x 10240, the shared gate 1 x 2560, DeltaNet's gates
// 48 x 2560), token counts off the tiles' multiples, a stride wider than the row. needs a GPU; skipped (77) without one.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "mv.h"

extern "C" {
#include "osmos/gpu/mv.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / (float) (1 << 23) - 1.0f;
}

static float* dev(size_t n, float scale) {
  std::vector<float> h(n);
  for (float& v : h) v = scale * uniform();
  float* d = nullptr;
  CHECK(hipMalloc(&d, n * 4) == hipSuccess && hipMemcpy(d, h.data(), n * 4, hipMemcpyHostToDevice) == hipSuccess);
  return d;
}

static void run(const char* name, uint32_t rows, uint32_t cols, uint32_t n, uint32_t xs) {
  float *w = dev((size_t) rows * cols, 1.0f), *x = dev((size_t) n * xs, 4.0f), *ya = nullptr, *yb = nullptr;
  const size_t yn = (size_t) n * rows;
  CHECK(hipMalloc(&ya, yn * 4) == hipSuccess && hipMalloc(&yb, yn * 4) == hipSuccess);
  CHECK(hipMemset(ya, 0x5A, yn * 4) == hipSuccess && hipMemset(yb, 0x5A, yn * 4) == hipSuccess);
  CHECK((mv::wide_launch<quant::kF32, 4, 8>(w, rows, cols, x, xs, n, yb, rows, nullptr)));
  CHECK(mv_strided(QTYPE_F32, w, rows, cols, x, xs, n, ya, rows, nullptr) && hipDeviceSynchronize() == hipSuccess);
  std::vector<uint32_t> a(yn), b(yn);
  CHECK(hipMemcpy(a.data(), ya, yn * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(b.data(), yb, yn * 4, hipMemcpyDeviceToHost) == hipSuccess);
  size_t bad = 0;
  for (size_t i = 0; i < yn; i++) bad += a[i] != b[i];
  if (bad) std::fprintf(stderr, "  %s %u x %u, %u tokens: %zu outputs differ\n", name, rows, cols, n, bad);
  CHECK(bad == 0);
  std::printf("  %-12s %3u x %5u, %4u tokens (stride %u): the 4 x 8 wide tile's bits\n", name, rows, cols, n, xs);
  CHECK(hipFree(w) == hipSuccess && hipFree(x) == hipSuccess && hipFree(ya) == hipSuccess && hipFree(yb) == hipSuccess);
}

int main() {
  std::printf("[test_mv_f32_wide_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_mv_f32_wide_gpu] skipped: no GPU\n");
    return 77;
  }
  for (uint32_t n : {9u, 37u, 2048u}) run("router", 512, 2560, n, 2560);
  for (uint32_t n : {9u, 37u, 2048u}) run("inject", 4, 10240, n, 10240);
  run("shared gate", 1, 2560, 2048, 2560);
  run("gates", 48, 2560, 333, 2560);
  run("odd rows", 7, 2560, 100, 2600);   // past the narrow tile, rows not the tile's multiple, a wider stride
  run("three rows", 3, 512, 21, 520);    // the narrow tile with a row of its 4 past the end
  std::printf("[test_mv_f32_wide_gpu] ok\n");
  return 0;
}
