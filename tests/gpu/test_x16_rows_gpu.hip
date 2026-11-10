// mm_x16_rows (the experts' gather into their f16 slots) against a host reference: slot s row idx[s] of x (-1: zero),
// each value f16-nearest at x16_at's place, every byte past the slots untouched; the overflow flag set by a value past
// f16's range or a NaN and only then; rows wider than the columns read; a stride that breaks the rows' alignment
// refused. needs a GPU; skipped (77) without one.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/mm.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr uint32_t kTile = MM_TILE;
constexpr uint8_t kFill = 0x5A;  // the output's bytes before the call: a byte not written stays this

static uint32_t rng = 1017;
static uint32_t next() { return rng = rng * 1664525u + 1013904223u; }

// mm.h's tiled layout (x16_at): token t's column c, rows of xts values
static uint64_t at(uint32_t t, uint32_t c, uint32_t xts) {
  return (uint64_t) (t / kTile) * kTile * xts + (c / kTile) * kTile * kTile + (t % kTile) * kTile + c % kTile;
}

// a value from every range the rounding meets: f16's normals, its subnormals and below, past its largest, -0
static float value(bool wild) {
  const uint32_t r = next();
  const float m = (float) (int32_t) (next() >> 8) / (float) (1 << 23);
  if (!wild) return std::ldexp(m, (int) (r % 30) - 26);
  switch (r % 8) {
    case 0: return std::ldexp(m, 16 + (int) (r >> 8) % 8);  // past 65504
    case 1: return (r >> 8) & 1 ? INFINITY : -INFINITY;
    case 2: return NAN;
    case 3: return -0.0f;
    case 4: return 65504.0f + (float) ((r >> 8) % 32);  // f16's largest and just past (rounds to it, or past)
    default: return std::ldexp(m, (int) (r % 40) - 30);
  }
}

// one call's case: n rows of x (stride xs), slots of cols columns gathered by idx
static void run(uint32_t n, uint32_t slots, uint32_t cols, uint32_t xs, bool wild) {
  std::vector<float> xh((size_t) n * xs);
  for (float& v : xh) v = value(wild);
  std::vector<int32_t> ih(slots);
  for (int32_t& i : ih) i = next() % 5 == 0 ? -1 : (int32_t) (next() % n);
  const uint32_t rows = (slots + kTile - 1) / kTile * kTile;
  const size_t hb = (size_t) rows * cols * 2 + 4096;  // past the last tile too: nothing written there
  std::vector<uint8_t> want(hb, kFill);
  bool over = false;
  for (uint32_t s = 0; s < slots; s++)
    for (uint32_t c = 0; c < cols; c++) {
      const float v = ih[s] >= 0 ? xh[(size_t) ih[s] * xs + c] : 0.0f;
      over |= !(std::fabs(v) <= 65504.0f);
      const _Float16 h = (_Float16) v;
      std::memcpy(&want[at(s, c, cols) * 2], &h, 2);
    }
  float* x = nullptr;
  int32_t *idx = nullptr, *ovf = nullptr;
  void* x16 = nullptr;
  CHECK(hipMalloc(&x, xh.size() * 4) == hipSuccess && hipMalloc(&idx, slots * 4) == hipSuccess);
  CHECK(hipMalloc(&ovf, 4) == hipSuccess && hipMalloc(&x16, hb) == hipSuccess);
  CHECK(hipMemcpy(x, xh.data(), xh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(idx, ih.data(), slots * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemset(ovf, 0, 4) == hipSuccess && hipMemset(x16, kFill, hb) == hipSuccess);
  CHECK(mm_x16_rows(x, xs, idx, slots, cols, x16, ovf, nullptr) && hipDeviceSynchronize() == hipSuccess);
  std::vector<uint8_t> got(hb);
  int32_t o = -1;
  CHECK(hipMemcpy(got.data(), x16, hb, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(&o, ovf, 4, hipMemcpyDeviceToHost) == hipSuccess);
  size_t bad = 0;
  for (size_t i = 0; i < hb; i++) bad += got[i] != want[i];
  if (bad) std::fprintf(stderr, "  %u slots x %u (stride %u): %zu bytes differ\n", slots, cols, xs, bad);
  CHECK(bad == 0);
  CHECK((o != 0) == over);
  CHECK(hipFree(x) == hipSuccess && hipFree(idx) == hipSuccess && hipFree(ovf) == hipSuccess && hipFree(x16) == hipSuccess);
  std::printf("  %u slots x %u columns of %u rows (stride %u)%s: the reference's bytes, overflow %s\n", slots, cols, n,
              xs, wild ? ", wild values" : "", over ? "set" : "clear");
}

int main() {
  std::printf("[test_x16_rows_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_x16_rows_gpu] skipped: no GPU\n");
    return 77;
  }
  run(7, 37, 48, 48, false);        // a partial last tile, a short row
  run(64, 640, 2560, 2560, false);  // Qwen3.8's gate/up input: 64 tokens x 10 experts
  run(19, 203, 640, 656, true);     // the down's width, a wider stride; every rounding edge
  run(5, 16, 16, 20, true);         // one tile of one column block, the stride a multiple of 4 alone
  // a stride whose rows lose the 16-byte alignment, columns not 16s: refused, nothing launched
  {
    float* x = nullptr;
    int32_t *idx = nullptr, *ovf = nullptr;
    void* x16 = nullptr;
    CHECK(hipMalloc(&x, 2 * 2562 * 4) == hipSuccess && hipMalloc(&x16, 16 * 2560 * 2) == hipSuccess);
    CHECK(hipMalloc(&idx, 16 * 4) == hipSuccess && hipMalloc(&ovf, 4) == hipSuccess);
    CHECK(hipMemset(idx, 0, 16 * 4) == hipSuccess);
    CHECK(!mm_x16_rows(x, 2562, idx, 16, 2560, x16, ovf, nullptr));
    CHECK(!mm_x16_rows(x, 64, idx, 16, 40, x16, ovf, nullptr));
    CHECK(hipDeviceSynchronize() == hipSuccess && hipFree(x) == hipSuccess && hipFree(x16) == hipSuccess);
    CHECK(hipFree(idx) == hipSuccess && hipFree(ovf) == hipSuccess);
  }
  std::printf("  refusals: a stride not a multiple of 4, columns not 16s\n");
  std::printf("[test_x16_rows_gpu] ok\n");
  return 0;
}
