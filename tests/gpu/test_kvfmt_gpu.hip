// the attention's state in the model's formats (substrate/gpu/kvfmt.h): rows rounded with their packed form beside them
// - the F32 the same bits as rounding alone, and the packed rows read back (ops_kv_unpack) the same bits again - for
// FP8 (the window), FP4 with an E4M3 scale a 16 (the compressed rows) and FP4 with a power of two a 32 (the indexer's
// keys); and FP8's rows rounded already (another's, the oracle's) packing to themselves. the values: normal ones, a
// block of zeros, -0, blocks under the scales' floors (subnormal codes, the E4M3 scale's own subnormals), large ones.
// needs a gfx1151; skipped (77) without one.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "substrate/gpu/kvfmt.h"
#include "substrate/gpu/ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr uint32_t kRows = 257;  // a row of each kind below, many times over
static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) ((rng >> 8) * (1.0 / (1 << 24)) * 2.0 - 1.0);
}

// row r's values: its kind r % 6 - normal, a zero block among normal, all -0 and +0, tiny (under the floors), mixed
// magnitudes across blocks, large (up to `large`: FP4 with an E4M3 scale holds a block's amax to 6 x 448 = 2,688)
static std::vector<float> values(uint32_t dim, uint32_t block, float large) {
  std::vector<float> v((size_t) kRows * dim);
  for (uint32_t r = 0; r < kRows; r++)
    for (uint32_t i = 0; i < dim; i++) {
      float x = uniform();
      switch (r % 6) {
        case 1: x = i / block == 1 ? 0.0f : x * 3.0f; break;
        case 2: x = i % 2 ? -0.0f : 0.0f; break;
        case 3: x *= 1e-7f; break;
        case 4: x *= std::ldexp(1.0f, (int) (i / block % (large > 1e4f ? 40 : 30)) - 20); break;
        case 5: x *= large; break;
        default: x *= 2.5f;
      }
      v[(size_t) r * dim + i] = x;
    }
  return v;
}

template <typename T> static T* dev(const std::vector<T>& h) {
  T* d = nullptr;
  CHECK(hipMalloc(&d, h.size() * sizeof(T)) == hipSuccess);
  CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice) == hipSuccess);
  return d;
}
template <typename T> static std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess);
  return h;
}

static size_t differ(const std::vector<float>& a, const std::vector<float>& b) {
  size_t bad = 0;
  for (size_t i = 0; i < a.size(); i++) bad += std::memcmp(&a[i], &b[i], 4) != 0;
  return bad;
}

// a format's round trip: round alone, round and pack, unpack - three times the same bits
static void trip(const char* name, KvFmt fmt, uint32_t dim) {
  const uint32_t block = kv_block(fmt);
  const std::vector<float> x = values(dim, block, fmt == KV_FP4_E4M3 ? 2000.0f : 3e4f);
  const uint64_t n = (uint64_t) kRows * dim, rb = kv_row_bytes(fmt, dim);
  float *a = dev(x), *b = dev(x), *c = nullptr;
  uint8_t* p = nullptr;
  CHECK(hipMalloc(&p, kRows * rb) == hipSuccess && hipMalloc(&c, n * 4) == hipSuccess);
  CHECK(hipMemset(p, 0xA5, kRows * rb) == hipSuccess && hipMemset(c, 0xA5, n * 4) == hipSuccess);
  if (fmt == KV_FP8)
    CHECK(ops_fp8_round(a, n, nullptr, 0, nullptr) && ops_fp8_round(b, n, p, dim, nullptr));
  else
    CHECK(ops_fp4_round(a, n, block, fmt == KV_FP4_E4M3, nullptr, 0, nullptr) &&
          ops_fp4_round(b, n, block, fmt == KV_FP4_E4M3, p, dim, nullptr));
  CHECK(ops_kv_unpack(fmt, p, kRows, dim, c, nullptr));
  const std::vector<float> ra = host(a, n), rb2 = host(b, n), rc = host(c, n);
  size_t bad = differ(ra, rb2);
  if (bad) std::printf("  %s: rounding beside the packing: %zu of %llu values differ\n", name, bad, (unsigned long long) n);
  CHECK(!bad);
  bad = differ(ra, rc);
  if (bad) {
    size_t i = 0;
    while (!std::memcmp(&ra[i], &rc[i], 4)) i++;
    std::printf("  %s: unpacked, %zu of %llu values differ; first at %zu (row %zu, kind %zu): %g, rounded %g\n", name,
                bad, (unsigned long long) n, i, i / dim, i / dim % 6, rc[i], ra[i]);
  }
  CHECK(!bad);
  if (fmt == KV_FP8) {  // FP8: a row rounded already rounds to itself, its packing too
    CHECK(ops_fp8_round(a, n, p, dim, nullptr) && ops_kv_unpack(fmt, p, kRows, dim, c, nullptr));
    CHECK(!differ(host(a, n), ra) && !differ(host(c, n), ra));
  }
  // every format: rounded values (another's: the probe's, the oracle's) packed as they are read back as themselves
  CHECK(hipMemset(p, 0xA5, kRows * rb) == hipSuccess && hipMemset(c, 0xA5, n * 4) == hipSuccess);
  CHECK(ops_kv_pack(fmt, a, kRows, dim, p, nullptr) && ops_kv_unpack(fmt, p, kRows, dim, c, nullptr));
  bad = differ(host(c, n), ra);
  if (bad) std::printf("  %s: rounded values packed as they are, %zu values differ\n", name, bad);
  CHECK(!bad);
  CHECK(hipFree(a) == hipSuccess && hipFree(b) == hipSuccess && hipFree(c) == hipSuccess && hipFree(p) == hipSuccess);
  std::printf("  %-12s %3u values a row, %llu bytes (F32 %u): %u rows rounded, packed and read back, the same bits;"
              " rounded rows packed as they are, themselves\n", name, dim, (unsigned long long) rb, dim * 4, kRows);
}

int main() {
  std::printf("[test_kvfmt_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_kvfmt_gpu] skipped: no GPU\n");
    return 77;
  }
  CHECK(kv_row_bytes(KV_FP8, 512) == 528 && kv_row_bytes(KV_FP4_E4M3, 512) == 288 &&
        kv_row_bytes(KV_FP4_POW2, 128) == 68 && kv_row_bytes(KV_F32, 512) == 2048);
  float* x = nullptr;
  CHECK(hipMalloc(&x, 1024 * 4) == hipSuccess);
  CHECK(!ops_fp8_round(x, 1024, x, 48, nullptr) && !ops_fp4_round(x, 1024, 16, true, x, 40, nullptr) &&
        !ops_kv_unpack(KV_FP4_E4M3, x, 1, 40, x, nullptr) && !ops_kv_unpack(9, x, 1, 512, x, nullptr));
  CHECK(hipFree(x) == hipSuccess);
  std::printf("  refusals: a row not whole blocks, n not whole rows, an unknown format\n");
  trip("FP8", KV_FP8, 512);
  trip("FP4 E4M3/16", KV_FP4_E4M3, 512);
  trip("FP4 pow2/32", KV_FP4_POW2, 128);
  std::printf("[test_kvfmt_gpu] ok\n");
  return 0;
}
