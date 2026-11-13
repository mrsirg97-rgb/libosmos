// the NPU's GPU side (kernels/npu_pack.hip) against host references written from npu_pack.h's contract: the A stream's
// bytes for random rows and the cases its rounding turns on (zeros, -0, subnormals, a mantissa that rounds to 128 -
// the exponent raised, -128, a block 32 binades wide, pad rows), parts of two kinds (gate + up's K 5120, the down's
// 2304); and the unpack for both kernels' C layouts (gate + up: two matrices, 24 rows a core a pass; the down: one, 16)
// - every value where its part's rows say, every other row untouched. needs a GPU; skipped (77) without one.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/npu_pack.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

static uint32_t seed = 1017;
static uint32_t next() { return seed = seed * 1664525u + 1013904223u; }
static float rnd() { return (float) ((next() >> 8) & 0xFFFF) / 32768.0f - 1.0f; }

template <typename T>
static T* dev_of(const std::vector<T>& h) {
  T* d;
  CHECK(hipMalloc(&d, h.size() * sizeof(T) + 4) == hipSuccess &&
        hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice) == hipSuccess);
  return d;
}
template <typename T>
static std::vector<T> host_of(const T* d, size_t n) {
  std::vector<T> h(n);
  CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess);
  return h;
}

static uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }

// a block of 8 as the contract says: its largest biased exponent e, each value v * 2^(133 - e) rounded to nearest
// (ties to even); a rounding past 127 raises e by one and rounds again; clamped to [-128, 127]
static void block_ref(const float* v, uint8_t out[9]) {
  uint32_t e = 0;
  for (int j = 0; j < 8; j++) e = std::max(e, (bits(v[j]) >> 23) & 0xFF);
  long q[8];
  for (int round = 0; round < 2; round++) {
    bool carry = false;
    for (int j = 0; j < 8; j++) q[j] = std::lrint(std::ldexp((double) v[j], 133 - (int) e)), carry |= q[j] > 127;
    if (!carry) break;
    e++;
  }
  out[0] = (uint8_t) e;
  for (int j = 0; j < 8; j++) out[1 + j] = (uint8_t) (int8_t) std::min(127L, std::max(-128L, q[j]));
}

// part p's A stream, as npu_pack.h lays it: [K-step][64-row half][K half] pieces of 64 x 128, each 8 x 8 sub-tiles in
// raster order, a sub-tile's 8 rows of one block
static std::vector<uint8_t> pack_ref(const std::vector<float>& x, uint32_t K, const std::vector<int32_t>& tok, uint32_t parts) {
  std::vector<uint8_t> a(parts * npu_a_bytes(K));
  const uint64_t piece = 64 * 128 * 9 / 8;
  for (uint32_t p = 0; p < parts; p++)
    for (uint32_t m = 0; m < NPU_TOKENS; m++)
      for (uint32_t kb = 0; kb < K / 8; kb++) {
        float v[8];
        const int32_t row = tok[p * NPU_TOKENS + m];
        for (int j = 0; j < 8; j++) v[j] = row < 0 ? 0.0f : x[(uint64_t) row * K + kb * 8 + j];
        const uint32_t s = kb / 32, kq = kb % 32 / 16, sx = kb % 16, h = m / 64, sy = m % 64 / 8, i = m % 8;
        block_ref(v, &a[p * npu_a_bytes(K) + ((s * 2 + h) * 2 + kq) * piece + ((sy * 16 + sx) * 8 + i) * 9]);
      }
  return a;
}

static void pack_case(uint32_t K, uint32_t rows, uint32_t parts) {
  std::vector<float> x((size_t) rows * K);
  for (auto& v : x) v = rnd() * std::ldexp(1.0f, (int) (next() % 40) - 20);
  // the cases the rounding turns on, in the first rows' first blocks
  const float cases[][8] = {
      {0, 0, 0, 0, 0, 0, 0, 0},                                 // a zero block
      {-0.0f, 0, 0, 0, 0, 0, 0, 1.0f},                          // -0
      {1e-40f, -2e-41f, 3e-39f, 0, 0, 0, 0, 0},                 // subnormals alone
      {1.9921875f, 1.0f, 0.5f, -1.0f, 0, 0, 0, 0},              // 1.9921875 * 2^6 = 127.5: ties to even, 128 - carry
      {1.9999f, 1.0f, 0, 0, 0, 0, 0, 0},                        // 1.9999 * 2^6 rounds to 128: carry
      {-1.0f, -0.5f, 0.25f, 0, 0, 0, 0, 0},                     // -128 (fits: no carry)
      {std::ldexp(1.0f, 20), std::ldexp(1.0f, -12), 1, 0, 0, 0, 0, 0},  // 32 binades wide: the small one to 0
      {2.5f / 64, 3.5f / 64, -2.5f / 64, -3.5f / 64, 1.0f, 0, 0, 0},    // ties to even at the block's scale
  };
  for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) std::memcpy(&x[c * 8], cases[c], sizeof cases[c]);
  std::vector<int32_t> tok(parts * NPU_TOKENS);
  for (uint32_t t = 0; t < tok.size(); t++) tok[t] = t % 7 == 6 ? -1 : (int32_t) (next() % rows);  // pad rows among them
  tok[0] = 0;
  float* dx = dev_of(x);
  int32_t* dt = dev_of(tok);
  uint8_t* da;
  CHECK(hipMalloc(&da, parts * npu_a_bytes(K)) == hipSuccess);
  CHECK(npu_pack_a(dx, K, dt, parts, da, nullptr) && hipDeviceSynchronize() == hipSuccess);
  const std::vector<uint8_t> got = host_of(da, parts * npu_a_bytes(K)), want = pack_ref(x, K, tok, parts);
  size_t diff = 0;
  for (size_t i = 0; i < got.size(); i++) diff += got[i] != want[i];
  std::printf("pack K %u, %u parts: %zu of %zu bytes differ\n", K, parts, diff, got.size());
  CHECK(diff == 0);
  hipFree(dx), hipFree(dt), hipFree(da);
}

// C as the kernels write it: [matrix][pass][core][half][row octet][column octet][8 m + n], f32, nr rows a core a pass
static uint64_t c_index(uint32_t N, uint32_t nr, uint32_t mats, uint32_t p, uint32_t mat, uint32_t m, uint32_t n) {
  const uint32_t P = N / (NPU_CORES * nr), ps = n / (NPU_CORES * nr), core = n % (NPU_CORES * nr) / nr, nn = n % nr;
  return (uint64_t) p * npu_c_floats(N, mats) +
         ((((((uint64_t) mat * P + ps) * NPU_CORES + core) * 2 + m / 64) * 8 + m % 64 / 8) * (nr / 8) + nn / 8) * 64 +
         m % 8 * 8 + nn % 8;
}

static void unpack_case(uint32_t N, uint32_t nr, uint32_t mats, uint32_t parts) {
  const uint64_t cf = npu_c_floats(N, mats);
  std::vector<float> c(parts * cf);
  for (auto& v : c) v = rnd();
  std::vector<uint32_t> row0(parts);
  const uint32_t rows = (parts + 2) * NPU_TOKENS;
  for (uint32_t p = 0; p < parts; p++) row0[p] = (parts - 1 - p) * NPU_TOKENS + 64;  // apart, out of order
  const float sentinel = -7.25f;
  std::vector<float> y0((size_t) rows * N, sentinel), y1 = y0;
  float *dc = dev_of(c), *dy0 = dev_of(y0), *dy1 = dev_of(y1);
  uint32_t* dr = dev_of(row0);
  CHECK(npu_unpack_c(dc, N, nr, mats, parts, dr, dy0, mats > 1 ? dy1 : nullptr, nullptr) && hipDeviceSynchronize() == hipSuccess);
  const std::vector<float> g0 = host_of(dy0, y0.size()), g1 = host_of(dy1, y1.size());
  std::vector<bool> theirs(rows, false);
  size_t bad = 0;
  for (uint32_t p = 0; p < parts; p++)
    for (uint32_t m = 0; m < NPU_TOKENS; m++) {
      theirs[row0[p] + m] = true;
      for (uint32_t mat = 0; mat < mats; mat++)
        for (uint32_t n = 0; n < N; n++)
          bad += (mat ? g1 : g0)[(uint64_t) (row0[p] + m) * N + n] != c[c_index(N, nr, mats, p, mat, m, n)];
    }
  size_t touched = 0;
  for (uint32_t r = 0; r < rows; r++)
    if (!theirs[r])
      for (uint32_t n = 0; n < N; n++) touched += g0[(uint64_t) r * N + n] != sentinel || g1[(uint64_t) r * N + n] != sentinel;
  if (mats == 1)
    for (float v : g1) touched += v != sentinel;
  std::printf("unpack N %u, %u rows a core, %u matrices, %u parts: %zu values wrong, %zu outside the parts touched\n", N,
              nr, mats, parts, bad, touched);
  CHECK(bad == 0 && touched == 0);
  hipFree(dc), hipFree(dy0), hipFree(dy1), hipFree(dr);
}

int main() {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || n == 0) return std::printf("no GPU: skipped\n"), 77;
  pack_case(5120, 300, 3);  // gate + up's input
  pack_case(2304, 300, 2);  // the down's
  unpack_case(2304, 24, 2, 3);  // gate + up
  unpack_case(5120, 16, 1, 2);  // the down
  // what the kernels refuse: a K not whole K-steps, an N not whole passes
  CHECK(!npu_pack_a(nullptr, 5000, nullptr, 1, nullptr, nullptr));
  CHECK(!npu_unpack_c(nullptr, 2300, 24, 2, 1, nullptr, nullptr, nullptr, nullptr));
  CHECK(!npu_unpack_c(nullptr, 5120, 12, 1, 1, nullptr, nullptr, nullptr, nullptr));
  std::printf("ok\n");
  return 0;
}
