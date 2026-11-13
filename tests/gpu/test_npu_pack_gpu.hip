// the NPU's GPU side (kernels/npu_pack.hip) against host references written from npu_pack.h's contract: the A stream's
// bytes for random rows and the cases its rounding turns on (zeros, -0, subnormals, a mantissa that rounds to 128 -
// the exponent raised, -128, a block 32 binades wide, pad rows), parts of DeepSeek's kinds (gate + up's K 5120, the
// down's 2304) from f32, and Qwen's down's (K 640 in K halves) from f16 rows; and Qwen's part selection (its tables'
// whole 128-token entries, in order, their counts zeroed, each's weights, offsets, rows and slots). needs a GPU;
// skipped (77) without one.
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
static std::vector<uint8_t> pack_ref(const std::vector<float>& x, uint32_t K, const std::vector<int32_t>& tok, uint32_t parts,
                                     bool halves = false) {
  std::vector<uint8_t> a(parts * npu_a_bytes(K));
  const uint64_t piece = 64 * 128 * 9 / 8;
  for (uint32_t p = 0; p < parts; p++)
    for (uint32_t m = 0; m < NPU_TOKENS; m++)
      for (uint32_t kb = 0; kb < K / 8; kb++) {
        float v[8];
        const int32_t row = tok[p * NPU_TOKENS + m];
        for (int j = 0; j < 8; j++) v[j] = row < 0 ? 0.0f : x[(uint64_t) row * K + kb * 8 + j];
        const uint32_t s = kb / 32, kq = kb % 32 / 16, sx = kb % 16, h = m / 64, sy = m % 64 / 8, i = m % 8;
        const uint32_t pc = halves ? (kb / 16) * 2 + h : (s * 2 + h) * 2 + kq;
        block_ref(v, &a[p * npu_a_bytes(K) + pc * piece + ((sy * 16 + sx) * 8 + i) * 9]);
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
  (void) hipFree(dx), (void) hipFree(dt), (void) hipFree(da);
}

// Qwen's down's input: f16 rows (the swiglu's x16 slots), packed in K halves - against the same reference, its values
// the f16's
static void pack16_case(uint32_t K, uint32_t rows, uint32_t parts) {
  std::vector<_Float16> x16((size_t) rows * K);
  std::vector<float> x((size_t) rows * K);
  for (size_t i = 0; i < x.size(); i++) x16[i] = (_Float16) (rnd() * 4), x[i] = (float) x16[i];
  std::vector<int32_t> slot(parts * NPU_TOKENS);
  for (uint32_t t = 0; t < slot.size(); t++) slot[t] = (int32_t) (next() % rows);
  _Float16* dx = dev_of(x16);
  int32_t* ds = dev_of(slot);
  uint8_t* da;
  CHECK(hipMalloc(&da, parts * npu_a_bytes(K)) == hipSuccess);
  CHECK(npu_pack_a_x16(dx, K, ds, parts, true, da, nullptr) && hipDeviceSynchronize() == hipSuccess);
  const std::vector<uint8_t> got = host_of(da, parts * npu_a_bytes(K)), want = pack_ref(x, K, slot, parts, true);
  size_t diff = 0;
  for (size_t i = 0; i < got.size(); i++) diff += got[i] != want[i];
  std::printf("pack f16 K %u in halves, %u parts: %zu of %zu bytes differ\n", K, parts, diff, got.size());
  CHECK(diff == 0);
  (void) hipFree(dx), (void) hipFree(ds), (void) hipFree(da);
}

// Qwen's selection over a table of entries whose 128-token run is [first, first + count)
static void select_case(uint32_t want) {
  const uint32_t entries = 40, first = 25, count = 12, D = 64, ff = 32;
  std::vector<int32_t> cnt(entries, 16);
  const int32_t run[12] = {128, 90, 128, 128, 0, 128, 128, 64, 128, 1, 128, 128};  // whole: 25, 27, 28, 30, ..
  for (uint32_t i = 0; i < count; i++) cnt[first + i] = run[i];
  cnt[3] = 128;                                                                    // whole, but not in the run
  std::vector<uintptr_t> ptr(3 * entries);
  std::vector<uint32_t> xg(entries), yg(entries), xd(entries), yd(entries);
  for (uint32_t e = 0; e < entries; e++) {
    for (uint32_t k = 0; k < 3; k++) ptr[k * entries + e] = 0x1000 * (k + 1) + e;
    xg[e] = e * 128 * D, yg[e] = e * 128 * ff + 7, xd[e] = e * 128 * ff, yd[e] = e * 128 * D + 3;
  }
  std::vector<int32_t> gidx(entries * 128);
  for (size_t i = 0; i < gidx.size(); i++) gidx[i] = (int32_t) (i * 7 % 1000);
  int32_t *dc = dev_of(cnt), *dg = dev_of(gidx), *dt, *ds;
  uintptr_t* dp = dev_of(ptr);
  uint32_t *dxg = dev_of(xg), *dyg = dev_of(yg), *dxd = dev_of(xd), *dyd = dev_of(yd), *dn;
  NpuPart* dsel;
  CHECK(hipMalloc(&dt, want * 128 * 4) == hipSuccess && hipMalloc(&ds, want * 128 * 4) == hipSuccess &&
        hipMalloc(&dsel, NPU_SELECT_MAX * sizeof(NpuPart)) == hipSuccess && hipMalloc(&dn, 4) == hipSuccess);
  CHECK(npu_select_q4x(dc, first, count, want, (const void* const*) dp, entries, dxg, dyg, dxd, dyd, dg, D, ff, dsel, dn,
                       dt, ds, nullptr) && hipDeviceSynchronize() == hipSuccess);
  const std::vector<int32_t> c2 = host_of(dc, entries), t2 = host_of(dt, want * 128), s2 = host_of(ds, want * 128);
  const std::vector<NpuPart> sel = host_of(dsel, want);
  const std::vector<uint32_t> n = host_of(dn, 1);
  const uint32_t whole[8] = {25, 27, 28, 30, 31, 33, 35, 36}, got = want < 8 ? want : 8;
  const uint32_t* taken = whole;
  CHECK(n[0] == got);
  for (uint32_t p = 0; p < got; p++) {
    const uint32_t e = taken[p];
    CHECK(c2[e] == 0 && (uintptr_t) sel[p].gate == 0x1000 + e && (uintptr_t) sel[p].up == 0x2000 + e &&
          (uintptr_t) sel[p].down == 0x3000 + e && sel[p].yg == yg[e] && sel[p].yd == yd[e]);
    for (uint32_t t = 0; t < 128; t++)
      CHECK(t2[p * 128 + t] == gidx[e * 128 + t] && s2[p * 128 + t] == (int32_t) (e * 128 + t));
  }
  for (uint32_t i = got * 128; i < want * 128; i++)  // the parts past those taken: zero rows
    CHECK(t2[i] == -1 && s2[i] == -1);
  for (uint32_t e = 0; e < entries; e++)  // every other entry's count as it was
    CHECK((e >= first && e < first + count && cnt[e] == 128 && e <= whole[got - 1]) ? c2[e] == 0 : c2[e] == cnt[e]);
  std::printf("select: the run's first %u whole entries (in order, %u wanted), their counts zeroed alone, the rest "
              "zero rows\n", got, want);
  (void) hipFree(dc), (void) hipFree(dg), (void) hipFree(dt), (void) hipFree(ds), (void) hipFree(dp), (void) hipFree(dsel);
  (void) hipFree(dxg), (void) hipFree(dyg), (void) hipFree(dxd), (void) hipFree(dyd), (void) hipFree(dn);
}

int main() {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || n == 0) return std::printf("no GPU: skipped\n"), 77;
  pack_case(5120, 300, 3);  // gate + up's input
  pack_case(2304, 300, 2);  // the down's
  pack16_case(640, 300, 2);  // Qwen's down's, in K halves
  select_case(3);
  select_case(10);  // more wanted than whole: 8 taken, 2 parts of zero rows
  CHECK(!npu_pack_a(nullptr, 5000, nullptr, 1, nullptr, nullptr));     // a K not whole K-steps
  CHECK(!npu_pack_a_x16(nullptr, 600, nullptr, 1, true, nullptr, nullptr));
  CHECK(!npu_select_q4x(nullptr, 0, 0, NPU_SELECT_MAX + 1, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 1, 1,
                        nullptr, nullptr, nullptr, nullptr, nullptr));  // more parts than the most
  std::printf("ok\n");
  return 0;
}
