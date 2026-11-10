// mm_run_gated_mean (the mixer's up with the gated mean as its epilogue) against the two calls it stands for, mm_run
// then q4x_hc_gated_mean: the mean's F32 and its f16 copy the same bytes, the overflow flag the same - on random Q8_0
// weights at Qwen3.8's mixer shape (4 streams of 2,560, rank 320) and a smaller one, at token counts across every
// column-tile count; the refusals (out16 = x16, a matrix the gemm takes, a type without the epilogue). needs a GPU;
// skipped (77) without one.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/mm.h"
#include "osmos/gpu/q4x_ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr uint32_t kQ8Values = 32, kQ8Bytes = 34, kQ8_0 = 8;
constexpr uint8_t kFill = 0x5A;  // the outputs' bytes before a call

static uint32_t rng = 1017;
static uint32_t next() { return rng = rng * 1664525u + 1013904223u; }
static float uniform() { return (float) (next() >> 8) / (float) (1 << 23) - 1.0f; }  // [-1, 1)

template <class V>
static V* dev(const std::vector<V>& h) {
  V* d = nullptr;
  CHECK(hipMalloc(&d, h.size() * sizeof(V)) == hipSuccess);
  CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(V), hipMemcpyHostToDevice) == hipSuccess);
  return d;
}
static std::vector<uint8_t> host(const void* d, size_t bytes) {
  std::vector<uint8_t> h(bytes);
  CHECK(hipMemcpy(h.data(), d, bytes, hipMemcpyDeviceToHost) == hipSuccess);
  return h;
}

// one shape: hc streams of dim (the up's hc dim rows), rank r (its columns), n tokens
static void run(uint32_t hc, uint32_t dim, uint32_t r, uint32_t n) {
  const uint32_t rows = hc * dim, tiles = (n + MM_TILE - 1) / MM_TILE * MM_TILE;
  std::vector<uint8_t> wh((size_t) rows * (r / kQ8Values) * kQ8Bytes);
  for (size_t b = 0; b < wh.size(); b += kQ8Bytes) {
    const _Float16 d = (_Float16) (std::ldexp(1.0f, -(int) (next() % 8) - 3) * (next() & 1 ? -1.0f : 1.0f));
    std::memcpy(&wh[b], &d, 2);
    for (uint32_t i = 2; i < kQ8Bytes; i++) wh[b + i] = (uint8_t) (next() >> 24);
  }
  std::vector<float> lo((size_t) n * r), xn((size_t) n * rows);
  for (float& v : lo) v = uniform();
  for (float& v : xn) v = 3.0f * uniform();
  if (n > 3) xn[(size_t) 2 * rows + 5] = 1e30f;  // one token's mean past f16's range: the flag set, the same way
  uint8_t* w = dev(wh);
  float *dlo = dev(lo), *dxn = dev(xn);
  float *y = nullptr, *outa = nullptr, *outb = nullptr;
  void *x16 = nullptr, *ha = nullptr, *hb = nullptr;
  int32_t *ova = nullptr, *ovb = nullptr;
  const size_t yb = (size_t) n * rows * 4, ob = (size_t) n * dim * 4, h16 = (size_t) tiles * dim * 2;
  CHECK(hipMalloc(&y, yb) == hipSuccess && hipMalloc(&outa, ob) == hipSuccess && hipMalloc(&outb, ob) == hipSuccess);
  CHECK(hipMalloc(&x16, (size_t) tiles * r * 2) == hipSuccess && hipMalloc(&ha, h16) == hipSuccess &&
        hipMalloc(&hb, h16) == hipSuccess);
  CHECK(hipMalloc(&ova, 4) == hipSuccess && hipMalloc(&ovb, 4) == hipSuccess);
  CHECK(hipMemset(outa, kFill, ob) == hipSuccess && hipMemset(outb, kFill, ob) == hipSuccess);
  CHECK(hipMemset(ha, kFill, h16) == hipSuccess && hipMemset(hb, kFill, h16) == hipSuccess);
  CHECK(hipMemset(ova, 0, 4) == hipSuccess && hipMemset(ovb, 0, 4) == hipSuccess);
  CHECK(mm_x16(dlo, r, n, r, x16, ova, nullptr) && hipMemset(ova, 0, 4) == hipSuccess);
  // the two calls: the up's F32 out, then the gated mean (its f16 beside)
  const Q4xX16 side = {hb, ovb, dim, 0};
  CHECK(mm_run(kQ8_0, w, rows, r, x16, n, y, rows, nullptr));
  CHECK(q4x_hc_gated_mean(dxn, y, n, hc, dim, outb, &side, nullptr));
  CHECK(mm_run_gated_mean(kQ8_0, w, hc, dim, r, x16, n, dxn, outa, ha, 0, ova, nullptr));
  CHECK(hipDeviceSynchronize() == hipSuccess);
  const std::vector<uint8_t> oa = host(outa, ob), obv = host(outb, ob), a16 = host(ha, h16), b16 = host(hb, h16);
  const std::vector<uint8_t> fa = host(ova, 4), fb = host(ovb, 4);
  size_t bad = 0, bad16 = 0;
  for (size_t i = 0; i < ob; i++) bad += oa[i] != obv[i];
  for (size_t i = 0; i < h16; i++) bad16 += a16[i] != b16[i];
  if (bad || bad16) std::fprintf(stderr, "  %u x %u, rank %u, %u tokens: %zu F32 bytes, %zu f16 bytes differ\n", hc, dim, r, n, bad, bad16);
  CHECK(bad == 0 && bad16 == 0 && std::memcmp(fa.data(), fb.data(), 4) == 0);
  // without the f16 copy: the F32 alone, the same
  CHECK(hipMemset(outa, kFill, ob) == hipSuccess);
  CHECK(mm_run_gated_mean(kQ8_0, w, hc, dim, r, x16, n, dxn, outa, nullptr, 0, nullptr, nullptr));
  CHECK(hipDeviceSynchronize() == hipSuccess && host(outa, ob) == obv);
  int32_t flag = 0;
  std::memcpy(&flag, fa.data(), 4);
  std::printf("  %u streams x %u, rank %u, %4u tokens: the two calls' bytes (F32, f16), overflow %s\n", hc, dim, r, n,
              flag ? "set alike" : "clear");
  CHECK(hipFree(w) == hipSuccess && hipFree(dlo) == hipSuccess && hipFree(dxn) == hipSuccess && hipFree(y) == hipSuccess);
  CHECK(hipFree(outa) == hipSuccess && hipFree(outb) == hipSuccess && hipFree(x16) == hipSuccess);
  CHECK(hipFree(ha) == hipSuccess && hipFree(hb) == hipSuccess && hipFree(ova) == hipSuccess && hipFree(ovb) == hipSuccess);
}

int main() {
  std::printf("[test_gated_mean_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_gated_mean_gpu] skipped: no GPU\n");
    return 77;
  }
  for (uint32_t n : {1u, 5u, 16u, 17u, 37u, 64u, 100u, 2048u}) run(4, 2560, 320, n);  // Qwen3.8's mixers
  for (uint32_t n : {3u, 33u}) run(2, 64, 64, n);
  {  // refusals: nothing launched
    void* p = nullptr;
    CHECK(hipMalloc(&p, 1 << 20) == hipSuccess);
    float* f = (float*) p;
    CHECK(!mm_run_gated_mean(kQ8_0, p, 4, 64, 64, p, 8, f, f, p, 0, (int32_t*) p, nullptr));   // out16 = x16
    CHECK(!mm_run_gated_mean(kQ8_0, p, 4, 64, 4096, p, 128, f, f, nullptr, 0, nullptr, nullptr));  // the gemm's
    CHECK(!mm_run_gated_mean(14, p, 4, 64, 256, p, 8, f, f, nullptr, 0, nullptr, nullptr));    // Q6_K: no epilogue
    CHECK(!mm_run_gated_mean(kQ8_0, p, 4, 40, 64, p, 8, f, f, nullptr, 0, nullptr, nullptr));  // dim not 16s
    CHECK(!mm_run_gated_mean(kQ8_0, p, MM_GATED_MEAN_MAX_HC + 1, 64, 64, p, 8, f, f, nullptr, 0, nullptr, nullptr));
    CHECK(hipDeviceSynchronize() == hipSuccess && hipFree(p) == hipSuccess);
    std::printf("  refusals: out16 the input, a matrix the gemm takes, Q6_K, dim not 16s, %d streams\n",
                MM_GATED_MEAN_MAX_HC + 1);
  }
  std::printf("[test_gated_mean_gpu] ok\n");
  return 0;
}
