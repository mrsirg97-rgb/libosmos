// the fused kernels against the kernels they stand for, bit for bit: ops_kv_finish (ops_rms, ops_rope, ops_fp8_round,
// ops_put_rows) - the keys and the window's rows, at positions far apart so the angles are large; ops_rms_x16,
// ops_hc_pre_rms_x16 and ops_swiglu_x16 (each op, then mm_x16) - the F32, every x16 half (the pad rows too: both
// buffers filled with garbage first) and the overflow flag (a value past f16's); ops_gather_sum_add (+ ops_add)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "substrate/gpu/mm.h"
#include "substrate/gpu/ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / (float) (1u << 23) - 1.0f;
}

template <class T> static T* dev(const std::vector<T>& h) {
  T* d = nullptr;
  CHECK(hipMalloc(&d, h.size() * sizeof(T) + 4) == hipSuccess);
  CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice) == hipSuccess);
  return d;
}
template <class T> static std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess);
  return h;
}

static void kv_finish() {
  constexpr uint32_t kHd = 512, kRd = 64, kTok = 5, kSlots = 128;
  std::vector<float> kv((size_t) kTok * kHd), w(kHd), freqs(kRd / 2), ring((size_t) kSlots * kHd, 0.0f);
  for (float& v : kv) v = uniform() * std::ldexp(1.0f, (int) ((uniform() + 1) * 6) - 6);
  for (float& v : w) v = 1.0f + uniform() * 0.5f;
  for (uint32_t i = 0; i < kRd / 2; i++) freqs[i] = (float) std::pow(10000.0, -2.0 * i / kRd);
  const std::vector<int32_t> pos = {7, 300, 4094, 8191, 129};  // rows 7, 44, 126, 127, 1: each its own
  float *a = dev(kv), *b = dev(kv), *dw = dev(w), *df = dev(freqs), *ra = dev(ring), *rb = dev(ring);
  int32_t* dp = dev(pos);
  const float eps = 1e-6f;
  CHECK(ops_rms(a, kHd, kHd, dw, eps, kTok, a, kHd, nullptr) &&
        ops_rope(a, 1, kHd, kRd, kHd, df, dp, false, kTok, nullptr) &&
        ops_fp8_round(a, (uint64_t) kTok * kHd, nullptr) &&
        ops_put_rows(ra, kSlots, a, dp, 1, kHd, kTok, nullptr));
  CHECK(ops_kv_finish(b, kHd, dw, eps, kRd, df, dp, rb, kSlots, kTok, nullptr));
  CHECK(hipDeviceSynchronize() == hipSuccess);
  const std::vector<float> ka = host(a, kv.size()), kb = host(b, kv.size());
  const std::vector<float> wa = host(ra, ring.size()), wb = host(rb, ring.size());
  CHECK(std::memcmp(ka.data(), kb.data(), ka.size() * 4) == 0 && std::memcmp(wa.data(), wb.data(), wa.size() * 4) == 0);
  CHECK(!ops_kv_finish(b, OPS_MAX_KV_DIM + 32, dw, eps, kRd, df, dp, rb, kSlots, kTok, nullptr) &&
        !ops_kv_finish(b, kHd + 16, dw, eps, kRd, df, dp, rb, kSlots, kTok, nullptr));
  for (void* p : {(void*) a, (void*) b, (void*) dw, (void*) df, (void*) ra, (void*) rb, (void*) dp})
    CHECK(hipFree(p) == hipSuccess);
  std::printf("  kv_finish: %u keys at positions to 8,191 - the keys and the window's rows the four kernels' bits\n",
              kTok);
}

// an op's F32 and x16 (and the overflow flag) against the op alone, then mm_x16: `fused(y, x16, over)` and
// `plain(y)` each fill y; the x16 of `rows` rows of `cols`
template <class Fused, class Plain>
static void x16_pair(const char* name, uint32_t rows, uint32_t cols, size_t ylen, Fused fused, Plain plain) {
  const size_t xl = (size_t) (rows + MM_TILE - 1) / MM_TILE * MM_TILE * cols;
  float *ya, *yb;
  uint16_t *xa, *xb;
  int32_t *oa, *ob;
  CHECK(hipMalloc(&ya, ylen * 4) == hipSuccess && hipMalloc(&yb, ylen * 4) == hipSuccess &&
        hipMalloc(&xa, xl * 2) == hipSuccess && hipMalloc(&xb, xl * 2) == hipSuccess &&
        hipMalloc(&oa, 4) == hipSuccess && hipMalloc(&ob, 4) == hipSuccess);
  CHECK(hipMemset(xa, 0xA5, xl * 2) == hipSuccess && hipMemset(xb, 0x5A, xl * 2) == hipSuccess &&
        hipMemset(oa, 0, 4) == hipSuccess && hipMemset(ob, 0, 4) == hipSuccess &&
        hipMemset(ya, 0, ylen * 4) == hipSuccess && hipMemset(yb, 0, ylen * 4) == hipSuccess);
  CHECK(plain(ya) && mm_x16(ya, cols, rows, cols, xa, oa, nullptr));
  CHECK(fused(yb, xb, ob));
  CHECK(hipDeviceSynchronize() == hipSuccess);
  CHECK(std::memcmp(host(ya, ylen).data(), host(yb, ylen).data(), ylen * 4) == 0);
  CHECK(host(xa, xl) == host(xb, xl) && host(oa, 1) == host(ob, 1));
  const int32_t flag = host(ob, 1)[0];
  for (void* p : {(void*) ya, (void*) yb, (void*) xa, (void*) xb, (void*) oa, (void*) ob})
    CHECK(hipFree(p) == hipSuccess);
  std::printf("  %s: %u rows of %u - the F32, the x16 to a whole tile and the overflow flag (%d) the op's and "
              "mm_x16's\n", name, rows, cols, flag);
}

static void x16_ops(uint32_t n, bool overflow) {
  constexpr uint32_t kD = 1024, kHc = 4, kFf = 256;
  std::vector<float> x((size_t) n * kD), w(kD), h((size_t) n * kHc * kD), pre((size_t) n * kHc);
  std::vector<float> g((size_t) n * kFf), u((size_t) n * kFf), sc(n);
  for (float& v : x) v = uniform();
  for (float& v : w) v = 1.0f + uniform() * 0.5f;
  for (float& v : h) v = uniform();
  for (float& v : pre) v = 0.5f + uniform() * 0.25f;
  for (float& v : g) v = uniform() * 8.0f;
  for (float& v : u) v = uniform() * 8.0f;
  for (float& v : sc) v = 0.5f + uniform() * 0.25f;
  if (overflow) w[17] = 1e9f, u[3] = 70000.0f, sc[0] = 1e6f;  // a value past f16 in each op's output
  float *dx = dev(x), *dw = dev(w), *dh = dev(h), *dp = dev(pre), *dg = dev(g), *du = dev(u), *ds = dev(sc);
  const float eps = 1e-6f, lim = 7.0f;
  x16_pair("rms_x16", n, kD, (size_t) n * kD,
           [&](float* y, void* x16, int32_t* o) { return ops_rms_x16(dx, kD, kD, dw, eps, n, y, kD, x16, o, nullptr); },
           [&](float* y) { return ops_rms(dx, kD, kD, dw, eps, n, y, kD, nullptr); });
  x16_pair("hc_pre_rms_x16", n, kD, (size_t) n * kD,
           [&](float* y, void* x16, int32_t* o) {
             return ops_hc_pre_rms_x16(dh, dp, kHc, kD, dw, eps, n, y, x16, o, nullptr);
           },
           [&](float* y) { return ops_hc_pre_rms(dh, dp, kHc, kD, dw, eps, n, y, nullptr); });
  x16_pair("swiglu_x16", n, kFf, (size_t) n * kFf,
           [&](float* y, void* x16, int32_t* o) { return ops_swiglu_x16(dg, du, kFf, lim, ds, n, y, x16, o, nullptr); },
           [&](float* y) { return ops_swiglu(dg, du, kFf, lim, ds, n, y, nullptr); });
  for (void* p : {(void*) dx, (void*) dw, (void*) dh, (void*) dp, (void*) dg, (void*) du, (void*) ds})
    CHECK(hipFree(p) == hipSuccess);
}

static void gather_sum_add() {
  constexpr uint32_t kD = 1024, kTok = 3, kK = 6, kRows = kTok * kK;
  std::vector<float> src((size_t) kRows * kD), add((size_t) kTok * kD);
  std::vector<int32_t> idx(kTok * kK);
  for (float& v : src) v = uniform();
  for (float& v : add) v = uniform();
  for (uint32_t i = 0; i < kTok * kK; i++) idx[i] = (int32_t) ((i * 7) % kRows);
  float *ds = dev(src), *da = dev(add), *ya, *yb;
  int32_t* di = dev(idx);
  CHECK(hipMalloc(&ya, add.size() * 4) == hipSuccess && hipMalloc(&yb, add.size() * 4) == hipSuccess);
  CHECK(ops_gather_sum(ya, ds, di, kK, kTok, kD, nullptr) && ops_add(ya, da, add.size(), nullptr));
  CHECK(ops_gather_sum_add(yb, ds, di, kK, kTok, kD, da, nullptr) && hipDeviceSynchronize() == hipSuccess);
  const std::vector<float> a = host(ya, add.size()), b = host(yb, add.size());
  CHECK(std::memcmp(a.data(), b.data(), a.size() * 4) == 0);
  for (void* p : {(void*) ds, (void*) da, (void*) ya, (void*) yb, (void*) di}) CHECK(hipFree(p) == hipSuccess);
  std::printf("  gather_sum_add: %u tokens' %u rows each, then the add - ops_gather_sum and ops_add's bits\n", kTok,
              kK);
}

int main() {
  std::printf("[test_fuse_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_fuse_gpu] skipped: no GPU\n");
    return 77;
  }
  kv_finish();
  x16_ops(3, false);   // a tile's 13 pad rows
  x16_ops(16, true);   // a whole tile, and the overflow flag
  gather_sum_add();
  std::printf("[test_fuse_gpu] ok\n");
  return 0;
}
