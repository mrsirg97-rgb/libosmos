// the fused kernels against the kernels they stand for, bit for bit: ops_kv_finish (ops_rms, ops_rope, ops_fp8_round,
// ops_put_rows) - the keys and the window's rows, at positions far apart so the angles are large; ops_rms_x16,
// ops_hc_pre_rms_x16 and ops_swiglu_x16 (each op, then mm_x16) - the F32, every x16 half (the pad rows too: both
// buffers filled with garbage first) and the overflow flag (a value past f16's); ops_gather_sum_add (+ ops_add);
// ops_hc_post_dot (ops_hc_post, ops_hc_dot) and ops_hc_mix_pre_rms (ops_hc_mix, ops_hc_pre_rms(_x16))
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "substrate/gpu/mm.h"
#include "substrate/gpu/kvfmt.h"
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
  const uint64_t rowb = kv_row_bytes(KV_FP8, kHd);  // the window's rows as a sequence keeps them (kvfmt.h)
  uint8_t* pa = nullptr;
  CHECK(hipMalloc(&pa, kTok * rowb) == hipSuccess);
  CHECK(ops_rms(a, kHd, kHd, dw, eps, kTok, a, kHd, nullptr) &&
        ops_rope(a, 1, kHd, kRd, kHd, df, dp, false, kTok, nullptr) &&
        ops_fp8_round(a, (uint64_t) kTok * kHd, pa, kHd, nullptr) &&
        ops_put_rows(ra, kSlots, pa, dp, 1, KV_FP8, kHd, kTok, nullptr));
  CHECK(ops_kv_finish(b, kHd, dw, eps, kRd, df, dp, rb, kSlots, kTok, nullptr));
  CHECK(hipDeviceSynchronize() == hipSuccess);
  const std::vector<float> ka = host(a, kv.size()), kb = host(b, kv.size());
  const std::vector<float> wa = host(ra, ring.size()), wb = host(rb, ring.size());  // the rings' KV_FP8 rows
  CHECK(std::memcmp(ka.data(), kb.data(), ka.size() * 4) == 0 && std::memcmp(wa.data(), wb.data(), kSlots * rowb) == 0);
  CHECK(!ops_kv_finish(b, OPS_MAX_KV_DIM + 32, dw, eps, kRd, df, dp, rb, kSlots, kTok, nullptr) &&
        !ops_kv_finish(b, kHd + 16, dw, eps, kRd, df, dp, rb, kSlots, kTok, nullptr));
  for (void* p : {(void*) a, (void*) b, (void*) pa, (void*) dw, (void*) df, (void*) ra, (void*) rb, (void*) dp})
    CHECK(hipFree(p) == hipSuccess);
  std::printf("  kv_finish: %u keys at positions to 8,191 - the keys and the window's packed rows the four kernels'"
              " bits\n",
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

constexpr uint32_t kHc = 4, kHcDim = 4096;  // DeepSeek-V4.1-Flash's streams: 4 of n_embd
constexpr uint32_t kHcRows = (2 + kHc) * kHc + 1;     // the mixes' sums and the norm's

// ops_hc_post_dot against ops_hc_post then ops_hc_dot: the streams out and every part's sums
static void hc_post_dot(uint32_t n) {
  const size_t len = (size_t) kHc * kHcDim;
  std::vector<float> y((size_t) n * kHcDim), res(n * len), post(n * kHc), comb(n * kHc * kHc), fn(kHcRows * len);
  for (float& v : y) v = uniform();
  for (float& v : res) v = uniform() * 2.0f;
  for (float& v : post) v = uniform() + 1.0f;
  for (float& v : comb) v = (uniform() + 1.0f) * 0.25f;
  for (float& v : fn) v = uniform() * 0.1f;
  float *dy = dev(y), *dr = dev(res), *dp = dev(post), *dc = dev(comb), *df = dev(fn);
  const size_t parts = (size_t) n * kHcRows * OPS_HC_PARTS;
  float *oa, *ob, *pa, *pb;
  CHECK(hipMalloc(&oa, n * len * 4) == hipSuccess && hipMalloc(&ob, n * len * 4) == hipSuccess &&
        hipMalloc(&pa, parts * 4) == hipSuccess && hipMalloc(&pb, parts * 4) == hipSuccess);
  CHECK(ops_hc_post(dy, dr, dp, dc, kHc, kHcDim, n, oa, nullptr) && ops_hc_dot(df, oa, kHc, kHcDim, n, pa, nullptr));
  CHECK(ops_hc_post_dot(dy, dr, dp, dc, kHc, kHcDim, n, ob, df, pb, nullptr) && hipDeviceSynchronize() == hipSuccess);
  CHECK(host(oa, n * len) == host(ob, n * len));
  const std::vector<float> ha = host(pa, parts), hb = host(pb, parts);
  CHECK(std::memcmp(ha.data(), hb.data(), parts * 4) == 0);
  for (void* q : {(void*) dy, (void*) dr, (void*) dp, (void*) dc, (void*) df, (void*) oa, (void*) ob, (void*) pa,
                  (void*) pb})
    CHECK(hipFree(q) == hipSuccess);
  std::printf("  hc_post_dot: %u token%s - ops_hc_post and ops_hc_dot's bits (the streams, every part's sums)\n", n,
              n == 1 ? "" : "s");
}

// ops_hc_mix_pre_rms against ops_hc_mix then ops_hc_pre_rms(_x16): the mixes, x, x16 and the flag - the norm's pre
// another's, or the mix's own
static void hc_mix_pre_rms(uint32_t n, bool x16, bool own) {
  const size_t len = (size_t) kHc * kHcDim;
  std::vector<float> h(n * len), fn(kHcRows * len), scale(3), base((2 + kHc) * kHc), pre(n * kHc), w(kHcDim);
  for (float& v : h) v = uniform() * 2.0f;
  for (float& v : fn) v = uniform() * 0.1f;
  for (float& v : scale) v = uniform() + 1.5f;
  for (float& v : base) v = uniform();
  for (float& v : pre) v = uniform() + 1.0f;
  for (float& v : w) v = 1.0f + uniform() * 0.5f;
  float *dh = dev(h), *df = dev(fn), *ds = dev(scale), *db = dev(base), *dpre = dev(pre), *dw = dev(w), *part;
  CHECK(hipMalloc(&part, (size_t) n * kHcRows * OPS_HC_PARTS * 4) == hipSuccess);
  CHECK(ops_hc_dot(df, dh, kHc, kHcDim, n, part, nullptr));
  const float eps = 1e-6f, hc_eps = 1e-6f;
  constexpr uint32_t kIters = 20;
  const uint32_t rows = x16 ? (n + MM_TILE - 1) / MM_TILE * MM_TILE : n;
  float *m[2][3], *xo[2];
  void* x16o[2] = {nullptr, nullptr};
  int32_t* ov[2] = {nullptr, nullptr};
  for (int k = 0; k < 2; k++) {
    for (int j = 0; j < 3; j++) CHECK(hipMalloc(&m[k][j], (size_t) n * kHc * kHc * 4) == hipSuccess);
    CHECK(hipMalloc(&xo[k], (size_t) n * kHcDim * 4) == hipSuccess);
    if (x16) {
      CHECK(hipMalloc(&x16o[k], (size_t) rows * kHcDim * 2) == hipSuccess && hipMalloc(&ov[k], 4) == hipSuccess);
      CHECK(hipMemset(x16o[k], 0x7F, (size_t) rows * kHcDim * 2) == hipSuccess && hipMemset(ov[k], 0, 4) == hipSuccess);
    }
  }
  const float* p0 = own ? m[0][0] : dpre;  // the norm's pre: another's, or the mix's own
  const float* p1 = own ? m[1][0] : dpre;
  CHECK(ops_hc_mix(part, ds, db, kHc, kHcDim, eps, hc_eps, kIters, n, m[0][0], m[0][1], m[0][2], nullptr) &&
        (x16 ? ops_hc_pre_rms_x16(dh, p0, kHc, kHcDim, dw, eps, n, xo[0], x16o[0], ov[0], nullptr)
             : ops_hc_pre_rms(dh, p0, kHc, kHcDim, dw, eps, n, xo[0], nullptr)));
  CHECK(ops_hc_mix_pre_rms(part, ds, db, kHc, kHcDim, eps, hc_eps, kIters, n, m[1][0], m[1][1], m[1][2], dh, p1, dw,
                           xo[1], x16o[1], ov[1], nullptr) &&
        hipDeviceSynchronize() == hipSuccess);
  CHECK(host(m[0][0], n * kHc) == host(m[1][0], n * kHc) && host(m[0][1], n * kHc) == host(m[1][1], n * kHc) &&
        host(m[0][2], n * kHc * kHc) == host(m[1][2], n * kHc * kHc));
  const std::vector<float> xa = host(xo[0], (size_t) n * kHcDim), xb = host(xo[1], (size_t) n * kHcDim);
  CHECK(std::memcmp(xa.data(), xb.data(), xa.size() * 4) == 0);
  if (x16) {
    const std::vector<uint16_t> a = host((const uint16_t*) x16o[0], (size_t) rows * kHcDim),
                                b = host((const uint16_t*) x16o[1], (size_t) rows * kHcDim);
    CHECK(a == b && host(ov[0], 1) == host(ov[1], 1));
  }
  for (int k = 0; k < 2; k++) {
    for (int j = 0; j < 3; j++) CHECK(hipFree(m[k][j]) == hipSuccess);
    CHECK(hipFree(xo[k]) == hipSuccess && hipFree(x16o[k]) == hipSuccess && hipFree(ov[k]) == hipSuccess);
  }
  for (void* q : {(void*) dh, (void*) df, (void*) ds, (void*) db, (void*) dpre, (void*) dw, (void*) part})
    CHECK(hipFree(q) == hipSuccess);
  std::printf("  hc_mix_pre_rms: %u token%s%s, the norm's pre %s - ops_hc_mix and ops_hc_pre_rms%s's bits\n", n,
              n == 1 ? "" : "s", x16 ? ", x16" : "", own ? "the mix's own" : "another's", x16 ? "_x16" : "");
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
  for (uint32_t n : {1u, 3u, 9u, 70u}) hc_post_dot(n);  // 70: a tile's worth, the two launches
  hc_mix_pre_rms(1, false, false);
  hc_mix_pre_rms(1, true, false);
  hc_mix_pre_rms(5, true, true);
  hc_mix_pre_rms(9, false, true);
  std::printf("[test_fuse_gpu] ok\n");
  return 0;
}
