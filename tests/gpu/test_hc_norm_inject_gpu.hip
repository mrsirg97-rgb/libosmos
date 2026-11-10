// q4x_hc_norm_inject (the mixer's norm and its inject, a block a token) against the two calls it stands for:
// q4x_hc_combine_norm or q4x_hc_norm, then mv_strided of the F32 inject over the xn written - h, xn, xn's f16 and the
// inject the same bytes, the inject in place (its input the step before's, read first, as the forward has it) and
// apart - at Qwen3.8's shape (4 streams of 2,560) over the matvec's paths (up to 8 tokens and past), and a smaller
// one whose streams' chunks are not the lanes' multiple; the refusals. needs a GPU; skipped (77) without one.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/mv.h"
#include "osmos/model/qtype.h"
#include "osmos/gpu/q4x_ops.h"
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
static std::vector<float> random(size_t n, float scale) {
  std::vector<float> h(n);
  for (float& v : h) v = scale * uniform();
  return h;
}
static float* dev(const std::vector<float>& h) {
  float* d = nullptr;
  CHECK(hipMalloc(&d, h.size() * 4) == hipSuccess && hipMemcpy(d, h.data(), h.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  return d;
}
static std::vector<uint8_t> bytes(const void* d, size_t n) {
  std::vector<uint8_t> h(n);
  CHECK(hipMemcpy(h.data(), d, n, hipMemcpyDeviceToHost) == hipSuccess);
  return h;
}

// a side's f16 rows padded to tiles of 16
static size_t x16_bytes(uint32_t n, uint64_t cols) { return (size_t) (n + 15) / 16 * 16 * cols * 2; }

static void run(uint32_t n, uint32_t hc, uint32_t dim, bool combine, bool in_place) {
  const uint64_t hd = (uint64_t) hc * dim;
  const std::vector<float> h0 = random(n * hd, 3.0f), out = random((size_t) n * dim, 2.0f), inj0 = random((size_t) n * hc, 2.0f);
  const std::vector<float> w = random(hd, 1.5f), winj = random(hc * hd, 0.05f);
  float *ha = dev(h0), *hb = dev(h0), *dout = dev(out), *dw = dev(w), *dwi = dev(winj);
  float *ia = dev(inj0), *ib = dev(inj0), *ja = in_place ? ia : dev(std::vector<float>((size_t) n * hc));
  float *xa = nullptr, *xb = nullptr;
  int32_t *ova = nullptr, *ovb = nullptr;
  void *sa = nullptr, *sb = nullptr;
  const size_t sx = x16_bytes(n, hd);
  CHECK(hipMalloc(&xa, n * hd * 4) == hipSuccess && hipMalloc(&xb, n * hd * 4) == hipSuccess);
  CHECK(hipMalloc(&sa, sx) == hipSuccess && hipMalloc(&sb, sx) == hipSuccess && hipMemset(sa, 0x5A, sx) == hipSuccess &&
        hipMemset(sb, 0x5A, sx) == hipSuccess);
  CHECK(hipMalloc(&ova, 4) == hipSuccess && hipMalloc(&ovb, 4) == hipSuccess && hipMemset(ova, 0, 4) == hipSuccess &&
        hipMemset(ovb, 0, 4) == hipSuccess);
  const Q4xX16 side_a = {sa, ova, (uint32_t) hd, 0}, side_b = {sb, ovb, (uint32_t) hd, 0};
  const float scale = 1.0f / (float) hc, eps = 1e-6f;
  // the two calls: the norm (its combine reading the step before's inject), then the inject over its xn, in place
  CHECK(combine ? q4x_hc_combine_norm(hb, dout, ib, n, hc, dim, scale, dw, eps, xb, &side_b, nullptr)
                : q4x_hc_norm(hb, dw, n, hc, dim, eps, xb, &side_b, nullptr));
  CHECK(mv_strided(QTYPE_F32, dwi, hc, (uint32_t) hd, xb, (uint32_t) hd, n, ib, hc, nullptr));
  CHECK(q4x_hc_norm_inject(ha, dout, ia, n, hc, dim, scale, dw, eps, xa, &side_a, dwi, ja, combine, nullptr));
  CHECK(hipDeviceSynchronize() == hipSuccess);
  const bool same = bytes(ha, n * hd * 4) == bytes(hb, n * hd * 4) && bytes(xa, n * hd * 4) == bytes(xb, n * hd * 4) &&
                    bytes(sa, sx) == bytes(sb, sx) && bytes(ja, (size_t) n * hc * 4) == bytes(ib, (size_t) n * hc * 4) &&
                    bytes(ova, 4) == bytes(ovb, 4);
  if (!same) {
    const std::vector<uint8_t> a = bytes(ja, (size_t) n * hc * 4), b = bytes(ib, (size_t) n * hc * 4);
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); i += 4) bad += std::memcmp(&a[i], &b[i], 4) != 0;
    std::fprintf(stderr, "  %u tokens, %u x %u%s: differ (%zu of the inject's %u)\n", n, hc, dim,
                 combine ? ", combined" : "", bad, n * hc);
  }
  CHECK(same);
  std::printf("  %4u tokens, %u streams x %4u%s, inject %s: the two calls' bytes\n", n, hc, dim,
              combine ? ", combined" : "", in_place ? "in place" : "apart");
  for (float* p : {ha, hb, dout, dw, dwi, ia, ib, xa, xb}) CHECK(hipFree(p) == hipSuccess);
  if (!in_place) CHECK(hipFree(ja) == hipSuccess);
  CHECK(hipFree(sa) == hipSuccess && hipFree(sb) == hipSuccess && hipFree(ova) == hipSuccess && hipFree(ovb) == hipSuccess);
}

int main() {
  std::printf("[test_hc_norm_inject_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_hc_norm_inject_gpu] skipped: no GPU\n");
    return 77;
  }
  for (uint32_t n : {1u, 3u, 8u, 9u, 37u, 2048u}) run(n, 4, 2560, true, true);  // Qwen3.8's mixers
  run(5, 4, 2560, false, true);   // the first mixer: no combine
  run(40, 4, 2560, true, false);  // the inject apart from its input
  run(17, 3, 208, true, true);    // a stream's 26 chunks: lanes cross the streams' edges (its f16 rows 16s)
  float* p = nullptr;
  CHECK(hipMalloc(&p, 1 << 20) == hipSuccess);
  CHECK(!q4x_hc_norm_inject(p, p, p, 4, 9, 64, 1, p, 1e-6f, p, nullptr, p, p, true, nullptr));       // 9 streams
  CHECK(!q4x_hc_norm_inject(p, p, p, 4, 4, 60, 1, p, 1e-6f, p, nullptr, p, p, true, nullptr));       // dim not 8s
  CHECK(!q4x_hc_norm_inject(p, p, p, 4, 4, 16384, 1, p, 1e-6f, p, nullptr, p, p, true, nullptr));    // past the LDS
  CHECK(!q4x_hc_norm_inject(p, nullptr, p, 4, 4, 64, 1, p, 1e-6f, p, nullptr, p, p, true, nullptr));  // combine, no out
  CHECK(hipFree(p) == hipSuccess);
  std::printf("  refusals: 9 streams, dim not 8s, a stream past the LDS, a combine without its output\n");
  std::printf("[test_hc_norm_inject_gpu] ok\n");
  return 0;
}
