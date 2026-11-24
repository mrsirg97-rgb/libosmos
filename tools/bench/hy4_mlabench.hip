// hy4_mlabench - Hy4-flash's MLA attention (hy4_mla_attn) at a prompt step's shapes: 2,048 tokens, the Q8_0 cache;
// a sliding layer (64 heads, a window of 512) and a sparse one (32 heads, 512 groups of 4 a token, at a context of
// 32K); random rows. the median of 5 after a warm-up, each on the matrix cores and the F32 path (a decode's: read false)
//   hy4_mlabench
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" {
#include "osmos/gpu/hy4_ops.h"
}

#define CHECK(x)                                                           \
  do {                                                                     \
    if (!(x)) {                                                            \
      std::fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

int main() {
  const uint32_t lat = 512, rd = 64, row = lat + rd, T = 2048, ctx = 32768, pool = 4, top = 512;
  const uint64_t rb = hy4_mla_row_bytes(lat, rd, true);
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint8_t> cache((size_t) ctx * rb);
  for (uint32_t r = 0; r < ctx; r++) {
    uint8_t* p = cache.data() + (size_t) r * rb;
    for (uint32_t b = 0; b < lat / 32; b++) {
      const _Float16 d = (_Float16) 0.02f;
      std::memcpy(p + b * 34, &d, 2);
      for (int i = 0; i < 32; i++) p[b * 34 + 2 + i] = (uint8_t) (int8_t) (nd(rng) * 30);
    }
    float* rp = (float*) (p + lat / 32 * 34);
    for (uint32_t i = 0; i < rd; i++) rp[i] = nd(rng);
  }
  void* dc;
  CHECK(hipMalloc(&dc, cache.size()) == hipSuccess);
  CHECK(hipMemcpy(dc, cache.data(), cache.size(), hipMemcpyHostToDevice) == hipSuccess);
  for (int sparse = 0; sparse < 2; sparse++) {
    const uint32_t H = sparse ? 32 : 64, window = sparse ? 0 : 512;
    std::vector<float> q((size_t) T * H * row), sinks(H, 0.5f);
    for (auto& v : q) v = nd(rng);
    std::vector<int32_t> pos(T), sel, nsel;
    for (uint32_t t = 0; t < T; t++) pos[t] = (int32_t) (ctx - T + t);
    if (sparse) {
      sel.resize((size_t) T * top), nsel.assign(T, (int32_t) top);
      for (uint32_t t = 0; t < T; t++) {
        const int32_t own = pos[t] / (int32_t) pool;
        for (uint32_t g = 0; g < top; g++) sel[(size_t) t * top + g] = own - (int32_t) (top - 1) + (int32_t) g;
      }
    }
    float *dq, *ds, *dout, *scr;
    int32_t *dp, *dsel = nullptr, *dn = nullptr;
    CHECK(hipMalloc(&dq, q.size() * 4) == hipSuccess && hipMalloc(&ds, H * 4) == hipSuccess &&
          hipMalloc(&dout, (size_t) T * H * lat * 4) == hipSuccess && hipMalloc(&dp, T * 4) == hipSuccess &&
          hipMalloc(&scr, hy4_mla_attn_scratch(H) * 4) == hipSuccess);
    CHECK(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(hipMemcpy(ds, sinks.data(), H * 4, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(hipMemcpy(dp, pos.data(), T * 4, hipMemcpyHostToDevice) == hipSuccess);
    if (sparse) {
      CHECK(hipMalloc(&dsel, sel.size() * 4) == hipSuccess && hipMalloc(&dn, T * 4) == hipSuccess);
      CHECK(hipMemcpy(dsel, sel.data(), sel.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
      CHECK(hipMemcpy(dn, nsel.data(), T * 4, hipMemcpyHostToDevice) == hipSuccess);
    }
    for (int exact = 0; exact < 2; exact++) {
      std::vector<double> ms;
      for (int r = 0; r < 6; r++) {
        CHECK(hipDeviceSynchronize() == hipSuccess);
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(hy4_mla_attn(dq, dc, 0, true, ds, H, lat, rd, window, dsel, dn, sparse ? top : 0, sparse ? pool : 0,
                           0.0625f, dp, T, dout, scr, !exact, nullptr));
        CHECK(hipDeviceSynchronize() == hipSuccess);
        if (r) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      }
      std::sort(ms.begin(), ms.end());
      const double keys = sparse ? top * pool : window, tf = 2.0 * T * H * keys * (row + lat) / (ms[2] * 1e9);
      std::printf("%s (%u heads, %4.0f keys), %u tokens, %s: %8.2f ms (%5.1f TFLOP/s)\n", sparse ? "sparse " : "sliding",
                  H, keys, T, exact ? "F32 path    " : "matrix cores", ms[2], tf);
    }
    CHECK(hipFree(dq) == hipSuccess && hipFree(ds) == hipSuccess && hipFree(dout) == hipSuccess &&
          hipFree(dp) == hipSuccess && hipFree(scr) == hipSuccess);
    if (sparse) CHECK(hipFree(dsel) == hipSuccess && hipFree(dn) == hipSuccess);
  }
  return 0;
}
