// hy4_ixbench - Hy4-flash's indexer selection (hy4_ix_select) at a decode's shapes: one token (and a verify's 6)
// against a context of 32K .. 1M tokens' groups (pool 4, 16 heads of 128, the top 512); random groups, queries and
// weights. each the median of 5 after a warm-up, a layer's (the forward runs 12 a token)
//   hy4_ixbench
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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
  const uint32_t heads = 16, dim = 128, pool = 4, top = 512, rd = 64;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (uint32_t ctx : {32768u, 131072u, 524288u, 1048576u}) {
    const uint32_t groups = ctx / pool, max_groups = ctx / pool + 1;
    for (uint32_t n_tok : {1u, 6u}) {
      std::vector<float> g((size_t) groups * dim), q((size_t) n_tok * heads * dim), w((size_t) n_tok * heads);
      for (auto& v : g) v = nd(rng);
      for (auto& v : q) v = nd(rng);
      for (auto& v : w) v = nd(rng);
      std::vector<int32_t> pos(n_tok);
      for (uint32_t t = 0; t < n_tok; t++) pos[t] = (int32_t) (ctx - n_tok + t);
      float *dg, *dq, *dw, *sc;
      int32_t *dp, *sel, *ns;
      CHECK(hipMalloc(&dg, g.size() * 4) == hipSuccess && hipMalloc(&dq, q.size() * 4) == hipSuccess &&
            hipMalloc(&dw, w.size() * 4) == hipSuccess && hipMalloc(&dp, n_tok * 4) == hipSuccess &&
            hipMalloc(&sc, (size_t) n_tok * max_groups * 4) == hipSuccess &&
            hipMalloc(&sel, (size_t) n_tok * top * 4) == hipSuccess && hipMalloc(&ns, n_tok * 4) == hipSuccess);
      CHECK(hipMemcpy(dg, g.data(), g.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
      CHECK(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
      CHECK(hipMemcpy(dw, w.data(), w.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
      CHECK(hipMemcpy(dp, pos.data(), n_tok * 4, hipMemcpyHostToDevice) == hipSuccess);
      
      std::vector<double> ms;
      for (int r = 0; r < 6; r++) {
        CHECK(hipDeviceSynchronize() == hipSuccess);
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(hy4_ix_select(dq, dw, dg, heads, dim, pool, top, dp, n_tok, max_groups, sc, sel, ns, false, nullptr));
        CHECK(hipDeviceSynchronize() == hipSuccess);
        if (r) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      }
      std::sort(ms.begin(), ms.end());
      std::printf("context %7u (%6u groups), %u token%s: %7.3f ms a layer (x 12: %7.1f ms a step)\n", ctx, groups, n_tok,
                  n_tok > 1 ? "s" : " ", ms[2], 12 * ms[2]);
      CHECK(hipFree(dg) == hipSuccess && hipFree(dq) == hipSuccess && hipFree(dw) == hipSuccess &&
            hipFree(dp) == hipSuccess && hipFree(sc) == hipSuccess && hipFree(sel) == hipSuccess &&
            hipFree(ns) == hipSuccess);
    }
  }
  return 0;
}
