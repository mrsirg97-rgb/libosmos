// qxbench: Qwen's QSA indexer (q4x_ix_pick: the scores and the pick) at a long context's shapes (0.7.3) - 4 heads x
// 128, blocks of 4, the top 512 blocks, a step's 256 tokens at positions near 64K, 256K, 512K; synthetic queries and
// pooled keys. the median of 5 after a warm-up, a token's share in ms.
//   qxbench
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/q4x_ops.h"
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
  return (float) ((rng >> 8) * (1.0 / (1 << 24)) * 2.0 - 1.0);
}

int main() {
  constexpr uint32_t H = 4, D = Q4X_IX_DIM, R = 4, T = 512, MAXCTX = 524288;
  const uint32_t N = getenv("QX_N") ? (uint32_t) atoi(getenv("QX_N")) : 256;  // a step's tokens (1, 6: decode, a verify)
  const uint32_t nb_max = MAXCTX / R;
  std::vector<float> qh((size_t) N * H * D), ph((size_t) nb_max * D);
  for (float& v : qh) v = uniform();
  for (float& v : ph) v = uniform();
  float *q, *pooled, *scratch;
  int32_t *sel, *npick;
  const uint64_t sf0 = q4x_ix_scratch(MAXCTX, R), sf_max = (uint64_t) 256 * nb_max;
  uint64_t sf = sf0;
  CHECK(hipMalloc(&q, qh.size() * 4) == hipSuccess && hipMalloc(&pooled, ph.size() * 4) == hipSuccess);
  CHECK(hipMalloc(&scratch, sf_max * 4) == hipSuccess && hipMalloc(&sel, (size_t) N * T * 4) == hipSuccess);
  CHECK(hipMalloc(&npick, N * 4) == hipSuccess);
  CHECK(hipMemcpy(q, qh.data(), qh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(pooled, ph.data(), ph.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  std::printf("qxbench: %u tokens a step, %u heads x %u, blocks of %u, top %u; scratch %llu floats\n", N, H, D, R, T,
              (unsigned long long) sf);
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a) == hipSuccess && hipEventCreate(&b) == hipSuccess);
  for (uint32_t P : {65536u, 262144u, 524288u - 256u}) {
    const int32_t pos0 = (int32_t) (P - N);
    const auto run = [&] { CHECK(q4x_ix_pick(q, pooled, N, H, pos0, R, T, scratch, sf, sel, npick, nullptr)); };
    const auto timed = [&] {
      run();
      CHECK(hipDeviceSynchronize() == hipSuccess);
      std::vector<double> t;
      for (int i = 0; i < 5; i++) {
        CHECK(hipEventRecord(a) == hipSuccess);
        run();
        CHECK(hipEventRecord(b) == hipSuccess && hipEventSynchronize(b) == hipSuccess);
        float ms = 0;
        CHECK(hipEventElapsedTime(&ms, a, b) == hipSuccess);
        t.push_back(ms);
      }
      std::sort(t.begin(), t.end());
      return t[2];
    };
    std::vector<int32_t> s0((size_t) N * T), s1(s0.size());
    q4x_ix_score_naive(true);
    const double naive = timed();
    CHECK(hipMemcpy(s0.data(), sel, s0.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    q4x_ix_score_naive(false);
    const double blocked = timed();
    CHECK(hipMemcpy(s1.data(), sel, s1.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    // the scores bit for bit: 16 tokens, one piece (their scores whole in the scratch)
    const uint32_t NS = N < 16 ? N : 16;  // (no more tokens than the q buffer holds)
    const int32_t ps = (int32_t) (P - NS);
    const uint32_t nv = P / R;
    std::vector<float> c0((size_t) NS * nv), c1(c0.size());
    q4x_ix_score_naive(true);
    CHECK(q4x_ix_pick(q, pooled, NS, H, ps, R, T, scratch, sf, sel, npick, nullptr));
    CHECK(hipMemcpy(c0.data(), scratch, c0.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    q4x_ix_score_naive(false);
    CHECK(q4x_ix_pick(q, pooled, NS, H, ps, R, T, scratch, sf, sel, npick, nullptr));
    CHECK(hipMemcpy(c1.data(), scratch, c1.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    uint64_t diff = 0, nsc = 0;
    for (uint32_t t = 0; t < NS; t++) {
      const uint32_t vis = (uint32_t) (ps + (int32_t) t + 1) / R;
      for (uint32_t j = 0; j < vis; j++, nsc++) diff += std::memcmp(&c0[(size_t) t * nv + j], &c1[(size_t) t * nv + j], 4) != 0;
    }
    uint64_t hash = 1469598103934665603ull;  // (0.7.4) the picks' FNV-1a: OSMOS_SELECT_ONE=1's run the same
    for (int32_t v : s1) hash = (hash ^ (uint32_t) v) * 1099511628211ull;
    std::printf("  at %6uK (%6u blocks): plain %8.2f ms, default %8.2f ms (%.1fx) a step of %u - %.3f ms a token a"
                " layer; the picks %s (%016llx), %llu scores %llu bits differ\n", P / 1024, P / R, naive, blocked,
                naive / blocked, N, blocked / N, s0 == s1 ? "the same" : "DIFFER", (unsigned long long) hash,
                (unsigned long long) nsc, (unsigned long long) diff);
  }
  // the scratch's size: a piece's tokens (the blocked kernel's tile 64 tokens) at 511K
  q4x_ix_score_naive(false);
  for (uint64_t toks : {32ull, 64ull, 128ull, 256ull}) {
    sf = toks * nb_max;
    const int32_t pos0 = (int32_t) (524288u - 256u - N);
    CHECK(q4x_ix_pick(q, pooled, N, H, pos0, R, T, scratch, sf, sel, npick, nullptr));
    CHECK(hipDeviceSynchronize() == hipSuccess);
    std::vector<double> t;
    for (int i = 0; i < 5; i++) {
      CHECK(hipEventRecord(a) == hipSuccess);
      CHECK(q4x_ix_pick(q, pooled, N, H, pos0, R, T, scratch, sf, sel, npick, nullptr));
      CHECK(hipEventRecord(b) == hipSuccess && hipEventSynchronize(b) == hipSuccess);
      float ms = 0;
      CHECK(hipEventElapsedTime(&ms, a, b) == hipSuccess);
      t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    std::printf("  511K, scratch for %3llu tokens (%5.0f MB): %7.2f ms a step of %u\n", (unsigned long long) toks,
                sf * 4 / 1e6, t[2], N);
  }
  return 0;
}
