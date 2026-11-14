// ink2_fabench: the prompt attention's time - ink2_fa (WMMA) against ink2_attn (F32, a block a token and head) on
// random q, k, v at Inkling's shapes, a full layer (causal over all) and a sliding one (window 1,024), T tokens.
//   ink2_fabench [T ...]   (default 4096 16384)
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
#include "osmos/gpu/ink2_ops.h"
}

#define CHECK(x)                                                         \
  do {                                                                   \
    if (!(x)) std::fprintf(stderr, "failed: %s\n", #x), std::exit(1);   \
  } while (0)

static float* rnd(size_t n, float s) {
  std::vector<float> h(n);
  for (auto& x : h) x = s * ((float) std::rand() / RAND_MAX - 0.5f);
  float* d;
  CHECK(hipMalloc(&d, n * 4) == hipSuccess && hipMemcpy(d, h.data(), n * 4, hipMemcpyHostToDevice) == hipSuccess);
  return d;
}

int main(int argc, char** argv) {
  std::vector<uint32_t> Ts;
  for (int i = 1; i < argc; i++) Ts.push_back((uint32_t) std::atoi(argv[i]));
  if (Ts.empty()) Ts = {4096, 16384};
  const uint32_t H = 48, KV = 8, D = 128, R = 16, E = 1024;
  for (uint32_t T : Ts)
    for (int full = 1; full >= 0; full--) {
      float *q = rnd((size_t) T * H * D, 2.0f), *k = rnd((size_t) T * KV * D, 2.0f), *v = rnd((size_t) T * KV * D, 2.0f);
      float *rel = rnd((size_t) T * H * R, 1.0f), *sink = rnd((size_t) T * H, 1.0f), *rp = rnd((size_t) R * E, 0.1f);
      float* out;
      CHECK(hipMalloc(&out, (size_t) T * H * D * 4) == hipSuccess);
      const Ink2Attn at = {H, KV, D, R, E, full ? 0u : 1024u, 0.0883883f};
      hipEvent_t a, b;
      hipEventCreate(&a), hipEventCreate(&b);
      float ms[2];
      for (int w = 0; w < 2; w++) {  // ink2_fa, then ink2_attn (skipped past 16K full: minutes)
        if (w == 1 && full && T > 16384) { ms[1] = -1; continue; }
        hipEventRecord(a);
        CHECK(w ? ink2_attn(&at, q, rel, sink, nullptr, T, k, v, T, rp, out, nullptr)
                : ink2_fa(&at, q, rel, sink, nullptr, T, k, v, T, rp, out, nullptr));
        hipEventRecord(b), hipEventSynchronize(b);
        hipEventElapsedTime(&ms[w], a, b);
      }
      std::printf("T %6u %-7s  ink2_fa %9.1f ms   ink2_attn %9.1f ms   x%.1f\n", T, full ? "full" : "sliding", ms[0],
                  ms[1], ms[1] > 0 ? ms[1] / ms[0] : 0.0f);
      for (float* p : {q, k, v, rel, sink, rp, out}) CHECK(hipFree(p) == hipSuccess);
    }
  return 0;
}
