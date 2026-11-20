// a prompt's attention (q4x_attn, its matrix cores: native mma on CUDA, WMMA on gfx1151) against the decode's F32
// attention (q35_attn_decode) a token at a time over the same Q8_0 cache: Qwen3.8-27B's shape (24 query heads on 4
// KV heads of 256), a step of tokens after a prefix of the cache's own; the gap within the prompt's f16 Q, K, V and P.
// skipped (77) without a GPU
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/q35_ops.h"
#include "osmos/gpu/q4x_ops.h"
}

#define CHECK(c)                                                                  \
  do {                                                                            \
    if (!(c)) {                                                                   \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

static uint32_t rng = 655;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / 16777216.0f - 0.5f;
}

int main() {
  std::printf("[test_q35_attn_gpu] start\n");
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || !n) {
    std::printf("[test_q35_attn_gpu] skipped: no GPU\n");
    return 77;
  }
  const uint32_t H = 24, KV = 4, D = 256, kv = KV * D, q = H * D;
  for (const uint32_t pos0 : {0u, 5000u, 8702u}) {  // (8,702: past 8,192 keys, a call across 8,704 = 17 x 512)
    const uint32_t T = 300, ctx = pos0 + T;
    std::vector<float> hq((size_t) T * q), hk((size_t) ctx * kv), hv((size_t) ctx * kv), hg((size_t) T * q);
    for (auto& x : hq) x = 4.0f * uniform();  // (scores of a few units: a softmax with a shape)
    for (auto& x : hk) x = 4.0f * uniform();
    for (auto& x : hv) x = uniform();
    for (auto& x : hg) x = uniform();
    const uint64_t row = q4x_kv_row_bytes(kv);
    float *dq, *dk, *dv, *dg, *out, *ref, *fa, *da;
    void *kc, *vc;
    CHECK(hipMalloc((void**) &dq, hq.size() * 4) == hipSuccess && hipMalloc((void**) &dk, hk.size() * 4) == hipSuccess);
    CHECK(hipMalloc((void**) &dv, hv.size() * 4) == hipSuccess && hipMalloc((void**) &dg, hg.size() * 4) == hipSuccess);
    CHECK(hipMalloc((void**) &out, hq.size() * 4) == hipSuccess && hipMalloc((void**) &ref, hq.size() * 4) == hipSuccess);
    CHECK(hipMalloc(&kc, ctx * row) == hipSuccess && hipMalloc(&vc, ctx * row) == hipSuccess);
    CHECK(hipMalloc((void**) &fa, q4x_attn_scratch(H, ctx) * 4) == hipSuccess);
    CHECK(hipMalloc((void**) &da, q35_attn_decode_scratch(H, Q35_ATTN_DECODE_MAX) * 4) == hipSuccess);
    CHECK(hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(hipMemcpy(dk, hk.data(), hk.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(hipMemcpy(dv, hv.data(), hv.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(hipMemcpy(dg, hg.data(), hg.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(q4x_kv_put(dk, dv, ctx, kv, kc, vc, nullptr));
    CHECK(hipDeviceSynchronize() == hipSuccess);
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(q4x_attn(dq, kc, vc, T, H, KV, D, pos0, dg, nullptr, nullptr, fa, out, nullptr, nullptr));
    CHECK(hipDeviceSynchronize() == hipSuccess);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (uint32_t t = 0; t < T; t++)
      CHECK(q35_attn_decode(dq + (size_t) t * q, kc, vc, 1, H, KV, D, nullptr, pos0 + t, dg + (size_t) t * q, da,
                            ref + (size_t) t * q, nullptr));
    std::vector<float> a(hq.size()), r(hq.size());
    CHECK(hipMemcpy(a.data(), out, a.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(hipMemcpy(r.data(), ref, r.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    double num = 0, den = 0, worst = 0;
    for (size_t i = 0; i < a.size(); i++) {
      num += ((double) a[i] - r[i]) * ((double) a[i] - r[i]), den += (double) r[i] * r[i];
      worst = std::fmax(worst, std::fabs((double) a[i] - r[i]));
    }
    std::printf("  %u tokens after %u: the prompt's attention against the decode's F32 - relative error %.2e, "
                "largest gap %.2e; %.2f ms\n", T, pos0, std::sqrt(num / den), worst, ms);
    CHECK(std::sqrt(num / den) < 5e-3);
    // the decode a step of Q35_ATTN_DECODE_MAX tokens at a time: each token's bits its one-token call's (0.11.2 - past
    // 8,192 keys a step's first tokens took its last's split)
    for (uint32_t t = 0; t < T; t += Q35_ATTN_DECODE_MAX) {
      const uint32_t m = T - t < Q35_ATTN_DECODE_MAX ? T - t : Q35_ATTN_DECODE_MAX;
      CHECK(q35_attn_decode(dq + (size_t) t * q, kc, vc, m, H, KV, D, nullptr, pos0 + t, dg + (size_t) t * q, da,
                            out + (size_t) t * q, nullptr));
    }
    CHECK(hipMemcpy(a.data(), out, a.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    size_t apart = 0;
    for (size_t i = 0; i < a.size(); i++) apart += std::memcmp(&a[i], &r[i], 4) != 0;
    std::printf("  the decode %u tokens a call: %zu of %zu values apart from a token a call\n", Q35_ATTN_DECODE_MAX,
                apart, a.size());
    CHECK(apart == 0);
    hipFree(dq), hipFree(dk), hipFree(dv), hipFree(dg), hipFree(out), hipFree(ref), hipFree(kc), hipFree(vc),
        hipFree(fa), hipFree(da);
  }
  std::printf("[test_q35_attn_gpu] ok\n");
  return 0;
}
