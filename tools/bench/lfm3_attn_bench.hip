// q4x_attn at LFM3's shape (16 q heads, 4 KV heads of 128, a gate a head, every key): a prompt step of T tokens at the
// end of a C-key cache, timed - lfm3_attn_bench [T=2048] C... (its effective TFLOPS: 4 x 128 flops a (head, key) pair,
// the causal pairs alone)
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
#include "osmos/gpu/attn_shape.h"
#include "osmos/gpu/q4x_ops.h"
}

#define CK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)

int main(int argc, char** argv) {
  const uint32_t H = 16, KV = 4, D = 128, T = argc > 1 ? (uint32_t) atoi(argv[1]) : 2048;
  for (int a = 2; a < argc; a++) {
    const uint32_t C = (uint32_t) atoi(argv[a]), pos0 = C - T;
    const uint64_t rb = q4x_kv_row_bytes(KV * D);
    void *kc, *vc;
    float *q, *g, *out, *fa, *tmp;
    CK(hipMalloc(&kc, C * rb) == hipSuccess && hipMalloc(&vc, C * rb) == hipSuccess);
    const uint64_t put = 65536;
    CK(hipMalloc((void**) &tmp, put * KV * D * 4) == hipSuccess);
    std::vector<float> h(put * KV * D);
    for (size_t i = 0; i < h.size(); i++) h[i] = (float) ((i * 2654435761u) % 1000) / 500.0f - 1.0f;
    CK(hipMemcpy(tmp, h.data(), h.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
    for (uint64_t p = 0; p < C; p += put) {
      const uint32_t n = (uint32_t) (C - p < put ? C - p : put);
      CK(q4x_kv_put(tmp, tmp, n, KV * D, (uint8_t*) kc + p * rb, (uint8_t*) vc + p * rb, nullptr));
    }
    CK(hipMalloc((void**) &q, (size_t) T * H * D * 4) == hipSuccess && hipMalloc((void**) &g, (size_t) T * H * 4) == hipSuccess);
    CK(hipMalloc((void**) &out, (size_t) T * H * D * 4) == hipSuccess);
    CK(hipMalloc((void**) &fa, q4x_attn_scratch(H, C) * 4) == hipSuccess);
    CK(hipMemcpy(q, h.data(), (size_t) T * H * D * 4 < h.size() * 4 ? (size_t) T * H * D * 4 : h.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
    CK(hipMemset(g, 0, (size_t) T * H * 4) == hipSuccess);
    const AttnShape sh = {D, 0, 0};
    CK(q4x_attn(q, kc, vc, T, H, KV, D, pos0, g, nullptr, &sh, fa, out, nullptr, nullptr));
    CK(hipDeviceSynchronize() == hipSuccess);
    const int reps = C > 200000 ? 1 : 3;
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) CK(q4x_attn(q, kc, vc, T, H, KV, D, pos0, g, nullptr, &sh, fa, out, nullptr, nullptr));
    CK(hipDeviceSynchronize() == hipSuccess);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
    const double pairs = (double) T * pos0 + (double) T * (T + 1) / 2, fl = pairs * H * D * 4;
    std::printf("  C %8u, T %u: %8.2f ms, %.2f TFLOPS, %.1f ns a key a token (one layer)\n", C, T, s * 1e3, fl / s / 1e12,
                s * 1e9 / pairs * T / T);
    for (void* x : {kc, vc, (void*) q, (void*) g, (void*) out, (void*) fa, (void*) tmp}) CK(hipFree(x) == hipSuccess);
  }
  return 0;
}
