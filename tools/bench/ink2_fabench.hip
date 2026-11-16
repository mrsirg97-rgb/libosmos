// ink2_fabench: the prompt attention's time - ink2_fa (WMMA) against ink2_attn (F32, a block a token and head) on
// random q, k, v at Inkling's shapes, a full layer (causal over all) and a sliding one (window 1,024), T tokens.
//   ink2_fabench [T ...]   (default 4096 16384)
//   DECODE=1: a decode's attention instead - one query over T keys stored f16 (ink2_fa16, the engine's), 20 calls
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
  if (std::getenv("DECODE")) {
    for (uint32_t T : Ts)
      for (int full = 1; full >= 0; full--) {
        const uint32_t keys = full ? T : 1039;  // a sliding layer's window: 1,023 + a tile's 15 + the query
        float *q = rnd((size_t) H * D, 2.0f), *k = rnd((size_t) keys * KV * D, 2.0f), *v = rnd((size_t) keys * KV * D, 2.0f);
        float *rel = rnd((size_t) H * R, 1.0f), *sink = rnd(H, 1.0f), *rp = rnd((size_t) R * E, 0.1f), *out;
        uint16_t *k16, *v16;
        CHECK(hipMalloc(&out, (size_t) H * D * 4) == hipSuccess && hipMalloc(&k16, (size_t) keys * KV * D * 2) == hipSuccess &&
              hipMalloc(&v16, (size_t) keys * KV * D * 2) == hipSuccess);
        CHECK(ink2_to_f16(k, (uint64_t) keys * KV * D, k16, nullptr) && ink2_to_f16(v, (uint64_t) keys * KV * D, v16, nullptr));
        const Ink2Attn at = {H, KV, D, R, E, full ? 0u : 1024u, 0.0883883f};
        float* scr;
        CHECK(hipMalloc(&scr, ink2_fa_scratch(H, keys)) == hipSuccess);
        if (full) {  // a full layer's cache as the engine keeps it: keys Q8, values Q5 records (ink2_fa85), split
          uint8_t *k8, *v5;
          CHECK(hipMalloc(&k8, (size_t) keys * KV * INK2_KV8_RECORD) == hipSuccess &&
                hipMalloc(&v5, (size_t) keys * KV * INK2_KV5_RECORD) == hipSuccess);
          CHECK(ink2_to_kv8(k, (uint64_t) keys * KV, k8, nullptr) && ink2_to_kv5(v, (uint64_t) keys * KV, v5, nullptr));
          CHECK(ink2_fa85(&at, q, rel, sink, nullptr, 1, k8, v5, keys, 0, rp, out, scr, nullptr));
          hipEvent_t a, b;
          hipEventCreate(&a), hipEventCreate(&b);
          hipEventRecord(a);
          for (int i = 0; i < 20; i++) CHECK(ink2_fa85(&at, q, rel, sink, nullptr, 1, k8, v5, keys, 0, rp, out, scr, nullptr));
          hipEventRecord(b), hipEventSynchronize(b);
          float ms;
          hipEventElapsedTime(&ms, a, b);
          const double gb = (double) keys * KV * (INK2_KV8_RECORD + INK2_KV5_RECORD) / 1e9;
          std::printf("decode %7u keys %-7s %-6s %8.3f ms a call   %6.1f GB/s of keys and values (Q8 / Q5)\n", keys,
                      "full", "q8q5", ms / 20, gb / (ms / 20 / 1e3));
          CHECK(hipFree(k8) == hipSuccess && hipFree(v5) == hipSuccess);
        }
        for (int split = 0; split < 2; split++) {  // one block a KV head, then a block a unit and KV head
          float* sp = split ? scr : nullptr;
          CHECK(ink2_fa16(&at, q, rel, sink, nullptr, 1, k16, v16, keys, 0, rp, out, sp, nullptr));
          hipEvent_t a, b;
          hipEventCreate(&a), hipEventCreate(&b);
          hipEventRecord(a);
          for (int i = 0; i < 20; i++) CHECK(ink2_fa16(&at, q, rel, sink, nullptr, 1, k16, v16, keys, 0, rp, out, sp, nullptr));
          hipEventRecord(b), hipEventSynchronize(b);
          float ms;
          hipEventElapsedTime(&ms, a, b);
          const double gb = 2.0 * keys * KV * D * 2 / 1e9;
          std::printf("decode %7u keys %-7s %-6s %8.3f ms a call   %6.1f GB/s of keys and values\n", keys,
                      full ? "full" : "sliding", split ? "split" : "whole", ms / 20, gb / (ms / 20 / 1e3));
        }
        CHECK(hipFree(scr) == hipSuccess);
        for (void* p : {(void*) q, (void*) k, (void*) v, (void*) rel, (void*) sink, (void*) rp, (void*) out, (void*) k16, (void*) v16})
          CHECK(hipFree(p) == hipSuccess);
        if (!full) break;
      }
    return 0;
  }
  if (std::getenv("PREFILL")) {  // a prompt's step at depth: 4,096 queries at the end of T keys, a full layer's cache
    const uint32_t n = 4096;     // (the engine's: keys Q8, values Q5 records - ink2_fa85; f16 beside it)
    for (uint32_t T : Ts) {
      float *q = rnd((size_t) n * H * D, 2.0f), *k = rnd((size_t) T * KV * D, 2.0f), *v = rnd((size_t) T * KV * D, 2.0f);
      float *rel = rnd((size_t) n * H * R, 1.0f), *sink = rnd((size_t) n * H, 1.0f), *rp = rnd((size_t) R * E, 0.1f), *out;
      uint8_t *k8, *v5;
      uint16_t *k16, *v16;
      CHECK(hipMalloc(&out, (size_t) n * H * D * 4) == hipSuccess &&
            hipMalloc(&k8, (size_t) T * KV * INK2_KV8_RECORD) == hipSuccess &&
            hipMalloc(&v5, (size_t) T * KV * INK2_KV5_RECORD) == hipSuccess &&
            hipMalloc(&k16, (size_t) T * KV * D * 2) == hipSuccess && hipMalloc(&v16, (size_t) T * KV * D * 2) == hipSuccess);
      CHECK(ink2_to_kv8(k, (uint64_t) T * KV, k8, nullptr) && ink2_to_kv5(v, (uint64_t) T * KV, v5, nullptr) &&
            ink2_to_f16(k, (uint64_t) T * KV * D, k16, nullptr) && ink2_to_f16(v, (uint64_t) T * KV * D, v16, nullptr));
      const Ink2Attn at = {H, KV, D, R, E, 0u, 0.0883883f};
      for (int f = 0; f < 2; f++) {
        auto call = [&] {
          return f ? ink2_fa16(&at, q, rel, sink, nullptr, n, k16, v16, T, 0, rp, out, nullptr, nullptr)
                   : ink2_fa85(&at, q, rel, sink, nullptr, n, k8, v5, T, 0, rp, out, nullptr, nullptr);
        };
        CHECK(call());
        hipEvent_t a, b;
        hipEventCreate(&a), hipEventCreate(&b);
        hipEventRecord(a);
        for (int i = 0; i < 5; i++) CHECK(call());
        hipEventRecord(b), hipEventSynchronize(b);
        float ms;
        hipEventElapsedTime(&ms, a, b);
        std::printf("prefill %7u keys %-5s %9.2f ms a 4,096-token step   %6.1f us a token\n", T, f ? "f16" : "q8q5",
                    ms / 5, ms / 5 * 1000 / n);
      }
      for (void* p : {(void*) q, (void*) k, (void*) v, (void*) rel, (void*) sink, (void*) rp, (void*) out, (void*) k8,
                      (void*) v5, (void*) k16, (void*) v16})
        CHECK(hipFree(p) == hipSuccess);
    }
    return 0;
  }
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
                : ink2_fa(&at, q, rel, sink, nullptr, T, k, v, T, 0, rp, out, nullptr, nullptr));
        hipEventRecord(b), hipEventSynchronize(b);
        hipEventElapsedTime(&ms[w], a, b);
      }
      std::printf("T %6u %-7s  ink2_fa %9.1f ms   ink2_attn %9.1f ms   x%.1f\n", T, full ? "full" : "sliding", ms[0],
                  ms[1], ms[1] > 0 ? ms[1] / ms[0] : 0.0f);
      for (float* p : {q, k, v, rel, sink, rp, out}) CHECK(hipFree(p) == hipSuccess);
    }
  return 0;
}
