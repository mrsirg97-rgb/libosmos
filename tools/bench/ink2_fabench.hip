// ink2_fabench: the prompt attention's time - the attention core (WMMA) against ink2_attn (F32, a block a token and head) on
// random q, k, v at Inkling's shapes, a full layer (causal over all) and a sliding one (window 1,024), T tokens.
//   ink2_fabench [T ...]   (default 4096 16384)
//   HEADS=n: the query heads (default Small's 48; Mini's 24); DUMP=path: PREFILL's outputs written (path.T.kind)
//   DECODE=1: a decode's attention instead - one query over T keys stored f16 (the engine's), 20 calls
//   PREFILL=1: a prompt's step at depth instead (4,096 queries after T keys); DET=1 with it: the same call twice, the
//   same bytes
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "attn_core_call.h"

extern "C" {
#include "osmos/gpu/ink2_ops.h"
}

#define CHECK(x)                                                         \
  do {                                                                   \
    if (!(x)) std::fprintf(stderr, "failed: %s\n", #x), std::exit(1);   \
  } while (0)

static uint64_t g_rng = 1017;  // the inputs' own generator: std::rand is reseeded by the HIP runtime's start
static float* rnd(size_t n, float s) {
  std::vector<float> h(n);
  for (auto& x : h) g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull, x = s * ((float) (g_rng >> 40) / 16777216.0f - 0.5f);
  float* d;
  CHECK(hipMalloc(&d, n * 4) == hipSuccess && hipMemcpy(d, h.data(), n * 4, hipMemcpyHostToDevice) == hipSuccess);
  return d;
}

int main(int argc, char** argv) {
  std::vector<uint32_t> Ts;
  for (int i = 1; i < argc; i++) Ts.push_back((uint32_t) std::atoi(argv[i]));
  if (Ts.empty()) Ts = {4096, 16384};
  const uint32_t H = std::getenv("HEADS") ? (uint32_t) std::atoi(std::getenv("HEADS")) : 48, KV = 8, D = 128, R = 16,
                 E = 1024;  // HEADS: the query heads (Small's 48, Mini's 24: 6 or 3 a KV head)
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
        if (full) {  // a full layer's cache as the engine keeps it: keys Q8, values Q5 records, split
          uint8_t *k8, *v5;
          CHECK(hipMalloc(&k8, (size_t) keys * KV * ATTN_KV8_RECORD) == hipSuccess &&
                hipMalloc(&v5, (size_t) keys * KV * ATTN_KV5_RECORD) == hipSuccess);
          CHECK(ink2_to_kv8(k, (uint64_t) keys * KV, k8, nullptr) && ink2_to_kv5(v, (uint64_t) keys * KV, v5, nullptr));
          auto call = [&] { return attn_core_call(H, KV, D, R, E, 0, at.scaling, ATTN_ROWS_KV8, ATTN_ROWS_KV5, q, rel, sink,
                                                  nullptr, 1, k8, v5, keys, 0, rp, out); };
          CHECK(call());
          hipEvent_t a, b;
          hipEventCreate(&a), hipEventCreate(&b);
          hipEventRecord(a);
          for (int i = 0; i < 20; i++) CHECK(call());
          hipEventRecord(b), hipEventSynchronize(b);
          float ms;
          hipEventElapsedTime(&ms, a, b);
          const double gb = (double) keys * KV * (ATTN_KV8_RECORD + ATTN_KV5_RECORD) / 1e9;
          std::printf("decode %7u keys %-7s %-6s %8.3f ms a call   %6.1f GB/s of keys and values (Q8 / Q5)\n", keys,
                      "full", "q8q5", ms / 20, gb / (ms / 20 / 1e3));
          CHECK(hipFree(k8) == hipSuccess && hipFree(v5) == hipSuccess);
        }
        {  // a block a unit and KV head, then the fold (the core's decode)
          auto call = [&] { return attn_core_call(H, KV, D, R, E, at.window, at.scaling, ATTN_ROWS_F16, ATTN_ROWS_F16, q, rel,
                                                  sink, nullptr, 1, k16, v16, keys, 0, rp, out); };
          CHECK(call());
          hipEvent_t a, b;
          hipEventCreate(&a), hipEventCreate(&b);
          hipEventRecord(a);
          for (int i = 0; i < 20; i++) CHECK(call());
          hipEventRecord(b), hipEventSynchronize(b);
          float ms;
          hipEventElapsedTime(&ms, a, b);
          const double gb = 2.0 * keys * KV * D * 2 / 1e9;
          std::printf("decode %7u keys %-7s %-6s %8.3f ms a call   %6.1f GB/s of keys and values\n", keys,
                      full ? "full" : "sliding", "split", ms / 20, gb / (ms / 20 / 1e3));
        }
        for (void* p : {(void*) q, (void*) k, (void*) v, (void*) rel, (void*) sink, (void*) rp, (void*) out, (void*) k16, (void*) v16})
          CHECK(hipFree(p) == hipSuccess);
        if (!full) break;
      }
    return 0;
  }
  if (std::getenv("PREFILL")) {  // a prompt's step at depth: 4,096 queries at the end of T keys, a full layer's cache
    const uint32_t n = 4096;     // (the engine's: keys Q8, values Q5 records ; f16 beside it)
    for (uint32_t T : Ts) {
      float *q = rnd((size_t) n * H * D, 2.0f), *k = rnd((size_t) T * KV * D, 2.0f), *v = rnd((size_t) T * KV * D, 2.0f);
      float *rel = rnd((size_t) n * H * R, 1.0f), *sink = rnd((size_t) n * H, 1.0f), *rp = rnd((size_t) R * E, 0.1f), *out;
      uint8_t *k8, *v5;
      uint16_t *k16, *v16;
      CHECK(hipMalloc(&out, (size_t) n * H * D * 4) == hipSuccess &&
            hipMalloc(&k8, (size_t) T * KV * ATTN_KV8_RECORD) == hipSuccess &&
            hipMalloc(&v5, (size_t) T * KV * ATTN_KV5_RECORD) == hipSuccess &&
            hipMalloc(&k16, (size_t) T * KV * D * 2) == hipSuccess && hipMalloc(&v16, (size_t) T * KV * D * 2) == hipSuccess);
      CHECK(ink2_to_kv8(k, (uint64_t) T * KV, k8, nullptr) && ink2_to_kv5(v, (uint64_t) T * KV, v5, nullptr) &&
            ink2_to_f16(k, (uint64_t) T * KV * D, k16, nullptr) && ink2_to_f16(v, (uint64_t) T * KV * D, v16, nullptr));
      const Ink2Attn at = {H, KV, D, R, E, 0u, 0.0883883f};
      for (int f = 0; f < 2; f++) {
        auto call = [&] {
          return f ? attn_core_call(H, KV, D, R, E, 0, at.scaling, ATTN_ROWS_F16, ATTN_ROWS_F16, q, rel, sink, nullptr, n,
                                    k16, v16, T, 0, rp, out)
                   : attn_core_call(H, KV, D, R, E, 0, at.scaling, ATTN_ROWS_KV8, ATTN_ROWS_KV5, q, rel, sink, nullptr, n,
                                    k8, v5, T, 0, rp, out);
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
        if (std::getenv("DET")) {  // the same call twice in one process: the same bytes? and the inputs' checksum
          std::vector<float> h1((size_t) n * H * D), h2(h1.size()), qh((size_t) n * H * D);
          CHECK(call() && hipMemcpy(h1.data(), out, h1.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
          CHECK(call() && hipMemcpy(h2.data(), out, h2.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
          auto sum = [](const float* d, size_t m) {
            std::vector<float> x(m);
            CHECK(hipMemcpy(x.data(), d, m * 4, hipMemcpyDeviceToHost) == hipSuccess);
            double c = 0;
            for (float y : x) c += y;
            return c;
          };
          std::printf("  in-process repeat %s; checksums q %.9g k %.9g v %.9g rel %.9g sink %.9g rp %.9g\n",
                      std::memcmp(h1.data(), h2.data(), h1.size() * 4) ? "DIFFERENT" : "the same",
                      sum(q, qh.size()), sum(k, (size_t) T * KV * D), sum(v, (size_t) T * KV * D),
                      sum(rel, (size_t) n * H * R), sum(sink, (size_t) n * H), sum(rp, (size_t) R * E));
        }
        if (const char* dump = std::getenv("DUMP")) {  // the output's bytes (a shape's bits against another's: cmp)
          std::vector<float> h((size_t) n * H * D);
          CHECK(hipMemcpy(h.data(), out, h.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
          FILE* o = std::fopen((std::string(dump) + "." + std::to_string(T) + (f ? ".f16" : ".q8q5")).c_str(), "wb");
          CHECK(o && std::fwrite(h.data(), 4, h.size(), o) == h.size());
          std::fclose(o);
        }
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
      for (int w = 0; w < 2; w++) {  // the core, then ink2_attn (skipped past 16K full: minutes)
        if (w == 1 && full && T > 16384) { ms[1] = -1; continue; }
        hipEventRecord(a);
        CHECK(w ? ink2_attn(&at, q, rel, sink, nullptr, T, k, v, T, rp, out, nullptr)
                : attn_core_call(H, KV, D, R, E, at.window, at.scaling, ATTN_ROWS_F32, ATTN_ROWS_F32, q, rel, sink, nullptr, T,
                                 k, v, T, 0, rp, out));
        hipEventRecord(b), hipEventSynchronize(b);
        hipEventElapsedTime(&ms[w], a, b);
      }
      std::printf("T %6u %-7s  the core %9.1f ms   ink2_attn %9.1f ms   x%.1f\n", T, full ? "full" : "sliding", ms[0],
                  ms[1], ms[1] > 0 ? ms[1] / ms[0] : 0.0f);
      for (float* p : {q, k, v, rel, sink, rp, out}) CHECK(hipFree(p) == hipSuccess);
    }
  return 0;
}
