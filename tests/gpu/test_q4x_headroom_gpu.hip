// a Qwen forward or sequence that would leave less than FIT_HEADROOM of the GPU's memory is refused with why (the
// runtime loads a kernel's code at its first launch: with no memory left it segfaulted), never built to crash later.
// the memory taken here to put each at its edge: what a forward (then a sequence) takes measured, then all but that
// and half a GiB held - the build refused, its own memory given back - and with the memory returned, both built.
// needs a gfx1151 and ~/Models/qwen38/UD-IQ3_XXS; skipped (77) without them.
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"
#include "osmos/run/fit.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kCtx = 4096, kMaxTokens = 512;
constexpr uint64_t kGiB = 1ull << 30, kUnder = kGiB / 2;  // left past a build: under FIT_HEADROOM

static uint64_t gpu_free() {
  size_t f = 0, t = 0;
  CHECK(hipMemGetInfo(&f, &t) == hipSuccess);
  return f;
}

// the GPU's memory taken until `leave` is free (one allocation, then smaller ones to close in)
struct Hog {
  void* p[64] = {};
  int n = 0;
  explicit Hog(uint64_t leave) {
    for (uint64_t piece = 1ull << 34; piece >= (1ull << 20) && n < 64; piece >>= 1)
      while (n < 64 && gpu_free() > leave + piece && hipMalloc(&p[n], piece) == hipSuccess) n++;
  }
  ~Hog() {
    for (int i = 0; i < n; i++) CHECK(hipFree(p[i]) == hipSuccess);
  }
};

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_q4x_headroom_gpu] start\n");
  const std::string home = std::getenv("HOME") ? std::getenv("HOME") : "";
  const std::string model = home + "/Models/qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf";
  int ndev = 0;
  char gerr[GGUF_ERR] = "", werr[Q4X_ERR] = "", err[Q4X_FWD_ERR] = "";
  Gguf* g = access(model.c_str(), R_OK) == 0 ? gguf_create(model.c_str(), gerr) : nullptr;
  if (!g || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_q4x_headroom_gpu] skipped: no model or GPU\n");
    return 77;
  }
  Q4xWeights* w = q4x_weights_create(g, nullptr, werr);
  CHECK(w);
  // what a forward and a sequence take
  const uint64_t f0 = gpu_free();
  Q4xForward* f = q4x_forward_create(w, nullptr, kCtx, kMaxTokens, err);
  CHECK(f);
  const uint64_t f1 = gpu_free();
  Q4xSeq* s = q4x_seq_create(f, err);
  CHECK(s);
  const uint64_t f2 = gpu_free(), fwd = f0 - f1, seq = f1 - f2;
  std::printf("  a forward takes %.2f GiB, a sequence %.2f\n", (double) fwd / kGiB, (double) seq / kGiB);
  q4x_seq_destroy(s);
  // a sequence that would leave half a GiB: refused, its memory given back
  {
    Hog hog(seq + kUnder);
    const uint64_t before = gpu_free();
    CHECK(!q4x_seq_create(f, err) && std::strstr(err, "GPU memory") && std::strstr(err, "the sequence"));
    CHECK(gpu_free() + (1ull << 20) >= before);  // the refused sequence's memory freed (within the pool's MiB)
    std::printf("  a sequence leaving %.2f GiB: refused (%s)\n", (double) (before - seq) / kGiB, err);
  }
  s = q4x_seq_create(f, err);
  CHECK(s);
  q4x_seq_destroy(s);
  q4x_forward_destroy(f);
  // a forward that would leave half a GiB: refused, its memory given back
  {
    Hog hog(fwd + kUnder);
    const uint64_t before = gpu_free();
    CHECK(!q4x_forward_create(w, nullptr, kCtx, kMaxTokens, err) && std::strstr(err, "GPU memory") &&
          std::strstr(err, "the forward"));
    CHECK(gpu_free() + (1ull << 20) >= before);
    std::printf("  a forward leaving %.2f GiB: refused (%s)\n", (double) (before - fwd) / kGiB, err);
  }
  f = q4x_forward_create(w, nullptr, kCtx, kMaxTokens, err);
  CHECK(f && (s = q4x_seq_create(f, err)));
  std::printf("  the memory given back: a forward and a sequence built\n");
  q4x_seq_destroy(s);
  q4x_forward_destroy(f);
  q4x_weights_destroy(w);
  gguf_destroy(g);
  std::printf("[test_q4x_headroom_gpu] ok\n");
  return 0;
}
