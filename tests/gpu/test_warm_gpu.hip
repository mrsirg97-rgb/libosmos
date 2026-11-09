// the warm start read once (DESIGN.md, "0.5.9"): the budget's experts are read at a forward's first step, not at its
// create - fit.h's sizing builds and tears down several forwards before the one it keeps, and a warm read at each
// create read the warm set once a build (UD-IQ2_XXS: ~600 GB at load for a 107 GiB budget). two creates of one
// budget (a sizing's), each reading only the dense weights; the kept one's first step reads the warm set; its second
// step does not again. bytes read from storage: the process's own (/proc/self/io), the experts' reads O_DIRECT (no
// page cache between). needs a gfx1151 and a model (test_model.h); skipped (77) without them.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/model/weights.h"
#include "osmos/run/forward.h"
}
#include "test_model.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kMaxTokens = 512, kMaxCtx = 4096;
// the dense weights (~7 GB) and ~33 GB of experts: a create's dense read under a quarter of it, the warm set over half.
// each step a single token's: its own misses a few GB (UD-IQ1_M: ~3), so the warm set alone passes half
constexpr uint64_t kBudget = 40ull << 30;

static uint64_t storage_read() {
  FILE* io = std::fopen("/proc/self/io", "r");
  CHECK(io);
  char key[64];
  unsigned long long v = 0, got = 0;
  while (std::fscanf(io, "%63s %llu", key, &v) == 2)
    if (!std::strcmp(key, "read_bytes:")) got = v;
  std::fclose(io);
  return got;
}

static Forward* create(Weights* w, char* err) {
  ForwardOptions o = {(int) w->geo.n_layers, kMaxCtx, kBudget, nullptr, kMaxTokens};
  Forward* f = forward_create_with(w, &o, err);
  if (!f) std::fprintf(stderr, "  forward_create_with: %s\n", err);
  CHECK(f && forward_lane(f, 0, NULL, err));
  return f;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_warm_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_warm_gpu] skipped: no model or no GPU\n");
    return 77;
  }
  Gguf *g = nullptr, *eg = nullptr;
  Weights* w = test_weights(path, which, &g, &eg);
  CHECK(w);
  char err[FORWARD_ERR] = "";
  const double GB = 1e9;

  uint64_t at = storage_read();
  forward_destroy(create(w, err)); /* a sizing's build, let go */
  const uint64_t sizing = storage_read() - at;
  at = storage_read();
  Forward* f = create(w, err);
  const uint64_t kept = storage_read() - at;
  std::printf("  creates read %.1f and %.1f GB (the budget %.1f)\n", sizing / GB, kept / GB, kBudget / GB);
  CHECK(sizing < kBudget / 4 && kept < kBudget / 4);

  Seq* s = seq_create(f, err);
  CHECK(s && seq_prompt(s, 1));
  int32_t t = 1017;
  at = storage_read();
  CHECK(forward_step(f, s, t, err));
  CHECK(forward_greedy(f, 0, 1, &t, err) && hipDeviceSynchronize() == hipSuccess);
  const uint64_t first = storage_read() - at;
  at = storage_read();
  CHECK(forward_step(f, s, t, err) && hipDeviceSynchronize() == hipSuccess);
  const uint64_t second = storage_read() - at;
  std::printf("  the first step read %.1f GB, the second %.1f\n", first / GB, second / GB);
  CHECK(first > kBudget / 2 && second < kBudget / 4);

  seq_destroy(s);
  forward_destroy(f);
  weights_destroy(w);
  gguf_destroy(g), gguf_destroy(eg);
  std::printf("[test_warm_gpu] ok\n");
  return 0;
}
