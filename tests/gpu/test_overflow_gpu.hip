// a step whose activations pass f16's range (mm's inputs are f16: the overflow flag) is refused, and the flag does
// not outlive it: the next step - another sequence's, the same tokens - is accepted and picks what a forward that
// never overflowed picks, and a second overflow is still caught. the overflow made by a probe that stands a huge
// value in for a layer's attention input (the matmul then converts it: mm_x16's flag). the flag is cleared where it
// is found set, not by a launch every step (DESIGN.md, "0.6.2: DeepSeek's decode, traced again"): without that
// clearing every step after the first overflow is refused - this test's mutant. needs a gfx1151 and a model
// (test_model.h); skipped (77) without them.
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

constexpr uint32_t kMaxTokens = 64, kMaxCtx = 4096, kPrompt = 24;
constexpr uint64_t kBudget = 40ull << 30;  // test_membrane_gpu's: the dense weights and a third of the experts
constexpr int kLayer = 2;                  // the layer whose attention input the probe blows up

static void blow_up(const char* step, int layer, int, void* dev, uint64_t n, bool ints, void*) {
  if (ints || layer != kLayer || std::strcmp(step, "attn_x")) return;
  std::vector<float> big(n, 1e30f);
  CHECK(hipMemcpy(dev, big.data(), n * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
}

// the prompt through a new sequence: true and its last token's pick, or false with why
static bool run(Forward* f, const std::vector<int32_t>& p, int32_t* t, char* err) {
  Seq* s = seq_create(f, err);
  CHECK(s && seq_prompt(s, (uint32_t) p.size()));
  std::vector<Seq*> ss(p.size(), s);
  const bool ok = forward_batch(f, ss.data(), p.data(), (uint32_t) p.size(), err) &&
                  forward_greedy(f, (uint32_t) p.size() - 1, 1, t, err);
  seq_destroy(s);
  return ok;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_overflow_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_overflow_gpu] skipped: no model or no GPU\n");
    return 77;
  }
  Gguf *g = nullptr, *eg = nullptr;
  Weights* w = test_weights(path, which, &g, &eg);
  CHECK(w);
  char err[FORWARD_ERR] = "";
  ForwardOptions o = {(int) w->geo.n_layers, kMaxCtx, kBudget, nullptr, kMaxTokens};
  o.no_guess = true;
  Forward* f = forward_create_with(w, &o, err);
  if (!f) std::fprintf(stderr, "  forward_create_with: %s\n", err);
  CHECK(f && forward_lane(f, 0, NULL, err));
  std::vector<int32_t> p(kPrompt);
  uint32_t seed = 1017;
  for (auto& x : p) seed = seed * 1664525u + 1013904223u, x = (int32_t) ((seed >> 8) % ((uint32_t) w->geo.n_vocab - 1000)) + 100;

  int32_t clean = -1, t = -1;
  CHECK(run(f, p, &clean, err));
  for (int round = 0; round < 2; round++) {  // twice: the flag set, cleared, and set again
    forward_probe(f, blow_up, nullptr);
    err[0] = '\0';
    CHECK(!run(f, p, &t, err) && std::strstr(err, "f16"));
    std::printf("  an overflowing step: refused (%s)\n", err);
    forward_probe(f, nullptr, nullptr);
    err[0] = '\0';
    const bool ok = run(f, p, &t, err);
    if (!ok) std::fprintf(stderr, "  the step after it: %s\n", err);
    CHECK(ok && t == clean);
    std::printf("  the step after it: accepted, token %d (the clean forward's %d)\n", t, clean);
  }
  // a decode-sized step after an overflow too (the lane's path)
  forward_probe(f, blow_up, nullptr);
  CHECK(!run(f, p, &t, err));
  forward_probe(f, nullptr, nullptr);
  Seq* s = seq_create(f, err);
  CHECK(s);
  for (uint32_t i = 0; i < 4; i++) CHECK(forward_step(f, s, p[i], err));
  seq_destroy(s);
  std::printf("  four single-token steps after an overflow: accepted\n");

  forward_destroy(f);
  weights_destroy(w);
  gguf_destroy(eg), gguf_destroy(g);
  std::printf("[test_overflow_gpu] ok\n");
  return 0;
}
