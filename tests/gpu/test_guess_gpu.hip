// the guess off (ForwardOptions' no_guess): a decode at a budget that leaves experts to stream, run twice through the
// same weights - the guess on, then off. on: experts guessed (the guess runs here, so off is a change). off: none
// guessed, no read begun for one. both: every step's logits row and token the same bytes. needs a gfx1151 and a
// model (test_model.h); skipped (77) without them.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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

constexpr uint32_t kMaxTokens = 512, kMaxCtx = 4096, kPrompt = 200;
constexpr int kNew = 32;
// experts streamed: the dense weights (~7 GB) and a fraction of a model's experts (UD-IQ1_S's 122 GB) - misses a step
constexpr uint64_t kBudget = 40ull << 30;

struct Run {
  uint64_t hash, streamed, guessed, read;
};

static uint64_t fnv(const void* p, size_t n, uint64_t h) {
  const uint8_t* b = (const uint8_t*) p;
  for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

static Run decode(const Weights* w, bool no_guess) {
  char err[FORWARD_ERR] = "";
  ForwardOptions o = {(int) w->geo.n_layers, kMaxCtx, kBudget, nullptr, kMaxTokens};
  o.no_guess = no_guess;
  Forward* f = forward_create_with(w, &o, err);
  if (!f) std::fprintf(stderr, "  forward_create_with: %s\n", err);
  CHECK(f && forward_lane(f, 0, NULL, err));
  const uint32_t V = (uint32_t) w->geo.n_vocab;
  std::vector<int32_t> p(kPrompt);
  uint32_t r = 1017;
  for (auto& x : p) r = r * 1664525u + 1013904223u, x = (int32_t) ((r >> 8) % (V - 1000)) + 100;
  Seq* s = seq_create(f, err);
  CHECK(s && seq_prompt(s, kPrompt));
  std::vector<Seq*> ss(kPrompt, s);
  CHECK(forward_batch(f, ss.data(), p.data(), kPrompt, err));
  int32_t t;
  CHECK(forward_greedy(f, kPrompt - 1, 1, &t, err));
  std::vector<float> lg(V);
  Run run = {1469598103934665603ull, 0, 0, 0};
  for (int i = 0; i < kNew; i++) {
    CHECK(forward_step(f, s, t, err));
    CHECK(forward_greedy(f, 0, 1, &t, err) && hipDeviceSynchronize() == hipSuccess);
    CHECK(hipMemcpy(lg.data(), forward_logits(f), (size_t) V * 4, hipMemcpyDeviceToHost) == hipSuccess);
    run.hash = fnv(&t, 4, fnv(lg.data(), lg.size() * 4, run.hash));
  }
  forward_guesses(f, &run.streamed, &run.guessed, &run.read);
  seq_destroy(s);
  forward_destroy(f);
  return run;
}

int main() {
  std::printf("[test_guess_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_guess_gpu] skipped: no model or no GPU\n");
    return 77;
  }
  const std::string dir = std::string(path).substr(0, std::string(path).rfind('/') + 1);
  char gerr[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "";
  Gguf* g = gguf_create(path, gerr);
  CHECK(g);
  Gguf* eg = which == TestModel::kGsq ? gguf_create((dir + "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf").c_str(), gerr)
                                      : nullptr;
  CHECK(which != TestModel::kGsq || eg);
  Weights* w = weights_create(g, eg, werr);
  CHECK(w);

  const Run on = decode(w, false), off = decode(w, true);
  std::printf("  guess on: %llu experts streamed, %llu guessed, %llu reads begun by the guess\n",
              (unsigned long long) on.streamed, (unsigned long long) on.guessed, (unsigned long long) on.read);
  std::printf("  guess off: %llu streamed, %llu guessed, %llu reads begun by the guess\n",
              (unsigned long long) off.streamed, (unsigned long long) off.guessed, (unsigned long long) off.read);
  CHECK(on.streamed > 0 && on.guessed > 0);    // the guess runs at this budget: off changes something
  CHECK(off.guessed == 0 && off.read == 0);    // off: nothing guessed, nothing read for a guess
  CHECK(on.hash == off.hash);                  // and no bit of the decode differs
  std::printf("  %d steps' logits and tokens the same bytes either way (%016llx)\n", kNew,
              (unsigned long long) on.hash);

  weights_destroy(w);
  gguf_destroy(eg);
  gguf_destroy(g);
  std::printf("[test_guess_gpu] ok\n");
  return 0;
}
