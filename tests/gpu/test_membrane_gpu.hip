// the membrane (DESIGN.md, "0.5.8"): every expert a unit of one cache. two sequences of the same prompt decode the
// same kRun tokens, one after the other: the second waits on no read - the first's experts all still in memory (a
// pool of two layers' streamed experts keeps one token's misses, not kRun's). then a step each again with a long
// prompt (another sequence's, 2,048 tokens: most of every layer's experts) between them: the prompt passes through,
// the second still waits on none. every step of the second the first's logits and token, bit for bit. needs a
// gfx1151 and a model (test_model.h); skipped (77) without them.
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

constexpr uint32_t kMaxTokens = 2048, kMaxCtx = 4096, kPrompt = 200, kLong = 2048;
constexpr int kRun = 128; /* their misses (32 tokens measured: 712) under the experts' budget (~33 GB at kBudget), past the stream room (~2 layers) and the swaps: 32 left today one */
// experts streamed: the dense weights (~7 GB) and a third of a model's experts (UD-IQ1_M's 140 GB) - a step misses
constexpr uint64_t kBudget = 40ull << 30;

static uint64_t waits(const Forward* f) {
  uint64_t n[2], ns[2];
  forward_waits(f, n, ns);
  return n[0] + n[1];
}

static std::vector<int32_t> tokens(uint32_t n, uint32_t seed, uint32_t V) {
  std::vector<int32_t> p(n);
  for (auto& x : p) seed = seed * 1664525u + 1013904223u, x = (int32_t) ((seed >> 8) % (V - 1000)) + 100;
  return p;
}

static void prompt(Forward* f, Seq* s, const std::vector<int32_t>& p, char* err) {
  CHECK(seq_prompt(s, (uint32_t) p.size()));
  std::vector<Seq*> ss(p.size(), s);
  CHECK(forward_batch(f, ss.data(), p.data(), (uint32_t) p.size(), err));
}

// one step of s at token t: its logits copied out, the token it picks
static int32_t step(Forward* f, Seq* s, int32_t t, std::vector<float>& lg, char* err) {
  CHECK(forward_step(f, s, t, err));
  int32_t next;
  CHECK(forward_greedy(f, 0, 1, &next, err) && hipDeviceSynchronize() == hipSuccess);
  CHECK(hipMemcpy(lg.data(), forward_logits(f), lg.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return next;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0); /* the counts printed before a failed check exits */
  std::printf("[test_membrane_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_membrane_gpu] skipped: no model or no GPU\n");
    return 77;
  }
  Gguf *g = nullptr, *eg = nullptr;
  Weights* w = test_weights(path, which, &g, &eg);
  CHECK(w);
  char err[FORWARD_ERR] = "";
  ForwardOptions o = {(int) w->geo.n_layers, kMaxCtx, kBudget, nullptr, kMaxTokens};
  o.no_guess = true; /* every read a pin's: the waits count every miss */
  Forward* f = forward_create_with(w, &o, err);
  if (!f) std::fprintf(stderr, "  forward_create_with: %s\n", err);
  CHECK(f && forward_lane(f, 0, NULL, err));
  const uint32_t V = (uint32_t) w->geo.n_vocab;
  const std::vector<int32_t> p = tokens(kPrompt, 1017, V);
  Seq *a = seq_create(f, err), *b = seq_create(f, err), *c = seq_create(f, err);
  CHECK(a && b && c);
  prompt(f, a, p, err), prompt(f, b, p, err);
  int32_t t;
  CHECK(forward_greedy(f, kPrompt - 1, 1, &t, err));
  std::vector<float> la(V), lb(V);

  /* the same kRun steps twice in a row */
  std::vector<int32_t> run(kRun + 1, t);
  std::vector<std::vector<float>> logits(kRun, std::vector<float>(V));
  const uint64_t w0 = waits(f);
  for (int i = 0; i < kRun; i++)
    run[i + 1] = step(f, a, run[i], logits[i], err);
  const uint64_t w1 = waits(f);
  for (int i = 0; i < kRun; i++)
    CHECK(step(f, b, run[i], lb, err) == run[i + 1] && std::memcmp(lb.data(), logits[i].data(), lb.size() * 4) == 0);
  const uint64_t w2 = waits(f);
  std::printf("  the same %d steps twice: the first waited on %llu reads, the second on %llu\n", kRun,
              (unsigned long long) (w1 - w0), (unsigned long long) (w2 - w1));
  CHECK(w1 > w0); /* the budget leaves experts to stream: the first misses */
  CHECK(w2 == w1);
  const int32_t ta = run[kRun], tb = run[kRun];

  /* again, a long prompt between the two */
  const int32_t ta2 = step(f, a, ta, la, err);
  const uint64_t w3 = waits(f);
  prompt(f, c, tokens(kLong, 4242, V), err);
  const uint64_t w4 = waits(f);
  const int32_t tb2 = step(f, b, tb, lb, err);
  const uint64_t w5 = waits(f);
  std::printf("  a %u-token prompt between them (it waited on %llu): the second step waited on %llu\n", kLong,
              (unsigned long long) (w4 - w3), (unsigned long long) (w5 - w4));
  CHECK(ta2 == tb2 && std::memcmp(la.data(), lb.data(), la.size() * 4) == 0);
  CHECK(w5 == w4);

  seq_destroy(a), seq_destroy(b), seq_destroy(c);
  forward_destroy(f);
  weights_destroy(w);
  gguf_destroy(g), gguf_destroy(eg);
  std::printf("[test_membrane_gpu] ok\n");
  return 0;
}
