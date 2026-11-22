// seq_rollback back to back with no step between (0.9.2): each within seq_reach, then the tokens stepped again - every
// row the first pass's bytes (before the fix, 16 then 8 more, or 6 then 16, left the window's ring and the Engram
// n-gram's history short: every row after differed). seq_reach after rollbacks SEQ_MAX_BACK together at most. needs a gfx1151 and a model (test_model.h); skipped (77) without them.
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>


extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/model/weights.h"
#include "osmos/run/drafter.h"
#include "osmos/run/forward.h"
}
#include "test_model.h"
#include "test_fit.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kMaxTokens = 512, kMaxCtx = 4096, kPromptA = 301;
constexpr int kNew = 40;
constexpr uint64_t kBudget = 100ull << 30;

static Forward* g_f;
static uint32_t g_vocab;
static char g_err[FORWARD_ERR];

static std::vector<int32_t> ids(uint32_t n, uint32_t seed) {
  std::vector<int32_t> v(n);
  uint32_t r = seed;
  for (auto& x : v) r = r * 1664525u + 1013904223u, x = (int32_t) ((r >> 8) % (g_vocab - 1000)) + 100;
  return v;
}

// one greedy step of token t at s's position: its logits row
static std::vector<float> step(Seq* s, int32_t t, int32_t* next) {
  std::vector<float> lg(g_vocab);
  CHECK(forward_step(g_f, s, t, g_err));
  CHECK(forward_greedy(g_f, 0, 1, next, g_err) && hipDeviceSynchronize() == hipSuccess);
  CHECK(hipMemcpy(lg.data(), forward_logits(g_f), g_vocab * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return lg;
}
int main() {
  std::printf("[test_rollback_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_rollback_gpu] skipped: no ds1017 or UD-IQ1_M, or no GPU\n");
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
  g_vocab = (uint32_t) w->geo.n_vocab;
  std::vector<uint64_t> picks((size_t) w->geo.n_layers * w->geo.n_experts);  // the bench's profile, where it is
  FILE* pf = std::fopen((dir + "osmos-picks.bin").c_str(), "rb");
  const bool ranked = pf && std::fread(picks.data(), 8, picks.size(), pf) == picks.size();
  if (pf) std::fclose(pf);
  ForwardOptions o = {(int) w->geo.n_layers, kMaxCtx, kBudget, ranked ? picks.data() : nullptr, kMaxTokens};
  TestFit fit = {[&](uint64_t b) {
                   o.budget = b;
                   return (g_f = forward_create_with(w, &o, g_err)) && forward_lane(g_f, 0, NULL, g_err);
                 },
                 [&] { return forward_budget(g_f); }, [&](uint64_t* r) { return forward_room(g_f, r); },
                 [&] { forward_destroy(g_f), g_f = nullptr; }};
  uint64_t budget = 0;  // (test_fit: the test's own, or this machine's fill where it is smaller)
  CHECK(test_fit(fit, kBudget, &budget));

  const std::vector<int32_t> pa = ids(kPromptA, 1017);
  // seq_reach after a rollback: every rollback it allows, back to back with no step between, then the same tokens
  // stepped again - each row the first pass's bytes (a sequence's state its tokens', however it got there)
  const uint32_t firsts[] = {16, 6, 1};
  int bad = 0;
  for (uint32_t first : firsts) {
    Seq* a = seq_create(g_f, g_err);
    CHECK(a);
    CHECK(seq_prompt(a, kPromptA));
    std::vector<Seq*> ss(kPromptA, a);
    CHECK(forward_batch(g_f, ss.data(), pa.data(), kPromptA, g_err));
    int32_t t;
    CHECK(forward_greedy(g_f, kPromptA - 1, 1, &t, g_err) && hipDeviceSynchronize() == hipSuccess);
    std::vector<int32_t> toks;
    std::vector<std::vector<float>> rows;
    for (int i = 0; i < kNew; i++) {
      int32_t n;
      toks.push_back(t);
      rows.push_back(step(a, t, &n));
      t = n;
    }
    CHECK(seq_rollback(a, first));
    const uint32_t more = seq_reach(a);
    uint32_t back = first;
    if (more) {
      CHECK(seq_rollback(a, more));
      back += more;
    }
    int diff = 0;
    for (uint32_t i = kNew - back; i < (uint32_t) kNew; i++) {
      int32_t n;
      const std::vector<float> lg = step(a, toks[i], &n);
      diff += std::memcmp(lg.data(), rows[i].data(), g_vocab * 4) != 0;
    }
    CHECK(back <= SEQ_MAX_BACK);
    std::printf("  rollback %u, then seq_reach %u more (%u back, no step between): %d of %u re-stepped rows differ\n",
                first, more, back, diff, back);
    bad += diff;
    seq_destroy(a);
  }
  std::printf("[test_rollback_gpu] %s\n", bad ? "FAIL" : "ok");
  return bad ? 1 : 0;
}
