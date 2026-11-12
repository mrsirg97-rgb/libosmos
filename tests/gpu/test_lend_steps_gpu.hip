// a long step in the expert cache's memory (0.6.9: its buffers lent - ForwardOptions' fixed_steps off) against the
// same step in buffers of its own (fixed_steps), and against a cache too small to lend them (the step a sub-batch at a
// time): a long step over sub-batches, a short one, a second long one, a short one - every step's kept logits the
// same bits each way. needs a gfx1151 and a model (test_model.h); skipped (77) without.
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

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

#include "test_model.h"

constexpr uint32_t kCtx = 16384, kMaxTokens = 8192;
constexpr uint32_t kSteps[] = {6000, 5, 3000, 5};  // long (three sub-batches), short, long (two), short
constexpr uint64_t kBudget = 24ull << 30;          // one forward at a time: a small pool, the rest streamed

struct Run {
  std::vector<std::vector<float>> rows;  // each step's kept rows' logits
  uint64_t bytes = 0, lent = 0, split = 0;
};

static Run run(Weights* w, bool fixed, uint64_t budget, const std::vector<int32_t>& ids) {
  const size_t V = w->geo.n_vocab;
  char err[FORWARD_ERR] = "";
  ForwardOptions o = {(int) w->geo.n_layers, kCtx, budget, nullptr, kMaxTokens, true, false, 0, 0, 0, fixed};
  Forward* f = forward_create_with(w, &o, err);
  if (!f) std::fprintf(stderr, "forward: %s\n", err);
  CHECK(f && forward_lane(f, 0, NULL, err));
  Seq* s = seq_create(f, err);
  CHECK(s);
  std::vector<Seq*> ss(kMaxTokens, s);
  Run r;
  uint32_t at = 0;
  for (uint32_t n : kSteps) {
    if (!forward_batch(f, ss.data(), ids.data() + at, n, err)) std::fprintf(stderr, "step: %s\n", err);
    CHECK(err[0] == 0);
    at += n;
    const uint32_t kept = forward_logit_rows(f);
    CHECK(kept > 0 && kept <= n);
    std::vector<float> x((size_t) kept * V);
    CHECK(hipDeviceSynchronize() == hipSuccess &&
          hipMemcpy(x.data(), forward_logits(f), x.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    r.rows.push_back(std::move(x));
  }
  forward_lends(f, &r.bytes, &r.lent, &r.split);
  seq_destroy(s);
  forward_destroy(f);
  return r;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_lend_steps_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_lend_steps_gpu] skipped: no model or GPU\n");
    return 77;
  }
  Gguf *g = nullptr, *eg = nullptr;
  Weights* w = test_weights(path, which, &g, &eg);
  CHECK(w);
  const uint32_t V = (uint32_t) w->geo.n_vocab;
  uint32_t total = 0;
  for (uint32_t n : kSteps) total += n;
  std::vector<int32_t> ids(total);
  uint32_t r = 1017;
  for (auto& x : ids) r = r * 1664525u + 1013904223u, x = (int32_t) ((r >> 8) % (V - 1000)) + 100;

  const Run own = run(w, true, kBudget, ids);
  CHECK(own.bytes == 0 && own.lent == 0 && own.split == 0);
  std::printf("  fixed steps: every buffer the forward's own, %zu steps\n", own.rows.size());

  const Run lent = run(w, false, kBudget, ids);
  CHECK(lent.bytes > 0 && lent.lent == 2 && lent.split == 0);
  for (size_t i = 0; i < own.rows.size(); i++) CHECK(lent.rows[i] == own.rows[i]);
  std::printf("  lent: the long steps' %.2f GiB from the expert cache, twice - every step's kept rows the same bits\n",
              lent.bytes / double(1ull << 30));

  // a cache smaller than the block: the dense weights' build and a little more (forward_budget at 0)
  char err[FORWARD_ERR] = "";
  ForwardOptions o0 = {(int) w->geo.n_layers, kCtx, 0, nullptr, kMaxTokens, true, false, 0, 0, 0, false};
  Forward* f0 = forward_create_with(w, &o0, err);
  CHECK(f0);
  const uint64_t dense = forward_budget(f0);
  forward_destroy(f0);
  const Run split = run(w, false, dense + (lent.bytes / 2), ids);
  CHECK(split.lent == 0 && split.split == 2);
  for (size_t i = 0; i < own.rows.size(); i++) CHECK(split.rows[i] == own.rows[i]);
  std::printf("  a cache too small to lend it: the long steps a sub-batch at a time - the same bits\n");
  std::printf("[test_lend_steps_gpu] ok\n");
  return 0;
}
