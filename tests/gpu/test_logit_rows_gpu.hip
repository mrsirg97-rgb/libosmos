// a forward that keeps a step's last rows' logits alone (ForwardOptions' logit_rows) against one that keeps every
// row: a step longer than them - its last rows the same bits at slots 0 .., forward_logit_rows theirs, a greedy pick
// of the last slot the same token, of a slot past them refused; a step within them - every row as before. needs a
// gfx1151 and a model (test_model.h); skipped (77) without.
#include <hip/hip_runtime.h>

#include <algorithm>
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

constexpr uint32_t kCtx = 1024, kMaxTokens = 64, kLong = 20, kShort = 5, kKeep = 8;
constexpr uint64_t kBudget = 24ull << 30;  // one forward at a time: a small pool, the rest streamed

struct Rows {
  std::vector<float> long_last, short_all;  // the long step's last kKeep rows, the short step's every row
  int32_t long_pick;
  uint32_t long_rows, short_rows;
};

static Rows run(Weights* w, uint32_t logit_rows, const std::vector<int32_t>& ids, bool refuse_past) {
  const size_t V = w->geo.n_vocab;
  char err[FORWARD_ERR] = "";
  ForwardOptions o = {(int) w->geo.n_layers, kCtx, kBudget, nullptr, kMaxTokens, true, false, 0, 0, logit_rows};
  Forward* f = forward_create_with(w, &o, err);
  if (!f) std::fprintf(stderr, "forward: %s\n", err);
  CHECK(f && forward_lane(f, 0, NULL, err) && forward_logit_rows(f) == 0);
  Seq* s = seq_create(f, err);
  CHECK(s);
  std::vector<Seq*> ss(kLong, s);
  Rows r;
  CHECK(forward_batch(f, ss.data(), ids.data(), kLong, err));
  r.long_rows = forward_logit_rows(f);
  CHECK(r.long_rows == kLong || r.long_rows == kKeep);
  r.long_last.resize(kKeep * V);
  CHECK(hipDeviceSynchronize() == hipSuccess &&
        hipMemcpy(r.long_last.data(), forward_logits(f) + (size_t) (r.long_rows - kKeep) * V, kKeep * V * 4,
                  hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(forward_greedy(f, r.long_rows - 1, 1, &r.long_pick, err));
  int32_t t;
  if (refuse_past)  // a slot past the kept ones, and a run of rows reaching one
    CHECK(!forward_greedy(f, kKeep, 1, &t, err) && !forward_greedy(f, 0, kKeep + 1, std::vector<int32_t>(kLong).data(), err));
  CHECK(forward_batch(f, ss.data(), ids.data() + kLong, kShort, err));
  r.short_rows = forward_logit_rows(f);
  r.short_all.resize(kShort * V);
  CHECK(hipDeviceSynchronize() == hipSuccess &&
        hipMemcpy(r.short_all.data(), forward_logits(f), kShort * V * 4, hipMemcpyDeviceToHost) == hipSuccess);
  seq_destroy(s);
  forward_destroy(f);
  return r;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_logit_rows_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_logit_rows_gpu] skipped: no model or GPU\n");
    return 77;
  }
  Gguf *g = nullptr, *eg = nullptr;
  Weights* w = test_weights(path, which, &g, &eg);
  CHECK(w);
  const uint32_t V = (uint32_t) w->geo.n_vocab;
  std::vector<int32_t> ids(kLong + kShort);
  uint32_t r = 1017;
  for (auto& x : ids) r = r * 1664525u + 1013904223u, x = (int32_t) ((r >> 8) % (V - 1000)) + 100;

  const Rows all = run(w, 0, ids, false);
  CHECK(all.long_rows == kLong && all.short_rows == kShort);
  const float* last = &all.long_last[(size_t) (kKeep - 1) * V];
  CHECK(all.long_pick == (int32_t) (std::max_element(last, last + V) - last));
  std::printf("  every row kept: a %u-token step %u rows, a %u-token step %u\n", kLong, all.long_rows, kShort,
              all.short_rows);

  const Rows kept = run(w, kKeep, ids, true);
  CHECK(kept.long_rows == kKeep && kept.short_rows == kShort);
  CHECK(kept.long_last == all.long_last && kept.long_pick == all.long_pick && kept.short_all == all.short_all);
  std::printf("  the last %u kept: the %u-token step's last %u rows the same bits at slots 0 .. %u, slot %u's pick "
              "token %d both, a slot past them refused; the %u-token step's every row the same\n",
              kKeep, kLong, kKeep, kKeep - 1, kKeep - 1, kept.long_pick, kShort);
  std::printf("[test_logit_rows_gpu] ok\n");
  return 0;
}
