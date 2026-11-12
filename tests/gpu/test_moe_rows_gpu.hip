// a large step's routed experts in chunks (ForwardOptions' moe_rows) against the step whole: one step of kLong tokens
// over sub-batches, whole, in chunks of a sub-batch and in chunks that cut across sub-batches - its kept rows' logits
// the same bits; then a short step on each (its logits read what the long one left in the cache: the same bits). needs
// a gfx1151 and a model (test_model.h); skipped (77) without.
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

constexpr uint32_t kCtx = 8192, kMaxTokens = 8192, kLong = 6000, kShort = 5;
constexpr uint64_t kBudget = 24ull << 30;  // one forward at a time: a small pool, the rest streamed

struct Out {
  std::vector<float> long_rows, short_rows;
};

static Out run(Weights* w, uint32_t moe_rows, const std::vector<int32_t>& ids) {
  const size_t V = w->geo.n_vocab;
  char err[FORWARD_ERR] = "";
  ForwardOptions o = {(int) w->geo.n_layers, kCtx, kBudget, nullptr, kMaxTokens, true, false, 0, 0, 0, moe_rows};
  Forward* f = forward_create_with(w, &o, err);
  if (!f) std::fprintf(stderr, "forward: %s\n", err);
  CHECK(f && forward_lane(f, 0, NULL, err));
  Seq* s = seq_create(f, err);
  CHECK(s);
  std::vector<Seq*> ss(kLong, s);
  Out r;
  CHECK(forward_batch(f, ss.data(), ids.data(), kLong, err));
  const uint32_t kept = forward_logit_rows(f);  // the last sub-batch's rows
  CHECK(kept > 0 && kept <= kLong);
  r.long_rows.resize((size_t) kept * V);
  CHECK(hipDeviceSynchronize() == hipSuccess &&
        hipMemcpy(r.long_rows.data(), forward_logits(f), r.long_rows.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(forward_batch(f, ss.data(), ids.data() + kLong, kShort, err));
  r.short_rows.resize((size_t) kShort * V);
  CHECK(hipDeviceSynchronize() == hipSuccess &&
        hipMemcpy(r.short_rows.data(), forward_logits(f), r.short_rows.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  seq_destroy(s);
  forward_destroy(f);
  return r;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_moe_rows_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_moe_rows_gpu] skipped: no model or GPU\n");
    return 77;
  }
  Gguf *g = nullptr, *eg = nullptr;
  Weights* w = test_weights(path, which, &g, &eg);
  CHECK(w);
  const uint32_t V = (uint32_t) w->geo.n_vocab;
  std::vector<int32_t> ids(kLong + kShort);
  uint32_t r = 1017;
  for (auto& x : ids) r = r * 1664525u + 1013904223u, x = (int32_t) ((r >> 8) % (V - 1000)) + 100;
  const Out whole = run(w, 0, ids);
  std::printf("  a %u-token step, its routed experts whole: %zu kept rows\n", kLong, whole.long_rows.size() / V);
  for (uint32_t c : {2048u, 1000u}) {
    const Out chunked = run(w, c, ids);
    CHECK(chunked.long_rows == whole.long_rows);
    CHECK(chunked.short_rows == whole.short_rows);
    std::printf("  in chunks of %u rows: the step's kept rows and the next step's %u rows the same bits\n", c, kShort);
  }
  std::printf("[test_moe_rows_gpu] ok\n");
  return 0;
}
