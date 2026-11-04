// DSpark drafting and the verify around it: a real-text prompt, then greedy tokens stepped one at a time and again
// drafted (drafter_draft), verified in one batch, the rejected undone (seq_rollback) and the accepted seeded
// (drafter_seed) - the same tokens, whatever the drafts; and the drafts mostly right (a drafter that computes
// nonsense has its drafts rejected: ~0 accepted). needs a gfx1151, ~/models/ds1017, ~/models/dspark and
// ~/osmos-oracle/ids_real1k.txt; skipped (77) without them.
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/model/weights.h"
#include "osmos/run/drafter.h"
#include "osmos/run/forward.h"
}
#include "test_model.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr uint64_t kBudget = 64ull << 30;  // the drafter's ~8 GB beside it, of this box's 125 GB
constexpr int kPrompt = 256, kTokens = 48;
constexpr float kConfMin = 0.5f, kCumMin = 0.5f;  // maya's defaults (--dspark-conf, MAYA_DS4_DSPARK_CUM)
constexpr double kAcceptFloor = 0.3;              // measured ~74% on this text; nonsense drafts ~0

static uint32_t V;
static std::vector<float> lg;
static int argmax_row(Forward* f, int i) {
  CHECK(hipMemcpy(lg.data(), forward_logits(f) + (size_t) i * V, V * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
}

int main() {
  std::printf("[test_drafter_gpu] start\n");
  const std::string home = std::getenv("HOME") ? std::getenv("HOME") : "";
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  // the model's own drafter: ds1017's beside it, UD's the packs' (dspark-iq2xs)
  const std::string ddir = which == TestModel::kGsq ? home + "/models/dspark" : home + "/Models/ds1017/dspark-iq2xs";
  FILE* idf = std::fopen((home + "/bench/osmos-prompts/calib16k.txt").c_str(), "r");
  int ndev = 0;
  if (!idf || which == TestModel::kNone || access((ddir + "/dspark.bin").c_str(), R_OK) || hipGetDeviceCount(&ndev) ||
      !ndev) {
    std::printf("[test_drafter_gpu] skipped: no model, drafter, the bench's text or GPU\n");
    return 77;
  }
  std::vector<int32_t> ids(kPrompt);
  for (auto& v : ids) CHECK(std::fscanf(idf, "%d,", &v) == 1);
  std::fclose(idf);
  char ferr[FORWARD_ERR] = "";
  Gguf *g = nullptr, *eg = nullptr;
  Weights* w = test_weights(path, which, &g, &eg);
  CHECK(w);
  V = (uint32_t) w->geo.n_vocab;
  lg.resize(V);
  Forward* f = forward_create(w, w->geo.n_layers, 4096, kBudget, ferr);
  CHECK(f);
  CHECK(!drafter_create(f, "/nonexistent", ferr));
  Drafter* d = drafter_create(f, ddir.c_str(), ferr);
  if (!d) std::printf("  %s\n", ferr);
  CHECK(d);
  const auto prompt = [&](Seq* s) {
    std::vector<Seq*> ss(kPrompt, s);
    const bool ok = forward_batch(f, ss.data(), ids.data(), kPrompt, ferr);
    if (!ok) std::fprintf(stderr, "  %s\n", ferr);
    CHECK(ok);
    return argmax_row(f, kPrompt - 1);
  };
  // drafted and verified against stepping one at a time: once in the default mode, once in the lane's (its small
  // steps' experts in the matvec's order: the plain steps and the verifies alike, so still the same tokens)
  const auto both = [&](const char* mode, bool first) {
    Seq* s = seq_create(f, ferr);
    int y = prompt(s);
    std::vector<int> plain;
    for (int i = 0; i < kTokens; i++) {
      plain.push_back(y);
      const bool stepped = forward_step(f, s, y, ferr);
      if (!stepped) std::fprintf(stderr, "  step %d: %s\n", i, ferr);
      CHECK(stepped);
      y = argmax_row(f, 0);
    }
    seq_destroy(s);

    s = seq_create(f, ferr);
    int32_t dr[DRAFTER_BLOCK];
    float cf[DRAFTER_BLOCK];
    y = prompt(s);
    if (first) {
      CHECK(drafter_draft(d, s, y, kConfMin, kCumMin, dr, cf, ferr) < 0);  // not seeded: refused
      CHECK(!drafter_seed(d, s, kPrompt, kPrompt + 1, ferr));               // a position the last step did not tap
    }
    CHECK(drafter_seed(d, s, kPrompt - 128, kPrompt, ferr));
    std::vector<int> spec;
    int32_t pos = kPrompt;
    int drafted = 0, accepted = 0, rounds = 0;
    while ((int) spec.size() < kTokens) {
      const int k = drafter_draft(d, s, y, kConfMin, kCumMin, dr, cf, ferr);
      CHECK(k >= 0 && k <= DRAFTER_BLOCK);
      std::vector<int32_t> toks{y};
      toks.insert(toks.end(), dr, dr + k);
      std::vector<Seq*> ss(toks.size(), s);
      CHECK(forward_batch(f, ss.data(), toks.data(), (uint32_t) toks.size(), ferr));
      int m = 0, next = -1;
      for (;; m++) {
        const int am = argmax_row(f, m);
        if (m == k || am != dr[m]) {
          next = am;
          break;
        }
      }
      CHECK(seq_rollback(s, (uint32_t) (k - m)) && drafter_seed(d, s, pos, pos + m + 1, ferr));
      spec.push_back(y);
      spec.insert(spec.end(), dr, dr + m);
      y = next, pos += m + 1, drafted += k, accepted += m, rounds++;
    }
    spec.resize(kTokens);
    CHECK(spec == plain);
    CHECK(drafted > 0 && accepted >= kAcceptFloor * drafted);
    std::printf("  %s: %d tokens drafted and verified in %d rounds: the tokens of stepping one at a time; drafts %d, "
                "accepted %d (%.0f%%)\n", mode, kTokens, rounds, drafted, accepted, 100.0 * accepted / drafted);
    seq_destroy(s);
  };
  both("default", true);
  CHECK(forward_lane(f, 0, ferr));
  both("the lane", false);
  drafter_destroy(d);
  forward_destroy(f);
  weights_destroy(w);
  gguf_destroy(g);
  gguf_destroy(eg);
  std::printf("[test_drafter_gpu] ok\n");
  return 0;
}
