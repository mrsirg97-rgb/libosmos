// DSpark drafting and the verify around it: a real-text prompt, then greedy tokens stepped one at a time and again
// drafted (drafter_draft), verified in one batch, the rejected undone (seq_rollback) and the accepted seeded
// (drafter_seed) - the same tokens, whatever the drafts; and the drafts mostly right (a drafter that computes
// nonsense has its drafts rejected: ~0 accepted). needs a gfx1151, ~/models/ds1017, ~/models/dspark and
// ~/osmos-oracle/ids_real1k.txt; skipped (77) without them.
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
#include "osmos/err.h"
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
  char ferr[OSMOS_ERR] = "";
  Gguf *g = nullptr, *eg = nullptr;
  Weights* w = test_weights(path, which, &g, &eg);
  CHECK(w);
  V = (uint32_t) w->geo.n_vocab;
  lg.resize(V);
  Forward* f = nullptr;
  Drafter* d = nullptr;
  ForwardOptions o = {(int) w->geo.n_layers, 4096, 0, nullptr, FORWARD_MAX_TOKENS};
  o.drafter = ddir.c_str(); /* (the forward and its drafter, its own: everything the test holds) */
  TestFit fit = {[&](uint64_t b) {
                   o.budget = b;
                   return (f = forward_create_with(w, &o, ferr)) && (d = forward_drafter(f));
                 },
                 [&] { return forward_budget(f); }, [&](uint64_t* r) { return forward_room(f, r); },
                 [&] { d = nullptr, forward_destroy(f), f = nullptr; }};
  uint64_t budget = 0;  // (test_fit: the test's own, or this machine's fill where it is smaller)
  if (!test_fit(fit, kBudget, &budget)) std::printf("  %s\n", ferr);
  CHECK(f && d);
  osmos_clear(ferr);
  ForwardOptions none = o; /* (a drafter not there: refused before the forward takes its GPU memory) */
  none.drafter = "/nonexistent";
  CHECK(!forward_create_with(w, &none, ferr));
  const auto prompt = [&](Seq* s) {
    std::vector<Seq*> ss(kPrompt, s);
    osmos_clear(ferr);
    const bool ok = forward_batch(f, ss.data(), ids.data(), kPrompt, ferr);
    if (!ok) std::fprintf(stderr, "  %s\n", ferr);
    CHECK(ok);
    return argmax_row(f, kPrompt - 1);
  };
  // drafted and verified against stepping one at a time: once in the default mode, once in the lane's (its small
  // steps' experts in the matvec's order: the plain steps and the verifies alike, so still the same tokens)
  const auto both = [&](const char* mode, bool first) {
    osmos_clear(ferr);
    Seq* s = seq_create(f, ferr);
    int y = prompt(s);
    std::vector<int> plain;
    for (int i = 0; i < kTokens; i++) {
      plain.push_back(y);
      osmos_clear(ferr);
      const bool stepped = forward_step(f, s, y, ferr);
      if (!stepped) std::fprintf(stderr, "  step %d: %s\n", i, ferr);
      CHECK(stepped);
      y = argmax_row(f, 0);
    }
    seq_destroy(s);

    osmos_clear(ferr);
    s = seq_create(f, ferr);
    int32_t dr[DRAFTER_BLOCK];
    float cf[DRAFTER_BLOCK];
    y = prompt(s);
    if (first) {
      osmos_clear(ferr);
      CHECK(drafter_draft(d, s, y, kConfMin, kCumMin, dr, cf, ferr) < 0);  // not seeded: refused
      osmos_clear(ferr);
      CHECK(!drafter_seed(d, s, kPrompt, kPrompt + 1, ferr));               // a position the last step did not tap
    }
    osmos_clear(ferr);
    CHECK(drafter_seed(d, s, kPrompt - 128, kPrompt, ferr));
    std::vector<int> spec;
    int32_t pos = kPrompt;
    int drafted = 0, accepted = 0, rounds = 0;
    while ((int) spec.size() < kTokens) {
      osmos_clear(ferr);
      const int k = drafter_draft(d, s, y, kConfMin, kCumMin, dr, cf, ferr);
      CHECK(k >= 0 && k <= DRAFTER_BLOCK);
      std::vector<int32_t> toks{y};
      toks.insert(toks.end(), dr, dr + k);
      std::vector<Seq*> ss(toks.size(), s);
      osmos_clear(ferr);
      CHECK(forward_batch(f, ss.data(), toks.data(), (uint32_t) toks.size(), ferr));
      int m = 0, next = -1;
      for (;; m++) {
        const int am = argmax_row(f, m);
        if (m == k || am != dr[m]) {
          next = am;
          break;
        }
      }
      osmos_clear(ferr);
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
  // a prompt whose step's last sub-batch is shorter than the drafter's window (FWD_MAX_BATCH + 77 tokens: sub-batches
  // of 2,048 and 77, the seed's 127 positions across both - a tool result continuing a cached conversation, rig's
  // 2026-11-16 fault "a position the last step did not tap"): seeded, and its first drafts the bits of the same
  // prompt read as two steps of one sub-batch each (1,995 + 130)
  {
    constexpr int kLong = 2048 + 77, kWin = 127;
    std::vector<int32_t> lid(kLong);
    FILE* lf = std::fopen((home + "/bench/osmos-prompts/calib16k.txt").c_str(), "r");
    CHECK(lf);
    for (auto& v : lid) CHECK(std::fscanf(lf, "%d,", &v) == 1);
    std::fclose(lf);
    const auto read = [&](Seq* s, int from, int n) {
      std::vector<Seq*> ss(n, s);
      osmos_clear(ferr);
      const bool ok = forward_batch(f, ss.data(), lid.data() + from, (uint32_t) n, ferr);
      if (!ok) std::fprintf(stderr, "  %s\n", ferr);
      CHECK(ok);
    };
    int32_t dra[DRAFTER_BLOCK], drb[DRAFTER_BLOCK];
    float cfa[DRAFTER_BLOCK], cfb[DRAFTER_BLOCK];
    osmos_clear(ferr);
    Seq* a1 = seq_create(f, ferr);
    read(a1, 0, kLong);
    const int ya = argmax_row(f, (int) forward_logit_rows(f) - 1);  // the step's last row: its last sub-batch's
    osmos_clear(ferr);
    const bool seeded = drafter_seed(d, a1, kLong - kWin, kLong, ferr);
    if (!seeded) std::fprintf(stderr, "  one step of %d: %s\n", kLong, ferr);
    CHECK(seeded);
    osmos_clear(ferr);
    const int na = drafter_draft(d, a1, ya, kConfMin, kCumMin, dra, cfa, ferr);
    osmos_clear(ferr);
    Seq* b1 = seq_create(f, ferr);
    read(b1, 0, kLong - 130);
    read(b1, kLong - 130, 130);
    const int yb = argmax_row(f, (int) forward_logit_rows(f) - 1);
    osmos_clear(ferr);
    CHECK(drafter_seed(d, b1, kLong - kWin, kLong, ferr));
    osmos_clear(ferr);
    const int nb = drafter_draft(d, b1, yb, kConfMin, kCumMin, drb, cfb, ferr);
    if (!(ya == yb && na == nb)) std::fprintf(stderr, "  one step: y %d, %d drafts (%d ..); two steps: y %d, %d drafts (%d ..)\n", ya, na, na > 0 ? dra[0] : -1, yb, nb, nb > 0 ? drb[0] : -1);
    CHECK(ya == yb && na >= 0 && na == nb && std::memcmp(dra, drb, na * sizeof *dra) == 0 &&
          std::memcmp(cfa, cfb, na * sizeof *cfa) == 0);
    std::printf("  a %d-token step (sub-batches of 2,048 and 77): its last %d positions seeded; %d drafts the bits of "
                "the same prompt in two steps\n", kLong, kWin, na);
    seq_destroy(a1), seq_destroy(b1);
  }
  forward_lane_mode(f, true);
  both("the lane", false);

  forward_destroy(f); /* (its drafter with it) */
  weights_destroy(w);
  gguf_destroy(g);
  gguf_destroy(eg);
  std::printf("[test_drafter_gpu] ok\n");
  return 0;
}
