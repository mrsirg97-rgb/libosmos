// the lane's bits are not its residency's: forward_lane over 20 layers at two budgets - the tight one streaming far
// more of each step's experts, read while the ready ones run (hip_lane.c: those in GPU memory first, the reading
// ones after) - an 8-token prompt, then 24 tokens a step at a time; every step's stream out the same bits at both.
// a token's rows summed in any order but ascending expert id fails it. and the lane's CPU half at the tight budget
// (forward_lane's cpu_threads: the experts not in GPU memory computed on the CPU) - the same bits again, the CPU's
// picks counted. the test model's (test_model.h: UD-IQ1_M, or OSMOS_TEST_UD's) and ~/osmos-oracle/ids32.txt;
// skipped (77) without them
#include <hip/hip_runtime.h>
#include <unistd.h>

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

constexpr int kLayers = 20, kPrompt = 8, kIds = 32, kCtx = 256;
constexpr uint64_t kLoose = 30ull << 30, kTight = 12ull << 30;  // the tight one: most experts streamed
constexpr int kCpuThreads = 6;

static uint64_t run(Weights* w, const std::vector<int32_t>& ids, uint64_t budget, int cpu, uint64_t* streamed,
                    uint64_t* on_cpu) {
  char ferr[FORWARD_ERR] = "";
  Forward* f = forward_create(w, kLayers, kCtx, budget, ferr);
  if (!f) std::fprintf(stderr, "  %s\n", ferr);
  CHECK(f && forward_lane(f, cpu, NULL, ferr));
  Seq* s = seq_create(f, ferr);
  CHECK(s);
  std::vector<Seq*> ss(kPrompt, s);
  CHECK(forward_batch(f, ss.data(), ids.data(), kPrompt, ferr));
  std::vector<float> st((size_t) w->geo.hc * w->geo.n_embd);
  uint64_t h = 1469598103934665603ull;
  for (int p = kPrompt; p < kIds; p++) {
    const bool ok = forward_step(f, s, ids[p], ferr);
    if (!ok) std::fprintf(stderr, "  step %d: %s\n", p, ferr);
    CHECK(ok);
    CHECK(hipMemcpy(st.data(), forward_stream(f), st.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t b = 0; b < st.size() * 4; b++) h = (h ^ ((const uint8_t*) st.data())[b]) * 1099511628211ull;
  }
  uint64_t guessed = 0;
  forward_guesses(f, streamed, &guessed, nullptr);
  *on_cpu = forward_cpu_picks(f);
  seq_destroy(s);
  forward_destroy(f);
  return h;
}

int main() {
  std::printf("[test_lane_gpu] start\n");
  const std::string home = std::getenv("HOME") ? std::getenv("HOME") : "";
  FILE* idf = std::fopen((home + "/osmos-oracle/ids32.txt").c_str(), "r");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  if (!idf || which == TestModel::kNone || hipGetDeviceCount(&ndev) || !ndev) {
    std::printf("[test_lane_gpu] skipped: no test model, ids or GPU\n");
    return 77;
  }
  std::vector<int32_t> ids(kIds);
  for (auto& v : ids) CHECK(std::fscanf(idf, "%d,", &v) == 1);
  std::fclose(idf);
  Gguf *g = nullptr, *eg = nullptr;
  Weights* w = test_weights(path, which, &g, &eg);
  CHECK(w);
  uint64_t loose_n = 0, tight_n = 0, cpu_n = 0, none = 0, cpu_picks = 0;
  const uint64_t loose = run(w, ids, kLoose, 0, &loose_n, &none), tight = run(w, ids, kTight, 0, &tight_n, &none);
  const uint64_t cpu = run(w, ids, kTight, kCpuThreads, &cpu_n, &cpu_picks);
  std::printf("  %d layers, %d steps: streamed picks %llu (%llu GiB) and %llu (%llu GiB); the streams' fingerprint "
              "%016llx and %016llx\n", kLayers, kIds - kPrompt, (unsigned long long) loose_n,
              (unsigned long long) (kLoose >> 30), (unsigned long long) tight_n, (unsigned long long) (kTight >> 30),
              (unsigned long long) loose, (unsigned long long) tight);
  CHECK(tight_n > loose_n && tight_n > 0);  // the tight budget's misses the many
  CHECK(loose == tight);
  std::printf("  the CPU half (%d threads) at %llu GiB: %llu of %llu streamed picks on the CPU; fingerprint %016llx\n",
              kCpuThreads, (unsigned long long) (kTight >> 30), (unsigned long long) cpu_picks,
              (unsigned long long) cpu_n, (unsigned long long) cpu);
  CHECK(none == 0 && cpu_picks > 0 && cpu == tight);
  weights_destroy(w);
  gguf_destroy(eg);
  gguf_destroy(g);
  std::printf("[test_lane_gpu] ok\n");
  return 0;
}
