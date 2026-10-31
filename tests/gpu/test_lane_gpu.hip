// the lane's bits are not its residency's: forward_lane over 20 layers at two budgets - the tight one streaming far
// more of each step's experts, read while the ready ones run (hip_lane.c: those in GPU memory first, the reading
// ones after) - an 8-token prompt, then 24 tokens a step at a time; every step's stream out the same bits at both.
// a token's rows summed in any order but ascending expert id fails it. needs ~/models/ds1017 and
// ~/substrate-oracle/ids32.txt; skipped (77) without them
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
#include "substrate/model/gguf.h"
#include "substrate/model/weights.h"
#include "substrate/run/forward.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr int kLayers = 20, kPrompt = 8, kIds = 32, kCtx = 256;
constexpr uint64_t kLoose = 30ull << 30, kTight = 12ull << 30;  // the tight one: most experts streamed

static uint64_t run(Weights* w, const std::vector<int32_t>& ids, uint64_t budget, uint64_t* streamed) {
  char ferr[FORWARD_ERR] = "";
  Forward* f = forward_create(w, kLayers, kCtx, budget, ferr);
  if (!f) std::fprintf(stderr, "  %s\n", ferr);
  CHECK(f && forward_lane(f, 0, ferr));
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
  forward_guesses(f, streamed, &guessed);
  seq_destroy(s);
  forward_destroy(f);
  return h;
}

int main() {
  std::printf("[test_lane_gpu] start\n");
  const std::string home = std::getenv("HOME") ? std::getenv("HOME") : "", mdir = home + "/models/ds1017/";
  FILE* idf = std::fopen((home + "/substrate-oracle/ids32.txt").c_str(), "r");
  int ndev = 0;
  if (!idf || access((mdir + "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf").c_str(), R_OK) || hipGetDeviceCount(&ndev) ||
      !ndev) {
    std::printf("[test_lane_gpu] skipped: no ds1017, ids or GPU\n");
    return 77;
  }
  std::vector<int32_t> ids(kIds);
  for (auto& v : ids) CHECK(std::fscanf(idf, "%d,", &v) == 1);
  std::fclose(idf);
  char err[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "";
  Gguf* g = gguf_create((mdir + "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf").c_str(), err);
  Gguf* eg = gguf_create((mdir + "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf").c_str(), err);
  CHECK(g && eg);
  Weights* w = weights_create(g, eg, werr);
  CHECK(w);
  uint64_t loose_n = 0, tight_n = 0;
  const uint64_t loose = run(w, ids, kLoose, &loose_n), tight = run(w, ids, kTight, &tight_n);
  std::printf("  %d layers, %d steps: streamed picks %llu (%llu GiB) and %llu (%llu GiB); the streams' fingerprint "
              "%016llx and %016llx\n", kLayers, kIds - kPrompt, (unsigned long long) loose_n,
              (unsigned long long) (kLoose >> 30), (unsigned long long) tight_n, (unsigned long long) (kTight >> 30),
              (unsigned long long) loose, (unsigned long long) tight);
  CHECK(tight_n > loose_n && tight_n > 0);  // the tight budget's misses the many
  CHECK(loose == tight);
  weights_destroy(w);
  gguf_destroy(eg);
  gguf_destroy(g);
  std::printf("[test_lane_gpu] ok\n");
  return 0;
}
