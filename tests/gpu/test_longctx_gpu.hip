// a 1M-token context (the packs' one: DESIGN.md, "osmos-api: the budgets") costs about its state - the compressed
// rows and indexer keys, ~0.9 KB a token - and no buffer a step's every token by every position: the GPU memory a
// forward and its sequence take at 1M less than kGrowthMax over 64K's (the indexer's scores were 2,048 tokens x 1M
// positions F32, 8 GiB). and the bits are not max_ctx's: a 4,096-token prompt in two full steps gives the last
// token's logits bit for bit at both. needs a gfx1151 and a model (test_model.h); skipped (77) without.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

constexpr uint32_t kMaxTokens = 2048, kPrompt = 4096;
constexpr uint32_t kShort = 65536, kLong = 1u << 20;
constexpr uint64_t kBudget = 20ull << 30;      // (the experts' room: the same at both, so it cancels)
constexpr uint64_t kGrowthMax = 3ull << 29;    // 1.5 GiB: the state's ~0.85 GiB and room to spare

struct Run {
  uint64_t bytes;
  std::vector<float> logits;
};

static Run run(Weights* w, uint32_t ctx, const std::vector<int32_t>& ids) {
  char err[FORWARD_ERR] = "";
  size_t free0 = 0, free1 = 0, total = 0;
  CHECK(hipMemGetInfo(&free0, &total) == hipSuccess);
  const ForwardOptions o = {(int) w->geo.n_layers, ctx, kBudget, nullptr, kMaxTokens};
  Forward* f = forward_create_with(w, &o, err);
  if (!f) std::printf("  %s\n", err);
  CHECK(f && forward_lane(f, 0, err));
  Seq* s = seq_create(f, err);
  CHECK(s && seq_prompt(s, kPrompt));
  CHECK(hipMemGetInfo(&free1, &total) == hipSuccess);
  std::vector<Seq*> ss(kMaxTokens, s);
  for (uint32_t at = 0; at < kPrompt; at += kMaxTokens) CHECK(forward_batch(f, ss.data(), ids.data() + at, kMaxTokens, err));
  Run r{free0 - free1, std::vector<float>(w->geo.n_vocab)};
  const float* row = forward_logits(f) + (uint64_t) ((kPrompt - 1) % kMaxTokens) * w->geo.n_vocab;
  CHECK(hipMemcpy(r.logits.data(), row, r.logits.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  seq_destroy(s);
  forward_destroy(f);
  return r;
}

int main() {
  std::printf("[test_longctx_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  const char* home = std::getenv("HOME");
  const std::string text = std::string(home ? home : "") + "/bench/osmos-prompts/calib16k.txt";
  FILE* tf = std::fopen(text.c_str(), "r");
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0 || !tf) {
    std::printf("[test_longctx_gpu] skipped: no model, GPU or the bench's text (%s)\n", text.c_str());
    return 77;
  }
  std::vector<int32_t> ids;
  for (long v; std::fscanf(tf, "%ld,", &v) == 1;) ids.push_back((int32_t) v);
  std::fclose(tf);
  CHECK(ids.size() >= kPrompt);
  ids.resize(kPrompt);
  const std::string dir = std::string(path).substr(0, std::string(path).rfind('/') + 1);
  char gerr[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "";
  Gguf* g = gguf_create(path, gerr);
  Gguf* eg = which == TestModel::kGsq ? gguf_create((dir + "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf").c_str(), gerr)
                                      : nullptr;
  Weights* w = weights_create(g, eg, werr);
  CHECK(w);
  const Run a = run(w, kShort, ids), b = run(w, kLong, ids);
  const uint64_t growth = b.bytes > a.bytes ? b.bytes - a.bytes : 0;
  std::printf("  GPU memory: %.2f GiB at %u, %.2f GiB at %u: %.2f GiB more (at most %.2f)\n", a.bytes / 1073741824.0,
              kShort, b.bytes / 1073741824.0, kLong, growth / 1073741824.0, kGrowthMax / 1073741824.0);
  const bool same = !std::memcmp(a.logits.data(), b.logits.data(), a.logits.size() * 4);
  std::printf("  the last token's logits at both: %s\n", same ? "the same bits" : "DIFFERENT");
  CHECK(same && growth <= kGrowthMax);
  weights_destroy(w);
  gguf_destroy(g);
  gguf_destroy(eg);
  std::printf("[test_longctx_gpu] ok\n");
  return 0;
}
