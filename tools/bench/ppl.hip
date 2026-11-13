// ppl budget_gib step n ids.txt: a model's perplexity over the first n tokens of a text (token ids, comma separated:
// tools/bench/prompts.py's) - one sequence, read `step` tokens a forward_batch (the prompt's path; no seq_prompt, so
// every row's logits), each row's next token scored by log-softmax in double on the host. prints the perplexity and
// the mean log-probability, and every 1,024 tokens the running perplexity. a check of a pack's quality beside the
// oracle's KL (32 positions): two packs on the same text and steps, the same arithmetic.
// environment: MODEL (a first shard), PICKS (forward_create_ranked's profile), OVERLAY (weights_create_with's),
// NPU=<its kernels' dir> and NPU_SHARE (forward_npu; NPU=1 builds: -lxrt_coreutil).
// build: hipcc -O2 -Iinclude --offload-arch=gfx1151 tools/bench/ppl.hip -Llib -losmos_hip -losmos
//        -lhsa-runtime64 -o bin/ppl   (after `make lib/libosmos.a lib/libosmos_hip.a`)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/model/weights.h"
#include "osmos/run/forward.h"
#include "osmos/run/engine.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kReport = 1024;  // the running perplexity, every this many tokens scored

int main(int argc, char** argv) {
  if (argc != 5) return std::fprintf(stderr, "usage: ppl budget_gib step n ids.txt (MODEL, PICKS)\n"), 2;
  const uint64_t budget = (uint64_t) (std::atof(argv[1]) * (double) (1ull << 30));
  const uint32_t step = (uint32_t) std::atoi(argv[2]), n = (uint32_t) std::atoi(argv[3]);
  std::vector<int32_t> ids;
  FILE* tf = std::fopen(argv[4], "r");
  CHECK(tf && step > 0 && step <= MM_MAX_TOKENS);  // (forward_logits holds a step's rows up to one sub-batch)
  for (long v; std::fscanf(tf, "%ld,", &v) == 1;) ids.push_back((int32_t) v);
  std::fclose(tf);
  CHECK(ids.size() >= n && n > 1);
  const char* mp = std::getenv("MODEL");
  CHECK(mp);
  char err[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "", ferr[FORWARD_ERR] = "";
  Gguf* g = gguf_create(mp, err);
  if (!g) std::fprintf(stderr, "gguf_create: %s\n", err), std::_Exit(1);
  const char* op = std::getenv("OVERLAY");
  Gguf* ov = op ? gguf_create(op, err) : nullptr;
  if (op && !ov) std::fprintf(stderr, "%s: %s\n", op, err), std::_Exit(1);
  const WeightsOptions wo = {g, nullptr, ov};
  Weights* w = weights_create_with(&wo, werr);
  if (!w) std::fprintf(stderr, "weights_create: %s\n", werr), std::_Exit(1);
  ForwardOptions o = {};
  o.n_layers = (int) w->geo.n_layers, o.max_ctx = n, o.budget = budget, o.max_tokens = step;
  std::vector<uint64_t> picks;
  if (const char* pp = std::getenv("PICKS")) {
    FILE* pf = std::fopen(pp, "rb");
    CHECK(pf);
    picks.resize((size_t) w->geo.n_layers * w->geo.n_experts);
    CHECK(std::fread(picks.data(), 8, picks.size(), pf) == picks.size());
    std::fclose(pf);
    o.picks = picks.data();
  }
  Forward* f = forward_create_with(w, &o, ferr);
  if (!f) std::fprintf(stderr, "forward_create_with: %s\n", ferr), std::_Exit(1);
  CHECK(forward_lane(f, 0, NULL, ferr));
  if (const char* nd = std::getenv("NPU"))  // 0.7.0: the NPU beside the GPU (forward_npu), NPU_SHARE of a layer's pairs
    if (!forward_npu(f, nd, std::getenv("NPU_SHARE") ? std::strtof(std::getenv("NPU_SHARE"), nullptr) : OSMOS_NPU_SHARE_DS, ferr))
      std::fprintf(stderr, "forward_npu: %s\n", ferr), std::_Exit(1);
  Seq* s = seq_create(f, ferr);
  CHECK(s);
  const size_t V = (size_t) w->geo.n_vocab;
  std::vector<Seq*> ss(step, s);
  std::vector<float> lg(step * V);
  double nll = 0;
  uint32_t scored = 0;
  for (uint32_t at = 0; at < n; at += step) {
    const uint32_t k = n - at < step ? n - at : step;
    if (!forward_batch(f, ss.data(), ids.data() + at, k, ferr))
      std::fprintf(stderr, "step at %u: %s\n", at, ferr), std::_Exit(1);
    CHECK(hipMemcpy(lg.data(), forward_logits(f), (size_t) k * V * 4, hipMemcpyDeviceToHost) == hipSuccess);
    for (uint32_t r = 0; r < k && at + r + 1 < n; r++) {  // row r predicts token at + r + 1
      const float* x = &lg[(size_t) r * V];
      double mx = x[0], sum = 0;
      for (size_t v = 1; v < V; v++) mx = std::fmax(mx, (double) x[v]);
      for (size_t v = 0; v < V; v++) sum += std::exp((double) x[v] - mx);
      CHECK(std::isfinite(sum));
      nll += mx + std::log(sum) - (double) x[ids[at + r + 1]];
      if (++scored % kReport == 0) std::printf("  %u tokens: ppl %.4f\n", scored, std::exp(nll / scored));
    }
  }
  std::printf("ppl %.4f over %u tokens (mean log-prob %.5f), steps of %u\n", std::exp(nll / scored), scored,
              -nll / scored, step);
  if (std::getenv("NPU")) {
    uint64_t nc, np, no;
    double nw, ni;
    forward_npu_stats(f, &nc, &np, &nw, &ni, &no);
    std::printf("the NPU: %llu calls, %llu parts\n", (unsigned long long) nc, (unsigned long long) np);
  }
  seq_destroy(s);
  forward_destroy(f);
  weights_destroy(w);
  gguf_destroy(g);
  return 0;
}
