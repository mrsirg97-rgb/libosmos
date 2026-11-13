// q4x_kld step n ids.txt: a Qwen3.8-Flash-Next pack's distance from a reference on a text, token by token - kld's
// protocol and files for the other family (DESIGN.md, "0.6.6: the reference, and how a quant is scored"; "0.6.9"):
// one sequence, `step` tokens a q4x_batch, each row's log-softmax in double on the host.
//   REF_OUT=path   the run is the reference: every scored row's log-probabilities written (kld's osmosref file)
//   REF=path       the run is a candidate: each scored row's KL(reference || run) over the whole vocabulary - the
//                  reference may be another engine's (tools/bench/llama_ref.cpp: llama.cpp's, the same file)
//   DUMP=path      either run: kld's record a row (kl, nll, top, ref_top) - tools/bench/kld_compare.py's
//   MASK=mask.txt  0 / 1 a token: row r scored only when token r + 1 is 1 (an agent trace's assistant tokens)
// environment: MODEL (a first shard), OVERLAY (q4x_weights_create's: an abliteration), CTX (the forward's context:
// past the native 262,144 YaRN, q4x_rope - the 262K / 512K question), STREAM (the n-gram table from its file).
// build: hipcc -O2 -Iinclude --offload-arch=gfx1151 tools/bench/q4x_kld.hip -Llib -losmos_hip -losmos
//        -lhsa-runtime64 -o bin/q4x_kld
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/q4x/forward.h"
#include "osmos/run/engine.h"
#include "osmos/q4x/weights.h"
#include "osmos/run/gpu.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

struct RefHead {  // kld.hip's: "osmosref", then a row of vocab F32 log-probabilities a scored row
  char magic[8];
  uint32_t vocab, n, step, rows;
};
struct Rec {
  float kl, nll;
  int32_t top, ref_top;
};

static std::vector<int32_t> read_list(const char* path) {
  std::vector<int32_t> v;
  FILE* f = std::fopen(path, "r");
  CHECK(f);
  for (long x; std::fscanf(f, "%ld,", &x) == 1;) v.push_back((int32_t) x);
  std::fclose(f);
  return v;
}

int main(int argc, char** argv) {
  if (argc != 4) return std::fprintf(stderr, "usage: q4x_kld step n ids.txt (MODEL, OVERLAY, CTX, STREAM, REF_OUT | REF, DUMP, MASK)\n"), 2;
  CHECK(gpu_sleep_on_waits());
  const uint32_t step = (uint32_t) std::atoi(argv[1]), n = (uint32_t) std::atoi(argv[2]);
  const std::vector<int32_t> ids = read_list(argv[3]);
  CHECK(step > 0 && ids.size() >= n && n > 1);
  std::vector<int32_t> mask(n, 1);
  if (const char* mp = std::getenv("MASK")) {
    mask = read_list(mp);
    CHECK(mask.size() >= n);
  }
  const char *ref_out = std::getenv("REF_OUT"), *ref_in = std::getenv("REF"), *dump = std::getenv("DUMP");
  CHECK(!(ref_out && ref_in));
  const char* mp = std::getenv("MODEL");
  CHECK(mp);
  char gerr[GGUF_ERR] = "", werr[Q4X_ERR] = "", ferr[Q4X_FWD_ERR] = "";
  Gguf* g = gguf_create(mp, gerr);
  if (!g) std::fprintf(stderr, "gguf_create: %s\n", gerr), std::_Exit(1);
  const char* op = std::getenv("OVERLAY");
  Gguf* ov = op ? gguf_create(op, gerr) : nullptr;
  if (op && !ov) std::fprintf(stderr, "%s: %s\n", op, gerr), std::_Exit(1);
  Q4xWeights* w = q4x_weights_create(g, ov, werr);
  if (!w) std::fprintf(stderr, "q4x_weights_create: %s\n", werr), std::_Exit(1);
  const uint32_t V = (uint32_t) w->geo.n_vocab;
  const uint32_t ctx = std::getenv("CTX") ? (uint32_t) std::atoi(std::getenv("CTX")) : n;
  CHECK(ctx >= n);

  uint32_t want = 0;
  for (uint32_t r = 0; r + 1 < n; r++) want += mask[r + 1] != 0;
  FILE *ro = nullptr, *ri = nullptr, *dp = nullptr;
  if (ref_out) {
    CHECK((ro = std::fopen(ref_out, "wb")));
    RefHead h = {{'o', 's', 'm', 'o', 's', 'r', 'e', 'f'}, V, n, step, want};
    CHECK(std::fwrite(&h, sizeof h, 1, ro) == 1);
  }
  if (ref_in) {
    RefHead h;
    CHECK((ri = std::fopen(ref_in, "rb")) && std::fread(&h, sizeof h, 1, ri) == 1);
    CHECK(!std::memcmp(h.magic, "osmosref", 8) && h.vocab == V && h.n == n && h.rows == want);
    if (h.step != step) std::fprintf(stderr, "note: the reference's step %u, this run's %u\n", h.step, step);
  }
  if (dump) CHECK((dp = std::fopen(dump, "wb")));

  const Q4xOptions qo = {ctx, step, std::getenv("STREAM") != nullptr, nullptr, 0, nullptr};
  Q4xForward* fw = q4x_forward_create_with(w, nullptr, &qo, ferr);
  // 0.7.0: the NPU beside the GPU (q4x_forward_npu): NPU=<its kernels' dir>, NPU_SHARE (NPU=1 builds)
  if (fw && std::getenv("NPU") &&
      !q4x_forward_npu(fw, std::getenv("NPU"), std::getenv("NPU_SHARE") ? std::strtof(std::getenv("NPU_SHARE"), nullptr) : OSMOS_NPU_SHARE_QWEN, ferr))
    q4x_forward_destroy(fw), fw = nullptr;
  Q4xSeq* s = fw ? q4x_seq_create(fw, ferr) : nullptr;
  if (!s) std::fprintf(stderr, "q4x forward: %s\n", ferr), std::_Exit(1);
  std::vector<float> lg((size_t) step * V), rl(V);
  std::vector<double> lp(V);
  std::vector<float> kls;
  double nll = 0;
  uint32_t scored = 0, agree = 0;
  for (uint32_t at = 0; at < n; at += step) {
    const uint32_t k = n - at < step ? n - at : step;
    if (!q4x_batch(fw, s, ids.data() + at, k, 0, ferr)) std::fprintf(stderr, "step at %u: %s\n", at, ferr), std::_Exit(1);
    CHECK(hipMemcpy(lg.data(), q4x_logits(fw), (size_t) k * V * 4, hipMemcpyDeviceToHost) == hipSuccess);
    for (uint32_t r = 0; r < k && at + r + 1 < n; r++) {  // row r predicts token at + r + 1
      if (!mask[at + r + 1]) continue;
      const float* x = &lg[(size_t) r * V];
      int32_t top = 0;
      for (uint32_t v = 1; v < V; v++)
        if (x[v] > x[top]) top = (int32_t) v;
      const double mx = x[top];
      double sum = 0;
      for (uint32_t v = 0; v < V; v++) sum += std::exp((double) x[v] - mx);
      CHECK(std::isfinite(sum));
      const double lse = mx + std::log(sum);
      for (uint32_t v = 0; v < V; v++) lp[v] = (double) x[v] - lse;
      Rec rec = {0.0f, (float) -lp[ids[at + r + 1]], top, -1};
      nll -= lp[ids[at + r + 1]];
      if (ro) {
        for (uint32_t v = 0; v < V; v++) rl[v] = (float) lp[v];
        CHECK(std::fwrite(rl.data(), 4, V, ro) == V);
      }
      if (ri) {
        CHECK(std::fread(rl.data(), 4, V, ri) == V);
        double kl = 0;
        int32_t rt = 0;
        for (uint32_t v = 0; v < V; v++) {
          if (rl[v] > rl[rt]) rt = (int32_t) v;
          const double p = std::exp((double) rl[v]);
          if (p > 0) kl += p * ((double) rl[v] - lp[v]);
        }
        rec.kl = (float) std::max(kl, 0.0), rec.ref_top = rt;
        agree += rt == top;
        kls.push_back(rec.kl);
      }
      if (dp) CHECK(std::fwrite(&rec, sizeof rec, 1, dp) == 1);
      scored++;
    }
    std::printf("  %u of %u tokens, %u scored\n", at + k, n, scored);
    std::fflush(stdout);
  }
  std::printf("ppl %.4f over %u scored rows (mean log-prob %.5f), steps of %u, context %u\n", std::exp(nll / scored),
              scored, -nll / scored, step, ctx);
  if (ri && !kls.empty()) {
    double m = 0;
    for (float v : kls) m += v;
    m /= (double) kls.size();
    std::vector<float> sorted = kls;
    std::sort(sorted.begin(), sorted.end());
    std::printf("kld mean %.6f median %.6f p99 %.6f, top-1 agreement %.2f%%\n", m, sorted[sorted.size() / 2],
                sorted[std::min(sorted.size() - 1, (size_t) ((double) sorted.size() * 0.99))],
                100.0 * agree / (double) kls.size());
  }
  if (ro) std::fclose(ro);
  if (ri) std::fclose(ri);
  if (dp) std::fclose(dp);
  q4x_seq_destroy(s);
  q4x_forward_destroy(fw);
  q4x_weights_destroy(w);
  gguf_destroy(g);
  return 0;
}
