// q4x_ppl <first shard> <ids> <n> [step]: Qwen3.8-Flash-Next's perplexity over the first n tokens of a text (its ids,
// comma separated - oracle/qwen_dump writes them with the model's own tokenizer): one sequence, `step` tokens a
// q4x_batch (256), each row's next token scored by log-softmax in double on the host - qwen_dump -ppl's protocol, so
// llama.cpp's number on the same ids is the comparison (DESIGN.md, "correctness first"). Q4X_PPL_CTX: a forward of
// that context (1,048,576: YaRN's factor 4; llama.cpp's with QWEN_DUMP_YARN=4).
// build: hipcc -O2 -Iinclude --offload-arch=gfx1151 tools/bench/q4x_ppl.hip -Llib -losmos_hip -losmos
//        -lhsa-runtime64 -o bin/q4x_ppl
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"
#include "osmos/run/gpu.h"
}

int main(int argc, char** argv) {
  if (argc < 4) return std::fprintf(stderr, "usage: q4x_ppl <first shard> <ids> <n> [step]\n"), 2;
  if (!gpu_sleep_on_waits()) return std::fprintf(stderr, "q4x_ppl: the GPU's wait mode not set\n"), 1;
  const uint32_t n = (uint32_t) std::atoi(argv[3]), step = argc > 4 ? (uint32_t) std::atoi(argv[4]) : 256;
  std::vector<int32_t> ids;
  FILE* f = std::fopen(argv[2], "r");
  for (int32_t x; f && ids.size() < n && std::fscanf(f, "%d,", &x) == 1;) ids.push_back(x);
  if (f) std::fclose(f);
  if (ids.size() != n || n < 2 || !step) return std::fprintf(stderr, "q4x_ppl: %zu ids, %u asked\n", ids.size(), n), 2;
  char gerr[GGUF_ERR] = "", werr[Q4X_ERR] = "", ferr[Q4X_FWD_ERR] = "";
  Gguf* g = gguf_create(argv[1], gerr);
  /* OVERLAY: a GGUF whose tensors stand in for the model's of their names (an abliteration: q4x_weights_create) */
  Gguf* ov = getenv("OVERLAY") ? gguf_create(getenv("OVERLAY"), gerr) : nullptr;
  if (getenv("OVERLAY") && !ov) return std::fprintf(stderr, "q4x_ppl: the overlay: %s\n", gerr), 1;
  Q4xWeights* w = g ? q4x_weights_create(g, ov, werr) : nullptr;
  if (!w) return std::fprintf(stderr, "q4x_ppl: %s%s\n", gerr, werr), 1;
  /* Q4X_PPL_CTX: the forward's context (past the native 262,144: YaRN, q4x_rope) - the text n tokens of it */
  const uint32_t ctx = getenv("Q4X_PPL_CTX") ? (uint32_t) atoi(getenv("Q4X_PPL_CTX")) : n;
  if (ctx < n) return std::fprintf(stderr, "q4x_ppl: Q4X_PPL_CTX below the text's %u tokens\n", n), 2;
  Q4xForward* fw = q4x_forward_create(w, nullptr, ctx, step, ferr);
  Q4xSeq* s = fw ? q4x_seq_create(fw, ferr) : nullptr;
  if (!s) return std::fprintf(stderr, "q4x_ppl: %s\n", ferr), 1;
  const uint32_t V = (uint32_t) w->geo.n_vocab;
  std::vector<float> lg((size_t) step * V);
  double nll = 0;
  for (uint32_t at = 0; at < n; at += step) {
    const uint32_t k = n - at < step ? n - at : step;
    if (!q4x_batch(fw, s, ids.data() + at, k, 0, ferr)) return std::fprintf(stderr, "q4x_ppl: %s\n", ferr), 1;
    if (hipMemcpy(lg.data(), q4x_logits(fw), (size_t) k * V * 4, hipMemcpyDeviceToHost) != hipSuccess) return 1;
    for (uint32_t r = 0; r < k && at + r + 1 < n; r++) {
      const float* l = &lg[(size_t) r * V];
      double m = l[0], sum = 0;
      for (uint32_t i = 1; i < V; i++) m = l[i] > m ? l[i] : m;
      for (uint32_t i = 0; i < V; i++) sum += std::exp(l[i] - m);
      nll -= l[ids[at + r + 1]] - m - std::log(sum);
    }
  }
  std::printf("q4x_ppl: perplexity %.4f over %u tokens (mean log-probability %.6f)\n", std::exp(nll / (n - 1)), n - 1,
              -nll / (n - 1));
  q4x_seq_destroy(s);
  q4x_forward_destroy(fw);
  q4x_weights_destroy(w);
  gguf_destroy(g);
  return 0;
}
