// Qwen's n-gram table (per_layer_token_embd: 28.8 GB of UD-IQ3_XXS) streamed from its file (Q4xOptions' stream_ngram,
// as DeepSeek's Engram): its rows read each step into a pinned stage, the GPU memory the table took left free. the same
// tokens through a forward holding the table and one streaming it - a prompt of 3,000 (past MM_MAX_TOKENS: the
// sub-batches' rows) then decode steps - every step's logits the same bits; the streaming forward took the table's
// bytes less GPU memory. needs a gfx1151 and the UD-IQ3_XXS set; skipped (77) without them.
#include "q4x_ref.h"

extern "C" {
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"
}

constexpr uint32_t kCtx = 4096, kStep = 4096, kPrompt = 3000, kDecode = 8;

struct Run {
  std::vector<std::vector<float>> logits;  // the prompt's last row, then each decode step's
  uint64_t took;                           // the GPU memory the forward took at create
};

static Run run(const Q4xWeights* w, bool stream, const std::vector<int32_t>& p, uint32_t vocab) {
  size_t free0 = 0, free1 = 0, total = 0;
  CHECK(hipDeviceSynchronize() == hipSuccess && hipMemGetInfo(&free0, &total) == hipSuccess);
  const Q4xOptions o = {kCtx, kStep, stream, nullptr, 0};
  char err[Q4X_FWD_ERR] = "";
  Q4xForward* f = q4x_forward_create_with(w, nullptr, &o, err);
  if (!f) std::fprintf(stderr, "  create: %s\n", err);
  CHECK(f);
  CHECK(hipDeviceSynchronize() == hipSuccess && hipMemGetInfo(&free1, &total) == hipSuccess);
  Run r{{}, free0 > free1 ? free0 - free1 : 0};
  Q4xSeq* s = q4x_seq_create(f, err);
  CHECK(s && q4x_batch(f, s, p.data(), kPrompt, 0, err));
  int32_t t;
  CHECK(q4x_greedy(f, kPrompt - 1, 1, &t, err));
  r.logits.push_back(host_of(q4x_logits(f) + (size_t) (kPrompt - 1) * vocab, vocab));
  for (uint32_t i = 0; i < kDecode; i++) {
    CHECK(q4x_batch(f, s, &t, 1, 0, err) && q4x_greedy(f, 0, 1, &t, err));
    r.logits.push_back(host_of(q4x_logits(f), vocab));
  }
  q4x_seq_destroy(s);
  q4x_forward_destroy(f);
  return r;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_q4x_ngram_gpu] start\n");
  const std::string first = home_path(kQwenFirst);
  int ndev = 0;
  if (access(first.c_str(), R_OK) || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_q4x_ngram_gpu] skipped: no Qwen3.8 UD-IQ3_XXS or no GPU\n");
    return 77;
  }
  char gerr[GGUF_ERR] = "", werr[Q4X_ERR] = "";
  Gguf* g = gguf_create(first.c_str(), gerr);
  CHECK(g);
  Q4xWeights* w = q4x_weights_create(g, nullptr, werr);
  CHECK(w);
  const uint32_t vocab = (uint32_t) w->geo.n_vocab;
  std::vector<int32_t> p(kPrompt);
  uint32_t seed = 1017;
  for (auto& x : p) seed = seed * 1664525u + 1013904223u, x = (int32_t) ((seed >> 8) % (vocab - 1000)) + 100;
  const Run held = run(w, false, p, vocab), streamed = run(w, true, p, vocab);
  const uint64_t table = w->ple_table->bytes;
  std::printf("  GPU memory taken: %.2f GiB holding the table, %.2f streaming it (the table %.2f GiB)\n",
              held.took / 1073741824.0, streamed.took / 1073741824.0, table / 1073741824.0);
  CHECK(held.took >= streamed.took + table);
  for (size_t i = 0; i < held.logits.size(); i++)
    CHECK(std::memcmp(held.logits[i].data(), streamed.logits[i].data(), vocab * 4) == 0);
  std::printf("  the prompt's and %u decode steps' logits the same bits, the table held or streamed\n", kDecode);
  q4x_weights_destroy(w);
  gguf_destroy(g);
  std::printf("[test_q4x_ngram_gpu] ok\n");
  return 0;
}
