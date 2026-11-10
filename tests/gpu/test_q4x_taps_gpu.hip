// a step's taps (Q4xOptions' tap_layers, q4x_taps): each tapped layer's output, its residual's hc streams' mean -
// DFlash2's features, layers 5, 15, 25, 35, 45. the same tokens - a prompt of 3,000 (past MM_MAX_TOKENS: its
// sub-batches) then decode steps - through a forward without taps and one with: every step's logits the same bits; each
// step's taps the bits of the probe's l_last at those layers averaged on the host in q4x_tap's order. and the refusals:
// a layer past the model's, one out of order, more than Q4X_MAX_TAPS. needs a gfx1151 and the UD-IQ3_XXS set; skipped
// (77) without them.
#include "q4x_ref.h"

extern "C" {
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"
}

constexpr uint32_t kCtx = 4096, kStep = 4096, kPrompt = 3000, kDecode = 4;
static const int32_t kLayers[5] = {5, 15, 25, 35, 45};

struct Probe {  // the probe's l_last at the tapped layers, averaged as q4x_tap averages: the reference
  uint32_t hc, D;
  std::vector<float> want;  // [token][5][D] of the step
};

static void probe(const char* step, int layer, float* dev, uint64_t n, void* arg) {
  Probe* p = (Probe*) arg;
  int k = -1;
  for (int i = 0; i < 5; i++) k = kLayers[i] == layer ? i : k;
  if (std::strcmp(step, "l_last") || k < 0) return;
  const std::vector<float> h = host_of(dev, n);
  const uint64_t tok = n / ((uint64_t) p->hc * p->D);
  p->want.resize(tok * 5 * p->D);
  for (uint64_t t = 0; t < tok; t++)
    for (uint32_t d = 0; d < p->D; d++) {
      float s = 0;
      for (uint32_t q = 0; q < p->hc; q++) s += h[(t * p->hc + q) * p->D + d];
      p->want[(t * 5 + k) * p->D + d] = s * (1.0f / (float) p->hc);
    }
}

struct Run {
  std::vector<std::vector<float>> logits, taps, want;  // a step each: the prompt, then each decode step
};

static Run run(const Q4xWeights* w, bool taps, bool with_probe, const std::vector<int32_t>& p, uint32_t vocab) {
  const Q4xOptions o = {kCtx, kStep, false, taps ? kLayers : nullptr, taps ? 5u : 0u};
  char err[Q4X_FWD_ERR] = "";
  Q4xForward* f = q4x_forward_create_with(w, nullptr, &o, err);
  if (!f) std::fprintf(stderr, "  create: %s\n", err);
  CHECK(f);
  CHECK(taps == (q4x_taps(f) != nullptr));
  Probe pr{(uint32_t) w->geo.hc, (uint32_t) w->geo.n_embd, {}};
  if (with_probe) q4x_forward_probe(f, probe, &pr);
  Run r;
  Q4xSeq* s = q4x_seq_create(f, err);
  auto keep = [&](uint32_t n, uint32_t row) {
    r.logits.push_back(host_of(q4x_logits(f) + (size_t) row * vocab, vocab));
    if (taps) r.taps.push_back(host_of(q4x_taps(f), (size_t) n * 5 * w->geo.n_embd));
    if (with_probe) r.want.push_back(pr.want);
  };
  CHECK(s && q4x_batch(f, s, p.data(), kPrompt, 0, err));
  int32_t t;
  CHECK(q4x_greedy(f, kPrompt - 1, 1, &t, err));
  keep(kPrompt, kPrompt - 1);
  for (uint32_t i = 0; i < kDecode; i++) {
    CHECK(q4x_batch(f, s, &t, 1, 0, err) && q4x_greedy(f, 0, 1, &t, err));
    keep(1, 0);
  }
  q4x_seq_destroy(s);
  q4x_forward_destroy(f);
  return r;
}

static bool refused(const Q4xWeights* w, const int32_t* layers, uint32_t n) {
  const Q4xOptions o = {kCtx, kStep, false, layers, n};
  char err[Q4X_FWD_ERR] = "";
  Q4xForward* f = q4x_forward_create_with(w, nullptr, &o, err);
  if (f) q4x_forward_destroy(f);
  std::printf("  refused: %s\n", err);
  return !f && err[0];
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_q4x_taps_gpu] start\n");
  const std::string first = home_path(kQwenFirst);
  int ndev = 0;
  if (access(first.c_str(), R_OK) || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_q4x_taps_gpu] skipped: no Qwen3.8 UD-IQ3_XXS or no GPU\n");
    return 77;
  }
  char gerr[GGUF_ERR] = "", werr[Q4X_ERR] = "";
  Gguf* g = gguf_create(first.c_str(), gerr);
  CHECK(g);
  Q4xWeights* w = q4x_weights_create(g, nullptr, werr);
  CHECK(w);
  const uint32_t vocab = (uint32_t) w->geo.n_vocab;
  std::vector<int32_t> p(kPrompt);
  uint32_t seed = 4242;
  for (auto& x : p) seed = seed * 1664525u + 1013904223u, x = (int32_t) ((seed >> 8) % (vocab - 1000)) + 100;

  const Run plain = run(w, false, false, p, vocab), tapped = run(w, true, false, p, vocab);
  for (size_t i = 0; i < plain.logits.size(); i++)
    CHECK(std::memcmp(plain.logits[i].data(), tapped.logits[i].data(), vocab * 4) == 0);
  std::printf("  the prompt's and %u decode steps' logits the same bits, taps or none\n", kDecode);
  const Run probed = run(w, true, true, p, vocab);
  for (size_t i = 0; i < probed.taps.size(); i++)
    CHECK(probed.taps[i].size() == probed.want[i].size() &&
          std::memcmp(probed.taps[i].data(), probed.want[i].data(), probed.taps[i].size() * 4) == 0 &&
          std::memcmp(probed.taps[i].data(), tapped.taps[i].data(), probed.taps[i].size() * 4) == 0);
  std::printf("  every step's taps the probe's l_last at layers 5 .. 45 averaged, bit for bit (and the same without it)\n");

  const int32_t past[1] = {(int32_t) w->geo.n_layers}, down[2] = {15, 5}, many[Q4X_MAX_TAPS + 1] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  CHECK(refused(w, past, 1) && refused(w, down, 2) && refused(w, many, Q4X_MAX_TAPS + 1) && refused(w, nullptr, 2));
  q4x_weights_destroy(w);
  gguf_destroy(g);
  std::printf("[test_q4x_taps_gpu] ok\n");
  return 0;
}
