// Qwen3.8-Flash-Next's whole forward (osmos/q4x/forward.h) against llama.cpp's own tensors (q4x_ref.h): the dump's
// 256 tokens one batch - every layer's residual (l_last) and the logits (result_output: the next token's
// distribution's KL from llama.cpp's, and its greatest the same) - then the same tokens as 128 + 128 and as 255 + 1,
// the state carried: the logits the bits of the one batch. needs a gfx1151, the UD-IQ3_XXS set (~78 GB of GPU
// memory) and the dump; skipped (77) without them.
#include <algorithm>

#include "q4x_ref.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"
}

constexpr uint32_t kTok = 256, kHalf = kTok / 2, kCtx = 4096;
// a layer alone (teacher-forced: llama.cpp's own residual in): its mixers', block's and MoE's quantized matmuls each
// within llama.cpp's 1% - three in a row (measured worst 2.7e-2, layer 32). the free run's drift compounds them
// over 48 layers: its logits held to the first build's measure (KL mean 2.9e-2, the greatest token 93.4% the same)
// with room, a guard against a change for the worse - correctness's measure is the perplexity (DESIGN.md)
constexpr double kLayerAloneRel = 3 * kRelL2, kQwenKlMean = 5e-2, kQwenAgree = 0.9;

static std::vector<int32_t> dump_ids() {
  FILE* f = std::fopen((home_path(kQwenDump) + "/ids.txt").c_str(), "r");
  std::vector<int32_t> v;
  for (int32_t x; f && std::fscanf(f, "%d,", &x) == 1;) v.push_back(x);
  if (f) std::fclose(f);
  return v;
}

struct Seen {
  std::vector<std::vector<float>> layer;  // l_last a layer
};

static void probe(const char* step, int layer, float* dev, uint64_t n, void* arg) {
  Seen* s = (Seen*) arg;
  if (std::strcmp(step, "l_last") || layer < 0) return;
  if ((size_t) layer >= s->layer.size()) s->layer.resize(layer + 1);
  s->layer[layer] = host_of(dev, n);
}

// teacher-forced: a layer's output against llama.cpp's, then llama.cpp's written in its place
struct Forced {
  std::vector<double> rel;
};

static void force(const char* step, int layer, float* dev, uint64_t n, void* arg) {
  if (std::strcmp(step, "l_last") || layer < 0) return;
  Forced* fz = (Forced*) arg;
  char name[32];
  std::snprintf(name, sizeof name, "l_last-%d", layer);
  const Ref r = ref(name);
  CHECK(r.v.size() == n);
  const std::vector<float> ours = host_of(dev, n);
  double num = 0, den = 0;
  for (size_t i = 0; i < n; i++) num += (ours[i] - r.v[i]) * (double) (ours[i] - r.v[i]), den += (double) r.v[i] * r.v[i];
  fz->rel.push_back(std::sqrt(num / den));
  CHECK(hipMemcpy(dev, r.v.data(), n * 4, hipMemcpyHostToDevice) == hipSuccess);
}

// KL(llama || osmos) a token over its vocabulary (softmax of each's logits, double), and their greatest the same
static void logits_against(const std::vector<float>& ours, const Ref& theirs, uint32_t vocab, double* kl_mean,
                           double* kl_max, double* agree) {
  *kl_mean = *kl_max = 0;
  int same = 0;
  std::vector<double> p(vocab), q(vocab);
  for (uint32_t t = 0; t < kTok; t++) {
    const float *a = &theirs.v[(size_t) t * vocab], *b = &ours[(size_t) t * vocab];
    const double ma = *std::max_element(a, a + vocab), mb = *std::max_element(b, b + vocab);
    double sa = 0, sb = 0;
    for (uint32_t i = 0; i < vocab; i++) sa += (p[i] = std::exp(a[i] - ma)), sb += (q[i] = std::exp(b[i] - mb));
    double kl = 0;
    for (uint32_t i = 0; i < vocab; i++) {
      const double pi = p[i] / sa, qi = q[i] / sb;
      if (pi > 0) kl += pi * std::log(pi / qi);
    }
    *kl_mean += kl / kTok, *kl_max = kl > *kl_max ? kl : *kl_max;
    same += std::max_element(a, a + vocab) - a == std::max_element(b, b + vocab) - b;
  }
  *agree = (double) same / kTok;
}

int main() {
  std::printf("[test_q4x_forward_gpu] start\n");
  int ndev = 0;
  Ref probe0;
  char err[GGUF_ERR] = "", werr[Q4X_ERR] = "", ferr[Q4X_FWD_ERR] = "";
  Gguf* g = gguf_create(home_path(kQwenFirst).c_str(), err);
  const std::vector<int32_t> ids = dump_ids();
  if (!g || !ref_read("result_norm", &probe0) || ids.size() != kTok || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_q4x_forward_gpu] skipped: no Qwen3.8 UD-IQ3_XXS, its dump or a GPU\n");
    return 77;
  }
  Q4xWeights* w = q4x_weights_create(g, werr);
  CHECK(w);
  const uint32_t vocab = (uint32_t) w->geo.n_vocab;
  Q4xForward* f = q4x_forward_create(w, nullptr, kCtx, kTok, ferr);
  if (!f) std::fprintf(stderr, "  %s\n", ferr);
  CHECK(f);
  CHECK(!q4x_forward_create(w, nullptr, OPS_Q4X_MAX_KEYS + 1, kTok, ferr));  // past the dense attention's reach

  // the batch, every layer seen
  Seen seen;
  q4x_forward_probe(f, probe, &seen);
  Q4xSeq* s = q4x_seq_create(f, ferr);
  CHECK(s && !q4x_batch(f, s, ids.data(), kTok + 1, 0, ferr) && q4x_seq_position(s) == 0);
  CHECK(q4x_batch(f, s, ids.data(), kTok, 0, ferr) && q4x_seq_position(s) == (int32_t) kTok);
  q4x_forward_probe(f, nullptr, nullptr);
  const std::vector<float> logits = host_of(q4x_logits(f), (size_t) kTok * vocab);
  CHECK(seen.layer.size() == (size_t) w->geo.n_layers);
  double worst = 0;
  for (int l = 0; l < w->geo.n_layers; l++) {
    char name[32];
    std::snprintf(name, sizeof name, "l_last-%d", l);
    const Ref r = ref(name);
    double num = 0, den = 0;
    for (size_t i = 0; i < r.v.size(); i++) {
      const double d = seen.layer[l][i] - r.v[i];
      num += d * d, den += (double) r.v[i] * r.v[i];
    }
    const double rel = std::sqrt(num / den);
    worst = rel > worst ? rel : worst;
    if (l % 8 == 0 || l == w->geo.n_layers - 1) std::printf("  l_last-%-2d rel L2 %.2e\n", l, rel);
  }
  double kl_mean, kl_max, agree;
  logits_against(logits, ref("result_output"), vocab, &kl_mean, &kl_max, &agree);
  std::printf("  every layer's residual within %.2e of llama.cpp's; the logits: KL mean %.2e max %.2e, the greatest "
              "token the same for %.1f%%\n", worst, kl_mean, kl_max, 100 * agree);
  CHECK(kl_mean <= kQwenKlMean && agree >= kQwenAgree);
  q4x_seq_destroy(s);

  // teacher-forced: each layer from llama.cpp's own residual before it - a layer's error alone
  Forced forced;
  q4x_forward_probe(f, force, &forced);
  s = q4x_seq_create(f, ferr);
  CHECK(q4x_batch(f, s, ids.data(), kTok, 0, ferr));
  q4x_forward_probe(f, nullptr, nullptr);
  double fw = 0;
  int at = -1;
  for (size_t l = 0; l < forced.rel.size(); l++)
    if (forced.rel[l] > fw) fw = forced.rel[l], at = (int) l;
  std::printf("  teacher-forced, a layer alone: worst %.2e (layer %d); layers 0 / 8 / 24 / 47: %.2e %.2e %.2e %.2e\n",
              fw, at, forced.rel[0], forced.rel[8], forced.rel[24], forced.rel[47]);
  CHECK(forced.rel.size() == (size_t) w->geo.n_layers && fw <= kLayerAloneRel);
  q4x_seq_destroy(s);

  // the same tokens as 128 + 128, and as 255 + 1: the same bits
  s = q4x_seq_create(f, ferr);
  CHECK(q4x_batch(f, s, ids.data(), kHalf, 0, ferr) && q4x_batch(f, s, ids.data() + kHalf, kHalf, kHalf, ferr));
  CHECK(std::memcmp(host_of(q4x_logits(f), (size_t) kTok * vocab).data(), logits.data(),
                    (size_t) kTok * vocab * 4) == 0);  // each half's rows where its row0 put them
  q4x_seq_destroy(s);
  s = q4x_seq_create(f, ferr);
  CHECK(q4x_batch(f, s, ids.data(), kTok - 1, 0, ferr) && q4x_batch(f, s, ids.data() + kTok - 1, 1, kTok - 1, ferr));
  CHECK(std::memcmp(host_of(q4x_logits(f) + (size_t) (kTok - 1) * vocab, vocab).data(),
                    &logits[(size_t) (kTok - 1) * vocab], vocab * 4) == 0);
  CHECK(!q4x_batch(f, s, ids.data(), 2, kTok - 1, ferr));  // rows past max_tokens
  int32_t top = -1;
  CHECK(q4x_greedy(f, kTok - 1, 1, &top, ferr) &&
        top == (int32_t) (std::max_element(&logits[(size_t) (kTok - 1) * vocab], &logits[(size_t) kTok * vocab]) -
                          &logits[(size_t) (kTok - 1) * vocab]));
  std::printf("  128 + 128 and 255 + 1: the logits the bits of the one batch\n");
  q4x_seq_destroy(s);
  q4x_forward_destroy(f);
  q4x_weights_destroy(w);
  gguf_destroy(g);
  std::printf("[test_q4x_forward_gpu] ok\n");
  return 0;
}
