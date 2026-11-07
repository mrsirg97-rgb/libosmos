// Qwen3.8-Flash-Next's whole forward (osmos/q4x/forward.h) against llama.cpp's own tensors (q4x_ref.h): the dump's
// 256 tokens one batch - every layer's residual (l_last) and the logits (result_output: the next token's
// distribution's KL from llama.cpp's, and its greatest the same) - then the same tokens as 128 + 128 and as 255 + 1,
// the state carried: the logits the bits of the one batch; then three sequences at their own positions sharing steps
// (q4x_step: 3 + 150 + 1 tokens, past one x16's 128, and 2 + 1 within it) - each one's rows the bits of the one
// batch's; then a step past MM_MAX_TOKENS against steps within it. needs a gfx1151, the UD-IQ3_XXS set (~78 GB of
// GPU memory) and the dump; skipped (77) without them.
#include <algorithm>

#include "q4x_ref.h"

extern "C" {
#include "osmos/gpu/mm.h"
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

// rows `rows` of the last step (from row `at`) against the one batch's from position `pos`
static bool same_rows(const Q4xForward* f, uint32_t at, uint32_t rows, const std::vector<float>& logits, uint32_t pos,
                      uint32_t vocab) {
  return std::memcmp(host_of(q4x_logits(f) + (size_t) at * vocab, (size_t) rows * vocab).data(),
                     &logits[(size_t) pos * vocab], (size_t) rows * vocab * 4) == 0;
}

// sequences at 100, 0 and 200 tokens in one step, then two of them again: each its own rows, the one batch's bits;
// a sequence twice, or one past max_ctx, refused with every position kept
static void shared_steps(Q4xForward* f, const std::vector<int32_t>& ids, const std::vector<float>& logits,
                         uint32_t vocab) {
  char ferr[Q4X_FWD_ERR] = "";
  Q4xSeq *a = q4x_seq_create(f, ferr), *b = q4x_seq_create(f, ferr), *c = q4x_seq_create(f, ferr);
  CHECK(a && b && c && q4x_batch(f, a, ids.data(), 100, 0, ferr) && q4x_batch(f, c, ids.data(), 200, 0, ferr));
  std::vector<int32_t> toks(ids.begin() + 100, ids.begin() + 103);
  toks.insert(toks.end(), ids.begin(), ids.begin() + 150);
  toks.push_back(ids[200]);
  Q4xSeq* seqs[] = {a, b, c};
  const uint32_t lens[] = {3, 150, 1};
  Q4xSeq* twice[] = {a, b, a};
  CHECK(!q4x_step(f, twice, lens, 3, toks.data(), 0, ferr));
  CHECK(!q4x_step(f, seqs, lens, 0, toks.data(), 0, ferr));
  CHECK(q4x_seq_position(a) == 100 && q4x_seq_position(b) == 0 && q4x_seq_position(c) == 200);
  CHECK(q4x_step(f, seqs, lens, 3, toks.data(), 0, ferr));
  CHECK(same_rows(f, 0, 3, logits, 100, vocab) && same_rows(f, 3, 150, logits, 0, vocab) &&
        same_rows(f, 153, 1, logits, 200, vocab));
  CHECK(q4x_seq_position(a) == 103 && q4x_seq_position(b) == 150 && q4x_seq_position(c) == 201);
  const int32_t two[] = {ids[103], ids[104], ids[201]};
  Q4xSeq* ac[] = {a, c};
  const uint32_t l2[] = {2, 1};
  CHECK(q4x_step(f, ac, l2, 2, two, 5, ferr));
  CHECK(same_rows(f, 5, 2, logits, 103, vocab) && same_rows(f, 7, 1, logits, 201, vocab));
  std::printf("  three sequences sharing steps (3 + 150 + 1, then 2 + 1): each one's rows the bits of the one batch\n");
  q4x_seq_destroy(a), q4x_seq_destroy(b), q4x_seq_destroy(c);
}

// a step past MM_MAX_TOKENS (its layers' parts a sub-batch of MM_MAX_TOKENS at a time, its experts over it whole):
// two sequences in one step of 3,000 (1,000 + 2,000: the bound 1,048 into the second) and one sequence of 3,000 -
// the rows across the bound and each one's last the bits of the same tokens in steps of MM_MAX_TOKENS or fewer
static void long_step(const Q4xWeights* w, const std::vector<int32_t>& ids, uint32_t vocab) {
  constexpr uint32_t kA = 1000, kB = 2000, kLong = kA + kB, kAround = 16;  // rows each side of the bound
  static_assert(kLong > MM_MAX_TOKENS && kB <= MM_MAX_TOKENS, "a step past the bound, its references within it");
  char ferr[Q4X_FWD_ERR] = "";
  Q4xForward* f = q4x_forward_create(w, nullptr, kCtx, kLong, ferr);
  CHECK(f);
  std::vector<int32_t> ta(kA), tb(kB), tl(kLong);  // the dump's ids, over and over
  for (uint32_t i = 0; i < kLong; i++) tl[i] = ids[i % kTok];
  for (uint32_t i = 0; i < kA; i++) ta[i] = ids[(i + 7) % kTok];
  for (uint32_t i = 0; i < kB; i++) tb[i] = ids[(i + 31) % kTok];
  const uint32_t bound_b = MM_MAX_TOKENS - kA;  // the bound's place in the second sequence
  auto rows = [&](uint32_t at, uint32_t n) { return host_of(q4x_logits(f) + (size_t) at * vocab, (size_t) n * vocab); };
  // the references: each alone, within MM_MAX_TOKENS a step
  Q4xSeq* s = q4x_seq_create(f, ferr);
  CHECK(s && q4x_batch(f, s, ta.data(), kA, 0, ferr));
  const std::vector<float> ra = rows(kA - 1, 1);
  q4x_seq_destroy(s);
  s = q4x_seq_create(f, ferr);
  CHECK(s && q4x_batch(f, s, tb.data(), kB, 0, ferr));
  const std::vector<float> rb = rows(bound_b - kAround, 2 * kAround), rb_last = rows(kB - 1, 1);
  q4x_seq_destroy(s);
  s = q4x_seq_create(f, ferr);
  CHECK(s && q4x_batch(f, s, tl.data(), MM_MAX_TOKENS, 0, ferr));
  const std::vector<float> rl0 = rows(MM_MAX_TOKENS - kAround, kAround);
  CHECK(q4x_batch(f, s, tl.data() + MM_MAX_TOKENS, kLong - MM_MAX_TOKENS, 0, ferr));
  const std::vector<float> rl1 = rows(0, kAround), rl_last = rows(kLong - MM_MAX_TOKENS - 1, 1);
  q4x_seq_destroy(s);
  // two sequences in one long step
  Q4xSeq *a = q4x_seq_create(f, ferr), *b = q4x_seq_create(f, ferr);
  Q4xSeq* seqs[] = {a, b};
  const uint32_t lens[] = {kA, kB};
  std::vector<int32_t> both(ta);
  both.insert(both.end(), tb.begin(), tb.end());
  CHECK(a && b && q4x_step(f, seqs, lens, 2, both.data(), 0, ferr));
  CHECK(rows(kA - 1, 1) == ra && rows(kA + bound_b - kAround, 2 * kAround) == rb && rows(kLong - 1, 1) == rb_last);
  CHECK(q4x_seq_position(a) == (int32_t) kA && q4x_seq_position(b) == (int32_t) kB);
  q4x_seq_destroy(a), q4x_seq_destroy(b);
  // one sequence of the long step's whole
  s = q4x_seq_create(f, ferr);
  CHECK(s && q4x_batch(f, s, tl.data(), kLong, 0, ferr));
  CHECK(rows(MM_MAX_TOKENS - kAround, kAround) == rl0 && rows(MM_MAX_TOKENS, kAround) == rl1 &&
        rows(kLong - 1, 1) == rl_last);
  q4x_seq_destroy(s);
  q4x_forward_destroy(f);
  std::printf("  a step of 3,000 (1,000 + 2,000, and one of 3,000): the rows across MM_MAX_TOKENS's bound and each "
              "one's last the bits of steps within it\n");
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
  {  // small steps' shared expert on the second stream forced to end last (q4x_forward_side_spin): 248 then 8 single
     // steps, each row the one batch's - a read of it without the wait would see the step before's
    constexpr uint32_t kSpinUs = 2000, kSingles = 8;
    q4x_forward_side_spin(f, kSpinUs);
    s = q4x_seq_create(f, ferr);
    CHECK(s && q4x_batch(f, s, ids.data(), kTok - kSingles, 0, ferr));
    for (uint32_t t = kTok - kSingles; t < kTok; t++) {
      CHECK(q4x_batch(f, s, ids.data() + t, 1, 0, ferr));
      CHECK(std::memcmp(host_of(q4x_logits(f), vocab).data(), &logits[(size_t) t * vocab], vocab * 4) == 0);
    }
    q4x_forward_side_spin(f, 0);
    std::printf("  single steps with the shared expert forced to end last: each row the one batch's\n");
  }
  q4x_seq_destroy(s);
  shared_steps(f, ids, logits, vocab);
  q4x_forward_destroy(f);
  long_step(w, ids, vocab);  // a forward of its own (the first's weights freed first)
  q4x_weights_destroy(w);
  gguf_destroy(g);
  std::printf("[test_q4x_forward_gpu] ok\n");
  return 0;
}
