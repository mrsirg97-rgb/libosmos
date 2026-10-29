// schedule invariance, as a property: a token's logits the same bits however the forward was given it. 4 sequences of
// real ids (16 positions each) decoded a token at a time and alone - the reference - then again: interleaved (the 4 a
// step, a token each: 4 conversations), chunked (each its own tokens in random chunks of 1-16: a prompt read in
// pieces, or drafts verified), mixed (random batches of up to 16 over the sequences, random chunks); every logit
// bit for bit the reference's. then the tokens a second: alone, interleaved, a prompt 16 a step. needs a gfx1151,
// ~/models/ds1017 and ~/substrate-oracle/ids32.txt; skipped (77) without them.
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr uint64_t kBudget = 72ull << 30;
constexpr int kSeqs = 4, kLen = 16, kMaxBatch = 16, kTrials = 3;

struct Run {
  Forward* f;
  size_t V;
  std::vector<std::vector<int32_t>> ids;
  std::vector<std::vector<float>> logits;  // [seq][pos * V]
  double ms = 0;
  int steps = 0;
};

// a schedule: a list of batches, each a list of sequence indices (each takes that sequence's next token)
using Schedule = std::vector<std::vector<int>>;

static void play(Run* r, const Schedule& sch) {
  char err[FORWARD_ERR] = "";
  std::vector<Seq*> seqs(kSeqs);
  for (auto& s : seqs) CHECK((s = seq_create(r->f, err)));
  std::vector<int> next(kSeqs, 0);
  r->logits.assign(kSeqs, std::vector<float>((size_t) kLen * r->V));
  r->ms = 0, r->steps = 0;
  for (const auto& batch : sch) {
    std::vector<Seq*> bs;
    std::vector<int32_t> toks;
    std::vector<std::pair<int, int>> at;  // (seq, pos) of each row
    for (int k : batch) bs.push_back(seqs[k]), toks.push_back(r->ids[k][next[k]]), at.push_back({k, next[k]++});
    const auto a = std::chrono::steady_clock::now();
    if (!forward_batch(r->f, bs.data(), toks.data(), (uint32_t) bs.size(), err)) std::printf("  batch: %s\n", err);
    CHECK(hipDeviceSynchronize() == hipSuccess);
    r->ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count(), r->steps++;
    std::vector<float> l(bs.size() * r->V);
    CHECK(hipMemcpy(l.data(), forward_logits(r->f), l.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 0; i < at.size(); i++)
      std::memcpy(&r->logits[at[i].first][(size_t) at[i].second * r->V], &l[i * r->V], r->V * 4);
  }
  for (auto* s : seqs) seq_destroy(s);
}

static Schedule alone() {
  Schedule s;
  for (int k = 0; k < kSeqs; k++)
    for (int p = 0; p < kLen; p++) s.push_back({k});
  return s;
}

static Schedule interleaved() {
  Schedule s;
  for (int p = 0; p < kLen; p++) s.push_back({0, 1, 2, 3});
  return s;
}

static Schedule prompt() {  // each sequence read kMaxBatch tokens a step: a prompt in chunks
  Schedule s;
  for (int k = 0; k < kSeqs; k++)
    for (int p = 0; p < kLen; p += kMaxBatch) s.push_back(std::vector<int>(kMaxBatch, k));
  return s;
}

static Schedule chunked(std::mt19937& rng) {
  Schedule s;
  for (int k = 0; k < kSeqs; k++)
    for (int p = 0; p < kLen;) {
      const int c = std::min<int>(1 + rng() % kMaxBatch, kLen - p);
      s.push_back(std::vector<int>(c, k));
      p += c;
    }
  return s;
}

static Schedule mixed(std::mt19937& rng) {
  Schedule s;
  std::vector<int> left(kSeqs, kLen);
  while (std::any_of(left.begin(), left.end(), [](int x) { return x > 0; })) {
    std::vector<int> batch;
    const int want = 1 + rng() % kMaxBatch;
    while ((int) batch.size() < want) {
      std::vector<int> live;
      for (int k = 0; k < kSeqs; k++)
        if (left[k] > 0) live.push_back(k);
      if (live.empty()) break;
      const int k = live[rng() % live.size()];
      const int c = std::min<int>({1 + (int) (rng() % 3), left[k], want - (int) batch.size()});
      for (int j = 0; j < c; j++) batch.push_back(k);
      left[k] -= c;
    }
    s.push_back(batch);
  }
  return s;
}

int main() {
  std::printf("[test_batch_gpu] start\n");
  const std::string home = std::getenv("HOME") ? std::getenv("HOME") : "", mdir = home + "/models/ds1017/";
  FILE* idf = std::fopen((home + "/substrate-oracle/ids32.txt").c_str(), "r");
  int ndev = 0;
  if (!idf || access((mdir + "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf").c_str(), R_OK) || hipGetDeviceCount(&ndev) ||
      !ndev) {
    std::printf("[test_batch_gpu] skipped: no ds1017, ids or GPU\n");
    return 77;
  }
  std::vector<int32_t> all(32);
  for (auto& v : all) CHECK(std::fscanf(idf, "%d,", &v) == 1);
  std::fclose(idf);
  char err[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "", ferr[FORWARD_ERR] = "";
  Gguf* g = gguf_create((mdir + "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf").c_str(), err);
  Gguf* eg = gguf_create((mdir + "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf").c_str(), err);
  CHECK(g && eg);
  Weights* w = weights_create(g, eg, werr);
  CHECK(w);
  Run r{forward_create(w, w->geo.n_layers, 4096, kBudget, ferr), (size_t) w->geo.n_vocab, {}, {}};
  CHECK(r.f);
  for (int k = 0; k < kSeqs; k++)  // 4 windows of the real ids: different sequences, real text
    r.ids.push_back(std::vector<int32_t>(all.begin() + 5 * k, all.begin() + 5 * k + kLen));
  Seq* s0 = seq_create(r.f, ferr);
  std::vector<int32_t> many(kMaxBatch + 1, 0);
  std::vector<Seq*> many_s(kMaxBatch + 1, s0);
  CHECK(!forward_batch(r.f, many_s.data(), many.data(), kMaxBatch + 1, ferr));
  CHECK(!forward_batch(r.f, many_s.data(), many.data(), 0, ferr));
  int32_t bad = (int32_t) r.V;
  CHECK(!forward_batch(r.f, many_s.data(), &bad, 1, ferr));
  seq_destroy(s0);
  std::printf("  refusals: %d tokens, 0 tokens, a token past the vocabulary\n", kMaxBatch + 1);

  play(&r, alone());  // twice: warm the pool, then the reference
  play(&r, alone());
  const auto ref = r.logits;
  const double alone_ms = r.ms;
  std::mt19937 rng(1017);
  int schedules = 0;
  auto same = [&](const char* name) {
    schedules++;
    for (int k = 0; k < kSeqs; k++)
      if (r.logits[k] != ref[k]) {
        std::printf("  %s: sequence %d's logits NOT the reference's\n", name, k);
        std::exit(1);
      }
    std::printf("  %-12s %3d steps: every logit of the 4 x %d positions bit for bit the reference's\n", name, r.steps,
                kLen);
  };
  play(&r, interleaved());
  same("interleaved");
  const double inter_ms = r.ms;
  play(&r, prompt());
  same("prompt x16");
  const double prompt_ms = r.ms;
  for (int t = 0; t < kTrials; t++) {
    play(&r, chunked(rng));
    same("chunked");
  }
  for (int t = 0; t < kTrials; t++) {
    play(&r, mixed(rng));
    same("mixed");
  }
  const int n = kSeqs * kLen;
  std::printf("  %d schedules, the same bits. %d tokens: alone %.0f ms (%.1f tok/s); interleaved 4 a step %.0f ms "
              "(%.1f tok/s, %.2fx); a prompt 16 a step %.0f ms (%.1f tok/s, %.2fx)\n", schedules, n, alone_ms,
              n * 1e3 / alone_ms, inter_ms, n * 1e3 / inter_ms, alone_ms / inter_ms, prompt_ms, n * 1e3 / prompt_ms,
              alone_ms / prompt_ms);
  forward_destroy(r.f);
  weights_destroy(w);
  gguf_destroy(g);
  gguf_destroy(eg);
  std::printf("[test_batch_gpu] ok\n");
  return 0;
}
