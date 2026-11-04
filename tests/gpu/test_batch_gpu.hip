// schedule invariance, as a property: a token's logits the same bits however the forward was given it. 4 sequences of
// real ids (16 positions each) decoded a token at a time and alone - the reference - then again: interleaved (the 4 a
// step, a token each: 4 conversations), chunked (each its own tokens in random chunks of 1-16: a prompt read in
// pieces, or drafts verified), mixed (random batches of up to 16 over the sequences, random chunks); every logit
// bit for bit the reference's. then the tokens a second: alone, interleaved, a prompt 16 a step. and sub-batches: 2
// sequences of 1,500 tokens interleaved in one batch of 3,000 (two sub-batches, each layer for both before the next),
// and one of 4,600 told its prompt's end (three; the first's tokens all before the decoder's reach: none past dec),
// each against the sequences alone in steps of 512 - the rows the batch leaves (its last sub-batch's) bit for bit.
// needs a gfx1151,
// ~/models/ds1017 and ~/osmos-oracle/ids32.txt; skipped (77) without them.
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
#include "osmos/model/gguf.h"
#include "osmos/model/weights.h"
#include "osmos/gpu/mm.h"
#include "osmos/run/forward.h"
}
#include "test_model.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr uint64_t kBudget = 72ull << 30;
constexpr int kSeqs = 4, kLen = 16, kMaxBatch = 16, kTrials = 3, kCtx = 8192;

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

// `ns` sequences of `len` tokens (ids drawn from the real ones), interleaved in one batch past MM_MAX_TOKENS,
// against each alone in steps of kSubStep: the logits the batch leaves (its last sub-batch's rows) the reference's,
// bit for bit. `skip`: each told its prompt's end (seq_prompt) - only its last token's logits
constexpr int kSubStep = 512, kMaxSeqs = 2;
static void sub_batches(Forward* f, size_t V, const std::vector<int32_t>& real, int ns, int len, bool skip) {
  char err[FORWARD_ERR] = "";
  std::mt19937 rng(41);
  std::vector<int32_t> ids[kMaxSeqs];
  for (int k = 0; k < ns; k++)
    for (int i = 0; i < len; i++) ids[k].push_back(real[rng() % real.size()]);
  const int n = ns * len, last = (n - 1) % MM_MAX_TOKENS + 1;  // the batch's last sub-batch: its rows left
  std::vector<std::vector<float>> want(ns);                      // [seq] the reference's rows of the last sub-batch
  std::vector<int> first(ns, -1);                                // a sequence's first position in it
  for (int i = n - last; i < n; i++)
    if (first[i % ns] < 0) first[i % ns] = i / ns;
  for (int k = 0; k < ns; k++) {
    Seq* s = seq_create(f, err);
    CHECK(s && (!skip || seq_prompt(s, (uint32_t) len)));
    for (int p = 0; p < len; p += kSubStep) {
      const int m = std::min(kSubStep, len - p);
      std::vector<Seq*> ss(m, s);
      CHECK(forward_batch(f, ss.data(), &ids[k][p], (uint32_t) m, err));
      for (int i = 0; i < m; i++)
        if (p + i >= first[k] && (!skip || p + i == len - 1)) {
          std::vector<float> row(V);
          CHECK(hipMemcpy(row.data(), forward_logits(f) + (size_t) i * V, V * 4, hipMemcpyDeviceToHost) == hipSuccess);
          want[k].insert(want[k].end(), row.begin(), row.end());
        }
    }
    seq_destroy(s);
  }
  Seq* s[kMaxSeqs];
  for (int k = 0; k < ns; k++) CHECK((s[k] = seq_create(f, err)) && (!skip || seq_prompt(s[k], (uint32_t) len)));
  std::vector<Seq*> bs(n);
  std::vector<int32_t> toks(n);
  for (int i = 0; i < n; i++) bs[i] = s[i % ns], toks[i] = ids[i % ns][i / ns];
  const auto a = std::chrono::steady_clock::now();
  CHECK(forward_batch(f, bs.data(), toks.data(), (uint32_t) n, err));
  CHECK(hipDeviceSynchronize() == hipSuccess);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
  std::vector<float> got((size_t) last * V);
  CHECK(hipMemcpy(got.data(), forward_logits(f), got.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  size_t at[kMaxSeqs] = {0}, rows = 0;
  for (int i = n - last; i < n; i++) {
    const int k = i % ns, p = i / ns;
    if (skip && p != len - 1) continue;
    CHECK(std::memcmp(&got[(size_t) (i - (n - last)) * V], &want[k][at[k]], V * 4) == 0);
    at[k] += V, rows++;
  }
  for (int k = 0; k < ns; k++) CHECK(at[k] == want[k].size() && rows > 0);
  std::printf("  sub-batches%s: %d x %d tokens in one batch of %d (%d sub-batches) in %.0f ms (%.0f tok/s): its %zu "
              "rows bit for bit each alone in steps of %d\n", skip ? " (the skip)" : "", ns, len, n,
              (n + MM_MAX_TOKENS - 1) / MM_MAX_TOKENS, ms, n * 1e3 / ms, rows, kSubStep);
  for (int k = 0; k < ns; k++) seq_destroy(s[k]);
}

// a verify's shape: at each position the token and two others after it in one batch, the two then undone
// (seq_rollback) - each row the token's own gets, and every later step, the bits of the sequence stepped one token at
// a time (the rejected tokens' window, compressor and Engram state rewritten before anything reads it)
static void rollback(Forward* f, size_t V, const std::vector<int32_t>& ids) {
  char err[FORWARD_ERR] = "";
  const int n = (int) ids.size(), junk = 2;
  Seq* a = seq_create(f, err);
  CHECK(a);
  std::vector<float> ref((size_t) n * V), got((size_t) n * V);
  for (int p = 0; p < n; p++) {
    CHECK(forward_step(f, a, ids[p], err));
    CHECK(hipMemcpy(&ref[(size_t) p * V], forward_logits(f), V * 4, hipMemcpyDeviceToHost) == hipSuccess);
  }
  seq_destroy(a);
  Seq* b = seq_create(f, err);
  CHECK(b);
  CHECK(!seq_rollback(b, 1));  // nothing stepped: nothing to undo
  int undone = 0;
  for (int p = 0; p < n; p++) {
    const int32_t t[3] = {ids[p], ids[(p + 7) % n], ids[(p + 13) % n]};
    Seq* const ss[3] = {b, b, b};
    CHECK(forward_batch(f, ss, t, 1 + junk, err));
    CHECK(hipMemcpy(&got[(size_t) p * V], forward_logits(f), V * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(seq_rollback(b, junk));
    undone += junk;
  }
  CHECK(got == ref);
  CHECK(!seq_rollback(b, SEQ_MAX_BACK + 1));
  CHECK(seq_prompt(b, (uint32_t) n) && !seq_rollback(b, 1));  // not before the prompt's end
  seq_destroy(b);
  std::printf("  rollback: %d positions each verified with %d tokens after it, those undone (%d in all) - every "
              "logit the bits of stepping one at a time\n", n, junk, undone);
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
  const std::string home = std::getenv("HOME") ? std::getenv("HOME") : "";
  FILE* idf = std::fopen((home + "/osmos-oracle/ids32.txt").c_str(), "r");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  if (!idf || which == TestModel::kNone || hipGetDeviceCount(&ndev) || !ndev) {
    std::printf("[test_batch_gpu] skipped: no model, ids or GPU\n");
    return 77;
  }
  std::vector<int32_t> all(32);
  for (auto& v : all) CHECK(std::fscanf(idf, "%d,", &v) == 1);
  std::fclose(idf);
  char ferr[FORWARD_ERR] = "";
  Gguf *g = nullptr, *eg = nullptr;
  Weights* w = test_weights(path, which, &g, &eg);
  CHECK(w);
  Run r{forward_create(w, w->geo.n_layers, kCtx, kBudget, ferr), (size_t) w->geo.n_vocab, {}, {}};
  CHECK(r.f);
  for (int k = 0; k < kSeqs; k++)  // 4 windows of the real ids: different sequences, real text
    r.ids.push_back(std::vector<int32_t>(all.begin() + 5 * k, all.begin() + 5 * k + kLen));
  Seq* s0 = seq_create(r.f, ferr);
  std::vector<int32_t> many(FORWARD_MAX_TOKENS + 1, 0);
  std::vector<Seq*> many_s(FORWARD_MAX_TOKENS + 1, s0);
  CHECK(!forward_batch(r.f, many_s.data(), many.data(), FORWARD_MAX_TOKENS + 1, ferr));
  CHECK(!forward_batch(r.f, many_s.data(), many.data(), 0, ferr));
  int32_t bad = (int32_t) r.V;
  CHECK(!forward_batch(r.f, many_s.data(), &bad, 1, ferr));
  seq_destroy(s0);
  std::printf("  refusals: %d tokens, 0 tokens, a token past the vocabulary\n", FORWARD_MAX_TOKENS + 1);

  rollback(r.f, r.V, all);
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
  // the decoder's reach (hip_skip.c): the skip's first sub-batch (its sequence's first 2,048) all before it, so that
  // sub-batch has none past dec
  int src = -1;
  for (int l = 0; l < w->geo.n_layers; l++)
    if (w->geo.kv_source[l]) src = l;
  const int reach = (w->geo.n_layers - 1 - src) * (w->geo.window - 1) + 1;
  constexpr int kSkipLen = 4600;
  CHECK(src >= 0 && MM_MAX_TOKENS - 1 + reach < kSkipLen - 1 && kSkipLen <= kCtx);
  sub_batches(r.f, r.V, all, 2, 1500, false);
  sub_batches(r.f, r.V, all, 1, kSkipLen, true);
  forward_destroy(r.f);
  weights_destroy(w);
  gguf_destroy(g);
  gguf_destroy(eg);
  std::printf("[test_batch_gpu] ok\n");
  return 0;
}
