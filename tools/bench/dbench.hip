// dbench budget_gib max_tokens n_new warmup.txt q1.txt [q2.txt ...]: libsubstrate's decode as maya's --bench runs
// it - one forward (the lane), an answer of 64 tokens to the warm-up prompt (keep it unlike the questions: the
// resident set follows what it answers), then each question's prompt and n_new greedy tokens; prints the prompt's
// tok/s (past 256 tokens), the answer's tok/s and a hash of its tokens (the same hash: the same tokens). a prompt
// file is token ids, comma separated (tools/bench/prompts.py writes them).
// environment: MODEL (a first shard: unsloth's UD-IQ1_M, say; ENGRAM its Engram file where it has one) or MODEL_DIR
// (ds1017's, default ~/models/ds1017), PICKS (forward_create_ranked's profile: the model's
// substrate-picks.bin), GREEDY=1 (the picks forward_greedy's), DRAFT=dir (DSpark: dspark.bin, dspark.json),
// CPU_THREADS (forward_lane's CPU half), MAX_CTX (default 4096), IDS=file (the undrafted answers' token ids).
// build: hipcc -O2 -Iinclude --offload-arch=gfx1151 tools/bench/dbench.hip -Llib -lsubstrate_hip -lsubstrate
//        -lhsa-runtime64 -o bin/dbench   (after `make lib/libsubstrate.a lib/libsubstrate_hip.a`)
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
#include "substrate/model/gguf.h"
#include "substrate/model/weights.h"
#include "substrate/run/forward.h"
#include "substrate/run/drafter.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

static std::vector<int32_t> read_ids(const char* path) {
  std::vector<int32_t> v;
  FILE* f = std::fopen(path, "r");
  CHECK(f);
  int x;
  while (std::fscanf(f, "%d,", &x) == 1) v.push_back(x);
  std::fclose(f);
  return v;
}

static int32_t argmax(const std::vector<float>& l) {
  size_t b = 0;
  for (size_t i = 1; i < l.size(); i++)
    if (l[i] > l[b]) b = i;
  return (int32_t) b;
}

static Drafter* g_draft;        // DRAFT=dir: the answers drafted (DSpark) and verified
static bool g_greedy;           // GREEDY=1: the picks forward_greedy's (on the GPU), not the logits copied and scanned
static uint32_t g_max_tokens;   // a step's most tokens (the prompt's steps)
static FILE* g_ids;             // IDS=file: each undrafted answer's token ids, a line each (to read the text back)

static int32_t pick(Forward* f, std::vector<float>& lg, uint32_t V, uint32_t row) {
  if (g_greedy) {
    char e[FORWARD_ERR] = "";
    int32_t id = -1;
    if (!forward_greedy(f, row, 1, &id, e)) std::fprintf(stderr, "greedy: %s\n", e), std::_Exit(1);
    return id;
  }
  CHECK(hipMemcpy(lg.data(), forward_logits(f) + (size_t) row * V, V * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return argmax(lg);
}
static int g_drafted, g_accepted, g_rounds;
static double g_draft_ms, g_verify_ms; // a drafted answer's time drafting, and verifying (with its picks)
constexpr float kConfMin = 0.5f, kCumMin = 0.5f;  // maya's defaults (as test_drafter_gpu)
constexpr int kRing = 128;                        // the drafter's window: a prompt's last rows seeded

// one answer: the prompt in one batch, then n greedy tokens (stepped, or drafted and verified); returns tok/s of them
static double answer(Forward* f, const Weights* w, const std::vector<int32_t>& ids, int n, uint64_t* h) {
  char err[FORWARD_ERR] = "";
  Seq* s = seq_create(f, err);
  CHECK(s);
  const uint32_t P = (uint32_t) ids.size(), V = w->geo.n_vocab;
  CHECK(seq_prompt(s, P));
  std::vector<Seq*> ss(P, s);
  const auto p0 = std::chrono::steady_clock::now();
  for (uint32_t at = 0; at < P;) {  // the prompt in steps of the forward's max_tokens
    const uint32_t c = P - at < g_max_tokens ? P - at : g_max_tokens;
    if (!forward_batch(f, ss.data(), ids.data() + at, c, err))
      std::fprintf(stderr, "prompt: %s\n", err), std::_Exit(1);
    at += c;
  }
  CHECK(hipDeviceSynchronize() == hipSuccess);
  const double ps = std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
  if (P > 256) std::printf("prompt %u tokens in %.0f ms (%.0f tok/s)\n", P, ps * 1e3, P / ps);
  std::vector<float> lg(V);
  // the last step's row: a step past MM_MAX_TOKENS keeps its last sub-batch's rows (forward_logits)
  const uint32_t last = P - (P - 1) / g_max_tokens * g_max_tokens;
  int32_t t = pick(f, lg, V, (last - 1) % MM_MAX_TOKENS);
  const auto t0 = std::chrono::steady_clock::now();
  if (g_draft) {
    char err2[FORWARD_ERR] = "";
    CHECK(drafter_seed(g_draft, s, (int32_t) P > kRing ? (int32_t) P - kRing : 0, (int32_t) P, err2));
    int32_t pos = (int32_t) P, dr[DRAFTER_BLOCK];
    float cf[DRAFTER_BLOCK];
    int made = 0;
    while (made < n) {
      const auto d0 = std::chrono::steady_clock::now();
      const int k = drafter_draft(g_draft, s, t, kConfMin, kCumMin, dr, cf, err2);
      const auto d1 = std::chrono::steady_clock::now();
      if (k < 0) std::fprintf(stderr, "draft: %s\n", err2), std::_Exit(1);
      std::vector<int32_t> toks{t};
      toks.insert(toks.end(), dr, dr + k);
      std::vector<Seq*> ss(toks.size(), s);
      if (!forward_batch(f, ss.data(), toks.data(), (uint32_t) toks.size(), err2))
        std::fprintf(stderr, "verify: %s\n", err2), std::_Exit(1);
      int m = 0, next = -1;
      int32_t picks[DRAFTER_BLOCK + 1];
      if (g_greedy && !forward_greedy(f, 0, (uint32_t) k + 1, picks, err2))
        std::fprintf(stderr, "greedy: %s\n", err2), std::_Exit(1);
      for (;; m++) {
        const int am = g_greedy ? picks[m] : pick(f, lg, V, (uint32_t) m);
        if (m == k || am != dr[m]) {
          next = am;
          break;
        }
      }
      CHECK(seq_rollback(s, (uint32_t) (k - m)) && drafter_seed(g_draft, s, pos, pos + m + 1, err2));
      const auto d2 = std::chrono::steady_clock::now();
      g_draft_ms += std::chrono::duration<double, std::milli>(d1 - d0).count();
      g_verify_ms += std::chrono::duration<double, std::milli>(d2 - d1).count();
      for (int i = -1; i < m && made < n; i++, made++)
        *h = (*h ^ (uint64_t) (i < 0 ? t : dr[i])) * 1099511628211ull;
      t = next, pos += m + 1, g_drafted += k, g_accepted += m, g_rounds++;
    }
  }
  for (int i = 0; !g_draft && i < n; i++) {
    *h = (*h ^ (uint64_t) t) * 1099511628211ull;
    if (g_ids) std::fprintf(g_ids, i ? ",%d" : "%d", t);
    if (!forward_step(f, s, t, err)) std::fprintf(stderr, "step %d: %s\n", i, err), std::_Exit(1);
    t = pick(f, lg, V, 0);
  }
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (g_ids) std::fprintf(g_ids, "\n"), std::fflush(g_ids);
  seq_destroy(s);
  return n / sec;
}

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: dbench budget_gib max_tokens n_new warmup.txt q1.txt [q2.txt ...]\n");
    return 2;
  }
  const uint64_t budget = (uint64_t) (std::atof(argv[1]) * (1ull << 30));
  const uint32_t max_tokens = (uint32_t) std::atoi(argv[2]);
  const int n_new = std::atoi(argv[3]);
  const char* md = std::getenv("MODEL_DIR");
  const std::string home = std::getenv("HOME") ? std::getenv("HOME") : "",
                    mdir = md ? std::string(md) + "/" : home + "/models/ds1017/";
  char err[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "", ferr[FORWARD_ERR] = "";
  // MODEL: a model's first shard (unsloth's UD: its Engram its own, ENGRAM unset); else ds1017's in MODEL_DIR
  const char* mp = std::getenv("MODEL");
  const char* ep = std::getenv("ENGRAM");
  const std::string model = mp ? mp : mdir + "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf",
                    engram = ep ? ep : mp ? "" : mdir + "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf";
  Gguf* g = gguf_create(model.c_str(), err);
  if (!g) std::fprintf(stderr, "%s: %s\n", model.c_str(), err), std::_Exit(1);
  Gguf* eg = engram.empty() ? nullptr : gguf_create(engram.c_str(), err);
  CHECK(engram.empty() || eg);
  Weights* w = weights_create(g, eg, werr);
  if (!w) std::fprintf(stderr, "weights_create: %s\n", werr), std::_Exit(1);
  ForwardOptions o = {};
  o.n_layers = (int) w->geo.n_layers;
  o.max_ctx = std::getenv("MAX_CTX") ? (uint32_t) std::atoi(std::getenv("MAX_CTX")) : 4096;
  o.budget = budget;
  o.max_tokens = max_tokens;
  g_max_tokens = max_tokens;
  std::vector<uint64_t> picks;
  if (const char* pp = std::getenv("PICKS")) {  // forward_create_ranked's counts ([layer][expert] uint64)
    FILE* pf = std::fopen(pp, "rb");
    CHECK(pf);
    picks.resize((size_t) w->geo.n_layers * w->geo.n_experts);
    CHECK(std::fread(picks.data(), 8, picks.size(), pf) == picks.size());
    std::fclose(pf);
    o.picks = picks.data();
    std::printf("ranked by %s\n", pp);
  }
  const auto l0 = std::chrono::steady_clock::now();
  Forward* f = forward_create_with(w, &o, ferr);
  if (!f) std::fprintf(stderr, "forward_create_with: %s\n", ferr), std::_Exit(1);
  const int cpu_threads = std::getenv("CPU_THREADS") ? std::atoi(std::getenv("CPU_THREADS")) : 0;
  if (!forward_lane(f, cpu_threads, ferr)) std::fprintf(stderr, "forward_lane: %s\n", ferr), std::_Exit(1);
  g_greedy = std::getenv("GREEDY") != nullptr;
  if (const char* ip = std::getenv("IDS")) CHECK((g_ids = std::fopen(ip, "w")));
  if (const char* dd = std::getenv("DRAFT")) {
    g_draft = drafter_create(f, dd, ferr);
    if (!g_draft) std::fprintf(stderr, "drafter_create: %s\n", ferr), std::_Exit(1);
  }
  std::printf("loaded %.0fs\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - l0).count());
  uint64_t h = 1469598103934665603ull;
  answer(f, w, read_ids(argv[4]), 64, &h);  // the warm-up, as maya's bench
  for (int i = 5; i < argc; i++) {
    h = 1469598103934665603ull;
    g_drafted = g_accepted = g_rounds = 0, g_draft_ms = g_verify_ms = 0;
    const double tps = answer(f, w, read_ids(argv[i]), n_new, &h);
    uint64_t gs, gg;
    forward_guesses(f, &gs, &gg);
    std::printf("%.1f tok/s (%d tokens) stream %016llx; so far streamed picks %llu, guessed %llu, on the CPU %llu, "
                "swaps %llu; drafted %d accepted %d in %d rounds (%.0f ms drafting, %.0f verifying)\n",
                tps, n_new, (unsigned long long) h, (unsigned long long) gs, (unsigned long long) gg,
                (unsigned long long) forward_cpu_picks(f), (unsigned long long) forward_swaps(f), g_drafted,
                g_accepted, g_rounds, g_draft_ms, g_verify_ms);
    std::fflush(stdout);
  }
  drafter_destroy(g_draft);
  forward_destroy(f);
  return 0;
}
