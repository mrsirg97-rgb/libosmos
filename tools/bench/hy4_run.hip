// hy4_run - Hy4-flash's forward timed (osmos/hy4/forward.h): a prompt of P ids (the oracle's wiki ids, or 1 .. P)
// in steps of S, then D greedy decode steps of one token - each phase's wall time and rate, and the decode's tokens'
// hash (two runs the same model the same ids). under rocprofv3 --kernel-trace --stats each kernel's share.
//
//   hy4_run <GGUF> [P=2048] [S=2048] [D=32]
//   IDS=path   the prompt's ids: a file of int32 (the first P), else 1000 + i * 7919 % 50000
//   DFLY=path  DFly2 (the release's model-00148-of-00148.safetensors): the decode in drafted rounds - the drafts and
//              the sampled token one decode step, the target's greedy picks keeping the leading drafts - its rate,
//              its drafts accepted a round; the tokens the plain decode's (the same hash). CUM=x: a round's drafts
//              kept while their probabilities' product is at least x (the engine's cum_min)
//   NLL=1      the prompt's every row kept: the mean NLL of each next id (teacher-forced), a cache's or a math's cost
//              over the whole text (KV_F32: the reference's F32 cache against the engine's Q8_0)
#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/hy4/forward.h"
#include "osmos/model/gguf.h"
}

#define CHECK(x)                                                           \
  do {                                                                     \
    if (!(x)) {                                                            \
      std::fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  if (argc < 2) return std::fprintf(stderr, "usage: hy4_run <GGUF> [prompt=2048] [step=2048] [decode=32]\n"), 2;
  const uint32_t P = argc > 2 ? (uint32_t) atoi(argv[2]) : 2048, S = argc > 3 ? (uint32_t) atoi(argv[3]) : 2048;
  const uint32_t D = argc > 4 ? (uint32_t) atoi(argv[4]) : 32;
  char err[HY4_ERR] = "", gerr[GGUF_ERR] = "";
  Gguf* g = gguf_create(argv[1], gerr);
  CHECK(g);
  Hy4Weights* w = hy4_weights_create(g, err);
  if (!w) std::fprintf(stderr, "%s\n", err);
  CHECK(w);
  std::vector<int> fds(g->n_files);
  for (int i = 0; i < g->n_files; i++) CHECK((fds[i] = open(g->paths[i], O_RDONLY)) >= 0);
  std::vector<int32_t> ids(P);
  for (uint32_t i = 0; i < P; i++) ids[i] = (int32_t) (1000 + i * 7919 % 50000);
  if (const char* ip = getenv("IDS")) {
    FILE* fi = std::fopen(ip, "rb");
    CHECK(fi && std::fread(ids.data(), 4, P, fi) == P);
    std::fclose(fi);
  }
  const char* dfly = getenv("DFLY");
  const double t0 = now();
  const bool nll_mode = getenv("NLL") != nullptr;
  const Hy4Options o = {S, P + D + 2 * HY4_DFLY_DRAFTS + HY4_MAX_BACK,
                        nll_mode ? S : dfly ? (uint32_t) HY4_DFLY_DRAFTS + 1 : 1u,
                       getenv("KV_F32") != nullptr};  // (KV_F32: the reference's F32 cache; else the engine's Q8_0)
  Hy4Forward* f = hy4_forward_create(w, fds.data(), &o, err);
  if (!f) std::fprintf(stderr, "%s\n", err);
  CHECK(f);
  Hy4Seq* s = hy4_seq_create(f, err);
  CHECK(s);
  if (dfly && !hy4_dfly_load(f, dfly, !getenv("DFLY_BF16"), err)) std::fprintf(stderr, "%s\n", err), std::exit(1);
  std::printf("hy4_run: loaded in %.1f s; prompt %u in steps of %u, decode %u\n", now() - t0, P, S, D);
  const double t1 = now();
  uint32_t n = 0;
  double nll = 0;
  uint64_t scored = 0;
  const uint32_t V = (uint32_t) w->geo.n_vocab;
  for (uint32_t at = 0; at < P; at += S) {
    n = P - at < S ? P - at : S;
    if (!hy4_step(f, s, ids.data() + at, n, nll_mode ? n : 1, err)) std::fprintf(stderr, "%s\n", err), std::exit(1);
    for (uint32_t r = 0; nll_mode && r < n && at + r + 1 < P; r++) {
      const float* x = hy4_logits(f) + (uint64_t) r * V;
      double m = -INFINITY, z = 0;
      for (uint32_t v = 0; v < V; v++) m = std::fmax(m, x[v]);
      for (uint32_t v = 0; v < V; v++) z += std::exp(x[v] - m);
      nll -= x[ids[at + r + 1]] - m - std::log(z), scored++;
    }
  }
  if (nll_mode) std::printf("NLL: %.5f over %llu tokens (%s cache)\n", nll / scored, (unsigned long long) scored,
                            getenv("KV_F32") ? "F32" : "Q8_0");
  const double t2 = now();
  int32_t next = 0;
  uint64_t hash = 1469598103934665603ull;
  CHECK(hy4_greedy(f, n - 1, 1, &next, err));  /* the step's last row: its one kept */
  const double t3 = now();
  uint32_t rounds = 0, accepted = 0, drafted = 0;
  for (uint32_t d = 0; d < D;) {
    if (!dfly) {
      CHECK(hy4_step(f, s, &next, 1, 1, err) && hy4_greedy(f, 0, 1, &next, err));
      hash = (hash ^ (uint64_t) next) * 1099511628211ull, d++;
      continue;
    }
    int32_t tok[1 + HY4_DFLY_DRAFTS], pick[1 + HY4_DFLY_DRAFTS];
    float conf[HY4_DFLY_DRAFTS];
    const int k = hy4_seq_draft(f, s, next, HY4_DFLY_DRAFTS, 0.0f, getenv("CUM") ? (float) atof(getenv("CUM")) : 0.0f,
                                tok + 1, conf, err);
    CHECK(k >= 0);
    tok[0] = next;
    const uint32_t len = 1 + (uint32_t) k;
    CHECK(hy4_step_runs(f, &s, &len, 1, tok, err) && hy4_greedy(f, 0, len, pick, err));
    int a = 0;
    while (a < k && tok[1 + a] == pick[a]) a++;
    if (k > a) CHECK(hy4_seq_rollback(s, (uint32_t) (k - a)));
    for (int j = 0; j <= a && d < D; j++, d++) hash = (hash ^ (uint64_t) pick[j]) * 1099511628211ull;
    next = pick[a];
    rounds++, accepted += (uint32_t) a, drafted += (uint32_t) k;
  }
  const double t4 = now();
  if (dfly)
    std::printf("drafted: %u rounds, %u drafts, %u accepted (%.2f a round, %.0f%%)\n", rounds, drafted, accepted,
                (double) accepted / rounds, 100.0 * accepted / (drafted ? drafted : 1));
  std::printf("prompt: %u tokens in %.2f s (%.1f tok/s)\ndecode: %u tokens in %.2f s (%.2f tok/s, %.1f ms a token); "
              "ids hash %016llx\n", P, t2 - t1, P / (t2 - t1), D, t4 - t3, D / (t4 - t3), 1000 * (t4 - t3) / D,
              (unsigned long long) hash);
  hy4_seq_destroy(s);
  hy4_forward_destroy(f);
  hy4_weights_destroy(w);
  gguf_destroy(g);
  return 0;
}
