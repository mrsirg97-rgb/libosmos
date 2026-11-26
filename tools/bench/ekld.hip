// ekld <first shard> <ids.txt> <step> <n>: any family's distance from a reference through the engine interface
// (osmos/run/engine.h) - kld's and q4x_kld's protocol for every pack: one sequence, the text's first n tokens `step` a
// osmos_batch (every row's logits: no prompt end), each row's log-softmax in double on the host. a step of at most
// MV_MAX_TOKENS takes a family's decode path, a longer one its prompt's: the two compared on the same text.
//   REF_OUT=path   the run is the reference: every scored row's log-probabilities written (F32, [rows][vocab])
//   REF=path       the run is a candidate: each scored row's KL(reference || run) over the vocabulary, the top-1
//                  agreement (the lowest id among equals)
// prints the mean NLL of each next token (and with REF the KL's mean, its standard error, median and 99th
// percentile). environment: BUDGET_GIB (a streaming family's expert budget, default 60).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
#include "osmos/run/engine.h"
}

#define CHECK(x)                                                         \
  do {                                                                   \
    if (!(x)) {                                                          \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

int main(int argc, char** argv) {
  if (argc < 5) return std::fprintf(stderr, "usage: ekld <first shard> <ids.txt> <step> <n>\n"), 2;
  std::vector<int32_t> ids;
  FILE* f = std::fopen(argv[2], "r");
  CHECK(f);
  for (int x; std::fscanf(f, "%d,", &x) == 1;) ids.push_back(x);
  std::fclose(f);
  const uint32_t S = (uint32_t) std::atoi(argv[3]), N = (uint32_t) std::atoi(argv[4]);
  CHECK(S > 0 && N > 1 && ids.size() >= N);
  char err[OSMOS_ERR] = "";
  const OsmosFiles files = {argv[1], nullptr, nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  if (!m) return std::fprintf(stderr, "ekld: %s\n", err), 1;
  const OsmosFacts facts = osmos_facts(m);
  OsmosOptions o = {};
  o.max_ctx = N + 64;
  o.max_tokens = S;
  o.budget = (uint64_t) (std::getenv("BUDGET_GIB") ? std::atof(std::getenv("BUDGET_GIB")) : 60.0) * (1ull << 30);
  Osmos* e = osmos_create(m, &o, err);
  if (!e) return std::fprintf(stderr, "ekld: %s\n", err), 1;
  OsmosSeq* s = oseq_create(e, err);
  if (!s) return std::fprintf(stderr, "ekld: %s\n", err), 1;
  FILE* lout = std::getenv("REF_OUT") ? std::fopen(std::getenv("REF_OUT"), "wb") : nullptr;
  FILE* lref = std::getenv("REF") ? std::fopen(std::getenv("REF"), "rb") : nullptr;
  CHECK(!std::getenv("REF_OUT") || lout);
  CHECK(!std::getenv("REF") || lref);
  const uint32_t V = facts.vocab;
  std::vector<float> lp(V), rp(V);
  std::vector<double> kls;
  double nll = 0, kl = 0;
  uint64_t scored = 0, agree = 0;
  std::printf("ekld: %s, %u tokens in steps of %u\n", facts.family, N, S);
  for (uint32_t at = 0; at + 1 < N; at += S) {
    const uint32_t n = N - at < S ? N - at : S;
    const std::vector<OsmosSeq*> seqs(n, s);  // (a step's sequence a token)
    if (!osmos_batch(e, seqs.data(), ids.data() + at, n, err)) return std::fprintf(stderr, "ekld: %s\n", err), 1;
    for (uint32_t r = 0; r < n && at + r + 1 < N; r++) {
      const float* x = osmos_logits(e, r);
      CHECK(x);
      double mx = -INFINITY, z = 0;
      for (uint32_t v = 0; v < V; v++) mx = std::fmax(mx, (double) x[v]);
      for (uint32_t v = 0; v < V; v++) z += std::exp((double) x[v] - mx);
      const double lz = mx + std::log(z);
      for (uint32_t v = 0; v < V; v++) lp[v] = (float) ((double) x[v] - lz);
      nll -= lp[(uint32_t) ids[at + r + 1]], scored++;
      if (lout) CHECK(std::fwrite(lp.data(), 4, V, lout) == V);
      if (lref) {
        CHECK(std::fread(rp.data(), 4, V, lref) == V);
        double k = 0;
        uint32_t ta = 0, tb = 0;
        for (uint32_t v = 0; v < V; v++) {
          k += std::exp((double) rp[v]) * ((double) rp[v] - (double) lp[v]);
          ta = rp[v] > rp[ta] ? v : ta, tb = lp[v] > lp[tb] ? v : tb;
        }
        kl += k, kls.push_back(k), agree += ta == tb;
      }
    }
  }
  std::printf("NLL: %.5f over %llu tokens\n", nll / scored, (unsigned long long) scored);
  if (lout) std::fclose(lout);
  if (lref) {
    std::sort(kls.begin(), kls.end());
    const double mean = kl / scored;
    double sd = 0;
    for (double k : kls) sd += (k - mean) * (k - mean);
    std::printf("KL(ref || run): mean %.5f (se %.5f), median %.5f, p99 %.5f; top-1 agreement %.2f%%\n", mean,
                std::sqrt(sd / scored / scored), kls[kls.size() / 2], kls[kls.size() * 99 / 100], 100.0 * agree / scored);
    std::fclose(lref);
  }
  oseq_destroy(s);
  osmos_destroy(e);
  osmos_close(m);
  return 0;
}
