// selprobe budget_gib step ids.txt P out.txt: the indexer's picks at one position of a long text (issue #47's trace) -
// the text's first P tokens read in `step`-token batches (their end told: seq_prompt, the prompt skip's exact path),
// then token P stepped with a probe on, every index layer's selected rows (its "sel": the compressed rows' indices,
// sorted) a line of out.txt: "<layer> <count> <row> <row> ...". two builds' (or two packs') picks compared tell where
// their selections part.
// environment: MODEL (a first shard), PICKS (forward_create_ranked's profile).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "bench_common.h"

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/model/weights.h"
#include "osmos/run/forward.h"
#include "osmos/run/engine.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

static std::vector<int32_t> read_list(const char* path) {
  std::vector<int32_t> v;
  FILE* f = std::fopen(path, "r");
  CHECK(f);
  for (long x; std::fscanf(f, "%ld,", &x) == 1;) v.push_back((int32_t) x);
  std::fclose(f);
  return v;
}

static void probe(const char* step, int layer, int, void* dev, uint64_t n, bool ints, void* arg) {
  if (std::strcmp(step, "sel") || !ints) return;
  std::vector<int32_t> v(n);
  if (n) CHECK(hipMemcpy(v.data(), dev, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  std::sort(v.begin(), v.end());
  FILE* o = (FILE*) arg;
  std::fprintf(o, "%d %llu", layer, (unsigned long long) n);
  for (int32_t x : v) std::fprintf(o, " %d", x);
  std::fprintf(o, "\n");
}

int main(int argc, char** argv) {
  if (argc != 6) return std::fprintf(stderr, "usage: selprobe budget_gib step ids.txt P out.txt (MODEL, PICKS)\n"), 2;
  const uint64_t budget = (uint64_t) (std::atof(argv[1]) * (double) (1ull << 30));
  const uint32_t step = (uint32_t) std::atoi(argv[2]), P = (uint32_t) std::atoi(argv[4]);
  const std::vector<int32_t> ids = read_list(argv[3]);
  CHECK(step > 0 && step <= FORWARD_MAX_TOKENS && P > 0 && ids.size() > P);
  const char* mp = std::getenv("MODEL");
  CHECK(mp);
  char err[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "", ferr[FORWARD_ERR] = "";
  Gguf* g = gguf_create(mp, err);
  if (!g) std::fprintf(stderr, "gguf_create: %s\n", err), std::_Exit(1);
  const WeightsOptions wo = {g, nullptr, nullptr};
  Weights* w = weights_create_with(&wo, werr);
  if (!w) std::fprintf(stderr, "weights_create: %s\n", werr), std::_Exit(1);
  ForwardOptions o = {};
  o.n_layers = (int) w->geo.n_layers, o.max_ctx = P + 1, o.budget = budget, o.max_tokens = step;
  std::vector<uint64_t> picks;
  if (const char* pp = std::getenv("PICKS")) {
    FILE* pf = std::fopen(pp, "rb");
    CHECK(pf);
    picks.resize((size_t) w->geo.n_layers * w->geo.n_experts);
    CHECK(std::fread(picks.data(), 8, picks.size(), pf) == picks.size());
    std::fclose(pf);
    o.picks = picks.data();
  }
  Forward* f = forward_create_with(w, &o, ferr);
  if (!f) std::fprintf(stderr, "forward_create_with: %s\n", ferr), std::_Exit(1);
  CHECK(forward_lane(f, 0, NULL, ferr));
  Seq* s = seq_create(f, ferr);
  CHECK(s && seq_prompt(s, P));
  std::vector<Seq*> ss(step, s);
  for (uint32_t at = 0; at < P; at += step) {
    const uint32_t k = P - at < step ? P - at : step;
    if (!forward_batch(f, ss.data(), ids.data() + at, k, ferr))
      std::fprintf(stderr, "step at %u: %s\n", at, ferr), std::_Exit(1);
    if (at / step % 16 == 0) std::fprintf(stderr, "read %u of %u\n", at + k, P);
  }
  FILE* out = std::fopen(argv[5], "w");
  CHECK(out);
  forward_probe(f, probe, out);
  if (!forward_step(f, s, ids[P], ferr)) std::fprintf(stderr, "the probed step: %s\n", ferr), std::_Exit(1);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  forward_probe(f, nullptr, nullptr);
  std::fclose(out);
  std::fprintf(stderr, "selprobe: position %u, picks in %s\n", P, argv[5]);
  seq_destroy(s);
  forward_destroy(f);
  weights_destroy(w);
  gguf_destroy(g);
  return 0;
}
