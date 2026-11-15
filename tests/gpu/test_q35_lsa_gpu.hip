// the q35 forward under latent sparse attention against itself dense, lsa.py's quality protocol (incarnate qwen/lsa.py
// quality: a held document's prompt read dense under YaRN x40, its last tokens scored): the document's tokens
// (Q35_IDS, int32) less the tail read in prompt steps, then the tail (Q35_TAIL, 256) a token a step - once dense
// (q35_forward_lsa NULL: the rotation alone), once under the read (Q35_LSA_MAPS) - each tail token's NLL of the next,
// the read's KL from dense and their top-1. the model Q35_MODEL; skipped (77) without the four
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/q35/forward.h"
}

#define CHECK(c)                                                                  \
  do {                                                                            \
    if (!(c)) {                                                                   \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

// the tail's log-probabilities (tail x V, host), the prompt in steps of `chunk`
static std::vector<float> run(Q35Weights* w, const std::vector<int32_t>& ids, uint32_t tail, const char* maps,
                              uint32_t chunk) {
  char err[Q35_ERR] = "";
  const uint32_t T = (uint32_t) ids.size(), V = (uint32_t) w->geo.n_vocab;
  Q35Forward* f = q35_forward_create(w, chunk, T + 8, err);
  CHECK(f || (std::fprintf(stderr, "%s\n", err), false));
  CHECK(q35_forward_lsa(f, maps, 40.0f, err) || (std::fprintf(stderr, "%s\n", err), false));
  Q35Seq* s = q35_seq_create(f, err);
  CHECK(s || (std::fprintf(stderr, "%s\n", err), false));
  const uint32_t P = T - tail;
  for (uint32_t t0 = 0; t0 < P; t0 += chunk)
    CHECK(q35_step(f, s, ids.data() + t0, P - t0 < chunk ? P - t0 : chunk, 1, err) || (std::fprintf(stderr, "%s\n", err), false));
  std::vector<float> lp((size_t) tail * V);
  for (uint32_t i = 0; i < tail; i++) {  // token P - 1 + i's logits came with the step before: re-read below
    CHECK(q35_step(f, s, ids.data() + P + i, 1, 1, err) || (std::fprintf(stderr, "%s\n", err), false));
    const float* lo = q35_logits(f);
    double mx = -1e30, z = 0;
    for (uint32_t v = 0; v < V; v++) mx = std::fmax(mx, lo[v]);
    for (uint32_t v = 0; v < V; v++) z += std::exp(lo[v] - mx);
    for (uint32_t v = 0; v < V; v++) lp[(size_t) i * V + v] = (float) (lo[v] - mx - std::log(z));
  }
  q35_seq_destroy(s);
  q35_forward_destroy(f);
  return lp;
}

int main() {
  std::printf("[test_q35_lsa_gpu] start\n");
  const char *mp = std::getenv("Q35_MODEL"), *ip = std::getenv("Q35_IDS"), *lp = std::getenv("Q35_LSA_MAPS");
  int devs = 0;
  if (!mp || !ip || !lp || hipGetDeviceCount(&devs) != hipSuccess || !devs) {
    std::printf("[test_q35_lsa_gpu] skipped: no Q35_MODEL / Q35_IDS / Q35_LSA_MAPS, or no GPU\n");
    return 77;
  }
  FILE* fi = std::fopen(ip, "rb");
  CHECK(fi);
  std::fseek(fi, 0, SEEK_END);
  std::vector<int32_t> ids((size_t) std::ftell(fi) / 4);
  std::fseek(fi, 0, SEEK_SET);
  CHECK(std::fread(ids.data(), 4, ids.size(), fi) == ids.size());
  std::fclose(fi);
  const uint32_t tail = std::getenv("Q35_TAIL") ? (uint32_t) std::atoi(std::getenv("Q35_TAIL")) : 256;
  char err[Q35_ERR] = "";
  Gguf* g = gguf_create(mp, err);
  CHECK(g);
  Q35Weights* w = q35_weights_create(g, err);
  CHECK(w || (std::fprintf(stderr, "%s\n", err), false));
  const uint32_t V = (uint32_t) w->geo.n_vocab, T = (uint32_t) ids.size();
  const std::vector<float> dense = run(w, ids, tail, nullptr, 2048);
  const std::vector<float> sparse = run(w, ids, tail, lp, 2048);
  double nd = 0, ns = 0, kl = 0;
  uint32_t agree = 0, scored = 0;
  for (uint32_t i = 0; i + 1 < tail; i++) {  // tail token i predicts token P + i + 1
    const int32_t y = ids[T - tail + i + 1];
    const float *a = &dense[(size_t) i * V], *b = &sparse[(size_t) i * V];
    nd -= a[y], ns -= b[y], scored++;
    uint32_t ia = 0, ib = 0;
    double k = 0;
    for (uint32_t v = 0; v < V; v++) {
      if (a[v] > a[ia]) ia = v;
      if (b[v] > b[ib]) ib = v;
      k += std::exp((double) a[v]) * ((double) a[v] - b[v]);
    }
    kl += k, agree += ia == ib;
  }
  std::printf("  %u tokens, the last %u a step: dense (YaRN x40) NLL %.4f, under the read %.4f; KL %.5f, top-1 %u / %u\n",
              T, tail, nd / scored, ns / scored, kl / scored, agree, scored);
  q35_weights_destroy(w);
  gguf_destroy(g);
  std::printf("[test_q35_lsa_gpu] ok\n");
  return 0;
}
