// LSA's read against dense attention (osmos/lfm3/forward.h's lsa option): lfm3_lsa_check <GGUF> <maps> <text> [P=16384]
// [D=64] - two forwards on the same weights, one dense, one with the maps; the text's first P tokens a prompt in
// each (dense in both: the read's buckets made as its blocks close), then its next D tokens one decode step at a time
// (teacher-forced: the text's own) - each step's logits compared: the share whose top token is dense's (the card's
// "same top token as dense"), the KL of LSA's from dense's, and each forward's mean NLL of the text's next token
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
#include "osmos/lfm3/forward.h"
#include "osmos/model/gguf.h"
#include "osmos/text/tokenizer.h"
}

#define CHECK(x)                                                                         \
  do {                                                                                   \
    if (!(x)) {                                                                          \
      std::fprintf(stderr, "%s:%d: failed: %s (%s)\n", __FILE__, __LINE__, #x, err);   \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double logsumexp(const float* x, uint32_t V) {
  double m = -INFINITY, z = 0;
  for (uint32_t i = 0; i < V; i++) m = std::fmax(m, x[i]);
  for (uint32_t i = 0; i < V; i++) z += std::exp(x[i] - m);
  return m + std::log(z);
}

int main(int argc, char** argv) {
  char err[LFM3_ERR] = "", gerr[GGUF_ERR] = "", terr[TOKENIZER_ERR] = "";
  if (argc < 4) return std::fprintf(stderr, "usage: lfm3_lsa_check <GGUF> <maps> <text> [P=16384] [D=64]\n"), 2;
  const uint32_t P = argc > 4 ? (uint32_t) atoi(argv[4]) : 16384, D = argc > 5 ? (uint32_t) atoi(argv[5]) : 64;
  Gguf* g = gguf_create(argv[1], gerr);
  if (!g) return std::fprintf(stderr, "%s\n", gerr), 1;
  Tokenizer* tok = tokenizer_create(g, terr);
  if (!tok) return std::fprintf(stderr, "%s\n", terr), 1;
  FILE* f = std::fopen(argv[3], "rb");
  if (!f) return std::fprintf(stderr, "no %s\n", argv[3]), 1;
  std::string text;
  for (int c; (c = std::fgetc(f)) != EOF;) text.push_back((char) c);
  std::fclose(f);
  const size_t want = tokenizer_encode(tok, text.data(), text.size(), false, nullptr, 0);
  std::vector<int32_t> ids(want);
  tokenizer_encode(tok, text.data(), text.size(), false, ids.data(), want);
  if (ids.size() < (size_t) P + D + 1)
    return std::fprintf(stderr, "the text's %zu tokens under P + D + 1 = %u\n", ids.size(), P + D + 1), 1;
  Lfm3Weights* w = lfm3_weights_create(g, err);
  CHECK(w);
  const uint32_t S = 2048 < P ? 2048 : P, V = (uint32_t) w->geo.n_vocab;
  const Lfm3Options od = {S, P + D + 1, 1, LFM3_HEAD_Q8_0, nullptr, 0}, ol = {S, P + D + 1, 1, LFM3_HEAD_Q8_0, argv[2], 0};
  Lfm3Forward *fd = lfm3_forward_create(w, &od, err), *fl = nullptr;
  CHECK(fd);
  fl = lfm3_forward_create(w, &ol, err);
  CHECK(fl);
  Lfm3Seq *sd = lfm3_seq_create(fd, err), *sl = lfm3_seq_create(fl, err);
  CHECK(sd && sl);
  const double t0 = now();
  for (uint32_t at = 0; at < P; at += S) {
    const uint32_t n = P - at < S ? P - at : S;
    CHECK(lfm3_step(fd, sd, ids.data() + at, n, 1, err) && lfm3_step(fl, sl, ids.data() + at, n, 1, err));
  }
  const double t1 = now();
  std::vector<float> ld(V);
  uint32_t same = 0;
  double kl_sum = 0, kl_max = 0, nll_d = 0, nll_l = 0, ms_d = 0, ms_l = 0;
  for (uint32_t i = 0; i < D; i++) {
    const int32_t x = ids[P + i], next = ids[P + i + 1];
    double a = now();
    CHECK(lfm3_step(fd, sd, &x, 1, 1, err));
    ms_d += now() - a;
    std::copy(lfm3_logits(fd), lfm3_logits(fd) + V, ld.begin());
    a = now();
    CHECK(lfm3_step(fl, sl, &x, 1, 1, err));
    ms_l += now() - a;
    const float* ll = lfm3_logits(fl);
    const double zd = logsumexp(ld.data(), V), zl = logsumexp(ll, V);
    double k = 0;
    uint32_t bd = 0, bl = 0;
    for (uint32_t v = 0; v < V; v++) {
      const double pd = ld[v] - zd, pl = ll[v] - zl;
      k += std::exp(pd) * (pd - pl);
      bd = ld[v] > ld[bd] ? v : bd, bl = ll[v] > ll[bl] ? v : bl;
    }
    same += bd == bl, kl_sum += k, kl_max = std::fmax(kl_max, k);
    nll_d += zd - ld[next], nll_l += zl - ll[next];
  }
  std::printf("lfm3_lsa_check: a prompt of %u (%.1f s, both forwards), %u decode steps teacher-forced\n", P, t1 - t0, D);
  std::printf("  the top token dense's: %u of %u (%.0f%%); KL(dense || LSA) mean %.3g, max %.3g nats\n", same, D,
              100.0 * same / D, kl_sum / D, kl_max);
  std::printf("  the text's next token, mean NLL: dense %.3f, LSA %.3f; a decode step: dense %.2f ms, LSA %.2f ms\n",
              nll_d / D, nll_l / D, 1e3 * ms_d / D, 1e3 * ms_l / D);
  lfm3_seq_destroy(sd), lfm3_seq_destroy(sl), lfm3_forward_destroy(fd), lfm3_forward_destroy(fl);
  lfm3_weights_destroy(w), tokenizer_destroy(tok), gguf_destroy(g);
  return 0;
}
