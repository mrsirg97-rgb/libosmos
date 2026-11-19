// LFM3's forward timed (osmos/lfm3/forward.h): lfm3_run <GGUF> [prompt=512] [step=512] [decode=64] - the wiki text's
// ids aren't needed: the prompt is a fixed pattern of ids (a speed run, not a quality one), then greedy decode of one
// token a step (the host's argmax over lfm3_logits); each phase's wall time and rate, the decoded ids' hash
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/lfm3/forward.h"
#include "osmos/model/gguf.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: failed: %s (%s)\n", __FILE__, __LINE__, #x, err); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  char err[LFM3_ERR] = "", gerr[GGUF_ERR] = "";
  if (argc < 2) return std::fprintf(stderr, "usage: lfm3_run <GGUF> [prompt=512] [step=512] [decode=64] [file: the BF16 head]\n"), 2;
  const uint32_t P = argc > 2 ? (uint32_t) atoi(argv[2]) : 512, S = argc > 3 ? (uint32_t) atoi(argv[3]) : 512;
  const uint32_t Dn = argc > 4 ? (uint32_t) atoi(argv[4]) : 64;
  const double t0 = now();
  Gguf* g = gguf_create(argv[1], gerr);
  if (!g) return std::fprintf(stderr, "%s\n", gerr), 1;
  Lfm3Weights* w = lfm3_weights_create(g, err);
  CHECK(w);
  const bool file_head = argc > 5 && !strcmp(argv[5], "file");  // the head as the file's BF16, not its Q8_0 copy
  const Lfm3Options o = {S, P + Dn + 1, 1, file_head ? LFM3_HEAD_FILE : LFM3_HEAD_Q8_0, nullptr, 0, nullptr, 0, 0, 0};
  Lfm3Forward* f = lfm3_forward_create(w, &o, err);
  CHECK(f);
  Lfm3Seq* s = lfm3_seq_create(f, err);
  CHECK(s);
  const uint32_t V = (uint32_t) w->geo.n_vocab;
  std::vector<int32_t> ids(P);
  for (uint32_t i = 0; i < P; i++) ids[i] = (int32_t) ((i * 7919u + 13u) % 120000u);
  std::printf("lfm3_run: loaded in %.2f s; prompt %u in steps of %u, decode %u\n", now() - t0, P, S, Dn);
  const double p0 = now();
  for (uint32_t at = 0; at < P; at += S) {
    const uint32_t n = P - at < S ? P - at : S;
    CHECK(lfm3_step(f, s, ids.data() + at, n, 1, err));
  }
  const double p1 = now();
  auto pick = [&]() {
    const float* l = lfm3_logits(f);
    int32_t b = 0;
    for (uint32_t i = 1; i < V; i++) b = l[i] > l[b] ? (int32_t) i : b;
    return b;
  };
  int32_t next = pick();
  uint64_t hash = 1469598103934665603ull;
  const double d0 = now();
  for (uint32_t i = 0; i < Dn; i++) {
    CHECK(lfm3_step(f, s, &next, 1, 1, err));
    next = pick();
    hash = (hash ^ (uint64_t) next) * 1099511628211ull;
  }
  const double d1 = now();
  std::printf("prompt: %u tokens in %.3f s (%.1f tok/s)\ndecode: %u tokens in %.3f s (%.2f tok/s, %.2f ms a token); "
              "ids hash %016llx\n", P, p1 - p0, P / (p1 - p0), Dn, d1 - d0, Dn / (d1 - d0), 1000 * (d1 - d0) / Dn,
              (unsigned long long) hash);
  lfm3_seq_destroy(s), lfm3_forward_destroy(f), lfm3_weights_destroy(w), gguf_destroy(g);
  return 0;
}
