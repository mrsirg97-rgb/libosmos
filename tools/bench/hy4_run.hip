// hy4_run - Hy4-flash's forward timed (osmos/hy4/forward.h): a prompt of P ids (the oracle's wiki ids, or 1 .. P)
// in steps of S, then D greedy decode steps of one token - each phase's wall time and rate, and the decode's tokens'
// hash (two runs the same model the same ids). under rocprofv3 --kernel-trace --stats each kernel's share.
//
//   hy4_run <GGUF> [P=2048] [S=2048] [D=32]
//   IDS=path  a fixture of the oracle's (oracle/hy4_forward.py: its "ids") for the prompt's ids
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
  const double t0 = now();
  const Hy4Options o = {S, P + D + HY4_MAX_BACK, 1};
  Hy4Forward* f = hy4_forward_create(w, fds.data(), &o, err);
  if (!f) std::fprintf(stderr, "%s\n", err);
  CHECK(f);
  Hy4Seq* s = hy4_seq_create(f, err);
  CHECK(s);
  std::printf("hy4_run: loaded in %.1f s; prompt %u in steps of %u, decode %u\n", now() - t0, P, S, D);
  const double t1 = now();
  uint32_t n = 0;
  for (uint32_t at = 0; at < P; at += S) {
    n = P - at < S ? P - at : S;
    if (!hy4_step(f, s, ids.data() + at, n, 1, err)) std::fprintf(stderr, "%s\n", err), std::exit(1);
  }
  const double t2 = now();
  int32_t next = 0;
  uint64_t hash = 1469598103934665603ull;
  CHECK(hy4_greedy(f, n - 1, 1, &next, err));  /* the step's last row: its one kept */
  const double t3 = now();
  for (uint32_t d = 0; d < D; d++) {
    CHECK(hy4_step(f, s, &next, 1, 1, err) && hy4_greedy(f, 0, 1, &next, err));
    hash = (hash ^ (uint64_t) next) * 1099511628211ull;
  }
  const double t4 = now();
  std::printf("prompt: %u tokens in %.2f s (%.1f tok/s)\ndecode: %u tokens in %.2f s (%.2f tok/s, %.1f ms a token); "
              "ids hash %016llx\n", P, t2 - t1, P / (t2 - t1), D, t4 - t3, D / (t4 - t3), 1000 * (t4 - t3) / D,
              (unsigned long long) hash);
  hy4_seq_destroy(s);
  hy4_forward_destroy(f);
  hy4_weights_destroy(w);
  gguf_destroy(g);
  return 0;
}
