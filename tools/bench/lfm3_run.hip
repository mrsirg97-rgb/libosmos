// LFM3's forward timed (osmos/lfm3/forward.h): lfm3_run <GGUF> [prompt=512] [step=512] [decode=64] - the wiki text's
// ids aren't needed: the prompt is a fixed pattern of ids (a speed run, not a quality one), then greedy decode of one
// token a step (the host's argmax over lfm3_logits); each phase's wall time and rate, the decoded ids' hash
//   IDS=path    the prompt's ids: a file of int32 (the first P), else the pattern
//   DFLASH=path the decode in drafted rounds with this DFlash2 (lfm3_dflash_*): y and its drafts one decode step
//               (lfm3_step_runs: the decode's math), the greedy picks keeping the leading drafts, the rest rolled back
//               - its rate, its drafts accepted a round; the tokens the plain decode's (the same hash). CUM=x: a
//               round's drafts kept while their confidences' product is at least x (the engine's cum_min; 0 all);
//               SHOW=1 the first rounds' drafts against the target's picks, and their confidences
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
#include "osmos/q4x/dflash.h"
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
  char err[OSMOS_ERR] = "", gerr[OSMOS_ERR] = "";
  if (argc < 2) return std::fprintf(stderr, "usage: lfm3_run <GGUF> [prompt=512] [step=512] [decode=64] [file: the BF16 head]\n"), 2;
  const uint32_t P = argc > 2 ? (uint32_t) atoi(argv[2]) : 512, S = argc > 3 ? (uint32_t) atoi(argv[3]) : 512;
  const uint32_t Dn = argc > 4 ? (uint32_t) atoi(argv[4]) : 64;
  const double t0 = now();
  Gguf* g = gguf_create(argv[1], gerr);
  if (!g) return std::fprintf(stderr, "%s\n", gerr), 1;
  Lfm3Weights* w = lfm3_weights_create(g, err);
  CHECK(w);
  const bool file_head = argc > 5 && !strcmp(argv[5], "file");  // the head as the file's BF16, not its Q8_0 copy
  const char* dpath = getenv("DFLASH");
  Gguf* dg = nullptr;
  Q4xDflash* df = nullptr;
  if (dpath) {
    dg = gguf_create(dpath, gerr);
    if (!dg) return std::fprintf(stderr, "%s\n", gerr), 1;
    const Q4xDflashTarget t = {w->geo.n_embd, w->geo.n_vocab, w->geo.n_layers};
    df = q4x_dflash_create_for(dg, &t, err);
    CHECK(df);
  }
  const uint32_t B = df ? (uint32_t) df->geo.block : 1;  // a round's rows: y and its drafts
  const Lfm3Options o = {S > B ? S : B, P + Dn + B + 1, B, file_head ? LFM3_HEAD_FILE : LFM3_HEAD_Q8_0, nullptr, 0,
                         nullptr, 0, 0, 0, df};
  Lfm3Forward* f = lfm3_forward_create(w, &o, err);
  CHECK(f);
  Lfm3Seq* s = lfm3_seq_create(f, err);
  CHECK(s);
  const uint32_t V = (uint32_t) w->geo.n_vocab;
  std::vector<int32_t> ids(P);
  for (uint32_t i = 0; i < P; i++) ids[i] = (int32_t) ((i * 7919u + 13u) % 120000u);
  if (const char* ip = getenv("IDS")) {
    FILE* fp = std::fopen(ip, "rb");
    if (!fp || std::fread(ids.data(), 4, P, fp) != P) return std::fprintf(stderr, "IDS: fewer than %u ids\n", P), 1;
    std::fclose(fp);
  }
  std::printf("lfm3_run: loaded in %.2f s; prompt %u in steps of %u, decode %u\n", now() - t0, P, S, Dn);
  const double p0 = now();
  for (uint32_t at = 0; at < P; at += S) {
    const uint32_t n = P - at < S ? P - at : S;
    CHECK(lfm3_step(f, s, ids.data() + at, n, 1, err));
    if (df) CHECK(lfm3_dflash_seed(f, s, (int32_t) at, (int32_t) (at + n), err));
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
  uint32_t rounds = 0, drafted = 0, accepted = 0;
  if (!df)
    for (uint32_t i = 0; i < Dn; i++) {
      CHECK(lfm3_step(f, s, &next, 1, 1, err));
      next = pick();
      hash = (hash ^ (uint64_t) next) * 1099511628211ull;
    }
  else {
    const float cum = getenv("CUM") ? (float) atof(getenv("CUM")) : 0.0f;
    int32_t tok[8], drafts[8], picks[8];
    float conf[8];
    for (uint32_t made = 0; made < Dn; rounds++) {
      const int k = lfm3_dflash_draft(f, s, next, B - 1, 0.0f, cum, drafts, conf, err);
      CHECK(k >= 0);
      tok[0] = next;
      for (int j = 0; j < k; j++) tok[j + 1] = drafts[j];
      const int32_t p0 = lfm3_seq_position(s);
      const uint32_t len = (uint32_t) k + 1;
      CHECK(lfm3_step_runs(f, &s, &len, 1, tok, err) && lfm3_greedy(f, 0, len, picks, err));
      int m = 0;
      while (m < k && drafts[m] == picks[m]) m++;
      if (getenv("SHOW") && rounds < 6) {  // the round's drafts against the target's picks
        std::printf("round %u at %d, y %d:", rounds, p0, next);
        for (int j = 0; j < k; j++) std::printf(" %d/%d(%.2f)", drafts[j], picks[j], conf[j]);
        std::printf("\n");
      }
      CHECK(lfm3_seq_rollback(s, (uint32_t) (k - m)));
      CHECK(lfm3_dflash_seed(f, s, p0, p0 + m + 1, err));
      drafted += (uint32_t) k, accepted += (uint32_t) m;
      for (int j = 0; j <= m && made < Dn; j++, made++)
        hash = (hash ^ (uint64_t) (j < m ? drafts[j] : picks[m])) * 1099511628211ull;
      next = picks[m];
    }
  }
  const double d1 = now();
  if (df)
    std::printf("drafted: %u rounds, %u drafts, %u accepted (%.2f a round, %.0f%%)\n", rounds, drafted, accepted,
                (double) accepted / rounds, drafted ? 100.0 * accepted / drafted : 0.0);
  std::printf("prompt: %u tokens in %.3f s (%.1f tok/s)\ndecode: %u tokens in %.3f s (%.2f tok/s, %.2f ms a token); "
              "ids hash %016llx\n", P, p1 - p0, P / (p1 - p0), Dn, d1 - d0, Dn / (d1 - d0), 1000 * (d1 - d0) / Dn,
              (unsigned long long) hash);
  lfm3_seq_destroy(s), lfm3_forward_destroy(f), lfm3_weights_destroy(w), gguf_destroy(g);
  q4x_dflash_destroy(df), gguf_destroy(dg);
  return 0;
}
