// which rotation the model reads best: lfm3_rope_nll <GGUF> <text> P... - the text's first P tokens read (steps of
// 2,048), the last 2,048's next-token NLL, under the native rotation and static YaRN over 262,144 at x2, x4, x8 and
// x40 (Lfm3Options' yarn_static_ctx: every position the factor's; x40 incarnate lsa.py's long-context measurements') -
// OSMOS_ROPE_MODES (a comma list of the names) runs those alone
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>

extern "C" {
#include "osmos/lfm3/forward.h"
#include "osmos/model/gguf.h"
#include "osmos/text/tokenizer.h"
}

#define CHECK(x)                                                                       \
  do {                                                                                 \
    if (!(x)) {                                                                        \
      std::fprintf(stderr, "%s:%d: failed: %s (%s)\n", __FILE__, __LINE__, #x, err); \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

int main(int argc, char** argv) {
  char err[OSMOS_ERR] = "", gerr[OSMOS_ERR] = "", terr[OSMOS_ERR] = "";
  if (argc < 4) return std::fprintf(stderr, "usage: lfm3_rope_nll <GGUF> <text> P...\n"), 2;
  Gguf* g = gguf_create(argv[1], gerr);
  if (!g) return std::fprintf(stderr, "%s\n", gerr), 1;
  Tokenizer* tok = tokenizer_create(g, terr);
  if (!tok) return std::fprintf(stderr, "%s\n", terr), 1;
  FILE* f = std::fopen(argv[2], "rb");
  if (!f) return std::fprintf(stderr, "no %s\n", argv[2]), 1;
  std::string text;
  for (int c; (c = std::fgetc(f)) != EOF && text.size() < 4000000;) text.push_back((char) c);
  std::fclose(f);
  const size_t want = tokenizer_encode(tok, text.data(), text.size(), false, nullptr, 0);
  std::vector<int32_t> ids(want);
  tokenizer_encode(tok, text.data(), text.size(), false, ids.data(), want);
  Lfm3Weights* w = lfm3_weights_create(g, err);
  CHECK(w);
  const uint32_t V = (uint32_t) w->geo.n_vocab, S = 2048;
  struct Mode {
    const char* name;
    uint32_t ctx;
  } modes[] = {{"native", 0}, {"x2", 524288}, {"x4", 1048576}, {"x8", 2097152}, {"x40", 10485760}};
  const char* only = getenv("OSMOS_ROPE_MODES");
  for (int a = 3; a < argc; a++) {
    const uint32_t P = (uint32_t) atoi(argv[a]);
    CHECK(ids.size() > (size_t) P + 1 && P >= S);
    std::printf("P %u:", P);
    for (const Mode& m : modes) {
      if (only && !std::strstr((std::string(",") + only + ",").c_str(), (std::string(",") + m.name + ",").c_str())) continue;
      const Lfm3Options o = {S, P + 1, S, LFM3_HEAD_Q8_0, nullptr, 0, nullptr, 0, m.ctx, 0};
      Lfm3Forward* fw = lfm3_forward_create(w, &o, err);
      CHECK(fw);
      Lfm3Seq* s = lfm3_seq_create(fw, err);
      CHECK(s);
      for (uint32_t at = 0; at + S < P; at += S) CHECK(lfm3_step(fw, s, ids.data() + at, S, 1, err));
      CHECK(lfm3_step(fw, s, ids.data() + P - S, S, S, err));
      std::vector<float> lg((size_t) S * V);
      std::copy(lfm3_logits(fw), lfm3_logits(fw) + lg.size(), lg.begin());
      double nll = 0;
      for (uint32_t r = 0; r < S; r++) {
        const float* x = lg.data() + (size_t) r * V;
        double mx = -INFINITY, z = 0;
        for (uint32_t v = 0; v < V; v++) mx = std::fmax(mx, x[v]);
        for (uint32_t v = 0; v < V; v++) z += std::exp(x[v] - mx);
        nll += mx + std::log(z) - x[ids[P - S + r + 1]];
      }
      std::printf("  %s %.4f", m.name, nll / S);
      std::fflush(stdout);
      lfm3_seq_destroy(s), lfm3_forward_destroy(fw);
    }
    std::printf("\n");
  }
  lfm3_weights_destroy(w), tokenizer_destroy(tok), gguf_destroy(g);
  return 0;
}
