// an overlay (weights_create_with): UD-IQ1_M with huihui-ai's abliteration patch as an overlay decodes the bits of
// huihui's own patched files (their script's output: UD-IQ1_M-abliterated) - every step's logits row and token - and
// not UD-IQ1_M's (the patch changes the model, so the comparison means something); its identity not the plain
// model's (a sequence's state does not cross). needs a gfx1151, UD-IQ1_M, its patched copy and the patch; skipped (77)
// without them.
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/model/weights.h"
#include "osmos/run/forward.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kMaxTokens = 512, kMaxCtx = 4096, kPrompt = 160;
constexpr int kNew = 24;
constexpr uint64_t kBudget = 100ull << 30;

struct Run {
  uint64_t hash, identity;
};

static uint64_t fnv(const void* p, size_t n, uint64_t h) {
  const uint8_t* b = (const uint8_t*) p;
  for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

// the model at `first` (with `overlay`, or none): a prompt, then kNew greedy steps, every logits row and token hashed
static Run decode(const std::string& first, const std::string& overlay) {
  char gerr[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "", err[FORWARD_ERR] = "";
  Gguf* g = gguf_create(first.c_str(), gerr);
  Gguf* ov = overlay.empty() ? nullptr : gguf_create(overlay.c_str(), gerr);
  CHECK(g && (overlay.empty() || ov));
  const WeightsOptions wo = {g, nullptr, ov};
  Weights* w = weights_create_with(&wo, werr);
  if (!w) std::fprintf(stderr, "  weights: %s\n", werr);
  CHECK(w);
  const ForwardOptions o = {(int) w->geo.n_layers, kMaxCtx, kBudget, nullptr, kMaxTokens, true};
  Forward* f = forward_create_with(w, &o, err);
  if (!f) std::fprintf(stderr, "  forward: %s\n", err);
  CHECK(f && forward_lane(f, 0, NULL, err));
  const uint32_t V = (uint32_t) w->geo.n_vocab;
  std::vector<int32_t> p(kPrompt);
  uint32_t r = 4242;
  for (auto& x : p) r = r * 1664525u + 1013904223u, x = (int32_t) ((r >> 8) % (V - 1000)) + 100;
  Seq* s = seq_create(f, err);
  CHECK(s && seq_prompt(s, kPrompt));
  std::vector<Seq*> ss(kPrompt, s);
  CHECK(forward_batch(f, ss.data(), p.data(), kPrompt, err));
  int32_t t;
  CHECK(forward_greedy(f, kPrompt - 1, 1, &t, err));
  std::vector<float> lg(V);
  Run run = {1469598103934665603ull, forward_identity(f)};
  for (int i = 0; i < kNew; i++) {
    CHECK(forward_step(f, s, t, err));
    CHECK(forward_greedy(f, 0, 1, &t, err) && hipDeviceSynchronize() == hipSuccess);
    CHECK(hipMemcpy(lg.data(), forward_logits(f), (size_t) V * 4, hipMemcpyDeviceToHost) == hipSuccess);
    run.hash = fnv(&t, 4, fnv(lg.data(), lg.size() * 4, run.hash));
  }
  seq_destroy(s);
  forward_destroy(f);
  weights_destroy(w);
  gguf_destroy(ov);
  gguf_destroy(g);
  return run;
}

int main() {
  std::printf("[test_overlay_gpu] start\n");
  const char* home = std::getenv("HOME");
  const std::string d = std::string(home ? home : "") + "/Models/ds1017/";
  const std::string name = "DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf";
  const std::string plain = d + "UD-IQ1_M/" + name, patched = d + "UD-IQ1_M-abliterated/" + name;
  const std::string patch = d + "huihui/unsloth/patch-UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-patch.gguf";
  int ndev = 0;
  if (access(plain.c_str(), R_OK) || access(patched.c_str(), R_OK) || access(patch.c_str(), R_OK) ||
      hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_overlay_gpu] skipped: no UD-IQ1_M, its abliterated copy, huihui's patch or a GPU\n");
    return 77;
  }
  const Run base = decode(plain, ""), over = decode(plain, patch), theirs = decode(patched, "");
  std::printf("  UD-IQ1_M %016llx, with the overlay %016llx, huihui's patched files %016llx\n",
              (unsigned long long) base.hash, (unsigned long long) over.hash, (unsigned long long) theirs.hash);
  CHECK(over.hash == theirs.hash);  // the overlay is the patch: the bits of the patched files
  CHECK(over.hash != base.hash);    // and the patch changes the model: the comparison means something
  CHECK(over.identity != base.identity);
  std::printf("  %d steps' logits and tokens: the overlay's the patched files' bytes, not the model's; another identity\n",
              kNew);
  std::printf("[test_overlay_gpu] ok\n");
  return 0;
}
