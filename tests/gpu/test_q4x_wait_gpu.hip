// Qwen's forward on GpuWait (DESIGN.md, "0.6.2: Qwen's waits"): which of its waits are marked - the wake under the
// head's matmul - and which sleep to the end (q4x_forward_waits). a prompt's step: one plain wait. a small step (a
// token; a round's token and its drafts): one marked wait each. a draft: one marked wait a depth. the bits are the
// forward's, MTP's and the engine's tests'. needs a gfx1151, the UD-IQ3_XXS set, the shared MTP head and the dump's
// ids; skipped (77) without them.
#include "q4x_ref.h"

extern "C" {
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"
#include "osmos/run/gpu.h"
}

constexpr uint32_t kCtx = 1024, kStep = 256, kPrompt = 200, kBlock = Q4X_MTP_BLOCK;
constexpr const char* kMtp = "%s/Models/qwen38/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";

struct Waits {
  uint64_t marked, plain;
};

// the waits since `last`, which then holds the forward's now
static Waits since(const Q4xForward* f, Waits* last) {
  Waits now;
  q4x_forward_waits(f, &now.marked, &now.plain);
  const Waits d{now.marked - last->marked, now.plain - last->plain};
  *last = now;
  return d;
}

static bool held(const char* what, Waits got, uint64_t marked, uint64_t plain) {
  const bool ok = got.marked == marked && got.plain == plain;
  std::printf("  %s: %llu marked, %llu plain%s\n", what, (unsigned long long) got.marked, (unsigned long long) got.plain,
              ok ? "" : "  <- NOT as designed");
  return ok;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_q4x_wait_gpu] start\n");
  const std::string first = home_path(kQwenFirst), head = home_path(kMtp);
  std::vector<int32_t> ids;
  FILE* fp = std::fopen((home_path(kQwenDump) + "/ids.txt").c_str(), "r");
  for (int32_t x; fp && ids.size() < kPrompt && std::fscanf(fp, "%d,", &x) == 1;) ids.push_back(x);
  if (fp) std::fclose(fp);
  int ndev = 0;
  if (access(first.c_str(), R_OK) || access(head.c_str(), R_OK) || ids.size() < kPrompt ||
      hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_q4x_wait_gpu] skipped: no Qwen3.8 UD-IQ3_XXS, its MTP head, the dump's ids or a GPU\n");
    return 77;
  }
  CHECK(gpu_sleep_on_waits());  // before the first kernel, as osmos-serve
  char gerr[GGUF_ERR] = "", werr[Q4X_ERR] = "", err[Q4X_FWD_ERR] = "";
  Gguf* g = gguf_create(first.c_str(), gerr);
  Gguf* mg = gguf_create(head.c_str(), gerr);
  CHECK(g && mg);
  Q4xWeights* w = q4x_weights_create(g, nullptr, werr);
  CHECK(w);
  Q4xMtp* mtp = q4x_mtp_create(mg, w, werr);
  CHECK(mtp);
  Q4xForward* f = q4x_forward_create(w, mtp, kCtx, kStep, err);
  if (!f) std::fprintf(stderr, "  forward: %s\n", err);
  CHECK(f);
  Q4xSeq* s = q4x_seq_create(f, err);
  CHECK(s);

  Waits last{};
  since(f, &last);  // (the forward's and the sequence's creation: not a step's)
  bool ok = true;

  CHECK(q4x_batch(f, s, ids.data(), kPrompt, 0, err));
  ok &= held("a 200-token step", since(f, &last), 0, 1);

  int32_t round[1 + kBlock];
  float conf[kBlock];
  CHECK(q4x_greedy(f, kPrompt - 1, 1, &round[0], err));
  CHECK(q4x_mtp_seed(f, s, 0, (int32_t) kPrompt, err));
  ok &= held("the greedy pick and the head's seeding", since(f, &last), 0, 0);

  const int k = q4x_mtp_draft(f, s, round[0], kBlock, 0.0f, 0.0f, round + 1, conf, err);  // no floors: every depth
  if (k < 0) std::fprintf(stderr, "  draft: %s\n", err);
  CHECK(k == (int) kBlock);
  ok &= held("a draft of every depth", since(f, &last), kBlock, 0);

  CHECK(q4x_batch(f, s, round, 1 + kBlock, 0, err));  // the round's verify: its token and its drafts
  ok &= held("a round's verify step", since(f, &last), 1, 0);

  int32_t next;
  CHECK(q4x_greedy(f, kBlock, 1, &next, err) && q4x_batch(f, s, &next, 1, 0, err));
  ok &= held("a token's step", since(f, &last), 1, 0);
  CHECK(ok);

  q4x_seq_destroy(s);
  q4x_forward_destroy(f);
  q4x_mtp_destroy(mtp);
  q4x_weights_destroy(w);
  gguf_destroy(mg), gguf_destroy(g);
  std::printf("[test_q4x_wait_gpu] ok\n");
  return 0;
}
