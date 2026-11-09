// the MTP head's seeding and the next draft (q4x_mtp_seed, q4x_mtp_draft) the same bits however the GPU and memory
// stood (0.5.10: drafts' confidences 0.1% apart run to run, two causes):
//   held   the seed's tokens reach the GPU through the pinned token buffer the next step writes too. the GPU held
//          busy before the seed (q4x_spin: its copy runs late), the draft right after it: the draft's own token must
//          not stand in for the seed's
//   reused a sequence's first seeded row at position 0 is zeros, never its pending h's memory (not cleared at
//          create): a sequence made where one that seeded another prompt was let go (its memory most likely handed
//          back as it was)
// each the reference's drafts and confidences, bit for bit. needs a gfx1151, the UD-IQ3_XXS set, the shared MTP head and the
// dump's ids; skipped (77) without them.
#include "q4x_ref.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"
}

constexpr uint32_t kCtx = 1024, kStep = 256, kPrompt = 200, kBlock = 5;
constexpr uint32_t kHoldUs = 200000; // the GPU's hold before the seed: far past the host's next tokens_in (~us)
constexpr const char* kMtp = "%s/Models/qwen38/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";

struct Drafted {
  int k;
  int32_t ids[kBlock];
  float conf[kBlock];
};

static Drafted run(Q4xForward* f, const std::vector<int32_t>& p, bool hold) {
  char err[Q4X_FWD_ERR] = "";
  Q4xSeq* s = q4x_seq_create(f, err);
  CHECK(s);
  CHECK(q4x_batch(f, s, p.data(), kPrompt, 0, err));
  int32_t y;
  CHECK(q4x_greedy(f, kPrompt - 1, 1, &y, err));
  if (hold) CHECK(q4x_spin(kHoldUs, nullptr));
  CHECK(q4x_mtp_seed(f, s, 0, (int32_t) kPrompt, err));
  Drafted d{};
  d.k = q4x_mtp_draft(f, s, y, kBlock, 0.0f, 0.0f, d.ids, d.conf, err); // no floors: the whole block
  if (d.k < 0) std::fprintf(stderr, "  draft: %s\n", err);
  CHECK(d.k == (int) kBlock && hipDeviceSynchronize() == hipSuccess);
  q4x_seq_destroy(s);
  return d;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_mtp_seed_gpu] start\n");
  const std::string first = home_path(kQwenFirst), head = home_path(kMtp);
  std::vector<int32_t> ids;
  FILE* fp = std::fopen((home_path(kQwenDump) + "/ids.txt").c_str(), "r");
  for (int32_t x; fp && ids.size() < kPrompt && std::fscanf(fp, "%d,", &x) == 1;) ids.push_back(x);
  if (fp) std::fclose(fp);
  int ndev = 0;
  if (access(first.c_str(), R_OK) || access(head.c_str(), R_OK) || ids.size() < kPrompt ||
      hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_mtp_seed_gpu] skipped: no Qwen3.8 UD-IQ3_XXS, its MTP head, the dump's ids or a GPU\n");
    return 77;
  }
  char gerr[GGUF_ERR] = "", werr[Q4X_ERR] = "", ferr[Q4X_FWD_ERR] = "";
  Gguf* g = gguf_create(first.c_str(), gerr);
  Gguf* mg = gguf_create(head.c_str(), gerr);
  CHECK(g && mg);
  Q4xWeights* w = q4x_weights_create(g, nullptr, werr);
  CHECK(w);
  Q4xMtp* mtp = q4x_mtp_create(mg, w, werr);
  CHECK(mtp);
  Q4xForward* f = q4x_forward_create(w, mtp, kCtx, kStep, ferr);
  if (!f) std::fprintf(stderr, "  forward: %s\n", ferr);
  CHECK(f);

  const Drafted ref = run(f, ids, false);
  auto same = [&](const char* what, const Drafted& d) {
    bool ok = true;
    for (int i = 0; i < (int) kBlock; i++) ok = ok && d.ids[i] == ref.ids[i] && std::memcmp(&d.conf[i], &ref.conf[i], 4) == 0;
    std::printf("  %s: the drafts %s (first %d %a against %d %a)\n", what, ok ? "the same bits" : "NOT the same",
                d.ids[0], d.conf[0], ref.ids[0], ref.conf[0]);
    return ok;
  };
  const bool held = same("the GPU held before the seed", run(f, ids, true));
  {  // another prompt's sequence seeded (its pending h written) and let go
    std::vector<int32_t> other(ids.rbegin(), ids.rend());
    run(f, other, false);
  }
  const bool reused = same("a sequence where another one was", run(f, ids, false));
  CHECK(held && reused);

  q4x_forward_destroy(f);
  q4x_mtp_destroy(mtp);
  q4x_weights_destroy(w);
  gguf_destroy(mg), gguf_destroy(g);
  std::printf("[test_mtp_seed_gpu] ok\n");
  return 0;
}
