// the engine's draft refuses a token outside the vocabulary, every family alike (osmos/run/engine.h: oseq_draft -
// "the count, or -1 with why"): after a step, y at the vocabulary's size, past it and negative: -1, saying so; y inside
// the vocabulary: a count (0.15.3: Hy4's drafter returned 0, no drafts, where Inkling's refused). each small pack with
// a drafter here; 77 if none.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

#include "small_packs.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

// the packs with a drafter: the small ones, and Hy4-flash at its smallest build (a budget of a byte: every expert
// streamed - its drafter's contract is the one that differed)
static const SmallPack kHy4 = {"hy4-flash", "/Models/osmos/hy4-flash/Hy4-flash-Q3_K-00001-of-00002.gguf",
                               "/Models/osmos/hy4-flash/model-00148-of-00148.safetensors"};

static bool one(const SmallPack& p, bool smallest) {
  if (!small_pack_here(p) || !p.drafter) return std::printf("  %s: no pack or drafter, skipped\n", p.name), false;
  const std::string path = small_pack_path(p), drafter = small_pack_drafter(p);
  char err[OSMOS_ERR] = "";
  OsmosFiles files = {path.c_str(), nullptr, nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  CHECK(m);
  const int32_t V = (int32_t) osmos_facts(m).vocab;
  OsmosOptions o = small_pack_options(m, 1024);
  o.drafter = drafter.c_str();
  if (smallest) o.budget = 1;
  Osmos* e = osmos_create(m, &o, err);
  if (!e) std::fprintf(stderr, "  %s\n", err);
  CHECK(e && osmos_block(e) > 0);
  OsmosSeq* s = oseq_create(e, err);
  CHECK(s);
  OsmosSeq* ss[16];
  int32_t t[16];
  for (int i = 0; i < 16; i++) ss[i] = s, t[i] = 100 + 7 * i;
  CHECK(osmos_batch(e, ss, t, 16, err) && oseq_seed(s, 0, 16, err));
  int32_t drafts[OSMOS_MAX_BLOCK];
  float conf[OSMOS_MAX_BLOCK];
  for (int32_t bad : {V, V + 1000, -1}) {
    err[0] = 0;
    const int got = oseq_draft(s, bad, 0.0f, 0.0f, drafts, conf, err);
    std::printf("  %s: a draft after y %d: %d%s%s\n", p.name, bad, got, err[0] ? " - " : "", err);
    CHECK(got == -1 && std::strstr(err, "vocabulary"));
  }
  {  // the batched form alike (oseq_draft_all: Qwen's MTP head drafts many at once)
    int32_t dd[1][OSMOS_MAX_BLOCK];
    float cc[1][OSMOS_MAX_BLOCK];
    int counts[1];
    OsmosSeq* one_s[1] = {s};
    const int32_t ys[1] = {V};
    err[0] = 0;
    CHECK(!oseq_draft_all(one_s, ys, 1, 0.0f, 0.0f, dd, cc, counts, err) && std::strstr(err, "vocabulary"));
  }
  int32_t y = -1;
  CHECK(osmos_greedy(e, 15, 1, &y, err));
  CHECK(oseq_draft(s, y, 0.0f, 0.0f, drafts, conf, err) >= 0);  // a token of the vocabulary: a count
  oseq_destroy(s);
  osmos_destroy(e);
  osmos_close(m);
  return true;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_engine_draft_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) return std::printf("[test_engine_draft_gpu] skipped: no GPU\n"), 77;
  int ran = 0;
  for (const SmallPack& p : kSmallPacks) ran += one(p, false);
  ran += one(kHy4, true);
  if (!ran) return std::printf("[test_engine_draft_gpu] skipped: no pack with a drafter\n"), 77;
  std::printf("[test_engine_draft_gpu] ok\n");
  return 0;
}
