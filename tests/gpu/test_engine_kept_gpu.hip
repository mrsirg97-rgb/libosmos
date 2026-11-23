// a step's kept rows, one contract every family keeps (osmos/run/engine.h's osmos_batch: every row's logits, or the
// family's last part of the step, always the batch's last row - osmos/gpu/kept.h the one shape under them, 0.16.0):
// a row has logits exactly when its pick holds - a row not kept has none (osmos_logits NULL) and its pick is refused,
// saying so; a kept row's pick is its logits' greatest token (the lowest among equals), read from the device row; the
// batch's last row always kept; a pick past the step's rows refused; a decode step's row likewise. a prompt marked
// (oseq_prompt) to its last row: no row before it kept - no family computes them (0.16.0: DeepSeek called its last
// round_tokens rows kept, those before the prompt's last never computed - an earlier step's logits read back). each
// pack here (the streaming ones at their smallest build); 77 if none.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "small_packs.h"
#include "test_fit.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

static const SmallPack kPacks[] = {
    kSmallPacks[0], kSmallPacks[1],
    {"qwen38-ud-iq3_xxs", "/Models/osmos/qwen38-ud-iq3_xxs/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf", nullptr},
    {"hy4-flash", "/Models/osmos/hy4-flash/Hy4-flash-Q3_K-00001-of-00002.gguf", nullptr},
    {"osmos-evo-inkling", "/Models/osmos/osmos-evo-inkling/Inkling-2-Small-UD-IQ1_S-00001-of-00005.gguf", nullptr},
    {"osmos-evo", "/Models/osmos/osmos-evo/DeepSeek-V4.1-Flash-1017-osmos-evo-A2-00001-of-00005.gguf", nullptr},
    {"coder27", "/Models/coder27/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-IQ3_S3.gguf", nullptr},
};
constexpr uint32_t kN = 12;

// the row's greatest token from its logits on the host (the lowest among equals)
static int32_t host_pick(const float* dev, uint32_t V) {
  std::vector<float> row(V);
  CHECK(hipMemcpy(row.data(), dev, V * 4, hipMemcpyDefault) == hipSuccess);
  uint32_t best = 0;
  for (uint32_t i = 1; i < V; i++)
    if (row[i] > row[best]) best = i;
  return (int32_t) best;
}

static bool one(const SmallPack& p) {
  if (!small_pack_here(p)) return std::printf("  %s: no pack, skipped\n", p.name), false;
  const std::string path = small_pack_path(p);
  char err[OSMOS_ERR] = "";
  OsmosFiles files = {path.c_str(), nullptr, nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  CHECK(m);
  const uint32_t V = osmos_facts(m).vocab;
  OsmosOptions o = small_pack_options(m, 1024);
  Osmos* e = nullptr;
  if (!std::strcmp(osmos_facts(m).family, "deepseek41")) {
    TestFit fit = {[&](uint64_t b) { o.budget = b; return (e = osmos_create(m, &o, err)) != nullptr; },
                   [&] { OsmosStats st; osmos_stats(e, &st); return st.budget; },
                   [&](uint64_t* r) { return osmos_room(e, r); }, [&] { osmos_destroy(e), e = nullptr; }};
    uint64_t budget = 0;
    CHECK(test_fit(fit, 64ull << 30, &budget));
  } else {
    if (osmos_facts(m).streams) o.budget = 1;
    e = osmos_create(m, &o, err);
  }
  if (!e) std::fprintf(stderr, "  %s\n", err);
  CHECK(e);
  OsmosSeq* s = oseq_create(e, err);
  CHECK(s);
  int32_t t[kN];
  OsmosSeq* ss[kN];
  for (uint32_t i = 0; i < kN; i++) t[i] = (int32_t) (200 + 11 * i), ss[i] = s;
  CHECK(oseq_prompt(s, kN));
  CHECK(osmos_batch(e, ss, t, kN, err));
  // each row: logits exactly when a pick holds; a pick its logits' greatest token, a refusal one message
  uint32_t kept = 0;
  for (uint32_t r = 0; r < kN; r++) {
    int32_t id = -2;
    const float* row = osmos_logits(e, r);
    const bool picked = osmos_greedy(e, r, 1, &id, err);
    CHECK(picked == (row != nullptr));
    if (row) {
      CHECK(id == host_pick(row, V));
      kept++;
    } else {
      CHECK(std::strstr(err, "kept no logits"));
    }
    if (r + 1 < kN && row) std::fprintf(stderr, "  %s: row %u kept, before the prompt's last (never computed)\n", p.name, r);
    CHECK(r + 1 == kN || !row);
  }
  int32_t id = -1;
  CHECK(osmos_logits(e, kN - 1) && osmos_greedy(e, kN - 1, 1, &id, err));  // the last row always kept
  // past the step's rows: refused
  CHECK(!osmos_greedy(e, kN, 1, &id, err) && !osmos_logits(e, kN));
  // a decode step's one row likewise
  CHECK(osmos_batch(e, ss, &id, 1, err));
  int32_t next = -1;
  CHECK(osmos_logits(e, 0) && osmos_greedy(e, 0, 1, &next, err) && next == host_pick(osmos_logits(e, 0), V));
  CHECK(!osmos_greedy(e, 1, 1, &next, err));
  std::printf("  %s: a prompt's %u rows - %u kept (each its logits' pick), %u not (no logits, pick refused); a "
              "decode's row likewise\n", p.name, kN, kept, kN - kept);
  oseq_destroy(s);
  osmos_destroy(e);
  osmos_close(m);
  return true;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_engine_kept_gpu] start\n");
  int ran = 0;
  for (const SmallPack& p : kPacks) ran += one(p);
  if (!ran) return std::printf("[test_engine_kept_gpu] skipped: no pack here\n"), 77;
  std::printf("[test_engine_kept_gpu] ok\n");
  return 0;
}
