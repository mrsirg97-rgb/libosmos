// a sequence's reset and its broken state, one contract every family keeps (osmos/run/engine.h: oseq_reset):
//   a reset sequence is a new one - after a prompt and a reset, a second prompt's logits the bits of a new sequence's
//     on it (its memory kept: a conversation's restart allocates nothing);
//   a step refused by its checks (before any work) leaves its sequence as it was - its next step the bits of one that
//     never saw it (0.15.3: Inkling's left it failed, osmos1.7b's as it was);
//   a broken sequence's steps refused until a reset holds - which mends it (osmos1.7b's reset failed through the fault
//     seam, osmos/err_hip.h).
// each pack here (the streaming ones at their smallest build); 77 if none.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/err_hip.h"
#include "osmos/lfm3/forward.h"
}
#include <functional>

#include "small_packs.h"
#include "test_fit.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

static bool failing;
extern "C" bool osmos_hip_fault(const char* what) { return failing && !std::strcmp(what, LFM3_SEQ_RESET); }

static const SmallPack kPacks[] = {
    kSmallPacks[0], kSmallPacks[1],
    {"qwen38-ud-iq3_xxs", "/Models/osmos/qwen38-ud-iq3_xxs/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf", nullptr},
    {"hy4-flash", "/Models/osmos/hy4-flash/Hy4-flash-Q3_K-00001-of-00002.gguf", nullptr},
    {"osmos-evo-inkling", "/Models/osmos/osmos-evo-inkling/Inkling-2-Small-UD-IQ1_S-00001-of-00005.gguf", nullptr},
    {"osmos-evo", "/Models/osmos/osmos-evo/DeepSeek-V4.1-Flash-1017-osmos-evo-A2-00001-of-00005.gguf", nullptr},
};
constexpr uint32_t kN = 24;

// a prompt read to its end: its last row's logits, copied out
static std::vector<float> read(Osmos* e, OsmosSeq* s, const int32_t* t, uint32_t V) {
  char err[OSMOS_ERR] = "";
  OsmosSeq* ss[kN];
  for (uint32_t i = 0; i < kN; i++) ss[i] = s;
  CHECK(oseq_prompt(s, (uint32_t) oseq_position(s) + kN));
  if (!osmos_batch(e, ss, t, kN, err)) std::fprintf(stderr, "  %s\n", err);
  std::vector<float> row(V);
  CHECK(osmos_logits(e, kN - 1) &&
        hipMemcpy(row.data(), osmos_logits(e, kN - 1), V * 4, hipMemcpyDefault) == hipSuccess);
  return row;
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
    // DeepSeek's budget its resident experts' alone (phase 4's one meaning): sized to this machine as its engine test
    // sizes it (test_fit.h: 64 GiB its own, lowered where the machine leaves no room)
    TestFit fit = {[&](uint64_t b) { o.budget = b; return (e = osmos_create(m, &o, err)) != nullptr; },
                   [&] { OsmosStats st; osmos_stats(e, &st); return st.budget; },
                   [&](uint64_t* r) { return osmos_room(e, r); }, [&] { osmos_destroy(e), e = nullptr; }};
    uint64_t budget = 0;
    CHECK(test_fit(fit, 64ull << 30, &budget));
  } else {
    if (osmos_facts(m).streams) o.budget = 1;  // (the smallest build: every expert streamed)
    e = osmos_create(m, &o, err);
  }
  if (!e) std::fprintf(stderr, "  %s\n", err);
  CHECK(e);
  int32_t a[kN], b[kN];
  for (uint32_t i = 0; i < kN; i++) a[i] = (int32_t) (100 + 11 * i), b[i] = (int32_t) (300 + 13 * i);
  // a reset sequence is a new one
  OsmosSeq* fresh = oseq_create(e, err);
  CHECK(fresh);
  const std::vector<float> want = read(e, fresh, b, V);
  OsmosSeq* s = oseq_create(e, err);
  CHECK(s);
  read(e, s, a, V);
  CHECK(oseq_reset(s, err) && oseq_position(s) == 0);
  const std::vector<float> got = read(e, s, b, V);
  CHECK(!std::memcmp(got.data(), want.data(), V * 4));
  std::printf("  %s: after a prompt and a reset, a prompt's logits a new sequence's, bit for bit\n", p.name);
  // a refused step leaves it as it was: a step to the context's end with every row's logits kept - refused by a family
  // keeping fewer rows than that (its logits' rows), before any work
  OsmosSeq* r = oseq_create(e, err);
  CHECK(r);
  read(e, r, a, V);
  const uint32_t n = 1024 - kN;
  std::vector<OsmosSeq*> many(n, r);
  std::vector<int32_t> t(n, 100);
  err[0] = 0;
  if (osmos_batch(e, many.data(), t.data(), n, err)) {
    std::printf("  %s: a step of %u rows all kept: run (its logits' rows hold them - nothing refused to check)\n", p.name, n);
  } else {
    std::printf("  %s: the refused step: %s\n", p.name, err);
    const bool as_was = oseq_position(r) == (int32_t) kN;
    err[0] = 0;
    OsmosSeq* ss[kN];
    for (uint32_t i = 0; i < kN; i++) ss[i] = r;
    CHECK(oseq_prompt(r, 2 * kN));
    const bool stepped = osmos_batch(e, ss, b, kN, err);
    std::printf("  %s: after it, the sequence %s%s%s\n", p.name, as_was && stepped ? "as it was" : "CHANGED",
                err[0] ? " - " : "", err);
    CHECK(as_was && stepped);
  }
  // a broken sequence refused until a reset mends it (osmos1.7b's reset failing through the seam)
  if (!std::strcmp(osmos_facts(m).family, "lfm3")) {
    const std::vector<uint8_t> bad(64, 0x5a);
    failing = true;
    CHECK(!oseq_load(s, bad.data(), bad.size(), err));
    failing = false;
    OsmosSeq* one_s[1] = {s};
    CHECK(!osmos_batch(e, one_s, a, 1, err));  // (broken: refused)
    (void) b;
    CHECK(oseq_reset(s, err));
    const std::vector<float> mended = read(e, s, b, V);
    CHECK(!std::memcmp(mended.data(), want.data(), V * 4));
    std::printf("  %s: a broken sequence refused, mended by a reset (a new one's bits)\n", p.name);
  }
  oseq_destroy(r), oseq_destroy(s), oseq_destroy(fresh);
  osmos_destroy(e);
  osmos_close(m);
  return true;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_engine_reset_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) return std::printf("[test_engine_reset_gpu] skipped: no GPU\n"), 77;
  int ran = 0;
  for (const SmallPack& p : kPacks) ran += one(p);
  if (!ran) return std::printf("[test_engine_reset_gpu] skipped: no pack\n"), 77;
  std::printf("[test_engine_reset_gpu] ok\n");
  return 0;
}
