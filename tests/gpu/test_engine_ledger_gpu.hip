// one ownership tree, every byte counted (SCOPE.md; DESIGN.md "phase 3's finish line"): an engine's every allocation
// through its ledger - what the library's own code holds from HIP and HSA (alloc_count.h: the linker's --wrap) the
// ledger's count, byte for byte, after osmos_create, with sequences, after a step and a drafted round, after a
// sequence's destroy; and after osmos_destroy and osmos_close nothing held at all. each pack its family's commit puts
// in kLedgered (0.16.0 phase 3, stage 4: one family a commit); 77 if none here.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "alloc_count.h"
#include "small_packs.h"
#include "test_fit.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

struct Pack {
  const char* name;
  const char* model;    // under $HOME
  const char* lsa;      // under $HOME, or NULL
  const char* drafter;  // under $HOME, or NULL
  bool smallest;        // the smallest build (every expert streamed): a pack past this machine's memory whole
};

static const Pack kPacks[] = {
    {"osmos1.7b", "/Models/osmos/osmos1.7b/osmos1.7b-Q4_0.gguf", "/Models/osmos/osmos1.7b/osmos1.7b-lsa-maps.safetensors",
     "/Models/osmos1.7b/osmos1.7b-dflash2-Q4_0.gguf"},
    {"inkling-2-mini", "/Models/osmos/inkling-2-mini/Inkling-2-Mini-00001-of-00002.gguf", nullptr,
     "/Models/osmos/inkling-2-mini/ink2-mini-mtp-q8.gguf"},
    {"osmos-evo-inkling", "/Models/osmos/osmos-evo-inkling/Inkling-2-Small-UD-IQ1_S-00001-of-00005.gguf", nullptr,
     nullptr, true},
    {"qwen38-ud-iq3_xxs", "/Models/osmos/qwen38-ud-iq3_xxs/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf", nullptr,
     "/Models/qwen38/DFlash2/dflash2-Q8_0.gguf", true},
    {"hy4-flash", "/Models/osmos/hy4-flash/Hy4-flash-Q3_K-00001-of-00002.gguf", nullptr,
     "/Models/osmos/hy4-flash/model-00148-of-00148.safetensors", true},
    {"osmos-evo", "/Models/osmos/osmos-evo/DeepSeek-V4.1-Flash-1017-osmos-evo-A2-00001-of-00005.gguf", nullptr, nullptr,
     false},
};
// the families whose builds go through the ledger (stage 4 adds one a commit; at its end every family, and this goes)
static const std::set<std::string> kLedgered = {"lfm3", "inkling2", "qwen4exp", "hy_v4_flash", "deepseek41"};

static std::string home(const char* p) {
  const char* h = std::getenv("HOME");
  return p ? std::string(h ? h : "") + p : std::string();
}

static uint64_t held_by(const Osmos* e) {
  OsmosStats st = {};
  osmos_stats(e, &st);
  return st.held;
}

// the library's bytes the ledger's, said where they part
static void same(const char* pack, const char* when, const Osmos* e, uint64_t outside) {
  const uint64_t lib = alloc_count::held() - outside, led = held_by(e);
  std::printf("  %s %s: %.2f MiB held, %.2f MiB counted\n", pack, when, lib / 1048576.0, led / 1048576.0);
  CHECK(lib == led);
}

static bool one(const Pack& p) {
  const std::string model = home(p.model), lsa = home(p.lsa), drafter = home(p.drafter);
  if (access(model.c_str(), R_OK)) return std::printf("  %s: no pack, skipped\n", p.name), false;
  const uint64_t before = alloc_count::held();
  char err[OSMOS_ERR] = "";
  OsmosFiles files = {model.c_str(), nullptr, nullptr, p.lsa && !access(lsa.c_str(), R_OK) ? lsa.c_str() : nullptr};
  OsmosModel* m = osmos_open(&files, err);
  CHECK(m);
  if (!kLedgered.count(osmos_facts(m).family)) {
    std::printf("  %s: %s not through the ledger yet (stage 4), skipped\n", p.name, osmos_facts(m).family);
    osmos_close(m);
    return false;
  }
  const uint64_t opened = alloc_count::held();  // (the model's own: not the engine's tree)
  OsmosOptions o = small_pack_options(m, 4096);
  if (p.drafter && !access(drafter.c_str(), R_OK)) o.drafter = drafter.c_str();
  if (p.smallest) o.budget = 1;
  Osmos* e = nullptr;
  if (!std::strcmp(osmos_facts(m).family, "deepseek41")) {
    // DeepSeek's budget its resident experts' alone (phase 4's one meaning): sized as its engine test sizes it
    // (test_fit.h: 64 GiB its own, lowered where the machine leaves no room)
    TestFit fit = {[&](uint64_t b) { o.budget = b; return (e = osmos_create(m, &o, err)) != nullptr; },
                   [&] { OsmosStats st; osmos_stats(e, &st); return st.budget; },
                   [&](uint64_t* r) { return osmos_room(e, r); }, [&] { osmos_destroy(e), e = nullptr; }};
    uint64_t budget = 0;
    CHECK(test_fit(fit, 64ull << 30, &budget));
  } else
    e = osmos_create(m, &o, err);
  if (!e) std::fprintf(stderr, "  %s\n", err);
  CHECK(e);
  same(p.name, "built", e, opened);

  OsmosSeq* s[2] = {oseq_create(e, err), oseq_create(e, err)};
  CHECK(s[0] && s[1]);
  same(p.name, "two sequences", e, opened);
  constexpr uint32_t kN = 32;
  std::vector<int32_t> t(kN);
  for (uint32_t i = 0; i < kN; i++) t[i] = (int32_t) (100 + 7 * i);
  OsmosSeq* ss[kN];
  for (uint32_t i = 0; i < kN; i++) ss[i] = s[0];
  CHECK(oseq_prompt(s[0], kN));
  CHECK(osmos_batch(e, ss, t.data(), kN, err));
  same(p.name, "a prompt read", e, opened);
  if (osmos_block(e)) {  // a drafted round: the seed and a draft
    int32_t y = 0;
    CHECK(osmos_greedy(e, kN - 1, 1, &y, err));
    CHECK(oseq_seed(s[0], 0, kN, err));
    int32_t drafts[OSMOS_MAX_BLOCK];
    float conf[OSMOS_MAX_BLOCK];
    CHECK(oseq_draft(s[0], y, 0.0f, 0.0f, drafts, conf, err) >= 0);
    same(p.name, "a drafted round", e, opened);
  }
  const uint64_t two = held_by(e);
  oseq_destroy(s[1]);
  CHECK(held_by(e) < two);
  same(p.name, "a sequence destroyed", e, opened);
  oseq_destroy(s[0]);
  osmos_destroy(e);
  if (alloc_count::held() != opened) alloc_count::say(opened);
  CHECK(alloc_count::held() == opened);  // the engine's whole tree given back
  osmos_close(m);
  if (alloc_count::held() != before) alloc_count::say(before);
  CHECK(alloc_count::held() == before);
  std::printf("  %s: destroyed, every byte given back\n", p.name);
  return true;
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_engine_ledger_gpu] start\n");
  int ran = 0;
  for (const Pack& p : kPacks)  // (a pack named: that one alone - a family's commit checks its own)
    if (argc < 2 || !std::strcmp(argv[1], p.name)) ran += one(p);
  if (!ran) return std::printf("[test_engine_ledger_gpu] skipped: no ledgered pack here\n"), 77;
  std::printf("[test_engine_ledger_gpu] ok\n");
  return 0;
}
