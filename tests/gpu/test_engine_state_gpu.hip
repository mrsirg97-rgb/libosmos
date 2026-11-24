// a saved sequence's load, one contract every family keeps (osmos/run/engine.h: oseq_save / oseq_load; osmos/gpu/
// state_parts.h's checks under them, 0.16.0): a state saved after a prompt loads back - the next step's logits the
// bits of the sequence it came from; a state a byte short, a byte long, of another format (its magic) or holding a
// token outside the vocabulary in its history (Qwen's and Hy4's n-gram tokens) refused, saying so, the sequence then
// a new one (position 0), never part one conversation and part another (0.16.0: Qwen loaded any token in its history;
// Hy4 refused it). each pack here (the streaming ones at their smallest build); 77 if none.
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
    {"coder27", "/Models/coder390-halo/models/osmos-vibes-nvfp4.gguf", nullptr},
};
constexpr uint32_t kN = 24;

// the family's history of tokens in its state: the first token's offset (its header's), or -1 none
static long history_at(const char* family) {
  if (!std::strcmp(family, "qwen4exp")) return 24;  // magic, version (4 + 4), identity (8), pos, n_last (4 + 4)
  return -1;
}

static std::vector<float> row(Osmos* e, uint32_t r, uint32_t V) {
  std::vector<float> v(V);
  CHECK(osmos_logits(e, r) && hipMemcpy(v.data(), osmos_logits(e, r), V * 4, hipMemcpyDefault) == hipSuccess);
  return v;
}

static bool one(const SmallPack& p) {
  if (!small_pack_here(p)) return std::printf("  %s: no pack, skipped\n", p.name), false;
  const std::string path = small_pack_path(p);
  char err[OSMOS_ERR] = "";
  OsmosFiles files = {path.c_str(), nullptr, nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  if (!m) std::fprintf(stderr, "  %s\n", err);
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
  OsmosSeq *a = oseq_create(e, err), *b = oseq_create(e, err);
  CHECK(a && b);
  int32_t t[kN];
  OsmosSeq* ss[kN];
  for (uint32_t i = 0; i < kN; i++) t[i] = (int32_t) (300 + 13 * i), ss[i] = a;
  CHECK(oseq_prompt(a, kN) && osmos_batch(e, ss, t, kN, err));
  std::vector<uint8_t> st(oseq_state_bytes(a));
  CHECK(!st.empty() && oseq_save(a, st.data(), st.size(), err));
  // a load back: the next step the sequence's own
  const int32_t next = 777;
  CHECK(osmos_batch(e, &a, &next, 1, err));
  const std::vector<float> want = row(e, 0, V);
  CHECK(oseq_load(b, st.data(), st.size(), err) && oseq_position(b) == (int32_t) kN);
  CHECK(osmos_batch(e, &b, &next, 1, err) && row(e, 0, V) == want);
  // refused, each saying so, the sequence a new one
  struct Bad {
    const char* what;
    std::vector<uint8_t> bytes;
  };
  std::vector<Bad> bad;
  bad.push_back({"a byte short", std::vector<uint8_t>(st.begin(), st.end() - 1)});
  std::vector<uint8_t> longer = st;
  longer.push_back(0);
  bad.push_back({"a byte long", longer});
  std::vector<uint8_t> other = st;
  other[0] ^= 0x5a;
  bad.push_back({"another format", other});
  const long h = history_at(osmos_facts(m).family);
  if (h >= 0) {
    std::vector<uint8_t> outside = st;
    const int32_t tok = (int32_t) V;  // the vocabulary's end: no token
    std::memcpy(outside.data() + h, &tok, 4);
    bad.push_back({"a history token outside the vocabulary", outside});
  }
  for (const Bad& x : bad) {
    err[0] = 0;
    const bool loaded = oseq_load(b, x.bytes.data(), x.bytes.size(), err);
    if (loaded) std::fprintf(stderr, "  %s: %s loaded\n", p.name, x.what);
    CHECK(!loaded && err[0] && oseq_position(b) == 0);
    std::printf("    %s: %s refused (%s)\n", p.name, x.what, err);
  }
  std::printf("  %s: a state loads back the bits; %zu bad states refused, the sequence a new one\n", p.name, bad.size());
  oseq_destroy(a), oseq_destroy(b);
  osmos_destroy(e);
  osmos_close(m);
  return true;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_engine_state_gpu] start\n");
  int ran = 0;
  for (const SmallPack& p : kPacks) ran += one(p);
  if (!ran) return std::printf("[test_engine_state_gpu] skipped: no pack here\n"), 77;
  std::printf("[test_engine_state_gpu] ok\n");
  return 0;
}
