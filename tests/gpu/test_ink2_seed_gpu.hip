// Inkling's drafter seed (0.9.2): the taps every step overwrites are a sequence's to seed from only until the next
// step - a seed after another sequence's step, or after a load, refused (it was accepted, its rings built from the
// other's rows); its own step's again accepted. then a drafted decode the bits of the plain one (the server's rounds:
// drafts after y, [y, drafts] one step, the agreeing prefix kept, the rest rolled back, the kept positions seeded).
// on Inkling-2-Small's osmos-evo-inkling pack (OSMOS_INK2_PACK: its folder) and on Inkling-2-Mini's reference with its
// drafter (OSMOS_INK2_MINI: the folder of ink2_ref_gguf.py's and ink2_mtp_gguf.py's GGUFs - a loop: the taps its steps,
// the slots in the order they run), each where it is; needs a gfx1151, skipped (77) without it or both.
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "osmos/run/engine.h"
}

#include "test_fit.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kCtx = 4096, kStep = 512, kPrompt = 300, kGen = 48;
constexpr uint64_t kBudget = 60ull << 30;

static char g_err[OSMOS_ERR] = "";

static std::vector<int32_t> ids(uint32_t n, uint32_t seed, uint32_t vocab) {
  std::vector<int32_t> v(n);
  uint32_t r = seed;
  for (auto& x : v) r = r * 1664525u + 1013904223u, x = (int32_t) ((r >> 8) % (vocab - 1000)) + 100;
  for (uint32_t i = n / 2; i < n; i++) v[i] = v[i - n / 2];  // its second half its first again: drafts that land
  return v;
}

static int32_t read(Osmos* e, OsmosSeq* s, const std::vector<int32_t>& p) {
  std::vector<OsmosSeq*> ss(p.size(), s);
  CHECK(oseq_prompt(s, (uint32_t) p.size()) && osmos_read(e, ss.data(), p.data(), (uint32_t) p.size(), g_err));
  int32_t y;
  CHECK(osmos_greedy(e, (uint32_t) p.size() - 1, 1, &y, g_err));
  return y;
}

static bool pack(const char* name, const std::string& model, const std::string& mtp, const std::string& picks) {
  if (access(model.c_str(), R_OK) || access(mtp.c_str(), R_OK)) {
    std::printf("  %s: not on this machine (%s), skipped\n", name, model.c_str());
    return false;
  }
  std::printf("  %s:\n", name);
  const OsmosFiles files = {model.c_str(), nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, g_err);
  CHECK(m);
  const OsmosOptions o = {kCtx, kStep, kBudget, picks.empty() || access(picks.c_str(), R_OK) ? nullptr : picks.c_str(), true,
                          mtp.c_str(), false, 0, false, nullptr, 0.0f};
  Osmos* e = test_osmos(m, o, g_err);  // (the test's budget, or this machine's fill where it is smaller)
  if (!e) std::fprintf(stderr, "  %s\n", g_err);
  CHECK(e && osmos_block(e) > 0);
  const OsmosFacts facts = osmos_facts(m);
  const std::vector<int32_t> pa = ids(kPrompt, 1017, facts.vocab), pb = ids(200, 4242, facts.vocab);

  OsmosSeq *a = oseq_create(e, g_err), *b = oseq_create(e, g_err);
  CHECK(a && b);
  read(e, a, pa);
  read(e, b, pb);
  CHECK(!oseq_seed(a, kPrompt - 8, kPrompt, g_err) && std::strstr(g_err, "not this sequence's"));  // B's step since
  CHECK(oseq_seed(b, 0, 200, g_err));                                                               // its own: taken
  std::vector<uint8_t> st(oseq_state_bytes(b));
  CHECK(oseq_save(b, st.data(), st.size(), g_err) && oseq_load(a, st.data(), st.size(), g_err));
  CHECK(!oseq_seed(a, 190, 200, g_err) && std::strstr(g_err, "not this sequence's"));  // loaded: no step of its own
  std::printf("  a seed after another sequence's step, and after a load: refused; after its own step: taken\n");
  oseq_destroy(a), oseq_destroy(b);

  // plain, then drafted: the same tokens
  OsmosSeq* p = oseq_create(e, g_err);
  std::vector<int32_t> plain = {read(e, p, pa)};
  while (plain.size() < kGen) {
    int32_t y = plain.back();
    CHECK(osmos_batch(e, &p, &y, 1, g_err) && osmos_greedy(e, 0, 1, &y, g_err));
    plain.push_back(y);
  }
  oseq_destroy(p);
  OsmosSeq* s = oseq_create(e, g_err);
  std::vector<int32_t> out = {read(e, s, pa)};
  CHECK(oseq_seed(s, kPrompt > facts.seed_window ? (int32_t) (kPrompt - facts.seed_window) : 0, kPrompt, g_err));
  int drafted = 0, accepted = 0;
  while (out.size() < kGen) {
    int32_t d[OSMOS_MAX_BLOCK], toks[OSMOS_MAX_BLOCK + 1], pk[OSMOS_MAX_BLOCK + 1];
    float cf[OSMOS_MAX_BLOCK];
    const int k = oseq_draft(s, out.back(), OSMOS_DRAFT_CONF_MIN, OSMOS_DRAFT_CUM_MIN, d, cf, g_err);
    CHECK(k >= 0);
    const int32_t pos = oseq_position(s);
    toks[0] = out.back();
    for (int i = 0; i < k; i++) toks[i + 1] = d[i];
    std::vector<OsmosSeq*> ss(k + 1, s);
    CHECK(osmos_batch(e, ss.data(), toks, (uint32_t) k + 1, g_err) && osmos_greedy(e, 0, (uint32_t) k + 1, pk, g_err));
    int keep = 1;
    while (keep <= k && d[keep - 1] == pk[keep - 1]) keep++;
    for (int i = 0; i < keep - 1; i++) out.push_back(d[i]);
    out.push_back(pk[keep - 1]);
    CHECK(oseq_rollback(s, (uint32_t) (k + 1 - keep)) && oseq_seed(s, pos, pos + keep, g_err));
    drafted += k, accepted += keep - 1;
  }
  out.resize(kGen);
  CHECK(out == plain);
  std::printf("  %u tokens drafted the bits of plain greedy: %d drafted, %d accepted\n", kGen, drafted, accepted);
  CHECK(accepted > 0);
  oseq_destroy(s);
  osmos_destroy(e);
  osmos_close(m);
  return true;
}

int main() {
  std::printf("[test_ink2_seed_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_ink2_seed_gpu] skipped: no GPU\n");
    return 77;
  }
  const char* home = std::getenv("HOME");
  const std::string h = home ? home : "";
  const std::string small = std::getenv("OSMOS_INK2_PACK") ? std::getenv("OSMOS_INK2_PACK") : h + "/Models/osmos/osmos-evo-inkling",
                    mini = std::getenv("OSMOS_INK2_MINI") ? std::getenv("OSMOS_INK2_MINI") : h + "/Models/inkling2-mini/osmos";
  const bool a = pack("Inkling-2-Small", small + "/Inkling-2-Small-UD-IQ1_S-00001-of-00005.gguf",
                      small + "/ink2-mtp-q5k.gguf", small + "/osmos-picks.bin");
  const bool b = pack("Inkling-2-Mini", mini + "/ink2-mini-00001-of-00002.gguf", mini + "/ink2-mini-mtp-bf16.gguf", "");
  if (!a && !b) {
    std::printf("[test_ink2_seed_gpu] skipped: neither model here\n");
    return 77;
  }
  std::printf("[test_ink2_seed_gpu] ok\n");
  return 0;
}
