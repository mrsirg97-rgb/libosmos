// a family's failure through the engine names the family once, its cause after it (osmos/err.h: the dispatch's
// osmos_context): an option the family refuses, a step past its logits' rows, a pick of a row not kept - each err
// "<arch>: <cause>", the arch once; the engine's own refusals (a token outside the vocabulary) no family's. each small
// pack here; 77 if none.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "small_packs.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

static int count(const char* s, const char* w) {
  int n = 0;
  for (const char* p = s; (p = std::strstr(p, w)); p += std::strlen(w)) n++;
  return n;
}

static void named_once(const char* arch, const char* what, const char* err) {
  char want[64];
  std::snprintf(want, sizeof want, "%s: ", arch);
  std::printf("    %s: %s\n", what, err);
  CHECK(!std::strncmp(err, want, std::strlen(want)) && count(err, want) == 1 && std::strlen(err) > std::strlen(want));
}

static bool one(const SmallPack& p) {
  if (!small_pack_here(p)) return std::printf("  %s: no pack, skipped\n", p.name), false;
  const std::string path = small_pack_path(p);
  char err[OSMOS_ERR] = "";
  OsmosFiles files = {path.c_str(), nullptr, nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  CHECK(m);
  const char* arch = osmos_facts(m).family;
  std::printf("  %s (%s):\n", p.name, arch);
  if (!std::strcmp(arch, "lfm3")) {  // an option this family refuses: osmos1.7b's every weight resident, kv_q8
    OsmosOptions bad = small_pack_options(m, 1024);
    bad.kv_q8 = true;
    err[0] = 0;
    CHECK(!osmos_create(m, &bad, err));
  } else {  // the others' LSA maps (osmos1.7b's alone)
    OsmosFiles lsa = files;
    lsa.lsa = path.c_str();
    err[0] = 0;
    CHECK(!osmos_open(&lsa, err));
  }
  named_once(arch, "a refused option", err);
  const OsmosOptions o = small_pack_options(m, 1024);
  Osmos* e = osmos_create(m, &o, err);
  CHECK(e);
  OsmosSeq* s = oseq_create(e, err);
  CHECK(s);
  std::vector<OsmosSeq*> ss(1024, s);
  std::vector<int32_t> t(1024, 100);
  err[0] = 0;
  CHECK(!osmos_batch(e, ss.data(), t.data(), 1024, err));  // (every row kept: past the logits' rows)
  named_once(arch, "a step past its logits' rows", err);
  oseq_destroy(s);  // (a sequence after a refused step: Inkling's failed, osmos1.7b's as it was - phase 3's contract)
  CHECK((s = oseq_create(e, err)));
  for (auto& q : ss) q = s;
  CHECK(oseq_prompt(s, 16) && osmos_batch(e, ss.data(), t.data(), 16, err));
  int32_t id = 0;
  err[0] = 0;
  CHECK(!osmos_greedy(e, 0, 1, &id, err));  // (a prompt's end told: row 0's logits not kept)
  named_once(arch, "a pick of a row not kept", err);
  const int32_t out[1] = {-1};
  err[0] = 0;
  CHECK(!osmos_batch(e, ss.data(), out, 1, err) && !std::strstr(err, arch));  // the engine's own: no family's
  std::printf("    the engine's own: %s\n", err);
  oseq_destroy(s);
  osmos_destroy(e);
  osmos_close(m);
  return true;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_engine_errors_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) return std::printf("[test_engine_errors_gpu] skipped: no GPU\n"), 77;
  int ran = 0;
  for (const SmallPack& p : kSmallPacks) ran += one(p);
  if (!ran) return std::printf("[test_engine_errors_gpu] skipped: no pack\n"), 77;
  std::printf("[test_engine_errors_gpu] ok\n");
  return 0;
}
