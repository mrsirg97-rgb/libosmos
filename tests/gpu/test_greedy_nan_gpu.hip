// the engine's greedy pick refuses a row of NaN logits (osmos/run/engine.h: osmos_greedy) on every family here - its
// argmax's -1 never handed on as a token (0.16.0: Inkling's own pick returned true with id -1; the others refused).
// each family's pack a step, then its last row's logits set to NaN in place: the pick refused, saying so; the row's
// own pick before it a token in the vocabulary. needs a gfx1151; a family whose pack is not here skipped, 77 if none.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include "small_packs.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr uint32_t kCtx = 1024, kPrompt = 16;

static bool one(const SmallPack& p) {
  if (!small_pack_here(p)) return std::printf("  %s: no pack, skipped\n", p.name), false;
  const std::string path = small_pack_path(p);
  char err[OSMOS_ERR] = "";
  OsmosFiles files = {path.c_str(), nullptr, nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  if (!m) std::fprintf(stderr, "  %s\n", err);
  CHECK(m);
  const uint32_t V = osmos_facts(m).vocab;
  const OsmosOptions o = small_pack_options(m, kCtx);
  Osmos* e = osmos_create(m, &o, err);
  if (!e) std::fprintf(stderr, "  %s\n", err);
  CHECK(e);
  OsmosSeq* s = oseq_create(e, err);
  CHECK(s);
  std::vector<OsmosSeq*> ss(kPrompt, s);
  std::vector<int32_t> t(kPrompt);
  for (uint32_t i = 0; i < kPrompt; i++) t[i] = (int32_t) (100 + 7 * i);
  CHECK(osmos_batch(e, ss.data(), t.data(), kPrompt, err));
  int32_t id = -2;
  CHECK(osmos_greedy(e, kPrompt - 1, 1, &id, err) && id >= 0 && (uint32_t) id < V);  // the row's own pick: a token
  const float* row = osmos_logits(e, kPrompt - 1);
  CHECK(row);
  std::vector<float> nan(V, std::nanf(""));
  CHECK(hipMemcpy((void*) row, nan.data(), V * sizeof(float), hipMemcpyDefault) == hipSuccess);  // (device or pinned)
  id = -2, err[0] = 0;
  const bool picked = osmos_greedy(e, kPrompt - 1, 1, &id, err);
  std::printf("  %s: a NaN row's pick %s (id %d%s%s)\n", p.name, picked ? "ANSWERED" : "refused", id, err[0] ? ": " : "",
              err);
  CHECK(!picked && std::strstr(err, "NaN"));
  oseq_destroy(s);
  osmos_destroy(e);
  osmos_close(m);
  return true;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_greedy_nan_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) return std::printf("[test_greedy_nan_gpu] skipped: no GPU\n"), 77;
  int ran = 0;
  for (const SmallPack& p : kSmallPacks) ran += one(p);
  if (!ran) return std::printf("[test_greedy_nan_gpu] skipped: no pack\n"), 77;
  std::printf("[test_greedy_nan_gpu] ok\n");
  return 0;
}
