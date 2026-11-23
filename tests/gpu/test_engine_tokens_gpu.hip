// the engine refuses a token outside the vocabulary before any family reads it (osmos/run/engine.h: osmos_batch) -
// past the last id, at it, and negative: refused, saying so, the sequence where it was; a token inside, read (0.16.0:
// Inkling, the 27B and Qwen embedded any id - a read past the table on the GPU). each small pack here; 77 if none.
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

static bool one(const SmallPack& p) {
  if (!small_pack_here(p)) return std::printf("  %s: no pack, skipped\n", p.name), false;
  const std::string path = small_pack_path(p);
  char err[OSMOS_ERR] = "";
  OsmosFiles files = {path.c_str(), nullptr, nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  CHECK(m);
  const int32_t V = (int32_t) osmos_facts(m).vocab;
  const OsmosOptions o = small_pack_options(m, 1024);
  Osmos* e = osmos_create(m, &o, err);
  if (!e) std::fprintf(stderr, "  %s\n", err);
  CHECK(e);
  OsmosSeq* s = oseq_create(e, err);
  CHECK(s);
  OsmosSeq* ss[4] = {s, s, s, s};
  for (int32_t bad : {V, V + 1000, -1, INT32_MIN}) {
    const int32_t t[4] = {100, 101, bad, 103};  // (one bad id among good ones: the whole step refused)
    err[0] = 0;
    CHECK(!osmos_batch(e, ss, t, 4, err) && std::strstr(err, "vocabulary"));
    CHECK(oseq_position(s) == 0);  // nothing read
  }
  const int32_t good[4] = {100, 101, V - 1, 0};
  CHECK(osmos_batch(e, ss, good, 4, err) && oseq_position(s) == 4);
  std::printf("  %s: ids %d, %d, -1 and INT32_MIN refused, nothing read; 0 .. %d read\n", p.name, V, V + 1000, V - 1);
  oseq_destroy(s);
  osmos_destroy(e);
  osmos_close(m);
  return true;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_engine_tokens_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) return std::printf("[test_engine_tokens_gpu] skipped: no GPU\n"), 77;
  int ran = 0;
  for (const SmallPack& p : kSmallPacks) ran += one(p);
  if (!ran) return std::printf("[test_engine_tokens_gpu] skipped: no pack\n"), 77;
  std::printf("[test_engine_tokens_gpu] ok\n");
  return 0;
}
