// Hy4's indexer scratch (the spread selection's) the forward's own, made whole when it is built (SCOPE.md): its
// allocation failing (the fault seam, osmos/err_hip.h) refuses the build, saying so - never a forward made without it,
// never an op launched over half of it (0.16.0: function statics in kernels/hy4.hip, made at first use - the first
// allocation held and the second refused, the next call launched with a null; never freed; shared by two engines).
// needs a gfx1151 and the hy4-flash pack; 77 without.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>

extern "C" {
#include "osmos/err_hip.h"
#include "osmos/hy4/forward.h"
}
#include "small_packs.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

static bool failing;  // the seam: the scratch's allocation fails while set
extern "C" bool osmos_hip_fault(const char* what) { return failing && !std::strcmp(what, HY4_IX_SPREAD); }

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_hy4_scratch_gpu] start\n");
  const SmallPack p = {"hy4-flash", "/Models/osmos/hy4-flash/Hy4-flash-Q3_K-00001-of-00002.gguf", nullptr};
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev || !small_pack_here(p))
    return std::printf("[test_hy4_scratch_gpu] skipped: no GPU or hy4-flash\n"), 77;
  const std::string path = small_pack_path(p);
  char err[OSMOS_ERR] = "";
  OsmosFiles files = {path.c_str(), nullptr, nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  CHECK(m);
  OsmosOptions o = small_pack_options(m, 1024);
  o.budget = 1;  // (its smallest build: every expert streamed)
  failing = true;
  Osmos* e = osmos_create(m, &o, err);
  failing = false;
  std::printf("  the scratch's allocation failing: %s%s%s\n", e ? "BUILT" : "refused", e ? "" : " - ", e ? "" : err);
  CHECK(!e && std::strstr(err, HY4_IX_SPREAD));
  CHECK((e = osmos_create(m, &o, err)));  // and holding: built
  osmos_destroy(e);
  osmos_close(m);
  std::printf("[test_hy4_scratch_gpu] ok\n");
  return 0;
}
