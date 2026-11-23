// the engine's room one rule on every family (osmos/run/engine.h: osmos_room): the GPU's free memory, or the system's
// RAM past what --fill keeps free, the less - no fill the most room; and a fill more than the machine has refuses the
// build, saying why (the engine's ledger: no allocation past the RAM --fill keeps - the operator's fail-closed
// --fill) (0.16.0: osmos1.7b and the 27B reported the GPU's free memory alone: no RAM bound, no --fill). each small
// pack here; 77 if none.
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

static uint64_t room_at(const OsmosModel* m, bool set, uint64_t fill) {
  char err[OSMOS_ERR] = "";
  OsmosOptions o = small_pack_options(m, 1024);
  o.fill_set = set, o.fill = fill;
  Osmos* e = osmos_create(m, &o, err);
  CHECK(e);
  uint64_t r = 0;
  CHECK(osmos_room(e, &r));
  osmos_destroy(e);
  return r;
}

static bool one(const SmallPack& p) {
  if (!small_pack_here(p)) return std::printf("  %s: no pack, skipped\n", p.name), false;
  const std::string path = small_pack_path(p);
  char err[OSMOS_ERR] = "";
  OsmosFiles files = {path.c_str(), nullptr, nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  CHECK(m);
  const uint64_t def = room_at(m, false, 0), none = room_at(m, true, 0);
  OsmosOptions huge = small_pack_options(m, 1024);
  huge.fill_set = true, huge.fill = 1000ull << 30;
  CHECK(!osmos_create(m, &huge, err) && std::strstr(err, " asked, "));
  std::printf("  %s: room %.2f GiB (8 kept free), %.2f (--fill 0); --fill 1000 refused (%s)\n", p.name,
              def / 1073741824.0, none / 1073741824.0, err);
  CHECK(none + (256ull << 20) >= def);  // (the system's count moves a little between builds)
  osmos_close(m);
  return true;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_engine_room_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) return std::printf("[test_engine_room_gpu] skipped: no GPU\n"), 77;
  int ran = 0;
  for (const SmallPack& p : kSmallPacks) ran += one(p);
  if (!ran) return std::printf("[test_engine_room_gpu] skipped: no pack\n"), 77;
  std::printf("[test_engine_room_gpu] ok\n");
  return 0;
}
