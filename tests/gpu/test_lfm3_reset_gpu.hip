// osmos1.7b's sequence after a refused load (osmos/run/engine.h: "load refuses ... the sequence then empty"): its
// convolutions' states zeroed - and where that reset fails (the fault seam, osmos/err_hip.h), the sequence broken, its
// next step refused with the cause, never "empty" over the last conversation's states (0.15.3: the reset's memsets
// unchecked). needs a gfx1151 and osmos1.7b's pack; 77 without.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/err_hip.h"
#include "osmos/lfm3/forward.h"
}
#include "small_packs.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

static bool failing;  // the seam: the sequence's reset fails while set
extern "C" bool osmos_hip_fault(const char* what) { return failing && !std::strcmp(what, LFM3_SEQ_RESET); }

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_lfm3_reset_gpu] start\n");
  int ndev = 0;
  const SmallPack& p = kSmallPacks[0];
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev || !small_pack_here(p))
    return std::printf("[test_lfm3_reset_gpu] skipped: no GPU or osmos1.7b\n"), 77;
  const std::string path = small_pack_path(p);
  char err[OSMOS_ERR] = "";
  OsmosFiles files = {path.c_str(), nullptr, nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  CHECK(m);
  const OsmosOptions o = small_pack_options(m, 1024);
  Osmos* e = osmos_create(m, &o, err);
  CHECK(e);
  OsmosSeq* s = oseq_create(e, err);
  CHECK(s);
  OsmosSeq* ss[16];
  int32_t t[16];
  for (int i = 0; i < 16; i++) ss[i] = s, t[i] = 100 + 7 * i;
  CHECK(osmos_batch(e, ss, t, 16, err) && oseq_position(s) == 16);
  const std::vector<uint8_t> bad(64, 0x5a);  // (not a state: the load refused)
  // the reset holding: refused, the sequence empty and stepping from 0
  CHECK(!oseq_load(s, bad.data(), bad.size(), err) && oseq_position(s) == 0);
  CHECK(osmos_batch(e, ss, t, 16, err) && oseq_position(s) == 16);
  std::printf("  a refused load, its reset holding: empty, stepped from 0\n");
  // the reset failing: the sequence broken - its next step refused with the cause
  failing = true;
  CHECK(!oseq_load(s, bad.data(), bad.size(), err));
  failing = false;
  err[0] = 0;
  const bool ran = osmos_batch(e, ss, t, 16, err);
  std::printf("  a refused load, its reset failing: the next step %s%s%s\n", ran ? "RAN" : "refused", err[0] ? " - " : "",
              err);
  CHECK(!ran && std::strstr(err, "a failed"));
  oseq_destroy(s);
  osmos_destroy(e);
  osmos_close(m);
  std::printf("[test_lfm3_reset_gpu] ok\n");
  return 0;
}
