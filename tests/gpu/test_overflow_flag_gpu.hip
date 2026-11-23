// the overflow flag's reset, one shape every family's step shares (osmos/gpu/overflow.h): after a step that went past
// f16's range the flag zeroed for the next; a reset that fails (through the fault seam, osmos/err_hip.h) owed - the
// step's own cause said, the next step's start trying it again and refusing with that cause while it fails, running
// once it holds (0.16.0: nine unchecked resets - a failed one left the flag set, every later step refused as "an
// activation past f16's range" forever). needs a GPU; 77 without.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "osmos/err_hip.h"
#include "osmos/gpu/overflow.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

static int failing;  // the seam: how many more calls of the flag's reset fail
extern "C" bool osmos_hip_fault(const char* what) {
  if (failing && !std::strcmp(what, OVERFLOW_RESET)) return failing--, true;
  return false;
}

static uint32_t flag_of(void* dev) {
  uint32_t v = 0;
  CHECK(hipMemcpy(&v, dev, 4, hipMemcpyDeviceToHost) == hipSuccess);
  return v;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_overflow_flag_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) return std::printf("[test_overflow_flag_gpu] skipped: no GPU\n"), 77;
  void* dev = nullptr;
  CHECK(hipMalloc(&dev, 8) == hipSuccess && hipMemset(dev, 0xff, 8) == hipSuccess);  // (a step went past f16's range)
  OverflowFlag o = {dev, 8, false};
  char err[OSMOS_ERR] = "";
  // a reset that holds: the flag zero, nothing owed
  CHECK(overflow_reset(&o, err) && !o.owed && flag_of(dev) == 0);
  // a reset that fails: owed, its cause said, the flag still set
  CHECK(hipMemset(dev, 0xff, 8) == hipSuccess);
  failing = 1;
  CHECK(!overflow_reset(&o, err) && o.owed && std::strstr(err, OVERFLOW_RESET));  // (the seam fails the call's report:
                                                                                  //  the flag itself zeroed or not)
  std::printf("  a failed reset: owed (%s)\n", err);
  // the next step's start: tried again - still failing, refused with that cause; holding, the step runs
  failing = 1, osmos_clear(err);
  CHECK(!overflow_settle(&o, err) && o.owed && std::strstr(err, OVERFLOW_RESET) && !std::strstr(err, "f16"));
  std::printf("  the next step while it fails: refused (%s)\n", err);
  osmos_clear(err);
  CHECK(overflow_settle(&o, err) && !o.owed && flag_of(dev) == 0 && !err[0]);
  std::printf("  the next step once it holds: run, the flag zero\n");
  // nothing owed: a step's start does nothing (no GPU call)
  failing = 5;
  CHECK(overflow_settle(&o, err) && failing == 5);
  failing = 0;
  CHECK(hipFree(dev) == hipSuccess);
  std::printf("[test_overflow_flag_gpu] ok\n");
  return 0;
}
