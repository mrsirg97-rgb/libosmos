// the ledger, a component (osmos/mem/ledger.h): its rules at a room it is given (any machine alike) - each allocation
// counted, one past the room less FIT_HEADROOM refused before it is taken (nothing taken, why kept), a free given back,
// an allocator not HIP's reserved, kept and released alike, a failed allocation not counted, any number of allocations
// given back (40,000), two ledgers apart (one's free never another's count), ledger_read re-anchoring its room on
// what it holds, taken back to zero when all is freed; a NULL ledger as HIP's own. needs a GPU; 77 without.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "osmos/mem/ledger.h"
#include "osmos/run/fit.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint64_t kMiB = 1ull << 20;

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_ledger_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) return std::printf("[test_ledger_gpu] skipped: no GPU\n"), 77;
  // a NULL ledger: HIP's own, nothing refused or counted
  CHECK(ledger_left(nullptr) == UINT64_MAX && ledger_taken(nullptr) == 0);
  void* p = nullptr;
  CHECK(ledger_malloc(nullptr, &p, 64 * kMiB) == hipSuccess && p && ledger_free(nullptr, p) == hipSuccess);

  // a room of the headroom and 256 MiB: 256 MiB may be taken, no more
  OsmosLedger* l = ledger_create_room(FIT_HEADROOM + 256 * kMiB, 8ull << 30);
  CHECK(l && ledger_left(l) == 256 * kMiB && ledger_taken(l) == 0);
  void *a = nullptr, *b = nullptr, *h = nullptr;
  CHECK(ledger_malloc(l, &a, 128 * kMiB) == hipSuccess && ledger_left(l) == 128 * kMiB);
  CHECK(ledger_host_malloc(l, &h, 64 * kMiB, 0) == hipSuccess && ledger_left(l) == 64 * kMiB);  // pinned host alike
  // past the room: refused before it is taken, why kept
  CHECK(ledger_malloc(l, &b, 65 * kMiB) == hipErrorOutOfMemory && ledger_left(l) == 64 * kMiB);
  CHECK(std::strstr(ledger_why(l), "GiB asked"));
  // a free given back
  CHECK(ledger_free(l, a) == hipSuccess && ledger_left(l) == 192 * kMiB);
  CHECK(ledger_host_free(l, h) == hipSuccess && ledger_left(l) == 256 * kMiB && ledger_taken(l) == 0);
  // an allocator not HIP's: reserved, kept, released; refused past the room
  CHECK(!ledger_reserve(l, 257 * kMiB) && ledger_left(l) == 256 * kMiB);
  CHECK(ledger_reserve(l, 100 * kMiB) && ledger_left(l) == 156 * kMiB);
  static char token;  // (a pointer of the other allocator's)
  ledger_keep(l, &token, 100 * kMiB);
  ledger_release(l, &token);
  CHECK(ledger_left(l) == 256 * kMiB);
  CHECK(ledger_reserve(l, 50 * kMiB));
  ledger_unreserve(l, 50 * kMiB);  // (its allocation failed)
  CHECK(ledger_left(l) == 256 * kMiB);
  ledger_destroy(l);

  // a failed allocation (past the device) not counted
  OsmosLedger* big = ledger_create_room(UINT64_MAX / 2, 0);
  const uint64_t before = ledger_left(big);
  void* huge = nullptr;
  CHECK(ledger_malloc(big, &huge, (size_t) 1 << 50) != hipSuccess && ledger_left(big) == before);
  ledger_destroy(big);

  // two ledgers apart: memory one took, freed through another (another engine's), never touches the other's count
  OsmosLedger *x = ledger_create_room(FIT_HEADROOM + 512 * kMiB, 0), *y = ledger_create_room(FIT_HEADROOM + 512 * kMiB, 0);
  void *c = nullptr, *d = nullptr;
  CHECK(ledger_malloc(x, &c, 32 * kMiB) == hipSuccess && ledger_malloc(y, &d, 64 * kMiB) == hipSuccess);
  CHECK(ledger_free(y, c) == hipSuccess && ledger_left(y) == 448 * kMiB && ledger_left(x) == 480 * kMiB);
  CHECK(ledger_free(y, d) == hipSuccess && ledger_left(y) == 512 * kMiB);
  ledger_destroy(x), ledger_destroy(y);

  // any number of allocations given back, freed out of order (deletions inside the table's probe runs)
  {
    constexpr int kMany = 40000;
    static void* many[kMany];
    OsmosLedger* m = ledger_create_room(FIT_HEADROOM + (1ull << 30), 0);
    for (int i = 0; i < kMany; i++) CHECK(ledger_malloc(m, &many[i], 4096) == hipSuccess);
    CHECK(ledger_taken(m) == (uint64_t) kMany * 4096);
    for (int i = kMany - 1; i >= 0; i -= 2) CHECK(ledger_free(m, many[i]) == hipSuccess);
    for (int i = kMany - 2; i >= 0; i -= 2) CHECK(ledger_free(m, many[i]) == hipSuccess);
    CHECK(ledger_taken(m) == 0 && ledger_left(m) == 1ull << 30);
    ledger_destroy(m);
  }

  // a ledger read from this machine: its room re-anchored on what it holds (what it holds is not room lost)
  char err[OSMOS_ERR] = "";
  OsmosLedger* r = ledger_create(8ull << 30, err);
  CHECK(r);
  void* e = nullptr;
  CHECK(ledger_malloc(r, &e, 256 * kMiB) == hipSuccess);
  const uint64_t held = ledger_left(r);
  ledger_read(r);
  const uint64_t again = ledger_left(r);
  CHECK((again > held ? again - held : held - again) < 256 * kMiB);  // (the machine's own drift between two reads)
  CHECK(ledger_free(r, e) == hipSuccess && ledger_taken(r) == 0);
  ledger_destroy(r);
  std::printf("  rules at a given room: counted, refused before taken, given back, reserved / kept / released, a failed "
              "allocation not counted, two ledgers apart, 40,000 given back, a read on what it holds\n");
  std::printf("[test_ledger_gpu] ok\n");
  return 0;
}
