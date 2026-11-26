// Inkling-2-Small's bits held to a release's: its pack (osmos-evo-inkling) through the engine as osmos-serve runs it
// - a prompt read in 4,096-token steps, then greedy decode steps - each row's logits hashed (FNV-1a 64 over their
// bytes) and the run's hash against kRef, 0.11.3's (this test built on 88df172's library). an 8,448-token prompt and a
// 2,099-token one, the ids a fixed generator's. a change that moves any of Small's sums (0.12.0's BF16 path did:
// its sliding layers' sink projections are BF16) fails here, whether or not a greedy token moves. the pack's first
// shard: OSMOS_TEST_INK2_SMALL, else ~/Models/osmos/osmos-evo-inkling's; skipped (77) without it or a GPU.
// OSMOS_TEST_PRINT=1: the hashes printed, not checked (a fixture's making). each case twice: the f16 path (the int8
// decode off: 0.11.3's bits) and the run's own (the int8 decode's: 0.18.0's)
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <memory>
#include <vector>

extern "C" {
#include "osmos/gpu/q8.h"
#include "osmos/run/engine.h"
}

#include "q8_off.h"
#include "test_fit.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr const char* kPack = "%s/Models/osmos/osmos-evo-inkling/Inkling-2-Small-UD-IQ1_S-00001-of-00005.gguf";
constexpr uint32_t kStep = 4096, kDecode = 16, kCtx = 16384;
constexpr uint64_t kBudget = 64ull << 30;  // the experts' room: Small's bits the same at any budget (streamed or not)
struct Case {
  uint32_t n;
  uint64_t seed, ref, ref8;
};
// ref: the f16 path's, 0.11.3's (88df172), measured 2026-11-21 - the same hashes as this build's after 22ebd95, held
// with the int8 decode off. ref8: the int8 decode's (0.18.0, measured 2026-11-25 after its KL: 0.0072 mean, top-1
// 96.9%; its prompt the f16 path's), held on the run's own path
static const Case kCases[] = {{8448, 1, 0x1f6a5baeee8ca834ull, 0xe40e711b95007553ull},
                              {2099, 2, 0xa970965e5b70be00ull, 0xfe347c36e57012daull}};

// a row's logits (osmos_logits': device memory) hashed into h
static uint64_t fnv_row(const float* dev, uint32_t V, uint64_t h) {
  std::vector<uint8_t> b((size_t) V * 4);
  CHECK(dev && hipMemcpy(b.data(), dev, b.size(), hipMemcpyDeviceToHost) == hipSuccess);
  for (uint8_t x : b) h = (h ^ x) * 1099511628211ull;
  return h;
}

// a case's prompt in osmos-serve's steps, then greedy decode a token a step: every kept row's logits hashed in order
static uint64_t run_case(Osmos* e, const Case& c, uint32_t V, char* err) {
  std::vector<int32_t> ids(c.n);
  uint64_t r = c.seed;  // ids below 100,000: ordinary tokens of Small's vocabulary
  for (auto& t : ids) r = r * 6364136223846793005ull + 1442695040888963407ull, t = (int32_t) ((r >> 33) % 100000);
  OsmosSeq* s = oseq_create(e, err);
  CHECK(s && oseq_prompt(s, c.n));
  const std::vector<OsmosSeq*> ss(kStep, s);  // (a step's sequence a token: osmos_read's seqs[i])
  uint64_t h = 1469598103934665603ull;
  uint32_t at = 0, len = 0;
  for (; at < c.n; at += len) {
    len = c.n - at < kStep ? c.n - at : kStep;
    CHECK(osmos_read(e, ss.data(), ids.data() + at, len, err) || (std::fprintf(stderr, "  %s\n", err), false));
  }
  int32_t y = 0;
  CHECK(osmos_greedy(e, len - 1, 1, &y, err));
  h = fnv_row(osmos_logits(e, len - 1), V, h);
  for (uint32_t k = 0; k < kDecode; k++) {
    CHECK(osmos_batch(e, &s, &y, 1, err) && osmos_greedy(e, 0, 1, &y, err));
    h = fnv_row(osmos_logits(e, 0), V, h);
  }
  oseq_destroy(s);
  return h;
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_ink2_small_ref_gpu] start\n");
  char path[512];
  if (std::getenv("OSMOS_TEST_INK2_SMALL"))
    std::snprintf(path, sizeof path, "%s", std::getenv("OSMOS_TEST_INK2_SMALL"));
  else
    std::snprintf(path, sizeof path, kPack, std::getenv("HOME") ? std::getenv("HOME") : "");
  int ndev = 0;
  if (access(path, R_OK) || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_ink2_small_ref_gpu] skipped: no Inkling-2-Small pack (%s) or no GPU\n", path);
    return 77;
  }
  const bool print = std::getenv("OSMOS_TEST_PRINT") != nullptr;
  char err[OSMOS_ERR] = "";
  const OsmosFiles files = {path, nullptr, nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  if (!m) std::fprintf(stderr, "  %s\n", err);
  CHECK(m);
  OsmosOptions o = {};
  o.max_ctx = kCtx, o.max_tokens = kStep, o.budget = kBudget;
  Osmos* e = test_osmos(m, o, err);  // (its bits any budget's: the test's own, or this machine's fill)
  if (!e) std::fprintf(stderr, "  %s\n", err);
  CHECK(e);
  const uint32_t V = osmos_facts(m).vocab;
  int bad = 0;
  for (const Case& c : kCases)
    for (const bool f16 : {true, false}) {
      std::unique_ptr<Q8Off> off = f16 ? std::make_unique<Q8Off>() : nullptr;
      const uint64_t h = run_case(e, c, V, err), want = f16 || !q8_enabled() ? c.ref : c.ref8;
      const char* path = f16 || !q8_enabled() ? "f16" : "int8 decode";
      if (print) {
        std::printf("  %u tokens, %s: hash 0x%016llxull\n", c.n, path, (unsigned long long) h);
      } else {
        std::printf("  a %u-token prompt and %u greedy steps, %s: the logits' hash %016llx %s\n", c.n, kDecode, path,
                    (unsigned long long) h, h == want ? "(held)" : "NOT the held one");
        bad += h != want;
      }
    }
  osmos_destroy(e);
  osmos_close(m);
  CHECK(!bad);
  std::printf("[test_ink2_small_ref_gpu] ok\n");
  return 0;
}
