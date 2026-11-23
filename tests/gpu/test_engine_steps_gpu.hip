// a step's most tokens one rule on every family (osmos/run/engine.h OsmosOptions.max_tokens): 0 the family's own, and
// never more than the context - a build at max_tokens past max_ctx made, its step the context's (0.15.3: Qwen alone
// refused it, and at 0 sized its step to the whole context - a 512K step's buffers). each pack here; 77 if none.
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

static const SmallPack kPacks[] = {kSmallPacks[0], kSmallPacks[1],
                                   {"qwen38-ud-iq3_xxs", "/Models/osmos/qwen38-ud-iq3_xxs/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf", nullptr}};

static bool one(const SmallPack& p) {
  if (!small_pack_here(p)) return std::printf("  %s: no pack, skipped\n", p.name), false;
  const std::string path = small_pack_path(p);
  char err[OSMOS_ERR] = "";
  OsmosFiles files = {path.c_str(), nullptr, nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  CHECK(m);
  for (uint32_t steps : {4096u, 0u}) {
    OsmosOptions o = small_pack_options(m, 1024);
    o.max_tokens = steps;
    Osmos* e = osmos_create(m, &o, err);
    std::printf("  %s: max_ctx 1024, max_tokens %u: %s%s\n", p.name, steps, e ? "built" : "refused: ", e ? "" : err);
    CHECK(e);
    OsmosSeq* s = oseq_create(e, err);
    CHECK(s);
    static int32_t t[1024];
    for (int i = 0; i < 1024; i++) t[i] = 100 + i % 50;
    static OsmosSeq* ss[1024];
    for (int i = 0; i < 1024; i++) ss[i] = s;
    CHECK(oseq_prompt(s, 1024));  // (a prompt's end told: its last row's logits kept, as a server reads one)
    const bool ran = osmos_batch(e, ss, t, 1024, err);
    if (!ran) std::printf("  %s: the context in one step refused: %s\n", p.name, err);
    CHECK(ran && oseq_position(s) == 1024);  // the whole context, one step
    oseq_destroy(s);
    osmos_destroy(e);
  }
  osmos_close(m);
  return true;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_engine_steps_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) return std::printf("[test_engine_steps_gpu] skipped: no GPU\n"), 77;
  int ran = 0;
  for (const SmallPack& p : kPacks) ran += one(p);
  if (!ran) return std::printf("[test_engine_steps_gpu] skipped: no pack\n"), 77;
  std::printf("[test_engine_steps_gpu] ok\n");
  return 0;
}
