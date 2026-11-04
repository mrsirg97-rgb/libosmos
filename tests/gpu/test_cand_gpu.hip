// the candidate blocks in the forward (DESIGN.md, "past 16K"): a prompt past them (22K tokens of text in steps of
// 2,048, then decode steps), the source layer's bits probed ("cand": every row of the sub-batch, its first position)
// and each later index layer's selection ("sel": its first token's rows - past the decoder's compaction, the first
// kept token's - and its position): every selected row in a block its own token's bits keep, the token found in the
// source's bits by position - so the bits are applied, and aligned after the compaction. masked selections counted (a
// test that never saw the mask bite proves nothing). needs a gfx1151 and a model (test_model.h); skipped (77) without.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "substrate/model/gguf.h"
#include "substrate/model/weights.h"
#include "substrate/run/forward.h"
}
#include "test_model.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kMaxTokens = 2048, kMaxCtx = 32768, kPrompt = 22000, kDecode = 4;  // (22,000: a sub-batch past the blocks whose tokens are only partly within the decoder's reach of the end - the compaction's gather of the bits not the identity)

struct Probe {
  int cand_layer = -1, cand_pos = -1;
  uint32_t words = 0, block = 0;
  std::vector<uint32_t> cand;  // the source layer's bits, this step's sub-batch
  uint64_t checked = 0, sels = 0, steps_masked = 0;
  int bad = 0;
};

static void probe(const char* step, int layer, int index, void* dev, uint64_t n, bool, void* arg) {
  Probe* p = (Probe*) arg;
  if (!std::strcmp(step, "cand")) {
    p->cand_layer = layer, p->cand_pos = index, p->cand.resize(n);
    CHECK(hipMemcpy(p->cand.data(), dev, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
    p->steps_masked++;
    return;
  }
  if (std::strcmp(step, "sel") || p->cand_layer < 0 || layer <= p->cand_layer || !n) return;
  const int64_t row = (int64_t) index - p->cand_pos;  // the token's row in the source's bits (one sequence, ascending)
  CHECK(row >= 0 && (uint64_t) (row + 1) * p->words <= p->cand.size());
  std::vector<int32_t> sel(n);
  CHECK(hipMemcpy(sel.data(), dev, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  const uint32_t* bits = &p->cand[(size_t) row * p->words];
  for (int32_t c : sel) {
    const uint32_t j = (uint32_t) c / p->block;
    p->checked++;
    if (!((bits[j / 32] >> (j % 32)) & 1u) && p->bad++ < 5)
      std::printf("  layer %d, position %d: row %d outside its candidate blocks\n", layer, index, c);
  }
  p->sels++;
}

int main() {
  std::printf("[test_cand_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  const char* home = std::getenv("HOME");
  const std::string text = std::string(home ? home : "") + "/bench/substrate-prompts/calib16k.txt";
  FILE* tf = std::fopen(text.c_str(), "r");
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0 || !tf) {
    std::printf("[test_cand_gpu] skipped: no model, GPU or the bench's text (%s)\n", text.c_str());
    return 77;
  }
  std::vector<int32_t> ids;
  for (long v; std::fscanf(tf, "%ld,", &v) == 1;) ids.push_back((int32_t) v);
  std::fclose(tf);
  const size_t base = ids.size();
  while (ids.size() < kPrompt) ids.push_back(ids[ids.size() % base]);  // (the text again past its end)
  ids.resize(kPrompt);
  const std::string dir = std::string(path).substr(0, std::string(path).rfind('/') + 1);
  char gerr[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "", err[FORWARD_ERR] = "";
  Gguf* g = gguf_create(path, gerr);
  Gguf* eg = which == TestModel::kGsq ? gguf_create((dir + "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf").c_str(), gerr)
                                      : nullptr;
  Weights* w = weights_create(g, eg, werr);
  CHECK(w);
  const ForwardOptions o = {(int) w->geo.n_layers, kMaxCtx, 100ull << 30, nullptr, kMaxTokens};
  Forward* f = forward_create_with(w, &o, err);
  CHECK(f && forward_lane(f, 0, err));
  Probe p;
  p.block = (uint32_t) w->geo.cand_block;
  p.words = ((kMaxCtx / 1 + p.block - 1) / p.block + 31) / 32;  // (the forward's: its smallest ratio 1)
  forward_probe(f, probe, &p);
  Seq* s = seq_create(f, err);
  CHECK(s && seq_prompt(s, kPrompt));
  std::vector<Seq*> ss(kMaxTokens, s);
  for (uint32_t at = 0; at < kPrompt; at += kMaxTokens) {
    p.cand_layer = -1;
    const uint32_t c = kPrompt - at < kMaxTokens ? kPrompt - at : kMaxTokens;
    CHECK(forward_batch(f, ss.data(), ids.data() + at, c, err));
  }
  int32_t t;
  CHECK(forward_greedy(f, (kPrompt - 1) % kMaxTokens, 1, &t, err));
  for (uint32_t i = 0; i < kDecode; i++) {
    p.cand_layer = -1;
    CHECK(forward_step(f, s, t, err) && forward_greedy(f, 0, 1, &t, err));
  }
  std::printf("  %u-token prompt and %u decode steps: the bits built in %llu sub-batches; %llu selections at the index "
              "layers after the source (%llu rows) - %d outside their token's candidate blocks\n",
              kPrompt, kDecode, (unsigned long long) p.steps_masked, (unsigned long long) p.sels,
              (unsigned long long) p.checked, p.bad);
  CHECK(p.steps_masked > 0 && p.sels > 0 && p.bad == 0);
  seq_destroy(s);
  forward_destroy(f);
  weights_destroy(w);
  gguf_destroy(g);
  gguf_destroy(eg);
  std::printf("[test_cand_gpu] ok\n");
  return 0;
}
