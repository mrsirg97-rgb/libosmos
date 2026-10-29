// the decoder skip (seq_prompt), bit for bit. a prompt of kTotal real ids (repeated) prefilled kChunk a step, whole
// and then told its end - its tokens before the decoder's reach skipping DeepSeek-V4.1's decoder - beside a second
// sequence decoding a token each step (the step's rows then compacted around it); then both decode kDecode tokens.
// every logit read - the prompt's last, the decoding sequence's every one, the decoded ones after - the whole run's.
// needs a gfx1151, ~/models/ds1017 and ~/substrate-oracle/ids32.txt; skipped (77) without them.
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "substrate/model/gguf.h"
#include "substrate/model/weights.h"
#include "substrate/gpu/mm.h"
#include "substrate/run/forward.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr uint64_t kBudget = 72ull << 30;
constexpr uint32_t kCtx = 4096, kTotal = 4096 - 64, kDecode = 4;
constexpr uint32_t kChunk = MM_MAX_TOKENS - 1;  // and the beside sequence's token: a whole step

struct Out {
  std::vector<std::vector<float>> logits;  // each read, in order
  double ms = 0;
};

static std::vector<float> row(Forward* f, size_t V, uint32_t i) {
  std::vector<float> l(V);
  CHECK(hipMemcpy(l.data(), forward_logits(f) + (size_t) i * V, V * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return l;
}

static Out play(Forward* f, size_t V, const int* ids, bool skip) {
  char err[FORWARD_ERR] = "";
  Seq* a = seq_create(f, err);  // the prompt
  Seq* b = seq_create(f, err);  // a conversation decoding beside it
  CHECK(a && b);
  if (skip) {
    CHECK(seq_prompt(a, kTotal));
    CHECK(!seq_prompt(a, kTotal - 1));  // lower than before: refused
    CHECK(!seq_prompt(a, kCtx + 1));
  }
  Out o;
  const auto t0 = std::chrono::steady_clock::now();
  int nb = 0;
  for (uint32_t p = 0; p < kTotal; p += kChunk) {
    const uint32_t k = kTotal - p < kChunk ? kTotal - p : kChunk;
    std::vector<Seq*> ss(k + 1, a);
    std::vector<int32_t> t(k + 1);
    ss[0] = b, t[0] = ids[(nb * 5) % 32];  // b's token first: a's kept ones then rows 1 .. of the step
    for (uint32_t i = 0; i < k; i++) t[i + 1] = ids[(p + i) % 32];
    CHECK(forward_batch(f, ss.data(), t.data(), k + 1, err));
    o.logits.push_back(row(f, V, 0)), nb++;
    if (p + k == kTotal) o.logits.push_back(row(f, V, k));
  }
  CHECK(hipDeviceSynchronize() == hipSuccess);
  o.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  for (uint32_t d = 0; d < kDecode; d++) {
    Seq* both[2] = {a, b};
    const int32_t t[2] = {ids[(kTotal + d * 7) % 32], ids[(nb + d) % 32]};
    CHECK(forward_batch(f, both, t, 2, err));
    o.logits.push_back(row(f, V, 0)), o.logits.push_back(row(f, V, 1));
  }
  seq_destroy(a);
  seq_destroy(b);
  return o;
}

int main() {
  std::printf("[test_skip_gpu] start\n");
  const char* home = std::getenv("HOME");
  const std::string mdir = std::string(home ? home : "") + "/models/ds1017/";
  const std::string idp = std::string(home ? home : "") + "/substrate-oracle/ids32.txt";
  int ndev = 0;
  if (access(mdir.c_str(), R_OK) != 0 || access(idp.c_str(), R_OK) != 0 || hipGetDeviceCount(&ndev) != hipSuccess ||
      !ndev) {
    std::printf("[test_skip_gpu] skipped: no ds1017, no ids, or no GPU\n");
    return 77;
  }
  int ids[32];
  FILE* idf = std::fopen(idp.c_str(), "r");
  CHECK(idf);
  for (int& i : ids) CHECK(std::fscanf(idf, "%d,", &i) == 1);
  std::fclose(idf);
  char err[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "", ferr[FORWARD_ERR] = "";
  Gguf* g = gguf_create((mdir + "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf").c_str(), err);
  Gguf* eg = gguf_create((mdir + "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf").c_str(), err);
  CHECK(g && eg);
  Weights* w = weights_create(g, eg, werr);
  CHECK(w);
  Forward* f = forward_create(w, w->geo.n_layers, kCtx, kBudget, ferr);
  CHECK(f);
  CHECK(!seq_prompt(nullptr, 1));
  const size_t V = (size_t) w->geo.n_vocab;
  const Out whole = play(f, V, ids, false), skip = play(f, V, ids, true);
  CHECK(whole.logits.size() == skip.logits.size());
  size_t same = 0;
  for (size_t i = 0; i < whole.logits.size(); i++) same += whole.logits[i] == skip.logits[i];
  std::printf("  %u prompt tokens, %u a step, a second sequence decoding beside: prefill %.0f ms whole, %.0f ms told "
              "its end (%.2fx)\n", kTotal, kChunk, whole.ms, skip.ms, whole.ms / skip.ms);
  std::printf("  logits bit for bit: %zu of %zu (the beside sequence's every step, the prompt's last, %u decoded "
              "each)\n", same, whole.logits.size(), kDecode);
  CHECK(same == whole.logits.size());
  forward_destroy(f);
  weights_destroy(w);
  gguf_destroy(eg);
  gguf_destroy(g);
  std::printf("[test_skip_gpu] ok\n");
  return 0;
}
