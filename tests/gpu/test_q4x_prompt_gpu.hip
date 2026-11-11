// a sequence told its prompt's end (q4x_seq_prompt) gets no logits for its tokens before end - 1: their rows left as
// they were and not kept (q4x_kept false, q4x_greedy refused), the output head not run for them. the row at end - 1
// and every later token's logits are the bits of a sequence never told: a prompt in one step, in two, two sequences
// sharing a step (one told, one not), the end lowered, and a load clearing it. needs a gfx1151 and
// ~/Models/qwen38/UD-IQ3_XXS; skipped (77) without them.
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kCtx = 512, kMaxTokens = 64, kPrompt = 40, kFirst = 25, kHalf = 20, kLower = 10;
constexpr int kMark = 0x5A;  // a row never written: every byte this

static Q4xForward* f;
static uint32_t V;
static char err[Q4X_FWD_ERR];

static std::vector<float> row(uint32_t r) {
  std::vector<float> v(V);
  CHECK(hipMemcpy(v.data(), q4x_logits(f) + (uint64_t) r * V, (uint64_t) V * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return v;
}

static bool same(const std::vector<float>& a, const std::vector<float>& b) {
  return std::memcmp(a.data(), b.data(), (uint64_t) V * 4) == 0;
}

static void mark() {
  CHECK(hipMemset((void*) q4x_logits(f), kMark, (uint64_t) kMaxTokens * V * 4) == hipSuccess);
}

static bool marked(uint32_t r) {
  const std::vector<float> v = row(r);
  for (uint64_t i = 0; i < (uint64_t) V * 4; i++)
    if (((const uint8_t*) v.data())[i] != kMark) return false;
  return true;
}

// rows r0 .. r1 - 1 not kept: untouched, q4x_kept false, a greedy pick of one refused
static void none(uint32_t r0, uint32_t r1) {
  for (uint32_t r = r0; r < r1; r++) {
    int32_t t;
    char e[Q4X_FWD_ERR] = "";
    CHECK(!q4x_kept(f, r) && marked(r) && !q4x_greedy(f, r, 1, &t, e) && std::strstr(e, "kept"));
  }
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_q4x_prompt_gpu] start\n");
  const std::string home = std::getenv("HOME") ? std::getenv("HOME") : "";
  const std::string model = home + "/Models/qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf";
  int ndev = 0;
  char gerr[GGUF_ERR] = "", werr[Q4X_ERR] = "";
  Gguf* g = access(model.c_str(), R_OK) == 0 ? gguf_create(model.c_str(), gerr) : nullptr;
  if (!g || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_q4x_prompt_gpu] skipped: no model or GPU\n");
    return 77;
  }
  V = (uint32_t) gguf_tensor(g, "token_embd.weight")->dims[1];
  std::vector<int32_t> p(kPrompt);
  uint32_t r = 1017;
  for (auto& x : p) r = r * 1664525u + 1013904223u, x = (int32_t) ((r >> 8) % (V - 1000)) + 100;
  Q4xWeights* w = q4x_weights_create(g, nullptr, werr);
  CHECK(w);
  f = q4x_forward_create(w, nullptr, kCtx, kMaxTokens, err);
  CHECK(f);
  CHECK(!q4x_kept(f, 0) && !q4x_kept(f, kMaxTokens));  // no step yet; a row past a step's

  // never told: every row kept. its rows the reference, its state after the prompt saved, then a token more
  Q4xSeq* a = q4x_seq_create(f, err);
  CHECK(a && q4x_batch(f, a, p.data(), kPrompt, 0, err));
  for (uint32_t i = 0; i < kPrompt; i++) CHECK(q4x_kept(f, i));
  const std::vector<float> ref_last = row(kPrompt - 1), ref_lower = row(kLower - 1), ref_half = row(kHalf - 1),
                           ref_mid = row(5);
  int32_t t1, t;
  CHECK(q4x_greedy(f, kPrompt - 1, 1, &t1, err));
  std::vector<uint8_t> saved;
  const auto sink = [](void* ctx, const void* b, uint64_t n) {
    auto* v = (std::vector<uint8_t>*) ctx;
    v->insert(v->end(), (const uint8_t*) b, (const uint8_t*) b + n);
    return true;
  };
  CHECK(q4x_seq_save_to(a, sink, &saved, err));
  CHECK(q4x_batch(f, a, &t1, 1, 0, err) && q4x_kept(f, 0));
  const std::vector<float> ref_next = row(0);
  std::printf("  never told: %u rows kept, the prompt's pick %d\n", kPrompt, t1);

  // told, the prompt one step: the last row alone, the reference's bits; the next token's too
  Q4xSeq* b = q4x_seq_create(f, err);
  CHECK(b && !q4x_seq_prompt(nullptr, 1) && !q4x_seq_prompt(b, kCtx + 1) && q4x_seq_prompt(b, kPrompt));
  mark();
  CHECK(q4x_batch(f, b, p.data(), kPrompt, 0, err));
  none(0, kPrompt - 1);
  CHECK(q4x_kept(f, kPrompt - 1) && same(row(kPrompt - 1), ref_last));
  CHECK(q4x_greedy(f, kPrompt - 1, 1, &t, err) && t == t1);
  CHECK(q4x_batch(f, b, &t1, 1, 0, err) && q4x_kept(f, 0) && same(row(0), ref_next));
  std::printf("  told, one step: rows 0 .. %u not kept, row %u and the next token the reference's bits\n",
              kPrompt - 2, kPrompt - 1);

  // told, the prompt two steps: the first keeps none, the second its last
  Q4xSeq* c = q4x_seq_create(f, err);
  CHECK(c && q4x_seq_prompt(c, kPrompt));
  mark();
  CHECK(q4x_batch(f, c, p.data(), kFirst, 0, err));
  none(0, kFirst);
  CHECK(q4x_batch(f, c, p.data() + kFirst, kPrompt - kFirst, 0, err));
  none(0, kPrompt - kFirst - 1);
  CHECK(q4x_kept(f, kPrompt - kFirst - 1) && same(row(kPrompt - kFirst - 1), ref_last));
  std::printf("  told, two steps: the first keeps none, the second its last row, the reference's bits\n");

  // two sequences a step: the told one's last row, the other's every row
  Q4xSeq* d = q4x_seq_create(f, err);
  Q4xSeq* e = q4x_seq_create(f, err);
  CHECK(d && e && q4x_seq_prompt(d, kHalf));
  std::vector<int32_t> two(p.begin(), p.begin() + kHalf);
  two.insert(two.end(), p.begin(), p.begin() + kHalf);
  Q4xSeq* const seqs[2] = {d, e};
  const uint32_t lens[2] = {kHalf, kHalf};
  mark();
  CHECK(q4x_step(f, seqs, lens, 2, two.data(), 0, err));
  none(0, kHalf - 1);
  CHECK(q4x_kept(f, kHalf - 1) && same(row(kHalf - 1), ref_half));
  for (uint32_t i = kHalf; i < 2 * kHalf; i++) CHECK(q4x_kept(f, i));
  CHECK(same(row(kHalf + 5), ref_mid) && same(row(2 * kHalf - 1), ref_half));
  std::printf("  two sequences a step: the told one's last row, the other's every row, the reference's bits\n");

  // the end lowered: the rows from the lower end - 1 on
  Q4xSeq* h = q4x_seq_create(f, err);
  CHECK(h && q4x_seq_prompt(h, kPrompt) && q4x_seq_prompt(h, kLower));
  mark();
  CHECK(q4x_batch(f, h, p.data(), kPrompt, 0, err));
  none(0, kLower - 1);
  for (uint32_t i = kLower - 1; i < kPrompt; i++) CHECK(q4x_kept(f, i));
  CHECK(same(row(kLower - 1), ref_lower) && same(row(kPrompt - 1), ref_last));
  std::printf("  the end lowered to %u: rows %u .. kept, the reference's bits\n", kLower, kLower - 1);

  // a load clears it: the loaded sequence's next token kept, the reference's bits
  struct Src {
    const std::vector<uint8_t>* v;
    uint64_t at;
  } src{&saved, 0};
  const auto source = [](void* ctx, void* b, uint64_t n) {
    auto* s = (Src*) ctx;
    if (s->at + n > s->v->size()) return false;
    std::memcpy(b, s->v->data() + s->at, n), s->at += n;
    return true;
  };
  CHECK(q4x_seq_prompt(h, kCtx) && q4x_seq_load_from(h, saved.size(), source, &src, err));
  mark();
  CHECK(q4x_batch(f, h, &t1, 1, 0, err) && q4x_kept(f, 0) && same(row(0), ref_next));
  std::printf("  a load clears the end: the next token kept, the reference's bits\n");

  for (Q4xSeq* s : {a, b, c, d, e, h}) q4x_seq_destroy(s);
  q4x_forward_destroy(f);
  q4x_weights_destroy(w);
  gguf_destroy(g);
  std::printf("[test_q4x_prompt_gpu] ok\n");
  return 0;
}
