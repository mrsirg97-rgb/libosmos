// the q35 sequence's contracts (osmos/q35/forward.h, DESIGN.md "0.9.0: q35 behind the engine") on the real model -
// every check a logits row's bits against another's:
//   rollback: a prompt, k decoded tokens, a rollback of k, the same k again - each row the first time's
//   runs: two sequences in one step against each alone
//   state: a save, a load into a new sequence, the next tokens' rows the original's; a damaged string refused (the
//     sequence then empty), a rollback past the reach refused (the sequence unchanged)
// the model Q35_MODEL, the tokens Q35_REF's ids.i32 (a held session's); skipped (77) without them
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "osmos/q35/forward.h"
}

#define CHECK(c)                                                                  \
  do {                                                                            \
    if (!(c)) {                                                                   \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)
#define OK(c) CHECK((c) || (std::fprintf(stderr, "%s\n", err), false))

static char err[Q35_ERR];
static uint32_t V;

static std::vector<float> row(Q35Forward* f, uint32_t r) {
  const float* d = q35_row_logits(f, r);
  CHECK(d);
  std::vector<float> h(V);
  CHECK(hipMemcpy(h.data(), d, (size_t) V * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return h;
}
static bool same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}
// n tokens of s one step; its last row's logits
static std::vector<float> feed(Q35Forward* f, Q35Seq* s, const int32_t* t, uint32_t n) {
  OK(q35_step_runs(f, &s, &n, 1, t, err));
  return row(f, n - 1);
}

struct Buf {
  std::vector<uint8_t> b;
  size_t at = 0;
};
static bool sink(void* c, const void* p, uint64_t n) {
  Buf* x = (Buf*) c;
  x->b.insert(x->b.end(), (const uint8_t*) p, (const uint8_t*) p + n);
  return true;
}
static bool source(void* c, void* p, uint64_t n) {
  Buf* x = (Buf*) c;
  if (x->at + n > x->b.size()) return false;
  std::memcpy(p, x->b.data() + x->at, n), x->at += n;
  return true;
}

// a tap that makes layer 0's residual NaN (a norm keeps it NaN: the next layer's f16 inputs past range, flagged)
static void poison(void* arg, int layer, const float* h, uint32_t n, uint32_t dim) {
  if (!*(bool*) arg || layer != 0) return;
  const std::vector<float> big((size_t) n * dim, std::nanf(""));
  CHECK(hipMemcpy((void*) h, big.data(), big.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
}

int main() {
  std::printf("[test_q35_seq_gpu] start\n");
  const char* mp = std::getenv("Q35_MODEL");
  const char* rp = std::getenv("Q35_REF");
  int devs = 0;
  if (!mp || !rp || hipGetDeviceCount(&devs) != hipSuccess || !devs) {
    std::printf("[test_q35_seq_gpu] skipped: no Q35_MODEL / Q35_REF, or no GPU\n");
    return 77;
  }
  std::vector<int32_t> ids;
  {
    FILE* fp = std::fopen((std::string(rp) + "/ids.i32").c_str(), "rb");
    CHECK(fp);
    int32_t v;
    while (std::fread(&v, 4, 1, fp) == 1) ids.push_back(v);
    std::fclose(fp);
  }
  constexpr uint32_t P = 40, K = 6;  // a prompt, then K decoded tokens (K <= Q35_MAX_BACK)
  CHECK(ids.size() >= P + K + 4);
  Gguf* g = gguf_create(mp, err);
  OK(g);
  Q35Weights* w = q35_weights_create(g, err);
  OK(w);
  V = (uint32_t) w->geo.n_vocab;
  const Q35Options o = {64, 1024, 0};
  Q35Forward* f = q35_forward_create_with(w, &o, err);
  OK(f);
  const int32_t* t = ids.data();

  // rollback: the decoded rows again after a drop of K
  Q35Seq* a = q35_seq_create(f, err);
  OK(a);
  feed(f, a, t, P);
  std::vector<std::vector<float>> first;
  for (uint32_t i = 0; i < K; i++) first.push_back(feed(f, a, t + P + i, 1));
  CHECK(q35_seq_reach(a) == Q35_MAX_BACK);  // P + K tokens fed: the ring's whole reach
  CHECK(!q35_seq_rollback(a, Q35_MAX_BACK + 1) && q35_seq_position(a) == (int32_t) (P + K));  // past it: unchanged
  CHECK(q35_seq_rollback(a, K) && q35_seq_position(a) == (int32_t) P);
  for (uint32_t i = 0; i < K; i++) CHECK(same(feed(f, a, t + P + i, 1), first[i]));
  // and a drop inside a step's tokens: K fed as one step, 2 dropped, the last 2 again one at a time
  CHECK(q35_seq_rollback(a, K));
  feed(f, a, t + P, K);
  CHECK(q35_seq_rollback(a, 2));
  CHECK(same(feed(f, a, t + P + K - 2, 1), first[K - 2]) && same(feed(f, a, t + P + K - 1, 1), first[K - 1]));
  std::printf("  rollback: %u decoded rows the same bits after a drop of %u (and of 2 inside a step)\n", K, K);

  // runs: b and c in one step against each alone (c the same prompt shifted by one: other positions, other rows)
  Q35Seq *b = q35_seq_create(f, err), *c = q35_seq_create(f, err);
  OK(b && c);
  feed(f, b, t, P), feed(f, c, t + 1, P);
  const std::vector<float> b_alone = feed(f, b, t + P, 1), c_alone = feed(f, c, t + 1 + P, 1);
  CHECK(q35_seq_rollback(b, 1) && q35_seq_rollback(c, 1));
  Q35Seq* both[2] = {b, c};
  const uint32_t lens[2] = {1, 1};
  const int32_t two[2] = {t[P], t[1 + P]};
  OK(q35_step_runs(f, both, lens, 2, two, err));
  CHECK(same(row(f, 0), b_alone) && same(row(f, 1), c_alone));
  std::printf("  runs: two sequences in one step, each one's bits alone\n");

  // state: b saved, loaded into d; both fed the same next tokens
  Buf saved;
  OK(q35_seq_save_to(b, sink, &saved, err));
  CHECK(saved.b.size() == q35_seq_state_bytes(b));
  Q35Seq* d = q35_seq_create(f, err);
  OK(d);
  OK(q35_seq_load_from(d, saved.b.size(), source, &saved, err));
  CHECK(q35_seq_position(d) == q35_seq_position(b) && q35_seq_reach(d) == 0);
  for (uint32_t i = 0; i < 3; i++) {
    const std::vector<float> x = feed(f, b, t + P + 1 + i, 1);
    CHECK(same(feed(f, d, t + P + 1 + i, 1), x));
  }
  Buf bad = saved;
  bad.at = 0, bad.b[bad.b.size() / 2] ^= 0x01;
  CHECK(!q35_seq_load_from(d, bad.b.size(), source, &bad, err) && q35_seq_position(d) == 0);
  std::printf("  state: %.1f MB saved and loaded, 3 tokens the same bits; a damaged one refused (%s)\n",
              saved.b.size() / 1e6, err);

  // a mark: taken at b's position, 2 more tokens, the mark loaded, the 2 again
  std::vector<uint8_t> mark(q35_seq_mark_bytes(b));
  OK(q35_seq_mark_save(b, mark.data(), mark.size(), err));
  const int32_t at = q35_seq_position(b);
  const std::vector<float> m0 = feed(f, b, t + at, 1), m1 = feed(f, b, t + at + 1, 1);
  OK(q35_seq_mark_load(b, mark.data(), mark.size(), err));
  CHECK(q35_seq_position(b) == at && same(feed(f, b, t + at, 1), m0) && same(feed(f, b, t + at + 1, 1), m1));
  std::printf("  mark: %.1f MB, the 2 tokens after it the same bits\n", mark.size() / 1e6);

  // an overflowing read refused, the forward's next step a new sequence's as ever (0.11.2: the f16 flag stayed set -
  // every step after refused, every sequence's, until a restart)
  {
    bool on = true;
    q35_forward_tap(f, poison, &on);
    Q35Seq *e = q35_seq_create(f, err), *h = q35_seq_create(f, err);
    OK(e && h);
    CHECK(!q35_step(f, e, t, 16, 1, err) && std::strstr(err, "f16"));
    on = false;
    q35_forward_tap(f, nullptr, nullptr);
    OK(q35_step(f, h, t, 16, 1, err));
    std::printf("  an overflowing read refused, the next step another sequence's as ever\n");
    q35_seq_destroy(e), q35_seq_destroy(h);
  }

  for (Q35Seq* s : {a, b, c, d}) q35_seq_destroy(s);
  q35_forward_destroy(f);
  q35_weights_destroy(w);
  gguf_destroy(g);
  std::printf("[test_q35_seq_gpu] ok\n");
  return 0;
}
