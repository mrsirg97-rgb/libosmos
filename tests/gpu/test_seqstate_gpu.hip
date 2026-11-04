// a sequence's state set aside and taken back (seq_save / seq_load): conversation A's prompt, its state saved, a
// conversation B run in a sequence of its own (its prompt and answer through the same forward), then A's state
// loaded into B's sequence - and A's answer from there the bits of A never set aside: every step's logits row byte
// for byte, greedy and (DSpark's weights present) drafted. the prompt's length leaves a compressor's group open (its
// pending rows in the state). refusals: a byte damaged, a size that disagrees, another identity - each false, the
// sequence reset to position 0. needs a gfx1151 and a model (test_model.h); skipped (77) without them.
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
#include "osmos/model/weights.h"
#include "osmos/run/drafter.h"
#include "osmos/run/forward.h"
}
#include "test_model.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kMaxTokens = 512, kMaxCtx = 4096;
constexpr uint32_t kPromptA = 301, kPromptB = 230;  // 301: 301 % 4 = 1, 301 % 128 = 45 positions not yet a row
constexpr int kNew = 24, kNewB = 8;
constexpr uint64_t kBudget = 100ull << 30;

static Forward* g_f;
static uint32_t g_vocab;
static char g_err[FORWARD_ERR];

static uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const uint8_t* b = (const uint8_t*) p;
  for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

static std::vector<int32_t> ids(uint32_t n, uint32_t seed) {
  std::vector<int32_t> v(n);
  uint32_t r = seed;
  for (auto& x : v) r = r * 1664525u + 1013904223u, x = (int32_t) ((r >> 8) % (g_vocab - 1000)) + 100;
  return v;
}

// the prompt in steps of max_tokens; the next token (greedy) and the last row's logits hashed into h
static int32_t prompt(Seq* s, const std::vector<int32_t>& p, uint64_t* h) {
  const uint32_t P = (uint32_t) p.size();
  CHECK(seq_prompt(s, P));
  std::vector<Seq*> ss(P, s);
  for (uint32_t at = 0; at < P;) {
    const uint32_t c = P - at < kMaxTokens ? P - at : kMaxTokens;
    CHECK(forward_batch(g_f, ss.data(), p.data() + at, c, g_err));
    at += c;
  }
  const uint32_t row = (P - 1) % kMaxTokens;
  int32_t t;
  CHECK(forward_greedy(g_f, row, 1, &t, g_err) && hipDeviceSynchronize() == hipSuccess);
  std::vector<float> lg(g_vocab);
  CHECK(hipMemcpy(lg.data(), forward_logits(g_f) + (uint64_t) row * g_vocab, g_vocab * 4, hipMemcpyDeviceToHost) ==
        hipSuccess);
  *h = fnv(lg.data(), lg.size() * 4, *h);
  return t;
}

// n greedy steps from token t: each step's logits row hashed into h
static void answer(Seq* s, int32_t t, int n, uint64_t* h) {
  std::vector<float> lg(g_vocab);
  for (int i = 0; i < n; i++) {
    CHECK(forward_step(g_f, s, t, g_err));
    CHECK(forward_greedy(g_f, 0, 1, &t, g_err) && hipDeviceSynchronize() == hipSuccess);
    CHECK(hipMemcpy(lg.data(), forward_logits(g_f), g_vocab * 4, hipMemcpyDeviceToHost) == hipSuccess);
    *h = fnv(lg.data(), lg.size() * 4, *h);
    *h = fnv(&t, 4, *h);
  }
}

// n tokens drafted and verified from token t (dbench's loop, greedy): every verify's logits rows and the accepted
// tokens hashed into h
static void drafted(Drafter* d, Seq* s, int32_t t, int n, uint64_t* h) {
  int32_t pos = seq_position(s), dr[DRAFTER_BLOCK];
  float cf[DRAFTER_BLOCK];
  std::vector<float> lg((size_t) (DRAFTER_BLOCK + 1) * g_vocab);
  for (int made = 0; made < n;) {
    const int k = drafter_draft(d, s, t, 0.0f, 0.0f, dr, cf, g_err);
    CHECK(k >= 0);
    std::vector<int32_t> toks{t};
    toks.insert(toks.end(), dr, dr + k);
    std::vector<Seq*> ss(toks.size(), s);
    CHECK(forward_batch(g_f, ss.data(), toks.data(), (uint32_t) toks.size(), g_err));
    int32_t picks[DRAFTER_BLOCK + 1];
    CHECK(forward_greedy(g_f, 0, (uint32_t) k + 1, picks, g_err) && hipDeviceSynchronize() == hipSuccess);
    CHECK(hipMemcpy(lg.data(), forward_logits(g_f), (size_t) (k + 1) * g_vocab * 4, hipMemcpyDeviceToHost) ==
          hipSuccess);
    *h = fnv(lg.data(), (size_t) (k + 1) * g_vocab * 4, *h);
    int m = 0;
    while (m < k && picks[m] == dr[m]) m++;
    CHECK(seq_rollback(s, (uint32_t) (k - m)) && drafter_seed(d, s, pos, pos + m + 1, g_err));
    *h = fnv(picks, (size_t) (m + 1) * 4, *h);
    t = picks[m], pos += m + 1, made += m + 1;
  }
}

int main() {
  std::printf("[test_seqstate_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_seqstate_gpu] skipped: no ds1017 or UD-IQ1_M, or no GPU\n");
    return 77;
  }
  const std::string dir = std::string(path).substr(0, std::string(path).rfind('/') + 1);
  char gerr[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "";
  Gguf* g = gguf_create(path, gerr);
  CHECK(g);
  Gguf* eg = which == TestModel::kGsq ? gguf_create((dir + "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf").c_str(), gerr)
                                      : nullptr;
  CHECK(which != TestModel::kGsq || eg);
  Weights* w = weights_create(g, eg, werr);
  CHECK(w);
  g_vocab = (uint32_t) w->geo.n_vocab;
  std::vector<uint64_t> picks((size_t) w->geo.n_layers * w->geo.n_experts);  // the bench's profile, where it is
  FILE* pf = std::fopen((dir + "osmos-picks.bin").c_str(), "rb");
  const bool ranked = pf && std::fread(picks.data(), 8, picks.size(), pf) == picks.size();
  if (pf) std::fclose(pf);
  const ForwardOptions o = {(int) w->geo.n_layers, kMaxCtx, kBudget, ranked ? picks.data() : nullptr, kMaxTokens};
  CHECK((g_f = forward_create_with(w, &o, g_err)) && forward_lane(g_f, 0, g_err));
  const char* home = std::getenv("HOME");
  const std::string ddir = std::string(home ? home : "") + "/Models/ds1017/dspark";
  Drafter* d = access((ddir + "/dspark.bin").c_str(), R_OK) == 0 ? drafter_create(g_f, ddir.c_str(), g_err) : nullptr;
  const std::vector<int32_t> pa = ids(kPromptA, 1017), pb = ids(kPromptB, 4242);

  for (int draft = 0; draft < (d ? 2 : 1); draft++) {
    // the reference: A from its prompt to its answer, never set aside
    uint64_t ref = 1469598103934665603ull;
    Seq* a = seq_create(g_f, g_err);
    CHECK(a);
    int32_t t = prompt(a, pa, &ref);
    if (draft) CHECK(drafter_seed(d, a, 0, (int32_t) kPromptA, g_err));
    draft ? drafted(d, a, t, kNew, &ref) : answer(a, t, kNew, &ref);
    seq_destroy(a);

    // A's prompt, its state saved; B run in its own sequence; A's state loaded into B's; A answers from there
    uint64_t got = 1469598103934665603ull;
    a = seq_create(g_f, g_err);
    CHECK(a);
    t = prompt(a, pa, &got);
    if (draft) CHECK(drafter_seed(d, a, 0, (int32_t) kPromptA, g_err));
    std::vector<uint8_t> st(seq_state_bytes(a));
    CHECK(!st.empty() && !seq_save(a, st.data(), st.size() - 1, g_err) && seq_save(a, st.data(), st.size(), g_err));
    seq_destroy(a);
    Seq* b = seq_create(g_f, g_err);
    CHECK(b);
    uint64_t hb = 0;
    const int32_t tb = prompt(b, pb, &hb);
    if (draft) CHECK(drafter_seed(d, b, 0, (int32_t) kPromptB, g_err));
    answer(b, tb, kNewB, &hb);
    CHECK(seq_load(b, st.data(), st.size(), g_err) && seq_position(b) == (int32_t) kPromptA);
    draft ? drafted(d, b, t, kNew, &got) : answer(b, t, kNew, &got);
    std::printf("  %s: A's state %zu bytes (%u tokens) set aside, B's %u + %d run, A taken back into B's sequence: "
                "every logits row the never-set-aside run's (%016llx)\n",
                draft ? "drafted" : "greedy", st.size(), kPromptA, kPromptB, kNewB, (unsigned long long) ref);
    CHECK(got == ref);

    if (!draft) {  // refusals, each leaving the sequence at position 0
      std::vector<uint8_t> bad = st;
      bad[bad.size() / 2] ^= 0x10;
      CHECK(!seq_load(b, bad.data(), bad.size(), g_err) && seq_position(b) == 0);
      std::printf("  a byte damaged: refused (%s), the sequence reset\n", g_err);
      CHECK(seq_load(b, st.data(), st.size(), g_err) && seq_position(b) == (int32_t) kPromptA);
      CHECK(!seq_load(b, st.data(), st.size() - 4, g_err) && seq_position(b) == 0);
      std::printf("  a size that disagrees: refused (%s), the sequence reset\n", g_err);
      // another identity: the header's (bytes 8 .. 16, after the magic and the version), the checksum made again
      bad = st;
      uint64_t id;
      std::memcpy(&id, bad.data() + 8, 8);
      id ^= 1;
      std::memcpy(bad.data() + 8, &id, 8);
      const uint64_t sum = fnv(bad.data(), bad.size() - 8);
      std::memcpy(bad.data() + bad.size() - 8, &sum, 8);
      CHECK(id != forward_identity(g_f));
      CHECK(seq_load(b, st.data(), st.size(), g_err));
      CHECK(!seq_load(b, bad.data(), bad.size(), g_err) && seq_position(b) == 0);
      std::printf("  another identity: refused (%s), the sequence reset\n", g_err);
    }
    seq_destroy(b);
  }
  if (!d) std::printf("  drafted: skipped (no DSpark weights at %s)\n", ddir.c_str());
  drafter_destroy(d);
  forward_destroy(g_f);
  weights_destroy(w);
  gguf_destroy(g);
  gguf_destroy(eg);
  std::printf("[test_seqstate_gpu] ok\n");
  return 0;
}
