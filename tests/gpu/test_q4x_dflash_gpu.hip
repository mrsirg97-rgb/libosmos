// DFlash2's drafter on Qwen (osmos/q4x/forward.h, q4x_dflash_feed / q4x_dflash_draft) against its reference
// (~/osmos-oracle/dflash2-chat: z-lab/dflash's model in torch, F32, the converted Q8_0 drafter's weights dequantized,
// fed osmos's own taps of a chat's greedy path): the prompt one step and fed, then each of the reference's rounds -
// a draft at the anchor, the round's accepted tokens a step, fed - its drafts the reference's, its final hidden
// (rounds 0 and 10) within mm's f16; then the same on a second sequence with two tokens past each round rolled back
// - its prompt shared with another sequence's step, fed in two parts: every draft and hidden the first's bits; a
// loaded sequence's drafts from its position on, whatever its rings held. refusals: a forward whose taps are not the
// drafter's, a draft before its context is fed, a feed past the last step or the position. needs a gfx1151, the
// UD-IQ3_XXS set, ~/Models/qwen38/DFlash2/dflash2-Q8_0.gguf and the reference; skipped (77) without them.
#include <map>

#include "q4x_ref.h"

extern "C" {
#include "osmos/q4x/dflash.h"
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"
}

constexpr const char* kDraft = "%s/Models/qwen38/DFlash2/dflash2-Q8_0.gguf";
constexpr const char* kRef = "%s/osmos-oracle/dflash2-chat";
constexpr uint32_t kCtx = 1024, kStep = 64, kJunk = 2;
constexpr int32_t kJunkToken = 13;  // the tokens fed past a round's and rolled back
// the block's hidden through 5 layers of mm's f16 matmuls and the head's norm: measured 1.0e-3 (rounds 0 and 10),
// within one kF16Rel; twice it the bound. the drafts every one the reference's (measured 511 of 511): the anchor's
// pairwise term moves 2 of the 73 rounds' first drafts (the sim's), so a bound short of all of them misses a wiring
// fault - a flip from a near tie shown one before the bound moves
constexpr double kHiddenRel = 2 * kF16Rel;

struct Round {
  int32_t start, produced;
  std::vector<int32_t> drafts;
};

static std::map<int, std::vector<float>> g_hid;  // a pass's df_out by round
static int g_round = -1;
static void probe(const char* step, int, float* dev, uint64_t n, void*) {
  if (std::strcmp(step, "df_out") || g_round < 0) return;
  std::vector<float> h(n);
  CHECK(hipMemcpy(h.data(), dev, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  g_hid[g_round] = h;
}

static double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double e = 0, r = 0;
  for (size_t i = 0; i < a.size(); i++) e += ((double) a[i] - b[i]) * ((double) a[i] - b[i]), r += (double) b[i] * b[i];
  return std::sqrt(e / r);
}

static void step(Q4xForward* f, Q4xSeq* s, const int32_t* tok, uint32_t n) {
  char err[Q4X_FWD_ERR] = "";
  if (!q4x_batch(f, s, tok, n, 0, err)) std::fprintf(stderr, "%s\n", err), std::exit(1);
}

static std::vector<uint8_t> g_state;  // a sequence saved at the plain pass's end
static bool sink(void* ctx, const void* b, uint64_t n) {
  auto* v = (std::vector<uint8_t>*) ctx;
  v->insert(v->end(), (const uint8_t*) b, (const uint8_t*) b + n);
  return true;
}
static bool source(void* ctx, void* b, uint64_t n) {
  auto* at = (size_t*) ctx;
  std::memcpy(b, g_state.data() + *at, n), *at += n;
  return true;
}

// a pass over the rounds. plain: the prompt a step, fed whole; the sequence saved at the end (g_state). junk: the
// prompt a step shared with another sequence before it (its taps' rows past the other's), fed in two parts; each
// round's step two tokens longer, those two rolled back - fed then dropped, or dropped first (a feed of them refused)
static std::vector<std::vector<int32_t>> pass(Q4xForward* f, const std::vector<int32_t>& ids, int32_t P,
                                              const std::vector<Round>& rounds, bool junk) {
  char err[Q4X_FWD_ERR] = "";
  Q4xSeq* s = q4x_seq_create(f, err);
  CHECK(s);
  int32_t d[8];
  g_round = -1;
  CHECK(q4x_dflash_draft(f, s, ids[0], 7, d, err) == 7);  // position 0: no context to feed, a draft from none
  if (junk) {
    Q4xSeq* other = q4x_seq_create(f, err);
    Q4xSeq* const both[2] = {other, s};
    const uint32_t lens[2] = {kJunk + 3, (uint32_t) P};
    std::vector<int32_t> tok(lens[0], kJunkToken);
    tok.insert(tok.end(), ids.begin(), ids.begin() + P);
    CHECK(other && q4x_step(f, both, lens, 2, tok.data(), 0, err));
    CHECK(q4x_dflash_feed(f, s, (uint32_t) P / 2, err) && q4x_dflash_feed(f, s, (uint32_t) (P - P / 2), err));
    q4x_seq_destroy(other);
  } else {
    step(f, s, ids.data(), (uint32_t) P);
    CHECK(q4x_dflash_draft(f, s, ids[P], 7, d, err) == -1 && std::strstr(err, "not fed"));  // the step's not yet fed
    CHECK(!q4x_dflash_feed(f, s, (uint32_t) P + 1, err));                                    // past the step
    CHECK(q4x_dflash_feed(f, s, (uint32_t) P, err));
  }
  CHECK(!q4x_dflash_feed(f, s, 1, err));  // past the position
  std::vector<std::vector<int32_t>> out;
  for (size_t r = 0; r < rounds.size(); r++) {
    const Round& R = rounds[r];
    CHECK(q4x_seq_position(s) == R.start);
    g_round = (int) r;
    const int n = q4x_dflash_draft(f, s, ids[R.start], 7, d, err);
    if (n != 7) std::fprintf(stderr, "round %zu: %d (%s)\n", r, n, err), std::exit(1);
    out.emplace_back(d, d + n);
    std::vector<int32_t> tok(ids.begin() + R.start, ids.begin() + R.start + R.produced);
    if (junk) tok.insert(tok.end(), kJunk, kJunkToken);
    step(f, s, tok.data(), (uint32_t) tok.size());
    if (junk && r % 2) {  // rolled back first: the dropped tokens past the position, refused; the kept fed
      CHECK(q4x_seq_rollback(s, kJunk) && !q4x_dflash_feed(f, s, (uint32_t) tok.size(), err));
      CHECK(q4x_dflash_feed(f, s, (uint32_t) R.produced, err));
      continue;
    }
    CHECK(q4x_dflash_feed(f, s, (uint32_t) tok.size(), err));
    if (junk) CHECK(q4x_seq_rollback(s, kJunk));
  }
  if (!junk) {
    g_state.clear();
    CHECK(q4x_seq_save_to(s, sink, &g_state, err));
  }
  q4x_seq_destroy(s);
  return out;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_q4x_dflash_gpu] start\n");
  int ndev = 0;
  char gerr[GGUF_ERR] = "", werr[Q4X_ERR] = "", err[Q4X_FWD_ERR] = "";
  const std::string ref = home_path(kRef);
  FILE* fi = std::fopen((ref + "/ids.txt").c_str(), "r");
  FILE* fp = std::fopen((ref + "/prompt.txt").c_str(), "r");
  FILE* fr = std::fopen((ref + "/rounds.txt").c_str(), "r");
  Gguf* model = (hipGetDeviceCount(&ndev) == hipSuccess && ndev && fi && fp && fr)
                    ? gguf_create(home_path(kQwenFirst).c_str(), gerr) : nullptr;
  Gguf* draft = model ? gguf_create(home_path(kDraft).c_str(), gerr) : nullptr;
  if (!draft) {
    std::printf("[test_q4x_dflash_gpu] skipped: no GPU, Qwen, the drafter or its reference (%s)\n", gerr);
    return 77;
  }
  std::vector<int32_t> ids;
  for (int32_t x; std::fscanf(fi, "%d,", &x) == 1;) ids.push_back(x);
  int32_t P = 0;
  CHECK(std::fscanf(fp, "%d", &P) == 1 && P > 0);
  std::vector<Round> rounds;
  for (Round r; std::fscanf(fr, "%d %d", &r.start, &r.produced) == 2;) {
    r.drafts.resize(7);
    for (int32_t& x : r.drafts) CHECK(std::fscanf(fr, "%d", &x) == 1);
    rounds.push_back(r);
  }
  std::fclose(fi), std::fclose(fp), std::fclose(fr);
  CHECK(rounds.size() > 10 && rounds[0].start == P);

  Q4xWeights* w = q4x_weights_create(model, nullptr, werr);
  CHECK(w);
  Q4xDflash* df = q4x_dflash_create(draft, w, werr);
  CHECK(df);
  const int32_t wrong[5] = {5, 15, 25, 35, 44};
  Q4xOptions o = {kCtx, kStep, false, wrong, 5, df};
  CHECK(!q4x_forward_create_with(w, nullptr, &o, err) && std::strstr(err, "tap "));  // taps not the drafter's
  o.tap_layers = nullptr, o.n_taps = 0;
  Q4xForward* f = q4x_forward_create_with(w, nullptr, &o, err);
  if (!f) std::fprintf(stderr, "%s\n", err);
  CHECK(f);
  q4x_forward_probe(f, probe, nullptr);

  const auto plain = pass(f, ids, P, rounds, false);
  const auto plain_hid = g_hid;
  uint32_t agree = 0, total = 0, made = 0;
  for (size_t r = 0; r < rounds.size(); r++) {
    for (int j = 0; j < 7; j++) agree += plain[r][j] == rounds[r].drafts[j], total++;
    int a = 0;  // its own drafts' acceptance on the true path
    while (a < 7 && rounds[r].start + 1 + a < (int32_t) ids.size() && plain[r][a] == ids[rounds[r].start + 1 + a]) a++;
    made += a + 1;
  }
  std::printf("  %zu rounds: drafts %u of %u the reference's (%.1f%%); its own %.2f tokens a round (the reference's"
              " %.2f)\n", rounds.size(), agree, total, 100.0 * agree / total, (double) made / rounds.size(),
              (double) (rounds.back().start + rounds.back().produced - P) / rounds.size());
  CHECK(agree == total);
  for (int r : {0, 10}) {
    Ref h;
    CHECK(ref_read("hid_r" + std::to_string(r), &h, kRef) && h.v.size() == plain_hid.at(r).size());
    const double e = rel_l2(plain_hid.at(r), h.v);
    std::printf("  round %d: the block's final hidden rel L2 %.2e (bound %.2e)\n", r, e, kHiddenRel);
    CHECK(e < kHiddenRel);
  }

  g_hid.clear();
  const auto junk = pass(f, ids, P, rounds, true);
  CHECK(junk == plain);
  for (auto& [r, h] : plain_hid) CHECK(std::memcmp(g_hid.at(r).data(), h.data(), h.size() * 4) == 0);
  std::printf("  the prompt shared with another sequence and fed in two parts, two tokens past each round rolled back"
              " (fed first, or refused): every draft and hidden the plain pass's bits\n");

  // a loaded sequence: its drafter context from its position on (the rings not saved) - none read from before it. one
  // loaded into a sequence whose rings hold the prompt's context, one into a new one: the same drafts
  int32_t a[8], b[8];
  const int32_t y = ids[rounds.back().start + rounds.back().produced];
  Q4xSeq *used = q4x_seq_create(f, err), *fresh = q4x_seq_create(f, err);
  CHECK(used && fresh);
  step(f, used, ids.data(), (uint32_t) P);
  CHECK(q4x_dflash_feed(f, used, (uint32_t) P, err));
  size_t at = 0;
  CHECK(q4x_seq_load_from(used, g_state.size(), source, &at, err));
  at = 0;
  CHECK(q4x_seq_load_from(fresh, g_state.size(), source, &at, err));
  CHECK(q4x_dflash_draft(f, used, y, 7, a, err) == 7 && q4x_dflash_draft(f, fresh, y, 7, b, err) == 7);
  CHECK(std::memcmp(a, b, sizeof a - sizeof *a) == 0);
  q4x_seq_destroy(used), q4x_seq_destroy(fresh);
  std::printf("  a loaded sequence's drafts from no context: the same whatever its rings held\n");

  q4x_forward_destroy(f);
  q4x_dflash_destroy(df);
  q4x_weights_destroy(w);
  gguf_destroy(draft), gguf_destroy(model);
  std::printf("[test_q4x_dflash_gpu] ok\n");
  return 0;
}
