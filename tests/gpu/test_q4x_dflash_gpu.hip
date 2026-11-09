// DFlash2's drafter on Qwen (osmos/q4x/forward.h, q4x_dflash_seed / q4x_dflash_draft) against its reference
// (~/osmos-oracle/dflash2-chat: z-lab/dflash's model in torch, F32, the converted Q8_0 drafter's weights dequantized,
// fed osmos's own taps of a chat's greedy path): the prompt one step and seeded, then each of the reference's rounds
// - a draft at the anchor, the round's accepted tokens a step, seeded - its drafts the reference's, its final hidden
// (rounds 0 and 10) within mm's f16; then the same on a second sequence with two tokens past each round rolled back
// - its prompt shared with another sequence's step, seeded in two parts: every draft and hidden the first's bits; a
// loaded sequence's drafts, and one seeded past a gap, from no context before it, whatever its rings held; the
// floors' count each draft's confidence's (the head's probability of it) prefix. refusals: a forward whose taps are
// not the drafter's, a draft before its context is seeded, a seed past the last step or the position. needs a
// gfx1151, the UD-IQ3_XXS set, ~/Models/qwen38/DFlash2/dflash2-Q8_0.gguf and the reference; skipped (77) without them.
#include <map>

#include "q4x_ref.h"

extern "C" {
#include "osmos/q4x/dflash.h"
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"
#include "osmos/run/engine.h"
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
// a draft's confidence: the head's logits of that hidden, a probability from two of them - measured 2.4e-3 from the
// reference's; bound twice the hidden's (a difference of two logits, each its error)
constexpr double kConfErr = 2 * kHiddenRel;
constexpr float kConfMin = OSMOS_DRAFT_CONF_MIN, kCumMin = OSMOS_DRAFT_CUM_MIN;  // the engine's floors

struct Round {
  int32_t start, produced;
  std::vector<int32_t> drafts;
  std::vector<float> conf;  // the reference's: the head's softmax probability of each draft at its place
};

static std::map<int, std::vector<float>> g_hid;  // a pass's df_out by round
static uint64_t g_floored, g_drafts;            // the drafts the engine's floors keep, of all
static double g_conf_err;                        // the confidences' greatest distance from the reference's
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
// round's step two tokens longer, those two rolled back - seeded then dropped, or dropped first (a seed of them
// refused)
static std::vector<std::vector<int32_t>> pass(Q4xForward* f, const std::vector<int32_t>& ids, int32_t P,
                                              const std::vector<Round>& rounds, bool junk) {
  char err[Q4X_FWD_ERR] = "";
  Q4xSeq* s = q4x_seq_create(f, err);
  CHECK(s);
  int32_t d[8];
  float cf[8];
  g_round = -1;
  CHECK(q4x_dflash_draft(f, s, ids[0], 7, 0, 0, d, cf, err) == 7);  // position 0: no context to seed, a draft from none
  if (junk) {
    Q4xSeq* other = q4x_seq_create(f, err);
    Q4xSeq* const both[2] = {other, s};
    const uint32_t lens[2] = {kJunk + 3, (uint32_t) P};
    std::vector<int32_t> tok(lens[0], kJunkToken);
    tok.insert(tok.end(), ids.begin(), ids.begin() + P);
    CHECK(other && q4x_step(f, both, lens, 2, tok.data(), 0, err));
    CHECK(q4x_dflash_seed(f, s, 0, P / 2, err) && q4x_dflash_seed(f, s, P / 2, P, err));
    q4x_seq_destroy(other);
  } else {
    step(f, s, ids.data(), (uint32_t) P);
    CHECK(q4x_dflash_draft(f, s, ids[P], 7, 0, 0, d, cf, err) == -1 && std::strstr(err, "not seeded"));  // not yet
    CHECK(!q4x_dflash_seed(f, s, 0, P + 1, err));                                                       // past the step
    CHECK(q4x_dflash_seed(f, s, 0, P, err));
  }
  CHECK(!q4x_dflash_seed(f, s, P, P + 1, err));  // past the step and the position
  std::vector<std::vector<int32_t>> out;
  for (size_t r = 0; r < rounds.size(); r++) {
    const Round& R = rounds[r];
    CHECK(q4x_seq_position(s) == R.start);
    g_round = (int) r;
    const int n = q4x_dflash_draft(f, s, ids[R.start], 7, 0, 0, d, cf, err);
    if (n != 7) std::fprintf(stderr, "round %zu: %d (%s)\n", r, n, err), std::exit(1);
    out.emplace_back(d, d + n);
    for (int j = 0; j < 7; j++) g_conf_err = std::fmax(g_conf_err, std::fabs((double) cf[j] - R.conf[j]));
    // the engine's floors: the drafts while each confidence and their product are at least them
    int want = 0;
    float cum = 1;
    for (; want < 7; want++) {
      CHECK(cf[want] > 0 && cf[want] <= 1);
      cum *= cf[want];
      if (cf[want] < kConfMin || cum < kCumMin) break;
    }
    int32_t d2[8];
    float cf2[8];
    CHECK(q4x_dflash_draft(f, s, ids[R.start], 7, kConfMin, kCumMin, d2, cf2, err) == want);
    CHECK(!want || (std::memcmp(d2, d, want * 4) == 0 && std::memcmp(cf2, cf, want * 4) == 0));
    g_floored += want, g_drafts += 7;
    std::vector<int32_t> tok(ids.begin() + R.start, ids.begin() + R.start + R.produced);
    if (junk) tok.insert(tok.end(), kJunk, kJunkToken);
    step(f, s, tok.data(), (uint32_t) tok.size());
    if (junk && r % 2) {  // rolled back first: the dropped tokens past the position, refused; the kept fed
      CHECK(q4x_seq_rollback(s, kJunk) && !q4x_dflash_seed(f, s, R.start, R.start + (int32_t) tok.size(), err));
      CHECK(q4x_dflash_seed(f, s, R.start, R.start + R.produced, err));
      continue;
    }
    CHECK(q4x_dflash_seed(f, s, R.start, R.start + (int32_t) tok.size(), err));
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
  FILE* fc = std::fopen((ref + "/conf.txt").c_str(), "r");
  Gguf* model = (hipGetDeviceCount(&ndev) == hipSuccess && ndev && fi && fp && fr && fc)
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
    r.conf.resize(7);
    for (float& x : r.conf) CHECK(std::fscanf(fc, "%f", &x) == 1);
    rounds.push_back(r);
  }
  std::fclose(fi), std::fclose(fp), std::fclose(fr), std::fclose(fc);
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
  std::printf("  under the engine's floors (%.1f, %.1f): %llu of %llu drafts kept; confidences within %.2e of the"
              " reference's\n", kConfMin, kCumMin, (unsigned long long) g_floored, (unsigned long long) g_drafts,
              g_conf_err);
  CHECK(g_conf_err < kConfErr);
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
  std::printf("  the prompt shared with another sequence and seeded in two parts, two tokens past each round rolled"
              " back (seeded first, or refused): every draft and hidden the plain pass's bits\n");

  // a loaded sequence: its drafter context from its position on (the rings not saved) - none read from before it. one
  // loaded into a sequence whose rings hold the prompt's context, one into a new one: the same drafts
  int32_t a[8], b[8];
  const int32_t y = ids[rounds.back().start + rounds.back().produced];
  Q4xSeq *used = q4x_seq_create(f, err), *fresh = q4x_seq_create(f, err);
  CHECK(used && fresh);
  step(f, used, ids.data(), (uint32_t) P);
  CHECK(q4x_dflash_seed(f, used, 0, P, err));
  size_t at = 0;
  CHECK(q4x_seq_load_from(used, g_state.size(), source, &at, err));
  at = 0;
  CHECK(q4x_seq_load_from(fresh, g_state.size(), source, &at, err));
  float ca[8], cb[8];
  CHECK(q4x_dflash_draft(f, used, y, 7, 0, 0, a, ca, err) == 7);
  CHECK(q4x_dflash_draft(f, fresh, y, 7, 0, 0, b, cb, err) == 7);
  CHECK(std::memcmp(a, b, sizeof a - sizeof *a) == 0);
  // a seed past a gap: the context from it on. the used sequence emptied by a load of an empty state (its rings
  // still the prompt's), the prompt again and its second half seeded - against a new sequence the same way
  std::vector<uint8_t> empty;
  Q4xSeq* none = q4x_seq_create(f, err);
  CHECK(none && q4x_seq_save_to(none, sink, &empty, err));
  g_state.swap(empty), at = 0;
  CHECK(q4x_seq_load_from(used, g_state.size(), source, &at, err) && q4x_seq_position(used) == 0);
  for (Q4xSeq* q : {used, none}) {
    step(f, q, ids.data(), (uint32_t) P);
    CHECK(q4x_dflash_seed(f, q, P / 2, P, err));
  }
  CHECK(q4x_dflash_draft(f, used, ids[P], 7, 0, 0, a, ca, err) == 7);
  CHECK(q4x_dflash_draft(f, none, ids[P], 7, 0, 0, b, cb, err) == 7);
  CHECK(std::memcmp(a, b, sizeof a - sizeof *a) == 0);
  q4x_seq_destroy(used), q4x_seq_destroy(fresh), q4x_seq_destroy(none);
  std::printf("  a loaded sequence's drafts, and a seed past a gap's: from no context before it, whatever the rings"
              " held\n");

  q4x_forward_destroy(f);
  q4x_dflash_destroy(df);
  q4x_weights_destroy(w);
  gguf_destroy(draft), gguf_destroy(model);
  std::printf("[test_q4x_dflash_gpu] ok\n");
  return 0;
}
