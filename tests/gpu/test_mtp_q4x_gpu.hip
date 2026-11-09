// Qwen's MTP head as the engine's drafter (osmos/run/engine.h, q4x_mtp_*): a prompt of real text, then greedy decode
// two ways on one engine - plain (a token a step) and drafted (the server's rounds: drafts after y, [y, drafts] one
// step, the agreeing prefix kept, the rest rolled back, the kept positions seeded) - the same tokens, every one;
// the acceptance printed; then three sequences at their own positions drafting at once (oseq_draft_all), under the
// engine's floors (some stopping early) and none (the first at the context's end after 3, the others their whole
// block - their rows moved up past it): each one's drafts, confidences and count
// the bits of its own oseq_draft. needs a gfx1151, the UD-IQ3_XXS set and the shared MTP head (~81 GB of GPU memory), the
// dump's ids; skipped (77) without them.
#include "q4x_ref.h"

extern "C" {
#include "osmos/run/engine.h"
}

constexpr uint32_t kCtx = 1024, kStep = 256, kPrompt = 200, kGen = 96;
constexpr const char* kMtp = "%s/Models/qwen38/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";

static std::vector<int32_t> dump_ids() {
  FILE* f = std::fopen((home_path(kQwenDump) + "/ids.txt").c_str(), "r");
  std::vector<int32_t> v;
  for (int32_t x; f && std::fscanf(f, "%d,", &x) == 1;) v.push_back(x);
  if (f) std::fclose(f);
  return v;
}

static int32_t prompt(Osmos* e, OsmosSeq* s, const std::vector<int32_t>& ids, bool seed, uint32_t from = 0,
                      uint32_t n = kPrompt) {
  char err[OSMOS_ERR];
  std::vector<OsmosSeq*> ss(n, s);
  CHECK(osmos_batch(e, ss.data(), ids.data() + from, n, err));
  CHECK(!seed || oseq_seed(s, 0, (int32_t) n, err));
  int32_t y;
  CHECK(osmos_greedy(e, n - 1, 1, &y, err));
  return y;
}

// a sequence 3 positions short of the context (the dump's ids over and over, a step of kStep at a time, each seeded):
// its drafts cut at the context's end while the others' go on
static int32_t near_the_end(Osmos* e, OsmosSeq* s, const std::vector<int32_t>& ids) {
  char err[OSMOS_ERR];
  std::vector<int32_t> toks(kCtx - 3);
  for (size_t i = 0; i < toks.size(); i++) toks[i] = ids[i % kPrompt];
  int32_t y = -1;
  for (uint32_t at = 0; at < toks.size(); at += kStep) {
    const uint32_t n = toks.size() - at < kStep ? (uint32_t) (toks.size() - at) : kStep;
    std::vector<OsmosSeq*> ss(n, s);
    CHECK(osmos_batch(e, ss.data(), toks.data() + at, n, err) && oseq_seed(s, (int32_t) at, (int32_t) (at + n), err));
    CHECK(osmos_greedy(e, n - 1, 1, &y, err));
  }
  return y;
}

// three sequences drafting at once against each alone: the first 3 short of the context (or a prompt of 200), then
// prompts of 150 and 120 from their own offsets
static void drafted_together(Osmos* e, const std::vector<int32_t>& ids, float conf_min, float cum_min, bool end) {
  char err[OSMOS_ERR];
  OsmosSeq* s[3];
  int32_t y[3];
  const uint32_t from[3] = {0, 30, 56}, len[3] = {200, 150, 120};
  for (int i = 0; i < 3; i++)
    s[i] = oseq_create(e, err),
    y[i] = i == 0 && end ? near_the_end(e, s[i], ids) : prompt(e, s[i], ids, true, from[i], len[i]);
  int32_t all[3][OSMOS_MAX_BLOCK] = {}, one[OSMOS_MAX_BLOCK];
  float cf[3][OSMOS_MAX_BLOCK] = {}, c1[OSMOS_MAX_BLOCK];
  int cnt[3], drafted = 0;
  CHECK(oseq_draft_all(s, y, 3, conf_min, cum_min, all, cf, cnt, err));
  OsmosSeq* twice[] = {s[0], s[1], s[0]};
  int32_t scratch[3][OSMOS_MAX_BLOCK];
  float sc[3][OSMOS_MAX_BLOCK];
  int sn[3];
  CHECK(!oseq_draft_all(twice, y, 3, conf_min, cum_min, scratch, sc, sn, err));  // a sequence twice
  for (int i = 0; i < 3; i++) {
    const int k = oseq_draft(s[i], y[i], conf_min, cum_min, one, c1, err);
    CHECK(k == cnt[i] && std::memcmp(one, all[i], k * 4) == 0 && std::memcmp(c1, cf[i], k * 4) == 0);
    drafted += k;
  }
  std::printf("  three sequences drafting at once (floors %.2f / %.2f): %d, %d, %d drafts - each the bits of its "
              "own\n", conf_min, cum_min, cnt[0], cnt[1], cnt[2]);
  CHECK(drafted > 0);
  for (int i = 0; i < 3; i++) oseq_destroy(s[i]);
}

int main() {
  std::printf("[test_mtp_q4x_gpu] start\n");
  const std::string first = home_path(kQwenFirst), mtp = home_path(kMtp);
  const std::vector<int32_t> ids = dump_ids();
  int ndev = 0;
  if (access(first.c_str(), R_OK) || access(mtp.c_str(), R_OK) || ids.size() < kPrompt ||
      hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_mtp_q4x_gpu] skipped: no Qwen3.8 UD-IQ3_XXS, its MTP head, the dump's ids or a GPU\n");
    return 77;
  }
  char err[OSMOS_ERR] = "";
  const OsmosFiles files = {first.c_str(), nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  CHECK(m);
  const OsmosOptions bad = {kCtx, kStep, 0, nullptr, false, first.c_str(), false, 0};  // the model as its own head
  CHECK(!osmos_create(m, &bad, err) && std::strstr(err, "nextn"));
  const OsmosOptions o = {kCtx, kStep, 0, nullptr, false, mtp.c_str(), false, 0};
  Osmos* e = osmos_create(m, &o, err);
  if (!e) std::fprintf(stderr, "  %s\n", err);
  CHECK(e);
  const uint32_t block = osmos_block(e);
  CHECK(block > 0 && block <= OSMOS_MAX_BLOCK);

  // plain
  OsmosSeq* a = oseq_create(e, err);
  std::vector<int32_t> plain = {prompt(e, a, ids, false)};
  while (plain.size() < kGen) {
    int32_t y = plain.back();
    CHECK(osmos_batch(e, &a, &y, 1, err) && osmos_greedy(e, 0, 1, &y, err));
    plain.push_back(y);
  }
  oseq_destroy(a);

  // drafted
  OsmosSeq* s = oseq_create(e, err);
  std::vector<int32_t> out = {prompt(e, s, ids, true)};
  int rounds = 0, drafted = 0, accepted = 0;
  while (out.size() < kGen) {
    int32_t d[OSMOS_MAX_BLOCK], toks[OSMOS_MAX_BLOCK + 1], picks[OSMOS_MAX_BLOCK + 1];
    float conf[OSMOS_MAX_BLOCK];
    const int k = oseq_draft(s, out.back(), OSMOS_DRAFT_CONF_MIN, OSMOS_DRAFT_CUM_MIN, d, conf, err);
    CHECK(k >= 0 && k <= (int) block);
    const int32_t pos = oseq_position(s);
    toks[0] = out.back();
    for (int i = 0; i < k; i++) toks[i + 1] = d[i];
    std::vector<OsmosSeq*> ss(k + 1, s);
    CHECK(osmos_batch(e, ss.data(), toks, (uint32_t) k + 1, err) && osmos_greedy(e, 0, (uint32_t) k + 1, picks, err));
    int keep = 1;  // y, then each draft its row before it picked
    while (keep <= k && d[keep - 1] == picks[keep - 1]) keep++;
    for (int i = 0; i < keep - 1; i++) out.push_back(d[i]);
    out.push_back(picks[keep - 1]);
    CHECK(oseq_rollback(s, (uint32_t) (k + 1 - keep)) && oseq_position(s) == pos + keep);
    CHECK(oseq_seed(s, pos, pos + keep, err));
    rounds++, drafted += k, accepted += keep - 1;
  }
  out.resize(kGen);
  for (uint32_t i = 0; i < kGen; i++)
    if (out[i] != plain[i]) std::printf("  token %u: drafted %d, plain %d\n", i, out[i], plain[i]), std::exit(1);
  std::printf("  %u tokens drafted the bits of plain greedy: %d rounds, %d drafted, %d accepted (%.0f%%), %.2f "
              "tokens a round\n", kGen, rounds, drafted, accepted, drafted ? 100.0 * accepted / drafted : 0.0,
              (double) (kGen - 1) / rounds);
  CHECK(accepted > 0);
  oseq_destroy(s);
  drafted_together(e, ids, OSMOS_DRAFT_CONF_MIN, OSMOS_DRAFT_CUM_MIN, false);
  drafted_together(e, ids, 0.0f, 0.0f, true);  // the first's 3 drafts then the context's end, the others' 5
  osmos_destroy(e);
  osmos_close(m);
  std::printf("[test_mtp_q4x_gpu] ok\n");
  return 0;
}
