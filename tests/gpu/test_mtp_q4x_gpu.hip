// Qwen's drafters as the engine's - the MTP head (q4x_mtp_*), then DFlash2 (q4x_dflash_*), each where its file is
// (osmos/run/engine.h): a prompt of real text, then greedy decode
// two ways on one engine - plain (a token a step) and drafted (the server's rounds: drafts after y, [y, drafts] one
// step, the agreeing prefix kept, the rest rolled back, the kept positions seeded) - the same tokens, every one;
// the acceptance printed; then three sequences at their own positions drafting at once (oseq_draft_all), under the
// engine's floors (some stopping early) and none (the first at the context's end after 3, the others their whole
// block - their rows moved up past it): each one's drafts, confidences and count
// the bits of its own oseq_draft. needs a gfx1151, the UD-IQ3_XXS set and a drafter (the shared MTP head, or
// tools/convert/dflash2_gguf.py's DFlash2: ~81 GB of GPU memory), the dump's ids; skipped (77) without them.
#include "q4x_ref.h"

extern "C" {
#include "osmos/run/engine.h"
}

constexpr uint32_t kCtx = 1024, kStep = 256, kPrompt = 200, kGen = 96;
constexpr const char* kMtp = "%s/Models/qwen38/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";
constexpr const char* kDflash = "%s/Models/qwen38/DFlash2/dflash2-Q8_0.gguf";

static std::vector<int32_t> dump_ids() {
  FILE* f = std::fopen((home_path(kQwenDump) + "/ids.txt").c_str(), "r");
  std::vector<int32_t> v;
  for (int32_t x; f && std::fscanf(f, "%d,", &x) == 1;) v.push_back(x);
  if (f) std::fclose(f);
  return v;
}

static int32_t prompt(Osmos* e, OsmosSeq* s, const std::vector<int32_t>& ids, bool seed, uint32_t from = 0,
                      uint32_t n = kPrompt) {
  char err[OSMOS_ERR] = "";
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
  char err[OSMOS_ERR] = "";
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
  char err[OSMOS_ERR] = "";
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

// 0.9.2: the MTP head's drafts after a state taken back into a sequence that held another conversation (the slots'
// load), and after a mark gone back to past later tokens - each the bits of the conversation never set aside (its
// head's cache, indexer and pending h the state's: before, the head attended the other conversation's rows)
static void restored(Osmos* e, const std::vector<int32_t>& ids) {
  char err[OSMOS_ERR] = "";
  int32_t ref[OSMOS_MAX_BLOCK], got[OSMOS_MAX_BLOCK];
  float rc[OSMOS_MAX_BLOCK], gc[OSMOS_MAX_BLOCK];
  OsmosSeq* a = oseq_create(e, err);
  const int32_t y = prompt(e, a, ids, true);
  const int k = oseq_draft(a, y, 0.0f, 0.0f, ref, rc, err);
  CHECK(k > 0);
  std::vector<uint8_t> st(oseq_state_bytes(a));
  CHECK(oseq_save(a, st.data(), st.size(), err));
  oseq_destroy(a);
  OsmosSeq* b = oseq_create(e, err);  // another conversation first: its head's rows where A's will be read
  int32_t d[OSMOS_MAX_BLOCK];
  float c[OSMOS_MAX_BLOCK];
  const int32_t yb = prompt(e, b, ids, true, 56, 144);
  CHECK(oseq_draft(b, yb, 0.0f, 0.0f, d, c, err) >= 0);
  CHECK(oseq_load(b, st.data(), st.size(), err));
  CHECK(oseq_draft(b, y, 0.0f, 0.0f, got, gc, err) == k && !std::memcmp(got, ref, k * 4) && !std::memcmp(gc, rc, k * 4));
  oseq_destroy(b);
  OsmosSeq* m = oseq_create(e, err);  // a mark at the prompt, 40 tokens more (seeded), the mark gone back to
  CHECK(prompt(e, m, ids, true) == y);
  std::vector<uint8_t> mk(oseq_mark_bytes(m));
  CHECK(oseq_mark_save(m, mk.data(), mk.size(), err));
  std::vector<OsmosSeq*> ss(40, m);
  CHECK(osmos_batch(e, ss.data(), ids.data() + 120, 40, err) && oseq_seed(m, kPrompt, kPrompt + 40, err));
  CHECK(oseq_mark_load(m, mk.data(), mk.size(), err));
  CHECK(oseq_draft(m, y, 0.0f, 0.0f, got, gc, err) == k && !std::memcmp(got, ref, k * 4) && !std::memcmp(gc, rc, k * 4));
  oseq_destroy(m);
  std::printf("  the head's %d drafts after a state taken back into another conversation's sequence, and after a "
              "mark gone back to: the bits of the conversation never set aside\n", k);
}

// greedy decode plain, then drafted the server's way, on an engine with the drafter at path: the same tokens
static void drafted(OsmosModel* m, const std::vector<int32_t>& ids, const std::string& path, const char* name) {
  char err[OSMOS_ERR] = "";
  const OsmosOptions o = {kCtx, kStep, 0, nullptr, false, path.c_str(), false, 0, false};
  Osmos* e = osmos_create(m, &o, err);
  if (!e) std::fprintf(stderr, "  %s\n", err);
  CHECK(e);
  const uint32_t block = osmos_block(e);
  CHECK(block > 0 && block <= OSMOS_MAX_BLOCK);
  std::printf("  %s: %u drafts a round at most\n", name, block);

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
  if (!std::strcmp(name, "the MTP head")) restored(e, ids);
  drafted_together(e, ids, OSMOS_DRAFT_CONF_MIN, OSMOS_DRAFT_CUM_MIN, false);
  drafted_together(e, ids, 0.0f, 0.0f, true);  // the first's drafts cut at the context's end, the others' whole
  osmos_destroy(e);
}

int main() {
  std::printf("[test_mtp_q4x_gpu] start\n");
  const std::string first = home_path(kQwenFirst), mtp = home_path(kMtp), dflash = home_path(kDflash);
  const std::vector<int32_t> ids = dump_ids();
  int ndev = 0;
  const bool has_mtp = !access(mtp.c_str(), R_OK), has_dflash = !access(dflash.c_str(), R_OK);
  if (access(first.c_str(), R_OK) || (!has_mtp && !has_dflash) || ids.size() < kPrompt ||
      hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_mtp_q4x_gpu] skipped: no Qwen3.8 UD-IQ3_XXS, a drafter, the dump's ids or a GPU\n");
    return 77;
  }
  char err[OSMOS_ERR] = "";
  const OsmosFiles files = {first.c_str(), nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  CHECK(m);
  // the model as its own drafter: not an MTP head
  const OsmosOptions bad = {kCtx, kStep, 0, nullptr, false, first.c_str(), false, 0, false};
  CHECK(!osmos_create(m, &bad, err) && std::strstr(err, "nextn"));
  if (has_mtp) drafted(m, ids, mtp, "the MTP head");
  if (has_dflash) drafted(m, ids, dflash, "DFlash2");
  osmos_close(m);
  std::printf("[test_mtp_q4x_gpu] ok\n");
  return 0;
}
