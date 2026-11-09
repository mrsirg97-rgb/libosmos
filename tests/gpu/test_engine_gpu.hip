// the engine interface (osmos/run/engine.h) against the family's own calls: a real-text prompt, greedy steps and
// drafted rounds through DeepSeek's forward and drafter as osmos-serve called them before 0.4.0, then through the
// engine - the same tokens, the same logits bit for bit, the same drafts; a sequence saved and loaded; the refusals
// (no file, another family's file named, an extra the family refuses, drafting without a drafter). needs a gfx1151,
// the UD-IQ1_M set (test_model.h), its drafter and the bench's text; skipped (77) without them.
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "state_stream.h"

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/model/weights.h"
#include "osmos/run/drafter.h"
#include "osmos/run/engine.h"
#include "osmos/run/forward.h"
}
#include "test_model.h"

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr uint64_t kBudget = 64ull << 30;  // the drafter's ~5 GB beside it, of this box's ~113 GiB
constexpr uint32_t kCtx = 4096;
constexpr int kPrompt = 256, kSteps = 8, kRounds = 6;
constexpr int kLong = MM_MAX_TOKENS + 52;  // a step past a sub-batch: the forward keeps its last 52 rows' logits

// what a run gives: its greedy tokens, the last step's logits, its drafted rounds' drafts, a saved state's bytes
struct Run {
  std::vector<int32_t> tokens, drafts;
  std::vector<float> logits;
  std::vector<uint8_t> state;
};

// a GGUF of no tensors, its one key general.architecture = arch (v3: magic, version, counts, the key and its string)
static bool write_header(const char* path, const char* arch) {
  FILE* f = std::fopen(path, "wb");
  if (!f) return false;
  const char* key = "general.architecture";
  const uint32_t version = 3, string_type = 8;
  const uint64_t n_tensors = 0, n_kv = 1, kl = std::strlen(key), al = std::strlen(arch);
  bool ok = std::fwrite("GGUF", 1, 4, f) == 4 && std::fwrite(&version, 4, 1, f) == 1 &&
            std::fwrite(&n_tensors, 8, 1, f) == 1 && std::fwrite(&n_kv, 8, 1, f) == 1 &&
            std::fwrite(&kl, 8, 1, f) == 1 && std::fwrite(key, 1, kl, f) == kl &&
            std::fwrite(&string_type, 4, 1, f) == 1 && std::fwrite(&al, 8, 1, f) == 1 &&
            std::fwrite(arch, 1, al, f) == al;
  return std::fclose(f) == 0 && ok;
}

static std::vector<int32_t> prompt_ids(int n) {
  const char* home = std::getenv("HOME");
  FILE* f = std::fopen((std::string(home ? home : "") + "/bench/osmos-prompts/calib16k.txt").c_str(), "r");
  std::vector<int32_t> ids;
  for (int32_t v; f && (int) ids.size() < n && std::fscanf(f, "%d,", &v) == 1;) ids.push_back(v);
  if (f) std::fclose(f);
  return ids;
}

// the run through any family's calls: batch, greedy, logits, save, seed, draft, rollback - as function objects
template <class Batch, class Greedy, class Logits, class Save, class Seed, class Draft, class Rollback>
static Run run(const std::vector<int32_t>& ids, uint32_t vocab, Batch batch, Greedy greedy, Logits logits, Save save,
               Seed seed, Draft draft, Rollback rollback) {
  Run r;
  CHECK(batch(ids.data(), (uint32_t) ids.size()));
  int32_t y = -1;
  CHECK(greedy((uint32_t) ids.size() - 1, &y));
  r.state = save();
  int32_t pos = kPrompt;
  CHECK(seed(pos - 128, pos));  // from the prompt's step: the drafter reads the step that just ran
  for (int k = 0; k < kRounds; k++) {  // drafted rounds: drafts, the verify, the rejected undone, the kept seeded
    int32_t dr[OSMOS_MAX_BLOCK], toks[OSMOS_MAX_BLOCK + 1], picks[OSMOS_MAX_BLOCK + 1];
    const int n = draft(y, dr);
    CHECK(n >= 0);
    toks[0] = y;
    std::memcpy(toks + 1, dr, n * sizeof *dr);
    CHECK(batch(toks, (uint32_t) n + 1));
    for (int i = 0; i <= n; i++) CHECK(greedy((uint32_t) i, &picks[i]));
    int m = 0;
    while (m < n && picks[m] == dr[m]) m++;
    r.drafts.insert(r.drafts.end(), dr, dr + n);
    r.tokens.insert(r.tokens.end(), toks, toks + m + 1);
    CHECK(rollback((uint32_t) (n - m)) && seed(pos, pos + m + 1));
    y = picks[m], pos += m + 1;
  }
  for (int i = 0; i < kSteps; i++) {  // then greedy steps, a token at a time
    r.tokens.push_back(y);
    CHECK(batch(&y, 1) && greedy(0, &y));
  }
  r.logits.resize(vocab);
  CHECK(hipMemcpy(r.logits.data(), logits(), vocab * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return r;
}

int main() {
  std::printf("[test_engine_gpu] start\n");
  const char* home = std::getenv("HOME");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  const std::string ddir = std::string(home ? home : "") + "/Models/ds1017/dspark-iq2xs";
  const std::vector<int32_t> ids = prompt_ids(kPrompt), longp = prompt_ids(kLong);
  int ndev = 0;
  if (which != TestModel::kUd || (int) longp.size() < kLong || access((ddir + "/dspark.bin").c_str(), R_OK) ||
      hipGetDeviceCount(&ndev) || !ndev) {
    std::printf("[test_engine_gpu] skipped: no UD-IQ1_M set, drafter, the bench's text or GPU\n");
    return 77;
  }
  char err[OSMOS_ERR] = "";

  // refusals, before any GPU memory: no files, no model, another family's file named, a table it cannot read
  CHECK(!osmos_open(nullptr, err) && err[0]);
  OsmosFiles none = {"/nonexistent.gguf", nullptr, nullptr};
  CHECK(!osmos_open(&none, err) && err[0]);
  {
    const std::string other = std::string("/tmp/osmos-test-llama-") + std::to_string(getpid()) + ".gguf";
    CHECK(write_header(other.c_str(), "llama"));
    OsmosFiles q = {other.c_str(), nullptr, nullptr};
    err[0] = '\0';
    CHECK(!osmos_open(&q, err) && std::strstr(err, "llama"));
    std::printf("  a file of an architecture no family runs refused: %s\n", err);
    unlink(other.c_str());
  }
  OsmosFiles badtable = {path, "/nonexistent-table.gguf", nullptr};
  CHECK(!osmos_open(&badtable, err) && err[0]);

  OsmosFiles files = {path, nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  if (!m) std::fprintf(stderr, "  %s\n", err);
  CHECK(m);
  const OsmosFacts fx = osmos_facts(m);

  // the family's own calls, as osmos-serve made them before 0.4.0 (forward_lane 0: its decode's order)
  Run direct;
  {
    char gerr[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "", ferr[FORWARD_ERR] = "";
    Gguf* g = gguf_create(path, gerr);
    const WeightsOptions wo = {g, nullptr, nullptr};
    Weights* w = g ? weights_create_with(&wo, werr) : nullptr;
    CHECK(w);
    CHECK(std::strcmp(fx.family, "deepseek41") == 0 && fx.vocab == (uint32_t) w->geo.n_vocab &&
          fx.round_tokens == (uint32_t) ((w->geo.n_experts - 1) / w->geo.n_used) &&
          fx.seed_window == (uint32_t) w->geo.window && fx.native_ctx > 0);
    const ForwardOptions fo = {(int) w->geo.n_layers, kCtx, kBudget, nullptr, 0, true};
    Forward* f = forward_create_with(w, &fo, ferr);
    CHECK(f && forward_lane(f, 0, NULL, ferr));
    Drafter* d = drafter_create(f, ddir.c_str(), ferr);
    CHECK(d);
    Seq* s = seq_create(f, ferr);
    CHECK(s);
    std::vector<Seq*> ss(OSMOS_MAX_BLOCK + 1 > kPrompt ? OSMOS_MAX_BLOCK + 1 : kPrompt, s);
    direct = run(
        ids, fx.vocab, [&](const int32_t* t, uint32_t n) { return forward_batch(f, ss.data(), t, n, ferr); },
        [&](uint32_t row, int32_t* id) { return forward_greedy(f, row, 1, id, ferr); },
        [&] { return forward_logits(f); },
        [&] {
          std::vector<uint8_t> b(seq_state_bytes(s));
          CHECK(save_buf(seq_save_to, s, b.data(), b.size(), ferr));
          return b;
        },
        [&](int32_t from, int32_t to) { return drafter_seed(d, s, from, to, ferr); },
        [&](int32_t y, int32_t* dr) {
          float cf[DRAFTER_BLOCK];
          return drafter_draft(d, s, y, OSMOS_DRAFT_CONF_MIN, OSMOS_DRAFT_CUM_MIN, dr, cf, ferr);
        },
        [&](uint32_t n) { return seq_rollback(s, n); });
    seq_destroy(s);
    drafter_destroy(d);
    forward_destroy(f);
    weights_destroy(w);
    gguf_destroy(g);
  }

  // the same through the engine
  OsmosOptions o = {kCtx, 0, kBudget, nullptr, true, "/nonexistent-drafter", false, 0};
  CHECK(!osmos_create(m, &o, err) && err[0]);  // a drafter that is not there: refused, not built without
  o.drafter = ddir.c_str();
  Osmos* e = osmos_create(m, &o, err);
  if (!e) std::fprintf(stderr, "  %s\n", err);
  CHECK(e);
  CHECK(osmos_block(e) == DRAFTER_BLOCK && osmos_identity(e) != 0);
  uint64_t room = 0;
  CHECK(osmos_room(e, &room) && room > 0);
  OsmosStats st;
  osmos_stats(e, &st);
  CHECK(st.budget > 0 && st.budget <= kBudget);
  OsmosSeq* s = oseq_create(e, err);
  CHECK(s && oseq_position(s) == 0 && oseq_reach(s) == 0);
  std::vector<OsmosSeq*> ss(kPrompt, s);
  const Run via = run(
      ids, fx.vocab, [&](const int32_t* t, uint32_t n) { return osmos_batch(e, ss.data(), t, n, err); },
      [&](uint32_t row, int32_t* id) { return osmos_greedy(e, row, 1, id, err); },
      [&] { return osmos_logits(e, 0); },
      [&] {
        std::vector<uint8_t> b(oseq_state_bytes(s));
        CHECK(oseq_save(s, b.data(), b.size(), err));
        return b;
      },
      [&](int32_t from, int32_t to) { return oseq_seed(s, from, to, err); },
      [&](int32_t y, int32_t* dr) {
        float cf[OSMOS_MAX_BLOCK];
        return oseq_draft(s, y, OSMOS_DRAFT_CONF_MIN, OSMOS_DRAFT_CUM_MIN, dr, cf, err);
      },
      [&](uint32_t n) { return oseq_rollback(s, n); });
  CHECK(via.tokens == direct.tokens && via.drafts == direct.drafts && !via.drafts.empty());
  CHECK(std::memcmp(via.logits.data(), direct.logits.data(), via.logits.size() * 4) == 0);
  CHECK(via.state == direct.state);
  CHECK(oseq_reach(s) == SEQ_MAX_BACK && !oseq_rollback(s, SEQ_MAX_BACK + 1));
  std::printf("  %zu tokens, %zu drafts and the last step's %u logits: the family's own calls' bits\n",
              via.tokens.size(), via.drafts.size(), fx.vocab);

  // a sequence saved after the prompt, loaded into another: its next step the saved one's; another size refused,
  // the sequence then empty
  OsmosSeq* t = oseq_create(e, err);
  CHECK(t && oseq_load(t, direct.state.data(), direct.state.size(), err) && oseq_position(t) == kPrompt);
  CHECK(!oseq_load(t, direct.state.data(), direct.state.size() - 1, err) && oseq_position(t) == 0);  // reset
  CHECK(oseq_load(t, direct.state.data(), direct.state.size(), err) && oseq_position(t) == kPrompt);
  int32_t y = direct.tokens[0], next = -1;
  const bool stepped = osmos_batch(e, &t, &y, 1, err) && osmos_greedy(e, 0, 1, &next, err);
  if (!stepped || next != direct.tokens[1])
    std::fprintf(stderr, "  loaded: stepped %d (%s), next %d against %d\n", stepped, err, next, direct.tokens[1]);
  CHECK(stepped && next == direct.tokens[1]);
  std::printf("  a state saved after the prompt and loaded: the next token the same\n");

  // streamed (oseq_save_to / oseq_load_from): pieces within STATE_CHUNK, the buffer's bytes; loaded from pieces, its
  // next token the same; a byte damaged mid-stream refused (the sequence empty); a sink failing part way, false
  {
    OsmosSeq* u = oseq_create(e, err);
    CHECK(u && oseq_load(u, direct.state.data(), direct.state.size(), err));
    StreamSink sk;
    CHECK(oseq_save_to(u, stream_sink, &sk, err) && !sk.over && sk.bytes == direct.state && sk.pieces >= 2);
    StreamSource src{&sk.bytes};
    OsmosSeq* v = oseq_create(e, err);
    CHECK(v && oseq_load_from(v, sk.bytes.size(), stream_source, &src, err) && !src.over &&
          src.at == sk.bytes.size() && oseq_position(v) == kPrompt);
    int32_t vn = -1;
    CHECK(osmos_batch(e, &v, &y, 1, err) && osmos_greedy(e, 0, 1, &vn, err) && vn == direct.tokens[1]);
    StreamSource bad{&sk.bytes};
    bad.flip = sk.bytes.size() / 2;
    CHECK(!oseq_load_from(v, sk.bytes.size(), stream_source, &bad, err) && oseq_position(v) == 0);
    StreamSink cut;
    cut.fail_after = sk.bytes.size() / 2;
    CHECK(!oseq_save_to(u, stream_sink, &cut, err));
    oseq_destroy(u), oseq_destroy(v);
    std::printf("  streamed: %llu pieces of at most %llu bytes, the buffer's bytes; loaded from pieces, the next token "
                "the same; damaged mid-stream refused\n", (unsigned long long) sk.pieces, (unsigned long long) sk.widest);
  }

  // a long step's rows by the batch's own numbers: the kept ones served, the one before them refused
  OsmosSeq* u = oseq_create(e, err);
  std::vector<OsmosSeq*> us(kLong, u);
  int32_t id = -1;
  CHECK(u && osmos_batch(e, us.data(), longp.data(), kLong, err));
  CHECK(!osmos_greedy(e, MM_MAX_TOKENS - 1, 1, &id, err) && !osmos_logits(e, MM_MAX_TOKENS - 1));
  CHECK(osmos_greedy(e, MM_MAX_TOKENS, 1, &id, err) && osmos_logits(e, kLong - 1) && !osmos_logits(e, kLong));
  CHECK(osmos_greedy(e, MM_MAX_TOKENS, kLong - MM_MAX_TOKENS, std::vector<int32_t>(kLong).data(), err));
  CHECK(!osmos_greedy(e, MM_MAX_TOKENS, kLong - MM_MAX_TOKENS + 1, std::vector<int32_t>(kLong).data(), err));
  oseq_destroy(u);
  std::printf("  a %d-token step: rows %d .. %d served by their own numbers, row %d refused\n", kLong, MM_MAX_TOKENS,
              kLong - 1, MM_MAX_TOKENS - 1);
  oseq_destroy(t);
  oseq_destroy(s);
  osmos_destroy(e);

  // no drafter: no block, and drafting refused
  o.drafter = nullptr;
  e = osmos_create(m, &o, err);
  CHECK(e && osmos_block(e) == 0);
  s = oseq_create(e, err);
  int32_t dr[OSMOS_MAX_BLOCK];
  float cf[OSMOS_MAX_BLOCK];
  CHECK(s && !oseq_seed(s, 0, 1, err) && oseq_draft(s, 0, 0.1f, 0.1f, dr, cf, err) < 0);
  oseq_destroy(s);
  osmos_destroy(e);
  osmos_close(m);
  std::printf("[test_engine_gpu] ok\n");
  return 0;
}
