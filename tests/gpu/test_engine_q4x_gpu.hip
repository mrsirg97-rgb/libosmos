// Qwen3.8-Flash-Next behind the engine interface (osmos/run/engine.h): its facts; the options and extras it refuses;
// a batch of two sequences' tokens the bits of each sequence run alone (and of the forward called directly); a
// batch whose sequence's tokens are not consecutive refused; greedy over a batch's rows; a sequence saved and loaded
// going on as the original, a damaged or a short state refused (the sequence then empty: it computes as a fresh
// one); rollbacks the bits of never having run the dropped tokens, refused past the reach; no drafter. needs a
// gfx1151, the UD-IQ3_XXS set (~78 GB of GPU memory) and the dump's ids; skipped (77) without them.
#include <algorithm>

#include "q4x_ref.h"
#include "state_stream.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
#include "osmos/model/fnv.h"
#include "osmos/run/engine.h"
}

constexpr uint32_t kCtx = 8192, kStep = 256;  // (8,192: a state whose cache parts pass STATE_CHUNK)
constexpr uint32_t kLong = 8000;               // the long sequence's tokens: each attention part ~4.7 MB
constexpr uint32_t kA = 40, kB = 24, kSaved = 100;  // a batch's two sequences' tokens; a saved sequence's
constexpr size_t kStatePos = 16;  // the state header's position: after its magic, version (4 bytes each) and identity

static std::vector<int32_t> dump_ids() {
  FILE* f = std::fopen((home_path(kQwenDump) + "/ids.txt").c_str(), "r");
  std::vector<int32_t> v;
  for (int32_t x; f && std::fscanf(f, "%d,", &x) == 1;) v.push_back(x);
  if (f) std::fclose(f);
  return v;
}

static std::vector<float> rows(const Osmos* e, uint32_t row0, uint32_t n, uint32_t vocab) {
  std::vector<float> out((size_t) n * vocab);
  for (uint32_t r = 0; r < n; r++) {
    const float* l = osmos_logits(e, row0 + r);
    CHECK(l && hipMemcpy(&out[(size_t) r * vocab], l, vocab * 4, hipMemcpyDeviceToHost) == hipSuccess);
  }
  return out;
}

int main() {
  std::printf("[test_engine_q4x_gpu] start\n");
  const std::string first = home_path(kQwenFirst);
  const std::vector<int32_t> ids = dump_ids();
  int ndev = 0;
  if (access(first.c_str(), R_OK) || ids.size() < kSaved + 1 || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_engine_q4x_gpu] skipped: no Qwen3.8 UD-IQ3_XXS, its dump's ids or a GPU\n");
    return 77;
  }
  char err[OSMOS_ERR] = "";

  // the facts, and the extra it has no use for (an Engram table; an overlay it takes: test_overlay_q4x_gpu)
  const OsmosFiles table = {first.c_str(), first.c_str(), nullptr};
  CHECK(!osmos_open(&table, err) && err[0]);
  const OsmosFiles files = {first.c_str(), nullptr, nullptr};
  OsmosModel* m = osmos_open(&files, err);
  if (!m) std::fprintf(stderr, "  %s\n", err);
  CHECK(m);
  const OsmosFacts fx = osmos_facts(m);
  CHECK(!std::strcmp(fx.family, "qwen4exp") && fx.vocab == 248320 && fx.native_ctx == 262144 && fx.seed_window == fx.native_ctx &&
        !fx.streams);
  CHECK(fx.round_tokens == (512 - 1) / 10);  // a decode round's tokens: under every expert's read (512 of them, 10 a token)
  const uint32_t V = fx.vocab;

  // the options it refuses: a budget, picks, a drafter, no_guess, a context past the dense attention's
  const OsmosOptions bad[] = {{kCtx, kStep, 1ull << 30, nullptr, false, nullptr},
                              {kCtx, kStep, 0, "/tmp/picks", false, nullptr},
                              {kCtx, kStep, 0, nullptr, false, "/tmp/drafter"},
                              {kCtx, kStep, 0, nullptr, true, nullptr},
                              {OPS_Q4X_MAX_KEYS + 1, kStep, 0, nullptr, false, nullptr}};
  for (const OsmosOptions& o : bad) CHECK(!osmos_create(m, &o, err) && err[0]);
  const OsmosOptions o = {kCtx, kStep, 0, nullptr, false, nullptr, false, 0, false};
  Osmos* e = osmos_create(m, &o, err);
  if (!e) std::fprintf(stderr, "  %s\n", err);
  CHECK(e);
  uint64_t room = 0;
  CHECK(osmos_room(e, &room) && room > 0 && osmos_identity(e) != 0 && osmos_block(e) == 0);

  // each sequence alone: its tokens' logits
  OsmosSeq* a = oseq_create(e, err);
  OsmosSeq* b = oseq_create(e, err);
  CHECK(a && b);
  std::vector<OsmosSeq*> sa(kA, a), sb(kB, b);
  CHECK(osmos_batch(e, sa.data(), ids.data(), kA, err));
  const std::vector<float> alone_a = rows(e, 0, kA, V);
  CHECK(osmos_batch(e, sb.data(), ids.data() + kA, kB, err));
  const std::vector<float> alone_b = rows(e, 0, kB, V);
  oseq_destroy(a), oseq_destroy(b);

  // the two in one batch: the same bits, row for row
  a = oseq_create(e, err), b = oseq_create(e, err);
  std::vector<OsmosSeq*> both(sa.begin(), sa.end());
  for (uint32_t i = 0; i < kB; i++) both.push_back(b);
  for (uint32_t i = 0; i < kA; i++) both[i] = a;
  std::vector<int32_t> toks(ids.begin(), ids.begin() + kA + kB);
  CHECK(osmos_batch(e, both.data(), toks.data(), kA + kB, err));
  CHECK(rows(e, 0, kA, V) == alone_a && rows(e, kA, kB, V) == alone_b);
  CHECK(oseq_position(a) == (int32_t) kA && oseq_position(b) == (int32_t) kB);
  std::vector<int32_t> picks(kA + kB);
  CHECK(osmos_greedy(e, 0, kA + kB, picks.data(), err));
  for (uint32_t r = 0; r < kA + kB; r++) {
    const float* l = &(r < kA ? alone_a : alone_b)[(size_t) (r < kA ? r : r - kA) * V];
    CHECK(picks[r] == (int32_t) (std::max_element(l, l + V) - l));
  }
  CHECK(!osmos_logits(e, kA + kB) && !osmos_greedy(e, kA + kB - 1, 2, picks.data(), err));
  std::printf("  two sequences in one batch: each one's logits the bits of it alone, greedy their greatest\n");

  // a sequence's tokens not consecutive: refused, no sequence moved
  OsmosSeq* split[3] = {a, b, a};
  CHECK(!osmos_batch(e, split, ids.data(), 3, err) && err[0]);
  CHECK(oseq_position(a) == (int32_t) kA && oseq_position(b) == (int32_t) kB);

  // no drafter
  CHECK(oseq_prompt(a, kA));
  int32_t drafts[OSMOS_MAX_BLOCK];
  float conf[OSMOS_MAX_BLOCK];
  CHECK(!oseq_seed(a, 0, 1, err) && oseq_draft(a, ids[0], 0.1f, 0.1f, drafts, conf, err) < 0);
  oseq_destroy(a), oseq_destroy(b);

  // rollback: a step's tokens dropped, then on - the bits of never having run them. the reference: the tokens run
  // straight, in steps of their own split (a token's bits its own however a sequence is split)
  {
    const int32_t* wrong = &ids[kSaved];  // tokens to drop: any others
    OsmosSeq* r = oseq_create(e, err);
    OsmosSeq* ref = oseq_create(e, err);
    CHECK(r && ref && oseq_reach(r) == 0 && oseq_rollback(r, 0) && !oseq_rollback(r, 1));
    std::vector<OsmosSeq*> sr(kA, r), sref(kA + 4, ref);
    CHECK(osmos_batch(e, sr.data(), ids.data(), kA, err) && oseq_reach(r) == OSMOS_MAX_BLOCK);
    CHECK(osmos_batch(e, sref.data(), ids.data(), kA + 4, err));
    const std::vector<float> want = rows(e, kA, 4, V);  // positions kA .. kA + 3
    // a 5-token step dropped whole, then the right 4 as one step
    CHECK(osmos_batch(e, sr.data(), wrong, 5, err) && oseq_rollback(r, 5) && oseq_position(r) == (int32_t) kA);
    CHECK(osmos_batch(e, sr.data(), &ids[kA], 4, err) && rows(e, 0, 4, V) == want);
    // the right 2 then 3 wrong as one step, the 3 dropped, the next right one: its row
    CHECK(oseq_rollback(r, 4));
    std::vector<int32_t> mixed = {ids[kA], ids[kA + 1], wrong[0], wrong[1], wrong[2]};
    CHECK(osmos_batch(e, sr.data(), mixed.data(), 5, err) && rows(e, 0, 2, V) ==
          std::vector<float>(want.begin(), want.begin() + 2 * (size_t) V));
    CHECK(oseq_rollback(r, 3) && oseq_position(r) == (int32_t) kA + 2);
    CHECK(osmos_batch(e, sr.data(), &ids[kA + 2], 1, err) && rows(e, 0, 1, V) ==
          std::vector<float>(want.begin() + 2 * (size_t) V, want.begin() + 3 * (size_t) V));
    // three one-token steps dropped across their steps' bounds, then the last right one
    for (int i = 0; i < 3; i++) CHECK(osmos_batch(e, sr.data(), &wrong[i], 1, err));
    CHECK(oseq_rollback(r, 3) && osmos_batch(e, sr.data(), &ids[kA + 3], 1, err) &&
          rows(e, 0, 1, V) == std::vector<float>(want.begin() + 3 * (size_t) V, want.end()));
    // a step longer than the ring (OSMOS_MAX_BLOCK + 1 tokens: a whole round of drafts and its y), all but its first
    // dropped - the reach's whole, the oldest state the step itself wrote - then the right ones on
    {
      OsmosSeq* d = oseq_create(e, err);
      CHECK(d);
      std::vector<OsmosSeq*> sd(kA, d);
      CHECK(osmos_batch(e, sd.data(), ids.data(), kA, err));
      std::vector<int32_t> long_step(OSMOS_MAX_BLOCK + 1, wrong[0]);
      long_step[0] = ids[kA];
      CHECK(osmos_batch(e, sd.data(), long_step.data(), OSMOS_MAX_BLOCK + 1, err) && oseq_reach(d) == OSMOS_MAX_BLOCK);
      CHECK(oseq_rollback(d, OSMOS_MAX_BLOCK) && oseq_position(d) == (int32_t) kA + 1);
      CHECK(osmos_batch(e, sd.data(), &ids[kA + 1], 3, err) &&
            rows(e, 0, 3, V) == std::vector<float>(want.begin() + (size_t) V, want.end()));
      oseq_destroy(d);
    }
    // past the reach: refused, nothing moved. the reach: the states kept behind - a rollback of r leaves r fewer, a
    // step of n n more, to OSMOS_MAX_BLOCK (here 8, then 3 / 7 / 3 / 8 / 5 / 6 / 8 / 5 / 6)
    const uint32_t reach = oseq_reach(r);
    CHECK(reach == 6 && !oseq_rollback(r, reach + 1) && oseq_position(r) == (int32_t) kA + 4);
    oseq_destroy(r), oseq_destroy(ref);
    std::printf("  rollback: a step dropped whole or in part, single steps across their bounds - the bits of never "
                "having run them\n");
  }

  // saved and loaded: the loaded one goes on as the original, bit for bit
  OsmosSeq* s = oseq_create(e, err);
  OsmosSeq* c = oseq_create(e, err);
  CHECK(s && c);
  std::vector<OsmosSeq*> ss(kSaved, s);
  CHECK(osmos_batch(e, ss.data(), ids.data(), kSaved, err));
  const uint64_t bytes = oseq_state_bytes(s);
  std::vector<uint8_t> st(bytes);
  CHECK(bytes > 0 && oseq_save(s, st.data(), bytes, err));
  CHECK(!oseq_save(s, st.data(), bytes - 1, err));
  CHECK(oseq_load(c, st.data(), bytes, err) && oseq_position(c) == (int32_t) kSaved && oseq_reach(c) == 0);
  CHECK(osmos_batch(e, &s, &ids[kSaved], 1, err));
  const std::vector<float> next_s = rows(e, 0, 1, V);
  CHECK(osmos_batch(e, &c, &ids[kSaved], 1, err));
  CHECK(rows(e, 0, 1, V) == next_s);
  std::printf("  a sequence saved (%.1f MB) and loaded goes on as the original, bit for bit\n", bytes / 1e6);

  // streamed (oseq_save_to / oseq_load_from): pieces within STATE_CHUNK, the buffer's bytes; loaded from pieces it
  // goes on as the original; a byte damaged mid-stream refused (the sequence empty); a sink failing part way, false
  {
    OsmosSeq* w = oseq_create(e, err);  // (s has stepped on since st: a sequence of st's own)
    CHECK(w && oseq_load(w, st.data(), bytes, err));
    StreamSink sk;
    CHECK(oseq_save_to(w, stream_sink, &sk, err) && !sk.over && sk.bytes == st && sk.pieces >= 2);
    OsmosSeq* d = oseq_create(e, err);
    StreamSource src{&sk.bytes};
    CHECK(d && oseq_load_from(d, bytes, stream_source, &src, err) && !src.over && src.at == bytes &&
          oseq_position(d) == (int32_t) kSaved);
    CHECK(osmos_batch(e, &d, &ids[kSaved], 1, err) && rows(e, 0, 1, V) == next_s);
    StreamSource bad{&sk.bytes};
    bad.flip = bytes / 2;
    CHECK(!oseq_load_from(d, bytes, stream_source, &bad, err) && oseq_position(d) == 0);
    StreamSink cut;
    cut.fail_after = bytes / 2;
    CHECK(!oseq_save_to(w, stream_sink, &cut, err));
    oseq_destroy(d), oseq_destroy(w);
    // a long sequence (kLong tokens, the ids cycled): its parts past STATE_CHUNK go in pieces, saved streamed and
    // loaded back its bytes again
    OsmosSeq* big = oseq_create(e, err);
    OsmosSeq* back = oseq_create(e, err);
    CHECK(big && back);
    std::vector<int32_t> lt(kLong);
    for (uint32_t i = 0; i < kLong; i++) lt[i] = ids[i % ids.size()];
    std::vector<OsmosSeq*> bs(kStep, big);
    for (uint32_t at = 0; at < kLong; at += kStep)
      CHECK(osmos_batch(e, bs.data(), &lt[at], std::min(kStep, kLong - at), err));
    StreamSink lk;
    CHECK(oseq_save_to(big, stream_sink, &lk, err) && !lk.over && lk.bytes.size() == oseq_state_bytes(big));
    StreamSource ls{&lk.bytes};
    CHECK(oseq_load_from(back, lk.bytes.size(), stream_source, &ls, err) && !ls.over &&
          oseq_position(back) == (int32_t) kLong);
    StreamSink again;
    CHECK(oseq_save_to(back, stream_sink, &again, err) && again.bytes == lk.bytes);
    std::printf("  a long sequence (%u tokens, %.1f MB): %llu pieces, none past STATE_CHUNK; loaded, its bytes again\n",
                kLong, lk.bytes.size() / 1e6, (unsigned long long) lk.pieces);
    oseq_destroy(big), oseq_destroy(back);
    std::printf("  streamed: %llu pieces of at most %llu bytes (a state of any size through %llu MiB), the buffer's "
                "bytes; loaded from pieces, the same bits; damaged mid-stream refused\n", (unsigned long long) sk.pieces,
                (unsigned long long) sk.widest, (unsigned long long) (STATE_CHUNK >> 20));
  }

  // damaged, short, of another position's size: refused, the sequence then empty
  std::vector<uint8_t> bad_st = st;
  bad_st[bytes / 2] ^= 1;
  CHECK(!oseq_load(c, bad_st.data(), bytes, err) && oseq_position(c) == 0);
  CHECK(!oseq_load(c, st.data(), bytes - 8, err) && oseq_position(c) == 0);
  std::vector<uint8_t> forged = st;  // another position in its header, its checksum made good: not that size
  int32_t pos;
  std::memcpy(&pos, &forged[kStatePos], 4), pos -= 1, std::memcpy(&forged[kStatePos], &pos, 4);
  const uint64_t sum = fnv(forged.data(), bytes - 8, FNV_SEED);
  std::memcpy(&forged[bytes - 8], &sum, 8);
  CHECK(!oseq_load(c, forged.data(), bytes, err) && oseq_position(c) == 0);
  std::vector<OsmosSeq*> sc(kA, c);  // empty indeed: its states a fresh sequence's
  CHECK(osmos_batch(e, sc.data(), ids.data(), kA, err) && rows(e, 0, kA, V) == alone_a);
  CHECK(oseq_load(c, st.data(), bytes, err) && oseq_position(c) == (int32_t) kSaved);
  oseq_destroy(s), oseq_destroy(c);

  osmos_destroy(e);
  osmos_close(m);
  std::printf("[test_engine_q4x_gpu] ok\n");
  return 0;
}
