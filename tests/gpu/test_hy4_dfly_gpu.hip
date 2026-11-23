// Hy4-flash's DFly2 drafter (osmos/hy4/forward.h: hy4_dfly_load, hy4_seq_draft) on the GGUF and the release's drafter
// file: a prompt, then rounds of drafts verified in one step each (greedy: the leading drafts the target's own picks
// kept, the rest rolled back) - the tokens bit for bit plain greedy decoding's, the drafts' mean acceptance stated and
// bounded below (a broken draft accepts ~0). OSMOS_TEST_HY4_DFLY_DUMP=dir: the first round's draft state and outputs
// for oracle/hy4_dfly.py (the reference's draft on osmos's own target state); with its fixture
// (OSMOS_TEST_DFLY_Q8: the engine's Q8_0 drafter, its agreement with the reference reported, not bounded)
// (~/osmos-oracle/hy4_dfly.bin, OSMOS_TEST_HY4_DFLY_REF) the draft's hidden rows and tokens against the reference's.
// needs the GGUF (OSMOS_TEST_HY4), the drafter (OSMOS_TEST_DFLY: model-00148-of-00148.safetensors) and the ids of
// ~/osmos-oracle/hy4_forward.bin; skipped (77) without. run under the memory guard (~95 GB on the GPU).
extern "C" {
#include <fcntl.h>
#include <unistd.h>

#include "osmos/err.h"
#include "osmos/gpu/hy4_ops.h"
#include "osmos/hy4/forward.h"
#include "osmos/model/gguf.h"
}

#include <string>

#include "hy4_fixture.h"
extern "C" {
#include "../../src/hy4/hy4_impl.h"  // (last: its F macro after HIP's headers) - the sequence's snapshot, for the dump
}
#undef F

constexpr uint32_t kPromptDefault = 600;  // the prompt's tokens (past the window, under the indexer's 2,048);
                                          // OSMOS_TEST_DFLY_PROMPT another (2,600: the indexer's picks sparse)
constexpr float kProbTol = 0.02f;  // a draft's probability against the reference's (osmos's k_b / v_b Q8_0, its BF16)
constexpr uint32_t kGen = 96;       // the tokens generated each way
constexpr double kMinAccept = 1.0;  // drafts accepted a round at least (a broken draft: ~0)

static void dump(const std::string& dir, const char* name, const void* dev, size_t bytes, bool host = false) {
  std::vector<char> b(bytes);
  if (host)
    std::memcpy(b.data(), dev, bytes);
  else
    CHECK(hipMemcpy(b.data(), dev, bytes, hipMemcpyDeviceToHost) == hipSuccess);
  FILE* f = std::fopen((dir + "/" + name).c_str(), "wb");
  CHECK(f && std::fwrite(b.data(), 1, bytes, f) == bytes);
  std::fclose(f);
}

int main() {
  char err[OSMOS_ERR] = "", gerr[OSMOS_ERR] = "";
  const char* home = getenv("HOME") ? getenv("HOME") : "";
  const std::string model = getenv("OSMOS_TEST_HY4") ? getenv("OSMOS_TEST_HY4")
                                                     : std::string(home) + "/Models/hy4-flash/Hy4-flash-Q3_K-00001-of-00002.gguf";
  const std::string dfly = getenv("OSMOS_TEST_DFLY") ? getenv("OSMOS_TEST_DFLY")
                                                     : std::string(home) + "/Models/hy4-flash/model-00148-of-00148.safetensors";
  int devs = 0;
  Gguf* g = nullptr;
  if (hipGetDeviceCount(&devs) != hipSuccess || !devs || !load((std::string(home) + "/osmos-oracle/hy4_forward.bin").c_str()) ||
      access(dfly.c_str(), R_OK) || !(g = gguf_create(model.c_str(), gerr))) {
    std::printf("[test_hy4_dfly_gpu] skipped: no GPU, ids, drafter or model (%s)\n", gerr);
    return 77;
  }
  osmos_clear(err);
  Hy4Weights* w = hy4_weights_create(g, err);
  CHECK(w);
  std::vector<int> fds(g->n_files);
  for (int i = 0; i < g->n_files; i++) CHECK((fds[i] = open(g->paths[i], O_RDONLY)) >= 0);
  const uint32_t kPrompt = getenv("OSMOS_TEST_DFLY_PROMPT") ? (uint32_t) atoi(getenv("OSMOS_TEST_DFLY_PROMPT"))
                                                             : kPromptDefault;
  std::vector<float> fid = A("ids");
  if (const char* ip = getenv("OSMOS_TEST_DFLY_IDS")) {  // (a longer prompt's ids, int32: the drafts past 2,600 tokens)
    FILE* fi = std::fopen(ip, "rb");
    CHECK(fi);
    std::vector<int32_t> v(kPrompt);
    CHECK(std::fread(v.data(), 4, kPrompt, fi) == kPrompt);
    std::fclose(fi);
    fid.assign(v.begin(), v.end());
  }
  CHECK(kPrompt > 0 && fid.size() >= kPrompt);
  const Hy4Options o = {kPrompt, kPrompt + kGen + 64, HY4_DFLY_DRAFTS + 1, getenv("OSMOS_TEST_HY4_KV8") == nullptr};
  osmos_clear(err);
  Hy4Forward* f = hy4_forward_create(w, fds.data(), &o, err);
  if (!f) std::printf("  refused: %s\n", err);
  CHECK(f);
  const bool q8 = getenv("OSMOS_TEST_DFLY_Q8") != nullptr;  // (the engine's: Q8_0; the reference's check strict in BF16)
  if (!hy4_dfly_load(f, dfly.c_str(), q8, err)) std::printf("  refused: %s\n", err);
  CHECK(f->dfly);
  std::printf("[test_hy4_dfly_gpu] DFly2 loaded beside the forward (%s); a prompt of %u tokens\n", dfly.c_str(), kPrompt);
  std::vector<int32_t> ids(kPrompt);
  for (uint32_t t = 0; t < kPrompt; t++) ids[t] = (int32_t) fid[t];

  // plain greedy: the prompt, then a token a step
  osmos_clear(err);
  Hy4Seq* a = hy4_seq_create(f, err);
  osmos_clear(err);
  CHECK(a && hy4_step(f, a, ids.data(), kPrompt, 1, err));
  std::vector<int32_t> plain;
  int32_t y = 0;
  osmos_clear(err);
  CHECK(hy4_greedy(f, kPrompt - 1, 1, &y, err));  // (the prompt's last row: a step row's index)
  for (uint32_t k = 0; k < kGen; k++) {
    plain.push_back(y);
    osmos_clear(err);
    CHECK(hy4_step(f, a, &y, 1, 1, err) && hy4_greedy(f, 0, 1, &y, err));
  }

  // drafted: rounds of [y, drafts] verified in one step, the rejected rolled back
  osmos_clear(err);
  Hy4Seq* b = hy4_seq_create(f, err);
  osmos_clear(err);
  CHECK(b && hy4_step(f, b, ids.data(), kPrompt, 1, err) && hy4_greedy(f, kPrompt - 1, 1, &y, err));
  std::vector<int32_t> out;
  uint32_t rounds = 0, drafted = 0, accepted = 0;
  const char* dir = getenv("OSMOS_TEST_HY4_DFLY_DUMP");
  while (out.size() < kGen) {
    int32_t d[HY4_DFLY_DRAFTS];
    float c[HY4_DFLY_DRAFTS];
    osmos_clear(err);
    const int k = hy4_seq_draft(f, b, y, HY4_DFLY_DRAFTS, 0.0f, 0.0f, d, c, err);
    if (k < 0) std::printf("  draft failed: %s\n", err);
    CHECK(k >= 0);
    if (rounds == 0) {
      CHECK(k == HY4_DFLY_DRAFTS);  // (the snapshot of the prompt's last row there)
      std::printf("  the first round's drafts after %d:", y);
      for (int j = 0; j < k; j++) std::printf(" %d (p %.3f)", d[j], c[j]);
      std::printf("\n");
      const std::string ref = getenv("OSMOS_TEST_HY4_DFLY_REF") ? getenv("OSMOS_TEST_HY4_DFLY_REF")
                                                                : std::string(home) + "/osmos-oracle/hy4_dfly_" +
                                                                      std::to_string(kPrompt) + ".bin";
      fx.clear();
      if (load(ref.c_str())) {  // the reference's drafts on this state (oracle/hy4_dfly.py on this test's dump)
        const std::vector<float>&rd = A("drafts"), &rp = A("probs");
        CHECK((int) rd.size() == k);
        float worst = 0;
        int same = 0;
        for (int j = 0; j < k; j++) {
          same += (int32_t) rd[j] == d[j];
          worst = std::fmax(worst, std::fabs(rp[j] - c[j]));
        }
        std::printf("  the reference's drafts on this state: %d of %d the same; their probabilities within %.4f%s\n",
                    same, k, worst, q8 ? " (Q8_0: reported, not bounded)" : "");
        if (!q8) CHECK(same == k && worst <= kProbTol);
      } else
        std::printf("  no reference (%s): the drafts not checked against it\n", ref.c_str());
      if (dir) {  // the draft's inputs and outputs (oracle/hy4_dfly.py)
        const Hy4Geometry* geo = &w->geo;
        const uint32_t top = (uint32_t) (geo->ix_topk / geo->ix_pool), D = (uint32_t) geo->n_embd;
        const int32_t anchor = b->pos, i = anchor - 1 - b->dpos0;
        const int32_t meta[4] = {anchor, y, i, (int32_t) top};
        dump(dir, "meta.i32", meta, sizeof meta, true);
        dump(dir, "drafts.i32", d, sizeof d, true);
        dump(dir, "hidden.f32", b->dhid + (uint64_t) i * D, (uint64_t) D * 4);
        for (int li = 0, l = 0; l < geo->n_layers; l++) {
          if (f->dfly_slot[l] < 0) continue;
          li = f->dfly_slot[l];
          const uint32_t lat = (uint32_t) geo->kv_lora, rd = (uint32_t) geo->rope_dim, rf = lat + rd;
          const uint64_t rb = hy4_mla_row_bytes(lat, rd, f->kv8);
          std::vector<uint8_t> raw((size_t) anchor * rb);
          CHECK(hipMemcpy(raw.data(), b->cache[l], raw.size(), hipMemcpyDeviceToHost) == hipSuccess);
          std::vector<float> rows((size_t) anchor * rf);  // (F32 for the oracle: a Q8_0 row's latent decoded)
          for (int32_t t = 0; t < anchor; t++) {
            const uint8_t* r = raw.data() + (size_t) t * rb;
            if (!f->kv8) {
              std::memcpy(&rows[(size_t) t * rf], r, rf * 4);
              continue;
            }
            for (uint32_t blk = 0; blk < lat / 32; blk++) {
              _Float16 d16;
              std::memcpy(&d16, r + blk * 34, 2);
              for (int i = 0; i < 32; i++) rows[(size_t) t * rf + blk * 32 + i] = (float) d16 * (float) (int8_t) r[blk * 34 + 2 + i];
            }
            std::memcpy(&rows[(size_t) t * rf + lat], r + lat / 32 * 34, rd * 4);
          }
          dump(dir, ("cache" + std::to_string(l) + ".f32").c_str(), rows.data(), rows.size() * 4, true);
          dump(dir, ("sel" + std::to_string(l) + ".i32").c_str(), b->dsel + ((uint64_t) li * HY4_DFLY_KEEP + i) * top,
               (uint64_t) top * 4);
          dump(dir, ("nsel" + std::to_string(l) + ".i32").c_str(), b->dn + (uint64_t) li * HY4_DFLY_KEEP + i, 4);
        }
        std::printf("  the first round's draft state written to %s\n", dir);
      }
    }
    std::vector<int32_t> step(1 + k);
    step[0] = y;
    for (int j = 0; j < k; j++) step[1 + j] = d[j];
    const uint32_t len = 1 + (uint32_t) k;  // (the decode's math, as the engine's verify: a token's bits its own)
    osmos_clear(err);
    CHECK(hy4_step_runs(f, &b, &len, 1, step.data(), err));
    std::vector<int32_t> pick(1 + k);
    osmos_clear(err);
    CHECK(hy4_greedy(f, 0, 1 + k, pick.data(), err));
    int acc = 0;
    while (acc < k && d[acc] == pick[acc]) acc++;
    out.push_back(y);
    for (int j = 0; j < acc; j++) out.push_back(d[j]);
    if (k > acc) CHECK(hy4_seq_rollback(b, (uint32_t) (k - acc)));
    y = pick[acc];
    rounds++, drafted += (uint32_t) k, accepted += (uint32_t) acc;
  }
  out.resize(kGen);
  uint32_t same = 0;
  while (same < kGen && out[same] == plain[same]) same++;
  const double mean = (double) accepted / rounds;
  std::printf("  %u tokens drafted in %u rounds: %u drafts, %u accepted (%.2f a round, %.0f%% of the drafted); the "
              "tokens plain greedy's: %u of %u\n",
              kGen, rounds, drafted, accepted, mean, 100.0 * accepted / (drafted ? drafted : 1), same, kGen);
  CHECK(same == kGen);
  CHECK(mean >= kMinAccept);
  hy4_seq_destroy(a), hy4_seq_destroy(b);
  hy4_forward_destroy(f);
  hy4_weights_destroy(w);
  for (int fd : fds) close(fd);
  gguf_destroy(g);
  std::printf("[test_hy4_dfly_gpu] ok\n");
  return 0;
}
