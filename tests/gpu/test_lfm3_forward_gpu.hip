// LFM3's forward (osmos/lfm3/forward.h) against Liquid's own (oracle/lfm3_forward.py: modeling_lfm3.py in F32 on the
// GGUF's own values): its ids through one step - every layer's output at the reference's kept rows, the last rows'
// logits (KL against the reference's, its top-1) - then the same ids in uneven steps, the last rows' logits the same
// bits. osmos1.7b's Q4_0 file (OSMOS_TEST_LFM3) and ~/osmos-oracle/lfm3_forward_64_kvq8.bin (OSMOS_TEST_LFM3_REF:
// another, lfm3_forward_2048_kvq8.bin past the window); skipped (77) without.
extern "C" {
#include "osmos/lfm3/forward.h"
#include "osmos/model/gguf.h"
}

#include "oracle_fixture.h"

// the reference rounds K and V through Q8_0 as osmos's cache does (lfm3_forward.py --kv-q8): F32 both sides over the
// same weights and the same cache values, the order of every sum another - and a key a rounding's boundary apart
// lands a code apart, so a layer's output drifts past F32's ~1e-6. measured (2026-11-18): the layers' worst 9.1e-4 of
// the largest (64 ids), 1.6e-3 (2,048); the logits' KL mean 4.8e-6 / 9.7e-6, max 2.3e-5 / 1.0e-4, top-1 64 / 63 of
// 64 (a near-tie). against the reference's own F32 cache (no --kv-q8): KL mean 8.2e-5 / 3.5e-5, max 8.7e-4 / 1.4e-4,
// top-1 64 / 64 - the cache's cost. the bounds the measurements' with room; a wrong term (the convolution's taps
// off by one, a window's edge) far past them
constexpr double kTolLayer = 5e-3, kKlMean = 5e-5, kKlMax = 5e-4;
constexpr uint32_t kTop1Miss = 1;
// a sequence crossing a YaRN stage against one begun in it: its earlier tokens read under the plain rotation, their
// keys turned and Q8_0 again - measured KL mean 8.2e-4, max 9.6e-3, top-1 64 / 64 (the turn skipped: 0.05, 0.18,
// 51 / 64)
constexpr double kStageKlMean = 2e-3, kStageKlMax = 2.5e-2;
// the head's Q8_0 copy against the file's BF16 head on the same streams, measured (2026-11-18): KL mean 3.9e-5 / 3.6e-5,
// max 1.0e-4 / 1.05e-4, top-1 64 / 64 (64 / 2,048 ids) - the Q8_0 cache's own order; the bounds with room
constexpr double kHeadKlMean = 2e-4, kHeadKlMax = 1e-3;
constexpr uint32_t kHeadTop1Miss = 1;

struct Taps {
  std::vector<int32_t> rows;
  std::vector<float> got;  // [layer][row][D]
  uint32_t layers = 0, dim = 0;
};

static void tap(void* arg, int layer, const char* what, const float* dev, uint32_t n, uint32_t dim) {
  Taps* t = (Taps*) arg;
  if (std::strcmp(what, "residual") || n < (uint32_t) t->rows.back() + 1) return;
  t->dim = dim;
  if (t->got.empty()) t->got.resize((size_t) t->layers * t->rows.size() * dim);
  for (size_t r = 0; r < t->rows.size(); r++)
    CHECK(hipMemcpy(t->got.data() + ((size_t) layer * t->rows.size() + r) * dim, dev + (size_t) t->rows[r] * dim,
                    dim * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
}

static double kl(const float* p, const float* q, uint32_t V) {  // KL(reference || ours), nats
  double mp = -INFINITY, mq = -INFINITY;
  for (uint32_t i = 0; i < V; i++) mp = std::fmax(mp, p[i]), mq = std::fmax(mq, q[i]);
  double zp = 0, zq = 0;
  for (uint32_t i = 0; i < V; i++) zp += std::exp(p[i] - mp), zq += std::exp(q[i] - mq);
  double k = 0;
  for (uint32_t i = 0; i < V; i++) {
    const double lp = p[i] - mp - std::log(zp), lq = q[i] - mq - std::log(zq);
    k += std::exp(lp) * (lp - lq);
  }
  return k;
}

static uint32_t argmax(const float* x, uint32_t V) {
  uint32_t b = 0;
  for (uint32_t i = 1; i < V; i++) b = x[i] > x[b] ? i : b;
  return b;
}

int main() {
  const char* home = getenv("HOME") ? getenv("HOME") : "";
  char model_path[512], ref_path[512], err[LFM3_ERR] = "", gerr[GGUF_ERR] = "";
  std::snprintf(model_path, sizeof model_path, "%s", getenv("OSMOS_TEST_LFM3") ? getenv("OSMOS_TEST_LFM3") : "");
  if (!model_path[0]) std::snprintf(model_path, sizeof model_path, "%s/Models/osmos1.7b/osmos1.7b-Q4_0.gguf", home);
  std::snprintf(ref_path, sizeof ref_path, "%s", getenv("OSMOS_TEST_LFM3_REF") ? getenv("OSMOS_TEST_LFM3_REF") : "");
  if (!ref_path[0]) std::snprintf(ref_path, sizeof ref_path, "%s/osmos-oracle/lfm3_forward_64_kvq8.bin", home);
  int devs = 0;
  Gguf* g = nullptr;
  if (hipGetDeviceCount(&devs) != hipSuccess || !devs || !load(ref_path) || !(g = gguf_create(model_path, gerr))) {
    std::printf("[test_lfm3_forward_gpu] skipped: no GPU, no %s (oracle/lfm3_forward.py) or no model (%s)\n", ref_path,
                gerr);
    return 77;
  }
  Lfm3Weights* w = lfm3_weights_create(g, err);
  if (!w) std::printf("  refused: %s\n", err);
  CHECK(w);
  const std::vector<float>& fid = A("ids");
  const uint32_t T = (uint32_t) fid.size(), V = (uint32_t) w->geo.n_vocab, last = 64 < T ? 64 : T;
  std::vector<int32_t> ids(T);
  for (uint32_t t = 0; t < T; t++) ids[t] = (int32_t) fid[t];
  const Lfm3Options o = {T, T, last, LFM3_HEAD_FILE, nullptr, 0, nullptr, 0, 0, 0};  // the reference's head: the file's BF16
  Lfm3Forward* f = lfm3_forward_create(w, &o, err);
  if (!f) std::printf("  refused: %s\n", err);
  CHECK(f);
  std::printf("[test_lfm3_forward_gpu] %u ids, %d layers; the last %u rows' logits\n", T, w->geo.n_layers, last);

  Taps taps;
  for (float r : A("rows")) taps.rows.push_back((int32_t) r);
  taps.layers = (uint32_t) w->geo.n_layers;
  lfm3_forward_tap(f, tap, &taps);
  Lfm3Seq* s = lfm3_seq_create(f, err);
  CHECK(s);
  if (!lfm3_step(f, s, ids.data(), T, last, err)) std::printf("  step refused: %s\n", err);
  CHECK(lfm3_seq_position(s) == (int32_t) T);
  lfm3_forward_tap(f, nullptr, nullptr);
  const std::vector<float>& ref_taps = A("taps");
  CHECK(taps.got.size() == ref_taps.size());
  const size_t per_layer = taps.rows.size() * taps.dim;
  double worst = 0;
  for (uint32_t l = 0; l < taps.layers; l++) {
    double e = 0, top = 0;
    for (size_t i = 0; i < per_layer; i++) {
      const double r = ref_taps[l * per_layer + i];
      e = std::fmax(e, std::fabs(taps.got[l * per_layer + i] - r)), top = std::fmax(top, std::fabs(r));
    }
    worst = std::fmax(worst, e / top);
    if (l % 6 == 0 || l + 1 == taps.layers || e > kTolLayer * top)
      std::printf("  layer %2u's output: max error %.3g of %.4g (%.2g)\n", l, e, top, e / top);
  }
  const std::vector<float> got(lfm3_logits(f), lfm3_logits(f) + (size_t) last * V);
  const std::vector<float>& ref = A("logits");
  const size_t ref_rows = taps.rows.size();
  double kl_sum = 0, kl_max = 0;
  uint32_t same = 0;
  for (uint32_t r = 0; r < last; r++) {
    const float* p = ref.data() + (ref_rows - last + r) * (size_t) V;
    const double k = kl(p, got.data() + (size_t) r * V, V);
    kl_sum += k, kl_max = std::fmax(kl_max, k), same += argmax(p, V) == argmax(got.data() + (size_t) r * V, V);
  }
  std::printf("  the last %u rows' logits: KL mean %.3g, max %.3g nats; top-1 the reference's %u of %u (layers' worst "
              "%.2g)\n", last, kl_sum / last, kl_max, same, last, worst);
  std::printf("  (the one step: a read - the matrix cores' math)\n");
  CHECK(worst <= kTolLayer && kl_sum / last <= kKlMean && kl_max <= kKlMax && same + kTop1Miss >= last);

  // the bits a token's own: the ids in three reads (each past LFM3_MV_TOKENS: lfm3_step's prompt math) - the last
  // step's rows the one step's, bit for bit
  Lfm3Seq* s2 = lfm3_seq_create(f, err);
  CHECK(s2);
  const uint32_t a = T / 3, b = T / 3, c = T - a - b, tail = c < last ? c : last;
  CHECK(a > LFM3_MV_TOKENS && b > LFM3_MV_TOKENS && c > LFM3_MV_TOKENS);
  CHECK(lfm3_step(f, s2, ids.data(), a, 1, err) && lfm3_step(f, s2, ids.data() + a, b, 1, err));
  CHECK(lfm3_step(f, s2, ids.data() + a + b, c, tail, err));
  CHECK(std::memcmp(lfm3_logits(f), got.data() + (size_t) (last - tail) * V, (size_t) tail * V * sizeof(float)) == 0);
  std::printf("  in read steps of %u, %u and %u: the last %u rows the one step's, bit for bit\n", a, b, c, tail);

  // a decode's math over the same ids, one step (lfm3_step_runs, its prompt's last rows kept): within the same bounds
  {
    Lfm3Seq* d = lfm3_seq_create(f, err);
    CHECK(d && lfm3_seq_prompt(d, T - last + 1));
    Lfm3Seq* const one[1] = {d};
    const uint32_t len1[1] = {T};
    CHECK(lfm3_step_runs(f, one, len1, 1, ids.data(), err));
    std::vector<float> row(V);
    double ks = 0, km = 0;
    uint32_t agree = 0;
    for (uint32_t r = 0; r < last; r++) {
      const float* dv = lfm3_row_logits(f, T - last + r);
      CHECK(dv && hipMemcpy(row.data(), dv, V * 4, hipMemcpyDeviceToHost) == hipSuccess);
      const float* p = ref.data() + (ref_rows - last + r) * (size_t) V;
      const double k = kl(p, row.data(), V);
      ks += k, km = std::fmax(km, k), agree += argmax(p, V) == argmax(row.data(), V);
    }
    std::printf("  a decode's math (the matvecs): KL mean %.3g, max %.3g; top-1 the reference's %u of %u\n", ks / last,
                km, agree, last);
    CHECK(ks / last <= kKlMean && km <= kKlMax && agree + kTop1Miss >= last);
    lfm3_seq_destroy(d);
  }

  // two sequences in one step (each its prompt's last row kept), each the bits of its own step alone
  {
    const uint32_t P = T / 4 > 8 ? T / 4 : 8;
    Lfm3Seq *a1 = lfm3_seq_create(f, err), *b1 = lfm3_seq_create(f, err), *a2 = lfm3_seq_create(f, err);
    Lfm3Seq* b2 = lfm3_seq_create(f, err);
    CHECK(a1 && b1 && a2 && b2);
    std::vector<int32_t> both(ids.begin(), ids.begin() + P);
    both.insert(both.end(), ids.begin() + 7, ids.begin() + 7 + P);
    CHECK(lfm3_seq_prompt(a1, P) && lfm3_seq_prompt(b1, P));
    Lfm3Seq* const pair[2] = {a1, b1};
    const uint32_t lens[2] = {P, P};
    CHECK(lfm3_step_read(f, pair, lens, 2, both.data(), err));  // a read: lfm3_step's math for P tokens
    std::vector<float> ra(V), rb(V);
    CHECK(lfm3_row_logits(f, P - 1) && lfm3_row_logits(f, 2 * P - 1) && !lfm3_row_logits(f, 0));
    CHECK(hipMemcpy(ra.data(), lfm3_row_logits(f, P - 1), V * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(hipMemcpy(rb.data(), lfm3_row_logits(f, 2 * P - 1), V * 4, hipMemcpyDeviceToHost) == hipSuccess);
    int32_t ga[1], gb[1];
    CHECK(lfm3_greedy(f, P - 1, 1, ga, err) && lfm3_greedy(f, 2 * P - 1, 1, gb, err));
    CHECK(lfm3_step(f, a2, ids.data(), P, 1, err) && std::memcmp(lfm3_logits(f), ra.data(), V * 4) == 0);
    CHECK(ga[0] == (int32_t) argmax(ra.data(), V));
    CHECK(lfm3_step(f, b2, ids.data() + 7, P, 1, err) && std::memcmp(lfm3_logits(f), rb.data(), V * 4) == 0);
    CHECK(gb[0] == (int32_t) argmax(rb.data(), V));
    std::printf("  two sequences of %u in one step: each the bits of its own step alone; the greedy ids the argmax's\n",
                P);

    // a rollback: 4 more tokens, 3 of them dropped and run again - the same bits
    const int32_t more[4] = {ids[1], ids[2], ids[3], ids[4]};
    CHECK(lfm3_step(f, a2, more, 4, 4, err));
    const std::vector<float> first(lfm3_logits(f), lfm3_logits(f) + (size_t) 4 * V);
    CHECK(lfm3_seq_reach(a2) >= 4 && lfm3_seq_rollback(a2, 3) && lfm3_seq_position(a2) == (int32_t) P + 1);
    CHECK(!lfm3_seq_rollback(a2, LFM3_MAX_BACK + 1));
    CHECK(lfm3_step(f, a2, more + 1, 3, 3, err));
    CHECK(std::memcmp(lfm3_logits(f), first.data() + V, (size_t) 3 * V * 4) == 0);
    std::printf("  4 tokens, 3 rolled back and run again: the same bits\n");

    // the state saved and loaded into another sequence: its next step the same bits; damaged, refused
    CHECK(lfm3_seq_rollback(a2, 4));
    std::vector<uint8_t> saved;
    auto sink = [](void* ctx, const void* bytes, uint64_t n) {
      auto* v = (std::vector<uint8_t>*) ctx;
      v->insert(v->end(), (const uint8_t*) bytes, (const uint8_t*) bytes + n);
      return true;
    };
    CHECK(lfm3_seq_save_to(a2, sink, &saved, err) && saved.size() == lfm3_seq_state_bytes(a2));
    struct Src {
      const std::vector<uint8_t>* v;
      size_t at;
    } src = {&saved, 0};
    auto source = [](void* ctx, void* bytes, uint64_t n) {
      auto* q = (Src*) ctx;
      if (q->at + n > q->v->size()) return false;
      std::memcpy(bytes, q->v->data() + q->at, n), q->at += n;
      return true;
    };
    Lfm3Seq* c = lfm3_seq_create(f, err);
    CHECK(c && lfm3_seq_load_from(c, saved.size(), source, &src, err) && lfm3_seq_position(c) == (int32_t) P);
    CHECK(lfm3_seq_reach(c) == 0 && lfm3_step(f, c, more, 4, 4, err));
    CHECK(std::memcmp(lfm3_logits(f), first.data(), first.size() * 4) == 0);
    saved[saved.size() / 2] ^= 1;
    src.at = 0;
    CHECK(!lfm3_seq_load_from(c, saved.size(), source, &src, err) && lfm3_seq_position(c) == 0);
    std::printf("  saved (%zu KB) and loaded into another sequence: its next step the same bits; damaged, refused\n",
                saved.size() >> 10);
    for (Lfm3Seq* q : {a1, b1, a2, b2, c}) lfm3_seq_destroy(q);
  }

  // the sliding layers' rings wrapping: another forward, steps of 300 (its rings 256 + 300 + LFM3_MAX_BACK rows) -
  // the prompt in its steps the one step's last rows, bit for bit
  {
    const uint32_t S = 300 < T ? 300 : T;
    const Lfm3Options o2 = {S, T, last, LFM3_HEAD_FILE, nullptr, 0, nullptr, 0, 0, 0};
    Lfm3Forward* f2 = lfm3_forward_create(w, &o2, err);
    CHECK(f2);
    Lfm3Seq* r = lfm3_seq_create(f2, err);
    CHECK(r);
    for (uint32_t at = 0; at < T; at += S) {
      const uint32_t n = T - at < S ? T - at : S, keep = T - at - n == 0 ? (n < last ? n : last) : 1;
      CHECK(lfm3_step(f2, r, ids.data() + at, n, keep, err));
      if (at + n == T)
        CHECK(std::memcmp(lfm3_logits(f2), got.data() + (size_t) (last - keep) * V, (size_t) keep * V * 4) == 0);
    }
    std::printf("  in steps of %u (the rings wrap past %u tokens): the last rows the one step's, bit for bit\n", S,
                256 + S + LFM3_MAX_BACK);
    lfm3_seq_destroy(r);
    lfm3_forward_destroy(f2);
  }
  // the head's Q8_0 copy (the engine's default): the same ids, its logits against the file's head's - the rounding's
  // own cost, measured (2026-11-18): see kHeadKl
  {
    const Lfm3Options oq = {T, T, last, LFM3_HEAD_Q8_0, nullptr, 0, nullptr, 0, 0, 0};
    Lfm3Forward* fq = lfm3_forward_create(w, &oq, err);
    CHECK(fq);
    Lfm3Seq* q = lfm3_seq_create(fq, err);
    CHECK(q && lfm3_step(fq, q, ids.data(), T, last, err));
    double ks = 0, km = 0;
    uint32_t agree = 0;
    for (uint32_t r = 0; r < last; r++) {
      const double k = kl(got.data() + (size_t) r * V, lfm3_logits(fq) + (size_t) r * V, V);
      ks += k, km = std::fmax(km, k);
      agree += argmax(got.data() + (size_t) r * V, V) == argmax(lfm3_logits(fq) + (size_t) r * V, V);
    }
    std::printf("  the head's Q8_0 copy against the file's BF16: KL mean %.3g, max %.3g; top-1 the same %u of %u\n",
                ks / last, km, agree, last);
    CHECK(ks / last <= kHeadKlMean && km <= kHeadKlMax && agree + kHeadTop1Miss >= last);
    lfm3_seq_destroy(q);
    lfm3_forward_destroy(fq);
  }
  // a prompt read sparse (LSA's read: OSMOS_TEST_LFM3_LSA, the release's maps), its tokens past position 1,024
  // (Lfm3Options' lsa_sparse_from) each over its read - under LSA_READ_MAX's positions every key, so against the
  // reference within the forward's bounds, its bits not the dense prompt's (the read's path); and in three steps the
  // one step's bits
  {
    std::string maps = getenv("OSMOS_TEST_LFM3_LSA") ? getenv("OSMOS_TEST_LFM3_LSA") : "";
    if (maps.empty()) maps = std::string(home) + "/Models/osmos1.7b/osmos1.7b-lsa-maps.safetensors";
    FILE* mf = std::fopen(maps.c_str(), "rb");
    if (T == 2048 && mf) {
      std::fclose(mf);
      const Lfm3Options os = {T, T, last, LFM3_HEAD_FILE, maps.c_str(), 0, nullptr, 0, 0, 1024};
      Lfm3Forward* fs = lfm3_forward_create(w, &os, err);
      CHECK(fs);
      Lfm3Seq *one = lfm3_seq_create(fs, err), *three = lfm3_seq_create(fs, err);
      CHECK(one && three && lfm3_step(fs, one, ids.data(), T, last, err));
      const std::vector<float> sp(lfm3_logits(fs), lfm3_logits(fs) + (size_t) last * V);
      double ks = 0, km = 0;
      uint32_t agree = 0;
      for (uint32_t r = 0; r < last; r++) {
        const float* p = ref.data() + (ref_rows - last + r) * (size_t) V;
        const double k = kl(p, sp.data() + (size_t) r * V, V);
        ks += k, km = std::fmax(km, k), agree += argmax(p, V) == argmax(sp.data() + (size_t) r * V, V);
      }
      std::printf("  a sparse read (from 1,024; every key under LSA_READ_MAX): against the reference KL mean %.3g, max "
                  "%.3g; top-1 %u of %u\n", ks / last, km, agree, last);
      CHECK(ks / last <= kKlMean && km <= kKlMax && agree + kTop1Miss >= last);
      CHECK(std::memcmp(sp.data(), got.data(), sp.size() * 4) != 0);  // the read's path, not the dense one's
      CHECK(lfm3_step(fs, three, ids.data(), a, 1, err) && lfm3_step(fs, three, ids.data() + a, b, 1, err));
      CHECK(lfm3_step(fs, three, ids.data() + a + b, c, tail, err));
      CHECK(std::memcmp(lfm3_logits(fs), sp.data() + (size_t) (last - tail) * V, (size_t) tail * V * 4) == 0);
      std::printf("  sparse, in read steps of %u, %u and %u: the last %u rows the one step's, bit for bit\n", a, b, c,
                  tail);
      lfm3_seq_destroy(one), lfm3_seq_destroy(three), lfm3_forward_destroy(fs);
    } else {
      if (mf) std::fclose(mf);
      std::printf("  the sparse read skipped: no %s (or not the 2,048-id fixture)\n", maps.c_str());
    }
  }

  // YaRN's stages (OSMOS_TEST_LFM3_YARN: oracle/lfm3_forward.py --kv-q8 --yarn 2 --native 1024 over the same ids):
  // a forward whose stages begin at 1,024 - the ids in one step its stage 1 (x2), against Liquid's YaRN x2; the ids
  // first under the base then past it (the cache turned to stage 1 at the crossing) against the one step; a rollback
  // across the crossing refused
  {
    std::string yarn_path = getenv("OSMOS_TEST_LFM3_YARN") ? getenv("OSMOS_TEST_LFM3_YARN") : "";
    if (yarn_path.empty()) yarn_path = std::string(home) + "/osmos-oracle/lfm3_forward_2048_yarn2_n1024.bin";
    std::map<std::string, Arr> base = fx;
    if (T == 2048 && load(yarn_path.c_str())) {
      const std::vector<float> yref = A("logits");
      const Lfm3Stage two_stages[] = {{0, 1}, {1024, 2}};  // native, then x2 past 1,024
      const Lfm3Options oy = {T, 4 * 1024, last, LFM3_HEAD_FILE, nullptr, 1024, two_stages, 2, 0, 0};
      Lfm3Forward* fy = lfm3_forward_create(w, &oy, err);
      CHECK(fy);
      Lfm3Seq *one = lfm3_seq_create(fy, err), *two = lfm3_seq_create(fy, err);
      CHECK(one && two && lfm3_step(fy, one, ids.data(), T, last, err) && lfm3_seq_stage(one) == 1);
      const std::vector<float> y1(lfm3_logits(fy), lfm3_logits(fy) + (size_t) last * V);
      double ks = 0, km = 0;
      uint32_t agree = 0;
      for (uint32_t r = 0; r < last; r++) {
        const float* p = yref.data() + (ref_rows - last + r) * (size_t) V;
        const double k = kl(p, y1.data() + (size_t) r * V, V);
        ks += k, km = std::fmax(km, k), agree += argmax(p, V) == argmax(y1.data() + (size_t) r * V, V);
      }
      std::printf("  YaRN x2 over 1,024 (stage 1 from the start): against Liquid's KL mean %.3g, max %.3g; top-1 %u "
                  "of %u\n", ks / last, km, agree, last);
      CHECK(ks / last <= kKlMean && km <= kKlMax && agree + kTop1Miss >= last);
      const uint32_t a = 1020, b = 8;  // under the base (plain rope), a step across it (the cache turned), the rest
      CHECK(lfm3_step(fy, two, ids.data(), a, 1, err) && lfm3_seq_stage(two) == 0);
      CHECK(lfm3_step(fy, two, ids.data() + a, b, 1, err) && lfm3_seq_stage(two) == 1);
      CHECK(lfm3_seq_reach(two) == b && !lfm3_seq_rollback(two, b + 1));  // never back across the crossing
      std::printf("  a rollback across the crossing refused (its reach %u: the crossing step's)\n", lfm3_seq_reach(two));
      CHECK(lfm3_step(fy, two, ids.data() + a + b, T - a - b, last, err) && lfm3_seq_stage(two) == 1);
      double kc = 0, kcm = 0;
      uint32_t same = 0;
      for (uint32_t r = 0; r < last; r++) {
        const double k = kl(y1.data() + (size_t) r * V, lfm3_logits(fy) + (size_t) r * V, V);
        kc += k, kcm = std::fmax(kcm, k), same += argmax(y1.data() + (size_t) r * V, V) == argmax(lfm3_logits(fy) + (size_t) r * V, V);
      }
      std::printf("  crossing into stage 1 at %u (the cache turned): against stage 1 from the start KL mean %.3g, max "
                  "%.3g; top-1 %u of %u\n", a + b, kc / last, kcm, same, last);
      CHECK(kc / last <= kStageKlMean && kcm <= kStageKlMax && same + kTop1Miss >= last);
      // static (Lfm3Options' yarn_static_ctx 2,048 over 1,024): every position YaRN x2, the ids in two steps (under the base, then past
      // it) against Liquid's x2 within the forward's bounds - no crossing, no turn
      {
        const Lfm3Options ox = {T, T, last, LFM3_HEAD_FILE, nullptr, 1024, nullptr, 0, 2 * 1024, 0};
        Lfm3Forward* fx = lfm3_forward_create(w, &ox, err);
        CHECK(fx);
        Lfm3Seq* sx = lfm3_seq_create(fx, err);
        CHECK(sx && lfm3_step(fx, sx, ids.data(), 1000, 1, err) && lfm3_step(fx, sx, ids.data() + 1000, T - 1000, last, err));
        double kx = 0, kxm = 0;
        uint32_t ax = 0;
        for (uint32_t r = 0; r < last; r++) {
          const float* p = yref.data() + (ref_rows - last + r) * (size_t) V;
          const double k = kl(p, lfm3_logits(fx) + (size_t) r * V, V);
          kx += k, kxm = std::fmax(kxm, k), ax += argmax(p, V) == argmax(lfm3_logits(fx) + (size_t) r * V, V);
        }
        std::printf("  static YaRN x2 (in steps of 1,000 and %u): against Liquid's KL mean %.3g, max %.3g; top-1 %u "
                    "of %u\n", T - 1000, kx / last, kxm, ax, last);
        CHECK(kx / last <= kKlMean && kxm <= kKlMax && ax + kTop1Miss >= last);
        lfm3_seq_destroy(sx), lfm3_forward_destroy(fx);
      }
      // a table's thresholds its own (not the base's powers): native, x2 past 1,500, x4 past 1,900 - one step to 2,048
      // its stage 2; steps ending at 1,400, 1,700 and 2,048 stages 0, 1, 2; tables out of order, not from 0, a factor
      // 0 or past 64 refused
      {
        const Lfm3Stage three[] = {{0, 1}, {1500, 2}, {1900, 4}};
        const Lfm3Options o3 = {T, T, last, LFM3_HEAD_FILE, nullptr, 1024, three, 3, 0, 0};
        Lfm3Forward* f3 = lfm3_forward_create(w, &o3, err);
        CHECK(f3);
        Lfm3Seq *a3 = lfm3_seq_create(f3, err), *b3 = lfm3_seq_create(f3, err);
        CHECK(a3 && b3 && lfm3_step(f3, a3, ids.data(), T, 1, err) && lfm3_seq_stage(a3) == 2);
        CHECK(lfm3_step(f3, b3, ids.data(), 1400, 1, err) && lfm3_seq_stage(b3) == 0);
        CHECK(lfm3_step(f3, b3, ids.data() + 1400, 300, 1, err) && lfm3_seq_stage(b3) == 1);
        CHECK(lfm3_step(f3, b3, ids.data() + 1700, T - 1700, 1, err) && lfm3_seq_stage(b3) == 2);
        lfm3_seq_destroy(a3), lfm3_seq_destroy(b3), lfm3_forward_destroy(f3);
        const Lfm3Stage desc[] = {{0, 1}, {1900, 2}, {1500, 4}}, late[] = {{10, 1}, {1500, 2}}, zero[] = {{0, 0}},
                        big[] = {{0, 1}, {100, 65}};
        for (const auto& bad : {std::make_pair(desc, 3u), std::make_pair(late, 2u), std::make_pair(zero, 1u),
                                std::make_pair(big, 2u)}) {
          const Lfm3Options ob = {T, T, last, LFM3_HEAD_FILE, nullptr, 1024, bad.first, bad.second, 0, 0};
          CHECK(!lfm3_forward_create(w, &ob, err));
        }
        std::printf("  a table's stages (1,500, 1,900): stages 0, 1, 2 at 1,400, 1,700, 2,048; bad tables refused\n");
      }
      lfm3_seq_destroy(one), lfm3_seq_destroy(two), lfm3_forward_destroy(fy);
    } else {
      std::printf("  YaRN's stages skipped: no %s (or not the 2,048-id fixture)\n", yarn_path.c_str());
    }
    fx = base;
  }
  lfm3_seq_destroy(s2);
  lfm3_seq_destroy(s);
  lfm3_forward_destroy(f);
  lfm3_weights_destroy(w);
  gguf_destroy(g);
  std::printf("[test_lfm3_forward_gpu] ok\n");
  return 0;
}
