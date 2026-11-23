// Hy4-flash's forward (osmos/hy4/forward.h) against the reference's own (oracle/hy4_forward.py: modeling_hy_v4_flash.py's
// decoder layers chained on CPU in F32 on the GGUF's weights, Engram's rows the table's): its ids through one step -
// every layer's streams at the reference's kept rows (the tap) and the last 64 rows' logits (KL against the
// reference's, its top-1); then the bits a token's own: the same ids in uneven steps (a prompt split, its last tokens
// one at a time) the same logits; a step rolled back and run again the same bits. needs the GGUF
// (OSMOS_TEST_HY4: its path) whole in GPU memory and ~/osmos-oracle/hy4_forward.bin (OSMOS_TEST_HY4_REF: another);
// skipped (77) without. run under the memory guard (~95 GB on the GPU).
extern "C" {
#include <fcntl.h>
#include <unistd.h>

#include "osmos/err.h"
#include "osmos/hy4/forward.h"
#include "osmos/model/gguf.h"
}

#include "hy4_fixture.h"

// F32 both sides, the order of every sum another: per step ~1e-6 of a value (step 1's pieces), over 42 layers ~1e-5.
// a short run (oracle/hy4_forward.py --tokens 64: ~/osmos-oracle/hy4_forward_64.bin) stays there: every layer's
// streams within kTolStreams, the logits' KL under kKl. a long one cannot: the router's top-16 margins run down to
// 3e-8 (AngelSlim's Q4_K_M at 2,600 ids: layer 1's row 1821), under the ~1e-5 the two orders of summation give the
// streams, so a token's 16th expert flips in one or the other and the difference spreads by attention - each layer
// itself still the reference's on osmos's own input (hy4_forward.py --check-layer on the decode's streams, 2026-11-19:
// layers 1-6, 11, 21, 31, 41 each within 5e-7 of the largest on every row but that one). there the logits are
// measured instead, against the spread two right runs have - the reference itself in BF16 (--dtype bf16, the release
// as served) on the same ids: KL mean 0.113, max 1.52, top-1 59 of 64 - and the decode's: 0.231, 9.92, 60 of 64 (one
// flip at layer 1 reaches every later row: SPARK-4Bit's flips came at layer 15 and gave 1.4e-3). the bounds that
// measurement's with room; a real fault shows in --check-layer first (a dropped sink, unclamped experts: a layer's
// error 1e-2 and up on most rows)
constexpr double kTolStreams = 1e-3, kKl = 1e-4;
constexpr double kKlMeanLong = 0.5, kKlMaxLong = 20.0;  // (Q8_0's cache, OSMOS_TEST_HY4_KV8: 0.323, 15.7)
constexpr uint32_t kTop1MissLong = 8;
// a read's (hy4_step_read: the matrix cores' f16 inputs) against the F32 reference, bounded by the reference itself run
// in BF16 (hy4_forward.py --dtype bf16, the release as served) on the same ids. Q4_K_M, 2026-11-19: 2,600 ids BF16
// 0.113, 1.52, 59 of 64; the read 0.331, 11.8, 57 of 64 (its f16 inputs flip near-ties in more rows). 64 ids (SPARK-4Bit,
// 2026-11-18) BF16 0.028, max 0.244, 62 of 64; Q4_K_M's read 0.019, 0.385, 62 of 64. the bounds: past both runs' with
// room at either length - a 64-row count moves by near-ties alone - and a real fault (a dropped sink: KL mean 1.0 at
// 64 ids) past the short ones
constexpr double kReadKlMean = 0.05, kReadKlMax = 1.0, kReadKlMeanLong = 0.6, kReadKlMaxLong = 20.0;
constexpr uint32_t kReadTop1Miss = 8, kReadTop1MissLong = 8;
// the Q8_0 cache's read at 64 tokens: the cache alone (the decode's math on it) 0.0553 mean, 1.63 max against the
// reference (2026-11-19, Q3_K's file); its read (the matrix cores' attention and matrices) 0.0557, 1.42
constexpr double kReadKlMean8 = 0.1, kReadKlMax8 = 3.0;

struct Taps {
  std::vector<int32_t> rows;  // the kept rows
  std::vector<float> got;     // [layer][row][hc][dim]
  uint32_t layers = 0, hc = 0, dim = 0;
  bool read = false;          // the read's step (its streams not dumped: --check-layer checks the decode's math)
};

static void tap(void* arg, int layer, const float* dev_h, uint32_t n, uint32_t hc, uint32_t dim) {
  Taps* t = (Taps*) arg;
  if (t->rows.empty() || n < (uint32_t) t->rows.back() + 1) return;  // the whole step's: the one-step run's alone
  // OSMOS_TEST_HY4_DUMP=dir and OSMOS_TEST_HY4_DUMP_LAYERS=a,b,..: those layers' whole streams to dir/h<layer>.f32
  // (oracle/hy4_forward.py --check-layer: the reference's layer on osmos's own input) - the decode's step's alone
  // (the read's f16 inputs would overwrite them, 1e-4 apart by design)
  const char *dir = getenv("OSMOS_TEST_HY4_DUMP"), *which = getenv("OSMOS_TEST_HY4_DUMP_LAYERS");
  char key[16];
  std::snprintf(key, sizeof key, ",%d,", layer);
  if (dir && which && !t->read && strstr((std::string(",") + which + ",").c_str(), key)) {
    std::vector<float> all((size_t) n * hc * dim);
    CHECK(hipMemcpy(all.data(), dev_h, all.size() * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
    char path[512];
    std::snprintf(path, sizeof path, "%s/h%d.f32", dir, layer);
    FILE* f = std::fopen(path, "wb");
    CHECK(f && std::fwrite(all.data(), sizeof(float), all.size(), f) == all.size());
    std::fclose(f);
  }
  t->hc = hc, t->dim = dim;
  const size_t per = (size_t) hc * dim;
  if (t->got.empty()) t->got.resize((size_t) t->layers * t->rows.size() * per);
  for (size_t r = 0; r < t->rows.size(); r++)
    CHECK(hipMemcpy(t->got.data() + ((size_t) layer * t->rows.size() + r) * per, dev_h + (size_t) t->rows[r] * per,
                    per * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
}

static double kl(const float* p_logits, const float* q_logits, uint32_t V) {  // KL(reference || ours), nats
  double mp = -INFINITY, mq = -INFINITY;
  for (uint32_t i = 0; i < V; i++) mp = std::fmax(mp, p_logits[i]), mq = std::fmax(mq, q_logits[i]);
  double zp = 0, zq = 0;
  for (uint32_t i = 0; i < V; i++) zp += exp(p_logits[i] - mp), zq += exp(q_logits[i] - mq);
  double k = 0;
  for (uint32_t i = 0; i < V; i++) {
    const double lp = p_logits[i] - mp - log(zp), lq = q_logits[i] - mq - log(zq);
    k += exp(lp) * (lp - lq);
  }
  return k;
}

static uint32_t argmax(const float* x, uint32_t V) {
  uint32_t b = 0;
  for (uint32_t i = 1; i < V; i++) b = x[i] > x[b] ? i : b;
  return b;
}

struct Stats {
  double kl_mean = 0, kl_max = 0, worst = 0, nll = 0, nll_ref = 0;
  uint32_t same = 0, first_bad = 0;
  std::vector<float> logits;  // the last rows'
};

// the prompt in one step - a decode's math (hy4_step_runs: its prompt's end marked, the last rows kept) or a read's
// (hy4_step_read) - every layer's streams at the reference's kept rows and the last rows' logits against its
static Stats measure(Hy4Forward* f, const std::vector<int32_t>& ids, uint32_t last, bool read) {
  char err[OSMOS_ERR] = "";
  const uint32_t T = (uint32_t) ids.size(), V = (uint32_t) fx["logits"].dims[1];
  Taps taps;
  for (float r : A("rows")) taps.rows.push_back((int32_t) r);
  taps.layers = (uint32_t) fx["taps"].dims[0];
  taps.read = read;
  hy4_forward_tap(f, tap, &taps);
  osmos_clear(err);
  Hy4Seq* s = hy4_seq_create(f, err);
  CHECK(s);
  CHECK(hy4_seq_prompt(s, T - last + 1));
  Hy4Seq* const seqs[1] = {s};
  const uint32_t lens[1] = {T};
  if (!(read ? hy4_step_read : hy4_step_runs)(f, seqs, lens, 1, ids.data(), err)) std::printf("  refused: %s\n", err);
  CHECK(hy4_seq_position(s) == (int32_t) T);
  hy4_forward_tap(f, nullptr, nullptr);
  Stats st;
  st.logits.resize((size_t) last * V);
  for (uint32_t r = 0; r < last; r++) {
    const float* d = hy4_row_logits(f, T - last + r);
    CHECK(d && hipMemcpy(st.logits.data() + (size_t) r * V, d, V * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
  }
  hy4_seq_destroy(s);
  const std::vector<float>& ref_taps = A("taps");
  CHECK(taps.got.size() == ref_taps.size());
  const size_t per_row = (size_t) taps.hc * taps.dim, per_layer = taps.rows.size() * per_row;
  st.first_bad = taps.layers;
  std::printf("  %s:\n", read ? "read (the matrix cores)" : "decode (the matvecs)");
  for (uint32_t l = 0; l < taps.layers; l++) {
    double err_l = 0, top = 0;
    for (size_t i = 0; i < per_layer; i++) {
      const double r = ref_taps[l * per_layer + i];
      err_l = std::fmax(err_l, std::fabs(taps.got[l * per_layer + i] - r)), top = std::fmax(top, std::fabs(r));
    }
    st.worst = std::fmax(st.worst, err_l / top);
    if (err_l > kTolStreams * top && st.first_bad == taps.layers) st.first_bad = l;
    if (l % 6 == 0 || l + 1 == taps.layers) {
      uint32_t off_rows = 0;
      for (size_t r = 0; r < taps.rows.size(); r++) {  // rows whose error passes 1e-4 of the layer's largest
        double e = 0;
        for (size_t i = 0; i < per_row; i++)
          e = std::fmax(e, std::fabs(taps.got[l * per_layer + r * per_row + i] - ref_taps[l * per_layer + r * per_row + i]));
        off_rows += e > 1e-4 * top;
      }
      std::printf("    layer %2u's streams: max error %.3g of %.4g (%.2g); %u of %zu rows past 1e-4\n", l, err_l, top,
                  err_l / top, off_rows, taps.rows.size());
    }
  }
  const std::vector<float>& ref = A("logits");
  const size_t ref_rows = taps.rows.size();
  for (uint32_t r = 0; r < last; r++) {
    const float* p = ref.data() + (ref_rows - last + r) * (size_t) V;  // the reference's last rows: its kept rows' tail
    const double k = kl(p, st.logits.data() + (size_t) r * V, V);
    st.kl_mean += k / last, st.kl_max = std::fmax(st.kl_max, k);
    st.same += argmax(p, V) == argmax(st.logits.data() + (size_t) r * V, V);
  }
  // the text's next id under each (the last rows but one: their next ids in the prompt) - its mean NLL, osmos's and
  // the reference's: a cache's or a math's cost in the model's own terms, not one trajectory's
  const std::vector<float>& ids_f = A("ids");
  double nll = 0, nll_ref = 0;
  for (uint32_t r = 0; r + 1 < last; r++) {
    const int32_t next = (int32_t) ids_f[ids_f.size() - last + r + 1];
    auto lp = [&](const float* x) {
      double m = -INFINITY, z = 0;
      for (uint32_t v = 0; v < V; v++) m = std::fmax(m, x[v]);
      for (uint32_t v = 0; v < V; v++) z += std::exp(x[v] - m);
      return x[next] - m - std::log(z);
    };
    nll -= lp(st.logits.data() + (size_t) r * V) / (last - 1);
    nll_ref -= lp(ref.data() + (ref_rows - last + r) * (size_t) V) / (last - 1);
  }
  st.nll = nll, st.nll_ref = nll_ref;
  std::printf("    the last %u rows' logits: KL mean %.3g, max %.3g nats; top-1 the reference's %u of %u; the text's NLL "
              "%.4f (the reference's %.4f)\n", last, st.kl_mean, st.kl_max, st.same, last, nll, nll_ref);
  return st;
}

int main() {
  const char* home = getenv("HOME") ? getenv("HOME") : "";
  char model_path[512], ref_path[512], err[OSMOS_ERR] = "", gerr[OSMOS_ERR] = "";
  std::snprintf(model_path, sizeof model_path, "%s", getenv("OSMOS_TEST_HY4") ? getenv("OSMOS_TEST_HY4") : "");
  if (!model_path[0]) std::snprintf(model_path, sizeof model_path, "%s/Models/hy4-flash/Hy4-flash-Q3_K-00001-of-00002.gguf", home);
  std::snprintf(ref_path, sizeof ref_path, "%s", getenv("OSMOS_TEST_HY4_REF") ? getenv("OSMOS_TEST_HY4_REF") : "");
  if (!ref_path[0]) std::snprintf(ref_path, sizeof ref_path, "%s/osmos-oracle/hy4_forward.bin", home);
  int devs = 0;
  Gguf* g = nullptr;
  if (hipGetDeviceCount(&devs) != hipSuccess || devs == 0 || !load(ref_path) || !(g = gguf_create(model_path, gerr))) {
    std::printf("[test_hy4_forward_gpu] skipped: no GPU, no %s (oracle/hy4_forward.py) or no model (%s)\n", ref_path,
                gerr);
    return 77;
  }
  osmos_clear(err);
  Hy4Weights* w = hy4_weights_create(g, err);
  if (!w) std::printf("  refused: %s\n", err);
  CHECK(w);
  std::vector<int> fds(g->n_files);
  for (int i = 0; i < g->n_files; i++) CHECK((fds[i] = open(g->paths[i], O_RDONLY)) >= 0);
  const std::vector<float>& fid = A("ids");
  const uint32_t T = (uint32_t) fid.size(), V = (uint32_t) w->geo.n_vocab, last = 64 < T ? 64 : T;
  std::vector<int32_t> ids(T);
  for (uint32_t t = 0; t < T; t++) ids[t] = (int32_t) fid[t];
  const bool kv8 = getenv("OSMOS_TEST_HY4_KV8") != nullptr;  // (the engine's Q8_0 cache; default F32: strict)
  Hy4Options o = {T, T + HY4_MAX_BACK, last, !kv8};
  o.budget = FIT_BUDGET_NOW;
  osmos_clear(err);
  Hy4Forward* f = hy4_forward_create(w, fds.data(), &o, err);
  if (!f) std::printf("  refused: %s\n", err);
  CHECK(f);
  std::printf("[test_hy4_forward_gpu] %u ids, %d layers; the last %u rows' logits\n", T, w->geo.n_layers, last);

  // the prompt in one step of each math: every layer's streams at the kept rows, the last rows' logits
  const Stats dec = measure(f, ids, last, false), rd = measure(f, ids, last, true);
  if (dec.first_bad == (uint32_t) w->geo.n_layers)
    CHECK(dec.kl_max <= kKl && dec.same == last);
  else {  // near-ties flipped (see kKlMeanLong): the logits within the measured bounds
    std::printf("  decode: streams apart from layer %u (the router's near-ties): the logits' bounds instead\n",
                dec.first_bad);
    CHECK(dec.kl_mean <= kKlMeanLong && dec.kl_max <= kKlMaxLong && dec.same + kTop1MissLong >= last);
  }
  // the read's: f16 inputs on the matrix cores, as far from the F32 reference as the release served in BF16 is
  const bool longer = ids.size() > 64;
  const double mean_b = longer ? kReadKlMeanLong : kv8 ? kReadKlMean8 : kReadKlMean;
  const double max_b = longer ? kReadKlMaxLong : kv8 ? kReadKlMax8 : kReadKlMax;
  const uint32_t miss_b = longer ? kReadTop1MissLong : kReadTop1Miss;
  std::printf("  read: KL mean %.3g (bound %.3g), max %.3g (%.3g), top-1 %u of %u (at least %u)\n", rd.kl_mean, mean_b,
              rd.kl_max, max_b, rd.same, last, last - miss_b);
  CHECK(rd.kl_mean <= mean_b && rd.kl_max <= max_b && rd.same + miss_b >= last);
  const std::vector<float>& got = rd.logits;

  // the bits a token's own: the prompt in uneven steps (each a read: hy4_step's prompt math past HY4_MV_TOKENS) -
  // the last step's rows the one step's, bit for bit (a decode's math its own: the rollback's and the state's below)
  osmos_clear(err);
  Hy4Seq* s2 = hy4_seq_create(f, err);
  CHECK(s2);
  const uint32_t a = T / 3, b = T / 3, c = T - a - b, tail = c < last ? c : last;  // three reads, the last's rows
  CHECK(a > HY4_MV_TOKENS && b > HY4_MV_TOKENS && c > HY4_MV_TOKENS);
  osmos_clear(err);
  CHECK(hy4_step(f, s2, ids.data(), a, 1, err) && hy4_step(f, s2, ids.data() + a, b, 1, err));
  osmos_clear(err);
  CHECK(hy4_step(f, s2, ids.data() + a + b, c, tail, err));
  CHECK(std::memcmp(hy4_logits(f), got.data() + (size_t) (last - tail) * V, (size_t) tail * V * sizeof(float)) == 0);
  std::printf("  in read steps of %u, %u and %u: the last %u rows the one step's, bit for bit\n", a, b, c, tail);

  // a decode's split: four tokens one step (each matrix its own launch) and four steps of one (a token's joint
  // matrices: hy4_impl.h's Hy4Joint) - every row the same bits
  {
    const int32_t four[4] = {ids[5], ids[6], ids[7], ids[8]};
    Hy4Seq* const one[1] = {s2};
    const uint32_t len4[1] = {4};
    CHECK(hy4_seq_prompt(s2, 0));
    osmos_clear(err);
    CHECK(hy4_step_runs(f, one, len4, 1, four, err));
    std::vector<float> together((size_t) 4 * V);
    for (uint32_t r = 0; r < 4; r++)
      CHECK(hipMemcpy(together.data() + (size_t) r * V, hy4_row_logits(f, r), V * sizeof(float), hipMemcpyDeviceToHost) ==
            hipSuccess);
    CHECK(hy4_seq_rollback(s2, 4));
    for (uint32_t r = 0; r < 4; r++) {
      osmos_clear(err);
      CHECK(hy4_step(f, s2, four + r, 1, 1, err));
      CHECK(std::memcmp(hy4_logits(f), together.data() + (size_t) r * V, V * sizeof(float)) == 0);
    }
    CHECK(hy4_seq_rollback(s2, 4));
    std::printf("  a decode's four tokens in one step and in four: the same bits\n");
  }

  // a step rolled back and run again: the same bits
  const int32_t more[4] = {ids[1], ids[2], ids[3], ids[4]};
  osmos_clear(err);
  CHECK(hy4_step(f, s2, more, 4, 4, err));
  const std::vector<float> first(hy4_logits(f), hy4_logits(f) + (size_t) 4 * V);
  CHECK(hy4_seq_reach(s2) >= 4 && hy4_seq_rollback(s2, 4) && hy4_seq_position(s2) == (int32_t) T);
  osmos_clear(err);
  CHECK(hy4_step(f, s2, more, 4, 4, err));
  CHECK(std::memcmp(hy4_logits(f), first.data(), first.size() * sizeof(float)) == 0);
  std::printf("  a step rolled back and run again: the same bits\n");

  // the state saved after the prompt and loaded into another sequence: its next step the same bits
  CHECK(hy4_seq_rollback(s2, 4));
  std::vector<uint8_t> saved;
  auto sink = [](void* ctx, const void* bytes, uint64_t n) {
    auto* v = (std::vector<uint8_t>*) ctx;
    v->insert(v->end(), (const uint8_t*) bytes, (const uint8_t*) bytes + n);
    return true;
  };
  osmos_clear(err);
  CHECK(hy4_seq_save_to(s2, sink, &saved, err) && saved.size() == hy4_seq_state_bytes(s2));
  osmos_clear(err);
  Hy4Seq* s3 = hy4_seq_create(f, err);
  CHECK(s3);
  struct Src {
    const std::vector<uint8_t>* v;
    size_t at;
  } src = {&saved, 0};
  auto source = [](void* ctx, void* bytes, uint64_t n) {
    auto* s = (Src*) ctx;
    if (s->at + n > s->v->size()) return false;
    std::memcpy(bytes, s->v->data() + s->at, n), s->at += n;
    return true;
  };
  osmos_clear(err);
  CHECK(hy4_seq_load_from(s3, saved.size(), source, &src, err) && hy4_seq_position(s3) == (int32_t) T);
  osmos_clear(err);
  CHECK(hy4_seq_reach(s3) == 0 && hy4_step(f, s3, more, 4, 4, err));
  CHECK(std::memcmp(hy4_logits(f), first.data(), first.size() * sizeof(float)) == 0);
  saved[saved.size() / 2] ^= 1;  // a damaged string refused (its sum), the sequence then empty
  src.at = 0;
  osmos_clear(err);
  CHECK(!hy4_seq_load_from(s3, saved.size(), source, &src, err) && hy4_seq_position(s3) == 0);
  std::printf("  saved (%zu MB) and loaded into another sequence: its next step the same bits; damaged, refused\n",
              saved.size() >> 20);
  hy4_seq_destroy(s3);
  hy4_seq_destroy(s2);
  hy4_forward_destroy(f);
  for (int fd : fds) close(fd);
  hy4_weights_destroy(w);
  gguf_destroy(g);
  std::printf("[test_hy4_forward_gpu] ok\n");
  return 0;
}
