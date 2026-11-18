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

#include "osmos/hy4/forward.h"
#include "osmos/model/gguf.h"
}

#include "hy4_fixture.h"

// F32 both sides, the order of every sum another: per step ~1e-6 of a value (step 1's pieces), over 42 layers ~1e-5.
// a short run (oracle/hy4_forward.py --tokens 64: ~/osmos-oracle/hy4_forward_64.bin) stays there: every layer's
// streams within kTolStreams, the logits' KL under kKl. a long one cannot: the router's top-16 margins run down to
// 3e-8 (2,600 ids: layers 15, 16, 23, 24), under the ~1e-5 the two orders of summation give the streams, so a token's
// 16th expert flips in one or the other and the difference spreads by attention - each layer itself still the
// reference's on osmos's own input (hy4_forward.py --check-layer: within 4.5e-7 of the largest, 2026-11-18). there
// the logits are measured instead: with every layer so checked, KL mean 1.44e-3, max 0.0176, top-1 64 of 64 - the
// bounds below that measurement's, rounded up (a real fault - step 1's dropped sink 1.0, unclamped experts 1.9 - is
// orders past them)
constexpr double kTolStreams = 1e-3, kKl = 1e-4;
constexpr double kKlMeanLong = 2e-3, kKlMaxLong = 2e-2;
constexpr uint32_t kTop1MissLong = 1;

struct Taps {
  std::vector<int32_t> rows;  // the kept rows
  std::vector<float> got;     // [layer][row][hc][dim]
  uint32_t layers = 0, hc = 0, dim = 0;
};

static void tap(void* arg, int layer, const float* dev_h, uint32_t n, uint32_t hc, uint32_t dim) {
  Taps* t = (Taps*) arg;
  if (t->rows.empty() || n < (uint32_t) t->rows.back() + 1) return;  // the whole step's: the one-step run's alone
  // OSMOS_TEST_HY4_DUMP=dir and OSMOS_TEST_HY4_DUMP_LAYERS=a,b,..: those layers' whole streams to dir/h<layer>.f32
  // (oracle/hy4_forward.py --check-layer: the reference's layer on osmos's own input)
  const char *dir = getenv("OSMOS_TEST_HY4_DUMP"), *which = getenv("OSMOS_TEST_HY4_DUMP_LAYERS");
  char key[16];
  std::snprintf(key, sizeof key, ",%d,", layer);
  if (dir && which && strstr((std::string(",") + which + ",").c_str(), key)) {
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

int main() {
  const char* home = getenv("HOME") ? getenv("HOME") : "";
  char model_path[512], ref_path[512], err[HY4_ERR] = "", gerr[GGUF_ERR] = "";
  std::snprintf(model_path, sizeof model_path, "%s", getenv("OSMOS_TEST_HY4") ? getenv("OSMOS_TEST_HY4") : "");
  if (!model_path[0]) std::snprintf(model_path, sizeof model_path, "%s/Models/hy4-flash/Hy4-flash-SPARK-4Bit.gguf", home);
  std::snprintf(ref_path, sizeof ref_path, "%s", getenv("OSMOS_TEST_HY4_REF") ? getenv("OSMOS_TEST_HY4_REF") : "");
  if (!ref_path[0]) std::snprintf(ref_path, sizeof ref_path, "%s/osmos-oracle/hy4_forward.bin", home);
  int devs = 0;
  Gguf* g = nullptr;
  if (hipGetDeviceCount(&devs) != hipSuccess || devs == 0 || !load(ref_path) || !(g = gguf_create(model_path, gerr))) {
    std::printf("[test_hy4_forward_gpu] skipped: no GPU, no %s (oracle/hy4_forward.py) or no model (%s)\n", ref_path,
                gerr);
    return 77;
  }
  Hy4Weights* w = hy4_weights_create(g, err);
  if (!w) std::printf("  refused: %s\n", err);
  CHECK(w);
  std::vector<int> fds(g->n_files);
  for (int i = 0; i < g->n_files; i++) CHECK((fds[i] = open(g->paths[i], O_RDONLY)) >= 0);
  const std::vector<float>& fid = A("ids");
  const uint32_t T = (uint32_t) fid.size(), V = (uint32_t) w->geo.n_vocab, last = 64 < T ? 64 : T;
  std::vector<int32_t> ids(T);
  for (uint32_t t = 0; t < T; t++) ids[t] = (int32_t) fid[t];
  const Hy4Options o = {T, T + HY4_MAX_BACK, last};
  Hy4Forward* f = hy4_forward_create(w, fds.data(), &o, err);
  if (!f) std::printf("  refused: %s\n", err);
  CHECK(f);
  std::printf("[test_hy4_forward_gpu] %u ids, %d layers; the last %u rows' logits\n", T, w->geo.n_layers, last);

  // one step: every layer's streams at the kept rows, the last rows' logits
  Taps taps;
  for (float r : A("rows")) taps.rows.push_back((int32_t) r);
  taps.layers = (uint32_t) w->geo.n_layers;
  hy4_forward_tap(f, tap, &taps);
  Hy4Seq* s = hy4_seq_create(f, err);
  CHECK(s);
  if (!hy4_step(f, s, ids.data(), T, last, err)) std::printf("  step refused: %s\n", err);
  CHECK(hy4_seq_position(s) == (int32_t) T);
  hy4_forward_tap(f, nullptr, nullptr);
  const std::vector<float>& ref_taps = A("taps");
  CHECK(taps.got.size() == ref_taps.size());
  const size_t per_layer = taps.rows.size() * taps.hc * taps.dim;
  double worst = 0;
  const size_t per_row = (size_t) taps.hc * taps.dim;
  uint32_t first_bad = taps.layers;
  for (uint32_t l = 0; l < taps.layers; l++) {
    double err_l = 0, top = 0;
    for (size_t i = 0; i < per_layer; i++) {
      const double r = ref_taps[l * per_layer + i];
      err_l = std::fmax(err_l, std::fabs(taps.got[l * per_layer + i] - r)), top = std::fmax(top, std::fabs(r));
    }
    worst = std::fmax(worst, err_l / top);
    if (err_l > kTolStreams * top && first_bad == taps.layers) first_bad = l;
    if (l % 6 == 0 || l + 1 == taps.layers || err_l > kTolStreams * top || (l > 0 && l < 20)) {
      uint32_t off_rows = 0;
      int32_t first_row = -1;
      for (size_t r = 0; r < taps.rows.size(); r++) {  // rows whose error passes 1e-4 of the layer's largest
        double e = 0;
        for (size_t i = 0; i < per_row; i++)
          e = std::fmax(e, std::fabs(taps.got[l * per_layer + r * per_row + i] - ref_taps[l * per_layer + r * per_row + i]));
        if (e > 1e-4 * top) off_rows++, first_row = first_row < 0 ? taps.rows[r] : first_row;
      }
      std::printf("  layer %2u's streams: max error %.3g of %.4g (%.2g); %u of %zu rows past 1e-4 (first: row %d)\n", l,
                  err_l, top, err_l / top, off_rows, taps.rows.size(), first_row);
    }
  }
  const std::vector<float> got(hy4_logits(f), hy4_logits(f) + (size_t) last * V);
  const std::vector<float>& ref = A("logits");
  const size_t ref_rows = taps.rows.size();
  double kl_sum = 0, kl_max = 0;
  uint32_t same = 0;
  for (uint32_t r = 0; r < last; r++) {
    const float* p = ref.data() + (ref_rows - last + r) * (size_t) V;  // the reference's last rows: its kept rows' tail
    const double k = kl(p, got.data() + (size_t) r * V, V);
    kl_sum += k, kl_max = std::fmax(kl_max, k), same += argmax(p, V) == argmax(got.data() + (size_t) r * V, V);
  }
  std::printf("  the last %u rows' logits: KL mean %.3g, max %.3g nats; top-1 the reference's %u of %u (streams' worst "
              "%.2g)\n", last, kl_sum / last, kl_max, same, last, worst);
  if (first_bad == taps.layers)
    CHECK(kl_max <= kKl && same == last);
  else {  // near-ties flipped (see kKlMeanLong): the logits within the measured bounds
    std::printf("  streams apart from layer %u (the router's near-ties): the logits' bounds instead\n", first_bad);
    CHECK(kl_sum / last <= kKlMeanLong && kl_max <= kKlMaxLong && same + kTop1MissLong >= last);
  }

  // the bits a token's own: the prompt in uneven steps, its last tokens one at a time - the same logits
  Hy4Seq* s2 = hy4_seq_create(f, err);
  CHECK(s2);
  const uint32_t tail = last < T / 4 ? last : T / 4, a = (T - tail) / 3 + 1, b = T - tail - a;  // tail of the kept rows
  CHECK(hy4_step(f, s2, ids.data(), a, 1, err) && hy4_step(f, s2, ids.data() + a, b, 1, err));
  for (uint32_t r = 0; r < tail; r++) {
    CHECK(hy4_step(f, s2, ids.data() + a + b + r, 1, 1, err));
    CHECK(std::memcmp(hy4_logits(f), got.data() + (size_t) (last - tail + r) * V, (size_t) V * sizeof(float)) == 0);
  }
  std::printf("  in steps of %u, %u and %u ones: the same logits, bit for bit\n", a, b, tail);

  // a step rolled back and run again: the same bits
  const int32_t more[4] = {ids[1], ids[2], ids[3], ids[4]};
  CHECK(hy4_step(f, s2, more, 4, 4, err));
  const std::vector<float> first(hy4_logits(f), hy4_logits(f) + (size_t) 4 * V);
  CHECK(hy4_seq_reach(s2) >= 4 && hy4_seq_rollback(s2, 4) && hy4_seq_position(s2) == (int32_t) T);
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
  CHECK(hy4_seq_save_to(s2, sink, &saved, err) && saved.size() == hy4_seq_state_bytes(s2));
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
  CHECK(hy4_seq_load_from(s3, saved.size(), source, &src, err) && hy4_seq_position(s3) == (int32_t) T);
  CHECK(hy4_seq_reach(s3) == 0 && hy4_step(f, s3, more, 4, 4, err));
  CHECK(std::memcmp(hy4_logits(f), first.data(), first.size() * sizeof(float)) == 0);
  saved[saved.size() / 2] ^= 1;  // a damaged string refused (its sum), the sequence then empty
  src.at = 0;
  CHECK(!hy4_seq_load_from(s3, saved.size(), source, &src, err) && hy4_seq_position(s3) == 0);
  std::printf("  saved (%zu MB) and loaded into another sequence: its next step the same bits; damaged, refused\n",
              saved.size() >> 20);
  hy4_seq_destroy(s3);
  hy4_seq_destroy(s2);
  hy4_seq_destroy(s);
  hy4_forward_destroy(f);
  for (int fd : fds) close(fd);
  hy4_weights_destroy(w);
  gguf_destroy(g);
  std::printf("[test_hy4_forward_gpu] ok\n");
  return 0;
}
