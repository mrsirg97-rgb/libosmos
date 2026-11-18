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

// F32 both sides, the order of every sum another: per step ~1e-6 of a value (step 1's pieces), over 42 layers and
// their residuals growing to ~1e-4 of the largest stream; the logits' KL a measure of the whole, below 1e-4 nats
// (a quant's is ~1e-3: this is the same model)
constexpr double kTolStreams = 1e-3, kKl = 1e-4;

struct Taps {
  std::vector<int32_t> rows;  // the kept rows
  std::vector<float> got;     // [layer][row][hc][dim]
  uint32_t layers = 0, hc = 0, dim = 0;
};

static void tap(void* arg, int layer, const float* dev_h, uint32_t n, uint32_t hc, uint32_t dim) {
  Taps* t = (Taps*) arg;
  if (t->rows.empty() || n < (uint32_t) t->rows.back() + 1) return;  // the whole step's: the one-step run's alone
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
  for (uint32_t l = 0; l < taps.layers; l++) {
    double err_l = 0, top = 0;
    for (size_t i = 0; i < per_layer; i++) {
      const double r = ref_taps[l * per_layer + i];
      err_l = std::fmax(err_l, std::fabs(taps.got[l * per_layer + i] - r)), top = std::fmax(top, std::fabs(r));
    }
    worst = std::fmax(worst, err_l / top);
    if (l % 6 == 0 || l + 1 == taps.layers || err_l > kTolStreams * top)
      std::printf("  layer %2u's streams: max error %.3g of %.4g (%.2g)\n", l, err_l, top, err_l / top);
    CHECK(err_l <= kTolStreams * top);
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
  CHECK(kl_max <= kKl && same == last);

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
  hy4_seq_destroy(s2);
  hy4_seq_destroy(s);
  hy4_forward_destroy(f);
  for (int fd : fds) close(fd);
  hy4_weights_destroy(w);
  gguf_destroy(g);
  std::printf("[test_hy4_forward_gpu] ok\n");
  return 0;
}
