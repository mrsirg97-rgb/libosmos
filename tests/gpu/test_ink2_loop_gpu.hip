// Inkling-2-Mini's looped forward (osmos/ink2/forward.h) against upstream llama.cpp's own (oracle/qwen_dump.cpp on its
// converter's GGUF of the release: ~/osmos-oracle/ink2-mini-dump-<n>, its ids, every cache slot's l_out, the logits):
// the ids through one step (every expert in memory: the MoE's tables on the GPU) - each slot's output (the prelude, the core's three passes, the tail) against the
// reference's, the logits (KL against the reference's, its top-1) - then the same ids in uneven steps, the last row's
// logits the same bits; then half the experts streamed (the host's tables), every row's logits the same bits; then a
// full layer's values Q8 (values_q8): the logits' bounds, its steps' bits. the
// reference GGUF (tools/convert/ink2_ref_gguf.py --name Inkling-2-Mini;
// OSMOS_TEST_INK2_MINI its first shard); the 64-id dump, and the 2,048-id one past the sliding window where it is
// there; skipped (77) without the model, the 64-id dump or a GPU
extern "C" {
#include "osmos/ink2/forward.h"
#include "osmos/ink2/weights.h"
#include "osmos/model/gguf.h"
}

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "q4x_ref.h"

// the oracle's own noise (measured 2026-11-20): llama.cpp's GPU run against its CPU run of the same file - BF16
// activations, BF16-rounded experts (the release's NVFP4, ~2e-3 of a weight), an F16 cache, each sum's order its own:
// a slot's output 1.1e-2 .. 6.6e-2 relative L2 apart at 64 ids, 6.5e-3 .. 1.09e-1 at 2,048 (the worst slot 41, pass
// 3's layer 11 - the noise's amplifier); the logits KL mean 3.0e-3 / 1.2e-3, max 3.0e-2 / 1.2e-2, top-1 63 / 64 and
// 64 / 64. osmos (its cache: a full layer's keys Q8, values Q5, a window's f16) against the GPU run, measured: the
// slots' worst 8.1e-2 / 1.30e-1 (slot 31 / 41), the logits KL mean 2.7e-3 / 1.4e-3, max 3.7e-2 / 2.3e-2, top-1 63 / 64
// and 64 / 64 - the oracle's own spread. the bounds those with room (1.5x the slots', 2x the logits'); a wrong term far
// past them (the tail on the wrong pass's input: 0.55 .. 0.76)
constexpr double kTolSlot = 2e-1, kKlMean = 6e-3, kKlMax = 8e-2;
constexpr double kTop1Miss = 0.05;  // of the rows
constexpr const char* kRefModel = "%s/Models/inkling2-mini/osmos/ink2-mini-00001-of-00002.gguf";
constexpr const char* kDump = "%s/osmos-oracle/ink2-mini-dump-%u";
constexpr uint32_t kSplits[] = {1, 7, 13};  // the uneven steps' first lengths: then the rest at once

struct Taps {
  uint32_t n = 0, dim = 0;
  std::vector<std::vector<float>> slot;  // [slot][row][dim]
};

static void tap(void* arg, int slot, const char* what, const float* dev, uint32_t n, uint32_t dim) {
  Taps* t = (Taps*) arg;
  if (std::strcmp(what, "l_out")) return;
  CHECK(slot >= 0 && (size_t) slot < t->slot.size() && n == t->n);
  t->dim = dim;
  t->slot[slot] = host_of(dev, (size_t) n * dim);
}

static std::vector<int32_t> dump_ids(const std::string& dir) {
  std::ifstream in(dir + "/ids.txt");
  std::vector<int32_t> ids;
  for (std::string s; std::getline(in, s, ',');) ids.push_back(std::atoi(s.c_str()));
  return ids;
}

// KL(ref || got) of a row's logits, in double
static double kl(const float* ref, const float* got, uint32_t V) {
  double mr = ref[0], mg = got[0], sr = 0, sg = 0, k = 0;
  for (uint32_t i = 1; i < V; i++) mr = std::max(mr, (double) ref[i]), mg = std::max(mg, (double) got[i]);
  for (uint32_t i = 0; i < V; i++) sr += std::exp(ref[i] - mr), sg += std::exp(got[i] - mg);
  for (uint32_t i = 0; i < V; i++) {
    const double lr = ref[i] - mr - std::log(sr), lg = got[i] - mg - std::log(sg);
    k += std::exp(lr) * (lr - lg);
  }
  return k;
}

static uint32_t argmax(const float* v, uint32_t V) { return (uint32_t) (std::max_element(v, v + V) - v); }

// whole: every expert in memory (room past theirs: the forward takes their all), its MoE's tables on the GPU; else
// half of them (the rest streamed), the tables the host's
static Ink2Forward* forward(const Ink2Weights* w, uint32_t n, bool whole, bool v8 = false) {
  uint64_t experts = 0;
  for (int l = 0; l < w->geo.n_layers; l++)
    experts += w->layer[l].exp_gate->bytes + w->layer[l].exp_up->bytes + w->layer[l].exp_down->bytes;
  const Ink2Options o = {n + 64, n, whole ? 2 * experts : experts / 2, nullptr, nullptr, 0, n, v8};
  char err[OSMOS_ERR] = "";
  Ink2Forward* f = ink2_forward_create(w, &o, err);
  if (!f) std::fprintf(stderr, "  ink2_forward_create: %s\n", err);
  CHECK(f && ink2_forward_whole(f) == whole);
  return f;
}

static void one_dump(const Ink2Weights* w, uint32_t n) {
  char dir[512];
  std::snprintf(dir, sizeof dir, kDump, std::getenv("HOME") ? std::getenv("HOME") : "", n);
  if (access((std::string(dir) + "/ids.txt").c_str(), R_OK)) {
    std::printf("  %u ids: no dump (%s), skipped\n", n, dir);
    return;
  }
  const std::vector<int32_t> ids = dump_ids(dir);
  CHECK(ids.size() == n);
  const Ink2Geometry* g = &w->geo;
  const uint32_t V = (uint32_t) g->n_vocab, C = (uint32_t) g->n_embd;
  Ink2Forward* f = forward(w, n, true);
  Taps t;
  t.n = n, t.slot.resize((size_t) g->n_slots);
  ink2_forward_tap(f, tap, &t);
  char err[OSMOS_ERR] = "";
  Ink2Seq* s = ink2_seq_create(f, err);
  CHECK(s);
  CHECK(ink2_step(f, &s, &n, 1, ids.data(), err) || (std::fprintf(stderr, "  %s\n", err), false));
  const std::vector<float> logits = host_of(ink2_logits(f), (size_t) n * V);

  // every slot's output, the reference's at the same rows
  double worst = 0;
  int worst_at = -1;
  for (int sl = 0; sl < g->n_slots; sl++) {
    const Ref r = ref("l_out-" + std::to_string(sl), dir);
    CHECK(r.ne[0] == C && r.ne[1] == n && t.slot[sl].size() == r.v.size());
    double e = 0, d = 0;
    for (size_t i = 0; i < r.v.size(); i++)
      e += (double) (t.slot[sl][i] - r.v[i]) * (t.slot[sl][i] - r.v[i]), d += (double) r.v[i] * r.v[i];
    const double rel = std::sqrt(e / d);
    if (rel > worst) worst = rel, worst_at = sl;
    if (rel > kTolSlot) std::printf("    slot %d (layer %d): relative L2 %.3e\n", sl, ink2_slot_layer(g, sl), rel);
  }
  std::printf("  %u ids, one step: every slot's output within %.3e relative L2 of the reference's (worst slot %d, "
              "layer %d)\n",
              n, worst, worst_at, ink2_slot_layer(g, worst_at));

  // the logits: the reference's rows the last ones
  const Ref lr = ref("result_output", dir);
  const uint32_t rows = (uint32_t) lr.ne[1], row0 = n - rows;
  CHECK(lr.ne[0] == V && rows <= n);
  double kl_sum = 0, kl_max = 0;
  uint32_t miss = 0;
  for (uint32_t r = 0; r < rows; r++) {
    const float *want = lr.v.data() + (size_t) r * V, *got = logits.data() + (size_t) (row0 + r) * V;
    const double k = kl(want, got, V);
    kl_sum += k, kl_max = std::max(kl_max, k), miss += argmax(want, V) != argmax(got, V);
  }
  std::printf("  the logits (%u rows): KL mean %.3e, max %.3e; top-1 %u / %u\n", rows, kl_sum / rows, kl_max,
              rows - miss, rows);
  CHECK(worst <= kTolSlot && kl_sum / rows <= kKlMean && kl_max <= kKlMax && miss <= kTop1Miss * rows);

  // the same ids in uneven steps: the last row's logits the same bits
  ink2_seq_destroy(s);
  ink2_forward_tap(f, nullptr, nullptr);
  for (uint32_t first : kSplits) {
    Ink2Seq* u = ink2_seq_create(f, err);
    CHECK(u);
    uint32_t at = 0, len = std::min(first, n);
    for (;;) {
      CHECK(ink2_step(f, &u, &len, 1, ids.data() + at, err) || (std::fprintf(stderr, "  %s\n", err), false));
      at += len;
      if (at == n) break;
      len = n - at;
    }
    const std::vector<float> last = host_of(ink2_logits(f) + (size_t) (len - 1) * V, V);  // the last step's last row
    CHECK(std::memcmp(last.data(), logits.data() + (size_t) (n - 1) * V, (size_t) V * 4) == 0);
    ink2_seq_destroy(u);
  }
  std::printf("  the same ids in steps of %u, %u and %u then the rest: the last row's logits the same bits\n",
              kSplits[0], kSplits[1], kSplits[2]);
  ink2_forward_destroy(f);

  // half the experts in memory (the rest streamed): the tables the host's, every row's logits the same bits
  Ink2Forward* hf = forward(w, n, false);
  Ink2Seq* hs = ink2_seq_create(hf, err);
  CHECK(hs && (ink2_step(hf, &hs, &n, 1, ids.data(), err) || (std::fprintf(stderr, "  %s\n", err), false)));
  CHECK(host_of(ink2_logits(hf), (size_t) n * V) == logits);
  std::printf("  half the experts streamed (the host's tables): every row's logits the same bits\n");
  ink2_seq_destroy(hs);
  ink2_forward_destroy(hf);

  // a full layer's values Q8 (values_q8, ink2_fa8): the logits within the same bounds, its uneven steps' last row
  // the one step's bits (a decode's split units over Q8 values)
  Ink2Forward* qf = forward(w, n, true, true);
  Ink2Seq* qs = ink2_seq_create(qf, err);
  CHECK(qs && (ink2_step(qf, &qs, &n, 1, ids.data(), err) || (std::fprintf(stderr, "  %s\n", err), false)));
  const std::vector<float> ql = host_of(ink2_logits(qf), (size_t) n * V);
  ink2_seq_destroy(qs);
  double qsum = 0, qmax = 0;
  uint32_t qmiss = 0;
  for (uint32_t r = 0; r < rows; r++) {
    const float *want = lr.v.data() + (size_t) r * V, *got = ql.data() + (size_t) (row0 + r) * V;
    const double k = kl(want, got, V);
    qsum += k, qmax = std::max(qmax, k), qmiss += argmax(want, V) != argmax(got, V);
  }
  std::printf("  values Q8: the logits (%u rows) KL mean %.3e, max %.3e; top-1 %u / %u\n", rows, qsum / rows, qmax,
              rows - qmiss, rows);
  CHECK(qsum / rows <= kKlMean && qmax <= kKlMax && qmiss <= kTop1Miss * rows);
  CHECK(ql != logits);  // (another cache: other bits)
  {
    Ink2Seq* u = ink2_seq_create(qf, err);
    CHECK(u);
    uint32_t at = 0, len = std::min(kSplits[1], n);
    for (;;) {
      CHECK(ink2_step(qf, &u, &len, 1, ids.data() + at, err) || (std::fprintf(stderr, "  %s\n", err), false));
      at += len;
      if (at == n) break;
      len = n - at;
    }
    CHECK(std::memcmp(host_of(ink2_logits(qf) + (size_t) (len - 1) * V, V).data(), ql.data() + (size_t) (n - 1) * V,
                      (size_t) V * 4) == 0);
    ink2_seq_destroy(u);
  }
  std::printf("  values Q8: in steps of %u then the rest, the last row's logits the same bits\n", kSplits[1]);
  ink2_forward_destroy(qf);
}

int main() {
  std::printf("[test_ink2_loop_gpu] start\n");
  char path[512], gerr[GGUF_ERR] = "", err[OSMOS_ERR] = "";
  if (std::getenv("OSMOS_TEST_INK2_MINI"))
    std::snprintf(path, sizeof path, "%s", std::getenv("OSMOS_TEST_INK2_MINI"));
  else
    std::snprintf(path, sizeof path, kRefModel, std::getenv("HOME") ? std::getenv("HOME") : "");
  char dump64[512];
  std::snprintf(dump64, sizeof dump64, kDump, std::getenv("HOME") ? std::getenv("HOME") : "", 64u);
  int ndev = 0;
  if (access(path, R_OK) || access((std::string(dump64) + "/ids.txt").c_str(), R_OK) ||
      hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_ink2_loop_gpu] skipped: no Inkling-2-Mini reference (%s), no 64-id dump or no GPU\n", path);
    return 77;
  }
  Gguf* m = gguf_create(path, gerr);
  CHECK(m);
  Ink2Weights* w = ink2_weights_create(m, err);
  if (!w) std::fprintf(stderr, "  %s\n", err);
  CHECK(w && w->geo.loop.passes == 3);
  one_dump(w, 64);
  one_dump(w, 2048);
  ink2_weights_destroy(w);
  gguf_destroy(m);
  std::printf("[test_ink2_loop_gpu] ok\n");
  return 0;
}
