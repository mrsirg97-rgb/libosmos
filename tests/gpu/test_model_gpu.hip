// the whole model: ds1017's 40 layers and its head, through the pool, the real ids a token at a time, against the
// oracle's full forward (oracle/ref_gguf.py --dump -> npz2bin.py: ~/osmos-oracle/ref<T>): each position's next-
// token distribution - KL(oracle || ours), its argmax - beside maya's (the bar's own logits on the same ids,
// ~/osmos-oracle/logits_a.bin), and each layer's stream drift. then the same ids again, unprobed and timed: the
// logits the same bits (determinism), and the decode's tokens a second. passes when every position's argmax is the
// oracle's wherever maya's is - or a tie: the oracle's two within twice our logits' measured error - and our mean KL
// is no more than maya's.
// the oracle is not schedule-invariant itself: positions 0-7 of its 8-token run and of its 32-token run part from
// layer 0 (torch's F32 blocking changes with the prompt's length) to KL 5.5e-5 .. 0.16 a position (mean 2.2e-2) -
// the floor any comparison against one of its runs stands on; the longer run (ref32) is the one used when present.
// and at every position forward_greedy's pick the first maximum of the logits' row (a second row refused).
// needs a gfx1151, ~/models/ds1017 and the oracle's dump; skipped (77) without them.
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "test_model.h"

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/model/weights.h"
#include "osmos/run/forward.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr uint64_t kBudget = 72ull << 30;
// the lane's F32 experts against maya's (its decode's experts F32 too: KL 1.22e-2 here): one token's route at layer 18
// is a near-tie the two orders tip apart (its stream's drift there 1.0, every other position's the default's), so
// the lane's KL ~1.27e-2 - within this of maya's, its argmax the oracle's or a tie
constexpr double kLaneKlSlack = 1.5;  // the layers' ~6.5 GB and a cache of experts, of this box's 125 GB

struct Arr {
  std::vector<uint64_t> dims;
  std::vector<float> v;
};

static bool load(const std::string& path, Arr* a) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  char m[4];
  uint32_t ty, nd;
  CHECK(std::fread(m, 1, 4, f) == 4 && std::memcmp(m, "SUBA", 4) == 0 && std::fread(&ty, 4, 1, f) == 1);
  CHECK(std::fread(&nd, 4, 1, f) == 1 && nd <= 4);
  a->dims.resize(nd);
  CHECK(std::fread(a->dims.data(), 8, nd, f) == nd);
  uint64_t n = 1;
  for (uint64_t d : a->dims) n *= d;
  a->v.resize(n);
  if (ty == 0) {
    CHECK(std::fread(a->v.data(), 4, n, f) == n);
  } else {
    std::vector<int32_t> iv(n);
    CHECK(std::fread(iv.data(), 4, n, f) == n);
    for (uint64_t i = 0; i < n; i++) a->v[i] = (float) iv[i];
  }
  std::fclose(f);
  return true;
}

// KL(p || q) of the softmaxes of two logit rows, in double
static double kl(const float* p, const float* q, size_t n) {
  double mp = -INFINITY, mq = -INFINITY, sp = 0, sq = 0, k = 0;
  for (size_t i = 0; i < n; i++) mp = std::max(mp, (double) p[i]), mq = std::max(mq, (double) q[i]);
  for (size_t i = 0; i < n; i++) sp += std::exp(p[i] - mp), sq += std::exp(q[i] - mq);
  const double lp = mp + std::log(sp), lq = mq + std::log(sq);
  for (size_t i = 0; i < n; i++) k += std::exp(p[i] - lp) * ((p[i] - lp) - (q[i] - lq));
  return k;
}

static size_t argmax(const float* v, size_t n) { return (size_t) (std::max_element(v, v + n) - v); }

struct Drift {
  const Arr* hs;
  int T;
  std::vector<double> rel;
  bool force = false;
  std::vector<std::vector<double>> by_pos;  // [layer][position]
};

static void probe(const char* step, int layer, int index, void* dev, uint64_t n, bool, void* arg) {
  Drift* d = (Drift*) arg;
  if (std::strcmp(step, "h_out")) return;
  std::vector<float> got(n);
  CHECK(hipMemcpy(got.data(), dev, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  const float* want = &d->hs->v[((size_t) layer * d->T + index) * n];
  double mx = 0, md = 0;
  for (uint64_t i = 0; i < n; i++) mx = std::max(mx, (double) std::fabs(want[i]));
  for (uint64_t i = 0; i < n; i++) md = std::max(md, std::fabs((double) got[i] - want[i]));
  d->rel[layer] = std::max(d->rel[layer], md / mx);
  d->by_pos[layer][index] = md / mx;
  if (d->force) CHECK(hipMemcpy(dev, want, n * 4, hipMemcpyHostToDevice) == hipSuccess);
}

int main() {
  std::printf("[test_model_gpu] start\n");
  const std::string home = std::getenv("HOME") ? std::getenv("HOME") : "", mdir = home + "/models/ds1017/";
  std::string ref = home + "/osmos-oracle/ref32";
  if (access((ref + "/logits.bin").c_str(), R_OK) != 0) ref = home + "/osmos-oracle/ref8";
  // OSMOS_ORACLE_REF / OSMOS_ORACLE_MAYA: another dump and maya's logits on the same ids (another pack's)
  if (const char* r = std::getenv("OSMOS_ORACLE_REF")) ref = r;
  const char* maya_env = std::getenv("OSMOS_ORACLE_MAYA");
  const std::string maya_path = maya_env ? maya_env : home + "/osmos-oracle/logits_a.bin";
  Arr ids, logits, hs, maya;
  int ndev = 0;
  if (!load(ref + "/ids.bin", &ids) || !load(ref + "/logits.bin", &logits) || !load(ref + "/hs.bin", &hs) ||
      hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_model_gpu] skipped: no oracle dump (oracle/ref_gguf.py --dump, npz2bin.py) or no GPU\n");
    return 77;
  }
  const int T = (int) ids.dims[0];
  const size_t V = logits.dims[1];
  {
    FILE* f = std::fopen(maya_path.c_str(), "rb");
    CHECK(f);
    maya.v.resize((size_t) T * V);
    CHECK(std::fread(maya.v.data(), 4, maya.v.size(), f) == maya.v.size());
    std::fclose(f);
  }
  char err[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "", ferr[FORWARD_ERR] = "";
  char path[512];
  const TestModel which = test_model(path, sizeof path);  // the oracle's dump must be of this model (its pack)
  CHECK(which != TestModel::kNone);
  Gguf* g = gguf_create(path, err);
  Gguf* eg = which == TestModel::kGsq ? gguf_create((mdir + "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf").c_str(), err)
                                      : nullptr;  // UD's Engram its own
  CHECK(g && (eg || which == TestModel::kUd));
  Weights* w = weights_create(g, eg, werr);
  if (!w) std::printf("  weights_create: %s\n", werr);
  CHECK(w);
  const auto t0 = std::chrono::steady_clock::now();
  Forward* f = forward_create(w, w->geo.n_layers, 4096, kBudget, ferr);
  if (!f) std::printf("  forward_create: %s\n", ferr);
  CHECK(f);
  std::printf("  the forward made (the layers' weights pinned) in %.1f s\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  const int L = w->geo.n_layers;
  Drift d{&hs, T, std::vector<double>(L, 0), false, std::vector<std::vector<double>>(L, std::vector<double>(T, 0))};
  Drift forced = d;
  forced.force = true;
  std::vector<std::vector<float>> got(3, std::vector<float>((size_t) T * V));
  std::vector<double> ms;
  for (int run = 0; run < 3; run++) {  // free and probed, unprobed and timed, forced (each layer from the oracle's)
    forward_probe(f, run == 1 ? nullptr : probe, run == 2 ? &forced : &d);
    Seq* s = seq_create(f, ferr);
    CHECK(s);
    for (int p = 0; p < T; p++) {
      const auto a = std::chrono::steady_clock::now();
      if (!forward_step(f, s, (int32_t) ids.v[p], ferr)) std::printf("  step %d: %s\n", p, ferr);
      CHECK(hipDeviceSynchronize() == hipSuccess);
      using Ms = std::chrono::duration<double, std::milli>;
      if (run == 1) ms.push_back(Ms(std::chrono::steady_clock::now() - a).count());
      CHECK(hipMemcpy(&got[run][(size_t) p * V], forward_logits(f), V * 4, hipMemcpyDeviceToHost) == hipSuccess);
      // forward_greedy: the GPU's pick the first maximum of the row; a row the step did not compute refused
      int32_t pick = -1, first = 0;
      const float* row = &got[run][(size_t) p * V];
      for (uint32_t i = 1; i < (uint32_t) V; i++)
        if (row[i] > row[first]) first = (int32_t) i;
      CHECK(forward_greedy(f, 0, 1, &pick, ferr) && pick == first && !forward_greedy(f, 0, 2, &pick, ferr));
    }
    seq_destroy(s);
  }
  std::printf("  %d positions; the next token's distribution against the oracle's (KL(oracle || x)), and its argmax:\n",
              T);
  // our logits' measured error: the rms of ours - the oracle's over each position's kTieTop leading tokens. a miss is
  // a tie when the oracle's two leading tokens are within twice it (the router's tie rule): an order of summing tips it
  constexpr int kTieTop = 8;
  double err2 = 0;
  for (int p = 0; p < T; p++) {
    const float* o = &logits.v[(size_t) p * V];
    std::vector<size_t> top(V);
    for (size_t i = 0; i < V; i++) top[i] = i;
    std::partial_sort(top.begin(), top.begin() + kTieTop, top.end(), [&](size_t a, size_t b) { return o[a] > o[b]; });
    for (int k = 0; k < kTieTop; k++) {
      const double e = (double) got[0][(size_t) p * V + top[k]] - o[top[k]];
      err2 += e * e;
    }
  }
  const double tie = 2 * std::sqrt(err2 / ((double) T * kTieTop));
  double ours_kl = 0, maya_kl = 0;
  int ours_top = 0, maya_top = 0, ok_top = 1, ties = 0;
  for (int p = 0; p < T; p++) {
    const float* o = &logits.v[(size_t) p * V];
    const double ko = kl(o, &got[0][(size_t) p * V], V), km = kl(o, &maya.v[(size_t) p * V], V);
    const size_t ao = argmax(o, V), au = argmax(&got[0][(size_t) p * V], V), am = argmax(&maya.v[(size_t) p * V], V);
    ours_kl += ko, maya_kl += km, ours_top += au == ao, maya_top += am == ao;
    const bool is_tie = au != ao && (double) o[ao] - o[au] <= tie;
    ties += is_tie;
    ok_top &= au == ao || am != ao || is_tie;
    std::printf("    %2d  ours KL %.2e %s   maya KL %.2e %s\n", p, ko,
                au == ao ? "argmax =" : is_tie ? "argmax T" : "argmax X", km, am == ao ? "argmax =" : "argmax X");
  }
  std::printf("  mean KL: ours %.2e, maya %.2e; argmax the oracle's: ours %d of %d, maya %d\n", ours_kl / T,
              maya_kl / T, ours_top, T, maya_top);
  std::printf("  our logits' error (rms, the oracle's %d leading tokens) %.3f; misses that are ties: %d (T: the two "
              "within %.3f)\n", kTieTop, tie / 2, ties, tie);
  std::printf("  the stream's drift from the oracle's, worst position (max |diff| / max |ref|), by layer:\n   ");
  for (int l = 0; l < L; l++) std::printf(" %d:%.1e%s", l, d.rel[l], l % 8 == 7 ? "\n   " : "");
  std::printf("\n  forced (each layer on the oracle's stream in), its own error, by layer:\n   ");
  for (int l = 0; l < L; l++) std::printf(" %d:%.1e%s", l, forced.rel[l], l % 8 == 7 ? "\n   " : "");
  std::printf("\n  free, by position (the worst layer):");
  for (int p = 0; p < T; p++) {
    double m = 0;
    int at = 0;
    for (int l = 0; l < L; l++)
      if (d.by_pos[l][p] > m) m = d.by_pos[l][p], at = l;
    std::printf(" %d:%.1e@%d", p, m, at);
  }
  std::vector<double> warm(ms.begin() + 1, ms.end());
  std::sort(warm.begin(), warm.end());
  std::printf("\n  unprobed: logits %s the probed run's; a token %.1f ms cold (the first), then p50 %.1f ms, "
              "max %.1f ms (%.2f tok/s)\n", got[0] == got[1] ? "bit for bit" : "NOT", ms[0], warm[warm.size() / 2],
              warm.back(), 1e3 / warm[warm.size() / 2]);
  uint64_t print = 1469598103934665603ull;  // FNV-1a over every position's logits: a change that keeps them keeps it
  const uint8_t* bytes = (const uint8_t*) got[1].data();
  for (size_t i = 0; i < got[1].size() * 4; i++) print = (print ^ bytes[i]) * 1099511628211ull;
  std::printf("  the logits, every position: fingerprint %016llx\n", (unsigned long long) print);
  CHECK(got[0] == got[1] && ok_top && ours_kl <= maya_kl);
  // the router's picks, counted: each run's every position, layer and slot; then a forward whose resident experts
  // are the least-picked first (another set of them) - the same bits, which experts are resident changing none
  const size_t nu = (size_t) L * w->geo.n_experts;
  std::vector<uint64_t> least(nu);
  uint64_t picks = 0, most = 0;
  for (size_t i = 0; i < nu; i++) picks += forward_picks(f)[i], most = std::max(most, forward_picks(f)[i]);
  CHECK(picks == 3ull * T * L * w->geo.n_used);
  for (size_t i = 0; i < nu; i++) least[i] = most - forward_picks(f)[i];
  forward_destroy(f);
  f = forward_create_ranked(w, L, 4096, kBudget, least.data(), ferr);
  if (!f) std::printf("  forward_create_ranked: %s\n", ferr);
  CHECK(f);
  std::vector<float> ranked((size_t) T * V);
  Seq* rs = seq_create(f, ferr);
  CHECK(rs);
  for (int p = 0; p < T; p++) {
    CHECK(forward_step(f, rs, (int32_t) ids.v[p], ferr));
    CHECK(hipMemcpy(&ranked[(size_t) p * V], forward_logits(f), V * 4, hipMemcpyDeviceToHost) == hipSuccess);
  }
  seq_destroy(rs);
  CHECK(ranked == got[1]);
  CHECK(forward_swaps(f) > 0);
  // forward_lane (0 CPU threads: its experts all on the GPU, in the matvec's order): two runs the same bits, its
  // logits within the oracle's reach (KL within kLaneKlSlack of maya's, argmax the oracle's or a tie)
  CHECK(forward_lane(f, 0, NULL, ferr));
  std::vector<std::vector<float>> lane(2, std::vector<float>((size_t) T * V));
  for (int run = 0; run < 2; run++) {
    Seq* ls = seq_create(f, ferr);
    CHECK(ls);
    for (int p = 0; p < T; p++) {
      CHECK(forward_step(f, ls, (int32_t) ids.v[p], ferr));
      CHECK(hipMemcpy(&lane[run][(size_t) p * V], forward_logits(f), V * 4, hipMemcpyDeviceToHost) == hipSuccess);
    }
    seq_destroy(ls);
  }
  CHECK(lane[0] == lane[1]);
  double lane_kl = 0;
  int lane_top = 0, lane_ok = 1;
  for (int p = 0; p < T; p++) {
    const float* o = &logits.v[(size_t) p * V];
    const size_t ao = argmax(o, V), al = argmax(&lane[0][(size_t) p * V], V), am = argmax(&maya.v[(size_t) p * V], V);
    lane_kl += kl(o, &lane[0][(size_t) p * V], V), lane_top += al == ao;
    lane_ok &= al == ao || am != ao || (double) o[ao] - o[al] <= tie;
  }
  CHECK(lane_ok && lane_kl <= kLaneKlSlack * maya_kl);
  std::printf("  the lane (experts in the matvec's order): two runs bit for bit; mean KL %.2e (maya %.2e), argmax the "
              "oracle's %d of %d\n", lane_kl / T, maya_kl / T, lane_top, T);
  std::printf("  picks counted: %llu (3 runs x %d positions x %d layers x %d); the least-picked resident first, %llu "
              "experts swapped in as it went: logits bit for bit\n", (unsigned long long) picks, T, L, w->geo.n_used,
              (unsigned long long) forward_swaps(f));
  forward_destroy(f);
  weights_destroy(w);
  gguf_destroy(g);
  gguf_destroy(eg);
  std::printf("[test_model_gpu] ok\n");
  return 0;
}
