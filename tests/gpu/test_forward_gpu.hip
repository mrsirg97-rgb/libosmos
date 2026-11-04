// the forward (src/run) over ds1017's first layers, as many as the oracle's dump holds ($OSMOS_LAYERS, a
// layer_dump.py -> npz2bin.py directory; ~/osmos-oracle/layers3 by default: 0 the window alone, 1 with Engram,
// 2 the first KV source - the compressor at ratio 2, the indexer), its ids a token at a time, every step it names
// against the oracle's (oracle/layer_dump.py -> npz2bin.py, ref.py's prompt path at that position), twice:
//   free: the forward on its own. the routes and the compressed rows attended exactly; the other steps reported (a
//     value at an FP8 / FP4 rounding edge flips a whole step - 1/16 of itself, or 4 -> 6 - and the difference then
//     carries on, so a chained run's differences are not each step's)
//   forced: each step's output, once compared, overwritten with the oracle's, so each step runs on the oracle's
//     input: every route and compressed row the oracle's - but for a tie: two experts whose scores, in double from
//     the oracle's own input, are closer than twice our router's error there (measured, the same way), where F32's
//     summation order decides and the oracle's pick is one roll of it (its weights and the FFN's output at that
//     position are then a different expert's, not compared); every float step within kMaxRel of its largest value;
//     the rounded rows (kv, ckv) not the oracle's in at most kMaxFlips of their values (each a step at an edge)
// the stream out's fingerprint the same with a probe and without (determinism; a probe that only reads changes
// nothing).
// needs a gfx1151, ~/models/ds1017 and the dump; skipped (77) without them.
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <set>
#include <string>
#include <vector>

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/model/weights.h"
#include "osmos/gpu/mm.h"
#include "osmos/run/forward.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

// a float step's worst element against its largest value: the matmul's f16 activations and weights cost 2-3e-4 rms
// (test_mm_gpu) and a step's worst of its thousands of elements ~5 times that; 3e-3 is that with room. a bug is 1e-1
// and up (positions one apart: 1e0)
constexpr double kMaxRel = 3e-3;
// a rounded row's values a step from the oracle's: one sits at an edge when the step before it moved it past one - its
// error (the matmul's f16 activations and weights: 2-3e-4 relative, test_mm_gpu) against E4M3's step (1/16 of a
// value at least): 2 x 3e-4 / (1 / 16) ~ 1% of them; 2% is that with room
constexpr double kMaxFlips = 2e-2;
// a router key (sqrt(softplus(logit)) + bias) is four F32 roundings - expf, log1pf, sqrtf, the add - each within an
// ulp: ours and the oracle's within 4 ulps each of the exact key, so two keys closer than 8 are not ordered in F32
constexpr double kKeyUlps = 8;
constexpr uint64_t kBudget = 72ull << 30;  // up to 40 layers' weights (~6.5 GB) and a cache of their experts

static std::string g_dir;

struct Arr {
  std::vector<uint64_t> dims;
  std::vector<float> v;
  size_t row_len() const { return v.size() / dims[0]; }
};

static bool load(const std::string& name, Arr* a) {
  FILE* f = std::fopen((g_dir + "/" + name + ".bin").c_str(), "rb");
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

struct Diff {
  double rel = 0;
  int seen = 0, exact_bad = 0;
  uint64_t values = 0, flips = 0;
};

// the router in double from the oracle's input: a layer's weights (BF16) and bias, read from the model once
struct Router {
  std::vector<double> w, bias;
};

struct Probe {
  std::map<std::string, Arr> ref;
  std::map<std::string, Diff> d;
  int T = 0;
  bool force = false;
  const Gguf* g = nullptr;
  const Weights* wts = nullptr;
  std::map<int, Router> router;
  std::vector<double> ours;      // our scores (+ bias), the position's, from its router_logits
  std::set<std::pair<int, int>> ties;
  int n_ties = 0;
  const Arr* get(const std::string& k) {
    auto it = ref.find(k);
    if (it != ref.end()) return &it->second;
    Arr a;
    if (!load(k, &a)) return nullptr;
    return &(ref[k] = a);
  }
};

static std::vector<uint8_t> bytes_of(const Gguf* g, const GgufTensor* t) {
  std::vector<uint8_t> b(t->bytes);
  const int fd = open(g->paths[t->file], O_RDONLY);
  CHECK(fd >= 0 && pread(fd, b.data(), b.size(), (off_t) t->offset) == (ssize_t) b.size());
  close(fd);
  return b;
}

static const Router& router_of(Probe* P, int layer) {
  auto it = P->router.find(layer);
  if (it != P->router.end()) return it->second;
  const LayerWeights& L = P->wts->layer[layer];
  Router r;
  const std::vector<uint8_t> w = bytes_of(P->g, L.router), b = bytes_of(P->g, L.router_bias);
  r.w.resize(w.size() / 2);
  for (size_t i = 0; i < r.w.size(); i++) {
    uint32_t u = (uint32_t) (w[2 * i] | w[2 * i + 1] << 8) << 16;
    float x;
    std::memcpy(&x, &u, 4);
    r.w[i] = x;
  }
  r.bias.resize(b.size() / 4);
  for (size_t i = 0; i < r.bias.size(); i++) r.bias[i] = ((const float*) b.data())[i];
  return P->router[layer] = r;
}

static double score(double logit) { return std::sqrt(logit > 20 ? logit : std::log1p(std::exp(logit))); }

// a forced route that is not the oracle's: a tie when every swapped expert's key is, in double from the oracle's
// input, within twice our router's logits' error (ours against double's, this position) and kKeyUlps of the key's
// F32 rounding of the expert it displaced
static bool is_tie(Probe* P, int layer, int index, const std::vector<float>& got, const float* want) {
  const Router& r = router_of(P, layer);
  const Arr* x = P->get(std::to_string(layer) + ".ffn_x");
  const size_t E = r.bias.size(), D = r.w.size() / E;
  std::vector<double> d(E);
  double err = 0;
  for (size_t e = 0; e < E; e++) {
    double s = 0;
    for (size_t c = 0; c < D; c++) s += r.w[e * D + c] * x->v[(size_t) index * D + c];
    d[e] = score(s) + r.bias[e];
    err = std::max(err, std::fabs(P->ours[e] - d[e]));
  }
  std::set<int> a(got.begin(), got.end()), b(want, want + got.size());
  bool all = true;
  for (int e : a)
    for (int o : b)
      if (!a.count(o) && !b.count(e)) {
        const double big = std::max(std::fabs(d[e]), std::fabs(d[o]));
        const double ulp = std::ldexp(1.0, std::ilogb(big) - 23), margin = std::fabs(d[e] - d[o]);
        const bool t = margin <= 2 * err + kKeyUlps * ulp;
        std::printf("    layer %d position %d: expert %d for the oracle's %d, %.1f F32 ulps apart in double: %s\n",
                    layer, index, e, o, margin / ulp, t ? "a tie" : "NOT a tie");
        all &= t;
      }
  return all;
}

static void probe(const char* step, int layer, int index, void* dev, uint64_t n, bool ints, void* arg) {
  Probe* P = (Probe*) arg;
  const std::string s = step;
  if (s == "router_logits") {  // our scores + bias, for a tie's measure
    std::vector<float> l(n);
    CHECK(hipMemcpy(l.data(), dev, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
    const Router& r = router_of(P, layer);
    P->ours.resize(n);
    for (uint64_t e = 0; e < n; e++) P->ours[e] = score(l[e]) + r.bias[e];
    return;
  }
  if ((s == "route_w" || s == "moe_out") && P->ties.count({layer, index})) return;  // another expert's: not compared
  const std::string key = std::to_string(layer) + "." + (s == "ckv" || s == "sel" ? "kv" : s);
  const Arr* r = P->get(s == "sel" ? std::to_string(layer) + ".idx" : key);
  if (!r) return;  // a step the oracle did not record
  std::vector<float> got(n);
  if (ints) {
    std::vector<int32_t> iv(n);
    if (n) CHECK(hipMemcpy(iv.data(), dev, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
    for (uint64_t i = 0; i < n; i++) got[i] = (float) iv[i];
  } else if (n) {
    CHECK(hipMemcpy(got.data(), dev, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  }
  Diff& d = P->d[std::to_string(layer) + "." + s];
  d.seen++;
  if (s == "sel" || s == "route_idx") {  // exact: the oracle's compressed part of idx is row + T, -1 for none
    std::vector<float> want;
    const float* row = &r->v[(size_t) index * r->row_len()];
    if (s == "sel") {
      const size_t win = std::min<size_t>(P->T, 128);
      for (size_t i = win; i < r->row_len(); i++)
        if (row[i] >= 0) want.push_back(row[i] - P->T);
      std::sort(want.begin(), want.end());
    } else {
      want.assign(row, row + n);
    }
    if (want != got && s == "route_idx" && P->force && is_tie(P, layer, index, got, row)) {
      P->ties.insert({layer, index});
      P->n_ties++;
      return;
    }
    d.exact_bad += want != got;
    return;
  }
  const size_t at = s == "ckv" ? (size_t) (P->T + index) : (size_t) index;
  CHECK(r->row_len() == n && at < r->dims[0]);
  const float* want = &r->v[at * n];
  double mx = 0, md = 0;
  for (uint64_t i = 0; i < n; i++) mx = std::max(mx, (double) std::fabs(want[i]));
  for (uint64_t i = 0; i < n; i++) md = std::max(md, std::fabs((double) got[i] - want[i]));
  for (uint64_t i = 0; i < n; i++) d.flips += got[i] != want[i];
  d.values += n;
  d.rel = std::max(d.rel, mx > 0 ? md / mx : md);
  if (P->force) CHECK(hipMemcpy(dev, want, n * 4, hipMemcpyHostToDevice) == hipSuccess);
}

int main() {
  std::printf("[test_forward_gpu] start\n");
  const char* home = std::getenv("HOME");
  const char* dump = std::getenv("OSMOS_LAYERS");
  g_dir = dump ? dump : std::string(home ? home : "") + "/osmos-oracle/layers3";
  const std::string mdir = std::string(home ? home : "") + "/models/ds1017/";
  int ndev = 0;
  int kLayers = 0;
  while (access((g_dir + "/" + std::to_string(kLayers) + ".h_out.bin").c_str(), R_OK) == 0) kLayers++;
  if (!kLayers || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_forward_gpu] skipped: no oracle dump (oracle/layer_dump.py) or no GPU\n");
    return 77;
  }
  char err[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "", ferr[FORWARD_ERR] = "";
  Gguf* g = gguf_create((mdir + "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf").c_str(), err);
  Gguf* eg = gguf_create((mdir + "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf").c_str(), err);
  CHECK(g && eg);
  Weights* w = weights_create(g, eg, werr);
  CHECK(w);
  CHECK(kLayers <= w->geo.n_layers);
  CHECK(!forward_create(w, 0, 64, kBudget, ferr) && !forward_create(w, kLayers, 64, 64ull << 20, ferr));
  std::printf("  refusals: 0 layers; a budget below the layers' weights (%s)\n", ferr);
  Forward* f = forward_create(w, kLayers, 64, kBudget, ferr);
  if (!f) std::printf("  forward_create: %s\n", ferr);
  CHECK(f);
  Probe P;
  P.g = g;
  P.wts = w;
  Arr ids;
  CHECK(load("ids", &ids));
  P.T = (int) ids.dims[0];
  // three runs: free and probed, forced, free and not probed
  std::map<std::string, Diff> free_d, forced_d;
  uint64_t print[3];
  for (int run = 0; run < 3; run++) {
    P.d.clear();
    P.force = run == 1;
    forward_probe(f, run < 2 ? probe : nullptr, &P);
    Seq* s = seq_create(f, ferr);
    CHECK(s);
    uint64_t h = 1469598103934665603ull;
    std::vector<float> out((size_t) w->geo.hc * w->geo.n_embd);
    for (int p = 0; p < P.T; p++) {
      if (!forward_step(f, s, (int32_t) ids.v[p], ferr)) std::printf("  step %d: %s\n", p, ferr);
      CHECK(hipMemcpy(out.data(), forward_stream(f), out.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
      for (size_t i = 0; i < out.size() * 4; i++) h = (h ^ ((const uint8_t*) out.data())[i]) * 1099511628211ull;
    }
    print[run] = h;
    seq_destroy(s);
    if (run == 0) free_d = P.d;
    if (run == 1) forced_d = P.d;
  }
  std::printf("  %d positions, a token at a time, layers 0-%d; the worst position of each step against ref.py,\n"
              "  free (the forward alone) | forced (each step on the oracle's input):\n", P.T, kLayers - 1);
  bool ok = true;
  for (auto& [k, d] : free_d) {
    const Diff& t = forced_d[k];
    const bool exact = k.find("sel") != std::string::npos || k.find("route_idx") != std::string::npos;
    const bool rounded = k.size() > 3 && (k.substr(k.size() - 3) == ".kv" || k.substr(k.size() - 4) == ".ckv");
    if (exact) {
      std::printf("    %-12s %3d positions, not the oracle's: free %d | forced %d\n", k.c_str(), d.seen, d.exact_bad,
                  t.exact_bad);
      ok &= t.exact_bad == 0;
    } else if (rounded) {
      std::printf("    %-12s %3d rows   free %.2e | forced: %llu of %llu values a step apart\n", k.c_str(), d.seen,
                  d.rel, (unsigned long long) t.flips, (unsigned long long) t.values);
      ok &= t.flips <= kMaxFlips * t.values;
    } else {
      std::printf("    %-12s %3d positions   free %.2e | forced %.2e\n", k.c_str(), d.seen, d.rel, t.rel);
      ok &= t.rel <= kMaxRel;
    }
  }
  std::printf("  forced routes that are ties (keys within F32 rounding, kKeyUlps): %d\n", P.n_ties);
  std::printf("  the stream out, every position: fingerprint %016llx (probed) %016llx (not)\n",
              (unsigned long long) print[0], (unsigned long long) print[2]);
  CHECK(ok && print[0] == print[2]);
  forward_destroy(f);
  // a step's most tokens its own (forward_create_with's max_tokens): a batch of them taken, one more refused
  {
    const ForwardOptions o = {kLayers, (uint32_t) MM_MAX_TOKENS + 64, kBudget, nullptr, MM_MAX_TOKENS};
    CHECK(!forward_create_with(w, nullptr, ferr));
    Forward* small = forward_create_with(w, &o, ferr);
    CHECK(small);
    Seq* s0 = seq_create(small, ferr);
    std::vector<int32_t> toks(MM_MAX_TOKENS + 1, 0);
    std::vector<Seq*> ss(MM_MAX_TOKENS + 1, s0);
    CHECK(!forward_batch(small, ss.data(), toks.data(), MM_MAX_TOKENS + 1, ferr));
    CHECK(forward_batch(small, ss.data(), toks.data(), MM_MAX_TOKENS, ferr));
    seq_destroy(s0);
    forward_destroy(small);
    std::printf("  max_tokens %d: a batch of them taken, one more refused\n", MM_MAX_TOKENS);
  }
  {  // a small step's forward (max_tokens 500: its buffers 512 rows): a batch of 500 taken, 501 refused, and the ids
     // a token at a time the stream's bits of the forward above (print[2])
    const ForwardOptions o = {kLayers, 1024, kBudget, nullptr, 500};
    Forward* small = forward_create_with(w, &o, ferr);
    CHECK(small);
    Seq* s0 = seq_create(small, ferr);
    std::vector<int32_t> toks(501, 0);
    std::vector<Seq*> ss(501, s0);
    CHECK(!forward_batch(small, ss.data(), toks.data(), 501, ferr));
    CHECK(forward_batch(small, ss.data(), toks.data(), 500, ferr));
    seq_destroy(s0);
    Seq* s = seq_create(small, ferr);
    CHECK(s);
    uint64_t h = 1469598103934665603ull;
    std::vector<float> out((size_t) w->geo.hc * w->geo.n_embd);
    for (int p = 0; p < P.T; p++) {
      CHECK(forward_step(small, s, (int32_t) ids.v[p], ferr));
      CHECK(hipMemcpy(out.data(), forward_stream(small), out.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
      for (size_t i = 0; i < out.size() * 4; i++) h = (h ^ ((const uint8_t*) out.data())[i]) * 1099511628211ull;
    }
    CHECK(h == print[2]);
    seq_destroy(s);
    forward_destroy(small);
    std::printf("  max_tokens 500: a batch of them taken, one more refused; a token at a time the same bits\n");
  }
  weights_destroy(w);
  gguf_destroy(g);
  gguf_destroy(eg);
  std::printf("[test_forward_gpu] ok\n");
  return 0;
}
