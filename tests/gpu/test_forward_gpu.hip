// the forward (src/run) over ds1017's first 3 layers - 0 the window alone, 1 with Engram, 2 the first KV source
// (the compressor at ratio 2, the indexer) - the 32 real ids a token at a time, every step it names (its probe)
// against the oracle's (oracle/layer_dump.py -> npz2bin.py, ref.py's prompt path at that position), twice:
//   free: the forward on its own. the routes and the compressed rows attended exactly; the other steps reported (a
//     value at an FP8 / FP4 rounding edge flips a whole step - 1/16 of itself, or 4 -> 6 - and the difference then
//     carries on, so a chained run's differences are not each step's)
//   forced: each step's output, once compared, overwritten with the oracle's, so each step runs on the oracle's
//     input: every float step within kMaxRel of its largest value; the rounded rows (kv, ckv) not the oracle's in at
//     most kMaxFlips of their values (each a step at an edge)
// the stream out's fingerprint the same with a probe and without (determinism; a probe that only reads changes
// nothing).
// needs a gfx1151, ~/models/ds1017 and ~/substrate-oracle/layers3; skipped (77) without them.
#include <hip/hip_runtime.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

extern "C" {
#include "substrate/model/gguf.h"
#include "substrate/model/weights.h"
#include "substrate/run/forward.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr int kLayers = 3;
constexpr double kMaxRel = 1e-3;
constexpr double kMaxFlips = 5e-3;
constexpr uint64_t kMaxResident = 24ull << 30;

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

struct Probe {
  std::map<std::string, Arr> ref;
  std::map<std::string, Diff> d;
  int T = 0;
  bool force = false;
  const Arr* get(const std::string& k) {
    auto it = ref.find(k);
    if (it != ref.end()) return &it->second;
    Arr a;
    if (!load(k, &a)) return nullptr;
    return &(ref[k] = a);
  }
};

static void probe(const char* step, int layer, int index, void* dev, uint64_t n, bool ints, void* arg) {
  Probe* P = (Probe*) arg;
  const std::string s = step, key = std::to_string(layer) + "." + (s == "ckv" || s == "sel" ? "kv" : s);
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
  g_dir = std::string(home ? home : "") + "/substrate-oracle/layers3";
  const std::string mdir = std::string(home ? home : "") + "/models/ds1017/";
  int ndev = 0;
  if (access((g_dir + "/2.h_out.bin").c_str(), R_OK) != 0 || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_forward_gpu] skipped: no oracle dump (oracle/layer_dump.py --layers 3) or no GPU\n");
    return 77;
  }
  char err[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "", ferr[FORWARD_ERR] = "";
  Gguf* g = gguf_create((mdir + "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf").c_str(), err);
  Gguf* eg = gguf_create((mdir + "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf").c_str(), err);
  CHECK(g && eg);
  Weights* w = weights_create(g, eg, werr);
  CHECK(w);
  CHECK(!forward_create(w, 0, 64, kMaxResident, ferr) && !forward_create(w, kLayers, 64, 1 << 20, ferr));
  std::printf("  refusals: 0 layers; more weights than max_resident (%s)\n", ferr);
  Forward* f = forward_create(w, kLayers, 64, kMaxResident, ferr);
  if (!f) std::printf("  forward_create: %s\n", ferr);
  CHECK(f);
  Probe P;
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
      std::printf("    %-12s %3d positions, %d not the oracle's\n", k.c_str(), d.seen, d.exact_bad);
      ok &= d.exact_bad == 0;
    } else if (rounded) {
      std::printf("    %-12s %3d rows   free %.2e | forced: %llu of %llu values a step apart\n", k.c_str(), d.seen,
                  d.rel, (unsigned long long) t.flips, (unsigned long long) t.values);
      ok &= t.flips <= kMaxFlips * t.values;
    } else {
      std::printf("    %-12s %3d positions   free %.2e | forced %.2e\n", k.c_str(), d.seen, d.rel, t.rel);
      ok &= t.rel <= kMaxRel;
    }
  }
  std::printf("  the stream out, every position: fingerprint %016llx (probed) %016llx (not)\n",
              (unsigned long long) print[0], (unsigned long long) print[2]);
  CHECK(ok && print[0] == print[2]);
  forward_destroy(f);
  weights_destroy(w);
  gguf_destroy(g);
  gguf_destroy(eg);
  std::printf("[test_forward_gpu] ok\n");
  return 0;
}
