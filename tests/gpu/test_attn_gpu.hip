// ops_sparse_attn at the model's shape and a long context's key count (64 heads x 512, a window of 128 and up to 512
// compressed rows: 640 keys, several of each kernel's tiles): every output within the rounding bound of its order
// against a double-precision attention over the same keys, and each token the same bits alone and among the others
// (schedule invariance). needs a GPU; skipped (77) without one.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "substrate/gpu/ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr uint32_t kHeads = 64, kHd = 512, kWindow = 128, kSel = 512, kRows = 2048, kSlots = 256, kTok = 24;
constexpr int32_t kPos0 = 6000;  // the tokens' first position: every window full
constexpr double kU = 1.0 / (1 << 24);
// an output's bound in units of u * (its values' sum of |p v|) / den: the scores' kHd terms and the values' up to
// kWindow + kSel, each relative error carried through exp; generous, an indexing bug is off by O(1)
constexpr double kBound = 4.0 * (kHd + kWindow + kSel);

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) ((rng >> 8) * (1.0 / (1 << 24)) * 2.0 - 1.0);
}

struct Host {
  std::vector<float> q, ring, ckv, sink;
  std::vector<int32_t> pos, sel, n_sel;
};

// token t's keys as ops_sparse_attn takes them: the window's ring rows, then its selected rows
static const float* key(const Host& h, uint32_t t, uint32_t k) {
  if (k < kWindow) return &h.ring[(size_t) ((h.pos[t] - (int32_t) kWindow + 1 + (int32_t) k) % kSlots) * kHd];
  return &h.ckv[(size_t) h.sel[(size_t) t * OPS_MAX_SELECT + k - kWindow] * kHd];
}

static double worst = 0;
static void reference(const Host& h, float scale, uint32_t t, const float* o) {
  const uint32_t n = kWindow + (uint32_t) h.n_sel[t];
  std::vector<double> s(n);
  for (uint32_t hd = 0; hd < kHeads; hd++) {
    const float* q = &h.q[((size_t) t * kHeads + hd) * kHd];
    double m = h.sink[hd];
    for (uint32_t k = 0; k < n; k++) {
      const float* kv = key(h, t, k);
      double dot = 0;
      for (uint32_t d = 0; d < kHd; d++) dot += (double) q[d] * kv[d];
      s[k] = dot * scale;
      m = std::fmax(m, s[k]);
    }
    double den = std::exp(h.sink[hd] - m);
    for (uint32_t k = 0; k < n; k++) den += (s[k] = std::exp(s[k] - m));
    for (uint32_t d = 0; d < kHd; d++) {
      double sum = 0, mag = 0;
      for (uint32_t k = 0; k < n; k++) {
        const double pv = s[k] * key(h, t, k)[d];
        sum += pv, mag += std::fabs(pv);
      }
      const double got = o[((size_t) t * kHeads + hd) * kHd + d], err = std::fabs(got - sum / den);
      const double unit = kU * (mag / den + 1e-30);
      CHECK(std::isfinite(got) && err <= kBound * unit);
      if (err / unit > worst) worst = err / unit;
    }
  }
}

int main() {
  std::printf("[test_attn_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_attn_gpu] skipped: no GPU\n");
    return 77;
  }
  Host h;
  h.q.resize((size_t) kTok * kHeads * kHd), h.ring.resize((size_t) kSlots * kHd), h.ckv.resize((size_t) kRows * kHd);
  h.sink.resize(kHeads);
  for (float& v : h.q) v = uniform();
  for (float& v : h.ring) v = uniform();
  for (float& v : h.ckv) v = uniform();
  for (float& v : h.sink) v = uniform();
  h.pos.resize(kTok), h.sel.resize((size_t) kTok * OPS_MAX_SELECT), h.n_sel.resize(kTok);
  for (uint32_t t = 0; t < kTok; t++) {
    h.pos[t] = kPos0 + (int32_t) t;
    h.n_sel[t] = (int32_t) (kSel - (t * 37) % 200);  // ragged: tiles ending part way, at different keys
    for (uint32_t k = 0; k < kSel; k++) {
      rng = rng * 1664525u + 1013904223u;
      h.sel[(size_t) t * OPS_MAX_SELECT + k] = (int32_t) ((rng >> 8) % kRows);
    }
  }
  const float scale = (float) std::pow((double) kHd, -0.5);
  float *q, *ring, *ckv, *sink, *sc, *o;
  int32_t *pos, *sel, *nsel;
  const size_t qb = h.q.size() * 4, scb = OPS_ATTN_SCRATCH(kTok, kHeads, kWindow + kSel) * 4;
  CHECK(hipMalloc(&q, qb) == hipSuccess && hipMalloc(&o, qb) == hipSuccess && hipMalloc(&sc, scb) == hipSuccess);
  CHECK(hipMalloc(&ring, h.ring.size() * 4) == hipSuccess && hipMalloc(&ckv, h.ckv.size() * 4) == hipSuccess);
  CHECK(hipMalloc(&sink, kHeads * 4) == hipSuccess && hipMalloc(&pos, kTok * 4) == hipSuccess);
  CHECK(hipMalloc(&sel, h.sel.size() * 4) == hipSuccess && hipMalloc(&nsel, kTok * 4) == hipSuccess);
  CHECK(hipMemcpy(q, h.q.data(), qb, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(ring, h.ring.data(), h.ring.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(ckv, h.ckv.data(), h.ckv.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(sink, h.sink.data(), kHeads * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(pos, h.pos.data(), kTok * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(sel, h.sel.data(), h.sel.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(nsel, h.n_sel.data(), kTok * 4, hipMemcpyHostToDevice) == hipSuccess);
  const auto attn = [&](uint32_t t0, uint32_t n) {
    CHECK(ops_sparse_attn(q + (size_t) t0 * kHeads * kHd, ring, kSlots, pos + t0, kWindow, 0, ckv,
                          sel + (size_t) t0 * OPS_MAX_SELECT, nsel + t0, OPS_MAX_SELECT, kSel, sink, scale, kHeads, kHd,
                          n, sc, o + (size_t) t0 * kHeads * kHd, nullptr));
  };

  attn(0, kTok);
  std::vector<float> all(h.q.size()), one(h.q.size());
  CHECK(hipMemcpy(all.data(), o, qb, hipMemcpyDeviceToHost) == hipSuccess);
  for (uint32_t t = 0; t < kTok; t++) reference(h, scale, t, all.data());
  std::printf("  %u tokens x %u heads, %u-%u keys: within %.0f u of a double attention (worst %.1f)\n", kTok, kHeads,
              kWindow + kSel - 199, kWindow + kSel, kBound, worst);

  CHECK(hipMemset(o, 0x7F, qb) == hipSuccess);
  for (uint32_t t = 0; t < kTok; t++) attn(t, 1);
  CHECK(hipMemcpy(one.data(), o, qb, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(std::memcmp(all.data(), one.data(), qb) == 0);
  std::printf("  each token alone: the bits it takes among the %u\n", kTok);

  CHECK(!ops_sparse_attn(q, ring, kSlots, pos, kWindow, 0, ckv, sel, nsel, OPS_MAX_SELECT, kSel, sink, scale, kHeads,
                         128, kTok, sc, o, nullptr));  // head_dim not a multiple of 256
  for (void* p : {(void*) q, (void*) o, (void*) sc, (void*) ring, (void*) ckv, (void*) sink, (void*) pos, (void*) sel,
                  (void*) nsel})
    CHECK(hipFree(p) == hipSuccess);
  std::printf("[test_attn_gpu] ok\n");
  return 0;
}
