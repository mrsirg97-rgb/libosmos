// ops_sparse_attn at the model's shape and a long context's key count (64 heads x 512, a window of 128 and up to 512
// compressed rows: 640 keys, several of each kernel's tiles): every output within the rounding bound of its order
// against a double-precision attention over the same keys, and each token the same bits alone and among the others
// (schedule invariance); the indexer's scores the same ways. needs a GPU; skipped (77) without one.
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

// ops_index_score at the indexer's shape (up to 32 heads x 128, ratio 2, a token past 6,000 seeing ~3,000 rows):
// each score the bits of its order replayed on the host and within its rounding bound of a double sum, and each
// token the same bits alone and among the others; `heads` below 32 leaves head slots empty
constexpr uint32_t kIxHd = 128, kIxRatio = 2, kIxTok = 13;
static void indexer(uint32_t heads) {
  const uint32_t rows = (uint32_t) (kPos0 + kIxTok) / kIxRatio, stride = rows;
  std::vector<float> qh((size_t) kIxTok * heads * kIxHd), kh((size_t) rows * kIxHd), wh((size_t) kIxTok * heads);
  for (float& v : qh) v = uniform();
  for (float& v : kh) v = uniform();
  for (float& v : wh) v = uniform();
  std::vector<int32_t> ph(kIxTok);
  for (uint32_t t = 0; t < kIxTok; t++) ph[t] = kPos0 + (int32_t) t;
  float *q, *k, *w, *sc;
  int32_t* pos;
  const size_t scb = (size_t) kIxTok * stride * 4;
  CHECK(hipMalloc(&q, qh.size() * 4) == hipSuccess && hipMalloc(&k, kh.size() * 4) == hipSuccess);
  CHECK(hipMalloc(&w, wh.size() * 4) == hipSuccess && hipMalloc(&sc, scb) == hipSuccess);
  CHECK(hipMalloc(&pos, kIxTok * 4) == hipSuccess);
  CHECK(hipMemcpy(q, qh.data(), qh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(k, kh.data(), kh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(w, wh.data(), wh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(pos, ph.data(), kIxTok * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(ops_index_score(q, k, w, pos, kIxRatio, rows, heads, kIxHd, kIxTok, sc, stride, nullptr));
  std::vector<float> all(scb / 4), one(scb / 4);
  CHECK(hipMemcpy(all.data(), sc, scb, hipMemcpyDeviceToHost) == hipSuccess);
  double worst = 0;
  for (uint32_t t = 0; t < kIxTok; t++)
    for (uint32_t c = 0; c < (uint32_t) (ph[t] + 1) / kIxRatio; c++) {
      double sum = 0, mag = 0;
      for (uint32_t h = 0; h < heads; h++) {
        double dot = 0, dmag = 0;
        for (uint32_t d = 0; d < kIxHd; d++) {
          const double p = (double) qh[((size_t) t * heads + h) * kIxHd + d] * kh[(size_t) c * kIxHd + d];
          dot += p, dmag += std::fabs(p);
        }
        const double wv = wh[(size_t) t * heads + h];
        sum += std::fmax(dot, 0.0) * wv, mag += dmag * std::fabs(wv);
      }
      // the kernel's order in F32 on the host (the build's -ffp-contract=off: a multiply, then an add): the same bits
      float x[32] = {};
      for (uint32_t h = 0; h < heads; h++) {
        float dot = 0;
        for (uint32_t d = 0; d < kIxHd; d++)
          dot += qh[((size_t) t * heads + h) * kIxHd + d] * kh[(size_t) c * kIxHd + d];
        x[h] = std::fmax(dot, 0.0f) * wh[(size_t) t * heads + h];
      }
      for (int off = 16; off > 0; off >>= 1)
        for (int h = 0; h < off; h++) x[h] = x[h] + x[h + off];
      CHECK(std::memcmp(&x[0], &all[(size_t) t * stride + c], 4) == 0);
      // the dot's kIxHd terms, relu * w, then the heads' 5-deep tree: (kIxHd + 2 + 5) u of the magnitudes
      const double err = std::fabs(all[(size_t) t * stride + c] - sum), bound = (kIxHd + 7) * kU * mag + 1e-30;
      CHECK(std::isfinite(all[(size_t) t * stride + c]) && err <= bound);
      if (mag > 0 && err / (kU * mag) > worst) worst = err / (kU * mag);
    }
  CHECK(hipMemset(sc, 0x7F, scb) == hipSuccess);
  for (uint32_t t = 0; t < kIxTok; t++)
    CHECK(ops_index_score(q + (size_t) t * heads * kIxHd, k, w + (size_t) t * heads, pos + t, kIxRatio, rows, heads,
                          kIxHd, 1, sc + (size_t) t * stride, stride, nullptr));
  CHECK(hipMemcpy(one.data(), sc, scb, hipMemcpyDeviceToHost) == hipSuccess);
  for (uint32_t t = 0; t < kIxTok; t++)
    CHECK(std::memcmp(&all[(size_t) t * stride], &one[(size_t) t * stride], (size_t) (ph[t] + 1) / kIxRatio * 4) == 0);
  for (void* p : {(void*) q, (void*) k, (void*) w, (void*) sc, (void*) pos}) CHECK(hipFree(p) == hipSuccess);
  std::printf("  the indexer, %u heads, %u tokens x ~%u rows: the bits of its order in F32 on the host, within"
              " (hd + 7) u of a double sum (worst %.1f), each token alone the bits it takes among them\n",
              heads, kIxTok, rows, worst);
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
  indexer(32);
  indexer(24);
  std::printf("[test_attn_gpu] ok\n");
  return 0;
}
