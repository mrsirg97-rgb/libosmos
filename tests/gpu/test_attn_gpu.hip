// ops_sparse_attn at the model's shape and a long context's key count (64 heads x 512, a window of 128 and up to 512
// compressed rows: 640 keys, several of each kernel's tiles): every output within the rounding bound of its order
// against a double-precision attention over the same keys, and each token the same bits alone and among the others
// (schedule invariance); the indexer's scores the same ways. the keys as a sequence keeps them (kvfmt.h: the window
// FP8, the compressed rows FP4 with an E4M3 scale, the indexer's FP4 with a power of two), rounded and packed on the
// GPU, the references on the rounded values - and the same keys handed over as F32 the same bits again (the packed
// rows read in the kernels exactly as the F32 they hold). needs a GPU; skipped (77) without one.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/kvfmt.h"
#include "osmos/gpu/ops.h"
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
  // the keys as the sequence keeps them: rounded (fp4, a power of two a 32) and packed; the host's the rounded values
  uint8_t* kp = nullptr;
  CHECK(hipMalloc(&kp, rows * kv_row_bytes(KV_FP4_POW2, kIxHd)) == hipSuccess);
  CHECK(ops_fp4_round(k, kh.size(), KV_FP4_POW2_BLOCK, false, kp, kIxHd, nullptr));
  CHECK(hipMemcpy(kh.data(), k, kh.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(ops_index_score(q, kp, KV_FP4_POW2, w, pos, kIxRatio, rows, heads, kIxHd, kIxTok, sc, stride, nullptr));
  std::vector<float> all(scb / 4), one(scb / 4);
  CHECK(hipMemcpy(all.data(), sc, scb, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(ops_index_score(q, k, KV_F32, w, pos, kIxRatio, rows, heads, kIxHd, kIxTok, sc, stride, nullptr));
  CHECK(hipMemcpy(one.data(), sc, scb, hipMemcpyDeviceToHost) == hipSuccess);
  for (uint32_t t = 0; t < kIxTok; t++)  // the packed keys read as the F32 they hold
    CHECK(std::memcmp(&all[(size_t) t * stride], &one[(size_t) t * stride], (size_t) (ph[t] + 1) / kIxRatio * 4) == 0);
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
    CHECK(ops_index_score(q + (size_t) t * heads * kIxHd, kp, KV_FP4_POW2, w + (size_t) t * heads, pos + t, kIxRatio,
                          rows, heads, kIxHd, 1, sc + (size_t) t * stride, stride, nullptr));
  CHECK(hipMemcpy(one.data(), sc, scb, hipMemcpyDeviceToHost) == hipSuccess);
  for (uint32_t t = 0; t < kIxTok; t++)
    CHECK(std::memcmp(&all[(size_t) t * stride], &one[(size_t) t * stride], (size_t) (ph[t] + 1) / kIxRatio * 4) == 0);
  for (void* p : {(void*) q, (void*) k, (void*) kp, (void*) w, (void*) sc, (void*) pos}) CHECK(hipFree(p) == hipSuccess);
  std::printf("  the indexer, %u heads, %u tokens x ~%u rows: the bits of its order in F32 on the host, within"
              " (hd + 7) u of a double sum (worst %.1f), each token alone the bits it takes among them; FP4 keys"
              " packed the bits of their F32\n",
              heads, kIxTok, rows, worst);
}

// ops_index_score_rows (0.7.3): the candidate layers' scoring over a token's kept blocks only - a ratio-1 source with
// 40,000 rows (5,000 blocks of 8), each token keeping a random 2,048 of its visible blocks (its own last always, as
// ops_cand_blocks keeps it), tokens of one sequence: ops_cand_list the bits' blocks ascending; each kept row's score
// the bits ops_index_score gives it, every other row untouched; and ops_topk under the mask the same rows from either
constexpr uint32_t kGRows = 40000, kGBlock = 8, kGKeep = 2048, kGTok = 7;
static void gathered(uint32_t heads) {
  const uint32_t ratio = 1, stride = kGRows, nb = kGRows / kGBlock, words = (nb + 31) / 32;
  std::vector<float> qh((size_t) kGTok * heads * kIxHd), kh((size_t) kGRows * kIxHd), wh((size_t) kGTok * heads);
  for (float& v : qh) v = uniform();
  for (float& v : kh) v = uniform();
  for (float& v : wh) v = uniform();
  std::vector<int32_t> ph(kGTok);
  for (uint32_t t = 0; t < kGTok; t++) ph[t] = (int32_t) (kGRows - kGTok + t);  // past the blocks kept
  std::vector<uint32_t> bits((size_t) kGTok * words, 0);
  for (uint32_t t = 0; t < kGTok; t++) {
    const uint32_t vis = (uint32_t) (ph[t] + 1) / ratio, vb = (vis + kGBlock - 1) / kGBlock;
    std::vector<uint32_t> order(vb);
    for (uint32_t i = 0; i < vb; i++) order[i] = i;
    for (uint32_t i = vb - 1; i > 0; i--) std::swap(order[i], order[(uint32_t) ((uniform() * 0.5f + 0.5f) * (float) i) % (i + 1)]);
    uint32_t* bt = &bits[(size_t) t * words];
    bt[(vb - 1) / 32] |= 1u << ((vb - 1) % 32);
    for (uint32_t i = 0, kept = 1; kept < kGKeep && i < vb; i++)
      if (!((bt[order[i] / 32] >> (order[i] % 32)) & 1u)) bt[order[i] / 32] |= 1u << (order[i] % 32), kept++;
  }
  float *q, *k, *w, *full, *part;
  int32_t *pos, *sel, *n_sel;
  uint32_t *db, *list, *n_list;
  uint8_t* kp;
  const size_t scb = (size_t) kGTok * stride * 4;
  CHECK(hipMalloc(&q, qh.size() * 4) == hipSuccess && hipMalloc(&k, kh.size() * 4) == hipSuccess);
  CHECK(hipMalloc(&w, wh.size() * 4) == hipSuccess && hipMalloc(&full, scb) == hipSuccess);
  CHECK(hipMalloc(&part, scb) == hipSuccess && hipMalloc(&pos, kGTok * 4) == hipSuccess);
  CHECK(hipMalloc(&db, bits.size() * 4) == hipSuccess && hipMalloc(&list, (size_t) kGTok * kGKeep * 4) == hipSuccess);
  CHECK(hipMalloc(&n_list, kGTok * 4) == hipSuccess && hipMalloc(&kp, kGRows * kv_row_bytes(KV_FP4_POW2, kIxHd)) == hipSuccess);
  CHECK(hipMalloc(&sel, (size_t) kGTok * OPS_MAX_SELECT * 4) == hipSuccess && hipMalloc(&n_sel, kGTok * 4) == hipSuccess);
  CHECK(hipMemcpy(q, qh.data(), qh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(k, kh.data(), kh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(w, wh.data(), wh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(pos, ph.data(), kGTok * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(db, bits.data(), bits.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(ops_fp4_round(k, kh.size(), KV_FP4_POW2_BLOCK, false, kp, kIxHd, nullptr));
  // the list: each token's set bits ascending
  CHECK(ops_cand_list(db, words, kGTok, list, kGKeep, n_list, nullptr));
  std::vector<uint32_t> lh((size_t) kGTok * kGKeep), nh(kGTok);
  CHECK(hipMemcpy(lh.data(), list, lh.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(nh.data(), n_list, kGTok * 4, hipMemcpyDeviceToHost) == hipSuccess);
  for (uint32_t t = 0; t < kGTok; t++) {
    uint32_t n = 0;
    for (uint32_t j = 0; j < nb; j++)
      if ((bits[(size_t) t * words + j / 32] >> (j % 32)) & 1u) CHECK(n < kGKeep && lh[(size_t) t * kGKeep + n++] == j);
    CHECK(nh[t] == n && n == kGKeep);
  }
  // the scores: full, and gathered over a buffer of a sentinel
  CHECK(ops_index_score(q, kp, KV_FP4_POW2, w, pos, ratio, kGRows, heads, kIxHd, kGTok, full, stride, nullptr));
  CHECK(hipMemset(part, 0x7F, scb) == hipSuccess);
  CHECK(ops_index_score_rows(q, kp, KV_FP4_POW2, w, pos, ratio, list, n_list, kGKeep, kGBlock, heads, kIxHd, kGTok,
                             part, stride, nullptr));
  std::vector<float> fh(scb / 4), gh(scb / 4);
  CHECK(hipMemcpy(fh.data(), full, scb, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(gh.data(), part, scb, hipMemcpyDeviceToHost) == hipSuccess);
  uint32_t same = 0, left = 0;
  uint32_t sentinel;
  std::memset(&sentinel, 0x7F, 4);
  for (uint32_t t = 0; t < kGTok; t++)
    for (uint32_t c = 0; c < (uint32_t) (ph[t] + 1) / ratio; c++) {
      const uint32_t j = c / kGBlock;
      const size_t i = (size_t) t * stride + c;
      if ((bits[(size_t) t * words + j / 32] >> (j % 32)) & 1u) {
        CHECK(std::memcmp(&fh[i], &gh[i], 4) == 0);
        same++;
      } else {
        CHECK(std::memcmp(&gh[i], &sentinel, 4) == 0);
        left++;
      }
    }
  // the masked selection from either buffer: the same rows
  std::vector<int32_t> s1((size_t) kGTok * OPS_MAX_SELECT), s2(s1.size()), n1(kGTok), n2(kGTok);
  for (int pass = 0; pass < 2; pass++) {
    CHECK(ops_topk(pass ? part : full, stride, pos, ratio, kSel, db, kGBlock, words, kGTok, sel, OPS_MAX_SELECT, n_sel,
                   nullptr));
    CHECK(hipMemcpy((pass ? s2 : s1).data(), sel, s1.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(hipMemcpy((pass ? n2 : n1).data(), n_sel, kGTok * 4, hipMemcpyDeviceToHost) == hipSuccess);
  }
  CHECK(n1 == n2 && s1 == s2 && n1[0] == (int32_t) kSel);
  for (void* p : {(void*) q, (void*) k, (void*) kp, (void*) w, (void*) full, (void*) part, (void*) pos, (void*) db,
                  (void*) list, (void*) n_list, (void*) sel, (void*) n_sel})
    CHECK(hipFree(p) == hipSuccess);
  std::printf("  the indexer gathered, %u heads, %u tokens x %u rows keeping %u of %u blocks: the list the bits' blocks,"
              " %u kept scores the full scoring's bits, %u others untouched, the masked top %u the same rows\n",
              heads, kGTok, kGRows, kGKeep, nb, same, left, kSel);
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
  // the keys as a sequence keeps them: the window's FP8, the compressed rows FP4 (an E4M3 scale a 16), each rounded
  // and packed; the host's reference on the rounded values
  uint8_t *ring_p = nullptr, *ckv_p = nullptr;
  CHECK(hipMalloc(&ring_p, kSlots * kv_row_bytes(KV_FP8, kHd)) == hipSuccess &&
        hipMalloc(&ckv_p, kRows * kv_row_bytes(KV_FP4_E4M3, kHd)) == hipSuccess);
  CHECK(ops_fp8_round(ring, h.ring.size(), ring_p, kHd, nullptr) &&
        ops_fp4_round(ckv, h.ckv.size(), KV_FP4_E4M3_BLOCK, true, ckv_p, kHd, nullptr));
  CHECK(hipMemcpy(h.ring.data(), ring, h.ring.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(h.ckv.data(), ckv, h.ckv.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(sink, h.sink.data(), kHeads * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(pos, h.pos.data(), kTok * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(sel, h.sel.data(), h.sel.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(nsel, h.n_sel.data(), kTok * 4, hipMemcpyHostToDevice) == hipSuccess);
  float* ring_f = nullptr;  // the window's rows unpacked: the ring a prompt's step reads
  CHECK(hipMalloc(&ring_f, h.ring.size() * 4) == hipSuccess && ops_kv_unpack(KV_FP8, ring_p, kSlots, kHd, ring_f, nullptr));
  const auto attn = [&](uint32_t t0, uint32_t n, uint32_t fmt = KV_FP4_E4M3, uint32_t rfmt = KV_FP8) {
    CHECK(ops_sparse_attn(q + (size_t) t0 * kHeads * kHd, rfmt == KV_F32 ? (const void*) ring_f : (const void*) ring_p,
                          rfmt, kSlots, pos + t0, kWindow, 0,
                          fmt == KV_F32 ? (const void*) ckv : (const void*) ckv_p, fmt,
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
  CHECK(hipMemset(o, 0x7F, qb) == hipSuccess);
  attn(0, kTok, KV_F32);  // the compressed rows handed over as the F32 they hold (the drafter's form)
  CHECK(hipMemcpy(one.data(), o, qb, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(std::memcmp(all.data(), one.data(), qb) == 0);
  std::printf("  the compressed rows packed (FP4, %llu bytes a row) or as F32: the same bits\n",
              (unsigned long long) kv_row_bytes(KV_FP4_E4M3, kHd));
  for (uint32_t cf : {(uint32_t) KV_FP4_E4M3, (uint32_t) KV_F32}) {  // the ring unpacked: alone, and every key F32
    CHECK(hipMemset(o, 0x7F, qb) == hipSuccess);
    attn(0, kTok, cf, KV_F32);
    CHECK(hipMemcpy(one.data(), o, qb, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(std::memcmp(all.data(), one.data(), qb) == 0);
    CHECK(hipMemset(o, 0x7F, qb) == hipSuccess);
    for (uint32_t t = 0; t < kTok; t++) attn(t, 1, cf, KV_F32);  // and the few tokens' tiles
    CHECK(hipMemcpy(one.data(), o, qb, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(std::memcmp(all.data(), one.data(), qb) == 0);
  }
  std::printf("  the window's ring unpacked to F32, the compressed rows packed or F32 (every key F32: the plain loads' "
              "kernels), a batch or a token at a time: the same bits\n");

  CHECK(!ops_sparse_attn(q, ring_p, KV_FP8, kSlots, pos, kWindow, 0, ckv_p, KV_FP4_E4M3, sel, nsel, OPS_MAX_SELECT, kSel, sink,
                         scale, kHeads, 128, kTok, sc, o, nullptr));  // head_dim not a multiple of 256
  CHECK(!ops_sparse_attn(q, ring_p, KV_FP8, kSlots, pos, kWindow, 0, ckv_p, 9, sel, nsel, OPS_MAX_SELECT, kSel, sink,
                         scale, kHeads, kHd, kTok, sc, o, nullptr));  // no such format
  CHECK(!ops_sparse_attn(q, ring_p, KV_FP4_E4M3, kSlots, pos, kWindow, 0, ckv_p, KV_FP4_E4M3, sel, nsel, OPS_MAX_SELECT,
                         kSel, sink, scale, kHeads, kHd, kTok, sc, o, nullptr));  // a ring neither FP8 nor F32
  for (void* p : {(void*) q, (void*) o, (void*) sc, (void*) ring, (void*) ckv, (void*) ring_p, (void*) ring_f, (void*) ckv_p,
                  (void*) sink, (void*) pos, (void*) sel, (void*) nsel})
    CHECK(hipFree(p) == hipSuccess);
  indexer(32);
  indexer(24);
  gathered(32);
  gathered(24);
  std::printf("[test_attn_gpu] ok\n");
  return 0;
}
