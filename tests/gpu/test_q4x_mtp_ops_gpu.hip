// the MTP head's own ops (osmos/gpu/q4x_ops.h): q4x_top against the host's argmax and softmax in double - the
// greatest the lowest among equals, its probability within F32's rounding of a sum of 248,320 terms; q4x_join the
// bytes it lays out. needs a gfx1151; skipped (77) without one.
#include <cmath>
#include <random>

#include "q4x_ref.h"

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

constexpr uint32_t kVocab = 248320, kTok = 3, kHc = 4, kDim = 2560;
constexpr double kTopRel = 1e-5;  // a probability's relative error: F32 sums of exp over the vocabulary

static void top_case(const std::vector<float>& l, const char* what) {
  float* d;
  int32_t* id;
  float* p;
  CHECK(hipMalloc(&d, l.size() * 4) == hipSuccess && hipMalloc(&id, 4) == hipSuccess && hipMalloc(&p, 4) == hipSuccess);
  CHECK(hipMemcpy(d, l.data(), l.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(q4x_top(d, (uint32_t) l.size(), 1, 0, id, p, nullptr));
  int32_t* id2;  // the pick alone (no probability): the same id
  CHECK(hipMalloc(&id2, 4) == hipSuccess && q4x_top(d, (uint32_t) l.size(), 1, 0, id2, nullptr, nullptr));
  CHECK(host_of(reinterpret_cast<float*>(id2), 1)[0] == host_of(reinterpret_cast<float*>(id), 1)[0]);
  int32_t got;
  float gp;
  CHECK(hipMemcpy(&got, id, 4, hipMemcpyDeviceToHost) == hipSuccess && hipMemcpy(&gp, p, 4, hipMemcpyDeviceToHost) == hipSuccess);
  size_t best = 0;
  for (size_t i = 1; i < l.size(); i++) best = l[i] > l[best] ? i : best;
  double sum = 0;
  for (float v : l) sum += std::exp((double) v - l[best]);
  const double want = 1.0 / sum;
  std::printf("  %-34s id %d (want %zu)  p %.7f (want %.7f, rel %.1e)\n", what, got, best, gp, want,
              std::fabs(gp - want) / want);
  CHECK(got == (int32_t) best && std::fabs(gp - want) <= kTopRel * want);
  CHECK(hipFree(d) == hipSuccess && hipFree(id) == hipSuccess && hipFree(id2) == hipSuccess && hipFree(p) == hipSuccess);
}

// q4x_argmax (the greedy pick split across blocks) the same id as q4x_top (one block a row) on every row: rows of
// the vocabulary's width, a tie far apart (the lower id), NaNs (never taken), a row all one value, the last id
static void argmax_cases(std::mt19937& rng) {
  constexpr uint32_t kRows = 7;
  std::normal_distribution<float> nd(0.0f, 3.0f);
  std::vector<float> l((size_t) kRows * kVocab);
  for (float& v : l) v = nd(rng);
  l[1ull * kVocab + 200000] = 50.0f, l[1ull * kVocab + 3] = 50.0f;           // a tie: 3
  for (uint32_t i = 0; i < kVocab; i += 97) l[2ull * kVocab + i] = NAN;      // NaNs among them
  l[2ull * kVocab + 1000] = 60.0f;
  for (uint32_t i = 0; i < kVocab; i++) l[3ull * kVocab + i] = 1.5f;         // all one: 0
  l[4ull * kVocab + kVocab - 1] = 70.0f;                                     // the last
  for (uint32_t i = 0; i < kVocab; i++) l[5ull * kVocab + i] = -INFINITY;   // none greater than -inf
  l[6ull * kVocab + 3879] = 80.0f, l[6ull * kVocab + 3880] = 80.0f;          // a tie across a slice's seam (3,880 a slice)
  float* d;
  int32_t *a, *b;
  void* part;
  CHECK(hipMalloc(&d, l.size() * 4) == hipSuccess && hipMalloc(&a, kRows * 4) == hipSuccess &&
        hipMalloc(&b, kRows * 4) == hipSuccess && hipMalloc(&part, q4x_argmax_scratch(kRows)) == hipSuccess);
  CHECK(hipMemcpy(d, l.data(), l.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(q4x_top(d, kVocab, kRows, kVocab, a, nullptr, nullptr) && q4x_argmax(d, kVocab, kRows, kVocab, b, part, nullptr));
  std::vector<int32_t> ha(kRows), hb(kRows);
  CHECK(hipMemcpy(ha.data(), a, kRows * 4, hipMemcpyDeviceToHost) == hipSuccess &&
        hipMemcpy(hb.data(), b, kRows * 4, hipMemcpyDeviceToHost) == hipSuccess);
  for (uint32_t r = 0; r < kRows; r++) std::printf("  argmax row %u: top %d, split %d\n", r, ha[r], hb[r]);
  CHECK(ha == hb && ha[1] == 3 && ha[2] == 1000 && ha[3] == 0 && ha[4] == (int32_t) kVocab - 1 && ha[6] == 3879);
  CHECK(!q4x_argmax(d, kVocab, kRows, kVocab, b, nullptr, nullptr) && !q4x_argmax(d, 0, 1, 0, b, part, nullptr));
  CHECK(hipFree(d) == hipSuccess && hipFree(a) == hipSuccess && hipFree(b) == hipSuccess && hipFree(part) == hipSuccess);
}

int main() {
  std::printf("[test_q4x_mtp_ops_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_q4x_mtp_ops_gpu] skipped: no GPU\n");
    return 77;
  }
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 3.0f);
  std::vector<float> l(kVocab);
  for (float& v : l) v = nd(rng);
  top_case(l, "random logits");
  l[123456] = 40.0f;
  top_case(l, "one far ahead (p near 1)");
  l[77] = 40.0f;  // a tie: the lower id
  top_case(l, "a tie: the lowest id");
  l[kVocab - 1] = 41.0f;
  top_case(l, "the last row's last");
  CHECK(!q4x_top(nullptr, kVocab, 1, 0, nullptr, nullptr, nullptr) && !q4x_top(l.data(), 0, 1, 0, nullptr, nullptr, nullptr));
  argmax_cases(rng);

  std::vector<float> e((size_t) kTok * kDim), h((size_t) kTok * kHc * kDim);
  for (float& v : e) v = nd(rng);
  for (float& v : h) v = nd(rng);
  float *de, *dh, *dout;
  CHECK(hipMalloc(&de, e.size() * 4) == hipSuccess && hipMalloc(&dh, h.size() * 4) == hipSuccess &&
        hipMalloc(&dout, h.size() * 8) == hipSuccess);
  CHECK(hipMemcpy(de, e.data(), e.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dh, h.data(), h.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(q4x_join(de, dh, kTok, kHc, kDim, dout, nullptr));
  const std::vector<float> out = host_of(dout, h.size() * 2);
  for (uint32_t t = 0; t < kTok; t++)
    for (uint32_t s = 0; s < kHc; s++) {
      const float* o = &out[((size_t) t * kHc + s) * 2 * kDim];
      CHECK(!std::memcmp(o, &e[(size_t) t * kDim], kDim * 4) &&
            !std::memcmp(o + kDim, &h[((size_t) t * kHc + s) * kDim], kDim * 4));
    }
  std::printf("  q4x_join: each stream [e ; h], the bytes\n");
  std::printf("[test_q4x_mtp_ops_gpu] ok\n");
  return 0;
}
