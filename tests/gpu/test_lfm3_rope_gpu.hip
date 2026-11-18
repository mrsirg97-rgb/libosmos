// LFM3's q and k (DESIGN.md, "vibes-flash: LFM3"): q4x_qk at its shape - head 128, no gate in q (attn_gate is its
// own matrix), the zero tail (32 of 64 pairs rotated, theta the whole dim's progression) - against Liquid's
// modeling_lfm3.py, computed here as it computes: RMS a head times the norm's weight, then NEOX pairs (i, i + 64) at
// position x inv_freq (float32: base^(-2i / 128) for i < 32, 0 past), cos and sin times the attention factor. plain at
// the native context; YaRN x40 (10M: by parts over the whole dim, the prefix kept - compute_rope_parameters) where
// the factor 1 + 0.1 ln 40 scales the unrotated tail too (cos 1, sin 0 times it: Liquid's, not llama.cpp's, which
// scales the rotated pairs alone). then the refusals: no pairs, more than half the dim.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

extern "C" {
#include "osmos/gpu/q4x_ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr uint32_t kHeads = 16, kKv = 4, kDim = 128, kPairs = 32, kTok = 64;
constexpr uint32_t kNative = 262144, kYarn = 40;
constexpr float kBase = 1e6f, kEps = 1e-5f;
constexpr int32_t kPos0 = 8192 - kTok;  // past the sliding window's reach, inside float32's exact positions

// Liquid's inv_freq: the zero tail's (compute_default_rope_parameters), or YaRN's over the whole dim, its prefix kept
// (compute_rope_parameters on transformers' yarn: beta_fast 32, beta_slow 1, attention factor 0.1 ln s + 1)
static std::vector<float> inv_freq(bool yarn, float* attn) {
  std::vector<float> f(kDim / 2, 0.0f);
  *attn = 1.0f;
  for (uint32_t i = 0; i < kDim / 2; i++) f[i] = 1.0f / std::pow(kBase, (float) (2 * i) / (float) kDim);
  if (yarn) {
    const float s = (float) kYarn;
    const auto corr = [](float beta) {
      return (float) kDim * std::log((float) kNative / (beta * 2.0f * (float) M_PI)) / (2.0f * std::log(kBase));
    };
    const float lo = std::max(0.0f, std::floor(corr(32.0f))), hi = std::min((float) kDim - 1, std::ceil(corr(1.0f)));
    for (uint32_t i = 0; i < kDim / 2; i++) {
      const float ramp = std::min(1.0f, std::max(0.0f, ((float) i - lo) / std::max(hi - lo, 0.001f)));
      const float extra = 1.0f - ramp;  // transformers' inv_freq_extrapolation_factor
      f[i] = f[i] / s * (1.0f - extra) + f[i] * extra;
    }
    *attn = 0.1f * std::log(s) + 1.0f;
  }
  for (uint32_t i = kPairs; i < kDim / 2; i++) f[i] = 0.0f;  // the zero tail
  return f;
}

static void reference(const std::vector<float>& x, const std::vector<float>& w, uint32_t rows, bool yarn,
                      std::vector<float>& out) {
  float attn;
  const std::vector<float> f = inv_freq(yarn, &attn);
  out.resize(x.size());
  for (uint32_t r = 0; r < rows; r++) {
    const uint32_t t = r / (uint32_t) (x.size() / kDim / kTok);
    const float* v = &x[(size_t) r * kDim];
    double ss = 0;
    for (uint32_t d = 0; d < kDim; d++) ss += (double) v[d] * v[d];
    const float inv = 1.0f / std::sqrt((float) (ss / kDim) + kEps);
    float y[kDim];
    for (uint32_t d = 0; d < kDim; d++) y[d] = v[d] * inv * w[d];
    for (uint32_t i = 0; i < kDim / 2; i++) {
      const float a = (float) (kPos0 + (int32_t) t) * f[i], c = std::cos(a) * attn, s = std::sin(a) * attn;
      out[(size_t) r * kDim + i] = y[i] * c - y[i + kDim / 2] * s;
      out[(size_t) r * kDim + i + kDim / 2] = y[i + kDim / 2] * c + y[i] * s;
    }
  }
}

template <class T>
static T* dev(const std::vector<T>& h) {
  T* p = nullptr;
  CHECK(hipMalloc((void**) &p, h.size() * sizeof(T)) == hipSuccess);
  CHECK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice) == hipSuccess);
  return p;
}

static std::vector<float> host(const float* p, size_t n) {
  std::vector<float> h(n);
  CHECK(hipMemcpy(h.data(), p, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return h;
}

static void run(bool yarn) {
  std::mt19937 rng(1017 + yarn);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> hq((size_t) kTok * kHeads * kDim), hk((size_t) kTok * kKv * kDim), wq(kDim), wk(kDim);
  for (auto& v : hq) v = nd(rng);
  for (auto& v : hk) v = nd(rng);
  for (auto& v : wq) v = 1.0f + 0.1f * nd(rng);
  for (auto& v : wk) v = 1.0f + 0.1f * nd(rng);
  Q4xRope rope;
  CHECK(q4x_rope(kDim, kPairs, kBase, kNative, yarn ? kYarn * kNative : kNative, Q4X_THETA_HF, &rope));
  float *dq = dev(hq), *dk = dev(hk), *dwq = dev(wq), *dwk = dev(wk), *oq, *ok;
  CHECK(hipMalloc((void**) &oq, hq.size() * 4) == hipSuccess && hipMalloc((void**) &ok, hk.size() * 4) == hipSuccess);
  CHECK(q4x_qk(dq, dk, kTok, kHeads, kKv, kDim, &rope, dwq, dwk, kEps, kPos0, oq, nullptr, ok, nullptr));
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<float> rq, rk;
  reference(hq, wq, kTok * kHeads, yarn, rq);
  reference(hk, wk, kTok * kKv, yarn, rk);
  const std::vector<float> gq = host(oq, hq.size()), gk = host(ok, hk.size());
  double worst = 0, tail = 0;
  for (size_t i = 0; i < gq.size(); i++) {
    const double e = std::fabs(gq[i] - rq[i]);
    worst = std::max(worst, e);
    const uint32_t d = i % kDim;
    if (d % (kDim / 2) >= kPairs) tail = std::max(tail, e);
  }
  for (size_t i = 0; i < gk.size(); i++) worst = std::max(worst, (double) std::fabs(gk[i] - rk[i]));
  std::printf("  %s: q and k against Liquid's, max |d| %.2e (the unrotated tail %.2e)\n", yarn ? "YaRN x40" : "native",
              worst, tail);
  CHECK(worst < 2e-4 && tail < 1e-5);
  CHECK(hipFree(dq) == hipSuccess && hipFree(dk) == hipSuccess && hipFree(dwq) == hipSuccess &&
        hipFree(dwk) == hipSuccess && hipFree(oq) == hipSuccess && hipFree(ok) == hipSuccess);
}

int main() {
  std::printf("[test_lfm3_rope_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_lfm3_rope_gpu] skipped: no GPU\n");
    return 77;
  }
  Q4xRope r;
  CHECK(!q4x_rope(kDim, 0, kBase, kNative, kNative, Q4X_THETA_HF, &r) && !q4x_rope(kDim, kDim / 2 + 1, kBase, kNative, kNative, Q4X_THETA_HF, &r));
  std::printf("  refusals: no pairs, more pairs than half the dim\n");
  run(false);
  run(true);
  std::printf("[test_lfm3_rope_gpu] ok\n");
  return 0;
}
