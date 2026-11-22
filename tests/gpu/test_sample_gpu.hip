// a sampled token (ops_sample, ops_draw_uniform: 0.14.0's sampling) against the definition on the host in double:
// temperature's softmax, top-p's kept set (every token at least the threshold whose kept mass reaches top_p, ties
// kept), the inverse CDF in id order at u - on random rows at Qwen's and Inkling's vocabularies and small ones, at
// temperatures 0.5 to 1.5 and top_p 0.05 to 1 (a draw near a CDF or kept-set edge, within the F32 sums' rounding,
// not judged); the same row and u the same token every launch; a small row's tokens at N evenly spaced u each its
// probability's share to within 1 / N (the draw's distribution, deterministically); NaN rows -1; refusals; the
// uniform a function of (seed, position) alone, in [0, 1)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
#include "osmos/gpu/ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

static uint64_t g_rng = 1017;
static double unit() {
  g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
  return (double) (g_rng >> 11) / 9007199254740992.0;
}
static double gauss() { return std::sqrt(-2 * std::log(unit() + 1e-300)) * std::cos(6.283185307179586 * unit()); }

// the definition at a cut: the token for kept set p >= tau, or -2 when u's point lies within `eps` of a CDF edge
static int32_t at_cut(const std::vector<double>& p, double tau, double u, double eps) {
  double kept = 0;
  for (double v : p)
    if (v >= tau) kept += v;
  const double r = u * kept;
  double c = 0;
  for (size_t i = 0; i < p.size(); i++)
    if (p[i] >= tau) {
      if (std::fabs(c + p[i] - r) <= eps * kept) return -2;  // r at an edge
      if ((c += p[i]) > r) return (int32_t) i;
    }
  return -2;
}

// the definition: the token, or -2 when the F32 sums' rounding could decide it (within 1e-6 of the mass: probed, no
// mismatch down to it) - u at a CDF edge, or the top-p cut (the fewest greatest reaching top_p of the mass, ties
// kept) within rounding of the target and one place either way giving another token
static int32_t reference(const std::vector<float>& x, double T, double top_p, double u, double eps) {
  const size_t n = x.size();
  double m = -INFINITY;
  for (float v : x) m = std::max(m, (double) v);
  std::vector<double> p(n);
  double z = 0;
  for (size_t i = 0; i < n; i++) z += p[i] = std::exp(((double) x[i] - m) / T);
  if (top_p >= 1) return at_cut(p, 0, u, eps);
  std::vector<double> s(p);
  std::sort(s.begin(), s.end(), std::greater<double>());
  const double target = top_p * z;
  double before = 0;  // the mass of s[0 .. k), short of the target
  size_t k = 0;
  while (k + 1 < n && before + s[k] < target) before += s[k++];
  const int32_t want = at_cut(p, s[k], u, eps);
  if (want == -2) return -2;
  // the cut moved by the sums' rounding: one place in (its mass short of the target within it), or one out
  if (k && before >= target - eps * z && at_cut(p, s[k - 1], u, eps) != want) return -2;
  if (k + 1 < n && before + s[k] < target + eps * z && at_cut(p, s[k + 1], u, eps) != want) return -2;
  return want;
}

static int32_t draw(const float* dx, uint32_t n, float T, float top_p, float u, int32_t* dout) {
  CHECK(ops_sample(dx, n, T, top_p, u, dout, nullptr));
  int32_t t = 0;
  CHECK(hipMemcpy(&t, dout, 4, hipMemcpyDeviceToHost) == hipSuccess);
  return t;
}

int main() {
  std::printf("[test_sample_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_sample_gpu] skipped: no GPU\n");
    return 77;
  }
  int32_t* dout;
  CHECK(hipMalloc(&dout, 4) == hipSuccess);
  // the definition's token, and the same token every launch
  int judged = 0, edges = 0;
  for (uint32_t n : {5u, 300u, 151936u, 200064u, 248320u}) {
    std::vector<float> x(n);
    float* dx;
    CHECK(hipMalloc(&dx, (size_t) n * 4) == hipSuccess);
    for (int row = 0; row < 4; row++) {
      const double spread = 1.0 + 3.0 * row;  // flat to peaked rows
      for (float& v : x) v = (float) (spread * gauss());
      CHECK(hipMemcpy(dx, x.data(), (size_t) n * 4, hipMemcpyHostToDevice) == hipSuccess);
      for (float T : {0.5f, 0.7f, 1.0f, 1.5f})
        for (float top_p : {0.05f, 0.5f, 0.9f, 0.95f, 1.0f})
          for (int k = 0; k < 6; k++) {
            const float u = (float) unit();
            const int32_t want = reference(x, T, top_p, u, 1e-6), got = draw(dx, n, T, top_p, u, dout);
            CHECK(got >= 0 && (uint32_t) got < n && got == draw(dx, n, T, top_p, u, dout));
            if (want == -2) {
              edges++;
              continue;
            }
            if (got != want)
              std::fprintf(stderr, "  n %u T %g top_p %g u %.7f: %d, the definition's %d\n", n, T, top_p, u, got, want);
            CHECK(got == want);
            judged++;
          }
    }
    CHECK(hipFree(dx) == hipSuccess);
  }
  std::printf("  %d draws the definition's (%d at an edge within the sums' rounding, not judged); each the same "
              "token at every launch\n", judged, edges);
  // the distribution: a small row, N evenly spaced u - each token's share its probability's to within 1 / N
  {
    const std::vector<float> x = {0.0f, 1.0f, -0.5f, 2.0f, 0.25f, -3.0f, 1.5f, 0.0f};
    const uint32_t n = (uint32_t) x.size(), N = 4096;
    float* dx;
    CHECK(hipMalloc(&dx, n * 4) == hipSuccess && hipMemcpy(dx, x.data(), n * 4, hipMemcpyHostToDevice) == hipSuccess);
    for (float T : {0.6f, 1.0f})
      for (float top_p : {1.0f, 0.8f}) {
        std::vector<int> count(n, 0);
        for (uint32_t k = 0; k < N; k++) count[(size_t) draw(dx, n, T, top_p, (k + 0.5f) / N, dout)]++;
        double m = -INFINITY, z = 0;
        for (float v : x) m = std::max(m, (double) v);
        std::vector<double> p(n);
        for (uint32_t i = 0; i < n; i++) z += p[i] = std::exp(((double) x[i] - m) / T);
        std::vector<double> s(p);
        std::sort(s.begin(), s.end(), std::greater<double>());
        double tau = 0, acc = 0;
        for (double v : s)
          if ((acc += v) >= top_p * z) {
            tau = v;
            break;
          }
        double kept = 0;
        for (double v : p) kept += v >= tau ? v : 0;
        for (uint32_t i = 0; i < n; i++) {
          const double want = p[i] >= tau ? p[i] / kept : 0, got = (double) count[i] / N;
          CHECK(std::fabs(got - want) <= 1.0 / N + 1e-6);
        }
      }
    CHECK(hipFree(dx) == hipSuccess);
    std::printf("  a row's tokens at %u evenly spaced u: each its probability's share within 1 / %u (T 0.6, 1; top_p "
                "1, 0.8: the cut tokens never)\n", N, N);
  }
  // a NaN or an inf: -1; refusals
  {
    std::vector<float> x(100, 0.5f);
    x[37] = NAN;
    float* dx;
    CHECK(hipMalloc(&dx, 400) == hipSuccess && hipMemcpy(dx, x.data(), 400, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(draw(dx, 100, 1.0f, 1.0f, 0.5f, dout) == -1);
    x[37] = INFINITY;
    CHECK(hipMemcpy(dx, x.data(), 400, hipMemcpyHostToDevice) == hipSuccess && draw(dx, 100, 1.0f, 1.0f, 0.5f, dout) == -1);
    CHECK(!ops_sample(nullptr, 100, 1, 1, 0.5f, dout, nullptr) && !ops_sample(dx, 0, 1, 1, 0.5f, dout, nullptr) &&
          !ops_sample(dx, 100, 0, 1, 0.5f, dout, nullptr) && !ops_sample(dx, 100, 1, 0, 0.5f, dout, nullptr) &&
          !ops_sample(dx, 100, 1, 1.5f, 0.5f, dout, nullptr) && !ops_sample(dx, 100, 1, 1, 1.0f, dout, nullptr) &&
          !ops_sample(dx, 100, 1, 1, -0.1f, dout, nullptr));
    CHECK(hipFree(dx) == hipSuccess);
    std::printf("  a NaN or an inf: -1; no row, temperature 0, top_p 0 or past 1, u past [0, 1): refused\n");
  }
  // the uniform: (seed, position) alone, in [0, 1)
  {
    double mean = 0;
    for (uint64_t pos = 0; pos < 10000; pos++) {
      const float u = ops_draw_uniform(42, pos);
      CHECK(u >= 0.0f && u < 1.0f && u == ops_draw_uniform(42, pos));
      mean += u;
    }
    CHECK(std::fabs(mean / 10000 - 0.5) < 0.01);
    CHECK(ops_draw_uniform(42, 7) != ops_draw_uniform(43, 7) && ops_draw_uniform(42, 7) != ops_draw_uniform(42, 8));
    std::printf("  the uniform: a function of (seed, position), in [0, 1), mean %.4f over 10,000 positions\n", mean / 10000);
  }
  CHECK(hipFree(dout) == hipSuccess);
  std::printf("[test_sample_gpu] ok\n");
  return 0;
}
