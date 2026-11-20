// lfm3_shortconv (osmos/gpu/lfm3_ops.h): LFM3's short convolution between its projections - C * conv(B * x), 7 taps,
// no activation - against a double-precision reference with a state before the step, at osmos1.7b's width (2,048);
// a step split in two the same bits and the same states; the ring's slots each token's state (a rollback's); refusals
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/lfm3_ops.h"
}

#define CHECK(c)                                                                  \
  do {                                                                            \
    if (!(c)) {                                                                   \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

constexpr uint32_t D = 2048, K = 7, H = K - 1, N = 37, kSlots = 4;

static uint32_t rng = 2048;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / 16777216.0f - 0.5f;
}

template <class T>
static T* up(const std::vector<T>& v) {
  T* d = nullptr;
  CHECK(hipMalloc((void**) &d, v.size() * sizeof(T)) == hipSuccess);
  CHECK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice) == hipSuccess);
  return d;
}

static std::vector<float> down(const float* d, size_t n) {
  std::vector<float> v(n);
  CHECK(hipDeviceSynchronize() == hipSuccess && hipMemcpy(v.data(), d, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return v;
}

int main() {
  std::printf("[test_lfm3_conv_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) return std::printf("[test_lfm3_conv_gpu] skipped: no GPU\n"), 77;
  std::vector<float> bcx((size_t) N * 3 * D), w((size_t) D * K), hist((size_t) H * D);
  for (auto& v : bcx) v = 4.0f * uniform();
  for (auto& v : w) v = uniform();
  for (auto& v : hist) v = 2.0f * uniform();
  // the reference: u' = the state then B * x, out = C * the conv
  std::vector<double> u((size_t) (H + N) * D);
  for (uint32_t i = 0; i < H * D; i++) u[i] = hist[i];
  for (uint32_t t = 0; t < N; t++)
    for (uint32_t c = 0; c < D; c++)
      u[(size_t) (H + t) * D + c] = (double) bcx[(size_t) t * 3 * D + c] * bcx[(size_t) t * 3 * D + 2 * D + c];
  std::vector<double> ref((size_t) N * D);
  for (uint32_t t = 0; t < N; t++)
    for (uint32_t c = 0; c < D; c++) {
      double s = 0;
      for (uint32_t k = 0; k < K; k++) s += (double) w[(size_t) c * K + k] * u[(size_t) (t + k) * D + c];
      ref[(size_t) t * D + c] = (double) bcx[(size_t) t * 3 * D + D + c] * s;
    }
  std::vector<float> ring((size_t) kSlots * H * D, 0.0f);
  std::memcpy(ring.data(), hist.data(), hist.size() * 4);  // slot 0 the state before
  float *dbcx = up(bcx), *dw = up(w), *dring = up(ring), *out = up(std::vector<float>((size_t) N * D));
  CHECK(lfm3_shortconv(dbcx, N, D, K, dw, dring, kSlots, 0, out, nullptr));
  const std::vector<float> got = down(out, (size_t) N * D), states = down(dring, ring.size());
  double worst = 0, top = 0;
  for (size_t i = 0; i < got.size(); i++)
    worst = std::fmax(worst, std::fabs(got[i] - ref[i])), top = std::fmax(top, std::fabs(ref[i]));
  std::printf("  %u tokens of %u, %u taps: max error %.2e of %.3f (%.1e)\n", N, D, K, worst, top, worst / top);
  CHECK(worst <= 1e-6 * top);
  // the ring: after token t (the last kSlots), slot (1 + t) % kSlots the last H rows of u' up to it
  for (uint32_t t = N - kSlots; t < N; t++)
    for (uint32_t k = 0; k < H; k++)
      for (uint32_t c = 0; c < D; c += 97) {
        const float want = (float) u[(size_t) (t + 1 + k) * D + c];
        CHECK(std::fabs(states[((size_t) ((1 + t) % kSlots) * H + k) * D + c] - want) <= 1e-6f * std::fabs(want));
      }
  std::printf("  the ring: each of the last %u tokens' state in its slot\n", kSlots);
  // the step in two (19 then 18: the second from slot 19 % kSlots) - the same outputs and states, bit for bit
  float *dring2 = up(ring), *out2 = up(std::vector<float>((size_t) N * D));
  const uint32_t a = 19;
  CHECK(lfm3_shortconv(dbcx, a, D, K, dw, dring2, kSlots, 0, out2, nullptr));
  CHECK(lfm3_shortconv(dbcx + (size_t) a * 3 * D, N - a, D, K, dw, dring2, kSlots, a % kSlots, out2 + (size_t) a * D,
                       nullptr));
  const std::vector<float> got2 = down(out2, (size_t) N * D), states2 = down(dring2, ring.size());
  CHECK(std::memcmp(got.data(), got2.data(), got.size() * 4) == 0);
  CHECK(std::memcmp(states.data() + (size_t) ((N % kSlots)) * H * D, states2.data() + (size_t) ((N % kSlots)) * H * D,
                    (size_t) H * D * 4) == 0);
  std::printf("  in steps of %u and %u: the same outputs and the same last state, bit for bit\n", a, N - a);
  CHECK(!lfm3_shortconv(dbcx, N, D, LFM3_CONV_TAPS_MAX + 1, dw, dring, kSlots, 0, out, nullptr));
  CHECK(!lfm3_shortconv(dbcx, N, D, K, dw, dring, kSlots, kSlots, out, nullptr));
  CHECK(!lfm3_shortconv(dbcx, 0, D, K, dw, dring, kSlots, 0, out, nullptr));
  std::printf("  refusals: taps past the kernel's, a slot past the ring, no tokens\n");
  for (float* p : {dbcx, dw, dring, out, dring2, out2}) CHECK(hipFree(p) == hipSuccess);
  std::printf("[test_lfm3_conv_gpu] ok\n");
  return 0;
}
