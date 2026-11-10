// q4x_conv's DeltaNet path (4 taps, dilation 1: a thread a channel over a run of tokens) and q4x_moe_sum (4 values a
// thread) against the one-value-a-thread kernels they replace, copied here as they were: every output the same
// bytes - token counts off the run's multiple, the conv's history rows, k picks in where's order, the shared
// expert's gate; moe_sum's fallbacks (a width not a multiple of 4, a misaligned row) the same too. needs a GPU;
// skipped (77) without one.
#pragma clang fp contract(off)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "block.h"

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

constexpr int kThreads = 256;

__device__ inline float silu_ref(float x) { return x / (1.0f + expf(-x)); }

// kernels/q4x_ssm.hip's conv_kernel before the run kernel
__global__ void conv_ref(const float* x, uint32_t n_tok, uint32_t ch, uint32_t taps, uint32_t dil, const float* w,
                         const float* hist, float* y) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * ch) return;
  const uint32_t c = (uint32_t) (i % ch), t = (uint32_t) (i / ch), h = (taps - 1) * dil;
  float sum = 0;
  for (uint32_t k = 0; k < taps; k++) {
    const uint32_t at = t + k * dil;
    const float v = at < h ? hist[(uint64_t) at * ch + c] : x[(uint64_t) (at - h) * ch + c];
    sum += v * w[(uint64_t) c * taps + k];
  }
  y[i] = silu_ref(sum);
}

// kernels/q4x_moe.hip's moe_sum_kernel before the 4-wide one
__global__ void moe_sum_ref(const float* y, const int32_t* where, const float* w, uint32_t n_tok, uint32_t k,
                            uint32_t dim, const float* shared, const float* shared_logit, float* out) {
  const uint64_t i = (uint64_t) blockIdx.x * kThreads + threadIdx.x;
  if (i >= (uint64_t) n_tok * dim) return;
  const uint64_t t = i / dim, d = i % dim;
  float sum = 0;
  for (uint32_t j = 0; j < k; j++) sum += y[(uint64_t) where[t * k + j] * dim + d] * w[t * k + j];
  out[i] = sum + shared[i] * sigmoid(shared_logit[t]);
}

static uint32_t rng = 1017;
static uint32_t next() { return rng = rng * 1664525u + 1013904223u; }
static float uniform() { return (float) (next() >> 8) / (float) (1 << 23) - 1.0f; }  // [-1, 1)

static float* dev(const std::vector<float>& h, size_t pad = 0) {
  float* d = nullptr;
  CHECK(hipMalloc(&d, (h.size() + pad) * 4) == hipSuccess);
  CHECK(hipMemcpy(d + pad, h.data(), h.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  return d;
}
static size_t differ(const float* a, const float* b, size_t n) {
  std::vector<uint32_t> ha(n), hb(n);
  CHECK(hipMemcpy(ha.data(), a, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(hb.data(), b, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  size_t bad = 0;
  for (size_t i = 0; i < n; i++) bad += ha[i] != hb[i];
  return bad;
}

static void conv(uint32_t n, uint32_t ch) {
  constexpr uint32_t kTaps = 4, kHist = kTaps - 1;
  std::vector<float> x((size_t) n * ch), w((size_t) ch * kTaps), hist((size_t) kHist * ch);
  for (float& v : x) v = 4.0f * uniform();
  for (float& v : w) v = uniform();
  for (float& v : hist) v = 4.0f * uniform();
  float *dx = dev(x), *dw = dev(w), *ring = dev(hist), *ya = nullptr, *yb = nullptr;
  CHECK(hipMalloc(&ya, x.size() * 4) == hipSuccess && hipMalloc(&yb, x.size() * 4) == hipSuccess);
  conv_ref<<<(unsigned) ((x.size() + kThreads - 1) / kThreads), kThreads>>>(dx, n, ch, kTaps, 1, dw, ring, yb);
  CHECK(hipGetLastError() == hipSuccess);
  CHECK(q4x_conv(dx, n, ch, kTaps, 1, dw, ring, 1, 0, ya, nullptr) && hipDeviceSynchronize() == hipSuccess);
  const size_t bad = differ(ya, yb, x.size());
  if (bad) std::fprintf(stderr, "  conv %u tokens x %u: %zu values differ\n", n, ch, bad);
  CHECK(bad == 0);
  std::printf("  conv, 4 taps: %4u tokens x %u channels, the one-a-thread kernel's bits\n", n, ch);
  CHECK(hipFree(dx) == hipSuccess && hipFree(dw) == hipSuccess && hipFree(ring) == hipSuccess);
  CHECK(hipFree(ya) == hipSuccess && hipFree(yb) == hipSuccess);
}

// pad: the rows' first float that many past an aligned base (1: misaligned, the fallback's)
static void moe_sum(uint32_t n, uint32_t k, uint32_t dim, size_t pad) {
  const uint32_t picks = n * k;
  std::vector<float> y((size_t) picks * dim), w(picks), sh((size_t) n * dim), sl(n);
  for (float& v : y) v = 8.0f * uniform();
  for (float& v : w) v = uniform();
  for (float& v : sh) v = 8.0f * uniform();
  for (float& v : sl) v = 6.0f * uniform();
  std::vector<float> wh(picks);
  std::vector<int32_t> where(picks);
  for (uint32_t i = 0; i < picks; i++) where[i] = (int32_t) i;
  for (uint32_t i = picks; i > 1; i--) std::swap(where[i - 1], where[next() % i]);  // the places shuffled
  float *dy = dev(y, pad), *dw = dev(w), *dsh = dev(sh, pad), *dsl = dev(sl), *oa = nullptr, *ob = nullptr;
  int32_t* dwh = nullptr;
  CHECK(hipMalloc(&dwh, picks * 4) == hipSuccess);
  CHECK(hipMemcpy(dwh, where.data(), picks * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMalloc(&oa, ((size_t) n * dim + pad) * 4) == hipSuccess && hipMalloc(&ob, (size_t) n * dim * 4) == hipSuccess);
  moe_sum_ref<<<(unsigned) (((size_t) n * dim + kThreads - 1) / kThreads), kThreads>>>(dy + pad, dwh, dw, n, k, dim,
                                                                                       dsh + pad, dsl, ob);
  CHECK(hipGetLastError() == hipSuccess);
  CHECK(q4x_moe_sum(dy + pad, dwh, dw, n, k, dim, dsh + pad, dsl, oa + pad, nullptr) &&
        hipDeviceSynchronize() == hipSuccess);
  const size_t bad = differ(oa + pad, ob, (size_t) n * dim);
  if (bad) std::fprintf(stderr, "  moe_sum %u tokens x %u (k %u, pad %zu): %zu values differ\n", n, dim, k, pad, bad);
  CHECK(bad == 0);
  std::printf("  moe_sum: %4u tokens x %u, k %u%s: the one-a-thread kernel's bits\n", n, dim, k,
              pad || dim % 4 ? " (the fallback)" : "");
  CHECK(hipFree(dy) == hipSuccess && hipFree(dw) == hipSuccess && hipFree(dsh) == hipSuccess && hipFree(dsl) == hipSuccess);
  CHECK(hipFree(dwh) == hipSuccess && hipFree(oa) == hipSuccess && hipFree(ob) == hipSuccess);
}

int main() {
  std::printf("[test_q4x_conv_sum_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_q4x_conv_sum_gpu] skipped: no GPU\n");
    return 77;
  }
  for (uint32_t n : {1u, 3u, 8u, 37u, 2048u}) conv(n, 10240);  // Qwen3.8's DeltaNet channels
  conv(5, 300);                                                // channels past a block's, not 256s
  for (uint32_t n : {1u, 7u, 512u}) moe_sum(n, 10, 2560, 0);  // Qwen3.8's experts
  moe_sum(9, 3, 2560, 1);                                      // a misaligned row: the fallback
  moe_sum(9, 3, 2562, 0);                                      // a width not a multiple of 4: the fallback
  std::printf("[test_q4x_conv_sum_gpu] ok\n");
  return 0;
}
