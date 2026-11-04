// the hyper-connection mixes at one token, against what they must equal bit for bit: ops_hc_dot's few-token path
// (a decode's) against a host replay of its sums (fmaf in each part's order) and against the tiled path (a prompt's,
// the same token among 64); ops_hc_mix's Sinkhorn a lane an entry against a copy of the one-lane kernel it replaced
// (kept here as the reference) - pre, post and comb the same bits, at the model's hc 4 and at 6 (one lane still)
#include <hip/hip_runtime.h>
// built as the library is (-ffp-contract=off): the reference's products and sums each rounded, none fused
#pragma clang fp contract(off)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

static uint32_t rng = 1017;
static float uniform() {  // -1 .. 1
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / (float) (1u << 23) - 1.0f;
}

namespace ref {  // the one-lane mix this repository ran before (kernels/ops.hip, 2026-10-30): the reference
constexpr int kWave = 32, kParts = OPS_HC_PARTS;
__device__ inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
template <uint32_t hc>
__global__ void mix(const float* partial, const float* scale, const float* base, uint32_t dim, float eps, float hc_eps,
                    uint32_t iters, float* pre, float* post, float* comb) {
  __shared__ float rows[(2 + OPS_MAX_HC) * OPS_MAX_HC + 1];
  const uint32_t t = blockIdx.x, n = hc * dim, nm = (2 + hc) * hc;
  const float* pt = partial + (uint64_t) t * (nm + 1) * kParts;
  for (uint32_t i = threadIdx.x; i <= nm; i += kWave) {
    float s = 0;
    for (int p = 0; p < kParts; p++) s += pt[i * kParts + p];
    rows[i] = s;
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  const float r = 1.0f / sqrtf(rows[nm] / (float) n + eps);
  float m[(2 + hc) * hc];
  for (uint32_t i = 0; i < nm; i++) m[i] = rows[i] * r;
  for (uint32_t i = 0; i < hc; i++) {
    pre[t * hc + i] = sigmoid(m[i] * scale[0] + base[i]) + hc_eps;
    post[t * hc + i] = 2.0f * sigmoid(m[hc + i] * scale[1] + base[hc + i]);
  }
  float c[hc][hc];
  for (uint32_t i = 0; i < hc; i++) {
    float mx = -INFINITY;
    for (uint32_t j = 0; j < hc; j++) {
      c[i][j] = m[2 * hc + i * hc + j] * scale[2] + base[2 * hc + i * hc + j];
      mx = fmaxf(mx, c[i][j]);
    }
    float s = 0;
    for (uint32_t j = 0; j < hc; j++) s += (c[i][j] = expf(c[i][j] - mx));
    for (uint32_t j = 0; j < hc; j++) c[i][j] = c[i][j] / s + hc_eps;
  }
  for (uint32_t it = 0; it < iters; it++) {
    if (it > 0)
      for (uint32_t i = 0; i < hc; i++) {
        float s = 0;
        for (uint32_t j = 0; j < hc; j++) s += c[i][j];
        for (uint32_t j = 0; j < hc; j++) c[i][j] = c[i][j] / (s + hc_eps);
      }
    for (uint32_t j = 0; j < hc; j++) {
      float s = 0;
      for (uint32_t i = 0; i < hc; i++) s += c[i][j];
      for (uint32_t i = 0; i < hc; i++) c[i][j] = c[i][j] / (s + hc_eps);
    }
  }
  for (uint32_t i = 0; i < hc; i++)
    for (uint32_t j = 0; j < hc; j++) comb[(t * hc + i) * hc + j] = c[i][j];
}
}  // namespace ref

template <class T> static T* dev(const std::vector<T>& h) {
  T* d = nullptr;
  CHECK(hipMalloc(&d, h.size() * sizeof(T) + 4) == hipSuccess);
  CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice) == hipSuccess);
  return d;
}
template <class T> static std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess);
  return h;
}

template <uint32_t hc> static void run(uint32_t dim, uint32_t iters) {
  constexpr uint32_t nm = (2 + hc) * hc, kTok = 64, kAt = 37;  // the token tested alone, at 37 of 64
  const uint32_t len = hc * dim, parts = (nm + 1) * OPS_HC_PARTS;
  std::vector<float> fn((size_t) nm * len), h((size_t) kTok * len), scale(3), base(nm);
  for (float& v : fn) v = uniform() * 0.05f;
  for (float& v : h) v = uniform();
  for (float& v : scale) v = 0.5f + uniform() * 0.25f;
  for (float& v : base) v = uniform();
  float *dfn = dev(fn), *dh = dev(h), *dsc = dev(scale), *db = dev(base), *part1, *part64;
  CHECK(hipMalloc(&part1, parts * 4) == hipSuccess && hipMalloc(&part64, (size_t) kTok * parts * 4) == hipSuccess);
  // the dots: one token alone (few-token path) = the host's replay = that token among 64 (tiled path)
  CHECK(ops_hc_dot(dfn, dh + (size_t) kAt * len, hc, dim, 1, part1, nullptr));
  CHECK(ops_hc_dot(dfn, dh, hc, dim, kTok, part64, nullptr));
  const std::vector<float> p1 = host(part1, parts), p64 = host(part64, (size_t) kTok * parts);
  CHECK(std::memcmp(p1.data(), &p64[(size_t) kAt * parts], parts * 4) == 0);
  const uint32_t seg = len / OPS_HC_PARTS;
  const float* x = &h[(size_t) kAt * len];
  for (uint32_t r = 0; r <= nm; r++)
    for (uint32_t p = 0; p < OPS_HC_PARTS; p++) {
      float acc = 0;
      for (uint32_t i = p * seg; i < (p + 1) * seg; i++)
        acc = std::fmaf(r < nm ? fn[(size_t) r * len + i] : x[i], x[i], acc);
      CHECK(std::memcmp(&acc, &p1[r * OPS_HC_PARTS + p], 4) == 0);
    }
  // the mixes of all 64 tokens: the library's against the reference's
  const float eps = 1e-6f, hc_eps = 1e-6f;
  float *pre, *post, *comb, *rpre, *rpost, *rcomb;
  CHECK(hipMalloc(&pre, kTok * hc * 4) == hipSuccess && hipMalloc(&post, kTok * hc * 4) == hipSuccess &&
        hipMalloc(&comb, kTok * hc * hc * 4) == hipSuccess && hipMalloc(&rpre, kTok * hc * 4) == hipSuccess &&
        hipMalloc(&rpost, kTok * hc * 4) == hipSuccess && hipMalloc(&rcomb, kTok * hc * hc * 4) == hipSuccess);
  CHECK(ops_hc_mix(part64, dsc, db, hc, dim, eps, hc_eps, iters, kTok, pre, post, comb, nullptr));
  ref::mix<hc><<<kTok, ref::kWave>>>(part64, dsc, db, dim, eps, hc_eps, iters, rpre, rpost, rcomb);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  CHECK(host(pre, kTok * hc) == host(rpre, kTok * hc) && host(post, kTok * hc) == host(rpost, kTok * hc));
  const std::vector<float> c = host(comb, kTok * hc * hc), rc = host(rcomb, kTok * hc * hc);
  CHECK(std::memcmp(c.data(), rc.data(), c.size() * 4) == 0);
  for (float* p : {dfn, dh, dsc, db, part1, part64, pre, post, comb, rpre, rpost, rcomb})
    CHECK(hipFree(p) == hipSuccess);
  std::printf("  hc %u: the dots alone, replayed and among %u the same bits; %u tokens' pre, post and comb (%u "
              "iterations) the one-lane kernel's\n", hc, kTok, kTok, iters);
}

int main() {
  std::printf("[test_hc_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_hc_gpu] skipped: no GPU\n");
    return 77;
  }
  run<4>(5120, 20);  // the model's
  run<6>(1024, 7);   // hc^2 past a wave: the one-lane path
  std::printf("[test_hc_gpu] ok\n");
  return 0;
}
