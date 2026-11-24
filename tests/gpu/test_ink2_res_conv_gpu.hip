// ink2_attn_res_conv (kernels/ink2.hip: the output's conv made by the FFN residual's dots, two launches not three)
// against ink2_conv_keep then ink2_attn_res on the same inputs: every output the same bits - the residual's out, the
// partial state (r: the conv plus add) and the conv's tail rows - at 1 .. 64 tokens, 0 .. 8 earlier states, with and
// without add and a history, 2,304 and 6,144 wide; values with ties, zeros and large magnitudes. 77 never
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern "C" {
#include "osmos/gpu/ink2_ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

static float* up(const std::vector<float>& v) {
  float* d = nullptr;
  CHECK(hipMalloc((void**) &d, v.size() * 4 + 64) == hipSuccess);
  CHECK(hipMemcpy(d, v.data(), v.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  return d;
}
static float* zero(size_t n) {
  float* d = nullptr;
  CHECK(hipMalloc((void**) &d, n * 4 + 64) == hipSuccess && hipMemset(d, 0x7f, n * 4 + 64) == hipSuccess);
  return d;
}
static bool same(const float* a, const float* b, size_t n) {
  std::vector<float> x(n), y(n);
  CHECK(hipMemcpy(x.data(), a, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(y.data(), b, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return !std::memcmp(x.data(), y.data(), n * 4);
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_ink2_res_conv_gpu] start\n");
  std::mt19937 rng(1017);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  int cases = 0;
  const uint32_t taps = 4, maxb = 9;
  for (const uint32_t dim : {2304u, 6144u})
    for (const uint32_t n : {1u, 2u, 3u, 4u, 7u, 16u, 64u})
      for (const uint32_t nb : {0u, 1u, 3u, 8u})
        for (int flags = 0; flags < 4; flags++) {  // bit 0: add, bit 1: a history
          const bool add = flags & 1, hist = flags & 2;
          auto rnd = [&](size_t k, float s) {
            std::vector<float> v(k);
            for (auto& x : v) x = s * nd(rng);
            return v;
          };
          std::vector<float> x = rnd((size_t) n * dim, 1.0f), w = rnd((size_t) dim * taps, 0.5f);
          for (size_t i = 0; i < x.size(); i += 97) x[i] = i % 2 ? 0.0f : 3e4f;  // zeros and large values
          std::vector<float> h = rnd((size_t) (taps - 1) * dim, 1.0f), a = rnd((size_t) n * dim, 1.0f);
          std::vector<float> blocks = rnd((size_t) n * maxb * dim, 1.0f), q = rnd(dim, 0.2f);
          float *dx = up(x), *dw = up(w), *dh = hist ? up(h) : nullptr, *da = add ? up(a) : nullptr;
          float *db = up(blocks), *dq = up(q);
          float *r0 = zero((size_t) n * dim), *k0 = zero((size_t) n * dim), *o0 = zero((size_t) n * dim);
          float *r1 = zero((size_t) n * dim), *k1 = zero((size_t) n * dim), *o1 = zero((size_t) n * dim);
          float *s0 = zero(INK2_RES_SCRATCH), *s1 = zero(INK2_RES_SCRATCH);
          const float res = 1.0f, cs = 0.4472136f, eps = 1e-6f, scale = 0.0208f;
          CHECK(ink2_conv_keep(dx, n, dim, taps, dw, dh, res, cs, da, r0, k0, nullptr) &&
                ink2_attn_res(dq, 1, db, nb, maxb, r0, n, dim, eps, scale, o0, s0, nullptr));
          CHECK(ink2_attn_res_conv(dx, taps, dw, dh, res, cs, da, k1, dq, 1, db, nb, maxb, r1, n, dim, eps, scale, o1,
                                   s1, nullptr));
          CHECK(hipDeviceSynchronize() == hipSuccess);
          const bool ok = same(r0, r1, (size_t) n * dim) && same(k0, k1, (size_t) n * dim) &&
                          same(o0, o1, (size_t) n * dim);
          if (!ok) std::fprintf(stderr, "  dim %u n %u nb %u add %d hist %d: an output differs\n", dim, n, nb, add, hist);
          CHECK(ok);
          for (float* p : {dx, dw, dh, da, db, dq, r0, k0, o0, r1, k1, o1, s0, s1})
            if (p) (void) hipFree(p);
          cases++;
        }
  // past its split, or without the tail's room, it refuses (the two calls then)
  CHECK(!ink2_attn_res_conv((const float*) 1, 4, (const float*) 1, nullptr, 1, 1, nullptr, nullptr, (const float*) 1,
                            1, nullptr, 0, 9, (float*) 1, 1, 2304, 1e-6f, 1, (float*) 1, (float*) 1, nullptr));
  CHECK(!ink2_attn_res_conv((const float*) 1, 4, (const float*) 1, nullptr, 1, 1, nullptr, (float*) 1, (const float*) 1,
                            1, nullptr, 0, 9, (float*) 1, INK2_RES_SPLIT_TOKENS + 1, 2304, 1e-6f, 1, (float*) 1,
                            (float*) 1, nullptr));
  std::printf("  %d cases: the residual's out, r and the tail rows the two calls' bits\n", cases);
  std::printf("[test_ink2_res_conv_gpu] ok\n");
  return 0;
}
