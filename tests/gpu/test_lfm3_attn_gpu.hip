// q35_attn_decode at LFM3's shape (DESIGN.md, "vibes-flash: LFM3"): 16 query heads on 4 KV heads of 128, a gate a
// head (AttnShape's gate_span the dim), a 256-key window or every key, rows by position or a ring - against a
// double-precision attention over the cache's own Q8_0 values (read back, decoded here: only the order of the sums
// differs), steps of 1 to 4 tokens past the window's reach; a token's bits the same alone or in its step; Qwen's
// shape (NULL) untouched (test_q35_attn_gpu); shapes it does not take refused. skipped (77) without a GPU
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/attn_shape.h"
#include "osmos/gpu/q35_ops.h"
#include "osmos/gpu/q4x_ops.h"
}

#define CHECK(c)                                                                  \
  do {                                                                            \
    if (!(c)) {                                                                   \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

constexpr uint32_t H = 16, KV = 4, D = 128, G = H / KV, kq = H * D, kkv = KV * D;
constexpr uint32_t LFM3_ATTN_LONG = 2097152;

static uint32_t rng = 1711;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / 16777216.0f - 0.5f;
}

static float f16(uint16_t h) {
  _Float16 v;
  std::memcpy(&v, &h, 2);
  return (float) v;
}

// a cache row's value i (its KV heads' dims in order): the Q8_0 block's scale times its int8
static double cached(const std::vector<uint8_t>& c, uint64_t row_bytes, uint64_t row, uint32_t i) {
  const uint8_t* b = c.data() + row * row_bytes + (i / Q4X_KV_BLOCK) * Q4X_KV_BLOCK_BYTES;
  uint16_t d;
  std::memcpy(&d, b, 2);
  return (double) f16(d) * (double) (int8_t) b[2 + i % Q4X_KV_BLOCK];
}

static void one(uint32_t pos0, uint32_t n, uint32_t window, uint32_t slots) {
  const uint32_t ctx = pos0 + n, rows = slots ? slots : ctx;
  const uint64_t rb = q4x_kv_row_bytes(kkv);
  std::vector<float> hq((size_t) n * kq), hg((size_t) n * H), hk((size_t) ctx * kkv), hv((size_t) ctx * kkv);
  for (auto& x : hq) x = 4.0f * uniform();
  for (auto& x : hg) x = 2.0f * uniform();
  for (auto& x : hk) x = 4.0f * uniform();
  for (auto& x : hv) x = uniform();
  void *kc, *vc;
  float *dk, *dv, *dq, *dg, *da, *out, *one_out;
  CHECK(hipMalloc(&kc, rows * rb) == hipSuccess && hipMalloc(&vc, rows * rb) == hipSuccess);
  CHECK(hipMalloc((void**) &dk, hk.size() * 4) == hipSuccess && hipMalloc((void**) &dv, hv.size() * 4) == hipSuccess);
  CHECK(hipMalloc((void**) &dq, hq.size() * 4) == hipSuccess && hipMalloc((void**) &dg, hg.size() * 4) == hipSuccess);
  CHECK(hipMalloc((void**) &out, hq.size() * 4) == hipSuccess && hipMalloc((void**) &one_out, kq * 4) == hipSuccess);
  CHECK(hipMalloc((void**) &da, q35_attn_decode_scratch(H, Q35_ATTN_DECODE_MAX) * 4) == hipSuccess);
  CHECK(hipMemcpy(dk, hk.data(), hk.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dv, hv.data(), hv.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dg, hg.data(), hg.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  // the cache: position p's row at p (slots 0) or p % slots - written in position order, a ring's later ones over its
  // earlier ones, as a sequence writes it
  if (!slots)  // rows by position: the cache in one put
    CHECK(q4x_kv_put(dk, dv, ctx, kkv, kc, vc, nullptr));
  for (uint32_t p = 0; slots && p < ctx; p++) {
    const uint64_t r = p % slots;
    CHECK(q4x_kv_put(dk + (size_t) p * kkv, dv + (size_t) p * kkv, 1, kkv, (uint8_t*) kc + r * rb,
                     (uint8_t*) vc + r * rb, nullptr));
  }
  const AttnShape shape = {D, window, slots};
  CHECK(q35_attn_decode(dq, kc, vc, n, H, KV, D, &shape, pos0, dg, da, out, nullptr));
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<uint8_t> ck(rows * rb), cv(rows * rb);
  std::vector<float> got(hq.size());
  CHECK(hipMemcpy(ck.data(), kc, ck.size(), hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(cv.data(), vc, cv.size(), hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(got.data(), out, got.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  double worst = 0, top = 0;
  for (uint32_t t = 0; t < n; t++) {
    const uint32_t p = pos0 + t, lo = window && p + 1 > window ? p + 1 - window : 0;
    for (uint32_t h = 0; h < H; h++) {
      const uint32_t g = h / G;
      std::vector<double> s(p + 1 - lo);
      double mx = -INFINITY;
      for (uint32_t k = lo; k <= p; k++) {
        double a = 0;
        for (uint32_t d = 0; d < D; d++)
          a += (double) hq[((size_t) t * H + h) * D + d] * cached(ck, rb, slots ? k % slots : k, g * D + d);
        s[k - lo] = a / std::sqrt((double) D), mx = std::fmax(mx, s[k - lo]);
      }
      double sum = 0;
      for (double& x : s) sum += (x = std::exp(x - mx));
      const double gate = 1.0 / (1.0 + std::exp(-(double) hg[(size_t) t * H + h]));
      for (uint32_t d = 0; d < D; d++) {
        double o = 0;
        for (uint32_t k = lo; k <= p; k++) o += s[k - lo] * cached(cv, rb, slots ? k % slots : k, g * D + d);
        const double want = o / sum * gate, e = std::fabs(want - got[((size_t) t * H + h) * D + d]);
        worst = std::fmax(worst, e), top = std::fmax(top, std::fabs(want));
      }
    }
  }
  // a token's bits alone: the step's last alone, its own position, the same keys
  CHECK(q35_attn_decode(dq + (size_t) (n - 1) * kq, kc, vc, 1, H, KV, D, &shape, pos0 + n - 1,
                        dg + (size_t) (n - 1) * H, da, one_out, nullptr));
  std::vector<float> alone(kq);
  CHECK(hipMemcpy(alone.data(), one_out, kq * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(std::memcmp(alone.data(), got.data() + (size_t) (n - 1) * kq, kq * 4) == 0);
  std::printf("  %u tokens at %u, window %u, %s: max error %.2e of %.3f (%.1e); the last alone the same bits\n", n,
              pos0, window, slots ? "a ring" : "rows by position", worst, top, worst / top);
  CHECK(worst <= 1e-5 * top);
  for (void* x : {kc, vc, (void*) dk, (void*) dv, (void*) dq, (void*) dg, (void*) da, (void*) out, (void*) one_out})
    CHECK(hipFree(x) == hipSuccess);
}

// q4x_attn (a prompt's attention on the matrix cores, f16 q, k, v and weights) at the same shape over a step of T
// tokens, against the decode's F32 attention a token at a time over the same cache - the gap the f16's; and the step
// in two pieces the bits of the whole
static void prompt(uint32_t pos0, uint32_t T, uint32_t window, uint32_t slots) {
  const uint32_t ctx = pos0 + T, rows = slots ? slots : ctx;
  const uint64_t rb = q4x_kv_row_bytes(kkv);
  std::vector<float> hq((size_t) T * kq), hg((size_t) T * H), hk((size_t) ctx * kkv), hv((size_t) ctx * kkv);
  for (auto& x : hq) x = 4.0f * uniform();
  for (auto& x : hg) x = 2.0f * uniform();
  for (auto& x : hk) x = 4.0f * uniform();
  for (auto& x : hv) x = uniform();
  void *kc, *vc;
  float *dk, *dv, *dq, *dg, *fa, *da, *out, *out2, *ref;
  CHECK(hipMalloc(&kc, rows * rb) == hipSuccess && hipMalloc(&vc, rows * rb) == hipSuccess);
  CHECK(hipMalloc((void**) &dk, hk.size() * 4) == hipSuccess && hipMalloc((void**) &dv, hv.size() * 4) == hipSuccess);
  CHECK(hipMalloc((void**) &dq, hq.size() * 4) == hipSuccess && hipMalloc((void**) &dg, hg.size() * 4) == hipSuccess);
  CHECK(hipMalloc((void**) &out, hq.size() * 4) == hipSuccess && hipMalloc((void**) &out2, hq.size() * 4) == hipSuccess &&
        hipMalloc((void**) &ref, hq.size() * 4) == hipSuccess);
  CHECK(hipMalloc((void**) &fa, q4x_attn_scratch(H, ctx) * 4) == hipSuccess);
  CHECK(hipMalloc((void**) &da, q35_attn_decode_scratch(H, Q35_ATTN_DECODE_MAX) * 4) == hipSuccess);
  CHECK(hipMemcpy(dk, hk.data(), hk.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dv, hv.data(), hv.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dg, hg.data(), hg.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  // the cache as a step writes it: the prefix's rows, then the step's - a ring's later rows over its earlier ones only
  // past the step's own reach (slots at least window + T - 1)
  if (!slots)  // rows by position: the cache in one put
    CHECK(q4x_kv_put(dk, dv, ctx, kkv, kc, vc, nullptr));
  for (uint32_t p = 0; slots && p < ctx; p++) {
    const uint64_t r = p % slots;
    CHECK(q4x_kv_put(dk + (size_t) p * kkv, dv + (size_t) p * kkv, 1, kkv, (uint8_t*) kc + r * rb,
                     (uint8_t*) vc + r * rb, nullptr));
  }
  const AttnShape shape = {D, window, slots};
  CHECK(q4x_attn(dq, kc, vc, T, H, KV, D, pos0, dg, nullptr, &shape, fa, out, nullptr, nullptr));
  const uint32_t a = T / 3 + 1;
  CHECK(q4x_attn(dq, kc, vc, a, H, KV, D, pos0, dg, nullptr, &shape, fa, out2, nullptr, nullptr));
  CHECK(q4x_attn(dq + (size_t) a * kq, kc, vc, T - a, H, KV, D, pos0 + a, dg + (size_t) a * H, nullptr, &shape, fa,
                 out2 + (size_t) a * kq, nullptr, nullptr));
  for (uint32_t t = 0; t < T; t++)
    CHECK(q35_attn_decode(dq + (size_t) t * kq, kc, vc, 1, H, KV, D, &shape, pos0 + t, dg + (size_t) t * H, da,
                          ref + (size_t) t * kq, nullptr));
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<float> g(hq.size()), g2(hq.size()), r(hq.size());
  CHECK(hipMemcpy(g.data(), out, g.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(g2.data(), out2, g2.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(r.data(), ref, r.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  double num = 0, den = 0, worst = 0;
  for (size_t i = 0; i < g.size(); i++) {
    num += ((double) g[i] - r[i]) * ((double) g[i] - r[i]), den += (double) r[i] * r[i];
    worst = std::fmax(worst, std::fabs((double) g[i] - r[i]));
  }
  CHECK(std::memcmp(g.data(), g2.data(), g.size() * 4) == 0);
  std::printf("  a prompt of %u at %u, window %u, %s: against the decode's F32 relative %.2e, largest gap %.2e; in "
              "pieces of %u and %u the same bits\n", T, pos0, window, slots ? "a ring" : "rows by position",
              std::sqrt(num / den), worst, a, T - a);
  CHECK(std::sqrt(num / den) < 5e-3);  // test_q35_attn_gpu's bound: the f16's
  for (void* x : {kc, vc, (void*) dk, (void*) dv, (void*) dq, (void*) dg, (void*) fa, (void*) da, (void*) out,
                  (void*) out2, (void*) ref})
    CHECK(hipFree(x) == hipSuccess);
}

int main() {
  std::printf("[test_lfm3_attn_gpu] start\n");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_lfm3_attn_gpu] skipped: no GPU\n");
    return 77;
  }
  float x[1];
  const AttnShape span3 = {3, 0, 0}, ring_short = {D, 256, 256}, ring_no_window = {D, 0, 512};
  CHECK(!q35_attn_decode(x, x, x, 1, H, KV, D, &span3, 0, x, x, x, nullptr));         // a gate neither 1 nor a head
  CHECK(!q35_attn_decode(x, x, x, 2, H, KV, D, &ring_short, 300, x, x, x, nullptr));  // a ring under window + step - 1
  CHECK(!q35_attn_decode(x, x, x, 1, H, KV, D, &ring_no_window, 0, x, x, x, nullptr)); // a ring without a window
  CHECK(!q35_attn_decode(x, x, x, 1, H, KV, 64, nullptr, 0, x, x, x, nullptr));       // a head neither 128 nor 256
  std::printf("  refusals: a gate's span, a ring too short or without a window, a head's dim\n");
  for (uint32_t n : {1u, 3u, 4u}) {
    one(700, n, 256, 0);
    one(700, n, 256, 256 + 3);
    one(700, n, 0, 0);
  }
  one(5, 4, 256, 0);  // within the window: every key
  prompt(0, 300, 256, 0);
  prompt(700, 300, 256, 256 + 300 + 8);
  prompt(700, 300, 0, 0);
  prompt(5000, 64, 256, 256 + 64 + 8);
  prompt(LFM3_ATTN_LONG - 64, 64, 0, 0);  // a 2M context's last keys (vibes1.7b's pack: past Qwen's 1M)
  std::printf("[test_lfm3_attn_gpu] ok\n");
  return 0;
}
