// q8_experts on route_tok's tables (osmos/gpu/q8.h, osmos/gpu/route.h): a step's routed experts three ways, each pick's
// output row byte for byte - exact (a launch of count 1, one of the rest at the step's width), by_count (0.18.1: a
// launch a range of counts at its most, over the entries that can hold it) and each token alone (a step of one: no
// expert shared) - at 1 .. 8 tokens, picks crowded onto a few hot experts (counts through every range, up to every
// token's), and every token the same experts (a range's entries as many as its grid bounds: one fewer fails), at Hy4's shape and types (Q3_K gate / up, Q4_K down) and Qwen's (IQ3_XXS, IQ4_NL down over 640 columns).
// random weights (their bits alone: the same computation the same bits, NaN or not).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

extern "C" {
#include "osmos/gpu/mv.h"
#include "osmos/gpu/q8.h"
#include "osmos/gpu/route.h"
#include "osmos/model/qtype.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

template <class T>
static T* dev(size_t n) {
  T* p = nullptr;
  CHECK(hipMalloc((void**) &p, n * sizeof(T) + 256) == hipSuccess);
  CHECK(hipMemset(p, 0, n * sizeof(T) + 256) == hipSuccess);
  return p;
}

template <class T>
static std::vector<T> down(const T* p, size_t n) {
  std::vector<T> v(n);
  CHECK(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess);
  return v;
}

struct Shape {
  const char* name;
  uint32_t d, ff, gate_up, down_type;
};

constexpr uint32_t kExperts = 64, kK = 8, kHot = 10;

// a step's tables (route_tok) and its experts (q8_experts, by_count or exact): each pick's output row (sidx's)
static std::vector<float> run(const Shape& S, const void* const* tab, const std::vector<int32_t>& idx,
                              const std::vector<float>& w, const float* x, uint32_t n, bool by_count) {
  const uint32_t nk = n * kK, chunk = n < MV_MAX_TOKENS ? n : MV_MAX_TOKENS;
  int32_t *d_idx = dev<int32_t>(nk), *gidx = dev<int32_t>(nk), *sidx = dev<int32_t>(nk), *cnt = dev<int32_t>(nk),
          *xrow = dev<int32_t>(nk), *fault = dev<int32_t>(1);
  float *d_w = dev<float>(nk), *wsel = dev<float>(nk), *h = dev<float>((size_t) nk * S.ff), *y = dev<float>((size_t) nk * S.d),
        *xqd = dev<float>((size_t) n * S.d / 32), *hqd = dev<float>((size_t) nk * S.ff / 32);
  int8_t *xq = dev<int8_t>((size_t) n * S.d), *hq = dev<int8_t>((size_t) nk * S.ff);
  const void** wptr = dev<const void*>(3 * (size_t) nk);
  CHECK(hipMemcpy(d_idx, idx.data(), nk * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(d_w, w.data(), nk * 4, hipMemcpyHostToDevice) == hipSuccess);
  const Q8Experts e = {.act = Q8_ACT_LANE, .gate_type = S.gate_up, .up_type = S.gate_up, .down_type = S.down_type,
                       .wg = wptr, .wu = wptr + nk, .wd = wptr + 2 * nk, .cnt = cnt, .xrow = xrow, .xin = gidx,
                       .n = nk, .max_cnt = chunk, .h_cnt = nullptr, .by_count = by_count, .d = S.d, .ff = S.ff,
                       .x = x, .x_rows = n, .xq = xq, .xqd = xqd, .scale = wsel, .win = nullptr, .lim = 7.0f, .h = h,
                       .h_rows = nk, .hq = hq, .hqd = hqd, .y = y};
  CHECK(route_tok(d_idx, d_w, n, kK, kExperts, chunk, tab, gidx, sidx, wsel, cnt, xrow, wptr, fault, nullptr));
  CHECK(q8_experts(&e, nullptr));
  CHECK(hipDeviceSynchronize() == hipSuccess && down(fault, 1)[0] == 0);
  const std::vector<int32_t> rows = down(sidx, nk);
  const std::vector<float> all = down(y, (size_t) nk * S.d);
  std::vector<float> out((size_t) nk * S.d);
  for (uint32_t m = 0; m < nk; m++)
    std::memcpy(out.data() + (size_t) m * S.d, all.data() + (size_t) rows[m] * S.d, S.d * 4);
  for (void* p : {(void*) d_idx, (void*) gidx, (void*) sidx, (void*) cnt, (void*) xrow, (void*) fault, (void*) d_w,
                  (void*) wsel, (void*) h, (void*) y, (void*) xqd, (void*) hqd, (void*) xq, (void*) hq, (void*) wptr})
    CHECK(hipFree(p) == hipSuccess);
  return out;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  int devs = 0;
  if (hipGetDeviceCount(&devs) != hipSuccess || !devs) return std::printf("[test_q8_experts_gpu] skipped: no GPU\n"), 77;
  const Shape shapes[] = {{"Hy4 (Q3_K / Q4_K)", 1536, 1536, QTYPE_Q3_K, QTYPE_Q4_K},
                          {"Qwen (IQ3_XXS / IQ4_NL)", 2560, 640, QTYPE_IQ3_XXS, QTYPE_IQ4_NL}};
  std::mt19937 rng(20261126);
  uint32_t steps = 0;
  for (const Shape& S : shapes) {
    CHECK(q8_takes(S.gate_up) && q8_takes(S.down_type));
    const QTypeInfo *gu = qtype_info(S.gate_up), *dn = qtype_info(S.down_type);
    const uint64_t gb = (uint64_t) S.d / gu->values * gu->bytes * S.ff, db = (uint64_t) S.ff / dn->values * dn->bytes * S.d;
    std::vector<uint8_t> hw(2 * gb + db);
    uint8_t* wts = dev<uint8_t>((2 * gb + db) * kExperts);
    for (uint32_t e = 0; e < kExperts; e++) {
      for (auto& b : hw) b = (uint8_t) (rng() >> 24);
      CHECK(hipMemcpy(wts + e * hw.size(), hw.data(), hw.size(), hipMemcpyHostToDevice) == hipSuccess);
    }
    std::vector<const void*> th(3 * kExperts);
    for (uint32_t e = 0; e < kExperts; e++) {
      const uint8_t* base = wts + e * hw.size();
      th[e] = base, th[kExperts + e] = base + gb, th[2 * kExperts + e] = base + 2 * gb;
    }
    const void** tab = dev<const void*>(th.size());
    CHECK(hipMemcpy(tab, th.data(), th.size() * sizeof(void*), hipMemcpyHostToDevice) == hipSuccess);
    std::vector<float> hx((size_t) MV_MAX_TOKENS * S.d);
    for (auto& v : hx) v = std::ldexp((float) ((int32_t) (rng() % 4001) - 2000), -10);
    float* x = dev<float>(hx.size());
    CHECK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
    for (uint32_t n = 1; n <= MV_MAX_TOKENS; n++)
      for (int rep = 0; rep < 4; rep++, steps++) {
        // each token's kK distinct picks, most from the hot few (counts up to n), the rest anywhere
        std::vector<int32_t> idx(n * kK), pool(kExperts);
        std::vector<float> w(n * kK);
        for (uint32_t t = 0; t < n; t++) {
          std::iota(pool.begin(), pool.end(), 0);
          std::shuffle(pool.begin(), pool.begin() + kHot, rng);
          std::shuffle(pool.begin() + kHot, pool.end(), rng);
          const uint32_t hot = rep == 2 ? kK : kK / 2 + rng() % (kK / 2);  // (rep 2: every pick hot)
          for (uint32_t j = 0; j < kK; j++) idx[t * kK + j] = j < hot ? pool[j] : pool[kHot + j];
          if (rep == 3)  // every token the same kK experts: each a count of n, as many as a range's bound holds
            for (uint32_t j = 0; j < kK; j++) idx[t * kK + j] = (int32_t) (j * 7 % kExperts);
          for (uint32_t j = 0; j < kK; j++) w[t * kK + j] = std::ldexp((float) (rng() % 1024 + 1), -10);
        }
        const std::vector<float> exact = run(S, tab, idx, w, x, n, false), by = run(S, tab, idx, w, x, n, true);
        CHECK(std::memcmp(exact.data(), by.data(), exact.size() * 4) == 0);
        for (uint32_t t = 0; t < n; t++) {  // each token alone: no expert shared, the same row
          const std::vector<int32_t> it(idx.begin() + t * kK, idx.begin() + (t + 1) * kK);
          const std::vector<float> iw(w.begin() + t * kK, w.begin() + (t + 1) * kK);
          const std::vector<float> one = run(S, tab, it, iw, x + (size_t) t * S.d, 1, true);
          CHECK(std::memcmp(one.data(), by.data() + (size_t) t * kK * S.d, one.size() * 4) == 0);
        }
      }
    std::printf("  %s: 1 .. %u tokens, picks crowded on %u of %u experts - by_count's rows exact's and each token's "
                "alone, bit for bit\n", S.name, MV_MAX_TOKENS, kHot, kExperts);
    CHECK(hipFree(tab) == hipSuccess && hipFree(wts) == hipSuccess && hipFree(x) == hipSuccess);
  }
  std::printf("[test_q8_experts_gpu] %u steps ok\n", steps);
  return 0;
}
