// hy4_route (osmos/gpu/hy4_ops.h): a decode's expert tables built on the GPU, against the host's grouping (the step's
// until now: experts ascending, an expert's tokens ascending, its rows in chunks): random picks (each token's k
// distinct) at Hy4-flash's shape (512 experts, 16 picks) and others, every table bit for bit, the padding count 0 to
// n_tok x k; the experts' matrices from a table of their addresses (the membrane's); a pick out of range or twice in
// a token faults; shapes it does not take refused.
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
#include "osmos/gpu/hy4_ops.h"
}

#define CHECK(x)                                                           \
  do {                                                                     \
    if (!(x)) {                                                            \
      std::fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

struct Tables {
  std::vector<int32_t> gidx, sidx, cnt, xrow;
  std::vector<float> wsel;
  std::vector<uintptr_t> wptr;
};

static Tables host(const std::vector<int32_t>& idx, const std::vector<float>& w, uint32_t n, uint32_t k, uint32_t E,
                   uint32_t chunk, const uintptr_t base[3], const uint64_t bytes[3]) {
  const uint32_t nk = n * k;
  Tables t{std::vector<int32_t>(nk), std::vector<int32_t>(nk), std::vector<int32_t>(nk, 0),
           std::vector<int32_t>(nk, 0), std::vector<float>(nk), std::vector<uintptr_t>(3 * nk, 0)};
  uint32_t r = 0, e_n = 0;
  for (uint32_t e = 0; e < E; e++) {
    const uint32_t first = r;
    for (uint32_t tok = 0; tok < n; tok++)
      for (uint32_t s = 0; s < k; s++)
        if ((uint32_t) idx[tok * k + s] == e) t.gidx[r] = (int32_t) tok, t.sidx[tok * k + s] = (int32_t) r, t.wsel[r] = w[tok * k + s], r++;
    for (uint32_t c = first; c < r; c += chunk, e_n++) {
      t.cnt[e_n] = (int32_t) std::min(chunk, r - c), t.xrow[e_n] = (int32_t) c;
      for (int m = 0; m < 3; m++) t.wptr[m * nk + e_n] = base[m] + e * bytes[m];
    }
  }
  CHECK(r == nk);
  return t;
}

template <class T>
static T* dev(size_t n) {
  T* p = nullptr;
  CHECK(hipMalloc(&p, n * sizeof(T) + 64) == hipSuccess);
  CHECK(hipMemset(p, 0x5A, n * sizeof(T) + 64) == hipSuccess);  // garbage: every value the kernel's own
  return p;
}

template <class T>
static std::vector<T> down(const T* p, size_t n) {
  std::vector<T> v(n);
  CHECK(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess);
  return v;
}

// one case: the GPU's tables against the host's (fault_at >= 0: that pick spoiled, the fault expected instead)
static void check(std::mt19937& rng, uint32_t n, uint32_t k, uint32_t E, uint32_t chunk, int fault_kind) {
  const uint32_t nk = n * k;
  std::vector<int32_t> idx(nk);
  std::vector<float> w(nk);
  std::vector<int32_t> all(E);
  std::iota(all.begin(), all.end(), 0);
  const uint32_t hot = std::max<uint32_t>(k, E / 8);  // a crowded few: experts past one chunk
  for (uint32_t t = 0; t < n; t++) {
    std::shuffle(all.begin(), all.begin() + (rng() % 2 ? hot : E), rng);
    for (uint32_t s = 0; s < k; s++) idx[t * k + s] = all[s], w[t * k + s] = std::ldexp((float) (rng() % 4096 + 1), -12);
  }
  if (fault_kind == 1) idx[rng() % nk] = (int32_t) (E + rng() % 7);
  if (fault_kind == 2 && k > 1) {
    const uint32_t t = rng() % n;
    idx[t * k + 1] = idx[t * k];
  }
  const uintptr_t base[3] = {0x10000000ull, 0x20000000ull, 0x30000000ull};
  const uint64_t bytes[3] = {1u << 20, 3u << 18, 5u << 16};
  int32_t *d_idx = dev<int32_t>(nk), *gidx = dev<int32_t>(nk), *sidx = dev<int32_t>(nk), *cnt = dev<int32_t>(nk);
  int32_t *xrow = dev<int32_t>(nk), *fault = dev<int32_t>(1);
  float *d_w = dev<float>(nk), *wsel = dev<float>(nk);
  const void** wptr = dev<const void*>(3 * nk);
  CHECK(hipMemset(fault, 0, 4) == hipSuccess);
  CHECK(hipMemcpy(d_idx, idx.data(), nk * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(d_w, w.data(), nk * 4, hipMemcpyHostToDevice) == hipSuccess);
  // the experts' addresses as the membrane holds them: a table [gate, up, down][expert] (here each tensor's base and
  // an expert's stride: the host's tables' pointers)
  std::vector<const void*> tab_h(3 * (size_t) E);
  for (int q = 0; q < 3; q++)
    for (uint32_t e = 0; e < E; e++) tab_h[(size_t) q * E + e] = (const void*) (base[q] + (uint64_t) e * bytes[q]);
  const void** tab = dev<const void*>(3 * (size_t) E);
  CHECK(hipMemcpy(tab, tab_h.data(), tab_h.size() * sizeof(void*), hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hy4_route(d_idx, d_w, n, k, E, chunk, tab, gidx, sidx, wsel, cnt, xrow, wptr, fault, nullptr));
  CHECK(hipDeviceSynchronize() == hipSuccess);
  const int32_t f = down(fault, 1)[0];
  if (fault_kind) {
    CHECK(f == 1);
  } else {
    CHECK(f == 0);
    const Tables h = host(idx, w, n, k, E, chunk, base, bytes);
    CHECK(down(gidx, nk) == h.gidx && down(sidx, nk) == h.sidx && down(cnt, nk) == h.cnt);
    const std::vector<float> g_w = down(wsel, nk);
    CHECK(std::memcmp(g_w.data(), h.wsel.data(), nk * 4) == 0);
    const std::vector<int32_t> g_x = down(xrow, nk);
    const std::vector<const void*> g_p = down(wptr, 3 * nk);
    for (uint32_t e = 0; e < nk; e++)  // a skipped entry's row and pointers unread: the used ones' alone
      if (h.cnt[e]) {
        CHECK(g_x[e] == h.xrow[e]);
        for (int m = 0; m < 3; m++) CHECK((uintptr_t) g_p[m * nk + e] == h.wptr[m * nk + e]);
      }
  }
  for (void* p : {(void*) d_idx, (void*) gidx, (void*) sidx, (void*) cnt, (void*) xrow, (void*) fault, (void*) d_w,
                  (void*) wsel, (void*) wptr, (void*) tab})
    CHECK(hipFree(p) == hipSuccess);
}

int main() {
  int devs = 0;
  if (hipGetDeviceCount(&devs) != hipSuccess || devs == 0) {
    std::printf("[test_hy4_route_gpu] skipped: no GPU\n");
    return 77;
  }
  const void** t = nullptr;
  int32_t* p = nullptr;
  CHECK(!hy4_route(p, nullptr, 1, 0, 512, 4, t, p, p, nullptr, p, p, nullptr, p, nullptr));
  CHECK(!hy4_route(p, nullptr, 1, HY4_ROUTE_MAX_K + 1, 512, 4, t, p, p, nullptr, p, p, nullptr, p, nullptr));
  CHECK(!hy4_route(p, nullptr, 1, 16, HY4_ROUTE_MAX_EXPERTS + 1, 4, t, p, p, nullptr, p, p, nullptr, p, nullptr));
  CHECK(!hy4_route(p, nullptr, 1, 16, 512, 0, t, p, p, nullptr, p, p, nullptr, p, nullptr));
  CHECK(!hy4_route(p, nullptr, 0, 16, 512, 4, t, p, p, nullptr, p, p, nullptr, p, nullptr));
  std::mt19937 rng(20261118);
  uint32_t cases = 0;
  for (uint32_t n : {1u, 2u, 3u, 9u, 64u, 65u, 300u, 2600u})
    for (uint32_t chunk : {1u, 4u, 16u})
      for (auto [k, E] : {std::pair<uint32_t, uint32_t>{16, 512}, {8, 64}, {1, 3}, {64, 1024}})
        check(rng, n, k, E, chunk, 0), cases++;
  for (int kind : {1, 2})
    for (uint32_t n : {1u, 7u, 300u}) check(rng, n, 16, 512, 4, kind), cases++;
  std::printf("[test_hy4_route_gpu] %u cases: the host's tables bit for bit; a bad pick faults\n", cases);
  std::printf("[test_hy4_route_gpu] ok\n");
  return 0;
}
