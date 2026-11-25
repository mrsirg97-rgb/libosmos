// expert_bench: a decode's routed experts two ways, at Hy4-flash's shape - 16 experts of 1536 x 1536 a token (its
// gate / up / down's), each call 16 others of a pool of 512 (cold: never the last call's) - Hy4's matvec
// (mv_group_tok: F32 x, a row's dot a thread group's) against DeepSeek's matrix cores (mm_group: f16 x, a tile's 16
// rows - one used); then the latent MoE's dense matvecs, short rows against long (routed_up 3072 x 1536,
// routed_down 1536 x 3072, Q8_0; cold: rotated through 64 copies). Q3_K and Q4_K (Hy4's pack mixes them by layer).
// microseconds a call and the weights' GB/s (each expert's bytes read once a call). random weights: speed alone.
//   expert_bench [iters]
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
#include "osmos/gpu/mm.h"
#include "osmos/gpu/mv.h"
#include "osmos/model/gguf.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

static uint32_t g_rng = 1017;
static uint32_t next() { return g_rng = g_rng * 1664525u + 1013904223u; }

template <class T>
static T* dev(size_t n) {
  T* p = nullptr;
  CHECK(hipMalloc((void**) &p, n * sizeof(T) + 256) == hipSuccess);
  return p;
}
// n bytes of noise on the device (the weights: their values do not matter here)
static uint8_t* noise(size_t n) {
  std::vector<uint8_t> h(n);
  for (size_t i = 0; i < n; i++) h[i] = (uint8_t) (next() >> 13);
  uint8_t* d = dev<uint8_t>(n);
  CHECK(hipMemcpy(d, h.data(), n, hipMemcpyHostToDevice) == hipSuccess);
  return d;
}
static float* floats(size_t n, float s) {
  std::vector<float> h(n);
  for (auto& v : h) v = s * ((float) (next() >> 8) / 16777216.0f - 0.5f);
  float* d = dev<float>(n);
  CHECK(hipMemcpy(d, h.data(), n * 4, hipMemcpyHostToDevice) == hipSuccess);
  return d;
}

template <class F>
static double time_us(uint32_t iters, F&& call) {
  for (int i = 0; i < 5; i++) CHECK(call(i));
  CHECK(hipDeviceSynchronize() == hipSuccess);
  const auto t0 = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < iters; i++) CHECK(call(i));
  CHECK(hipDeviceSynchronize() == hipSuccess);
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
}

int main(int argc, char** argv) {
  const uint32_t iters = argc > 1 ? (uint32_t) atoi(argv[1]) : 400;
  const uint32_t R = 1536, C = 1536, E = 512, K = 16;
  struct Ty { const char* name; uint32_t type; } types[] = {{"Q3_K", 11}, {"Q4_K", 12}};
  float* x = floats((size_t) 256 * C, 1.0f);  // the picks' rows: row m at 16 m (each its own tile for mm_group)
  float* y = dev<float>((size_t) 256 * R);
  void* x16 = dev<uint16_t>((size_t) 256 * C);
  int32_t* ovf = dev<int32_t>(1);
  CHECK(hipMemset(ovf, 0, 4) == hipSuccess && mm_x16(x, C, 256, C, x16, ovf, nullptr));
  const void** dw = dev<const void*>(K);
  int32_t *dcnt = dev<int32_t>(K), *dxrow = dev<int32_t>(K);
  uint32_t *dxo = dev<uint32_t>(K), *dyo = dev<uint32_t>(K);
  std::vector<int32_t> cnt(K, 1), xrow(K);
  std::vector<uint32_t> xo(K), yo(K);
  for (uint32_t m = 0; m < K; m++) xrow[m] = (int32_t) (16 * m), xo[m] = 16 * m * C, yo[m] = 16 * m * R;
  CHECK(hipMemcpy(dcnt, cnt.data(), K * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dxrow, xrow.data(), K * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dxo, xo.data(), K * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dyo, yo.data(), K * 4, hipMemcpyHostToDevice) == hipSuccess);
  std::printf("16 experts of %u x %u a call (a decode token's gate, up or down), cold\n", R, C);
  for (const Ty& ty : types) {
    const uint64_t mb = gguf_bytes(ty.type, (uint64_t) R * C);
    uint8_t* pool = noise(mb * E);
    // each call's 16: a window of the pool that moves 16 on (never the last call's)
    std::vector<std::vector<const void*>> picks(E / K);
    for (uint32_t c = 0; c < E / K; c++)
      for (uint32_t m = 0; m < K; m++) picks[c].push_back(pool + (uint64_t) (c * K + m) * mb);
    const auto set = [&](uint32_t i) {
      return hipMemcpyAsync(dw, picks[i % picks.size()].data(), K * sizeof(void*), hipMemcpyHostToDevice, nullptr) ==
             hipSuccess;
    };
    const double mv = time_us(iters, [&](uint32_t i) {
      return set(i) && mv_group_tok(ty.type, dw, dcnt, dxrow, K, 1, R, C, x, C, y, R, nullptr);
    });
    const double mm = time_us(iters, [&](uint32_t i) {
      return set(i) && mm_group(ty.type, dw, dcnt, K, 1, R, C, x16, dxo, C, y, dyo, R, nullptr);
    });
    const double gb = (double) mb * K / 1e3;
    std::printf("  %s: mv_group_tok %7.1f us (%5.1f GB/s) | mm_group %7.1f us (%5.1f GB/s)\n", ty.name, mv, gb / mv, mm,
                gb / mm);
    // the same experts, each one token, launched as a verify's table bound (max_cnt 2 .. 6): the cost of the rows past
    // an expert's count (summed, not stored)
    std::printf("   one token each, max_cnt:");
    for (uint32_t mc = 2; mc <= 6; mc++) {
      const double us = time_us(iters, [&](uint32_t i) {
        return set(i) && mv_group_tok(ty.type, dw, dcnt, dxrow, K, mc, R, C, x, C, y, R, nullptr);
      });
      std::printf(" %u: %.1f us", mc, us);
    }
    std::printf("\n");
    CHECK(hipFree(pool) == hipSuccess);
  }
  // a prompt step's experts: every one of the 512 with 32 tokens (a 1,024-token step's 16,384 picks), 1536 x 1536, on
  // the matrix cores (mm_group: f16 x, its tiles) - Q3_K, Q4_K, and Q8_0 (its decode near free: the K-quants' cost)
  std::printf("a prompt step's experts: 16,384 (expert, token) pairs of 1536 x 1536 (mm_group), warm\n");
  for (uint32_t PT : {16u, 32u, 64u, 128u}) {
    const uint32_t PE = 16384 / PT;
    std::printf(" %u experts x %u tokens\n", PE, PT);
    void* px16 = dev<uint16_t>((size_t) PE * PT * C);
    float* pxf = floats((size_t) PE * PT * C, 1.0f);
    for (uint32_t r0 = 0; r0 < PE * PT; r0 += MM_MAX_TOKENS)  // (mm_x16's rows at most MM_MAX_TOKENS a call: tiles whole)
      CHECK(mm_x16(pxf + (size_t) r0 * C, C, MM_MAX_TOKENS, C, (uint16_t*) px16 + (size_t) r0 * C, ovf, nullptr));
    float* py = dev<float>((size_t) PE * PT * R);
    const void** pw = dev<const void*>(PE);
    int32_t* pc = dev<int32_t>(PE);
    uint32_t *pxo = dev<uint32_t>(PE), *pyo = dev<uint32_t>(PE);
    std::vector<int32_t> c32(PE, (int32_t) PT);
    std::vector<uint32_t> xo2(PE), yo2(PE);
    for (uint32_t m = 0; m < PE; m++) xo2[m] = m * PT * C, yo2[m] = m * PT * R;  // (PT a multiple of 16: tiles whole)
    CHECK(hipMemcpy(pc, c32.data(), PE * 4, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(hipMemcpy(pxo, xo2.data(), PE * 4, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(hipMemcpy(pyo, yo2.data(), PE * 4, hipMemcpyHostToDevice) == hipSuccess);
    for (const Ty& ty : {Ty{"Q3_K", 11}, Ty{"Q4_K", 12}, Ty{"Q8_0", 8}}) {
      const uint64_t mb = gguf_bytes(ty.type, (uint64_t) R * C);
      uint8_t* pool = noise(mb * PE);
      std::vector<const void*> ptr(PE);
      for (uint32_t m = 0; m < PE; m++) ptr[m] = pool + (uint64_t) m * mb;
      CHECK(hipMemcpy(pw, ptr.data(), PE * sizeof(void*), hipMemcpyHostToDevice) == hipSuccess);
      const double us = time_us(iters / 20 + 5, [&](uint32_t) {
        return mm_group(ty.type, pw, pc, PE, PT, R, C, px16, pxo, C, py, pyo, R, nullptr);
      });
      const double tf = 2.0 * PE * PT * R * C / (us * 1e6);
      std::printf("  %s: %8.1f us (%5.2f TFLOP/s, %5.1f GB/s of weights)\n", ty.name, us, tf, (double) mb * PE / 1e3 / us);
      CHECK(hipFree(pool) == hipSuccess);
    }
  }
  // MLA's absorptions: a head's own Q8_0 matrix on its own row of x, every head at once (k_b: 512 x 128, v_b: 128 x
  // 512), 64 heads a call, rotated through 43 layers' sets
  std::printf("MLA's absorptions: 64 heads' Q8_0 matrices a call (one token), cold\n");
  for (auto [rows, cols, what] : {std::tuple<uint32_t, uint32_t, const char*>{512, 128, "k_b (latent x nope)"},
                                  {128, 512, "v_b (v_dim x latent)"}}) {
    const uint32_t H = 64, L = 43;
    const uint64_t mb = gguf_bytes(8, (uint64_t) rows * cols);
    uint8_t* pool = noise(mb * H * L);
    float *xa = floats((size_t) H * cols, 1.0f), *ya = dev<float>((size_t) H * rows);
    const void** hw = dev<const void*>(H * L);
    int32_t *hc = dev<int32_t>(H), *hx = dev<int32_t>(H);
    std::vector<const void*> ptr(H * L);
    std::vector<int32_t> one(H, 1), xr(H);
    for (uint32_t i = 0; i < H * L; i++) ptr[i] = pool + (uint64_t) i * mb;
    for (uint32_t h = 0; h < H; h++) xr[h] = (int32_t) h;
    CHECK(hipMemcpy(hw, ptr.data(), ptr.size() * sizeof(void*), hipMemcpyHostToDevice) == hipSuccess);
    CHECK(hipMemcpy(hc, one.data(), H * 4, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(hipMemcpy(hx, xr.data(), H * 4, hipMemcpyHostToDevice) == hipSuccess);
    const double us = time_us(iters, [&](uint32_t i) {
      return mv_group_tok(8, hw + (uint64_t) (i % L) * H, hc, hx, H, 1, rows, cols, xa, cols, ya, rows, nullptr);
    });
    std::printf("  %4u x %4u %s: %6.1f us (%5.1f GB/s)\n", rows, cols, what, us, (double) mb * H / 1e3 / us);
    CHECK(hipFree(pool) == hipSuccess);
  }
  // the dense matvecs, Q8_0, one token: short rows (3072 x 1536) and long (1536 x 3072), rotated through 64 copies
  std::printf("a dense Q8_0 matvec, one token, cold\n");
  for (auto [rows, cols, what] : {std::tuple<uint32_t, uint32_t, const char*>{3072, 1536, "routed_up / shared down"},
                                  {1536, 3072, "routed_down / shared gate"}}) {
    const uint64_t mb = gguf_bytes(8, (uint64_t) rows * cols);
    uint8_t* pool = noise(mb * 64);
    float* xi = floats(cols, 1.0f);
    float* yo1 = dev<float>(rows);
    const double us = time_us(iters, [&](uint32_t i) {
      return mv_strided(8, pool + (uint64_t) (i % 64) * mb, rows, cols, xi, cols, 1, yo1, rows, nullptr);
    });
    std::printf("  %4u x %4u (%s): %6.1f us (%5.1f GB/s)\n", rows, cols, what, us, (double) mb / 1e3 / us);
    CHECK(hipFree(pool) == hipSuccess);
  }
  return 0;
}
