// mmsweep [experts|dense|q4]: mm's tile shapes (kernels/mm.h's Shape: SL values a slab, W waves a block, U chunks in
// flight a lane) timed at one token on Qwen3.8-Flash-Next's shapes - each expert type's gate / up (640 x 2560) and
// down (2560 x 640), 10 experts a launch (mm_group), and the dense Q6_K / Q8_0 - every call reading cold memory (its
// matrices rotated through 1 GiB). a shape is a K split, so it is the sums' order: a type's for every token count
// (the bits then the same however a token is batched); this picks the one-token time.
// build: hipcc -O3 -Iinclude -Ikernels --offload-arch=gfx1151 tools/bench/mmsweep.hip -o bin/mmsweep
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "mm.h"

constexpr uint64_t kPool = 1ull << 30;
constexpr int kCalls = 200, kExperts = 10;

#define HIP(x)                                                    \
  do {                                                            \
    if ((x) != hipSuccess) return std::printf("hip: %s\n", #x), 1; \
  } while (0)

__global__ void fill(uint32_t* p, uint64_t n) {
  for (uint64_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += gridDim.x * 256ull) {
    uint32_t v = (uint32_t) i * 2654435761u;
    v ^= v >> 15, v *= 2246822519u, v ^= v >> 13;
    p[i] = v & 0x3bff3bffu;  // f16-sized halves kept finite (the K-quants' scales)
  }
}

static uint8_t* pool;
static _Float16* x16;
static float* y;
static const void** ptrs;
static int32_t* cnt;
static uint32_t *xoff, *yoff;

template <quant::Type T>
static uint64_t row_bytes(uint32_t cols) {
  return (uint64_t) cols / quant::Block<T>::values * quant::Block<T>::bytes;
}

template <typename F>
static double time_us(F launch) {
  hipEvent_t a, b;
  hipEventCreate(&a), hipEventCreate(&b);
  float ms = 0;
  for (int pass = 0; pass < 2; pass++) {
    hipEventRecord(a);
    for (int c = 0; c < kCalls; c++) launch(c);
    hipEventRecord(b);
    hipEventSynchronize(b);
  }
  hipEventElapsedTime(&ms, a, b);
  return 1e3 * ms / kCalls;
}

template <quant::Type T, int SL, int W, int U>
static void group(const char* type, uint32_t rows, uint32_t cols) {
  using S = mm::Shape<SL, W, U>;
  const uint64_t mat = row_bytes<T>(cols) * rows, n_mats = kPool / mat;
  std::vector<const void*> host(kExperts * kCalls);
  for (int c = 0; c < kCalls; c++)
    for (int e = 0; e < kExperts; e++) host[c * kExperts + e] = pool + ((uint64_t) (c * kExperts + e) % n_mats) * mat;
  hipMemcpy(ptrs, host.data(), host.size() * sizeof(void*), hipMemcpyHostToDevice);
  const double us = time_us([&](int c) {
    mm::group_kernel<T, S, 1><<<dim3(rows / mm::kTile, kExperts, 1), mm::kWave * S::kWaves>>>(
        (const uint8_t* const*) (ptrs + c * kExperts), cnt, cols, x16, xoff, cols, y, yoff, rows);
  });
  std::printf("  %-8s %5u x %-5u SL %3d W %d U %d  %7.1f us  %6.1f GB/s\n", type, rows, cols, SL, W, U, us,
              kExperts * mat / us / 1e3);
}

template <quant::Type T, int SL, int W, int U>
static void dense(const char* type, uint32_t rows, uint32_t cols) {
  using S = mm::Shape<SL, W, U>;
  const uint64_t mat = row_bytes<T>(cols) * rows, n_mats = kPool / mat > 0 ? kPool / mat : 1;
  const double us = time_us([&](int c) {
    mm::kernel<T, S, 1><<<dim3(rows / mm::kTile, 1, 1), mm::kWave * S::kWaves>>>(
        pool + (c % n_mats) * mat, nullptr, rows, 0, cols, x16, 1, y, nullptr, rows, 0);
  });
  std::printf("  %-8s %6u x %-5u SL %3d W %d U %d  %7.1f us  %6.1f GB/s\n", type, rows, cols, SL, W, U, us, mat / us / 1e3);
}

// the 32-value slabs (Q5_K's 32 x 8 x 2 among them): the K-quants' sweep
#define SHAPES32(F, T, name, rows, cols)                                                                       \
  F<T, 32, 8, 2>(name, rows, cols), F<T, 32, 8, 1>(name, rows, cols), F<T, 32, 4, 2>(name, rows, cols),        \
      F<T, 32, 4, 1>(name, rows, cols) /* a 32-value slab: 2 chunks a lane, U 1 or 2 */
// 16 waves a block (512 threads; a long row's K split 16 ways)
#define SHAPES16(F, T, name, rows, cols)                                                                       \
  F<T, 64, 16, 4>(name, rows, cols), F<T, 64, 16, 2>(name, rows, cols), F<T, 32, 16, 2>(name, rows, cols),     \
      F<T, 32, 16, 1>(name, rows, cols) /* (128 x 16: 68 KB of LDS, past the 64) */
#define SHAPES(F, T, name, rows, cols)                                                                       \
  F<T, 64, 4, 2>(name, rows, cols), F<T, 64, 4, 4>(name, rows, cols), F<T, 64, 8, 2>(name, rows, cols),      \
      F<T, 64, 8, 4>(name, rows, cols), F<T, 128, 4, 4>(name, rows, cols), F<T, 128, 4, 8>(name, rows, cols), \
      F<T, 128, 8, 4>(name, rows, cols), F<T, 128, 8, 8>(name, rows, cols), F<T, 256, 4, 8>(name, rows, cols)

int main(int argc, char** argv) {
  const bool experts = argc < 2 || !std::strcmp(argv[1], "experts"), q4 = argc > 1 && !std::strcmp(argv[1], "q4");
  HIP(hipMalloc(&pool, kPool));
  HIP(hipMalloc(&x16, (size_t) mm::kTile * 10240 * 2));
  HIP(hipMalloc(&y, (size_t) 16 * 262144 * 4));
  HIP(hipMalloc(&ptrs, kExperts * kCalls * sizeof(void*)));
  HIP(hipMalloc(&cnt, kExperts * 4));
  HIP(hipMalloc(&xoff, kExperts * 4));
  HIP(hipMalloc(&yoff, kExperts * 4));
  fill<<<4096, 256>>>((uint32_t*) pool, kPool / 4);
  HIP(hipMemset(x16, 0, (size_t) mm::kTile * 10240 * 2));
  std::vector<int32_t> ones(kExperts, 1);
  std::vector<uint32_t> zero(kExperts, 0), ys(kExperts);
  for (int e = 0; e < kExperts; e++) ys[e] = e * 2560;
  HIP(hipMemcpy(cnt, ones.data(), kExperts * 4, hipMemcpyHostToDevice));
  HIP(hipMemcpy(xoff, zero.data(), kExperts * 4, hipMemcpyHostToDevice));
  HIP(hipMemcpy(yoff, ys.data(), kExperts * 4, hipMemcpyHostToDevice));
  HIP(hipDeviceSynchronize());
  if (q4) {  // UD-Q4_K_XL's expert types: Q4_K gate / up, Q5_1 down, a few layers' Q5_K gates
    SHAPES(group, quant::kQ4_K, "Q4_K", 640, 2560);
    SHAPES32(group, quant::kQ4_K, "Q4_K", 640, 2560);
    SHAPES16(group, quant::kQ4_K, "Q4_K", 640, 2560);
    SHAPES(group, quant::kQ5_1, "Q5_1", 2560, 640);
    SHAPES32(group, quant::kQ5_1, "Q5_1", 2560, 640);
    group<quant::kQ5_1, 32, 16, 2>("Q5_1", 2560, 640), group<quant::kQ5_1, 32, 16, 1>("Q5_1", 2560, 640);
    SHAPES(group, quant::kQ5_K, "Q5_K", 640, 2560);
  } else if (experts) {
    SHAPES(group, quant::kIQ4_NL, "IQ4_NL", 2560, 640);
    SHAPES32(group, quant::kIQ4_NL, "IQ4_NL", 2560, 640);
    group<quant::kIQ4_NL, 32, 16, 2>("IQ4_NL", 2560, 640);
    SHAPES(group, quant::kIQ3_XXS, "IQ3_XXS", 640, 2560);
    SHAPES32(group, quant::kIQ3_XXS, "IQ3_XXS", 640, 2560);
    SHAPES16(group, quant::kIQ3_XXS, "IQ3_XXS", 640, 2560);
    SHAPES(group, quant::kIQ2_S, "IQ2_S", 640, 2560);
    SHAPES16(group, quant::kIQ2_S, "IQ2_S", 640, 2560);
    SHAPES(group, quant::kIQ4_XS, "IQ4_XS", 2560, 640);
    SHAPES32(group, quant::kIQ4_XS, "IQ4_XS", 2560, 640);
    group<quant::kIQ4_XS, 32, 16, 2>("IQ4_XS", 2560, 640);
    SHAPES(group, quant::kQ2_0, "Q2_0", 640, 2560);
    SHAPES16(group, quant::kQ2_0, "Q2_0", 640, 2560);
  } else {
    SHAPES(dense, quant::kQ6_K, "Q6_K", 10240, 2560);
    SHAPES(dense, quant::kQ6_K, "Q6_K", 2560, 6144);
    SHAPES(dense, quant::kQ6_K, "Q6_K", 248320, 2560);
    SHAPES(dense, quant::kQ8_0, "Q8_0", 320, 10240);
    // 320 columns: 64-value slabs alone
    dense<quant::kQ8_0, 64, 4, 2>("Q8_0", 10240, 320), dense<quant::kQ8_0, 64, 4, 4>("Q8_0", 10240, 320);
    dense<quant::kQ8_0, 64, 8, 2>("Q8_0", 10240, 320), dense<quant::kQ8_0, 64, 8, 4>("Q8_0", 10240, 320);
    dense<quant::kQ8_0, 64, 2, 4>("Q8_0", 10240, 320), dense<quant::kQ8_0, 64, 2, 2>("Q8_0", 10240, 320);
  }
  return 0;
}
