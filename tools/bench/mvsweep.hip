// mvsweep [experts|dense|f32|ink2|hy4] [tokens]: (hy4: Hy4-flash's Q3_K / Q4_K experts and Q8_0 attention, 1 or 6) (ink2: Inkling-2's expert types at its shapes - gate / up 2304 x 4608,
// down 4608 x 2304 - at 1, 2, 4 or 8 tokens) the matvec kernel's shapes (kernels/mv.h: R rows a wave, U chunks in flight a
// lane) timed on Qwen3.8-Flash-Next's shapes at 1, 2, 3 or 4 tokens (an expert read for that many) - each expert type's gate / up (640 x 2560) and down (2560 x 640), 10 experts
// a call, and the dense matrices' - every call reading cold memory (its matrices rotated through 1 GiB). R and U
// change no bit (mv.h: a lane's chunks summed in the same order); what this picks is time.
// build: hipcc -O3 -Iinclude -Ikernels --offload-arch=gfx1151 tools/bench/mvsweep.hip -o bin/mvsweep
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "mv.h"

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
static float *x, *y;
static const void** ptrs;
static int32_t *cnt, *xrow;

template <quant::Type T>
static uint64_t row_bytes(uint32_t cols) {
  return (uint64_t) cols / quant::Block<T>::values * quant::Block<T>::bytes;
}

template <quant::Type T, int NT, int R, int U>
static void group(const char* type, uint32_t rows, uint32_t cols) {
  const uint64_t mat = row_bytes<T>(cols) * rows, n_mats = kPool / mat;
  std::vector<const void*> host(kExperts * kCalls);
  for (int c = 0; c < kCalls; c++)
    for (int e = 0; e < kExperts; e++) host[c * kExperts + e] = pool + ((uint64_t) (c * kExperts + e) % n_mats) * mat;
  hipMemcpy(ptrs, host.data(), host.size() * sizeof(void*), hipMemcpyHostToDevice);
  const uint32_t per_block = mv::kWavesPerBlock * R;
  const dim3 grid((rows + per_block - 1) / per_block, kExperts);
  hipEvent_t a, b;
  hipEventCreate(&a), hipEventCreate(&b);
  for (int pass = 0; pass < 2; pass++) {  // the first warms the kernel; the second is timed
    hipEventRecord(a);
    for (int c = 0; c < kCalls; c++)
      mv::group_tok_kernel<T, NT, R, U><<<grid, mv::kWave * mv::kWavesPerBlock>>>(
          (const uint8_t* const*) (ptrs + c * kExperts), cnt, xrow, rows, cols, x, cols, y, rows);
    hipEventRecord(b);
    hipEventSynchronize(b);
  }
  float ms;
  hipEventElapsedTime(&ms, a, b);
  const double us = 1e3 * ms / kCalls;
  std::printf("  %-8s %5u x %-5u NT %d R %d U %d  %7.1f us  %6.1f GB/s\n", type, rows, cols, NT, R, U, us,
              kExperts * mat / us / 1e3);
}

template <quant::Type T, int NT, int R, int U>
static void dense(const char* type, uint32_t rows, uint32_t cols) {
  const uint64_t mat = row_bytes<T>(cols) * rows, n_mats = kPool / mat > 0 ? kPool / mat : 1;
  const uint32_t per_block = mv::kWavesPerBlock * R;
  hipEvent_t a, b;
  hipEventCreate(&a), hipEventCreate(&b);
  for (int pass = 0; pass < 2; pass++) {
    hipEventRecord(a);
    for (int c = 0; c < kCalls; c++)
      mv::kernel<T, NT, R, U><<<(rows + per_block - 1) / per_block, mv::kWave * mv::kWavesPerBlock>>>(
          pool + (c % n_mats) * mat, rows, cols, x, cols, y, rows);
    hipEventRecord(b);
    hipEventSynchronize(b);
  }
  float ms;
  hipEventElapsedTime(&ms, a, b);
  const double us = 1e3 * ms / kCalls;
  std::printf("  %-8s %6u x %-5u NT %d R %d U %d  %7.1f us  %6.1f GB/s\n", type, rows, cols, NT, R, U, us,
              mat / us / 1e3);
}

#define SWEEP_G(T, NT, name, rows, cols)                                                                   \
  group<T, NT, 1, 1>(name, rows, cols), group<T, NT, 1, 2>(name, rows, cols), group<T, NT, 1, 4>(name, rows, cols), \
      group<T, NT, 2, 1>(name, rows, cols), group<T, NT, 2, 2>(name, rows, cols),                                 \
      group<T, NT, 2, 4>(name, rows, cols), group<T, NT, 4, 1>(name, rows, cols), group<T, NT, 4, 2>(name, rows, cols)
#define SWEEP_D(T, NT, name, rows, cols)                                                                   \
  dense<T, NT, 1, 1>(name, rows, cols), dense<T, NT, 1, 2>(name, rows, cols), dense<T, NT, 1, 4>(name, rows, cols), \
      dense<T, NT, 2, 1>(name, rows, cols), dense<T, NT, 2, 2>(name, rows, cols),                                 \
      dense<T, NT, 2, 4>(name, rows, cols), dense<T, NT, 4, 1>(name, rows, cols), dense<T, NT, 4, 2>(name, rows, cols)

// the F32 matvecs a Qwen step's few tokens read (a verify's: 2 .. 8): the shared expert's gate (1 x 2560), the
// mixers' inject (4 x 10240), the DeltaNet's alpha / beta (32 x 2560), the router (512 x 2560) - U up to 8 too
#define SWEEP_F(NT, rows, cols)                                                                                     \
  dense<quant::kF32, NT, 1, 1>("F32", rows, cols), dense<quant::kF32, NT, 1, 2>("F32", rows, cols),               \
      dense<quant::kF32, NT, 1, 4>("F32", rows, cols), dense<quant::kF32, NT, 1, 8>("F32", rows, cols),           \
      dense<quant::kF32, NT, 2, 2>("F32", rows, cols), dense<quant::kF32, NT, 2, 4>("F32", rows, cols)
template <int NT>
static void sweep_f32() {
  SWEEP_F(NT, 1, 2560);
  SWEEP_F(NT, 4, 10240);
  SWEEP_F(NT, 32, 2560);
  SWEEP_F(NT, 512, 2560);
}

template <int NT>
static void sweep(bool experts) {
  if (experts) {
    SWEEP_G(quant::kIQ4_NL, NT, "IQ4_NL", 2560, 640);
    SWEEP_G(quant::kIQ3_XXS, NT, "IQ3_XXS", 640, 2560);
    SWEEP_G(quant::kIQ2_S, NT, "IQ2_S", 640, 2560);
    SWEEP_G(quant::kIQ4_XS, NT, "IQ4_XS", 2560, 640);
    SWEEP_G(quant::kQ2_0, NT, "Q2_0", 640, 2560);
  } else {
    SWEEP_D(quant::kQ6_K, NT, "Q6_K", 10240, 2560);
    SWEEP_D(quant::kQ6_K, NT, "Q6_K", 2560, 6144);
    SWEEP_D(quant::kQ6_K, NT, "Q6_K", 248320, 2560);
    SWEEP_D(quant::kQ8_0, NT, "Q8_0", 320, 10240);
    SWEEP_D(quant::kQ8_0, NT, "Q8_0", 10240, 320);
    SWEEP_D(quant::kF32, NT, "F32", 512, 2560);
    SWEEP_D(quant::kF32, NT, "F32", 48, 2560);
  }
}

template <int NT>
static void sweep_ink2_dense() {  // Inkling-2's attention at Q5_K (GPTQ'd): q / gate 6144 x 4608, o 4608 x 6144, k 1024 x 4608
  SWEEP_D(quant::kQ5_K, NT, "Q5_K", 6144, 4608);
  SWEEP_D(quant::kQ5_K, NT, "Q5_K", 4608, 6144);
  SWEEP_D(quant::kQ5_K, NT, "Q5_K", 1024, 4608);
  SWEEP_D(quant::kQ5_KR, NT, "Q5_KR", 6144, 4608);
  SWEEP_D(quant::kQ5_KR, NT, "Q5_KR", 4608, 6144);
  SWEEP_D(quant::kQ5_KR, NT, "Q5_KR", 1024, 4608);
}

template <int NT>
static void sweep_ink2() {
  SWEEP_G(quant::kIQ1_S, NT, "IQ1_S", 2304, 4608);
  SWEEP_G(quant::kIQ2_XXS, NT, "IQ2_XXS", 2304, 4608);
  SWEEP_G(quant::kIQ2_S, NT, "IQ2_S", 4608, 2304);
  SWEEP_G(quant::kIQ3_XXS, NT, "IQ3_XXS", 4608, 2304);
  SWEEP_G(quant::kIQ4_XS, NT, "IQ4_XS", 4608, 2304);
}

template <int NT>
static void sweep_hy4() {  // Hy4-flash's experts (Q3_K's file): gate / up 1536 x 1536 Q3_K or Q4_K, down 1536 x 1536 Q4_K
  SWEEP_G(quant::kQ3_K, NT, "Q3_K", 1536, 1536);
  SWEEP_G(quant::kQ4_K, NT, "Q4_K", 1536, 1536);
  SWEEP_D(quant::kQ8_0, NT, "Q8_0", 16384, 3072);  // (a sliding layer's gate / out: 64 x 256 wide)
  SWEEP_D(quant::kQ8_0, NT, "Q8_0", 3072, 16384);
  SWEEP_D(quant::kQ6_K, NT, "Q6_K", 16384, 3072);  // (the gate and output requantized: tools/evo/requant_hy4.py)
  SWEEP_D(quant::kQ6_K, NT, "Q6_K", 3072, 16384);
  SWEEP_D(quant::kQ5_K, NT, "Q5_K", 16384, 3072);
  SWEEP_D(quant::kQ5_K, NT, "Q5_K", 3072, 16384);
  SWEEP_D(quant::kQ4_K, NT, "Q4_K", 16384, 3072);
  SWEEP_D(quant::kQ4_K, NT, "Q4_K", 3072, 16384);
  SWEEP_D(quant::kQ8_0, NT, "Q8_0", 120832, 3072);  // (DFly2's head, Q8_0)
  SWEEP_D(quant::kQ8_0, NT, "Q8_0", 3072, 6144);    // (DFly2's fusion)
  SWEEP_D(quant::kQ8_0, NT, "Q8_0", 1024, 3072);   // (q_a)
  SWEEP_D(quant::kQ8_0, NT, "Q8_0", 8192, 1024);   // (a sparse layer's q_b)
  SWEEP_D(quant::kQ8_0, NT, "Q8_0", 1536, 3072);   // (the latent's down)
  SWEEP_G(quant::kQ4_K, NT, "Q4_K", 640, 2560);    // (Qwen3.8 UD-Q4_K_XL's experts: the shared tune's other user)
  SWEEP_G(quant::kQ4_K, NT, "Q4_K", 2560, 640);
}

int main(int argc, char** argv) {
  const char* which = argc > 1 ? argv[1] : "experts";
  HIP(hipMalloc(&pool, kPool));
  HIP(hipMalloc(&x, 16 * 12288 * 4));
  HIP(hipMalloc(&y, 16 * 262144 * 4));
  HIP(hipMalloc(&ptrs, kExperts * kCalls * sizeof(void*)));
  HIP(hipMalloc(&cnt, kExperts * 4));
  HIP(hipMalloc(&xrow, kExperts * 4));
  fill<<<4096, 256>>>((uint32_t*) pool, kPool / 4);
  fill<<<64, 256>>>((uint32_t*) x, 12288);
  const int nt = argc > 2 ? std::atoi(argv[2]) : 1;
  std::vector<int32_t> cnts(kExperts, nt), zeros(kExperts, 0);
  HIP(hipMemcpy(cnt, cnts.data(), kExperts * 4, hipMemcpyHostToDevice));
  HIP(hipMemcpy(xrow, zeros.data(), kExperts * 4, hipMemcpyHostToDevice));
  HIP(hipDeviceSynchronize());
  const bool ex = !std::strcmp(which, "experts");
  if (!std::strcmp(which, "ink2dense")) {
    switch (nt) {
      case 1: sweep_ink2_dense<1>(); break;
      case 4: sweep_ink2_dense<4>(); break;
      case 8: sweep_ink2_dense<8>(); break;
      default: return std::printf("ink2dense tokens: 1, 4, 8\n"), 2;
    }
    return 0;
  }
  if (!std::strcmp(which, "hy4")) {
    switch (nt) {
      case 1: sweep_hy4<1>(); break;
      case 3: sweep_hy4<3>(); break;
      case 4: sweep_hy4<4>(); break;
      case 5: sweep_hy4<5>(); break;
      case 6: sweep_hy4<6>(); break;
      default: return std::printf("hy4 tokens: 1, 3, 4, 5, 6\n"), 2;
    }
    return 0;
  }
  if (!std::strcmp(which, "ink2")) {
    switch (nt) {
      case 1: sweep_ink2<1>(); break;
      case 2: sweep_ink2<2>(); break;
      case 4: sweep_ink2<4>(); break;
      case 8: sweep_ink2<8>(); break;
      default: return std::printf("ink2 tokens: 1, 2, 4, 8\n"), 2;
    }
    return 0;
  }
  if (!std::strcmp(which, "f32")) {
    switch (nt) {
      case 1: sweep_f32<1>(); break;
      case 2: sweep_f32<2>(); break;
      case 4: sweep_f32<4>(); break;
      case 6: sweep_f32<6>(); break;
      case 8: sweep_f32<8>(); break;
      default: return std::printf("f32 tokens: 1, 2, 4, 6, 8\n"), 2;
    }
    return 0;
  }
  switch (nt) {
    case 1: sweep<1>(ex); break;
    case 2: sweep<2>(ex); break;
    case 3: sweep<3>(ex); break;
    case 4: sweep<4>(ex); break;
    default: return std::printf("tokens: 1 to 4\n"), 2;
  }
  return 0;
}
