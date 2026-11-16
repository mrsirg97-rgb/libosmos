// fp4bench [tokens]: mv_fp4 (kernels/mv_fp4.hip) timed on Qwen3.8-27B's NVFP4 shapes (the q35 family's decode: its
// FFN's gate / up 17408 x 5120 and down 5120 x 17408, the DeltaNet's qkv 10240 x 5120, z 6144 x 5120 and out
// 5120 x 6144, attention's q 12288 x 5120) at 1 .. 4 tokens, every call reading cold memory (its matrices rotated
// through 2 GiB): microseconds a call and the weights' GB/s. a decode token's whole set in the last line (64 layers:
// 48 DeltaNet, 16 attention, each its FFN), the matrices' share of the token's time at that rate. READ=1: a plain
// read's GB/s on the same bytes (the ceiling: 244 at low-power); SWEEP=1 [SHAPE=i]: direct_kernel's R and U on shape i
// against fp4_kernel's bits. measured, not kept: the blocks split at load (scales apart, the codes' 32-byte runs: two
// uint4 loads a block) - 224 GB/s against 225 on gate / up, 205-211 against 210-213 on the long rows; x copied to the
// workgroup's LDS once (every wave's reads from there) - 191-217 against 212-225 (its load and sync more than the L2
// reads they save); many tokens a call with fp4_kernel's bits (a sub-block decoded once, each token's x in turn) -
// 4 tokens 646 us, 16 1,820 against one's 226 (x from L2 once a row a token: 5.7 GB at 16 tokens - a matmul's job)
// build: make fp4bench (or hipcc -O3 -Iinclude -Ikernels --offload-arch=gfx1151 tools/bench/fp4bench.hip)
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../../kernels/mv_fp4.hip"
extern "C" {
#include "osmos/gpu/mm.h"
}

constexpr uint64_t kPool = 2ull << 30;
constexpr int kCalls = 100;

#define HIP(x)                                                     \
  do {                                                             \
    if ((x) != hipSuccess) return std::printf("hip: %s\n", #x), 1; \
  } while (0)

__global__ void fill(uint32_t* p, uint64_t n) {
  for (uint64_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += gridDim.x * 256ull) {
    uint32_t v = (uint32_t) i * 2654435761u;
    v ^= v >> 15, v *= 2246822519u, v ^= v >> 13;
    p[i] = v & 0x77777777u;  // UE4M3 scales kept under the NaN code, any E2M1 code
  }
}

// the ceiling: a plain read of the matrices' bytes (each lane uint4 loads summed, one write a block), the GB/s no
// matvec of these bytes can pass
__global__ void read_kernel(const uint4* p, uint64_t n, uint32_t* out) {
  uint32_t acc = 0;
  for (uint64_t i = blockIdx.x * 256ull + threadIdx.x; i < n; i += gridDim.x * 256ull) {
    const uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) out[blockIdx.x] = acc;
}

struct Shape {
  const char* name;
  uint32_t rows, cols, per_token;  // per_token: the decode token's count of this matrix
};

int main(int argc, char** argv) {
  const uint32_t n = argc > 1 ? (uint32_t) std::atoi(argv[1]) : 1;
  if (n < 1 || n > 4) return std::printf("fp4bench [1-4]\n"), 2;
  const Shape shapes[] = {{"ffn gate/up", 17408, 5120, 128}, {"ffn down", 5120, 17408, 64},
                          {"dn qkv", 10240, 5120, 48},       {"dn z", 6144, 5120, 48},
                          {"dn out", 5120, 6144, 48},         {"attn q", 12288, 5120, 16},
                          {"attn out", 5120, 6144, 16},       {"attn k/v", 1024, 5120, 32}};
  uint8_t* pool;
  float *x, *y;
  HIP(hipMalloc(&pool, kPool));
  HIP(hipMalloc(&x, 4ull * 17408 * 4));
  HIP(hipMalloc(&y, 4ull * 17408 * 4));
  fill<<<4096, 256>>>((uint32_t*) pool, kPool / 4);
  fill<<<64, 256>>>((uint32_t*) x, 4ull * 17408);
  HIP(hipMemset(x, 0, 4ull * 17408 * 4));  // finite activations (the timing's, not the values')
  HIP(hipDeviceSynchronize());
  hipEvent_t a, b;
  HIP(hipEventCreate(&a));
  HIP(hipEventCreate(&b));
  double token_us = 0, token_bytes = 0;
  std::printf("mv_fp4 at %u token(s), cold\n", n);
  for (const Shape& s : shapes) {
    const uint64_t bytes = (uint64_t) s.rows * (s.cols / 64) * 36, slots = kPool / bytes;
    for (int w = 0; w < 5; w++) mv_fp4(pool + (w % slots) * bytes, s.rows, s.cols, x, s.cols, n, y, s.rows, 1.0f, nullptr);
    HIP(hipEventRecord(a));
    for (int c = 0; c < kCalls; c++)
      mv_fp4(pool + (uint64_t) (c % slots) * bytes, s.rows, s.cols, x, s.cols, n, y, s.rows, 1.0f, nullptr);
    HIP(hipEventRecord(b));
    HIP(hipEventSynchronize(b));
    float ms = 0;
    HIP(hipEventElapsedTime(&ms, a, b));
    const double us = 1000.0 * ms / kCalls;
    std::printf("  %-12s %5u x %5u  %7.1f us  %6.1f GB/s\n", s.name, s.rows, s.cols, us, bytes / us / 1e3);
    token_us += us * s.per_token, token_bytes += (double) bytes * s.per_token;
  }
  std::printf("a decode token's NVFP4 matrices: %.2f GB in %.1f ms (%.1f GB/s): %.1f tok/s on these alone\n",
              token_bytes / 1e9, token_us / 1e3, token_bytes / token_us / 1e3, 1e6 / token_us);
  if (std::getenv("MM")) {  // the matmul (mm_tokens, WMMA over f16) on the same NVFP4 bytes: its time a step at n tokens
    void* x16;
    int32_t* over;
    float* yy;
    HIP(hipMalloc(&x16, (uint64_t) MM_MAX_TOKENS * 17408 * 2));
    HIP(hipMalloc(&over, 4));
    HIP(hipMalloc(&yy, 16ull * 17408 * 4));
    float* xx;
    HIP(hipMalloc(&xx, 16ull * 17408 * 4));
    HIP(hipMemset(xx, 0, 16ull * 17408 * 4));
    for (const Shape& s : {shapes[0], shapes[1], shapes[2], shapes[4]})
      for (uint32_t nt : {1u, 2u, 4u, 8u, 16u}) {
        const uint64_t bytes = (uint64_t) s.rows * (s.cols / 64) * 36, slots = kPool / bytes;
        auto go = [&](uint8_t* w) {
          return mm_tokens(40, w, s.rows, yy, s.rows, nullptr, 0, nullptr, 0, s.cols, xx, s.cols, nt, x16, over, false, nullptr);
        };
        if (!go(pool)) { std::printf("  mm refused %s at %u\n", s.name, nt); continue; }
        for (int w = 0; w < 5; w++) go(pool + (w % slots) * bytes);
        hipEventRecord(a);
        for (int c = 0; c < kCalls; c++) go(pool + (uint64_t) (c % slots) * bytes);
        hipEventRecord(b);
        hipEventSynchronize(b);
        float ms = 0;
        hipEventElapsedTime(&ms, a, b);
        const double us = 1000.0 * ms / kCalls;
        std::printf("  mm %-12s %2u tokens %7.1f us  %6.1f GB/s\n", s.name, nt, us, bytes / us / 1e3);
      }
  }
  if (std::getenv("READ")) {  // the read's ceiling on one gate / up matrix's bytes, cold, at a few grid sizes
    const uint64_t bytes = 17408ull * 80 * 36, slots = kPool / bytes;
    for (uint32_t grid : {1024u, 2048u, 4096u, 8192u}) {
      for (int w = 0; w < 5; w++) read_kernel<<<grid, 256>>>((const uint4*) (pool + (w % slots) * bytes), bytes / 16, (uint32_t*) y);
      hipEventRecord(a);
      for (int c = 0; c < kCalls; c++)
        read_kernel<<<grid, 256>>>((const uint4*) (pool + (uint64_t) (c % slots) * bytes), bytes / 16, (uint32_t*) y);
      hipEventRecord(b);
      hipEventSynchronize(b);
      float ms = 0;
      hipEventElapsedTime(&ms, a, b);
      std::printf("  read %4u blocks: %7.1f us  %6.1f GB/s\n", grid, 1000.0 * ms / kCalls, bytes / (1000.0 * ms / kCalls) / 1e3);
    }
  }
  if (std::getenv("SWEEP")) {  // RDNA's direct_kernel at R rows a warp, U blocks in flight, against fp4_kernel's bits
    const Shape& s = shapes[std::getenv("SHAPE") ? std::atoi(std::getenv("SHAPE")) : 0];  // SHAPE: shapes' index
    const uint64_t bytes = (uint64_t) s.rows * (s.cols / 64) * 36, slots = kPool / bytes;
    std::vector<float> h0(4ull * s.rows), h1(4ull * s.rows);
    std::vector<float> hx(4ull * s.cols);
    for (uint64_t i = 0; i < hx.size(); i++) hx[i] = (float) ((int) (i * 2654435761u % 2001u) - 1000) / 337.0f;
    HIP(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    auto bits = [&](auto fn) {  // fp4_kernel's (R 1) against fn's, the whole n x rows
      hipMemset(y, 0, 4ull * s.rows * 4);
      if (n == 1) run<1, 1>(pool, s.rows, s.cols, x, s.cols, y, s.rows, 1.0f, nullptr);
      if (n == 2) run<1, 2>(pool, s.rows, s.cols, x, s.cols, y, s.rows, 1.0f, nullptr);
      if (n == 3) run<1, 3>(pool, s.rows, s.cols, x, s.cols, y, s.rows, 1.0f, nullptr);
      if (n == 4) run<1, 4>(pool, s.rows, s.cols, x, s.cols, y, s.rows, 1.0f, nullptr);
      hipMemcpy(h0.data(), y, (uint64_t) n * s.rows * 4, hipMemcpyDeviceToHost);
      hipMemset(y, 0, 4ull * s.rows * 4);
      fn(pool);
      hipMemcpy(h1.data(), y, (uint64_t) n * s.rows * 4, hipMemcpyDeviceToHost);
      return std::memcmp(h0.data(), h1.data(), (uint64_t) n * s.rows * 4) == 0;
    };
    auto time = [&](auto fn, const char* tag) {
      const bool same = bits(fn);
      for (int w = 0; w < 5; w++) fn(pool + (w % slots) * bytes);
      hipEventRecord(a);
      for (int c = 0; c < kCalls; c++) fn(pool + (uint64_t) (c % slots) * bytes);
      hipEventRecord(b);
      hipEventSynchronize(b);
      float ms = 0;
      hipEventElapsedTime(&ms, a, b);
      std::printf("  %-10s %7.1f us  %6.1f GB/s  %s\n", tag, 1000.0 * ms / kCalls, bytes / (1000.0 * ms / kCalls) / 1e3,
                  same ? "fp4_kernel's bits" : "BITS DIFFER");
    };
#define TRY(R, U)                                                                                                  \
  {                                                                                                                \
    auto fn = [&](uint8_t* w) {                                                                                    \
      if (n == 1) run_direct<R, 1, U>(w, s.rows, s.cols, x, s.cols, y, s.rows, 1.0f, nullptr);                     \
      if (n == 2) run_direct<R, 2, U>(w, s.rows, s.cols, x, s.cols, y, s.rows, 1.0f, nullptr);                     \
      if (n == 3) run_direct<R, 3, U>(w, s.rows, s.cols, x, s.cols, y, s.rows, 1.0f, nullptr);                     \
      if (n == 4) run_direct<R, 4, U>(w, s.rows, s.cols, x, s.cols, y, s.rows, 1.0f, nullptr);                     \
    };                                                                                                             \
    time(fn, "R" #R " U" #U);                                                                                      \
  }
    std::printf("direct_kernel at %u token(s), %s %u x %u:\n", n, s.name, s.rows, s.cols);
    TRY(1, 1) TRY(1, 2) TRY(2, 1) TRY(2, 2) TRY(4, 1) TRY(4, 2)
  }
  return 0;
}
