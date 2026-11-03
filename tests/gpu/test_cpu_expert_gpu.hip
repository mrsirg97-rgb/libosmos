// the CPU lane's matvec (src/cpu/expert.c) against the GPU's (mv_strided: kernels/mv.h's rows_of) on ds1017's
// experts - an IQ1_M, an IQ2_XXS and an IQ2_XS layer's gate and down, 1 to 8 tokens: every output the same bits.
// and the CPU's own: rows split anywhere, tokens one at a time or together, the same bits; refusals
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "test_model.h"

extern "C" {
#include "substrate/cpu/expert.h"
#include "substrate/gpu/mv.h"
#include "substrate/model/gguf.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / (float) (1u << 23) - 1.0f;
}

static void expert(const Gguf* g, const char* name, int e) {
  const GgufTensor* t = gguf_tensor(g, name);
  CHECK(t && t->n_dims == 3);
  const uint32_t cols = (uint32_t) t->dims[0], rows = (uint32_t) t->dims[1];
  const uint64_t bytes = t->bytes / t->dims[2];
  std::vector<uint8_t> w(bytes);
  const int fd = open(g->paths[t->file], O_RDONLY);
  CHECK(fd >= 0 && pread(fd, w.data(), bytes, (off_t) (t->offset + (uint64_t) e * bytes)) == (ssize_t) bytes);
  close(fd);
  constexpr uint32_t kTok = 8;
  std::vector<float> x((size_t) kTok * cols), cpu((size_t) kTok * rows), gpu(cpu.size()), part(cpu.size());
  for (float& v : x) v = uniform();
  uint8_t* dw;
  float *dx, *dy;
  CHECK(hipMalloc(&dw, bytes) == hipSuccess && hipMalloc(&dx, x.size() * 4) == hipSuccess &&
        hipMalloc(&dy, cpu.size() * 4) == hipSuccess);
  CHECK(hipMemcpy(dw, w.data(), bytes, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(cpu_mv_known(t->type));
  for (uint32_t n = 1; n <= kTok; n++) {  // the GPU at n tokens, the CPU at all 8: token t's rows its own either way
    CHECK(mv_strided(t->type, dw, rows, cols, dx, cols, n, dy, rows, nullptr) && hipDeviceSynchronize() == hipSuccess);
    CHECK(hipMemcpy(gpu.data(), dy, (size_t) n * rows * 4, hipMemcpyDeviceToHost) == hipSuccess);
    if (n == 1) CHECK(cpu_mv_rows(t->type, w.data(), cols, 0, rows, x.data(), cols, kTok, cpu.data(), rows));
    CHECK(std::memcmp(gpu.data(), cpu.data(), (size_t) n * rows * 4) == 0);
  }
  // the CPU's rows in three uneven parts, each token alone: the same bits
  const uint32_t a = rows / 3 + 5, b = rows - 7;
  for (uint32_t tk = 0; tk < kTok; tk++)
    for (uint32_t r0 : {0u, a, b}) {
      const uint32_t r1 = r0 == 0 ? a : r0 == a ? b : rows;
      CHECK(cpu_mv_rows(t->type, w.data(), cols, r0, r1, x.data() + (size_t) tk * cols, cols, 1,
                        part.data() + (size_t) tk * rows, rows));
    }
  CHECK(std::memcmp(part.data(), cpu.data(), cpu.size() * 4) == 0);
  CHECK(hipFree(dw) == hipSuccess && hipFree(dx) == hipSuccess && hipFree(dy) == hipSuccess);
  std::printf("  %-28s[%d] type %2u %4u x %4u: the GPU's bits at 1 .. %u tokens; rows split, tokens alone the same\n",
              name, e, t->type, rows, cols, kTok);
}

int main() {
  std::printf("[test_cpu_expert_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_cpu_expert_gpu] skipped: no ds1017 or UD-IQ1_M, or no GPU\n");
    return 77;
  }
  const float x[256] = {0};
  float y[1];
  CHECK(!cpu_mv_known(13) && !cpu_mv_rows(13, x, 256, 0, 1, x, 256, 1, y, 1));  // Q5_K: no CPU decode
  CHECK(!cpu_mv_rows(16, x, 255, 0, 1, x, 256, 1, y, 1) && !cpu_mv_rows(16, x, 256, 2, 1, x, 256, 1, y, 1));
  char err[GGUF_ERR] = "";
  Gguf* g = gguf_create(path, err);
  CHECK(g);
  for (const char* name : which == TestModel::kUd ? kUdExperts : kGsqExperts) expert(g, name, 7);
  gguf_destroy(g);
  std::printf("[test_cpu_expert_gpu] ok\n");
  return 0;
}
