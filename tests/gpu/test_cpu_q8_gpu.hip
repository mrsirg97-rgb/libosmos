// the CPU lane's int8 experts (src/cpu/q8.c) against the GPU's (q8_round, q8_group_tok) on ds1017's experts - an
// IQ1_M, an IQ2_XXS and an IQ2_XS layer's gate and down, experts 7 and 300, 1 to 8 tokens: the rounding's bytes and
// scales and every output the same bits; the CPU's rows split in three, tokens alone, the same bits
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "substrate/cpu/q8.h"
#include "substrate/gpu/mv.h"
#include "substrate/gpu/q8.h"
#include "substrate/model/gguf.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kTok = MV_MAX_TOKENS, kBlock = 32;

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / (float) (1u << 23) - 1.0f;
}

static void expert(const Gguf* g, const char* name, int e) {
  const GgufTensor* t = gguf_tensor(g, name);
  CHECK(t && t->n_dims == 3);
  const uint32_t cols = (uint32_t) t->dims[0], rows = (uint32_t) t->dims[1], nb = cols / kBlock;
  const uint64_t bytes = t->bytes / t->dims[2];
  std::vector<uint8_t> w(bytes);
  const int fd = open(g->paths[t->file], O_RDONLY);
  CHECK(fd >= 0 && pread(fd, w.data(), bytes, (off_t) (t->offset + (uint64_t) e * bytes)) == (ssize_t) bytes);
  close(fd);
  std::vector<float> x((size_t) kTok * cols), qd((size_t) kTok * nb), gqd(qd.size());
  for (float& v : x) v = uniform() * 3.0f;
  for (uint32_t i = 0; i < kBlock; i++) x[i] = 0.0f, x[cols + kBlock + i] = i == 5 ? -2.5f : 0.0f;
  std::vector<int8_t> q((size_t) kTok * cols), gq(q.size());
  std::vector<float> cpu((size_t) kTok * rows), gpu(cpu.size()), part(cpu.size());
  uint8_t *dw, **dp;
  float *dx, *dd, *dy;
  int8_t* dq;
  int32_t *dcnt, *dxrow;
  CHECK(hipMalloc(&dw, bytes) == hipSuccess && hipMalloc(&dp, sizeof(void*)) == hipSuccess &&
        hipMalloc(&dx, x.size() * 4) == hipSuccess && hipMalloc(&dq, q.size()) == hipSuccess &&
        hipMalloc(&dd, qd.size() * 4) == hipSuccess && hipMalloc(&dy, cpu.size() * 4) == hipSuccess &&
        hipMalloc(&dcnt, 4) == hipSuccess && hipMalloc(&dxrow, 4) == hipSuccess);
  const int32_t zero = 0;
  CHECK(hipMemcpy(dw, w.data(), bytes, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dp, &dw, sizeof dw, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dxrow, &zero, 4, hipMemcpyHostToDevice) == hipSuccess);
  // the rounding: the GPU's bytes and scales
  CHECK(q8_round(dx, kTok, cols, cols, dq, dd, nullptr) && hipDeviceSynchronize() == hipSuccess);
  CHECK(hipMemcpy(gq.data(), dq, q.size(), hipMemcpyDeviceToHost) == hipSuccess &&
        hipMemcpy(gqd.data(), dd, qd.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(cpu_q8_round(x.data(), kTok, cols, cols, q.data(), qd.data()));
  CHECK(std::memcmp(gq.data(), q.data(), q.size()) == 0 && std::memcmp(gqd.data(), qd.data(), qd.size() * 4) == 0);
  // the rows: the CPU at all 8 tokens, the GPU at 1 .. 8 - token t's rows its own either way
  CHECK(cpu_q8_rows(t->type, w.data(), cols, 0, rows, q.data(), qd.data(), kTok, cpu.data(), rows));
  for (uint32_t n = 1; n <= kTok; n++) {
    const int32_t cnt = (int32_t) n;
    CHECK(hipMemcpy(dcnt, &cnt, 4, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(q8_group_tok(t->type, (const void* const*) dp, dcnt, dxrow, 1, n, rows, cols, dq, dd, dy, rows, nullptr) &&
          hipDeviceSynchronize() == hipSuccess);
    CHECK(hipMemcpy(gpu.data(), dy, (size_t) n * rows * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(std::memcmp(gpu.data(), cpu.data(), (size_t) n * rows * 4) == 0);
  }
  const uint32_t a = rows / 3 + 5, b = rows - 7;  // the CPU's rows in three uneven parts, each token alone
  for (uint32_t tk = 0; tk < kTok; tk++)
    for (uint32_t r0 : {0u, a, b}) {
      const uint32_t r1 = r0 == 0 ? a : r0 == a ? b : rows;
      CHECK(cpu_q8_rows(t->type, w.data(), cols, r0, r1, &q[(size_t) tk * cols], &qd[(size_t) tk * nb], 1,
                        &part[(size_t) tk * rows], rows));
    }
  CHECK(std::memcmp(part.data(), cpu.data(), cpu.size() * 4) == 0);
  CHECK(hipFree(dw) == hipSuccess && hipFree(dp) == hipSuccess && hipFree(dx) == hipSuccess &&
        hipFree(dq) == hipSuccess && hipFree(dd) == hipSuccess && hipFree(dy) == hipSuccess &&
        hipFree(dcnt) == hipSuccess && hipFree(dxrow) == hipSuccess);
  std::printf("  %-28s[%3d] type %2u %4u x %4u: the GPU's bytes and bits at 1 .. %u tokens; rows split the same\n",
              name, e, t->type, rows, cols, kTok);
}

int main() {
  std::printf("[test_cpu_q8_gpu] start\n");
  const char* home = std::getenv("HOME");
  char path[512];
  std::snprintf(path, sizeof path, "%s/models/ds1017/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf",
                home ? home : "");
  int ndev = 0;
  if (access(path, R_OK) != 0 || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0 || !cpu_q8_ready()) {
    std::printf("[test_cpu_q8_gpu] skipped: no ds1017, no GPU or no AVX-512 VNNI\n");
    return 77;
  }
  char err[GGUF_ERR] = "";
  Gguf* g = gguf_create(path, err);
  CHECK(g);
  for (const char* name : {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight", "blk.10.ffn_gate_exps.weight",
                           "blk.10.ffn_down_exps.weight", "blk.17.ffn_gate_exps.weight", "blk.17.ffn_down_exps.weight",
                           "blk.23.ffn_down_exps.weight"})
    for (int e : {7, 300}) expert(g, name, e);
  gguf_destroy(g);
  std::printf("[test_cpu_q8_gpu] ok\n");
  return 0;
}
