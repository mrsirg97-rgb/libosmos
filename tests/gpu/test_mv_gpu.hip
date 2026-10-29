// the decode's matrix-vector products on ds1017's own matrices (an expert's slice as the pool cuts it): within the
// rounding bound of its order against a double-precision sum over the exact weights (quant_dequant, bit for bit
// gguf-py's); a token's y the same bits alone or among 8 (schedule invariance) and every run (determinism); and the
// bandwidth, over enough copies of the matrix that the cache holds none (cycled through 512 MB). needs a gfx1151 and
// ~/models/ds1017; skipped (77) without them.
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "substrate/gpu/mv.h"
#include "substrate/gpu/quant.h"
#include "substrate/model/gguf.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr int kNt = MV_MAX_TOKENS;
constexpr uint32_t kCheckRows = 1024;
constexpr uint64_t kCycleBytes = 512ull << 20;
constexpr int kTimed = 50;
constexpr double kU = 1.0 / (1 << 24);  // F32's unit roundoff
constexpr int kWave = 32, kTree = 5;     // mv.hip's lanes and their tree's depth

struct Shape {
  const char* tensor;
  int expert;  // -1: the whole tensor
};

static const Shape kShapes[] = {
  {"blk.0.attn_q_a.weight", -1},     {"blk.0.attn_kv.weight", -1},      {"blk.0.attn_q_b.weight", -1},
  {"blk.0.attn_output_b.weight", -1}, {"blk.0.ffn_gate_inp.weight", -1}, {"blk.0.hc_attn_fn.weight", -1},
  {"blk.0.ffn_gate_shexp.weight", -1}, {"blk.0.ffn_down_shexp.weight", -1}, {"blk.0.ffn_gate_exps.weight", 7},
  {"blk.0.ffn_down_exps.weight", 7}, {"blk.10.ffn_gate_exps.weight", 7}, {"blk.7.ffn_down_exps.weight", 7},
  {"blk.17.ffn_gate_exps.weight", 7}, {"blk.23.ffn_down_exps.weight", 7}, {"output.weight", -1},
};

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) ((rng >> 8) * (1.0 / (1 << 24)) * 2.0 - 1.0);
}

// FNV-1a over y's bytes: a kernel's change that keeps every bit keeps this
static uint64_t fnv(const std::vector<float>& v) {
  uint64_t h = 1469598103934665603ull;
  const uint8_t* p = (const uint8_t*) v.data();
  for (size_t i = 0; i < v.size() * 4; i++) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

static void run(uint32_t type, const void* w, uint32_t rows, uint32_t cols, const float* x, uint32_t nt, float* y) {
  CHECK(mv_run(type, w, rows, cols, x, nt, y, nullptr));
}

static void shape(const Gguf* g, const Shape& s) {
  const GgufTensor* t = gguf_tensor(g, s.tensor);
  CHECK(t);
  const uint32_t cols = (uint32_t) t->dims[0], rows = (uint32_t) t->dims[1];
  const uint64_t bytes = s.expert < 0 ? t->bytes : t->bytes / t->dims[2];
  const uint64_t at = t->offset + (s.expert < 0 ? 0 : (uint64_t) s.expert * bytes);
  std::vector<uint8_t> host(bytes);
  const int fd = open(g->paths[t->file], O_RDONLY);
  CHECK(fd >= 0 && pread(fd, host.data(), bytes, (off_t) at) == (ssize_t) bytes);
  close(fd);

  const int copies = (int) (bytes >= kCycleBytes ? 1 : kCycleBytes / bytes);
  uint8_t* w = nullptr;
  CHECK(hipMalloc(&w, bytes * copies) == hipSuccess);
  for (int i = 0; i < copies; i++)
    CHECK(hipMemcpy(w + i * bytes, host.data(), bytes, hipMemcpyHostToDevice) == hipSuccess);
  std::vector<float> xh((size_t) kNt * cols);
  for (float& v : xh) v = uniform();
  float *x = nullptr, *y = nullptr, *y1 = nullptr;
  CHECK(hipMalloc(&x, xh.size() * 4) == hipSuccess && hipMalloc(&y, (size_t) kNt * rows * 4) == hipSuccess);
  CHECK(hipMalloc(&y1, (size_t) rows * 4) == hipSuccess);
  CHECK(hipMemcpy(x, xh.data(), xh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);

  // 8 tokens, twice; then each alone: all the same bits
  std::vector<float> a((size_t) kNt * rows), b(a.size()), one(rows);
  run(t->type, w, rows, cols, x, kNt, y);
  CHECK(hipMemcpy(a.data(), y, a.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemset(y, 0x7F, a.size() * 4) == hipSuccess);
  run(t->type, w, rows, cols, x, kNt, y);
  CHECK(hipMemcpy(b.data(), y, b.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(std::memcmp(a.data(), b.data(), a.size() * 4) == 0);
  for (int k = 0; k < kNt; k++) {
    run(t->type, w, rows, cols, x + (size_t) k * cols, 1, y1);
    CHECK(hipMemcpy(one.data(), y1, rows * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(std::memcmp(one.data(), a.data() + (size_t) k * rows, rows * 4) == 0);
  }

  // against a double sum over the exact weights, the first rows: |err| <= (terms a lane + tree + 1) u sum |w x|
  const uint32_t n = rows < kCheckRows ? rows : kCheckRows;
  std::vector<float> wf((size_t) n * cols);
  if (t->type == 0) {
    std::memcpy(wf.data(), host.data(), wf.size() * 4);
  } else {
    float* d = nullptr;
    CHECK(hipMalloc(&d, wf.size() * 4) == hipSuccess);
    CHECK(quant_dequant(t->type, w, (uint64_t) n * cols, d, nullptr) && hipDeviceSynchronize() == hipSuccess);
    CHECK(hipMemcpy(wf.data(), d, wf.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(hipFree(d) == hipSuccess);
  }
  const double depth = (double) ((cols / 8 + kWave - 1) / kWave * 8 + kTree + 1);
  double worst = 0;
  for (int k = 0; k < kNt; k++)
    for (uint32_t r = 0; r < n; r++) {
      double sum = 0, mag = 0;
      for (uint32_t c = 0; c < cols; c++) {
        const double p = (double) wf[(size_t) r * cols + c] * xh[(size_t) k * cols + c];
        sum += p, mag += std::fabs(p);
      }
      const double err = std::fabs(a[(size_t) k * rows + r] - sum), bound = depth * kU * mag;
      CHECK(std::isfinite(a[(size_t) k * rows + r]) && err <= bound);
      if (mag > 0 && err / (kU * mag) > worst) worst = err / (kU * mag);
    }

  // bandwidth: kTimed products, each on the next copy
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0) == hipSuccess && hipEventCreate(&e1) == hipSuccess);
  float ms[2] = {0, 0};
  const uint32_t nts[2] = {1, (uint32_t) kNt};
  for (int m = 0; m < 2; m++) {
    for (int i = 0; i < copies && i < kTimed; i++) run(t->type, w + i * bytes, rows, cols, x, nts[m], y);
    CHECK(hipEventRecord(e0, nullptr) == hipSuccess);
    for (int i = 0; i < kTimed; i++) run(t->type, w + (i % copies) * bytes, rows, cols, x, nts[m], y);
    CHECK(hipEventRecord(e1, nullptr) == hipSuccess && hipEventSynchronize(e1) == hipSuccess);
    CHECK(hipEventElapsedTime(&ms[m], e0, e1) == hipSuccess);
  }
  const double us1 = ms[0] * 1e3 / kTimed, us8 = ms[1] * 1e3 / kTimed;
  char name[80];
  std::snprintf(name, sizeof name, s.expert < 0 ? "%s" : "%s[%d]", s.tensor, s.expert);
  std::printf("  %-31s %6u x %-6u %7.2f MB  err <= %3.1f u|wx| (bound %3.0f)  y %016llx  1 tok %7.1f us %5.1f GB/s"
              "  8 tok %7.1f us %5.1f GB/s\n", name, cols, rows, bytes / 1e6, worst, depth, (unsigned long long) fnv(a),
              us1, bytes / us1 / 1e3, us8, bytes / us8 / 1e3);
  CHECK(hipEventDestroy(e0) == hipSuccess && hipEventDestroy(e1) == hipSuccess);
  CHECK(hipFree(w) == hipSuccess && hipFree(x) == hipSuccess && hipFree(y) == hipSuccess && hipFree(y1) == hipSuccess);
}

int main() {
  std::printf("[test_mv_gpu] start\n");
  const char* home = std::getenv("HOME");
  char path[512];
  std::snprintf(path, sizeof path, "%s/models/ds1017/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf",
                home ? home : "");
  int ndev = 0;
  if (access(path, R_OK) != 0 || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_mv_gpu] skipped: no ds1017 or no GPU\n");
    return 77;
  }
  CHECK(!mv_run(99, nullptr, 8, 256, nullptr, 1, nullptr, nullptr));
  CHECK(!mv_run(13, nullptr, 8, 100 * 8, nullptr, 1, nullptr, nullptr));  // 800 is not whole 256-value blocks
  CHECK(!mv_run(8, nullptr, 8, 256, nullptr, 0, nullptr, nullptr) && !mv_run(8, nullptr, 8, 256, nullptr, 9, nullptr,
                                                                                nullptr));
  std::printf("  refusals: unknown type, cols not whole blocks, 0 or 9 tokens\n");
  char err[GGUF_ERR] = "";
  Gguf* g = gguf_create(path, err);
  CHECK(g);
  for (const Shape& s : kShapes) shape(g, s);
  gguf_destroy(g);
  std::printf("[test_mv_gpu] ok\n");
  return 0;
}
