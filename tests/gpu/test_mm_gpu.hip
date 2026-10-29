// the quantized weights' matmul (mm) on ds1017's own matrices: a token's outputs the same bits alone, among 16 and
// among 16 reordered (schedule invariance); within the rounding bound of its order against a double sum over the
// f16 weights and activations it reads; the f16 roundings' own cost against the exact weights and F32 activations;
// and the
// bandwidth at 1, 4 and 16 tokens beside the matvec's at 1. needs a gfx1151 and ~/models/ds1017; skipped (77)
// without them.
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "substrate/gpu/mm.h"
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

constexpr int kT = MM_TILE;
constexpr int kBig = MM_MAX_TOKENS;  // a prompt's chunk: a token's bits the same in it
constexpr uint32_t kCheckRows = 512;
constexpr uint64_t kCycleBytes = 512ull << 20;
constexpr int kTimed = 30;
constexpr double kU = 1.0 / (1 << 24);

struct Shape {
  const char* tensor;
  int expert;
};

static const Shape kShapes[] = {
  {"blk.0.attn_q_a.weight", -1},         {"blk.0.attn_q_b.weight", -1},        {"blk.0.attn_output_b.weight", -1},
  {"blk.0.ffn_gate_shexp.weight", -1},   {"blk.0.ffn_down_shexp.weight", -1},  {"blk.0.ffn_gate_exps.weight", 7},
  {"blk.0.ffn_down_exps.weight", 7},     {"blk.10.ffn_gate_exps.weight", 7},   {"blk.17.ffn_gate_exps.weight", 7},
  {"output.weight", -1},
};

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) ((rng >> 8) * (1.0 / (1 << 24)) * 2.0 - 1.0);
}

template <class K> static double time_us(K run) {
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a) == hipSuccess && hipEventCreate(&b) == hipSuccess);
  for (int i = 0; i < kTimed; i++) run(i);  // the clocks up before the measure
  CHECK(hipEventRecord(a, nullptr) == hipSuccess);
  for (int i = 0; i < kTimed; i++) run(i);
  CHECK(hipEventRecord(b, nullptr) == hipSuccess && hipEventSynchronize(b) == hipSuccess);
  float ms;
  CHECK(hipEventElapsedTime(&ms, a, b) == hipSuccess);
  return ms * 1e3 / kTimed;
}

static void shape(const Gguf* g, const Shape& s) {
  const GgufTensor* t = gguf_tensor(g, s.tensor);
  CHECK(t && mm_known(t->type));
  const uint32_t cols = (uint32_t) t->dims[0], rows = (uint32_t) t->dims[1];
  const uint64_t bytes = s.expert < 0 ? t->bytes : t->bytes / t->dims[2];
  std::vector<uint8_t> host(bytes);
  const int fd = open(g->paths[t->file], O_RDONLY);
  CHECK(fd >= 0 && pread(fd, host.data(), bytes, (off_t) (t->offset + (s.expert < 0 ? 0 : s.expert * bytes))) ==
                       (ssize_t) bytes);
  close(fd);
  const int copies = (int) (bytes >= kCycleBytes ? 1 : kCycleBytes / bytes);
  uint8_t* w = nullptr;
  CHECK(hipMalloc(&w, bytes * copies) == hipSuccess);
  for (int i = 0; i < copies; i++)
    CHECK(hipMemcpy(w + i * bytes, host.data(), bytes, hipMemcpyHostToDevice) == hipSuccess);
  std::vector<float> xh((size_t) kT * cols), xr((size_t) kT * cols);
  for (float& v : xh) v = uniform();
  for (int tk = 0; tk < kT; tk++)  // reversed: token tk at row kT - 1 - tk
    std::memcpy(&xr[(size_t) (kT - 1 - tk) * cols], &xh[(size_t) tk * cols], cols * 4);
  float *x, *xrv, *y, *y1;
  void* x16;
  int32_t* over;
  CHECK(hipMalloc(&x, xh.size() * 4) == hipSuccess && hipMalloc(&xrv, xh.size() * 4) == hipSuccess);
  CHECK(hipMalloc(&y, (size_t) kT * rows * 4) == hipSuccess && hipMalloc(&y1, (size_t) kT * rows * 4) == hipSuccess);
  CHECK(hipMalloc(&x16, (size_t) kT * cols * 2) == hipSuccess);
  CHECK(hipMalloc(&over, 4) == hipSuccess && hipMemset(over, 0, 4) == hipSuccess);
  CHECK(hipMemcpy(x, xh.data(), xh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(xrv, xr.data(), xr.size() * 4, hipMemcpyHostToDevice) == hipSuccess);

  // 16 at once; reversed; each alone: the same bits
  std::vector<float> a((size_t) kT * rows), r(a.size()), one(rows);
  CHECK(mm_x16(x, cols, kT, cols, x16, over, nullptr) && mm_run(t->type, w, rows, cols, x16, kT, y, rows, nullptr));
  CHECK(hipMemcpy(a.data(), y, a.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(mm_x16(xrv, cols, kT, cols, x16, over, nullptr) && mm_run(t->type, w, rows, cols, x16, kT, y, rows, nullptr));
  CHECK(hipMemcpy(r.data(), y, r.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  for (int tk = 0; tk < kT; tk++) {
    CHECK(std::memcmp(&a[(size_t) tk * rows], &r[(size_t) (kT - 1 - tk) * rows], rows * 4) == 0);
    CHECK(mm_x16(x + (size_t) tk * cols, cols, 1, cols, x16, over, nullptr) &&
          mm_run(t->type, w, rows, cols, x16, 1, y1, rows, nullptr));
    CHECK(hipMemcpy(one.data(), y1, rows * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(std::memcmp(one.data(), &a[(size_t) tk * rows], rows * 4) == 0);
  }
  {  // kBig tokens, the first kT the same x: those tokens' rows the same bits as among 16 (and token 3 at 200, too)
    std::vector<float> big((size_t) kBig * cols);
    for (size_t i = 0; i < big.size(); i++) big[i] = uniform();
    std::memcpy(big.data(), xh.data(), xh.size() * 4);
    std::memcpy(&big[(size_t) 200 * cols], &xh[(size_t) 3 * cols], cols * 4);
    float *xb, *yb;
    void* x16b;
    CHECK(hipMalloc(&xb, big.size() * 4) == hipSuccess && hipMalloc(&yb, (size_t) kBig * rows * 4) == hipSuccess);
    CHECK(hipMalloc(&x16b, (size_t) kBig * cols * 2) == hipSuccess);
    CHECK(hipMemcpy(xb, big.data(), big.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
    for (int nb : {kBig, 64, 33}) {
      CHECK(mm_x16(xb, cols, (uint32_t) std::max(nb, 201), cols, x16b, over, nullptr));
      CHECK(mm_run(t->type, w, rows, cols, x16b, (uint32_t) std::max(nb, 201), yb, rows, nullptr));
      std::vector<float> yh((size_t) std::max(nb, 201) * rows);
      CHECK(hipMemcpy(yh.data(), yb, yh.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
      CHECK(std::memcmp(yh.data(), a.data(), (size_t) kT * rows * 4) == 0);
      CHECK(std::memcmp(&yh[(size_t) 200 * rows], &a[(size_t) 3 * rows], rows * 4) == 0);
    }
    CHECK(hipFree(xb) == hipSuccess && hipFree(yb) == hipSuccess && hipFree(x16b) == hipSuccess);
  }
  int32_t overflow = 0;
  CHECK(hipMemcpy(&overflow, over, 4, hipMemcpyDeviceToHost) == hipSuccess && overflow == 0);

  // against double over the exact weights: x as f16 (what mm reads; the bound), and x as F32 (the f16's cost)
  const uint32_t n = std::min(rows, kCheckRows);
  std::vector<float> wf((size_t) n * cols);
  float* d = nullptr;
  CHECK(hipMalloc(&d, wf.size() * 4) == hipSuccess);
  CHECK(quant_dequant(t->type, w, (uint64_t) n * cols, d, nullptr) && hipDeviceSynchronize() == hipSuccess);
  CHECK(hipMemcpy(wf.data(), d, wf.size() * 4, hipMemcpyDeviceToHost) == hipSuccess && hipFree(d) == hipSuccess);
  const double depth = (double) (cols / 16 + 16 + 4);
  double worst = 0, f16_rms = 0, ref_rms = 0;
  for (int tk = 0; tk < kT; tk++)
    for (uint32_t row = 0; row < n; row++) {
      double s16 = 0, s32 = 0, mag = 0;
      for (uint32_t c = 0; c < cols; c++) {
        const double wv = wf[(size_t) row * cols + c], xv = xh[(size_t) tk * cols + c];
        const double wv16 = (double) (float) (_Float16) wf[(size_t) row * cols + c];
        const double xv16 = (double) (float) (_Float16) xv;
        s16 += wv16 * xv16, s32 += wv * xv, mag += std::fabs(wv16 * xv16);
      }
      const double got = a[(size_t) tk * rows + row], err = std::fabs(got - s16);
      CHECK(std::isfinite(got) && err <= depth * kU * mag);
      if (mag > 0) worst = std::max(worst, err / (kU * mag));
      f16_rms += (got - s32) * (got - s32), ref_rms += s32 * s32;
    }

  // bandwidth: mm at 1, 4, 16 tokens; mv at 1
  const uint32_t ns[4] = {1, 4, (uint32_t) kT, (uint32_t) kBig};
  double us[4];
  void* x16t = nullptr;
  float* yt = nullptr;
  CHECK(hipMalloc(&x16t, (size_t) kBig * cols * 2) == hipSuccess &&
        hipMalloc(&yt, (size_t) kBig * rows * 4) == hipSuccess);
  CHECK(hipMemset(x16t, 0, (size_t) kBig * cols * 2) == hipSuccess);
  for (int k = 3; k >= 0; k--)
    us[k] = time_us([&](int i) {
      CHECK(mm_run(t->type, w + (i % copies) * bytes, rows, cols, x16t, ns[k], yt, rows, nullptr));
    });
  CHECK(hipFree(x16t) == hipSuccess && hipFree(yt) == hipSuccess);
  const double mv_us =
      time_us([&](int i) { CHECK(mv_run(t->type, w + (i % copies) * bytes, rows, cols, x, 1, y, nullptr)); });
  char name[80];
  std::snprintf(name, sizeof name, s.expert < 0 ? "%s" : "%s[%d]", s.tensor, s.expert);
  std::printf("  %-31s %5u x %-6u err <= %4.1f u|wx| (bound %4.0f), f16 %.1e rms | GB/s: mv 1 %5.1f, mm 1 %5.1f, "
              "4 %5.1f, 16 %5.1f (%.1f tokens a mv pass), 256 %5.1f (%.0f)\n", name, cols, rows, worst, depth,
              std::sqrt(f16_rms / ref_rms), bytes / mv_us / 1e3, bytes / us[0] / 1e3, bytes / us[1] / 1e3,
              bytes / us[2] / 1e3, kT * mv_us / us[2], bytes / us[3] / 1e3, kBig * mv_us / us[3]);
  CHECK(hipFree(w) == hipSuccess && hipFree(x) == hipSuccess && hipFree(xrv) == hipSuccess);
  CHECK(hipFree(y) == hipSuccess && hipFree(y1) == hipSuccess);
  CHECK(hipFree(x16) == hipSuccess && hipFree(over) == hipSuccess);
}

int main() {
  std::printf("[test_mm_gpu] start\n");
  const char* home = std::getenv("HOME");
  char path[512];
  std::snprintf(path, sizeof path, "%s/models/ds1017/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf",
                home ? home : "");
  int ndev = 0;
  if (access(path, R_OK) != 0 || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_mm_gpu] skipped: no ds1017 or no GPU\n");
    return 77;
  }
  CHECK(!mm_known(0) && !mm_known(30) && mm_known(29));
  CHECK(!mm_run(13, nullptr, 24, 256, nullptr, 1, nullptr, 24, nullptr));   // rows not 16s
  CHECK(!mm_run(13, nullptr, 32, 256, nullptr, MM_MAX_TOKENS + 1, nullptr, 32, nullptr));  // too many tokens
  std::printf("  refusals: F32 and BF16 (the matvec's), rows not a multiple of 16, %d tokens\n", MM_MAX_TOKENS + 1);
  char err[GGUF_ERR] = "";
  Gguf* g = gguf_create(path, err);
  CHECK(g);
  for (const Shape& s : kShapes) shape(g, s);
  gguf_destroy(g);
  std::printf("[test_mm_gpu] ok: every token the same bits alone, among 16, among 16 reversed, and among 33, 64 and "
              "%d (at its place and at another)\n", kBig);
  return 0;
}
