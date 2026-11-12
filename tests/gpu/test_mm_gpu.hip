// the quantized weights' matmul (mm) on ds1017's own matrices: a token's outputs the same bits alone, among 16 and
// among 16 reordered (schedule invariance); within the rounding bound of its order against a double sum over the
// f16 weights and activations it reads; the f16 roundings' own cost against the exact weights and F32 activations;
// and the
// bandwidth at 1, 4 and 16 tokens beside the matvec's at 1 - and Qwen3.8's mixers' (a K split across blocks) where
// its shards are. needs a gfx1151 and ~/models/ds1017; skipped (77) without them.
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

#include "test_model.h"

extern "C" {
#include "osmos/gpu/mm.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/gpu/mv.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/quant.h"
#include "osmos/model/gguf.h"
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

static const Shape kUdShapes[] = {
  {"blk.0.attn_q_a.weight", -1},        {"blk.0.attn_q_b.weight", -1},         {"blk.0.attn_output_b.weight", -1},
  {"blk.0.ffn_gate_shexp.weight", -1},  {"blk.27.ffn_down_shexp.weight", -1},  {"blk.0.ffn_gate_exps.weight", 7},
  {"blk.0.ffn_down_exps.weight", 7},    {"blk.20.ffn_gate_exps.weight", 7},    {"blk.38.ffn_gate_exps.weight", 7},
  {"blk.20.ffn_down_exps.weight", 7},   {"output.weight", -1},
};

// Qwen3.8's (when its first shard is here): the mixers' down - Q8_0 320 x 10240, its K split across blocks (mm.hip's
// split_parts) - and up (320 columns: the odd width's shape)
static const Shape kQwenShapes[] = {
    {"blk.0.hc_attn_down.weight", -1}, {"blk.0.hc_attn_up.weight", -1},
    // UD-IQ3_XXS's expert types: IQ3_XXS gate, IQ4_NL / Q2_0 down, IQ4_XS gate
    {"blk.0.ffn_gate_exps.weight", 7}, {"blk.0.ffn_down_exps.weight", 7}, {"blk.8.ffn_down_exps.weight", 7},
    {"blk.47.ffn_gate_exps.weight", 7}, {"blk.14.ffn_gate_exps.weight", 7}};
// DeepSeek UD-IQ2_XXS's expert types UD-IQ1_M lacks: Q2_K and IQ3_XXS downs
constexpr const char* kUdX2 = "%s/Models/ds1017/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf";
static const Shape kUdX2Shapes[] = {{"blk.0.ffn_down_exps.weight", 7}, {"blk.20.ffn_down_exps.weight", 7}};
// the reference's experts as deepseek-ai stores them (tools/convert/ds4_experts_mxfp4.py): MXFP4 gate / up and down,
// its first shard alone (gguf_header: layers 0-4)
constexpr const char* kRef = "%s/Models/ds1017/ref/ds41-experts-mxfp4-00001-of-00009.gguf";
static const Shape kRefShapes[] = {{"blk.0.ffn_gate_exps.weight", 7}, {"blk.0.ffn_down_exps.weight", 7},
                                   {"blk.4.ffn_up_exps.weight", 300}};
constexpr const char* kQwenFirst = "%s/Models/qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf";
// UD-Q4_K_XL's expert types (Q4_K gate / up, Q5_1 down), from several layers
static const Shape kQwenQ4Shapes[] = {
    {"blk.0.ffn_gate_exps.weight", 7}, {"blk.30.ffn_gate_exps.weight", 300}, {"blk.47.ffn_up_exps.weight", 11},
    {"blk.0.ffn_down_exps.weight", 7}, {"blk.0.ffn_down_exps.weight", 300}, {"blk.2.ffn_gate_exps.weight", 7}};
constexpr const char* kQwenQ4 = "%s/Models/qwen38/q4kxl-dl/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";

static std::vector<float> host_of(const float* d, size_t n) {
  std::vector<float> h(n);
  CHECK(hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return h;
}
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
    if (s.expert >= 0) {  // the experts' launch (mm_group: one matrix, its multi-tile plans - mm.hip's GroupPlan): the
                          // first n tokens' rows mm_run's bits at 1 .. 8 column tiles, a partial tile and a partial block
      std::vector<float> yr((size_t) 201 * rows);  // x16b and yb: the last pass's 201 tokens, mm_run's
      CHECK(hipMemcpy(yr.data(), yb, yr.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
      const void** dw;
      int32_t* dc;
      uint32_t *dx, *dy;
      float* yg;
      CHECK(hipMalloc(&dw, sizeof(void*)) == hipSuccess && hipMalloc(&dc, 4) == hipSuccess &&
            hipMalloc(&dx, 4) == hipSuccess && hipMalloc(&dy, 4) == hipSuccess &&
            hipMalloc(&yg, (size_t) 128 * rows * 4) == hipSuccess);
      const void* wp = w;
      const uint32_t zero = 0;
      CHECK(hipMemcpy(dw, &wp, sizeof wp, hipMemcpyHostToDevice) == hipSuccess &&
            hipMemcpy(dx, &zero, 4, hipMemcpyHostToDevice) == hipSuccess &&
            hipMemcpy(dy, &zero, 4, hipMemcpyHostToDevice) == hipSuccess);
      for (int32_t n : {24, 37, 64, 100, 128}) {
        CHECK(hipMemcpy(dc, &n, 4, hipMemcpyHostToDevice) == hipSuccess && hipMemset(yg, 0xff, (size_t) 128 * rows * 4) == hipSuccess);
        CHECK(mm_group(t->type, dw, dc, 1, (uint32_t) n, rows, cols, x16b, dx, cols, yg, dy, rows, nullptr));
        CHECK(host_of(yg, (size_t) n * rows) == std::vector<float>(yr.begin(), yr.begin() + (size_t) n * rows));
      }
      CHECK(hipFree(dw) == hipSuccess && hipFree(dc) == hipSuccess && hipFree(dx) == hipSuccess &&
            hipFree(dy) == hipSuccess && hipFree(yg) == hipSuccess);
      std::printf("  %s: mm_group at 24 / 37 / 64 / 100 / 128 tokens, mm_run's bits\n", s.tensor);
    }
    CHECK(hipFree(xb) == hipSuccess && hipFree(yb) == hipSuccess && hipFree(x16b) == hipSuccess);
  }
  if (rows > 2 * MM_TILE) {  // a pair in one launch (mm_run2): the matrix's first rows - 16 and its last rows - 32,
                            // each mm_run's bits (rows of their own lengths)
    const uint32_t r0 = rows - MM_TILE, r1 = rows - 2 * MM_TILE;
    const uint8_t* w1 = w + bytes / rows * 2 * MM_TILE;
    float* y2;
    CHECK(hipMalloc(&y2, (size_t) kT * r1 * 4) == hipSuccess);
    CHECK(mm_x16(x, cols, kT, cols, x16, over, nullptr) &&
          mm_run2(t->type, w, r0, y, rows, w1, r1, y2, r1, cols, x16, kT, nullptr));
    std::vector<float> p0(a.size()), p1((size_t) kT * r1);
    CHECK(hipMemcpy(p0.data(), y, p0.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(hipMemcpy(p1.data(), y2, p1.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    for (int tk = 0; tk < kT; tk++) {
      CHECK(std::memcmp(&p0[(size_t) tk * rows], &a[(size_t) tk * rows], r0 * 4) == 0);
      CHECK(std::memcmp(&p1[(size_t) tk * r1], &a[(size_t) tk * rows + 2 * MM_TILE], r1 * 4) == 0);
    }
    if (t->type == 8) {  // mm_run_silu: mm_run then q4x_scale_silu's bits, y and the f16 written, at 1, 16 and 64 tokens
      constexpr float kScale = 0.25f;
      float *ys1, *ys2;
      void *xo1, *xo2;
      void* x16w;
      CHECK(hipMalloc(&ys1, (size_t) 64 * rows * 4) == hipSuccess && hipMalloc(&ys2, (size_t) 64 * rows * 4) == hipSuccess);
      CHECK(hipMalloc(&xo1, (size_t) 64 * rows * 2) == hipSuccess && hipMalloc(&xo2, (size_t) 64 * rows * 2) == hipSuccess);
      CHECK(hipMalloc(&x16w, (size_t) 64 * cols * 2) == hipSuccess);
      std::vector<float> xw((size_t) 64 * cols);
      for (float& v : xw) v = uniform();
      float* xd;
      CHECK(hipMalloc(&xd, xw.size() * 4) == hipSuccess &&
            hipMemcpy(xd, xw.data(), xw.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
      for (uint32_t nt : {1u, 16u, 64u}) {
        CHECK(mm_x16(xd, cols, nt, cols, x16w, over, nullptr));
        CHECK(hipMemset(xo1, 0, (size_t) 64 * rows * 2) == hipSuccess && hipMemset(xo2, 0, (size_t) 64 * rows * 2) == hipSuccess);
        CHECK(mm_run_silu(t->type, w, rows, cols, x16w, nt, ys1, rows, kScale, xo1, over, nullptr));
        const Q4xX16 side = {xo2, over, rows, 0};
        CHECK(mm_run(t->type, w, rows, cols, x16w, nt, ys2, rows, nullptr) &&
              q4x_scale_silu(ys2, (uint64_t) nt * rows, kScale, &side, nullptr));
        CHECK(host_of(ys1, (size_t) nt * rows) == host_of(ys2, (size_t) nt * rows));
        std::vector<uint16_t> h1((size_t) 64 * rows), h2(h1.size());
        CHECK(hipMemcpy(h1.data(), xo1, h1.size() * 2, hipMemcpyDeviceToHost) == hipSuccess &&
              hipMemcpy(h2.data(), xo2, h2.size() * 2, hipMemcpyDeviceToHost) == hipSuccess);
        CHECK(h1 == h2);
      }
      CHECK(!mm_run_silu(t->type, w, rows, cols, x16w, 65, ys1, rows, kScale, xo1, over, nullptr));    // past 64
      CHECK(!mm_run_silu(t->type, w, rows, cols, x16w, 1, ys1, rows, kScale, x16w, over, nullptr));    // its own x16
      CHECK(!mm_run_silu(14, w, rows, cols, x16w, 1, ys1, rows, kScale, xo1, over, nullptr));           // no epilogue
      CHECK(hipFree(ys1) == hipSuccess && hipFree(ys2) == hipSuccess && hipFree(xo1) == hipSuccess &&
            hipFree(xo2) == hipSuccess && hipFree(x16w) == hipSuccess && hipFree(xd) == hipSuccess);
      std::printf("  %s: mm_run_silu the bits of mm_run then q4x_scale_silu (y and its f16), 1 / 16 / 64 tokens\n", s.tensor);
    }
    CHECK(!mm_run2(t->type, w, r0, y, rows, nullptr, r1, y2, r1, cols, x16, kT, nullptr));
    CHECK(!mm_run2(t->type, w, r0, y, rows, w1, r1, y2, r1 - 1, cols, x16, kT, nullptr));  // y1's rows too short
    CHECK(hipFree(y2) == hipSuccess);
  }
  double r_us = 0;
  if (t->type == 13) {  // Q5_K repacked (quant_repack_q5k): mm at 1 and 16 tokens and mv at 3, each Q5_K's bits
    const uint64_t rb = QUANT_Q5_KR_BYTES(rows, cols);
    uint8_t* wr = nullptr;
    CHECK(hipMalloc(&wr, rb * copies) == hipSuccess);
    for (int i = 0; i < copies; i++) CHECK(quant_repack_q5k(w + i * bytes, rows, cols, wr + i * rb, nullptr));
    CHECK(!quant_repack_q5k(w, rows + 8, cols, wr, nullptr) && !quant_repack_q5k(w, rows, cols, w, nullptr));
    std::vector<float> got(a.size()), ref(a.size());
    CHECK(mm_x16(x, cols, kT, cols, x16, over, nullptr) &&
          mm_run(QUANT_Q5_KR, wr, rows, cols, x16, kT, y, rows, nullptr));
    CHECK(hipMemcpy(got.data(), y, got.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(std::memcmp(got.data(), a.data(), a.size() * 4) == 0);
    CHECK(mm_x16(x + 5 * cols, cols, 1, cols, x16, over, nullptr) &&
          mm_run(QUANT_Q5_KR, wr, rows, cols, x16, 1, y1, rows, nullptr));
    CHECK(hipMemcpy(one.data(), y1, rows * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(std::memcmp(one.data(), &a[(size_t) 5 * rows], rows * 4) == 0);
    CHECK(mv_strided(13, w, rows, cols, x, cols, 3, y, rows, nullptr) &&
          hipMemcpy(ref.data(), y, (size_t) 3 * rows * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(mv_strided(QUANT_Q5_KR, wr, rows, cols, x, cols, 3, y, rows, nullptr) &&
          hipMemcpy(got.data(), y, (size_t) 3 * rows * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(std::memcmp(got.data(), ref.data(), (size_t) 3 * rows * 4) == 0);
    void* x16t = nullptr;
    CHECK(hipMalloc(&x16t, (size_t) MM_TILE * cols * 2) == hipSuccess &&
          hipMemset(x16t, 0, (size_t) MM_TILE * cols * 2) == hipSuccess);
    r_us = time_us([&](int i) {
      CHECK(mm_run(QUANT_Q5_KR, wr + (i % copies) * rb, rows, cols, x16t, 1, y1, rows, nullptr));
    });
    CHECK(hipFree(x16t) == hipSuccess && hipFree(wr) == hipSuccess);
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
              "4 %5.1f, 16 %5.1f (%.1f tokens a mv pass), 256 %5.1f (%.0f)", name, cols, rows, worst, depth,
              std::sqrt(f16_rms / ref_rms), bytes / mv_us / 1e3, bytes / us[0] / 1e3, bytes / us[1] / 1e3,
              bytes / us[2] / 1e3, kT * mv_us / us[2], bytes / us[3] / 1e3, kBig * mv_us / us[3]);
  if (r_us > 0) std::printf("; repacked mm 1 %5.1f (its bits Q5_K's)", bytes / r_us / 1e3);
  std::printf("\n");
  CHECK(hipFree(w) == hipSuccess && hipFree(x) == hipSuccess && hipFree(xrv) == hipSuccess);
  CHECK(hipFree(y) == hipSuccess && hipFree(y1) == hipSuccess);
  CHECK(hipFree(x16) == hipSuccess && hipFree(over) == hipSuccess);
}

// mm_x16_attn against the three calls it stands for (ops_rope's inverse, ops_fp8_round, mm_x16) on the attention's
// output shape: every f16 the same bits, the padding rows too. a block's values spread over 2^-12 .. 2^4 so its FP8
// scales differ, positions to 8K so the angles are large
static void attn_x16() {
  constexpr uint32_t kHeads = 64, kHd = 512, kRd = 64, kN = 37, kCols = kHeads * kHd;
  const uint32_t rows = (kN + MM_TILE - 1) / MM_TILE * MM_TILE;
  std::vector<float> xh((size_t) kN * kCols), fh(kRd / 2);
  for (float& v : xh) v = uniform() * std::ldexp(1.0f, (int) ((uniform() + 1) * 8) - 12);
  for (uint32_t i = 0; i < kRd / 2; i++) fh[i] = (float) std::pow(10000.0, -2.0 * i / kRd);
  std::vector<int32_t> ph(kN);
  for (uint32_t i = 0; i < kN; i++) ph[i] = (int32_t) (i * 223);
  float *x = nullptr, *fr = nullptr;
  int32_t *pos = nullptr, *ovf = nullptr;
  void *a = nullptr, *b = nullptr;
  const size_t xb = xh.size() * 4, hb = (size_t) rows * kCols * 2;
  CHECK(hipMalloc(&x, xb) == hipSuccess && hipMalloc(&fr, fh.size() * 4) == hipSuccess);
  CHECK(hipMalloc(&pos, kN * 4) == hipSuccess && hipMalloc(&ovf, 4) == hipSuccess);
  CHECK(hipMalloc(&a, hb) == hipSuccess && hipMalloc(&b, hb) == hipSuccess);
  CHECK(hipMemcpy(fr, fh.data(), fh.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(pos, ph.data(), kN * 4, hipMemcpyHostToDevice) == hipSuccess && hipMemset(ovf, 0, 4) == hipSuccess);
  CHECK(hipMemset(a, 0x55, hb) == hipSuccess && hipMemset(b, 0x33, hb) == hipSuccess);
  CHECK(hipMemcpy(x, xh.data(), xb, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(mm_x16_attn(x, kCols, kN, kCols, kHd, kRd, fr, pos, a, ovf, nullptr));
  CHECK(ops_rope(x, kHeads, kHd, kRd, kCols, fr, pos, true, kN, nullptr) &&
        ops_fp8_round(x, (uint64_t) kN * kCols, nullptr, 0, nullptr) && mm_x16(x, kCols, kN, kCols, b, ovf, nullptr));
  std::vector<uint8_t> ha(hb), hbv(hb);
  int32_t o = 1;
  CHECK(hipMemcpy(ha.data(), a, hb, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(hbv.data(), b, hb, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(&o, ovf, 4, hipMemcpyDeviceToHost) == hipSuccess && o == 0);
  CHECK(std::memcmp(ha.data(), hbv.data(), hb) == 0);
  CHECK(!mm_x16_attn(x, kCols, kN, kCols, kHd, kRd + 2, fr, pos, a, ovf, nullptr));  // the rope's part mid-block
  CHECK(hipFree(x) == hipSuccess && hipFree(fr) == hipSuccess && hipFree(pos) == hipSuccess);
  CHECK(hipFree(ovf) == hipSuccess && hipFree(a) == hipSuccess && hipFree(b) == hipSuccess);
  std::printf("  the attention's output to f16 (rope back, FP8, f16 at once): %u tokens x %u, the bits of the three"
              " calls\n", kN, kCols);
}

int main() {
  std::printf("[test_mm_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  const Shape* shapes = which == TestModel::kUd ? kUdShapes : kShapes;
  const size_t n_shapes = which == TestModel::kUd ? sizeof kUdShapes / sizeof kUdShapes[0] : sizeof kShapes / sizeof kShapes[0];
  int ndev = 0;
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_mm_gpu] skipped: no ds1017 or UD-IQ1_M, or no GPU\n");
    return 77;
  }
  CHECK(!mm_known(0) && !mm_known(30) && mm_known(29));
  CHECK(!mm_run(13, nullptr, 24, 256, nullptr, 1, nullptr, 24, nullptr));   // rows not 16s
  CHECK(!mm_run(13, nullptr, 32, 256, nullptr, MM_MAX_TOKENS + 1, nullptr, 32, nullptr));  // too many tokens
  std::printf("  refusals: F32 and BF16 (the matvec's), rows not a multiple of 16, %d tokens\n", MM_MAX_TOKENS + 1);
  attn_x16();
  char err[GGUF_ERR] = "";
  Gguf* g = gguf_create(path, err);
  CHECK(g);
  for (size_t i = 0; i < n_shapes; i++) shape(g, shapes[i]);
  gguf_destroy(g);
  char qpath[512];
  std::snprintf(qpath, sizeof qpath, kQwenFirst, std::getenv("HOME") ? std::getenv("HOME") : "");
  if (Gguf* q = access(qpath, R_OK) ? nullptr : gguf_create(qpath, err)) {
    for (const Shape& sh : kQwenShapes) shape(q, sh);
    gguf_destroy(q);
  } else {
    std::printf("  Qwen3.8's shapes skipped: no %s\n", qpath);
  }
  std::snprintf(qpath, sizeof qpath, kQwenQ4, std::getenv("HOME") ? std::getenv("HOME") : "");
  if (Gguf* q = access(qpath, R_OK) ? nullptr : gguf_create(qpath, err)) {
    for (const Shape& sh : kQwenQ4Shapes) shape(q, sh);
    gguf_destroy(q);
  } else {
    std::printf("  Qwen3.8 UD-Q4_K_XL's shapes skipped: no %s\n", qpath);
  }
  std::snprintf(qpath, sizeof qpath, kUdX2, std::getenv("HOME") ? std::getenv("HOME") : "");
  if (Gguf* q = access(qpath, R_OK) ? nullptr : gguf_create(qpath, err)) {
    for (const Shape& sh : kUdX2Shapes) shape(q, sh);
    gguf_destroy(q);
  } else {
    std::printf("  DeepSeek UD-IQ2_XXS's shapes skipped: no %s\n", qpath);
  }
  std::snprintf(qpath, sizeof qpath, kRef, std::getenv("HOME") ? std::getenv("HOME") : "");
  if (Gguf* q = access(qpath, R_OK) ? nullptr : gguf_header(qpath, err)) {
    for (const Shape& sh : kRefShapes) shape(q, sh);
    gguf_destroy(q);
  } else {
    std::printf("  the reference's MXFP4 shapes skipped: no %s\n", qpath);
  }
  std::printf("[test_mm_gpu] ok: every token the same bits alone, among 16, among 16 reversed, and among 33, 64 and "
              "%d (at its place and at another); a pair in one launch each its own bits; Q5_K repacked its bits\n",
              kBig);
  return 0;
}
