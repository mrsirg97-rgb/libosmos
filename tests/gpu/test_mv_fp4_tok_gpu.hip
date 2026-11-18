// mv_fp4_tok (osmos/gpu/mv.h): NVFP4 matrices each over its own tokens - random ggml NVFP4 blocks (finite UE4M3
// scales, every E2M1 code), Hy4-flash's expert shape (1,536 x 1,536) and two others (a 48- and a 96-block row): every
// value within a few ulps of the float64 sum (the blocks' values exact, their products exact: only the order of the
// sums differs), and a token's bits the same whatever its entry's count (1 .. 4) or the entries beside it.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" {
#include "osmos/gpu/mv.h"
}

#define CHECK(x)                                                           \
  do {                                                                     \
    if (!(x)) {                                                            \
      std::fprintf(stderr, "%s:%d: failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

constexpr int kBlock = 64, kBlockBytes = 36;
constexpr float kE2m1[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};  // doubled, as ggml's

static float ue4m3(uint8_t c) {  // ggml's: the sign not read, 0x7F zero
  c &= 0x7F;
  if (c == 0x7F) return 0;
  const int e = c >> 3, m = c & 7;
  return e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp((float) m, -9);
}

static double value(const uint8_t* row, uint32_t i) {  // value i of a row, as dequantize_row_nvfp4
  const uint8_t* b = row + (i / kBlock) * kBlockBytes;
  const uint32_t k = i % kBlock, s = k / 16, j = k % 16;
  const uint8_t q = b[4 + 8 * s + (j % 8)];
  return (double) kE2m1[j < 8 ? q & 15 : q >> 4] * ue4m3(b[s]) * 0.5;
}

static void run(uint32_t rows, uint32_t cols, uint32_t n_mats, std::mt19937& rng) {
  const uint64_t mat_bytes = (uint64_t) rows * cols / kBlock * kBlockBytes;
  std::vector<uint8_t> w(mat_bytes * n_mats);
  for (auto& b : w) b = (uint8_t) rng();
  for (uint64_t at = 0; at < w.size(); at += kBlockBytes)  // scales in a sane range (2^-9 .. 2^1)
    for (int s = 0; s < 4; s++) w[at + s] = (uint8_t) (8 + rng() % 56);
  // entries: matrix m's tokens - counts 1 .. 4 across the table, the same tokens again at count 1 each
  std::vector<int32_t> cnt, xrow;
  std::vector<uint32_t> mat_of;
  uint32_t rows_x = 0;
  for (uint32_t m = 0; m < n_mats; m++) {
    const int32_t c = 1 + (int32_t) (m % 4);
    cnt.push_back(c), xrow.push_back((int32_t) rows_x), mat_of.push_back(m), rows_x += (uint32_t) c;
  }
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> x((size_t) (rows_x + 4) * cols);
  for (auto& v : x) v = nd(rng);
  uint8_t* dw = nullptr;
  float *dx = nullptr, *dy = nullptr, *dy1 = nullptr;
  int32_t *dc = nullptr, *dr = nullptr;
  const void** dp = nullptr;
  CHECK(hipMalloc(&dw, w.size()) == hipSuccess && hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMalloc(&dx, x.size() * 4) == hipSuccess && hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMalloc(&dy, (size_t) rows_x * rows * 4) == hipSuccess && hipMalloc(&dy1, (size_t) rows_x * rows * 4) == hipSuccess);
  std::vector<const void*> ptr;
  for (uint32_t m : mat_of) ptr.push_back(dw + m * mat_bytes);
  CHECK(hipMalloc(&dp, ptr.size() * sizeof(void*)) == hipSuccess && hipMalloc(&dc, cnt.size() * 4) == hipSuccess &&
        hipMalloc(&dr, xrow.size() * 4) == hipSuccess);
  CHECK(hipMemcpy(dp, ptr.data(), ptr.size() * sizeof(void*), hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dc, cnt.data(), cnt.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dr, xrow.data(), xrow.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(mv_fp4_tok(dp, dc, dr, (uint32_t) cnt.size(), rows, cols, dx, cols, dy, rows, nullptr));
  // each token alone (count 1): its own entry
  std::vector<const void*> p1;
  std::vector<int32_t> c1, r1;
  for (size_t e = 0; e < cnt.size(); e++)
    for (int32_t t = 0; t < cnt[e]; t++) p1.push_back(ptr[e]), c1.push_back(1), r1.push_back(xrow[e] + t);
  const void** dp1 = nullptr;
  int32_t *dc1 = nullptr, *dr1 = nullptr;
  CHECK(hipMalloc(&dp1, p1.size() * sizeof(void*)) == hipSuccess && hipMalloc(&dc1, c1.size() * 4) == hipSuccess &&
        hipMalloc(&dr1, r1.size() * 4) == hipSuccess);
  CHECK(hipMemcpy(dp1, p1.data(), p1.size() * sizeof(void*), hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dc1, c1.data(), c1.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dr1, r1.data(), r1.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(mv_fp4_tok(dp1, dc1, dr1, (uint32_t) c1.size(), rows, cols, dx, cols, dy1, rows, nullptr));
  std::vector<float> y((size_t) rows_x * rows), y1(y.size());
  CHECK(hipDeviceSynchronize() == hipSuccess);
  CHECK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(y1.data(), dy1, y1.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(std::memcmp(y.data(), y1.data(), y.size() * 4) == 0);
  double worst = 0;
  for (size_t e = 0; e < cnt.size(); e++)
    for (int32_t t = 0; t < cnt[e]; t++)
      for (uint32_t r = 0; r < rows; r += 7) {
        const uint8_t* row = w.data() + mat_of[e] * mat_bytes + (uint64_t) r * cols / kBlock * kBlockBytes;
        double s = 0, mag = 0;
        for (uint32_t i = 0; i < cols; i++) {
          const double p = value(row, i) * x[(size_t) (xrow[e] + t) * cols + i];
          s += p, mag += std::fabs(p);
        }
        const double err = std::fabs(y[(size_t) (xrow[e] + t) * rows + r] - s) / (mag + 1e-30);
        worst = std::fmax(worst, err);
      }
  // F32 sums of `cols` exact products: each addition's rounding at most 2^-24 of the running magnitude
  CHECK(worst <= cols * 6e-8);
  std::printf("  %u x %u, %u matrices (counts 1 .. 4): error at most %.2g of the terms' magnitude; each token alone "
              "the same bits\n", rows, cols, n_mats, worst);
  for (void* p : {(void*) dw, (void*) dx, (void*) dy, (void*) dy1, (void*) dp, (void*) dc, (void*) dr, (void*) dp1,
                  (void*) dc1, (void*) dr1})
    CHECK(hipFree(p) == hipSuccess);
}

// mv_fp4_rows: one matrix over n tokens - every n's tokens the bits each takes alone, and mv_fp4_tok's at count 1
static void dense(uint32_t rows, uint32_t cols, std::mt19937& rng) {
  const uint64_t bytes = (uint64_t) rows * cols / kBlock * kBlockBytes;
  std::vector<uint8_t> w(bytes);
  for (auto& b : w) b = (uint8_t) rng();
  for (uint64_t at = 0; at < w.size(); at += kBlockBytes)
    for (int s = 0; s < 4; s++) w[at + s] = (uint8_t) (8 + rng() % 56);
  constexpr uint32_t kN = 13, kXs = 4;  // x's rows cols + kXs apart (a strided slice), y's rows + 3
  const uint32_t xs = cols + kXs, ys = rows + 3;
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> x((size_t) kN * xs);
  for (auto& v : x) v = nd(rng);
  uint8_t* dw = nullptr;
  float *dx = nullptr, *dy = nullptr, *dy1 = nullptr;
  CHECK(hipMalloc(&dw, bytes) == hipSuccess && hipMemcpy(dw, w.data(), bytes, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMalloc(&dx, x.size() * 4) == hipSuccess && hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMalloc(&dy, (size_t) kN * ys * 4) == hipSuccess && hipMalloc(&dy1, (size_t) kN * ys * 4) == hipSuccess);
  CHECK(hipMemset(dy, 0, (size_t) kN * ys * 4) == hipSuccess && hipMemset(dy1, 0, (size_t) kN * ys * 4) == hipSuccess);
  CHECK(mv_fp4_rows(dw, rows, cols, dx, xs, kN, dy, ys, nullptr));
  for (uint32_t t = 0; t < kN; t++) CHECK(mv_fp4_rows(dw, rows, cols, dx + (size_t) t * xs, xs, 1, dy1 + (size_t) t * ys, ys, nullptr));
  std::vector<float> y((size_t) kN * ys), y1(y.size()), y2(y.size());
  CHECK(hipDeviceSynchronize() == hipSuccess);
  CHECK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(y1.data(), dy1, y1.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(std::memcmp(y.data(), y1.data(), y.size() * 4) == 0);
  // mv_fp4_tok's: each token its own count-1 entry
  const void** dp = nullptr;
  int32_t *dc = nullptr, *dr = nullptr;
  std::vector<const void*> p(kN, dw);
  std::vector<int32_t> c(kN, 1), r(kN);
  for (uint32_t t = 0; t < kN; t++) r[t] = (int32_t) t;
  CHECK(hipMalloc(&dp, kN * sizeof(void*)) == hipSuccess && hipMalloc(&dc, kN * 4) == hipSuccess && hipMalloc(&dr, kN * 4) == hipSuccess);
  CHECK(hipMemcpy(dp, p.data(), kN * sizeof(void*), hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dc, c.data(), kN * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dr, r.data(), kN * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemset(dy1, 0, (size_t) kN * ys * 4) == hipSuccess);
  CHECK(mv_fp4_tok(dp, dc, dr, kN, rows, cols, dx, xs, dy1, ys, nullptr));
  CHECK(hipDeviceSynchronize() == hipSuccess);
  CHECK(hipMemcpy(y2.data(), dy1, y2.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(std::memcmp(y.data(), y2.data(), y.size() * 4) == 0);
  CHECK(!mv_fp4_rows(dw, rows, cols, dx, xs, 0, dy, ys, nullptr) && !mv_fp4_rows(dw, rows, cols + 1, dx, xs, 1, dy, ys, nullptr));
  std::printf("  dense %u x %u over %u tokens (x and y strided): each token alone and mv_fp4_tok's, bit for bit\n", rows,
              cols, kN);
  for (void* q : {(void*) dw, (void*) dx, (void*) dy, (void*) dy1, (void*) dp, (void*) dc, (void*) dr}) CHECK(hipFree(q) == hipSuccess);
}

int main() {
  int devs = 0;
  if (hipGetDeviceCount(&devs) != hipSuccess || !devs) return std::printf("[test_mv_fp4_tok_gpu] skipped: no GPU\n"), 77;
  std::printf("[test_mv_fp4_tok_gpu] start\n");
  std::mt19937 rng(1536);
  CHECK(!mv_fp4_tok(nullptr, nullptr, nullptr, 1, 16, 64, nullptr, 64, nullptr, 16, nullptr));
  run(1536, 1536, 16, rng);
  run(256, 3072, 8, rng);
  run(128, 6144, 4, rng);
  dense(3072, 16384, rng);  // Hy4-flash's attention output (64 heads x 256 to the embedding)
  dense(16384, 3072, rng);  // its gate
  dense(1024, 3072, rng);   // its q_a
  std::printf("[test_mv_fp4_tok_gpu] ok\n");
  return 0;
}
