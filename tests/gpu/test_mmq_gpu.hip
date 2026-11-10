// the experts' int8 matmul (mmq) on Qwen3.8 UD-Q4_K_XL's own Q4_K experts: mmq_x8_rows' codes and scales the bits of
// a host copy of its contract; mmq_group's outputs the bits of a host copy of its operations in its order (so a
// token's bits the same at 1 / 16 / 37 / 64 tokens, two matrices in one launch); within the int8 rounding's bound of
// a double sum over the dequantized weights and the F32 activations; and its refusals. needs a gfx1151 and the
// UD-Q4_K_XL file; skipped (77) without them.
#pragma clang fp contract(off)
#include <hip/hip_fp16.h>
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
#include "osmos/gpu/mmq.h"
#include "osmos/gpu/quant.h"
#include "osmos/model/gguf.h"
#include "osmos/model/qtype.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr const char* kQwenQ4 = "%s/Models/qwen38/q4kxl-dl/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf";
constexpr int kBlockValues = 256, kBlockBytes = 144, kGroups = kBlockValues / MMQ_GROUP, kWaves = 4;
constexpr int kMaxTokens = 64;
constexpr float kQ8Max = 127.0f;
constexpr double kU = 1.0 / (1 << 24);

struct Expert {
  const char* tensor;
  int expert;
};
// two matrices a launch: gate and up of several layers
static const Expert kPairs[][2] = {{{"blk.0.ffn_gate_exps.weight", 7}, {"blk.0.ffn_up_exps.weight", 300}},
                                   {{"blk.30.ffn_gate_exps.weight", 300}, {"blk.47.ffn_up_exps.weight", 11}}};

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) ((rng >> 8) * (1.0 / (1 << 24)) * 2.0 - 1.0);
}
static float f16(uint32_t bits) { return __half2float(__ushort_as_half((unsigned short) bits)); }
template <class T> static std::vector<T> host_of(const T* d, size_t n) {
  std::vector<T> h(n);
  CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess);
  return h;
}

struct X8 {
  std::vector<int8_t> q;
  std::vector<float> dx, sx;
};
// the contract's quantization (include/osmos/gpu/mmq.h)
static X8 x8_ref(const std::vector<float>& x, uint32_t xs, const std::vector<int32_t>& idx, uint32_t cols) {
  const uint32_t groups = cols / MMQ_GROUP;
  X8 o{std::vector<int8_t>(idx.size() * cols), std::vector<float>(idx.size() * groups), std::vector<float>(idx.size() * groups)};
  for (size_t s = 0; s < idx.size(); s++)
    for (uint32_t g = 0; g < groups; g++) {
      float v[MMQ_GROUP], a = 0.0f;
      for (int i = 0; i < MMQ_GROUP; i++) {
        v[i] = idx[s] >= 0 ? x[(size_t) idx[s] * xs + g * MMQ_GROUP + i] : 0.0f;
        a = std::fmax(a, std::fabs(v[i]));
      }
      const float d = a / kQ8Max;
      int sum = 0;
      for (int i = 0; i < MMQ_GROUP; i++) {
        const int q = a == 0.0f ? 0 : (int) std::fmin(std::fmax(std::rint(v[i] / d), -kQ8Max), kQ8Max);
        o.q[mmq_x8_at((uint32_t) s, g * MMQ_GROUP + i, cols)] = (int8_t) q;
        sum += q;
      }
      const uint64_t at = mmq_dx_at((uint32_t) s, g, groups);
      o.dx[at] = a == 0.0f ? 0.0f : d;
      o.sx[at] = o.dx[at] * (float) sum;
    }
  return o;
}

// Q4_K group g of a block: (d sc, dmin m) and its 32 codes
static void q4k_group(const uint8_t* blk, int g, float* dsc, float* dm, int* q) {
  const float d = f16(blk[0] | blk[1] << 8), dmin = f16(blk[2] | blk[3] << 8);
  const uint8_t* s = blk + 4;
  const int sc = g < 4 ? s[g] & 0x3F : (s[g + 4] & 0x0F) | ((s[g - 4] >> 2) & 0x30);
  const int mn = g < 4 ? s[g + 4] & 0x3F : (s[g + 4] >> 4) | ((s[g] >> 2) & 0x30);
  *dsc = d * (float) sc;
  *dm = dmin * (float) mn;
  for (int i = 0; i < MMQ_GROUP; i++) q[i] = (blk[16 + (g / 2) * 32 + i] >> (4 * (g % 2))) & 0x0F;
}

// the contract's operations in its order: wave p's blocks p, p + 4, ..., a group at a time; the waves' sums in order
static float mmq_ref(const uint8_t* row, uint32_t cols, const X8& x, uint32_t slot) {
  const uint32_t nb = cols / kBlockValues;
  float part[kWaves] = {};
  for (int p = 0; p < kWaves; p++)
    for (uint32_t b = p; b < nb; b += kWaves)
      for (int g = 0; g < kGroups; g++) {
        float dsc, dm;
        int q[MMQ_GROUP], i_sum = 0;
        q4k_group(row + (size_t) b * kBlockBytes, g, &dsc, &dm, q);
        const uint32_t gg = b * kGroups + g;
        for (int i = 0; i < MMQ_GROUP; i++) i_sum += q[i] * x.q[mmq_x8_at(slot, gg * MMQ_GROUP + i, cols)];
        const uint64_t at = mmq_dx_at(slot, gg, cols / MMQ_GROUP);
        const float v = dsc * x.dx[at];
        part[p] = std::fmaf(v, (float) i_sum, part[p]);
        part[p] = std::fmaf(dm, -x.sx[at], part[p]);
      }
  float s = part[0];
  for (int p = 1; p < kWaves; p++) s = s + part[p];
  return s;
}

struct Mat {
  std::vector<uint8_t> host;
  uint8_t* dev = nullptr;
  uint32_t rows = 0, cols = 0;
};
static Mat load(const Gguf* g, const Expert& e) {
  const GgufTensor* t = gguf_tensor(g, e.tensor);
  CHECK(t && t->type == QTYPE_Q4_K && t->dims[2] > (uint64_t) e.expert);
  Mat m;
  m.cols = (uint32_t) t->dims[0], m.rows = (uint32_t) t->dims[1];
  const uint64_t bytes = t->bytes / t->dims[2];
  m.host.resize(bytes);
  const int fd = open(g->paths[t->file], O_RDONLY);
  CHECK(fd >= 0 && pread(fd, m.host.data(), bytes, (off_t) (t->offset + e.expert * bytes)) == (ssize_t) bytes);
  close(fd);
  CHECK(hipMalloc(&m.dev, bytes) == hipSuccess &&
        hipMemcpy(m.dev, m.host.data(), bytes, hipMemcpyHostToDevice) == hipSuccess);
  return m;
}

// a pair: tokens of their own x for each, mmq_x8_rows over both's slots (one -1), mmq_group over both at n0 / n1
static void pair(const Gguf* g, const Expert (&e)[2]) {
  Mat m[2] = {load(g, e[0]), load(g, e[1])};
  CHECK(m[0].cols == m[1].cols && m[0].rows == m[1].rows);
  const uint32_t rows = m[0].rows, cols = m[0].cols, groups = cols / MMQ_GROUP, xs = cols + 64;
  // x: 2 * kMaxTokens + 1 rows, xs floats apart; some groups scaled up or down by powers of two, one all zero
  const uint32_t x_rows = 2 * kMaxTokens + 1;
  std::vector<float> x((size_t) x_rows * xs);
  for (uint32_t r = 0; r < x_rows; r++)
    for (uint32_t gi = 0; gi < groups; gi++) {
      const float scale = std::ldexp(1.0f, (int) ((r * 7 + gi * 3) % 13) - 6);
      for (int i = 0; i < MMQ_GROUP; i++) x[(size_t) r * xs + gi * MMQ_GROUP + i] = (r == 5 && gi == 3) ? 0.0f : uniform() * scale;
    }
  // slots: matrix 0's tokens rows 0 .. kMaxTokens - 1 of x, then matrix 1's reversed from row x_rows - 1; slot 9 -1
  std::vector<int32_t> idx(2 * kMaxTokens);
  for (int i = 0; i < kMaxTokens; i++) idx[i] = i, idx[kMaxTokens + i] = (int32_t) (x_rows - 1 - i);
  idx[9] = -1;
  const X8 ref = x8_ref(x, xs, idx, cols);

  float *dxs, *dxd, *dsx, *y;
  int32_t *didx, *dcnt;
  int8_t* dx8;
  uint32_t *dsoff, *dyoff;
  const void** dw;
  const size_t slots = idx.size();
  CHECK(hipMalloc(&dxs, x.size() * 4) == hipSuccess && hipMalloc(&didx, slots * 4) == hipSuccess &&
        hipMalloc(&dx8, slots * cols) == hipSuccess && hipMalloc(&dxd, slots * groups * 4) == hipSuccess &&
        hipMalloc(&dsx, slots * groups * 4) == hipSuccess && hipMalloc(&y, slots * rows * 4) == hipSuccess &&
        hipMalloc(&dcnt, 8) == hipSuccess && hipMalloc(&dsoff, 8) == hipSuccess && hipMalloc(&dyoff, 8) == hipSuccess &&
        hipMalloc(&dw, 2 * sizeof(void*)) == hipSuccess);
  CHECK(hipMemcpy(dxs, x.data(), x.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(didx, idx.data(), slots * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(mmq_x8_rows(dxs, xs, didx, (uint32_t) slots, cols, dx8, dxd, dsx, nullptr));
  CHECK(host_of(dx8, slots * cols) == ref.q);
  CHECK(std::memcmp(host_of(dxd, slots * groups).data(), ref.dx.data(), ref.dx.size() * 4) == 0);
  CHECK(std::memcmp(host_of(dsx, slots * groups).data(), ref.sx.data(), ref.sx.size() * 4) == 0);

  // the host reference for every slot (matrix 0 the first kMaxTokens, 1 the rest)
  std::vector<float> want(slots * rows);
  for (size_t s = 0; s < slots; s++) {
    const Mat& mm = m[s < (size_t) kMaxTokens ? 0 : 1];
    const size_t row_bytes = mm.host.size() / rows;
    for (uint32_t r = 0; r < rows; r++)
      want[s * rows + r] = mmq_ref(mm.host.data() + r * row_bytes, cols, ref, (uint32_t) s);
  }
  const void* wp[2] = {m[0].dev, m[1].dev};
  const uint32_t soff[2] = {0, kMaxTokens}, yoff[2] = {0, kMaxTokens * rows};
  CHECK(hipMemcpy(dw, wp, sizeof wp, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dsoff, soff, 8, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dyoff, yoff, 8, hipMemcpyHostToDevice) == hipSuccess);
  for (int32_t n : {1, 16, 37, 64}) {
    const int32_t cnt[2] = {n, n == 1 ? 3 : n - 1};  // the pair's counts their own
    CHECK(hipMemcpy(dcnt, cnt, 8, hipMemcpyHostToDevice) == hipSuccess &&
          hipMemset(y, 0xff, slots * rows * 4) == hipSuccess);
    CHECK(mmq_group(QTYPE_Q4_K, dw, dcnt, 2, (uint32_t) (cnt[0] > cnt[1] ? cnt[0] : cnt[1]), rows, cols, dx8, dxd, dsx, dsoff, y, dyoff, rows, nullptr));
    const std::vector<float> got = host_of(y, slots * rows);
    for (int k = 0; k < 2; k++) {
      const size_t at = (size_t) soff[k] * rows;
      CHECK(std::memcmp(&got[at], &want[at], (size_t) cnt[k] * rows * 4) == 0);
      const uint32_t* tail = (const uint32_t*) &got[at + (size_t) cnt[k] * rows];  // past its count: untouched
      CHECK(cnt[k] == kMaxTokens || tail[0] == 0xffffffffu);
    }
  }

  // against the dequantized weights and F32 activations: a group's int8 rounding at most dx / 2 a value, a
  // sum's float roundings a few units of the absolute sum
  for (int k = 0; k < 2; k++) {
    float* wf;
    CHECK(hipMalloc(&wf, (size_t) rows * cols * 4) == hipSuccess);
    CHECK(quant_dequant(QTYPE_Q4_K, m[k].dev, (uint64_t) rows * cols, wf, nullptr));
    const std::vector<float> w = host_of(wf, (size_t) rows * cols);
    CHECK(hipFree(wf) == hipSuccess);
    double worst = 0.0;
    for (int s = 0; s < kMaxTokens; s++) {
      const size_t slot = (size_t) k * kMaxTokens + s;
      for (uint32_t r = 0; r < rows; r++) {
        double exact = 0.0, abs_sum = 0.0, bound = 0.0;
        for (uint32_t c = 0; c < cols; c++) {
          const double wv = w[(size_t) r * cols + c];
          const double xv = idx[slot] >= 0 ? x[(size_t) idx[slot] * xs + c] : 0.0;
          exact += wv * xv;
          abs_sum += std::fabs(wv * xv);
          bound += std::fabs(wv) * ref.dx[mmq_dx_at((uint32_t) slot, c / MMQ_GROUP, groups)] * 0.5;
        }
        const double err = std::fabs(want[slot * rows + r] - exact);
        CHECK(err <= bound + 64 * kU * (abs_sum + bound));
        worst = std::fmax(worst, err / (bound + 1e-30));
      }
    }
    std::printf("  %s #%d: worst error %.3f of the int8 bound\n", e[k].tensor, e[k].expert, worst);
  }
  std::printf("  %s #%d + %s #%d (%u x %u): mmq_x8_rows and mmq_group the host's bits at 1 / 16 / 37 / 64 tokens\n",
              e[0].tensor, e[0].expert, e[1].tensor, e[1].expert, rows, cols);
  for (void* p : {(void*) dxs, (void*) didx, (void*) dx8, (void*) dxd, (void*) dsx, (void*) y, (void*) dcnt,
                  (void*) dsoff, (void*) dyoff, (void*) dw, (void*) m[0].dev, (void*) m[1].dev})
    CHECK(hipFree(p) == hipSuccess);
}

int main() {
  std::printf("[test_mmq_gpu] start\n");
  char path[512];
  std::snprintf(path, sizeof path, kQwenQ4, std::getenv("HOME") ? std::getenv("HOME") : "");
  int ndev = 0;
  if (access(path, R_OK) || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_mmq_gpu] skipped: no %s, or no GPU\n", path);
    return 77;
  }
  CHECK(mmq_known(QTYPE_Q4_K) && !mmq_known(QTYPE_Q5_1) && !mmq_known(0));
  int8_t* p8 = (int8_t*) 1;
  float* pf = (float*) 1;
  const void* const* pw = (const void* const*) 1;
  const int32_t* pc = (const int32_t*) 1;
  const uint32_t* pu = (const uint32_t*) 1;
  CHECK(!mmq_x8_rows(pf, 48, pc, 1, 48, p8, pf, pf, nullptr));     // cols not a multiple of 32
  CHECK(!mmq_x8_rows(pf, 32, pc, 1, 64, p8, pf, pf, nullptr));     // rows overlapping
  CHECK(!mmq_group(QTYPE_Q5_1, pw, pc, 1, 1, 16, 256, p8, pf, pf, pu, pf, pu, 16, nullptr));  // not its type
  CHECK(!mmq_group(QTYPE_Q4_K, pw, pc, 1, 1, 24, 256, p8, pf, pf, pu, pf, pu, 24, nullptr));  // rows not 16s
  CHECK(!mmq_group(QTYPE_Q4_K, pw, pc, 1, 1, 16, 288, p8, pf, pf, pu, pf, pu, 16, nullptr));  // cols not 256s
  CHECK(!mmq_group(QTYPE_Q4_K, pw, pc, 1, 1, 32, 256, p8, pf, pf, pu, pf, pu, 16, nullptr));  // outputs overlapping
  std::printf("  refusals: Q5_1, cols not 32s / 256s, rows not 16s, overlapping rows\n");
  char err[GGUF_ERR] = "";
  Gguf* g = gguf_create(path, err);
  CHECK(g);
  for (const auto& p : kPairs) pair(g, p);
  gguf_destroy(g);
  std::printf("[test_mmq_gpu] ok\n");
  return 0;
}
