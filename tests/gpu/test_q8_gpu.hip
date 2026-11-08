// the lane's int8 experts (q8_round, q8_group_tok) against the host's replay of kernels/lane.h's contract on ds1017's
// experts - an IQ1_M, an IQ2_XXS and an IQ2_XS layer's gate and down, two experts a launch, 1 to 8 tokens: the
// rounding's bytes and scales, every output the same bits (a token's alone whatever the launch's width); refusals;
// and how far the int8 sums are from the F32 matvec's (printed)
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

#pragma clang fp contract(off)  // the reference sums as the contract writes them: no fused product

#include "lane.h"
#include "quant_grids.h"
#include "test_model.h"

extern "C" {
#include "osmos/gpu/mv.h"
#include "osmos/gpu/q8.h"
#include "osmos/model/gguf.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kQK = 256, kTok = MV_MAX_TOKENS, kExperts = 2;
constexpr int kIQ2_XXS = QTYPE_IQ2_XXS, kIQ2_XS = QTYPE_IQ2_XS, kIQ1_M = QTYPE_IQ1_M, kIQ1_S = QTYPE_IQ1_S,
              kIQ2_S = QTYPE_IQ2_S, kIQ3_XXS = QTYPE_IQ3_XXS, kQ2_K = QTYPE_Q2_K;

static uint64_t g_xxs[256], g_xs[512], g_2s[1024], g_1s[2048];
static uint32_t g_3xxs[256];
static uint8_t g_signs[128];

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / (float) (1u << 23) - 1.0f;
}

static float f16(uint32_t bits) {
  const uint16_t b = (uint16_t) bits;
  _Float16 h;
  std::memcpy(&h, &b, sizeof h);
  return (float) h;
}
static uint32_t ld16(const uint8_t* p) { return (uint32_t) p[0] | (uint32_t) p[1] << 8; }

// the block's scale, half h's 2s + 1 and integer dot for 32-value block j of a 256-value block b against q (32 int8);
// Q2_K's also its dmin, a half's 8 m and x's sum
struct Half {
  float d, dmin;
  int s21[2], isum[2], m8[2], xsum[2];
};
static void halves(int type, const uint8_t* b, int j, const int8_t* q, Half& hf) {
  float* d = &hf.d;
  int* s21 = hf.s21;
  int* isum = hf.isum;
  isum[0] = isum[1] = hf.xsum[0] = hf.xsum[1] = 0;
  for (int c = 0; c < 4; c++) {  // the block's four chunks of 8
    const int k = 4 * j + c, h = c / 2;
    int v[8];
    if (type == kIQ2_XXS) {
      const uint8_t* qq = b + 2 + 8 * j;
      const uint32_t aux = ld16(qq + 4) | ld16(qq + 6) << 16;
      const uint64_t gr = g_xxs[qq[c]];
      const uint8_t sg = g_signs[(aux >> (7 * c)) & 0x7F];
      for (int i = 0; i < 8; i++) v[i] = (sg >> i) & 1 ? -(int) (uint8_t) (gr >> (8 * i)) : (int) (uint8_t) (gr >> (8 * i));
      *d = f16(ld16(b)), s21[h] = 2 * (int) (aux >> 28) + 1;
    } else if (type == kIQ2_XS) {
      const uint32_t qq = ld16(b + 2 + 2 * k);
      const uint64_t gr = g_xs[qq & 511];
      const uint8_t sg = g_signs[qq >> 9];
      for (int i = 0; i < 8; i++) v[i] = (sg >> i) & 1 ? -(int) (uint8_t) (gr >> (8 * i)) : (int) (uint8_t) (gr >> (8 * i));
      *d = f16(ld16(b)), s21[h] = 2 * ((b[66 + k / 4] >> (4 * ((k / 2) % 2))) & 0x0F) + 1;
    } else if (type == kIQ2_S) {  // gguf-py's IQ2_S: d, qs[32], signs[32], qh[8] (2 bits an 8), scales[8] (a nibble a 16)
      const uint64_t gr = g_2s[b[2 + k] | ((b[66 + k / 4] >> (2 * (k % 4))) & 3) << 8];
      const uint8_t sg = b[34 + k];
      for (int i = 0; i < 8; i++) v[i] = (sg >> i) & 1 ? -(int) (uint8_t) (gr >> (8 * i)) : (int) (uint8_t) (gr >> (8 * i));
      *d = f16(ld16(b)), s21[h] = 2 * ((b[74 + k / 4] >> (4 * ((k / 2) % 2))) & 0x0F) + 1;
    } else if (type == kIQ3_XXS) {  // gguf-py's IQ3_XXS: d, qs[64] (a grid byte a 4), then a u32 a 32 (4 7-bit signs, s)
      const uint8_t* qq = b + 2 + 8 * j + 2 * c;
      const uint32_t aux = ld16(b + 66 + 4 * j) | ld16(b + 68 + 4 * j) << 16;
      const uint64_t gr = (uint64_t) g_3xxs[qq[0]] | (uint64_t) g_3xxs[qq[1]] << 32;
      const uint8_t sg = g_signs[(aux >> (7 * c)) & 0x7F];
      for (int i = 0; i < 8; i++) v[i] = (sg >> i) & 1 ? -(int) (uint8_t) (gr >> (8 * i)) : (int) (uint8_t) (gr >> (8 * i));
      *d = f16(ld16(b)), s21[h] = 2 * (2 * (int) (aux >> 28) + 1);  // d (0.5 + s) 0.5: 2 (2s + 1) eighths
    } else if (type == kQ2_K) {  // gguf-py's Q2_K: scales[16] (sc low nibble, m high), qs[64], d, dmin
      const int at = 32 * (j / 4) + 8 * c, sh = 2 * (j % 4);
      for (int i = 0; i < 8; i++) v[i] = (b[16 + at + i] >> sh) & 3, hf.xsum[h] += q[8 * c + i];
      *d = f16(ld16(b + 80)), hf.dmin = f16(ld16(b + 82));
      s21[h] = 8 * (b[2 * j + h] & 0x0F), hf.m8[h] = 8 * (b[2 * j + h] >> 4);
    } else if (type == kIQ1_S) {  // gguf-py's IQ1_S: d, qs[32], qh u16[8] (3 bits an 8, the scale bits 12-14, sign 15)
      const uint32_t qh = ld16(b + 34 + 2 * (k / 4));
      const uint64_t gr = g_1s[b[2 + k] | ((qh >> (3 * (k % 4))) & 7) << 8];
      for (int i = 0; i < 8; i++) v[i] = 8 * (int) (int8_t) (gr >> (8 * i)) + (qh & 0x8000 ? -1 : 1);
      *d = f16(ld16(b)), s21[h] = 2 * (int) ((qh >> 12) & 7) + 1;
    } else {
      CHECK(type == kIQ1_M);
      uint64_t sc = 0;
      std::memcpy(&sc, b + 48, 8);
      const uint32_t s0 = (uint32_t) sc & 0xFFFF, s1 = (uint32_t) (sc >> 16) & 0xFFFF, s2 = (uint32_t) (sc >> 32) & 0xFFFF,
                     s3 = (uint32_t) (sc >> 48);
      const uint32_t hb = (b[32 + k / 2] >> (4 * (k % 2))) & 0x0F;
      const uint64_t gr = g_1s[b[k] | (hb & 0x07) << 8];
      for (int i = 0; i < 8; i++) v[i] = 8 * (int) (int8_t) (gr >> (8 * i)) + (hb & 0x08 ? -1 : 1);
      *d = f16((s0 >> 12) | ((s1 >> 8) & 0x00F0) | ((s2 >> 4) & 0x0F00) | (s3 & 0xF000));
      s21[h] = 2 * (int) ((sc >> (16 * (k / 8) + 3 * ((k / 2) % 4))) & 0x07) + 1;
    }
    for (int i = 0; i < 8; i++) isum[h] += v[i] * q[8 * c + i];
  }
}

static int block_bytes(int type) { return (int) qtype_info((uint32_t) type)->bytes; }
static int block_values(int type) { return (int) qtype_info((uint32_t) type)->values; }

// one row's sum by the contract: blocks to LANE_Q8_LANES(cols) lanes in turn, then the xor tree
static float row_ref(int type, const uint8_t* row, uint32_t cols, const int8_t* q, const float* qd) {
  const int L = LANE_Q8_LANES(cols);
  float acc[32] = {0};
  for (uint32_t s = 0; s < cols / LANE_Q8_VALUES; s++) {
    Half h;
    const int per = block_values(type) / LANE_Q8_VALUES;  // a block's blocks of 32 (the type's values over 32)
    halves(type, row + (s / per) * block_bytes(type), (int) (s % per), q + s * LANE_Q8_VALUES, h);
    const float sub = type == kQ2_K ? lane_q8_part_min(h.d, h.s21[0], h.isum[0], h.dmin, h.m8[0], h.xsum[0]) +
                                          lane_q8_part_min(h.d, h.s21[1], h.isum[1], h.dmin, h.m8[1], h.xsum[1])
                                    : lane_q8_part(h.d, h.s21[0], h.isum[0]) + lane_q8_part(h.d, h.s21[1], h.isum[1]);
    acc[s % L] = fmaf(qd[s], sub, acc[s % L]);
  }
  for (int o = L / 2; o > 0; o >>= 1) {
    float nx[32];
    for (int l = 0; l < L; l++) nx[l] = acc[l] + acc[l ^ o];
    std::memcpy(acc, nx, sizeof nx);
  }
  return acc[0];
}

static void round_ref(const float* x, uint32_t cols, int8_t* q, float* d) {
  for (uint32_t s = 0; s < cols / LANE_Q8_VALUES; s++) {
    float a = 0.0f;
    for (uint32_t i = 0; i < LANE_Q8_VALUES; i++) a = fmaxf(a, fabsf(x[s * LANE_Q8_VALUES + i]));
    const float inv = lane_q8_inv(a);
    for (uint32_t i = 0; i < LANE_Q8_VALUES; i++) q[s * LANE_Q8_VALUES + i] = (int8_t) lane_q8_round(x[s * LANE_Q8_VALUES + i], inv);
    d[s] = lane_q8_scale(a);
  }
}

struct Experts {
  uint32_t type, rows, cols;
  uint64_t bytes;  // an expert's
  std::vector<uint8_t> w;  // kExperts of them
};

static void check(const Experts& ex, const char* name);

static void experts(const Gguf* g, const char* name) {
  const GgufTensor* t = gguf_tensor(g, name);
  CHECK(t && t->n_dims == 3);
  Experts ex{t->type, (uint32_t) t->dims[1], (uint32_t) t->dims[0], t->bytes / t->dims[2], {}};
  ex.w.resize(kExperts * ex.bytes);
  const int fd = open(g->paths[t->file], O_RDONLY);
  for (uint32_t e = 0; e < kExperts; e++)  // experts 7 and 300
    CHECK(fd >= 0 && pread(fd, ex.w.data() + e * ex.bytes, ex.bytes,
                           (off_t) (t->offset + (uint64_t) (e ? 300 : 7) * ex.bytes)) == (ssize_t) ex.bytes);
  close(fd);
  check(ex, name);
}

// a type no model here holds: random blocks (any bytes are one) with sane f16 scales at the type's places
static void synthetic(uint32_t type, uint32_t rows, uint32_t cols, const char* name) {
  const uint32_t bb = (uint32_t) block_bytes((int) type);
  Experts ex{type, rows, cols, (uint64_t) rows * (cols / (uint32_t) block_values((int) type)) * bb, {}};
  ex.w.resize(kExperts * ex.bytes);
  for (uint8_t& b : ex.w) b = (uint8_t) (uniform() * 128.0f + 128.0f);
  const std::vector<int> scale_at = type == (uint32_t) kQ2_K ? std::vector<int>{80, 82} : std::vector<int>{0};
  for (size_t at = 0; at < ex.w.size(); at += bb)
    for (int o : scale_at) {
      const _Float16 h = (_Float16) (std::fabs(uniform()) * 0.01f);
      std::memcpy(&ex.w[at + o], &h, 2);
    }
  check(ex, name);
}

static void check(const Experts& ex, const char* name) {
  const uint32_t cols = ex.cols, rows = ex.rows, nb = cols / LANE_Q8_VALUES;
  const uint64_t bytes = ex.bytes;
  const std::vector<uint8_t>& w = ex.w;
  // x: kExperts runs of kTok rows (row r of expert e at e * kTok + r); a row's block of zeros and one of a single value
  const uint32_t nrow = kExperts * kTok;
  std::vector<float> x((size_t) nrow * cols);
  for (float& v : x) v = uniform() * 3.0f;
  for (uint32_t i = 0; i < LANE_Q8_VALUES; i++) x[i] = 0.0f, x[cols + LANE_Q8_VALUES + i] = i == 5 ? -2.5f : 0.0f;
  std::vector<int8_t> q((size_t) nrow * cols), gq(q.size());
  std::vector<float> qd((size_t) nrow * nb), gqd(qd.size());
  for (uint32_t r = 0; r < nrow; r++) round_ref(&x[(size_t) r * cols], cols, &q[(size_t) r * cols], &qd[(size_t) r * nb]);
  uint8_t *dw, **dp;
  float *dx, *dd, *dy;
  int8_t* dq;
  int32_t *dcnt, *dxrow;
  CHECK(hipMalloc(&dw, w.size()) == hipSuccess && hipMalloc(&dp, kExperts * sizeof(void*)) == hipSuccess &&
        hipMalloc(&dx, x.size() * 4) == hipSuccess && hipMalloc(&dq, q.size()) == hipSuccess &&
        hipMalloc(&dd, qd.size() * 4) == hipSuccess && hipMalloc(&dy, (size_t) nrow * rows * 4) == hipSuccess &&
        hipMalloc(&dcnt, kExperts * 4) == hipSuccess && hipMalloc(&dxrow, kExperts * 4) == hipSuccess);
  const uint8_t* hp[kExperts] = {dw, dw + bytes};
  const int32_t xrow[kExperts] = {0, (int32_t) kTok};
  CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dp, hp, sizeof hp, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dxrow, xrow, sizeof xrow, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(q8_round(dx, nrow, cols, cols, dq, dd, nullptr) && hipDeviceSynchronize() == hipSuccess);
  CHECK(hipMemcpy(gq.data(), dq, q.size(), hipMemcpyDeviceToHost) == hipSuccess &&
        hipMemcpy(gqd.data(), dd, qd.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(std::memcmp(gq.data(), q.data(), q.size()) == 0 && std::memcmp(gqd.data(), qd.data(), qd.size() * 4) == 0);
  CHECK(qd[0] == 0.0f && q[cols + LANE_Q8_VALUES + 5] == -127);  // the zero block, the lone value
  std::vector<float> ref((size_t) nrow * rows), got(ref.size());
  for (uint32_t e = 0; e < kExperts; e++)
    for (uint32_t r = 0; r < kTok; r++)
      for (uint32_t o = 0; o < rows; o++)
        ref[(size_t) (e * kTok + r) * rows + o] = row_ref(ex.type, &w[e * bytes + (size_t) o * (bytes / rows)], cols,
                                                          &q[(size_t) (e * kTok + r) * cols], &qd[(size_t) (e * kTok + r) * nb]);
  for (uint32_t n = 1; n <= kTok; n++) {  // n tokens each, at a launch of width n and of width kTok
    const int32_t cnt[kExperts] = {(int32_t) n, (int32_t) (kTok + 1 - n)};
    CHECK(hipMemcpy(dcnt, cnt, sizeof cnt, hipMemcpyHostToDevice) == hipSuccess);
    for (uint32_t width : {std::max(n, kTok + 1 - n), kTok}) {
      CHECK(hipMemset(dy, 0xFF, (size_t) nrow * rows * 4) == hipSuccess);
      CHECK(q8_group_tok(ex.type, (const void* const*) dp, dcnt, dxrow, nullptr, kExperts, width, rows, cols, dq, dd, dy, rows,
                         nullptr) &&
            hipDeviceSynchronize() == hipSuccess);
      CHECK(hipMemcpy(got.data(), dy, got.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
      for (uint32_t e = 0; e < kExperts; e++)
        CHECK(std::memcmp(&got[(size_t) e * kTok * rows], &ref[(size_t) e * kTok * rows], (size_t) cnt[e] * rows * 4) == 0);
    }
  }
  // the F32 matvec's sums of the same rows of x (expert 7, kTok tokens): how far the int8 inputs move them
  float* df;
  CHECK(hipMalloc(&df, (size_t) kTok * rows * 4) == hipSuccess);
  CHECK(mv_strided(ex.type, dw, rows, cols, dx, cols, kTok, df, rows, nullptr) && hipDeviceSynchronize() == hipSuccess);
  std::vector<float> f32((size_t) kTok * rows);
  CHECK(hipMemcpy(f32.data(), df, f32.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  double num = 0, den = 0;
  for (size_t i = 0; i < f32.size(); i++) num += (ref[i] - f32[i]) * (double) (ref[i] - f32[i]), den += (double) f32[i] * f32[i];
  CHECK(hipFree(df) == hipSuccess && hipFree(dw) == hipSuccess && hipFree(dp) == hipSuccess && hipFree(dx) == hipSuccess &&
        hipFree(dq) == hipSuccess && hipFree(dd) == hipSuccess && hipFree(dy) == hipSuccess &&
        hipFree(dcnt) == hipSuccess && hipFree(dxrow) == hipSuccess);
  std::printf("  %-28s type %2u %4u x %4u: the contract's bits at 1 .. %u tokens, two experts a launch; against F32 "
              "%.2e\n", name, ex.type, rows, cols, kTok, std::sqrt(num / den));
}

int main() {
  std::printf("[test_q8_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  const std::vector<const char*> names = which == TestModel::kUd ? kUdExperts : kGsqExperts;
  int ndev = 0;
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_q8_gpu] skipped: no ds1017 or UD-IQ1_M, or no GPU\n");
    return 77;
  }
  CHECK(hipMemcpyFromSymbol(g_xxs, HIP_SYMBOL(kIq2xxsGrid), sizeof g_xxs) == hipSuccess &&
        hipMemcpyFromSymbol(g_xs, HIP_SYMBOL(kIq2xsGrid), sizeof g_xs) == hipSuccess &&
        hipMemcpyFromSymbol(g_2s, HIP_SYMBOL(kIq2sGrid), sizeof g_2s) == hipSuccess &&
        hipMemcpyFromSymbol(g_1s, HIP_SYMBOL(kIq1sGrid), sizeof g_1s) == hipSuccess &&
        hipMemcpyFromSymbol(g_3xxs, HIP_SYMBOL(kIq3xxsGrid), sizeof g_3xxs) == hipSuccess &&
        hipMemcpyFromSymbol(g_signs, HIP_SYMBOL(kKsigns), sizeof g_signs) == hipSuccess);
  // refusals: a type without the contract (Q5_K), cols not whole blocks, no matrices, too many tokens
  const void* const* nw = nullptr;
  CHECK(!q8_round(nullptr, 1, 33, 33, nullptr, nullptr, nullptr) && !q8_round(nullptr, 0, 256, 256, nullptr, nullptr, nullptr));
  CHECK(!q8_group_tok(13, nw, nullptr, nullptr, nullptr, 1, 1, 1, kQK, nullptr, nullptr, nullptr, 1, nullptr));
  CHECK(!q8_group_tok(kIQ2_XXS, nw, nullptr, nullptr, nullptr, 1, 1, 1, kQK + 32, nullptr, nullptr, nullptr, 1, nullptr));
  CHECK(!q8_group_tok(kIQ2_XXS, nw, nullptr, nullptr, nullptr, 0, 1, 1, kQK, nullptr, nullptr, nullptr, 1, nullptr));
  CHECK(!q8_group_tok(kIQ2_XXS, nw, nullptr, nullptr, nullptr, 1, kTok + 1, 1, kQK, nullptr, nullptr, nullptr, 1, nullptr));
  char err[GGUF_ERR] = "";
  Gguf* g = gguf_create(path, err);
  CHECK(g);
  for (const char* name : names) experts(g, name);
  gguf_destroy(g);
  // UD-IQ2_XXS's types beside the model's (its down experts' IQ3_XXS and Q2_K): a gate's shape (32 lanes) and a down's (16)
  for (uint32_t type : {(uint32_t) kIQ3_XXS, (uint32_t) kQ2_K}) {
    const char* tn = qtype_info(type)->name;
    char a[64], b[64];
    std::snprintf(a, sizeof a, "random %s, 4096 cols", tn), std::snprintf(b, sizeof b, "random %s, 2048 cols", tn);
    synthetic(type, 512, 4096, a);
    synthetic(type, 512, 2048, b);
  }
  std::printf("[test_q8_gpu] ok\n");
  return 0;
}
