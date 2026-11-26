// the int8 lane's decoders (kernels/q8.h: q8_group_tok, q8_gate_up) on every type they take, against exact arithmetic:
// a matrix of random blocks (each type's scales made valid: no NaN, no overflow) decoded to F32 by the engine's own
// decoder (quant_dequant), the input rounded to int8 by q8_round (its codes and scales read back), each row's sum
// sum over blocks of d_x (sum over the block's values of w x_code) in double on the host - the lane's within float
// rounding of it (a code's wrong place, a scale's wrong bits: an error of the sum's own size). needs a GPU; skipped
// (77) without one.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "osmos/gpu/q8.h"
#include "osmos/gpu/quant.h"
#include "osmos/model/gguf.h"
}

static int g_fail = 0, g_pass = 0;
static void check(bool ok, const std::string& what) {
  if (ok) g_pass++;
  else g_fail++, std::printf("  FAIL %s\n", what.c_str());
}
#define HIP_OK(x)                                                         \
  do {                                                                    \
    if ((x) != hipSuccess) {                                              \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);        \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

static uint32_t g_rng = 1017;
static uint32_t next() { return g_rng = g_rng * 1664525u + 1013904223u; }
static float uni() { return ((float) (next() >> 8) + 0.5f) / 16777216.0f; }

struct Ty {
  const char* name;
  uint32_t id, values, bytes;
};
static const Ty kTypes[] = {
    {"Q2_K", 10, 256, 84},   {"Q3_K", 11, 256, 110},  {"Q4_K", 12, 256, 144},  {"Q5_K", 13, 256, 176},
    {"Q6_K", 14, 256, 210},  {"Q4_0", 2, 32, 18},     {"Q5_1", 7, 32, 24},     {"Q8_0", 8, 32, 34},
    {"IQ2_XXS", 16, 256, 66}, {"IQ2_XS", 17, 256, 74}, {"IQ2_S", 22, 256, 82}, {"IQ3_XXS", 18, 256, 98},
    {"IQ1_S", 19, 256, 50},  {"IQ1_M", 29, 256, 56},  {"IQ4_NL", 20, 32, 18},  {"IQ4_XS", 23, 256, 136},
    {"MXFP4", 39, 32, 17},   {"NVFP4", 40, 64, 36},   {"Q2_0", 42, 64, 18},
};

static void put16(uint8_t* p, uint16_t v) { std::memcpy(p, &v, 2); }
// a small f16 (2^-8 .. 2^-7): every product and sum well inside F32's range
static uint16_t small_f16() { return (uint16_t) (0x1C00 | (next() >> 22 & 0x3FF)); }

// one block of random bytes, its scales made valid for the type
static void block(const Ty& t, uint8_t* b) {
  for (uint32_t i = 0; i < t.bytes; i++) b[i] = (uint8_t) (next() >> 13);
  switch (t.id) {
    case 10: put16(b + 80, small_f16()), put16(b + 82, small_f16()); break;  // Q2_K d, dmin
    case 11: put16(b + 108, small_f16()); break;                              // Q3_K d
    case 12: case 13: put16(b, small_f16()), put16(b + 2, small_f16()); break; // Q4_K / Q5_K d, dmin
    case 14: put16(b + 208, small_f16()); break;                              // Q6_K d
    case 7: put16(b, small_f16()), put16(b + 2, small_f16()); break;          // Q5_1 d, m
    case 39: b[0] = (uint8_t) (120 + next() % 8); break;                      // MXFP4's E8M0: 2^-8 .. 2^-1
    case 40:                                                                  // NVFP4's UE4M3s: not 0x7F (NaN)
      for (int s = 0; s < 4; s++) b[s] = (uint8_t) (0x20 + next() % 0x20);
      break;
    case 29: {  // IQ1_M: d the top nibbles of its 4 scale words
      const uint16_t d = small_f16();
      for (int k = 0; k < 4; k++) b[48 + 2 * k + 1] = (uint8_t) ((b[48 + 2 * k + 1] & 0x0F) | ((d >> (4 * k)) & 0x0F) << 4);
      break;
    }
    default: put16(b, small_f16()); break;  // d first: Q4_0, Q8_0, Q2_0, IQ4_NL, IQ4_XS, the IQ2s, IQ3_XXS, IQ1_S
  }
}

int main() {
  int n_dev = 0;
  if (hipGetDeviceCount(&n_dev) != hipSuccess || !n_dev) return std::printf("no GPU: skipped\n"), 77;
  const uint32_t R = 48, C = 1536;  // a matrix's rows, its columns (whole blocks of every type)
  // the input: one row, rounded to int8 (its codes and its blocks' scales read back)
  std::vector<float> xh(C);
  for (auto& v : xh) v = 2.0f * uni() - 1.0f;
  float *x, *dq;
  int8_t* q;
  HIP_OK(hipMalloc(&x, C * 4));
  HIP_OK(hipMalloc(&q, C));
  HIP_OK(hipMalloc(&dq, C / 32 * 4));
  HIP_OK(hipMemcpy(x, xh.data(), C * 4, hipMemcpyHostToDevice));
  if (!q8_round(x, 1, C, C, q, dq, nullptr)) return std::printf("q8_round refused\n"), 1;
  std::vector<int8_t> qh(C);
  std::vector<float> dqh(C / 32);
  HIP_OK(hipMemcpy(qh.data(), q, C, hipMemcpyDeviceToHost));
  HIP_OK(hipMemcpy(dqh.data(), dq, C / 32 * 4, hipMemcpyDeviceToHost));
  int32_t *cnt, *xrow;
  HIP_OK(hipMalloc(&cnt, 4));
  HIP_OK(hipMalloc(&xrow, 4));
  const int32_t one = 1, zero = 0;
  HIP_OK(hipMemcpy(cnt, &one, 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(xrow, &zero, 4, hipMemcpyHostToDevice));
  for (const Ty& t : kTypes) {
    const uint64_t row = (uint64_t) C / t.values * t.bytes;
    std::vector<uint8_t> wh(row * R);
    for (uint64_t b = 0; b < (uint64_t) R * C / t.values; b++) block(t, wh.data() + b * t.bytes);
    uint8_t* w;
    float *wf, *y;
    HIP_OK(hipMalloc(&w, wh.size()));
    HIP_OK(hipMalloc(&wf, (size_t) R * C * 4));
    HIP_OK(hipMalloc(&y, R * 4));
    HIP_OK(hipMemcpy(w, wh.data(), wh.size(), hipMemcpyHostToDevice));
    const void** wp;
    HIP_OK(hipMalloc(&wp, sizeof(void*)));
    HIP_OK(hipMemcpy(wp, &w, sizeof(void*), hipMemcpyHostToDevice));
    const bool dec = quant_dequant(t.id, w, (uint64_t) R * C, wf, nullptr);
    const bool ran = q8_group_tok(t.id, wp, cnt, xrow, nullptr, 1, 1, R, C, q, dq, y, R, nullptr);
    HIP_OK(hipDeviceSynchronize());
    if (!dec || !ran) {
      check(false, std::string(t.name) + (dec ? ": q8_group_tok refused it" : ": quant_dequant refused it"));
      HIP_OK(hipFree(w));
      HIP_OK(hipFree(wf));
      HIP_OK(hipFree(y));
      HIP_OK(hipFree(wp));
      continue;
    }
    std::vector<float> wfh((size_t) R * C), yh(R);
    HIP_OK(hipMemcpy(wfh.data(), wf, wfh.size() * 4, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(yh.data(), y, R * 4, hipMemcpyDeviceToHost));
    double worst = 0;
    bool finite = true;
    for (uint32_t r = 0; r < R; r++) {
      double ref = 0, mag = 0;
      for (uint32_t b = 0; b < C / 32; b++) {
        double s = 0, sm = 0;
        for (uint32_t i = 0; i < 32; i++) {
          const double p = (double) wfh[(size_t) r * C + 32 * b + i] * (double) qh[32 * b + i];
          s += p, sm += std::fabs(p);
        }
        ref += (double) dqh[b] * s, mag += std::fabs((double) dqh[b]) * sm;
      }
      finite &= std::isfinite(yh[r]) && std::isfinite(ref);
      worst = std::fmax(worst, std::fabs((double) yh[r] - ref) / (mag > 0 ? mag : 1));
    }
    // float rounding of a block's few sums and the row's 48 fmafs: ~1e-6 of the magnitude; a decoding error O(1)
    check(finite && worst < 2e-5, std::string(t.name) + ": every row within float rounding of exact (worst " +
                                      std::to_string(worst) + " of the terms' magnitude)");
    std::printf("  %-8s worst %.2e of the terms' magnitude\n", t.name, worst);
    HIP_OK(hipFree(w));
    HIP_OK(hipFree(wf));
    HIP_OK(hipFree(y));
    HIP_OK(hipFree(wp));
  }
  std::printf("q8 types: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
