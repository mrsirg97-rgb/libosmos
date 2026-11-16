// W4A4 on the FP4 matrix cores (mm_fp4): the activations' quantization against the host's (ModelOpt's act_quant,
// bit for bit), the product against the host's double sum over the same quantized values (the matrix cores' F32
// accumulation within its rounding), then its time against today's W4A16 matmul (mm_tokens) - an NVFP4 matrix of the
// GGUF (Q35_MODEL, its .input_scale; else a random one). skipped (77) without an sm_120 GPU
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/mm.h"
#include "osmos/model/gguf.h"
#include "osmos/model/qtype.h"
}

#define CHECK(c)                                                                  \
  do {                                                                            \
    if (!(c)) {                                                                   \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / 16777216.0f - 0.5f;
}
static float gauss() { return (uniform() + uniform() + uniform() + uniform()) * 1.7f; }

static const float kE2m1[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
static float e4m3(uint32_t c) {  // a finite E4M3 code's value (no sign: scales are positive)
  const uint32_t e = (c >> 3) & 15, m = c & 7;
  return e ? std::ldexp(1.0f + m / 8.0f, (int) e - 7) : std::ldexp((float) m, -9);
}
static uint32_t e4m3_bits(float a) {  // nearest, ties to even, saturating: the device's by search over the 127 codes
  if (!(a > 0.0f)) return 0;
  uint32_t best = 0;
  double bd = 1e30;
  for (uint32_t c = 0; c < 0x7F; c++) {
    const double d = std::fabs((double) e4m3(c) - a);
    if (d < bd || (d == bd && (c & 1) == 0 && (best & 1))) bd = d, best = c;
  }
  return best;
}
static uint32_t e2m1_idx(float a) {
  return a <= 0.25f ? 0 : a < 0.75f ? 1 : a <= 1.25f ? 2 : a < 1.75f ? 3 : a <= 2.5f ? 4 : a < 3.5f ? 5 : a <= 5.0f ? 6 : 7;
}

int main() {
  std::printf("[test_mm_fp4_gpu] start\n");
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || !n || !mm_fp4_ok()) {
    std::printf("[test_mm_fp4_gpu] skipped: no sm_120 GPU\n");
    return 77;
  }
  uint32_t rows = 17408, cols = 5120;
  std::vector<uint8_t> w;
  float wscale = 1.0f, gs = 0.05f;
  const char* mp = std::getenv("Q35_MODEL");
  char err[GGUF_ERR] = "";
  Gguf* g = mp ? gguf_create(mp, err) : nullptr;
  const GgufTensor* t = g ? gguf_tensor(g, "blk.3.ffn_gate.weight") : nullptr;
  if (t && t->type == QTYPE_NVFP4) {
    cols = (uint32_t) t->dims[0], rows = (uint32_t) t->dims[1];
    w.resize(t->bytes);
    FILE* f = std::fopen(g->paths[t->file], "rb");
    CHECK(f && std::fseek(f, (long) t->offset, SEEK_SET) == 0 && std::fread(w.data(), 1, w.size(), f) == w.size());
    for (const char* nm : {"blk.3.ffn_gate.scale", "blk.3.ffn_gate.input_scale"}) {
      const GgufTensor* s = gguf_tensor(g, nm);
      CHECK(s && s->bytes == 4);
      float v;
      CHECK(std::fseek(f, (long) s->offset, SEEK_SET) == 0 && std::fread(&v, 4, 1, f) == 1);
      (nm[13] == 's' ? wscale : gs) = v;
    }
    std::fclose(f);
  } else {
    w.resize((size_t) rows * (cols / 64) * 36);
    for (size_t b = 0; b < w.size(); b++) w[b] = (uint8_t) (rng = rng * 1664525u + 1013904223u) >> 0;
    for (size_t b = 0; b < w.size(); b += 36)
      for (int s = 0; s < 4; s++) w[b + s] = (uint8_t) (0x30 + (w[b + s] & 15));  // scales finite, moderate
  }
  const uint32_t T = 2048;
  std::vector<float> x((size_t) T * cols);
  for (auto& v : x) v = gauss() * gs * 6.0f;  // (around the calibrated range)
  // the host's quantization: an E4M3 scale a 16 (amax / 6 / gs), E2M1 codes (ties to even)
  std::vector<uint8_t> hq((size_t) T * cols / 2), hs((size_t) T * cols / 16);
  for (uint32_t r = 0; r < T; r++)
    for (uint32_t b = 0; b < cols / 16; b++) {
      const float* v = &x[(size_t) r * cols + b * 16];
      float amax = 0;
      for (int j = 0; j < 16; j++) amax = std::fmax(amax, std::fabs(v[j]));
      const uint32_t s = e4m3_bits(std::fmin(amax / 6.0f / gs, 448.0f));
      const float sf = e4m3(s) * gs;
      hs[(size_t) r * (cols / 16) + b] = (uint8_t) s;
      for (int j = 0; j < 16; j++) {
        const float a = sf > 0 ? std::fmin(std::fmax(v[j] / sf, -6.0f), 6.0f) : 0.0f;
        const uint32_t i = e2m1_idx(std::fabs(a)), c = i ? (a < 0 ? 8u : 0u) | i : 0u;
        hq[((size_t) r * cols + b * 16 + j) / 2] |= (uint8_t) (c << (4 * (j & 1)));
      }
    }
  float *dx, *dy, *dy16;
  void *dw, *dq, *ds, *x16;
  int32_t* ovf;
  CHECK(hipMalloc((void**) &dx, x.size() * 4) == hipSuccess && hipMalloc(&dw, w.size()) == hipSuccess);
  CHECK(hipMalloc(&dq, hq.size()) == hipSuccess && hipMalloc(&ds, hs.size()) == hipSuccess);
  CHECK(hipMalloc((void**) &dy, (size_t) T * rows * 4) == hipSuccess && hipMalloc((void**) &dy16, (size_t) T * rows * 4) == hipSuccess);
  CHECK(hipMalloc(&x16, (size_t) MM_MAX_TOKENS * cols * 2) == hipSuccess && hipMalloc((void**) &ovf, 4) == hipSuccess);
  CHECK(hipMemset(ovf, 0, 4) == hipSuccess);
  CHECK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice) == hipSuccess);
  CHECK(mm_fp4_act(dx, cols, T, cols, gs, dq, ds, nullptr));
  std::vector<uint8_t> gq(hq.size()), gsb(hs.size());
  CHECK(hipMemcpy(gq.data(), dq, gq.size(), hipMemcpyDeviceToHost) == hipSuccess);
  CHECK(hipMemcpy(gsb.data(), ds, gsb.size(), hipMemcpyDeviceToHost) == hipSuccess);
  size_t qd = 0, sd = 0;
  for (size_t i = 0; i < gq.size(); i++) qd += gq[i] != hq[i];
  for (size_t i = 0; i < gsb.size(); i++) sd += gsb[i] != hs[i];
  std::printf("  the activations (%u x %u, input scale %g): scales %zu / %zu off the host's, codes %zu / %zu bytes\n", T,
              cols, gs, sd, gsb.size(), qd, gq.size());
  CHECK(qd == 0 && sd == 0);
  CHECK(mm_fp4(dw, rows, cols, dq, ds, T, wscale * gs, dy, rows, nullptr));
  std::vector<float> y((size_t) T * rows);
  CHECK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
  // the host's sums over the quantized values: rows and tokens sampled
  double num = 0, den = 0;
  for (int k = 0; k < 400; k++) {
    const uint32_t r = (uint32_t) (k * 2654435761u % rows), tk = (uint32_t) (k * 40503u % T);
    double s = 0;
    for (uint32_t b = 0; b < cols / 64; b++) {
      const uint8_t* blk = &w[((size_t) r * (cols / 64) + b) * 36];
      for (int sb = 0; sb < 4; sb++)
        for (int j = 0; j < 16; j++) {
          const uint8_t byte = blk[4 + 8 * sb + (j & 7)];
          const uint32_t cw = (j < 8 ? byte : byte >> 4) & 15;
          const double wv = (cw & 8 ? -1 : 1) * kE2m1[cw & 7] * e4m3(blk[sb] & 0x7F);
          const size_t c = b * 64 + sb * 16 + j;
          const uint32_t ca = (hq[((size_t) tk * cols + c) / 2] >> (4 * (c & 1))) & 15;
          const double av = (ca & 8 ? -1 : 1) * kE2m1[ca & 7] * e4m3(hs[(size_t) tk * (cols / 16) + c / 16]);
          s += wv * av;
        }
    }
    s *= (double) wscale * gs;
    const double got = y[(size_t) tk * rows + r];
    num += (got - s) * (got - s), den += s * s;
  }
  std::printf("  the product (%u x %u, %u tokens): against the host's double sums, relative error %.2e\n", rows, cols, T,
              std::sqrt(num / den));
  CHECK(std::sqrt(num / den) < 1e-5);
  // time: W4A4 (quantize + FP4 matmul) against W4A16 (mm_tokens on the f32 activations)
  for (int pass = 0; pass < 2; pass++) {
    CHECK(hipDeviceSynchronize() == hipSuccess);
    const int reps = 10;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
      CHECK(mm_fp4_act(dx, cols, T, cols, gs, dq, ds, nullptr) && mm_fp4(dw, rows, cols, dq, ds, T, 1.0f, dy, rows, nullptr));
    CHECK(hipDeviceSynchronize() == hipSuccess);
    const double fp4 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; i++)
      CHECK(mm_tokens(QTYPE_NVFP4, dw, rows, dy16, rows, nullptr, 0, nullptr, 0, cols, dx, cols, T, x16, ovf, false, nullptr));
    CHECK(hipDeviceSynchronize() == hipSuccess);
    const double w16 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    const double flop = 2.0 * rows * cols * T;
    if (pass)
      std::printf("  %u x %u at %u tokens: W4A4 %.2f ms (%.0f TFLOPS), W4A16 %.2f ms (%.0f TFLOPS): %.1fx\n", rows, cols,
                  T, fp4, flop / fp4 / 1e9, w16, flop / w16 / 1e9, w16 / fp4);
  }
  std::printf("[test_mm_fp4_gpu] ok\n");
  return 0;
}
