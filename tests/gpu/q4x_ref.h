// Qwen's GPU tests against llama.cpp's own tensors (oracle/qwen_dump.cpp, ~/osmos-oracle/qwen-dump-256): a dumped
// tensor read with its shape, a model tensor's bytes uploaded, and a result held to its reference within a relative
// L2 error. llama.cpp's matmuls take the activations at 8 bits (q8_1, a value's error up to half a step of 127,
// ~0.4%): kRelL2 bounds a step through one such matmul, kChainRel two in a row (the mixer's down, then its up);
// against an exact (double, exact weights) reference osmos is held to F32's own rounding (kExactRel). a wrong
// formula is off by order 1
#pragma once
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "osmos/model/gguf.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr double kRelL2 = 1e-2, kChainRel = 2 * kRelL2, kExactRel = 1e-5;
constexpr const char* kQwenFirst = "%s/Models/qwen38/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf";
constexpr const char* kQwenDump = "%s/osmos-oracle/qwen-dump-256";

struct Ref {
  std::vector<float> v;
  int64_t ne[4] = {1, 1, 1, 1};
};

inline std::string home_path(const char* fmt) {
  char p[512];
  std::snprintf(p, sizeof p, fmt, std::getenv("HOME") ? std::getenv("HOME") : "");
  return p;
}

// a dumped tensor: "QDMP", u32 dims, 4 x i64 ne, the values; false where the file is not there
inline bool ref_read(const std::string& name, Ref* r) {
  FILE* f = std::fopen((home_path(kQwenDump) + "/" + name + ".bin").c_str(), "rb");
  if (!f) return false;
  char magic[4];
  uint32_t dims = 0;
  CHECK(std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "QDMP", 4) == 0 && std::fread(&dims, 4, 1, f) == 1);
  CHECK(std::fread(r->ne, 8, 4, f) == 4);
  r->v.resize((size_t) (r->ne[0] * r->ne[1] * r->ne[2] * r->ne[3]));
  CHECK(std::fread(r->v.data(), 4, r->v.size(), f) == r->v.size());
  std::fclose(f);
  return true;
}

inline Ref ref(const std::string& name) {
  Ref r;
  CHECK(ref_read(name, &r));
  return r;
}

template <class T>
inline T* dev_alloc(size_t n) {
  T* p = nullptr;
  CHECK(hipMalloc(&p, n * sizeof(T)) == hipSuccess);
  return p;
}

inline float* dev_from(const std::vector<float>& v) {
  float* p = dev_alloc<float>(v.size());
  CHECK(hipMemcpy(p, v.data(), v.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  return p;
}

inline std::vector<float> host_of(const float* p, size_t n) {
  std::vector<float> v(n);
  CHECK(hipDeviceSynchronize() == hipSuccess && hipMemcpy(v.data(), p, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return v;
}

// a model tensor's bytes on the GPU (its type the file's)
inline void* dev_tensor(const Gguf* g, const char* name, const GgufTensor** out = nullptr) {
  const GgufTensor* t = gguf_tensor(g, name);
  CHECK(t);
  std::vector<uint8_t> b(t->bytes);
  FILE* f = std::fopen(g->paths[t->file], "rb");
  CHECK(f && pread(fileno(f), b.data(), b.size(), (off_t) t->offset) == (ssize_t) b.size());
  std::fclose(f);
  void* p = dev_alloc<uint8_t>(b.size());
  CHECK(hipMemcpy(p, b.data(), b.size(), hipMemcpyHostToDevice) == hipSuccess);
  if (out) *out = t;
  return p;
}

// got against the reference: the relative L2 error (|got - ref| / |ref|), within bound; scale multiplies got first
inline double held(const char* what, const std::vector<float>& got, const Ref& r, double scale = 1.0,
                   double bound = kRelL2) {
  CHECK(got.size() == r.v.size());
  double num = 0, den = 0, worst = 0;
  for (size_t i = 0; i < got.size(); i++) {
    const double g = got[i] * scale, d = g - r.v[i];
    CHECK(std::isfinite(g));
    num += d * d, den += (double) r.v[i] * r.v[i];
    worst = std::fabs(d) > worst ? std::fabs(d) : worst;
  }
  const double rel = std::sqrt(num / (den > 0 ? den : 1));
  std::printf("  %-34s %9zu values  rel L2 %.2e  max |err| %.2e\n", what, got.size(), rel, worst);
  CHECK(rel <= bound);
  return rel;
}
