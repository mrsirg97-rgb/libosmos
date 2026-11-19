// the GPU tests' oracle fixtures (oracle/fixture.py's writer: named F32 arrays, the reference's) and their comparison
#pragma once
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

struct Arr {
  std::vector<uint64_t> dims;
  std::vector<float> v;
};
static std::map<std::string, Arr> fx;

static bool load(const char* path) {
  fx.clear();
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  char magic[4];
  uint32_t head[2];
  CHECK(std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "ORCL", 4) == 0);
  CHECK(std::fread(head, 4, 2, f) == 2 && head[0] == 1);
  for (uint32_t i = 0; i < head[1]; i++) {
    uint32_t len = 0, nd = 0;
    CHECK(std::fread(&len, 4, 1, f) == 1 && len < 64);
    std::string name(len, '\0');
    CHECK(std::fread(name.data(), 1, len, f) == len && std::fread(&nd, 4, 1, f) == 1 && nd <= 4);
    Arr a;
    a.dims.resize(nd);
    CHECK(std::fread(a.dims.data(), 8, nd, f) == nd);
    uint64_t n = 1;
    for (uint64_t d : a.dims) n *= d;
    a.v.resize(n);
    CHECK(std::fread(a.v.data(), 4, n, f) == n);
    fx[name] = std::move(a);
  }
  std::fclose(f);
  return true;
}

static const std::vector<float>& A(const char* name) {
  CHECK(fx.count(name));
  return fx[name].v;
}

static float* up(const std::vector<float>& v) {
  float* d = nullptr;
  CHECK(hipMalloc(&d, v.size() * sizeof(float)) == hipSuccess);
  CHECK(hipMemcpy(d, v.data(), v.size() * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
  return d;
}

static float* scratch(size_t n) {
  float* d = nullptr;
  CHECK(hipMalloc(&d, n * sizeof(float)) == hipSuccess);
  return d;
}

static std::vector<float> down(const float* d, size_t n) {
  std::vector<float> v(n);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  CHECK(hipMemcpy(v.data(), d, n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
  return v;
}

// the largest error against the reference, as a fraction of the reference's largest value: at most tol
static void match(const char* what, const std::vector<float>& got, const char* ref, double tol) {
  const std::vector<float>& want = A(ref);
  CHECK(got.size() == want.size());
  double err = 0, top = 0;
  size_t at = 0;
  for (size_t i = 0; i < got.size(); i++) {
    CHECK(std::isfinite(got[i]));
    const double e = std::fabs((double) got[i] - want[i]);
    if (e > err) err = e, at = i;
    top = std::fmax(top, std::fabs((double) want[i]));
  }
  std::printf("  %-36s %zu values: max error %.3g of %.4g (%.2g; at %zu: %.9g, the reference's %.9g)\n", what,
              got.size(), err, top, err / top, at, got[at], want[at]);
  CHECK(err <= tol * top);
}

