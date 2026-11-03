// the weights' types decoded on the GPU against gguf-py's values, every float bit for bit: a model's own blocks (the
// first tensor of each type: its start, a middle, its end) and random ones (oracle/dequant_ref.py --model writes them
// to ~/substrate-oracle/dequant). every type the table decodes that has a file; the others named. needs a gfx1151 and
// at least one file; skipped (77) without them.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "substrate/gpu/quant.h"
#include "substrate/model/qtype.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

constexpr uint32_t kSweep = 256;  // past every ggml type id, as test_qtype's

struct Ref {
  uint32_t type = 0;
  std::vector<uint8_t> bytes;
  std::vector<float> values;
};

static bool load(const char* path, Ref* r) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  char magic[4];
  uint32_t version = 0;
  uint64_t n_values = 0, n_bytes = 0;
  CHECK(std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "SUBQ", 4) == 0);
  CHECK(std::fread(&version, 4, 1, f) == 1 && version == 1 && std::fread(&r->type, 4, 1, f) == 1);
  CHECK(std::fread(&n_values, 8, 1, f) == 1 && std::fread(&n_bytes, 8, 1, f) == 1);
  r->bytes.resize(n_bytes);
  r->values.resize(n_values);
  CHECK(std::fread(r->bytes.data(), 1, n_bytes, f) == n_bytes);
  CHECK(std::fread(r->values.data(), 4, n_values, f) == n_values);
  std::fclose(f);
  return true;
}

// the type decoded twice (the second into a buffer of other bytes): both bit for bit the reference's
static void check(const char* name, const Ref& r) {
  const uint64_t n = r.values.size();
  uint8_t* src = nullptr;
  float* dst = nullptr;
  CHECK(hipMalloc(&src, r.bytes.size()) == hipSuccess && hipMalloc(&dst, n * 4) == hipSuccess);
  CHECK(hipMemcpy(src, r.bytes.data(), r.bytes.size(), hipMemcpyHostToDevice) == hipSuccess);
  std::vector<float> got(n);
  for (int run = 0; run < 2; run++) {
    CHECK(hipMemset(dst, run ? 0xA5 : 0, n * 4) == hipSuccess);
    CHECK(quant_dequant(r.type, src, n, dst, nullptr));
    CHECK(hipDeviceSynchronize() == hipSuccess);
    CHECK(hipMemcpy(got.data(), dst, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
    uint64_t bad = 0, first = n;
    for (uint64_t i = 0; i < n; i++)
      if (std::memcmp(&got[i], &r.values[i], 4) != 0 && bad++ == 0) first = i;
    if (bad) {
      uint32_t g, w;
      std::memcpy(&g, &got[first], 4);
      std::memcpy(&w, &r.values[first], 4);
      std::printf("  %s: %llu of %llu values differ; first at %llu: %g (0x%08x), gguf-py %g (0x%08x)\n", name,
                  (unsigned long long) bad, (unsigned long long) n, (unsigned long long) first, got[first], g,
                  r.values[first], w);
      std::exit(1);
    }
  }
  /* as f16 (the matmul's decode): gguf-py's value rounded to f16, every one bit for bit */
  _Float16* dh = nullptr;
  CHECK(hipMalloc(&dh, n * 2) == hipSuccess && hipMemset(dh, 0xA5, n * 2) == hipSuccess);
  CHECK(quant_dequant_f16(r.type, src, n, dh, nullptr));
  CHECK(hipDeviceSynchronize() == hipSuccess);
  std::vector<_Float16> goth(n);
  CHECK(hipMemcpy(goth.data(), dh, n * 2, hipMemcpyDeviceToHost) == hipSuccess);
  uint64_t bad_h = 0, first_h = n;
  for (uint64_t i = 0; i < n; i++) {
    const _Float16 want = (_Float16) r.values[i];
    if (std::memcmp(&goth[i], &want, 2) != 0 && bad_h++ == 0) first_h = i;
  }

  if (bad_h) {
    std::printf("  %s as f16: %llu of %llu values differ; first at %llu: %g, gguf-py's rounded %g\n", name,
                (unsigned long long) bad_h, (unsigned long long) n, (unsigned long long) first_h,
                (double) goth[first_h], (double) (_Float16) r.values[first_h]);
    std::exit(1);
  }
  CHECK(hipFree(src) == hipSuccess && hipFree(dst) == hipSuccess && hipFree(dh) == hipSuccess);
  std::printf("  %-8s %9llu values, bit for bit gguf-py's, twice; as f16 its rounding\n", name, (unsigned long long) n);
}

int main() {
  std::printf("[test_quant_gpu] start\n");
  const char* home = std::getenv("HOME");
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_quant_gpu] skipped: no GPU\n");
    return 77;
  }
  CHECK(!quant_known(0) && !quant_known(12) && quant_known(29));
  CHECK(!quant_dequant(0, nullptr, 8, nullptr, nullptr) && !quant_dequant(99, nullptr, 8, nullptr, nullptr));
  CHECK(!quant_dequant(13, nullptr, 100, nullptr, nullptr) && !quant_dequant(30, nullptr, 9, nullptr, nullptr));
  CHECK(!quant_dequant_f16(0, nullptr, 8, nullptr, nullptr) && !quant_dequant_f16(16, nullptr, 100, nullptr, nullptr));
  std::printf("  refusals: F32 and unknown types, values not whole blocks\n");
  // every type the table decodes; each with a reference file checked, the others named (a model holds some types)
  uint64_t total = 0;
  int checked = 0;
  std::string missing;
  for (uint32_t id = 0; id < kSweep; id++) {
    const QTypeInfo* t = qtype_info(id);
    if (!t || !(t->paths & QTYPE_DECODE)) continue;
    char path[512];
    std::snprintf(path, sizeof path, "%s/substrate-oracle/dequant/%s.bin", home ? home : "", t->name);
    Ref r;
    if (!load(path, &r)) {
      missing += std::string(" ") + t->name;
      continue;
    }
    CHECK(r.type == id);
    check(t->name, r);
    total += r.values.size();
    checked++;
  }
  if (!checked) {
    std::printf("[test_quant_gpu] skipped: no reference in ~/substrate-oracle/dequant (oracle/dequant_ref.py)\n");
    return 77;
  }
  if (!missing.empty()) std::printf("  NOT CHECKED (no reference file: a model without them):%s\n", missing.c_str());
  std::printf("[test_quant_gpu] ok: %llu values\n", (unsigned long long) total);
  return 0;
}
