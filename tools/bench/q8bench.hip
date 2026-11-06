// q8bench <first shard> [layer]: the lane's int8 expert kernels timed on a model's own experts - 19 of the layer's (a
// drafted step's mean, DESIGN.md "0.3.2"), each NT tokens (1 .. 6), the fused gate+up (q8_gate_up) and the down
// (q8_group_tok) launched as the forward does (one launch for the 19 at one count); per NT: us a launch, the weights'
// GB/s (their bytes over the time - each expert's weights read once a launch, whatever its tokens).
// build: hipcc -O2 -Iinclude -Ikernels --offload-arch=gfx1151 tools/bench/q8bench.hip -Llib -losmos_hip -losmos
//        -lhsa-runtime64 -o bin/q8bench
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <random>
#include <unistd.h>
#include <vector>

extern "C" {
#include "osmos/gpu/q8.h"
#include "osmos/model/gguf.h"
#include "osmos/run/gpu.h"
}

#define CHECK(x) \
  do { \
    if (!(x)) std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #x, __FILE__, __LINE__), std::exit(1); \
  } while (0)

constexpr uint32_t kMaxExperts = 19, kMaxTok = 6, kIters = 200;
static uint32_t kExperts = kMaxExperts;  // EXPERTS: a launch's experts (19: a drafted step's mean a layer)

struct Mat {  // kExperts consecutive experts of one of the layer's expert tensors, on the GPU
  uint32_t type, rows, cols;
  uint64_t bytes;  // an expert's
  uint8_t* dev;
  void** ptrs;
};

static Mat load(const Gguf* g, const char* name) {
  const GgufTensor* t = gguf_tensor(g, name);
  CHECK(t && t->n_dims == 3);
  Mat m{t->type, (uint32_t) t->dims[1], (uint32_t) t->dims[0], t->bytes / t->dims[2], nullptr, nullptr};
  std::vector<uint8_t> w(kExperts * m.bytes);
  const int fd = open(g->paths[t->file], O_RDONLY);
  CHECK(fd >= 0 && pread(fd, w.data(), w.size(), (off_t) t->offset) == (ssize_t) w.size());
  close(fd);
  std::vector<void*> p(kExperts);
  CHECK(hipMalloc(&m.dev, w.size()) == hipSuccess && hipMalloc(&m.ptrs, kExperts * sizeof(void*)) == hipSuccess);
  for (uint32_t e = 0; e < kExperts; e++) p[e] = m.dev + e * m.bytes;
  CHECK(hipMemcpy(m.dev, w.data(), w.size(), hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(m.ptrs, p.data(), kExperts * sizeof(void*), hipMemcpyHostToDevice) == hipSuccess);
  return m;
}

int main(int argc, char** argv) {
  CHECK(gpu_sleep_on_waits());
  if (const char* e = std::getenv("EXPERTS")) {
    const int n = std::atoi(e);
    kExperts = n > 0 && (uint32_t) n <= kMaxExperts ? (uint32_t) n : kMaxExperts;
  }
  if (argc < 2) return std::fprintf(stderr, "usage: q8bench <first shard> [layer]\n"), 2;
  const int layer = argc > 2 ? std::atoi(argv[2]) : 20;
  char err[GGUF_ERR] = "";
  Gguf* g = gguf_create(argv[1], err);
  CHECK(g);
  char ng[64], nu[64], nd[64];
  std::snprintf(ng, sizeof ng, "blk.%d.ffn_gate_exps.weight", layer);
  std::snprintf(nu, sizeof nu, "blk.%d.ffn_up_exps.weight", layer);
  std::snprintf(nd, sizeof nd, "blk.%d.ffn_down_exps.weight", layer);
  const Mat G = load(g, ng), U = load(g, nu), Dn = load(g, nd);
  CHECK(G.type == U.type);
  const uint32_t D = G.cols, FF = G.rows, nrow = kMaxExperts * kMaxTok;
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::vector<float> x((size_t) nrow * D), hx((size_t) nrow * FF), sc(nrow);
  for (float& v : x) v = uni(rng);
  for (float& v : hx) v = uni(rng);
  for (float& v : sc) v = 0.5f + 0.5f * uni(rng);
  float *dx, *dhx, *dd, *dhd, *dsc, *dh, *dy;
  int8_t *dq, *dhq;
  int32_t *dcnt, *dxrow;
  CHECK(hipMalloc(&dx, x.size() * 4) == hipSuccess && hipMalloc(&dhx, hx.size() * 4) == hipSuccess &&
        hipMalloc(&dq, x.size()) == hipSuccess && hipMalloc(&dhq, hx.size()) == hipSuccess &&
        hipMalloc(&dd, x.size() / 32 * 4) == hipSuccess && hipMalloc(&dhd, hx.size() / 32 * 4) == hipSuccess &&
        hipMalloc(&dsc, nrow * 4) == hipSuccess && hipMalloc(&dh, (size_t) nrow * FF * 4) == hipSuccess &&
        hipMalloc(&dy, (size_t) nrow * D * 4) == hipSuccess && hipMalloc(&dcnt, kExperts * 4) == hipSuccess &&
        hipMalloc(&dxrow, kExperts * 4) == hipSuccess);
  CHECK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dhx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(dsc, sc.data(), nrow * 4, hipMemcpyHostToDevice) == hipSuccess);
  CHECK(q8_round(dx, nrow, D, D, dq, dd, nullptr) && q8_round(dhx, nrow, FF, FF, dhq, dhd, nullptr));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0) == hipSuccess && hipEventCreate(&e1) == hipSuccess);
  std::printf("layer %d: %u experts, gate/up type %u (%u x %u, %.2f MB each), down type %u (%.2f MB)\n", layer, kExperts,
              G.type, FF, D, G.bytes / 1e6, Dn.type, Dn.bytes / 1e6);
  std::printf("%3s %12s %10s %12s %10s\n", "NT", "gate+up us", "GB/s", "down us", "GB/s");
  for (uint32_t nt = 1; nt <= kMaxTok; nt++) {
    std::vector<int32_t> cnt(kExperts, (int32_t) nt), xrow(kExperts);
    for (uint32_t e = 0; e < kExperts; e++) xrow[e] = (int32_t) (e * nt);
    CHECK(hipMemcpy(dcnt, cnt.data(), kExperts * 4, hipMemcpyHostToDevice) == hipSuccess &&
          hipMemcpy(dxrow, xrow.data(), kExperts * 4, hipMemcpyHostToDevice) == hipSuccess);
    float ms_gu = 0, ms_dn = 0;
    for (int pass = 0; pass < 2; pass++) {  // the first a warm-up
      CHECK(hipEventRecord(e0, nullptr) == hipSuccess);
      for (uint32_t i = 0; i < kIters; i++)
        CHECK(q8_gate_up(G.type, (const void* const*) G.ptrs, (const void* const*) U.ptrs, dcnt, dxrow, nullptr, nullptr,
                         kExperts, nt, FF, D, dq, dd, dsc, 10.0f, dh, FF, nullptr));
      CHECK(hipEventRecord(e1, nullptr) == hipSuccess && hipEventSynchronize(e1) == hipSuccess &&
            hipEventElapsedTime(&ms_gu, e0, e1) == hipSuccess);
      CHECK(hipEventRecord(e0, nullptr) == hipSuccess);
      for (uint32_t i = 0; i < kIters; i++)
        CHECK(q8_group_tok(Dn.type, (const void* const*) Dn.ptrs, dcnt, dxrow, nullptr, kExperts, nt, D, FF, dhq, dhd, dy, D,
                           nullptr));
      CHECK(hipEventRecord(e1, nullptr) == hipSuccess && hipEventSynchronize(e1) == hipSuccess &&
            hipEventElapsedTime(&ms_dn, e0, e1) == hipSuccess);
    }
    const double gu = ms_gu * 1e3 / kIters, dn = ms_dn * 1e3 / kIters;
    std::printf("%3u %12.1f %10.1f %12.1f %10.1f\n", nt, gu, 2.0 * kExperts * G.bytes / (gu * 1e3), dn,
                (double) kExperts * Dn.bytes / (dn * 1e3));
  }
  gguf_destroy(g);
  return 0;
}
