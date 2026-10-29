// the pool on the real model and the real GPU: one ds1017 layer's 384 experts (gate, up, down: 3 parts each, from
// gguf), pinned through hip_pinner with a budget of a quarter of them, so pins push others out. the GPU sums every
// part where it lies and the CPU checks it. needs a gfx1151 and ~/models/ds1017; skipped (77) without them.
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "substrate/mem/hip_pinner.h"
#include "substrate/mem/pool.h"
#include "substrate/model/gguf.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

__global__ void sum_kernel(const uint8_t* __restrict__ p, size_t n, unsigned long long* out) {
  unsigned long long s = 0;
  for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) s += p[i];
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o, 32);
  if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// reads in the test are the kernel's: begun with WILLNEED, and "done" at once - a registration of pages not yet read
// reads them itself (the pin's own I/O). the binary's fetcher is not this one
static std::vector<uint32_t> done;
static void fetch(uint32_t unit, const void* const*, const uint64_t*, int, void*) { done.push_back(unit); }

int main() {
  std::printf("[test_pool_gpu] start\n");
  const char* home = std::getenv("HOME");
  char path[512];
  std::snprintf(path, sizeof path, "%s/models/ds1017/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf",
                home ? home : "");
  int ndev = 0;
  if (access(path, R_OK) != 0 || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_pool_gpu] skipped: no ds1017 or no GPU\n");
    return 77;
  }
  char gerr[GGUF_ERR] = "";
  Gguf* g = gguf_create(path, gerr);
  CHECK(g);
  std::vector<int> fds(g->n_files);
  for (int i = 0; i < g->n_files; i++) CHECK((fds[i] = open(g->paths[i], O_RDONLY)) >= 0);
  const char* names[3] = {"blk.3.ffn_gate_exps.weight", "blk.3.ffn_up_exps.weight", "blk.3.ffn_down_exps.weight"};
  const GgufTensor* t[3];
  for (int k = 0; k < 3; k++) CHECK((t[k] = gguf_tensor(g, names[k])) && t[k]->n_dims == 3);
  const int E = (int) t[0]->dims[2];
  std::vector<PoolUnit> units(E);
  uint64_t unit_bytes = 0;
  for (int e = 0; e < E; e++) {
    units[e].n_parts = 3;
    for (int k = 0; k < 3; k++) {
      const uint64_t per = t[k]->bytes / (uint64_t) E;
      units[e].part[k] = PoolRange{fds[t[k]->file], t[k]->offset + (uint64_t) e * per, per};
      if (e == 0) unit_bytes += per;
    }
  }
  const uint64_t budget = unit_bytes * (uint64_t) E / 4;
  char err[POOL_ERR] = "";
  Pool* p = pool_create(units.data(), units.size(), budget, hip_pinner(), Fetcher{fetch, nullptr}, nullptr, nullptr,
                        err);
  if (!p) std::fprintf(stderr, "pool_create: %s\n", err);
  CHECK(p);
  unsigned long long* d_sum = nullptr;
  CHECK(hipMalloc(&d_sum, sizeof *d_sum) == hipSuccess);
  // every expert twice, in two orders: pinned (pushing older ones out), summed on the GPU, checked, released
  int ok = 0, waited = 0;
  double pin_ms = 0;
  for (int round = 0; round < 2; round++)
    for (int i = 0; i < E; i++) {
      const uint32_t e = (uint32_t) (round == 0 ? i : (i * 157) % E);
      const void* dev[POOL_PARTS] = {nullptr};
      const double t0 = now_ms();
      PoolPin r = pool_pin(p, e, dev);
      if (r == POOL_WAIT) {
        waited++;
        while (!done.empty()) {
          const uint32_t u = done.back();
          done.pop_back();
          pool_ready(p, u);
        }
        CHECK(pool_dev(p, e, dev));
        r = POOL_OK;
      }
      pin_ms += now_ms() - t0;
      CHECK(r == POOL_OK && pool_pinned(p) <= budget);
      for (int k = 0; k < 3; k++) {
        const size_t n = (size_t) units[e].part[k].bytes;
        CHECK(hipMemset(d_sum, 0, sizeof *d_sum) == hipSuccess);
        sum_kernel<<<64, 256>>>((const uint8_t*) dev[k], n, d_sum);
        unsigned long long got = 0, want = 0;
        CHECK(hipMemcpy(&got, d_sum, sizeof got, hipMemcpyDeviceToHost) == hipSuccess);
        for (size_t b = 0; b < n; b++) want += ((const uint8_t*) dev[k])[b];
        CHECK(got == want);
      }
      pool_release(p, e);
      ok++;
    }
  std::printf("  %d experts of blk.3 (3 parts, %.2f MB each), budget a quarter (%.0f MB): %d pins, %d waited for a "
              "read, every GPU sum the CPU's; %.2f ms a pin on average, %.0f MB pinned at the end\n",
              E, (double) unit_bytes / 1e6, (double) budget / 1e6, ok, waited, pin_ms / ok, (double) pool_pinned(p) / 1e6);
  pool_destroy(p);
  CHECK(hipFree(d_sum) == hipSuccess);
  for (int fd : fds) close(fd);
  gguf_destroy(g);
  std::printf("[test_pool_gpu] ok\n");
  return 0;
}
