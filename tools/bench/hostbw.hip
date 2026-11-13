// 0.6.10: can a long step's buffers live in the expert cache's own host memory, with no memory swapped (the lend's
// churn: DESIGN.md "0.6.10")? the GPU's write, read and read-write bandwidth on 1 GiB of each kind, against hipMalloc:
//   malloc    hipMalloc (the lent block's kind)
//   register  mmap + hipHostRegister (the cache's kind now: coherent, the GPU's writes not cached)
//   nc        hipHostMalloc(hipHostMallocNonCoherent)
//   coarse    mmap + hsa_amd_memory_lock_to_pool, the CPU's coarse-grained pool (coherent at dispatch boundaries)
// and whether an O_DIRECT read lands in it (the cache's reads). build:
//   hipcc -O2 --offload-arch=gfx1151 tools/bench/hostbw.hip -lhsa-runtime64 -o bin/hostbw
//   bin/hostbw <a file to read O_DIRECT, 64 MiB of it>
#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr size_t kBytes = 1ull << 30;

__global__ void write_k(float4* p, size_t n, float v) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = make_float4(v, v, v, v);
}
__global__ void read_k(const float4* p, size_t n, float* out) {
  float s = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float4 x = p[i];
    s += x.x + x.y + x.z + x.w;
  }
  if (s == 1.2345f) *out = s;  // kept, never taken
}
__global__ void rw_k(float4* p, size_t n) {  // a step buffer's use: read, then written again
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float4 x = p[i];
    x.x += 1, x.y += 1, x.z += 1, x.w += 1;
    p[i] = x;
  }
}

static double gbps(void (*launch)(void*), void* p, int reps, size_t bytes_moved) {
  hipEvent_t a, b;
  CHECK(hipEventCreate(&a) == hipSuccess && hipEventCreate(&b) == hipSuccess);
  launch(p);
  CHECK(hipDeviceSynchronize() == hipSuccess);
  CHECK(hipEventRecord(a, nullptr) == hipSuccess);
  for (int i = 0; i < reps; i++) launch(p);
  CHECK(hipEventRecord(b, nullptr) == hipSuccess && hipEventSynchronize(b) == hipSuccess);
  float ms = 0;
  CHECK(hipEventElapsedTime(&ms, a, b) == hipSuccess);
  return bytes_moved * (double)reps / (ms / 1e3) / 1e9;
}

static float* g_out;
static size_t g_win = kBytes;  // the working set: a window of the buffer, again and again (the GPU's caches)
static void lw(void* p) { write_k<<<1024, 256>>>((float4*)p, g_win / 16, 1.0f); }
static void lr(void* p) { read_k<<<1024, 256>>>((const float4*)p, g_win / 16, g_out); }
static void lrw(void* p) { rw_k<<<1024, 256>>>((float4*)p, g_win / 16); }

static hsa_agent_t g_cpu, g_gpu;
static hsa_amd_memory_pool_t g_coarse;
static bool g_has_coarse;
static hsa_status_t find_pool(hsa_amd_memory_pool_t p, void*) {
  hsa_amd_segment_t seg;
  uint32_t fl = 0;
  if (hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg) == HSA_STATUS_SUCCESS && seg == HSA_AMD_SEGMENT_GLOBAL &&
      hsa_amd_memory_pool_get_info(p, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &fl) == HSA_STATUS_SUCCESS &&
      (fl & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) && !g_has_coarse)
    g_coarse = p, g_has_coarse = true;
  return HSA_STATUS_SUCCESS;
}
static hsa_status_t find_agents(hsa_agent_t a, void*) {
  hsa_device_type_t t;
  if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t) != HSA_STATUS_SUCCESS) return HSA_STATUS_SUCCESS;
  if (t == HSA_DEVICE_TYPE_CPU) g_cpu = a, hsa_amd_agent_iterate_memory_pools(a, find_pool, nullptr);
  if (t == HSA_DEVICE_TYPE_GPU) g_gpu = a;
  return HSA_STATUS_SUCCESS;
}

static bool odirect(const char* file, void* p) {
  const int fd = open(file, O_RDONLY | O_DIRECT);
  if (fd < 0) return false;
  const ssize_t n = pread(fd, p, 64u << 20, 0);
  close(fd);
  return n == (ssize_t)(64u << 20);
}

int main(int argc, char** argv) {
  if (argc < 2) return std::printf("usage: hostbw <file>\n"), 2;
  CHECK(hipMalloc(&g_out, 4) == hipSuccess);
  CHECK(hsa_init() == HSA_STATUS_SUCCESS && hsa_iterate_agents(find_agents, nullptr) == HSA_STATUS_SUCCESS);
  std::printf("the CPU's coarse-grained pool: %s\n", g_has_coarse ? "found" : "none");
  struct Kind {
    const char* name;
    void* p;
    bool host;
  };
  std::vector<Kind> kinds;
  void* d = nullptr;
  CHECK(hipMalloc(&d, kBytes) == hipSuccess);
  kinds.push_back({"malloc", d, false});
  void* r = mmap(nullptr, kBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  CHECK(r != MAP_FAILED && hipHostRegister(r, kBytes, hipHostRegisterMapped) == hipSuccess);
  kinds.push_back({"register", r, true});
  void* nc = nullptr;
  if (hipHostMalloc(&nc, kBytes, hipHostMallocNonCoherent | hipHostMallocMapped) == hipSuccess) kinds.push_back({"nc", nc, true});
  else std::printf("nc: hipHostMalloc refused\n");
  void* cm = mmap(nullptr, kBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  void* cg = nullptr;
  if (g_has_coarse && hsa_amd_memory_lock_to_pool(cm, kBytes, &g_gpu, 1, g_coarse, 0, &cg) == HSA_STATUS_SUCCESS)
    kinds.push_back({"coarse", cg, true}), std::printf("coarse: GPU address %s the host's\n", cg == cm ? "is" : "is not");
  else std::printf("coarse: hsa_amd_memory_lock_to_pool refused\n");
  std::printf("%-9s %8s %9s %9s %12s  %s\n", "kind", "window", "write", "read", "read-write", "O_DIRECT read into it");
  for (const Kind& k : kinds) {
    const char* od = !k.host ? "-" : odirect(argv[1], k.name == std::string("coarse") ? cm : k.p) ? "ok" : "REFUSED";
    for (size_t win : {(size_t) 4 << 20, (size_t) 16 << 20, (size_t) 64 << 20, kBytes}) {
      g_win = win;
      const int reps = (int)(kBytes / win) * 4;
      const double w = gbps(lw, k.p, reps, win), rd = gbps(lr, k.p, reps, win), rw = gbps(lrw, k.p, reps, 2 * win);
      std::printf("%-9s %6zu MB %7.0f GB/s %7.0f GB/s %10.0f GB/s  %s\n", k.name, win >> 20, w, rd, rw, od);
    }
  }
  return 0;
}
