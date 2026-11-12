// 0.7.0 spike: can the NPU take osmos's memory without a copy? XRT buffer objects over (1) host memory registered with
// HIP, as host_slots' expert cache is (a userptr BO), and (2) hipMalloc's memory exported as a dma-buf (HSA's portable
// export) and imported by XRT. no NPU kernel yet (no AIE compiler): this asks whether XRT accepts and maps each, and
// whether the CPU sees the GPU's bytes through the BO's own map - not yet whether the NPU reads them.
// build: hipcc -O2 --offload-arch=gfx1151 spikes/npu/share.hip -I/usr/include -lxrt_coreutil -lhsa-runtime64 -o bin/npu_share
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"

static void report(const char* what, const std::exception& e) { std::printf("  %-44s refused: %s\n", what, e.what()); }

__global__ void fill(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
    p[i] = seed ^ (uint32_t) i;
}

int main() {
  const size_t bytes = 64u << 20, n = bytes / 4;
  xrt::device dev(0);
  std::printf("NPU: %s\n", dev.get_info<xrt::info::device::name>().c_str());

  // (1) host_slots' kind: anonymous pages, registered with HIP (mapped: the GPU's address the host's)
  void* m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
  void* gp = nullptr;
  if (hipHostRegister(m, bytes, hipHostRegisterMapped) != hipSuccess || hipHostGetDevicePointer(&gp, m, 0) != hipSuccess)
    return std::printf("hipHostRegister refused\n"), 1;
  fill<<<256, 256>>>((uint32_t*) gp, n, 0x1017u);
  (void) hipDeviceSynchronize();
  for (auto fl : {xrt::bo::flags::host_only, xrt::bo::flags::normal, xrt::bo::flags::cacheable}) {
    const char* name = fl == xrt::bo::flags::host_only ? "userptr BO, host_only" : fl == xrt::bo::flags::normal ? "userptr BO, normal" : "userptr BO, cacheable";
    try {
      xrt::bo bo(dev, m, bytes, fl, 0);
      const uint32_t* v = bo.map<const uint32_t*>();
      bool same = v != nullptr;
      for (size_t i = 0; same && i < n; i += 4096) same = v[i] == (0x1017u ^ (uint32_t) i);
      std::printf("  %-44s ok: map %s the pages (%p vs %p), the GPU's bytes %s, device address 0x%llx\n", name,
                  (const void*) v == m ? "is" : "is not", (const void*) v, m, same ? "seen" : "NOT seen",
                  (unsigned long long) bo.address());
    } catch (const std::exception& e) {
      report(name, e);
    }
  }

  // (2) hipMalloc's memory as a dma-buf, imported
  void* d = nullptr;
  if (hipMalloc(&d, bytes) != hipSuccess) return std::printf("hipMalloc refused\n"), 1;
  fill<<<256, 256>>>((uint32_t*) d, n, 0x0610u);
  (void) hipDeviceSynchronize();
  int fd = -1;
  uint64_t off = 0;
  const hsa_status_t st = hsa_amd_portable_export_dmabuf(d, bytes, &fd, &off);
  if (st != HSA_STATUS_SUCCESS || fd < 0) {
    std::printf("  %-44s refused by HSA (%d)\n", "hipMalloc -> dma-buf export", (int) st);
  } else {
    std::printf("  %-44s ok: fd %d, offset %llu\n", "hipMalloc -> dma-buf export", fd, (unsigned long long) off);
    try {
      xrt::bo bo(dev, (xrt::bo::export_handle) fd);
      std::printf("  %-44s ok: %zu bytes, device address 0x%llx\n", "dma-buf -> XRT import", bo.size(),
                  (unsigned long long) bo.address());
      try {
        const uint32_t* v = bo.map<const uint32_t*>();
        bool same = v != nullptr;
        for (size_t i = 0; same && i < n; i += 4096) same = v[i] == (0x0610u ^ (uint32_t) i);
        std::printf("  %-44s %s\n", "the imported BO mapped by the CPU", same ? "ok: the GPU's bytes seen" : "mapped, bytes NOT the GPU's");
      } catch (const std::exception& e) {
        report("the imported BO mapped by the CPU", e);
      }
    } catch (const std::exception& e) {
      report("dma-buf -> XRT import", e);
    }
    (void) hsa_amd_portable_close_dmabuf(fd);
  }
  (void) hipFree(d);
  (void) hipHostUnregister(m);
  munmap(m, bytes);
  return 0;
}
