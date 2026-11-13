// 0.7.0: what slows the GPU beside the NPU - the gate + up kernel run back to back until a stop file appears, its
// weights either (bo) in one XRT buffer of its own, or (userptr) views of a large HIP-registered host region as the
// expert cache's - each launch another expert's place in it (a stride through region), the IOMMU's translations
// (SVA) the NPU's for every page. the zeros of A and W: the time and the traffic, not the values.
// build: as ds_npu.hip; bin/ds_loop <artifact dir> bo|userptr <region GiB> <stop file>
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

int main(int argc, char** argv) {
  if (argc < 5) return std::printf("usage: ds_loop <dir> bo|userptr <GiB> <stop>\n"), 2;
  const std::string dir = argv[1], mode = argv[2], stop = argv[4];
  const size_t region = (size_t)std::atof(argv[3]) * (1ull << 30), w = 3041280, slot = (w + 4095) & ~(size_t)4095;
  xrt::device dev(0);
  xrt::xclbin xb(dir + "/final.xclbin");
  dev.register_xclbin(xb);
  xrt::hw_context ctx(dev, xb.get_uuid());
  xrt::kernel k(ctx, "MLIR_AIE");
  std::ifstream f(dir + "/insts.bin", std::ios::binary);
  std::vector<char> ins((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  xrt::bo instr(dev, ins.size(), xrt::bo::flags::cacheable, k.group_id(1));
  std::memcpy(instr.map<char*>(), ins.data(), ins.size()), instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  xrt::bo a(dev, 737280, xrt::bo::flags::host_only, k.group_id(5)), c(dev, 2359296, xrt::bo::flags::host_only, k.group_id(6)),
      spare(dev, 4096, xrt::bo::flags::host_only, k.group_id(7));
  std::memset(a.map<char*>(), 0, a.size()), a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  xrt::bo own, whole;
  void* m = nullptr;
  size_t n_slots = 1;
  if (mode == "bo") {
    own = xrt::bo(dev, 2 * slot, xrt::bo::flags::host_only, k.group_id(3));
    std::memset(own.map<char*>(), 0, own.size()), own.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  } else {
    m = mmap(nullptr, region, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (m == MAP_FAILED || hipHostRegister(m, region, hipHostRegisterMapped) != hipSuccess)
      return std::printf("region refused\n"), 1;
    whole = xrt::bo(dev, m, region, xrt::bo::flags::host_only, k.group_id(3));
    n_slots = region / slot - 1;
  }
  const auto t0 = std::chrono::steady_clock::now();
  uint64_t runs = 0;
  for (size_t i = 0; access(stop.c_str(), F_OK) != 0; i++, runs++) {
    xrt::bo g, u;
    if (mode == "bo")
      g = xrt::bo(own, w, 0), u = xrt::bo(own, w, slot);
    else  // two experts' places 97 slots apart, the next launch's past them: every page of the region in turn
      g = xrt::bo(whole, w, (i * 2 % n_slots) * slot), u = xrt::bo(whole, w, ((i * 2 + 97) % n_slots) * slot);
    auto r = k(3u, instr, (uint32_t)(ins.size() / 4), g, u, a, c, spare);
    r.wait();
  }
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%s: %llu launches in %.1f s, %.0f us each\n", mode.c_str(), (unsigned long long)runs, s, s / runs * 1e6);
  if (m) (void)hipHostUnregister(m), munmap(m, region);
  return 0;
}
