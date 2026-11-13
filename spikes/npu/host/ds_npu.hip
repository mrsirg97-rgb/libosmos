// 0.7.0: the engine's side of the NPU expert kernel (fused/fused_ds.py: DeepSeek's gate + up, 128 tokens) - its
// xclbin and instruction stream run from C++ through XRT, as osmos will, three ways, each against the Python run's
// output bit for bit: (1) XRT's own buffers, the inputs copied in; (2) the weights in host pages registered with HIP
// (host_slots' kind: the expert cache) as a userptr BO, no copy; (3) A and C in hipMalloc's memory (the step's
// buffers) imported as dma-bufs, no copy - the GPU writing A and reading C through them. each timed end to end.
// build: hipcc -O2 --offload-arch=gfx1151 spikes/npu/host/ds_npu.hip -Ispikes/npu/include -I/usr/include \
//          -lxrt_coreutil -lhsa-runtime64 -o bin/ds_npu
//   bin/ds_npu <artifact dir: final.xclbin, insts.bin> <dump dir: wg/wu/a/c.bin>
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <sys/mman.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

static std::vector<char> slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + p);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Npu {
  xrt::device dev;
  xrt::hw_context ctx;
  xrt::kernel k;
  xrt::bo instr, spare;
  uint32_t ninstr;
  Npu(const std::string& dir) : dev(0) {
    xrt::xclbin xb(dir + "/final.xclbin");
    dev.register_xclbin(xb);
    ctx = xrt::hw_context(dev, xb.get_uuid());
    k = xrt::kernel(ctx, "MLIR_AIE");
    const std::vector<char> ins = slurp(dir + "/insts.bin");
    ninstr = (uint32_t)(ins.size() / 4);
    instr = xrt::bo(dev, ins.size(), xrt::bo::flags::cacheable, k.group_id(1));
    std::memcpy(instr.map<char*>(), ins.data(), ins.size());
    instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    spare = xrt::bo(dev, 4096, xrt::bo::flags::host_only, k.group_id(7));
  }
  // one launch, waited; its wall time
  double run(xrt::bo& wg, xrt::bo& wu, xrt::bo& a, xrt::bo& c) {
    const double t0 = now_us();
    auto r = k(3u, instr, ninstr, wg, wu, a, c, spare);
    r.wait();
    return now_us() - t0;
  }
};

static size_t diff_words(const void* x, const void* y, size_t bytes) {
  const uint32_t *p = (const uint32_t*)x, *q = (const uint32_t*)y;
  size_t d = 0;
  for (size_t i = 0; i < bytes / 4; i++) d += p[i] != q[i];
  return d;
}

static void timed(Npu& n, const char* what, xrt::bo& wg, xrt::bo& wu, xrt::bo& a, xrt::bo& c, const void* got,
                  const std::vector<char>& want, bool sync_c) {
  std::vector<double> t;
  for (int i = 0; i < 12; i++) {
    std::memset((void*)got, 0xFF, want.size()), sync_c ? c.sync(XCL_BO_SYNC_BO_TO_DEVICE) : void();
    t.push_back(n.run(wg, wu, a, c));
    if (sync_c) c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }
  std::vector<double> s(t.begin() + 2, t.end());
  std::sort(s.begin(), s.end());
  const size_t d = diff_words(got, want.data(), want.size());
  std::printf("  %-52s %s (%zu of %zu words differ); median %.0f us a launch (min %.0f) -> %.2f TFLOP/s\n", what,
              d ? "DIFFERS" : "bit-identical", d, want.size() / 4, s[s.size() / 2], s[0],
              6.0398e9 / s[s.size() / 2] / 1e6);
}

int main(int argc, char** argv) {
  if (argc < 3) return std::printf("usage: ds_npu <artifacts> <dump>\n"), 2;
  const std::string art = argv[1], dump = argv[2];
  try {
    Npu n(art);
    const std::vector<char> wg = slurp(dump + "/wg.bin"), wu = slurp(dump + "/wu.bin"), a = slurp(dump + "/a.bin"),
                            c = slurp(dump + "/c.bin");
    std::printf("NPU %s: %u instruction words; W %zu + %zu, A %zu, C %zu bytes\n",
                n.dev.get_info<xrt::info::device::name>().c_str(), n.ninstr, wg.size(), wu.size(), a.size(), c.size());

    // (1) XRT's buffers
    {
      xrt::bo bg(n.dev, wg.size(), xrt::bo::flags::host_only, n.k.group_id(3)),
          bu(n.dev, wu.size(), xrt::bo::flags::host_only, n.k.group_id(4)),
          ba(n.dev, a.size(), xrt::bo::flags::host_only, n.k.group_id(5)),
          bc(n.dev, c.size(), xrt::bo::flags::host_only, n.k.group_id(6));
      std::memcpy(bg.map<char*>(), wg.data(), wg.size()), std::memcpy(bu.map<char*>(), wu.data(), wu.size());
      std::memcpy(ba.map<char*>(), a.data(), a.size());
      bg.sync(XCL_BO_SYNC_BO_TO_DEVICE), bu.sync(XCL_BO_SYNC_BO_TO_DEVICE), ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      timed(n, "(1) XRT's buffers, inputs copied in", bg, bu, ba, bc, bc.map<char*>(), c, true);
    }

    // (2) the weights in HIP-registered host pages (the expert cache's kind), both in one region at a slot's offsets
    const size_t slot = (wg.size() + 4095) & ~(size_t)4095, region = 2 * slot;
    void* m = mmap(nullptr, region, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    void* gm = nullptr;
    if (m == MAP_FAILED || hipHostRegister(m, region, hipHostRegisterMapped) != hipSuccess ||
        hipHostGetDevicePointer(&gm, m, 0) != hipSuccess)
      return std::printf("host pages refused\n"), 1;
    std::memcpy(m, wg.data(), wg.size()), std::memcpy((char*)m + slot, wu.data(), wu.size());
    xrt::bo hg(n.dev, m, wg.size(), xrt::bo::flags::host_only, n.k.group_id(3));
    xrt::bo hu(n.dev, (char*)m + slot, wu.size(), xrt::bo::flags::host_only, n.k.group_id(4));
    {
      xrt::bo ba(n.dev, a.size(), xrt::bo::flags::host_only, n.k.group_id(5)),
          bc(n.dev, c.size(), xrt::bo::flags::host_only, n.k.group_id(6));
      std::memcpy(ba.map<char*>(), a.data(), a.size()), ba.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      timed(n, "(2) W in HIP-registered host pages (userptr)", hg, hu, ba, bc, bc.map<char*>(), c, true);
    }

    // (3) A and C in hipMalloc's memory, imported as dma-bufs; the GPU writes A and reads C
    void *da = nullptr, *dc = nullptr;
    if (hipMalloc(&da, a.size()) != hipSuccess || hipMalloc(&dc, c.size()) != hipSuccess)
      return std::printf("hipMalloc refused\n"), 1;
    (void)hipMemcpy(da, a.data(), a.size(), hipMemcpyHostToDevice);
    int fa = -1, fc = -1;
    uint64_t oa = 0, oc = 0;
    if (hsa_amd_portable_export_dmabuf(da, a.size(), &fa, &oa) != HSA_STATUS_SUCCESS ||
        hsa_amd_portable_export_dmabuf(dc, c.size(), &fc, &oc) != HSA_STATUS_SUCCESS)
      return std::printf("dma-buf export refused\n"), 1;
    std::printf("  dma-buf offsets: A %llu, C %llu\n", (unsigned long long)oa, (unsigned long long)oc);
    xrt::bo ia(n.dev, (xrt::bo::export_handle)fa), ic(n.dev, (xrt::bo::export_handle)fc);
    std::vector<char> back(c.size());
    {
      std::vector<double> t;
      for (int i = 0; i < 12; i++) {
        (void)hipMemset(dc, 0xFF, c.size()), (void)hipDeviceSynchronize();
        t.push_back(n.run(hg, hu, ia, ic));
      }
      (void)hipMemcpy(back.data(), dc, c.size(), hipMemcpyDeviceToHost);
      std::vector<double> s(t.begin() + 2, t.end());
      std::sort(s.begin(), s.end());
      const size_t d = diff_words(back.data(), c.data(), c.size());
      std::printf("  %-52s %s (%zu of %zu words differ); median %.0f us a launch (min %.0f) -> %.2f TFLOP/s\n",
                  "(3) + A, C hipMalloc'd (dma-buf), C read by the GPU", d ? "DIFFERS" : "bit-identical", d,
                  c.size() / 4, s[s.size() / 2], s[0], 6.0398e9 / s[s.size() / 2] / 1e6);
    }
    // (4) what a launch's buffers cost to make: a userptr BO over an expert's pages (made each launch), against
    // sub-BOs of one BO over the whole region (made once), the latter run for its bits
    {
      double t0 = now_us();
      for (int i = 0; i < 50; i++) { xrt::bo x(n.dev, m, wg.size(), xrt::bo::flags::host_only, n.k.group_id(3)); }
      const double up = (now_us() - t0) / 50;
      xrt::bo whole(n.dev, m, region, xrt::bo::flags::host_only, n.k.group_id(3));
      t0 = now_us();
      for (int i = 0; i < 1000; i++) { xrt::bo x(whole, wg.size(), 0); }
      const double sub = (now_us() - t0) / 1000;
      xrt::bo sg(whole, wg.size(), 0), su(whole, wu.size(), slot);
      (void)hipMemset(dc, 0xFF, c.size()), (void)hipDeviceSynchronize();
      const double t = n.run(sg, su, ia, ic);
      (void)hipMemcpy(back.data(), dc, c.size(), hipMemcpyDeviceToHost);
      std::printf("  (4) a userptr BO over an expert: %.0f us to make; a sub-BO of the region's: %.2f us; the sub-BOs' run "
                  "%s, %.0f us\n", up, sub, diff_words(back.data(), c.data(), c.size()) ? "DIFFERS" : "bit-identical", t);
    }
    (void)hsa_amd_portable_close_dmabuf(fa), (void)hsa_amd_portable_close_dmabuf(fc);
    (void)hipFree(da), (void)hipFree(dc), (void)hipHostUnregister(m), munmap(m, region);
  } catch (const std::exception& e) {
    return std::printf("failed: %s\n", e.what()), 1;
  }
  return 0;
}
