// 0.7.0: the engine's NPU chain for one expert part, on the engine's own memory - the GPU packs the part's tokens'
// rows of x (f32) into the kernel's A stream (kernels/npu_pack.hip: nearest-even bfp16), the NPU runs gate + up from
// the expert's bytes in HIP-registered host pages, the GPU unpacks C into gate and up rows. checked: (1) the GPU's A
// bytes against numpy's encoding of the same rule (a_even.bin), (2) the unpack against the Python run's own C, bit
// for bit (a.bin's floor A through the NPU, got.bin), (3) the chain's gate and up against the float64 reference of
// the nearest-even A (ref_even.bin). the dumps: fused/fused_ds.py with DUMP=dir.
// build: hipcc -O2 --offload-arch=gfx1151 spikes/npu/host/ds_chain.hip kernels/npu_pack.hip -Iinclude -Ikernels \
//          -Ispikes/npu/include -I/usr/include -lxrt_coreutil -lhsa-runtime64 -o bin/ds_chain
//   bin/ds_chain <artifact dir> <dump dir>
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

extern "C" {
#include "osmos/gpu/npu_pack.h"
}

static std::vector<char> slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + p);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void HIP(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }

struct DmaBo {  // hipMalloc'd memory, the NPU's through a dma-buf
  void* d = nullptr;
  int fd = -1;
  xrt::bo bo;
  DmaBo(xrt::device& dev, size_t bytes) {
    HIP(hipMalloc(&d, bytes));
    uint64_t off = 0;
    if (hsa_amd_portable_export_dmabuf(d, bytes, &fd, &off) != HSA_STATUS_SUCCESS || off)
      throw std::runtime_error("dma-buf export");
    bo = xrt::bo(dev, (xrt::bo::export_handle)fd);
  }
  ~DmaBo() { (void)hsa_amd_portable_close_dmabuf(fd), (void)hipFree(d); }
};

int main(int argc, char** argv) {
  if (argc < 3) return std::printf("usage: ds_chain <artifacts> <dump>\n"), 2;
  const std::string art = argv[1], dump = argv[2];
  const uint32_t K = 5120, N = 2304, T = NPU_TOKENS;
  try {
    xrt::device dev(0);
    xrt::xclbin xb(art + "/final.xclbin");
    dev.register_xclbin(xb);
    xrt::hw_context ctx(dev, xb.get_uuid());
    xrt::kernel k(ctx, "MLIR_AIE");
    const std::vector<char> ins = slurp(art + "/insts.bin");
    xrt::bo instr(dev, ins.size(), xrt::bo::flags::cacheable, k.group_id(1));
    std::memcpy(instr.map<char*>(), ins.data(), ins.size()), instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::bo spare(dev, 4096, xrt::bo::flags::host_only, k.group_id(7));
    const uint32_t ninstr = (uint32_t)(ins.size() / 4);

    const std::vector<char> wg = slurp(dump + "/wg.bin"), wu = slurp(dump + "/wu.bin"), x = slurp(dump + "/x.bin"),
                            afloor = slurp(dump + "/a.bin"), aeven = slurp(dump + "/a_even.bin"),
                            got = slurp(dump + "/got.bin"), ref = slurp(dump + "/ref_even.bin");
    // the expert cache's kind: one registered region, one BO over it, an expert's matrices sub-BOs
    const size_t slot = (wg.size() + 4095) & ~(size_t)4095, region = 2 * slot;
    void* m = mmap(nullptr, region, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    HIP(hipHostRegister(m, region, hipHostRegisterMapped));
    std::memcpy(m, wg.data(), wg.size()), std::memcpy((char*)m + slot, wu.data(), wu.size());
    xrt::bo whole(dev, m, region, xrt::bo::flags::host_only, k.group_id(3));
    xrt::bo bg(whole, wg.size(), 0), bu(whole, wu.size(), slot);

    const uint64_t ab = npu_a_bytes(K), cf = npu_c_floats(N, 2);
    DmaBo A(dev, ab), C(dev, cf * 4);
    float *dx = nullptr, *dg = nullptr, *du = nullptr;
    int32_t* dtok = nullptr;
    uint32_t* drow = nullptr;
    HIP(hipMalloc(&dx, x.size())), HIP(hipMalloc(&dg, (size_t)T * N * 4)), HIP(hipMalloc(&du, (size_t)T * N * 4));
    HIP(hipMalloc(&dtok, T * 4)), HIP(hipMalloc(&drow, 4));
    HIP(hipMemcpy(dx, x.data(), x.size(), hipMemcpyHostToDevice));
    std::vector<int32_t> tok(T);
    for (uint32_t t = 0; t < T; t++) tok[t] = (int32_t)t;
    const uint32_t row0 = 0;
    HIP(hipMemcpy(dtok, tok.data(), T * 4, hipMemcpyHostToDevice)), HIP(hipMemcpy(drow, &row0, 4, hipMemcpyHostToDevice));
    auto run = [&]() { auto r = k(3u, instr, ninstr, bg, bu, A.bo, C.bo, spare); r.wait(); };

    // (1) the GPU's packing
    if (!npu_pack_a(dx, K, dtok, 1, (uint8_t*)A.d, nullptr)) throw std::runtime_error("npu_pack_a");
    HIP(hipDeviceSynchronize());
    std::vector<uint8_t> abytes(ab);
    HIP(hipMemcpy(abytes.data(), A.d, ab, hipMemcpyDeviceToHost));
    size_t da = 0;
    for (size_t i = 0; i < ab; i++) da += abytes[i] != (uint8_t)aeven[i];
    std::printf("(1) the GPU's A: %s (%zu of %llu bytes differ from numpy's)\n", da ? "DIFFERS" : "byte-identical", da,
                (unsigned long long)ab);

    // (2) the unpack: Python's floor A through the NPU, unpacked, against the Python run's gate and up
    HIP(hipMemcpy(A.d, afloor.data(), ab, hipMemcpyHostToDevice));
    run();
    if (!npu_unpack_c((const float*)C.d, N, 24, 2, 1, drow, dg, du, nullptr)) throw std::runtime_error("npu_unpack_c");
    std::vector<float> g((size_t)T * N), u((size_t)T * N);
    HIP(hipMemcpy(g.data(), dg, g.size() * 4, hipMemcpyDeviceToHost)), HIP(hipMemcpy(u.data(), du, u.size() * 4, hipMemcpyDeviceToHost));
    const float* gw = (const float*)got.data();
    size_t dd = 0;
    for (size_t i = 0; i < g.size(); i++) dd += std::memcmp(&g[i], &gw[i], 4) != 0, dd += std::memcmp(&u[i], &gw[g.size() + i], 4) != 0;
    std::printf("(2) the unpack: %s (%zu of %zu values differ from the Python run's)\n", dd ? "DIFFERS" : "bit-identical",
                dd, 2 * g.size());

    // (3) the whole chain, timed: pack on the GPU, the NPU, unpack on the GPU
    std::vector<double> t;
    for (int i = 0; i < 12; i++) {
      const double t0 = now_us();
      if (!npu_pack_a(dx, K, dtok, 1, (uint8_t*)A.d, nullptr)) throw std::runtime_error("npu_pack_a");
      HIP(hipDeviceSynchronize());
      const double t1 = now_us();
      run();
      const double t2 = now_us();
      if (!npu_unpack_c((const float*)C.d, N, 24, 2, 1, drow, dg, du, nullptr)) throw std::runtime_error("npu_unpack_c");
      HIP(hipDeviceSynchronize());
      if (i >= 2) t.push_back(t1 - t0), t.push_back(t2 - t1), t.push_back(now_us() - t2);
    }
    double tp = 0, tn = 0, tu = 0;
    for (size_t i = 0; i < t.size(); i += 3) tp += t[i], tn += t[i + 1], tu += t[i + 2];
    const double n = t.size() / 3.0;
    HIP(hipMemcpy(g.data(), dg, g.size() * 4, hipMemcpyDeviceToHost)), HIP(hipMemcpy(u.data(), du, u.size() * 4, hipMemcpyDeviceToHost));
    const float* rf = (const float*)ref.data();
    double worst = 0, rmax = 0, sxy = 0, sxx = 0, syy = 0, sx = 0, sy = 0;
    const size_t V = 2 * g.size();
    for (size_t i = 0; i < V; i++) {
      const double a = i < g.size() ? g[i] : u[i - g.size()], b = rf[i];
      worst = std::max(worst, std::fabs(a - b)), rmax = std::max(rmax, std::fabs(b));
      sx += a, sy += b, sxx += a * a, syy += b * b, sxy += a * b;
    }
    const double corr = (sxy - sx * sy / V) / std::sqrt((sxx - sx * sx / V) * (syy - sy * sy / V));
    std::printf("(3) the chain: corr %.6f, max err / max |C| %.2e against the nearest-even reference; pack %.0f us, "
                "NPU %.0f us, unpack %.0f us (each with its sync)\n", corr, worst / rmax, tp / n, tn / n, tu / n);
    (void)hipHostUnregister(m), munmap(m, region);
  } catch (const std::exception& e) {
    return std::printf("failed: %s\n", e.what()), 1;
  }
  return 0;
}
