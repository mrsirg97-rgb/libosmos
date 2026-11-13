// 0.7.0: two formats' kernels at once - DeepSeek's layers alternate IQ1_S and IQ2_XXS experts, each its own xclbin.
// both registered, a hardware context each (both the whole array): may they live together, and what does going from
// one to the other cost against running one? each launch's wall time: A A A ..., B B B ..., then A B A B ...
// (zeros for weights and A: the time, not the values). build: as ds_npu.hip; bin/ds_two <dir A> <dir B>
#include <hip/hip_runtime.h>

#include <algorithm>
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

static double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct K {
  xrt::hw_context ctx;
  xrt::kernel k;
  xrt::bo instr, w, a, c, spare;
  uint32_t n;
  K(xrt::device& dev, const std::string& dir) {
    xrt::xclbin xb(dir + "/final.xclbin");
    dev.register_xclbin(xb);
    ctx = xrt::hw_context(dev, xb.get_uuid());
    k = xrt::kernel(ctx, "MLIR_AIE");
    std::ifstream f(dir + "/insts.bin", std::ios::binary);
    std::vector<char> ins((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    n = (uint32_t)(ins.size() / 4);
    instr = xrt::bo(dev, ins.size(), xrt::bo::flags::cacheable, k.group_id(1));
    std::memcpy(instr.map<char*>(), ins.data(), ins.size()), instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    w = xrt::bo(dev, 3041280, xrt::bo::flags::host_only, k.group_id(3));
    a = xrt::bo(dev, 737280, xrt::bo::flags::host_only, k.group_id(5));
    c = xrt::bo(dev, 4u << 20, xrt::bo::flags::host_only, k.group_id(6));  // a matrix's rows: gate / up 1.2 MB, the down 2.6
    spare = xrt::bo(dev, 4u << 20, xrt::bo::flags::host_only, k.group_id(7));  // the second output (up's)
    std::memset(w.map<char*>(), 0, w.size()), std::memset(a.map<char*>(), 0, a.size());
    w.sync(XCL_BO_SYNC_BO_TO_DEVICE), a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
  double run() {
    const double t0 = now_us();
    auto r = k(3u, instr, n, w, w, a, c, spare);
    r.wait();
    return now_us() - t0;
  }
};

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  if (argc < 3) return std::printf("usage: ds_two <dir A> <dir B>\n"), 2;
  try {
    xrt::device dev(0);
    K a(dev, argv[1]);
    std::printf("A loaded\n");
    K b(dev, argv[2]);
    std::printf("B loaded: both contexts live\n");
    std::vector<double> ta, tb, tab_a, tab_b;
    for (int i = 0; i < 3; i++) a.run(), b.run();
    for (int i = 0; i < 20; i++) ta.push_back(a.run());
    for (int i = 0; i < 20; i++) tb.push_back(b.run());
    for (int i = 0; i < 20; i++) tab_a.push_back(a.run()), tab_b.push_back(b.run());
    std::printf("A alone %.0f us, B alone %.0f us; alternating A %.0f us, B %.0f us (medians, 20 each)\n", median(ta),
                median(tb), median(tab_a), median(tab_b));
    // a layer's worth: 20 launches of one, then 20 of the other (a layer of A's parts, the next of B's)
    std::vector<double> blk;
    for (int r = 0; r < 4; r++) {
      const double t0 = now_us();
      for (int i = 0; i < 20; i++) a.run();
      for (int i = 0; i < 20; i++) b.run();
      blk.push_back((now_us() - t0) / 40);
    }
    std::printf("layers of 20 each, alternating: %.0f us a launch\n", median(blk));
    // queued: 40 launches started back to back, then waited (the engine's way) - against one at a time
    for (int r = 0; r < 3; r++) {
      const double t0 = now_us();
      std::vector<xrt::run> q;
      for (int i = 0; i < 40; i++) q.push_back(a.k(3u, a.instr, a.n, a.w, a.w, a.a, a.c, a.spare));
      for (auto& x : q) x.wait();
      std::printf("40 queued: %.0f us a launch\n", (now_us() - t0) / 40);
    }
  } catch (const std::exception& e) {
    return std::printf("failed: %s\n", e.what()), 1;
  }
  return 0;
}
