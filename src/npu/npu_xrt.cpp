// the NPU through XRT (include/osmos/npu/npu.h): built with NPU=1
#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <chrono>
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

extern "C" {
#include "osmos/gpu/npu_pack.h"
#include "osmos/npu/npu.h"
}

namespace {

struct Region {
  const uint8_t* base;
  uint64_t bytes;
  xrt::bo bo;
};

struct Dma {  // hipMalloc'd, the NPU's through its dma-buf
  void* d = nullptr;
  int fd = -1;
  xrt::bo bo;
};

double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void say(char* err, const char* what, const char* why) { std::snprintf(err, NPU_ERR, "npu: %s%s%s", what, why ? ": " : "", why ? why : ""); }

}  // namespace

struct Kern {  // a format and kind's kernel: its own hardware context
  NpuKind kind;
  uint32_t qtype;
  xrt::hw_context ctx;
  xrt::kernel k;
  xrt::bo instr, spare;
  uint32_t ninstr;
};

struct Npu {
  xrt::device dev;
  std::vector<Kern> kern;
  uint32_t max_parts = 0;
  uint64_t a_bytes[NPU_KINDS] = {}, c_bytes[NPU_KINDS] = {};  // a part's, by kind
  Dma a, c;
  std::vector<Region> regions;
  std::vector<xrt::run> runs;
  NpuStats st{};
};

static bool dma(Npu* n, Dma& m, uint64_t bytes, char* err) {
  uint64_t off = 0;
  if (hipMalloc(&m.d, bytes) != hipSuccess) return say(err, "hipMalloc", nullptr), false;
  if (hsa_amd_portable_export_dmabuf(m.d, bytes, &m.fd, &off) != HSA_STATUS_SUCCESS || off)
    return say(err, "a dma-buf export", nullptr), false;
  m.bo = xrt::bo(n->dev, (xrt::bo::export_handle) m.fd);
  return true;
}

extern "C" bool npu_built(void) { return true; }

static const struct {
  const char* sub;
  NpuKind kind;
  uint32_t qtype;
} kFormats[] = {{"iq2xxs", NPU_GATE_UP, 16}, {"iq1s", NPU_GATE_UP, 19},  // GGUF's IQ2_XXS, IQ1_S
                {"iq2xxs-down", NPU_DOWN, 16}, {"iq1s-down", NPU_DOWN, 19}};

extern "C" Npu* npu_open(const char* dir, uint32_t D, uint32_t FF, uint32_t max_parts, char err[NPU_ERR]) {
  Npu* n = nullptr;
  try {
    n = new Npu();
    n->dev = xrt::device(0);
    for (const auto& fm : kFormats) {
      const std::string d = std::string(dir) + "/" + fm.sub;
      std::ifstream f(d + "/insts.bin", std::ios::binary);
      if (!f) continue;
      const std::vector<char> ins((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      xrt::xclbin xb(d + "/final.xclbin");
      n->dev.register_xclbin(xb);
      Kern kn{fm.kind, fm.qtype, xrt::hw_context(n->dev, xb.get_uuid()), {}, {}, {}, (uint32_t) (ins.size() / 4)};
      kn.k = xrt::kernel(kn.ctx, "MLIR_AIE");
      kn.instr = xrt::bo(n->dev, ins.size(), xrt::bo::flags::cacheable, kn.k.group_id(1));
      std::memcpy(kn.instr.map<char*>(), ins.data(), ins.size());
      kn.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      kn.spare = xrt::bo(n->dev, 4096, xrt::bo::flags::host_only, kn.k.group_id(7));
      n->kern.push_back(std::move(kn));
    }
    if (n->kern.empty()) throw std::runtime_error(std::string("no kernel in ") + dir + " (iq2xxs/, iq1s/, ...-down/)");
    n->max_parts = max_parts;
    n->a_bytes[NPU_GATE_UP] = npu_a_bytes(D), n->c_bytes[NPU_GATE_UP] = npu_c_floats(FF, 2) * 4;
    n->a_bytes[NPU_DOWN] = npu_a_bytes(FF), n->c_bytes[NPU_DOWN] = npu_c_floats(D, 1) * 4;
    const uint64_t a = std::max(n->a_bytes[0], n->a_bytes[1]), c = std::max(n->c_bytes[0], n->c_bytes[1]);
    if (!dma(n, n->a, max_parts * a, err) || !dma(n, n->c, max_parts * c, err)) return npu_close(n), nullptr;
    n->runs.reserve(max_parts);
    return n;
  } catch (const std::exception& e) {
    say(err, "open", e.what());
    npu_close(n);
    return nullptr;
  }
}

static Kern* kern(Npu* n, NpuKind kind, uint32_t qtype) {
  for (Kern& k : n->kern)
    if (k.kind == kind && k.qtype == qtype) return &k;
  return nullptr;
}

extern "C" bool npu_has(const Npu* n, NpuKind kind, uint32_t qtype) { return n && kern((Npu*) n, kind, qtype); }

extern "C" void npu_close(Npu* n) {
  if (!n) return;
  n->runs.clear(), n->regions.clear(), n->kern.clear();
  for (Dma* m : {&n->a, &n->c}) {
    m->bo = xrt::bo();
    if (m->fd >= 0) (void) hsa_amd_portable_close_dmabuf(m->fd);
    if (m->d) (void) hipFree(m->d);
  }
  delete n;
}

extern "C" bool npu_region(Npu* n, void* base, uint64_t bytes, char err[NPU_ERR]) {
  try {
    n->regions.push_back({(const uint8_t*) base, bytes, xrt::bo(n->dev, base, bytes, xrt::bo::flags::host_only, n->kern[0].k.group_id(3))});
    return true;
  } catch (const std::exception& e) {
    return say(err, "a region", e.what()), false;
  }
}

extern "C" void* npu_a(const Npu* n) { return n->a.d; }
extern "C" void* npu_c(const Npu* n) { return n->c.d; }
extern "C" NpuStats npu_stats(const Npu* n) { return n ? n->st : NpuStats{}; }

// w's view in the region that holds all of its bytes
static bool view(Npu* n, const void* w, uint64_t bytes, xrt::bo& out) {
  const uint8_t* p = (const uint8_t*) w;
  for (Region& r : n->regions)
    if (p >= r.base && p + bytes <= r.base + r.bytes) return out = xrt::bo(r.bo, bytes, (size_t) (p - r.base)), true;
  return false;
}

extern "C" bool npu_run(Npu* n, NpuKind kind, uint32_t qtype, uint32_t parts, const void* const* w0,
                        const void* const* w1, uint64_t w_bytes, char err[NPU_ERR]) {
  if (parts > n->max_parts) return say(err, "more parts than its buffers", nullptr), false;
  Kern* kn = kern(n, kind, qtype);
  if (!kn) return say(err, "no kernel for the kind and format", nullptr), false;
  const double t0 = now_us();
  const uint64_t ab = n->a_bytes[kind], cb = n->c_bytes[kind];
  try {
    n->runs.clear();
    for (uint32_t p = 0; p < parts; p++) {
      xrt::bo x, y;
      if (!view(n, w0[p], w_bytes, x) || (kind == NPU_GATE_UP && !view(n, w1[p], w_bytes, y)))
        return say(err, "an expert outside its regions", nullptr), false;
      if (kind == NPU_DOWN) y = x;  // the down kernel's second weight argument is unread
      xrt::bo a(n->a.bo, ab, p * ab), c(n->c.bo, cb, p * cb);
      n->runs.push_back(kn->k(3u, kn->instr, kn->ninstr, x, y, a, c, kn->spare));
    }
    for (xrt::run& r : n->runs)
      if (r.wait() != ERT_CMD_STATE_COMPLETED) return say(err, "a launch did not complete", nullptr), false;
  } catch (const std::exception& e) {
    return say(err, "a launch", e.what()), false;
  }
  n->st.calls++, n->st.parts += parts, n->st.wait_us += now_us() - t0;
  return true;
}
