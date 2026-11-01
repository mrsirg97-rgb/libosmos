// the CPU lane's pool (src/cpu/lane.c) against the GPU lane's steps for the same experts (q8_round, q8_gate_up,
// q8_round, q8_group_tok) on ds1017's - an IQ1_M, an IQ2_XXS and an IQ2_XS layer, three experts of 1, 2 and 3 rows,
// a clamp that bites and one that does not: every output row the same bits. and the hand-off: a stream queued to wait
// (ops_wait) on the job's value in host memory before the job is given, then its rows copied in (ops_copy_rows) -
// the GPU's copies the CPU's bits, after the post
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

extern "C" {
#include "substrate/cpu/lane.h"
#include "substrate/cpu/q8.h"
#include "substrate/gpu/ops.h"
#include "substrate/gpu/q8.h"
#include "substrate/model/gguf.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

constexpr uint32_t kNe = 3, kRows = 6, kBlock = 32, kThreads = 6;
constexpr uint32_t kCnt[kNe] = {1, 2, 3};
constexpr int kExpert[kNe] = {7, 300, 41};
constexpr int kDelayMs = 30;  // the job given this long after the stream's wait is queued

static uint32_t rng = 1017;
static float uniform() {
  rng = rng * 1664525u + 1013904223u;
  return (float) (rng >> 8) / (float) (1u << 23) - 1.0f;
}

static std::vector<uint8_t> slice(const Gguf* g, const char* name, int e, uint32_t* type, uint32_t* cols,
                                  uint32_t* rows) {
  const GgufTensor* t = gguf_tensor(g, name);
  CHECK(t && t->n_dims == 3);
  *type = t->type, *cols = (uint32_t) t->dims[0], *rows = (uint32_t) t->dims[1];
  const uint64_t bytes = t->bytes / t->dims[2];
  std::vector<uint8_t> w(bytes);
  const int fd = open(g->paths[t->file], O_RDONLY);
  CHECK(fd >= 0 && pread(fd, w.data(), bytes, (off_t) (t->offset + (uint64_t) e * bytes)) == (ssize_t) bytes);
  close(fd);
  return w;
}

template <class T>
static T* dev_of(const std::vector<T>& v) {
  T* d;
  CHECK(hipMalloc(&d, v.size() * sizeof(T)) == hipSuccess &&
        hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice) == hipSuccess);
  return d;
}

static void layer(const Gguf* g, int l) {
  char n[3][64];
  std::snprintf(n[0], 64, "blk.%d.ffn_gate_exps.weight", l), std::snprintf(n[1], 64, "blk.%d.ffn_up_exps.weight", l);
  std::snprintf(n[2], 64, "blk.%d.ffn_down_exps.weight", l);
  std::vector<std::vector<uint8_t>> w(3 * kNe);  // [matrix][expert]
  uint32_t type[3], cols[3], rows[3];
  for (int m = 0; m < 3; m++)
    for (uint32_t k = 0; k < kNe; k++) w[m * kNe + k] = slice(g, n[m], kExpert[k], &type[m], &cols[m], &rows[m]);
  const uint32_t D = cols[0], FF = rows[0];
  CHECK(type[0] == type[1] && cols[2] == FF && rows[2] == D);
  std::vector<float> x((size_t) kRows * D), sc(kRows);
  for (float& v : x) v = uniform() * 3.0f;
  for (float& v : sc) v = 0.25f + uniform() * uniform();
  std::vector<int32_t> cnt(kNe), xrow(kNe), idx(kRows);
  for (uint32_t k = 0, r = 0; k < kNe; r += kCnt[k], k++) cnt[k] = (int32_t) kCnt[k], xrow[k] = (int32_t) r;
  for (uint32_t r = 0; r < kRows; r++) idx[r] = (int32_t) r;
  // the GPU's: the matrices, their pointer tables, x, the weights
  std::vector<uint8_t*> dw(3 * kNe);
  for (size_t i = 0; i < dw.size(); i++) dw[i] = dev_of(w[i]);
  uint8_t** dtab = dev_of(dw);
  int32_t *dcnt = dev_of(cnt), *dxrow = dev_of(xrow), *didx = dev_of(idx);
  float *dx = dev_of(x), *dsc = dev_of(sc), *dd, *dh, *dhd, *dy, *dcopy;
  int8_t *dq, *dqh;
  CHECK(hipMalloc(&dq, (size_t) kRows * D) == hipSuccess && hipMalloc(&dd, kRows * (D / kBlock) * 4) == hipSuccess &&
        hipMalloc(&dh, (size_t) kRows * FF * 4) == hipSuccess && hipMalloc(&dqh, (size_t) kRows * FF) == hipSuccess &&
        hipMalloc(&dhd, kRows * (FF / kBlock) * 4) == hipSuccess && hipMalloc(&dy, (size_t) kRows * D * 4) == hipSuccess &&
        hipMalloc(&dcopy, (size_t) kRows * D * 4) == hipSuccess);
  // the CPU's: y and the job's value in host memory the GPU maps
  float* y;
  uint32_t* seq;
  CHECK(hipHostMalloc((void**) &y, (size_t) kRows * D * 4, hipHostMallocMapped | hipHostMallocCoherent) == hipSuccess &&
        hipHostMalloc((void**) &seq, 4, hipHostMallocMapped | hipHostMallocCoherent) == hipSuccess);
  float* ydev;
  uint32_t* seqdev;
  CHECK(hipHostGetDevicePointer((void**) &ydev, y, 0) == hipSuccess &&
        hipHostGetDevicePointer((void**) &seqdev, seq, 0) == hipSuccess);
  CpuLaneExpert e[kNe];
  for (uint32_t k = 0; k < kNe; k++)
    e[k] = {w[k].data(), w[kNe + k].data(), w[2 * kNe + k].data(), (uint32_t) xrow[k], kCnt[k]};
  CpuLane* c = cpu_lane_create((int) kThreads, kRows, D, FF);
  CHECK(c);
  *seq = 0;
  uint32_t value = 0;
  for (float lim : {0.75f, 1e30f}) {
    // the GPU lane's steps
    CHECK(q8_round(dx, kRows, D, D, dq, dd, nullptr) &&
          q8_gate_up(type[0], (const void* const*) dtab, (const void* const*) dtab + kNe, dcnt, dxrow, kNe, 3, FF, D, dq,
                     dd, dsc, lim, dh, FF, nullptr) &&
          q8_round(dh, kRows, FF, FF, dqh, dhd, nullptr) &&
          q8_group_tok(type[2], (const void* const*) dtab + 2 * kNe, dcnt, dxrow, kNe, 3, D, FF, dqh, dhd, dy, D,
                       nullptr) &&
          hipDeviceSynchronize() == hipSuccess);
    std::vector<float> ref((size_t) kRows * D), got(ref.size());
    CHECK(hipMemcpy(ref.data(), dy, ref.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    // the hand-off: the stream waits on the value before the job is given, then takes the rows
    value++;
    CHECK(hipMemset(dcopy, 0xFF, got.size() * 4) == hipSuccess && hipDeviceSynchronize() == hipSuccess);
    CHECK(ops_wait(seqdev, value, nullptr) && ops_copy_rows(dcopy, ydev, didx, kRows, D, nullptr));
    std::this_thread::sleep_for(std::chrono::milliseconds(kDelayMs));
    CHECK(hipStreamQuery(nullptr) == hipErrorNotReady);  // still waiting
    const CpuLaneJob job = {type[0], type[2], D, FF, lim, x.data(), sc.data(), kRows, e, kNe, y, seq, value};
    CHECK(cpu_lane_submit(c, &job) && hipDeviceSynchronize() == hipSuccess && cpu_lane_wait(c));
    CHECK(std::memcmp(y, ref.data(), ref.size() * 4) == 0);  // the CPU's rows the GPU's bits
    CHECK(hipMemcpy(got.data(), dcopy, got.size() * 4, hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(std::memcmp(got.data(), ref.data(), ref.size() * 4) == 0);  // and the GPU's copy of them
    std::printf("  blk.%-2d types %2u / %2u, %u experts (%u rows), clamp %g: the GPU lane's bits; the stream waited "
                "for the post and took them\n", l, type[0], type[2], kNe, kRows, lim);
  }
  cpu_lane_destroy(c);
  CHECK(hipHostFree(y) == hipSuccess && hipHostFree(seq) == hipSuccess);
  for (uint8_t* p : dw) CHECK(hipFree(p) == hipSuccess);
  for (void* p : {(void*) dtab, (void*) dcnt, (void*) dxrow, (void*) didx, (void*) dx, (void*) dsc, (void*) dd,
                  (void*) dh, (void*) dhd, (void*) dy, (void*) dcopy, (void*) dq, (void*) dqh})
    CHECK(hipFree(p) == hipSuccess);
}

int main() {
  std::printf("[test_cpu_lane_gpu] start\n");
  const char* home = std::getenv("HOME");
  char path[512];
  std::snprintf(path, sizeof path, "%s/models/ds1017/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf",
                home ? home : "");
  int ndev = 0;
  if (access(path, R_OK) != 0 || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0 || !cpu_q8_ready()) {
    std::printf("[test_cpu_lane_gpu] skipped: no ds1017, no GPU or no AVX-512 VNNI\n");
    return 77;
  }
  char err[GGUF_ERR] = "";
  Gguf* g = gguf_create(path, err);
  CHECK(g);
  for (int l : {0, 10, 17}) layer(g, l);  // IQ1_M; IQ2_XXS; IQ2_XS gate and up, IQ1_M down
  gguf_destroy(g);
  std::printf("[test_cpu_lane_gpu] ok\n");
  return 0;
}
