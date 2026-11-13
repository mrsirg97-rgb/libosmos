#define _GNU_SOURCE // O_DIRECT
// the pool and the resident set on the real model and the real GPU (test_model.h's): one layer's 384 experts (gate, up, down:
// 3 parts each, from gguf), read through reads.h (libevt) into hip_slots' GPU memory - the pool with a budget of a quarter
// of them, so pins push others out and their memory is written again for the next; the resident set all of them.
// the GPU sums every part in its memory, checked against the file's own bytes (read apart): a stale cache line of a
// memory's last unit would show. needs a gfx1151 and ~/models/ds1017; skipped (77) without them.
#include <hip/hip_runtime.h>

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <mutex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "test_model.h"
#include "osmos/mem/reads.h"
#include "osmos/mem/host_slots.h"
#include "osmos/mem/arena.h"
#include "osmos/mem/hip_slots.h"
#include "osmos/mem/pool.h"
#include "osmos/mem/resident.h"
#include "osmos/model/gguf.h"
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

// the binary's path: the pool's reads through reads.h (libevt: their order by need, their ends events this thread
// pumps into pool_ready); the pool's events counted - a failed read is a FAILED one
static Reads* g_reads = nullptr;
static int bad = 0, pins = 0;
static void on_event(const PoolEvent* ev, void*) {
  bad += ev->kind == POOL_EV_FAILED;
  pins += ev->kind == POOL_EV_PIN;
}
static int drain(Pool* p) {
  char rerr[READS_ERR];
  const int before = pins;
  Pool* const pools[READS_POOLS] = {p, nullptr};
  CHECK(reads_pump(g_reads, pools, true, rerr));
  return pins - before;
}

// a part's byte sum, from the file (not the memory under test)
static unsigned long long file_sum(const PoolRange& r) {
  std::vector<uint8_t> b(r.bytes);
  CHECK(pread(r.fd, b.data(), r.bytes, (off_t) r.offset) == (ssize_t) r.bytes);
  unsigned long long s = 0;
  for (uint8_t x : b) s += x;
  return s;
}

static unsigned long long gpu_sum(const void* dev, size_t n, unsigned long long* d_sum) {
  CHECK(hipMemset(d_sum, 0, sizeof *d_sum) == hipSuccess);
  sum_kernel<<<64, 256>>>((const uint8_t*) dev, n, d_sum);
  unsigned long long got = 0;
  CHECK(hipMemcpy(&got, d_sum, sizeof got, hipMemcpyDeviceToHost) == hipSuccess);
  return got;
}

int main() {
  std::printf("[test_pool_gpu] start\n");
  char path[512];
  int ndev = 0;
  if (test_model(path, sizeof path) == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::printf("[test_pool_gpu] skipped: no model (test_model.h) or no GPU\n");
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
  std::vector<unsigned long long> want((size_t) E * 3);
  for (int e = 0; e < E; e++)
    for (int k = 0; k < 3; k++) want[(size_t) e * 3 + k] = file_sum(units[e].part[k]);
  const uint64_t budget = unit_bytes * (uint64_t) E / 4;
  char err[POOL_ERR] = "";
  CHECK((g_reads = reads_create((uint32_t) E, 4, err)));
  Pool* p = pool_create(units.data(), units.size(), budget, hip_slots(), reads_fetcher(g_reads, 0), 0, on_event, nullptr, err);
  if (!p) std::fprintf(stderr, "pool_create: %s\n", err);
  CHECK(p);
  unsigned long long* d_sum = nullptr;
  CHECK(hipMalloc(&d_sum, sizeof *d_sum) == hipSuccess);
  // every expert twice, in two orders: pinned (pushing older ones out, their memory written again), summed on the
  // GPU, checked against the file, released
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
        while (pool_state(p, e) == POOL_READING) drain(p);
        CHECK(pool_dev(p, e, dev));
        r = POOL_OK;
      }
      pin_ms += now_ms() - t0;
      CHECK(r == POOL_OK && pool_bytes(p) <= budget);
      for (int k = 0; k < 3; k++)
        CHECK(gpu_sum(dev[k], (size_t) units[e].part[k].bytes, d_sum) == want[(size_t) e * 3 + k]);
      pool_release(p, e, 0);
      ok++;
    }
  std::printf("  %d experts of blk.3 (3 parts, %.2f MB each), budget a quarter (%.0f MB): %d pins, %d waited for a "
              "read, every GPU sum the file's; %.2f ms a pin on average, %.0f MB held at the end\n",
              E, (double) unit_bytes / 1e6, (double) budget / 1e6, ok, waited, pin_ms / ok, (double) pool_bytes(p) / 1e6);
  CHECK(bad == 0);
  reads_destroy(g_reads);
  pool_destroy(p);

  // the whole path, cold: the layer's experts evicted, all asked at once, read by 4 readers by need (reads.h;
  // DESIGN.md "the fetcher"), each read's end an event pumped into the pool on this thread - then a GPU sum each
  for (int k = 0; k < 3; k++) posix_fadvise(fds[t[k]->file], (off_t) t[k]->offset, (off_t) t[k]->bytes, POSIX_FADV_DONTNEED);
  CHECK((g_reads = reads_create((uint32_t) E, 4, err)));
  p = pool_create(units.data(), units.size(), unit_bytes * (uint64_t) E + (64u << 20), hip_slots(),
                  reads_fetcher(g_reads, 0), 0, on_event, nullptr, err);
  CHECK(p);
  const double t0 = now_ms();
  for (int e = 0; e < E; e++) {
    const void* dev[POOL_PARTS];
    CHECK(pool_pin(p, (uint32_t) e, dev) == POOL_WAIT);
  }
  for (int pinned = 0; pinned < E;) pinned += drain(p);
  for (int e = 0; e < E; e++) CHECK(pool_state(p, (uint32_t) e) == POOL_PINNED);
  const double cold_ms = now_ms() - t0;
  for (int e = 0; e < E; e += 37) {
    const void* dev[POOL_PARTS];
    CHECK(pool_dev(p, (uint32_t) e, dev));
    for (int k = 0; k < 3; k++)
      CHECK(gpu_sum(dev[k], (size_t) units[e].part[k].bytes, d_sum) == want[(size_t) e * 3 + k]);
  }
  std::printf("  cold, the whole path: %d experts (%.2f GB) read through reads.h into GPU memory in %.0f ms = %.2f GB/s, %d "
              "reads failed; the GPU's sums the file's\n", E, (double) unit_bytes * E / 1e9, cold_ms,
              (double) unit_bytes * E / (cold_ms * 1e-3) / 1e9, bad);
  CHECK(bad == 0);
  reads_destroy(g_reads);
  pool_destroy(p);

  // the binary's path since 0.3.0: the same, read O_DIRECT (each part its 4 KiB-aligned range) straight into registered
  // host pages (host_slots) - the GPU's sums of every part at its pointer the file's, and the GB/s
  {
    std::vector<int> dfds(g->n_files);
    for (int i = 0; i < g->n_files; i++) CHECK((dfds[i] = open(g->paths[i], O_RDONLY | O_DIRECT)) >= 0);
    std::vector<PoolUnit> du = units;
    for (int e = 0; e < E; e++)
      for (int k = 0; k < 3; k++) du[e].part[k].fd = dfds[t[k]->file];
    const uint64_t room = unit_bytes * (uint64_t) E + (64u << 20);
    HostSlots* hs = host_slots_create(room + (64u << 20), err);
    if (!hs) std::fprintf(stderr, "host_slots_create: %s\n", err);
    CHECK(hs);
    CHECK((g_reads = reads_create((uint32_t) E, 4, err)));
    p = pool_create(du.data(), du.size(), room, host_slots(hs), reads_fetcher(g_reads, 0), ARENA_GRAIN, on_event, nullptr, err);
    if (!p) std::fprintf(stderr, "pool_create: %s\n", err);
    CHECK(p);
    const double d0 = now_ms();
    for (int e = 0; e < E; e++) {
      const void* dev[POOL_PARTS];
      CHECK(pool_pin(p, (uint32_t) e, dev) == POOL_WAIT);
    }
    for (int pinned = 0; pinned < E;) pinned += drain(p);
    const double direct_ms = now_ms() - d0;
    for (int e = 0; e < E; e++) {
      const void* dev[POOL_PARTS];
      CHECK(pool_state(p, (uint32_t) e) == POOL_PINNED && pool_dev(p, (uint32_t) e, dev));
      for (int k = 0; k < 3; k++)
        CHECK(gpu_sum(dev[k], (size_t) units[e].part[k].bytes, d_sum) == want[(size_t) e * 3 + k]);
    }
    std::printf("  direct: %d experts (%.2f GB) read O_DIRECT into registered host pages in %.0f ms = %.2f GB/s, %d "
                "reads failed; every part's GPU sum the file's\n", E, (double) unit_bytes * E / 1e9, direct_ms,
                (double) unit_bytes * E / (direct_ms * 1e-3) / 1e9, bad);
    CHECK(bad == 0);
    reads_destroy(g_reads);
    pool_destroy(p);
    host_slots_destroy(hs);
    for (int fd : dfds) close(fd);
  }

  // the resident set: the layer's experts, cold, read once into GPU memory; every part's GPU sum the file's, and the
  // file's pages dropped from the page cache (the copy is the one kept)
  for (int k = 0; k < 3; k++) posix_fadvise(fds[t[k]->file], (off_t) t[k]->offset, (off_t) t[k]->bytes, POSIX_FADV_DONTNEED);
  double r0 = now_ms();
  Resident* rs = resident_create(units.data(), units.size(), hip_slots(), 4, err);
  if (!rs) std::fprintf(stderr, "resident_create: %s\n", err);
  CHECK(rs);
  const double res_ms = now_ms() - r0;
  for (int e = 0; e < E; e++) {
    const void* dev[POOL_PARTS];
    CHECK(resident_dev(rs, (size_t) e, dev));
    for (int k = 0; k < 3; k++)
      CHECK(gpu_sum(dev[k], (size_t) units[e].part[k].bytes, d_sum) == want[(size_t) e * 3 + k]);
  }
  std::printf("  resident: %d experts (%.2f GB) read cold into %.2f GB of GPU memory in %.0f ms = %.2f GB/s; every "
              "GPU sum the file's\n", E, (double) unit_bytes * E / 1e9, (double) resident_bytes(rs) / 1e9, res_ms,
              (double) unit_bytes * E / (res_ms * 1e-3) / 1e9);
  resident_destroy(rs);
  CHECK(hipFree(d_sum) == hipSuccess);
  for (int fd : fds) close(fd);
  gguf_destroy(g);
  std::printf("[test_pool_gpu] ok\n");
  return 0;
}
