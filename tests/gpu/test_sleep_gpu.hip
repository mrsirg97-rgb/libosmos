// the forward's waits for the GPU sleep (osmos/run/gpu.h): a kernel busy for ~300 ms, waited for as the forward waits
// (hipStreamSynchronize, hipEventSynchronize), the thread's CPU time against the wall - spinning burns ~all of it, a
// sleep a few percent; set where a binary sets it - before its first kernel (after one it does not take: measured).
// NO_SLEEP=1: the wait as before it, which must spin (the test's own check that it can tell).
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sys/resource.h>

extern "C" {
#include "osmos/run/gpu.h"
}

#define CHECK(x) \
  do { \
    if (!(x)) std::printf("CHECK failed: %s (%s:%d)\n", #x, __FILE__, __LINE__), std::exit(1); \
  } while (0)

__global__ void busy(uint32_t* out, uint64_t ticks) {  // wall_clock64's ticks at the device's rate
  const uint64_t t0 = wall_clock64();
  while (wall_clock64() - t0 < ticks) __builtin_amdgcn_s_sleep(127);
  *out = 7;
}

static double thread_cpu() {
  rusage u;
  getrusage(RUSAGE_THREAD, &u);
  return u.ru_utime.tv_sec + u.ru_stime.tv_sec + (u.ru_utime.tv_usec + u.ru_stime.tv_usec) / 1e6;
}

/* a 300 ms kernel waited for by `how`: the thread's CPU share of the wall */
static double share(uint32_t* out, uint64_t ticks, bool event) {
  hipEvent_t ev;
  CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming) == hipSuccess);
  *out = 0;
  const double c0 = thread_cpu();
  const auto t0 = std::chrono::steady_clock::now();
  busy<<<1, 1>>>(out, ticks);
  CHECK(event ? hipEventRecord(ev, NULL) == hipSuccess && hipEventSynchronize(ev) == hipSuccess
              : hipStreamSynchronize(NULL) == hipSuccess);
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), cpu = thread_cpu() - c0;
  CHECK(*out == 7 && wall > 0.25);
  CHECK(hipEventDestroy(ev) == hipSuccess);
  std::printf("  a %.0f ms kernel, waited on %s: the thread's CPU %.1f ms (%.0f%% of the wall)\n", wall * 1e3,
              event ? "an event" : "the stream", cpu * 1e3, 100 * cpu / wall);
  return cpu / wall;
}

int main() {
  std::printf("[test_sleep_gpu] start\n");
  int ndev = 0, khz = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) return std::printf("[test_sleep_gpu] skipped: no GPU\n"), 77;
  CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0) == hipSuccess && khz > 0);
  uint32_t* out = nullptr;
  void* d = nullptr;
  CHECK(hipHostMalloc((void**) &out, 4, 0) == hipSuccess && hipMalloc(&d, 1 << 20) == hipSuccess);
  if (!std::getenv("NO_SLEEP")) CHECK(gpu_sleep_on_waits()); /* after allocations, before any kernel: a binary's start */
  const uint64_t ticks = (uint64_t) khz * 300;
  busy<<<1, 1>>>(out, 1); /* the first launch's own setup out of the measured ones */
  CHECK(hipDeviceSynchronize() == hipSuccess);
  CHECK(share(out, ticks, true) < 0.1);
  CHECK(share(out, ticks, false) < 0.1);
  CHECK(hipFree(d) == hipSuccess && hipHostFree(out) == hipSuccess);
  std::printf("[test_sleep_gpu] ok\n");
  return 0;
}
