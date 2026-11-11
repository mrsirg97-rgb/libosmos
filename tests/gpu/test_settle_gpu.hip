// GpuWait (osmos/run/gpu.h; DESIGN.md, "0.6.2: Qwen's waits"): a wait asleep to a mark, polling the tail - the GPU's
// idle after its work, by the device's own clock. a trial: a busy kernel (the body), a busy kernel (the tail), a
// kernel that stamps the wall clock; the wait; a second stamp launched the moment the wait returns - the stamps'
// distance is the time the GPU had nothing queued. three waits in one process, gpu_sleep_on_waits first as the
// binaries: the stream's sleep (0.6.1's), an event polled (the floor), GpuWait's with its mark between body and tail.
// held to: the sleep idles longer than the poll (the premise: a runtime that wakes at once makes GpuWait moot - this
// fails and says so); GpuWait's idle under the midpoint of the two; its thread's CPU under the tails' time (it slept
// the bodies - a wait that polls everything is the mutant). needs a GPU; skipped (77) without one.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sys/resource.h>
#include <vector>

extern "C" {
#include "osmos/run/gpu.h"
}

#define CHECK(x) \
  do { \
    if (!(x)) std::printf("CHECK failed: %s (%s:%d)\n", #x, __FILE__, __LINE__), std::exit(1); \
  } while (0)

// the tail: the shortest one the forward marks - an MTP draft depth's shortlist head, 137 MB at the dense's ~200 GB/s
constexpr uint32_t kTailUs = 700;
// the body: several tails, so a wait that polls it all spends several times the tails' time in CPU
constexpr uint32_t kBodyUs = 8 * kTailUs;
constexpr int kTrials = 64, kWarm = 4;  // (the first trials load the kernels' code)

enum class Wait { kSleep, kPoll, kMarked };

__global__ void busy(uint64_t ticks) {  // wall_clock64's ticks at the device's rate
  const uint64_t t0 = wall_clock64();
  while (wall_clock64() - t0 < ticks) __builtin_amdgcn_s_sleep(8);
}

__global__ void stamp(uint64_t* at) { *at = wall_clock64(); }

static double thread_cpu() {
  rusage u;
  getrusage(RUSAGE_THREAD, &u);
  return u.ru_utime.tv_sec + u.ru_stime.tv_sec + (u.ru_utime.tv_usec + u.ru_stime.tv_usec) / 1e6;
}

struct Measured {
  double idle_us;  // the median trial's
  double cpu_s;    // the thread's, inside the waits
};

static Measured run(Wait how, GpuWait* w, hipEvent_t ev, uint64_t* at, int khz) {
  const uint64_t body = (uint64_t) kBodyUs * khz / 1000, tail = (uint64_t) kTailUs * khz / 1000;
  std::vector<double> idle;
  double cpu = 0;
  for (int i = 0; i < kWarm + kTrials; i++) {
    busy<<<1, 1>>>(body);
    if (how == Wait::kMarked) CHECK(gpu_wait_mark(w));
    busy<<<1, 1>>>(tail);
    stamp<<<1, 1>>>(at);
    const double c0 = thread_cpu();
    if (how == Wait::kSleep) CHECK(hipStreamSynchronize(NULL) == hipSuccess);
    if (how == Wait::kMarked) CHECK(gpu_wait_settle(w));
    if (how == Wait::kPoll) {
      CHECK(hipEventRecord(ev, NULL) == hipSuccess);
      hipError_t e;
      while ((e = hipEventQuery(ev)) == hipErrorNotReady) {
      }
      CHECK(e == hipSuccess);
    }
    const double c1 = thread_cpu();
    stamp<<<1, 1>>>(at + 1);
    CHECK(hipStreamSynchronize(NULL) == hipSuccess && at[1] > at[0]);
    if (i < kWarm) continue;
    idle.push_back((double) (at[1] - at[0]) * 1000.0 / khz);
    cpu += c1 - c0;
  }
  std::sort(idle.begin(), idle.end());
  return {idle[idle.size() / 2], cpu};
}

int main() {
  std::printf("[test_settle_gpu] start\n");
  int ndev = 0, khz = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || !ndev) return std::printf("[test_settle_gpu] skipped: no GPU\n"), 77;
  CHECK(gpu_sleep_on_waits());  // before the first kernel, as osmos-serve and q4x_bench
  CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0) == hipSuccess && khz > 0);
  uint64_t* at = nullptr;  // mapped: the stamps read with no copy
  CHECK(hipHostMalloc((void**) &at, 2 * sizeof *at, hipHostMallocMapped | hipHostMallocCoherent) == hipSuccess);
  uint64_t* at_dev = nullptr;
  CHECK(hipHostGetDevicePointer((void**) &at_dev, at, 0) == hipSuccess);
  hipEvent_t ev;
  CHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming) == hipSuccess);
  GpuWait* w = gpu_wait_create();
  CHECK(w);

  const Measured sleep = run(Wait::kSleep, w, ev, at_dev, khz), poll = run(Wait::kPoll, w, ev, at_dev, khz);
  const Measured marked = run(Wait::kMarked, w, ev, at_dev, khz);
  const double tails_s = kTrials * kTailUs / 1e6;
  std::printf("  the GPU idle after its work (median of %d): asleep %.0f us, polled %.0f us, GpuWait %.0f us\n", kTrials,
              sleep.idle_us, poll.idle_us, marked.idle_us);
  std::printf("  the thread's CPU in the waits: asleep %.1f ms, polled %.1f ms, GpuWait %.1f ms (the tails %.1f ms)\n",
              sleep.cpu_s * 1e3, poll.cpu_s * 1e3, marked.cpu_s * 1e3, tails_s * 1e3);
  CHECK(sleep.idle_us > poll.idle_us);                          // the premise: a sleep's wake costs the GPU idle time
  CHECK(marked.idle_us < (sleep.idle_us + poll.idle_us) / 2);   // the wake under the tail
  CHECK(marked.cpu_s < tails_s);                                // asleep through the bodies

  // no mark: the sleep alone, to the end - and the work done when it returns
  at[0] = 0;
  busy<<<1, 1>>>((uint64_t) kBodyUs * khz / 1000);
  stamp<<<1, 1>>>(at_dev);
  CHECK(gpu_wait_settle(w) && at[0] != 0);

  gpu_wait_destroy(w);
  CHECK(hipEventDestroy(ev) == hipSuccess && hipHostFree(at) == hipSuccess);
  std::printf("[test_settle_gpu] ok\n");
  return 0;
}
