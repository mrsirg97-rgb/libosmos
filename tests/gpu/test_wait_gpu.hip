// the forward's waits for the GPU (src/run/hip_wait.c): a decode-sized step's spin (its waits short and many, a wake
// ~0.2 ms), a prompt's step's sleep - the forward's thread's CPU time against the wall of each, on the real model.
// gpu_sleep_on_waits first, as the binaries do. needs the model (test_model.h); skipped (77) without it.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sys/resource.h>
#include <vector>

#include "test_model.h"

extern "C" {
#include "osmos/run/forward.h"
#include "osmos/run/gpu.h"
}

#define CHECK(x) \
  do { \
    if (!(x)) std::printf("CHECK failed: %s (%s:%d)\n", #x, __FILE__, __LINE__), std::exit(1); \
  } while (0)

constexpr uint64_t kBudget = 72ull << 30;
constexpr uint32_t kCtx = 4096, kPrompt = 2048, kDecode = 32;

static double thread_cpu() {
  rusage u;
  getrusage(RUSAGE_THREAD, &u);
  return u.ru_utime.tv_sec + u.ru_stime.tv_sec + (u.ru_utime.tv_usec + u.ru_stime.tv_usec) / 1e6;
}

int main() {
  std::printf("[test_wait_gpu] start\n");
  char path[512];
  const TestModel which = test_model(path, sizeof path);
  int ndev = 0;
  if (which == TestModel::kNone || hipGetDeviceCount(&ndev) != hipSuccess || !ndev)
    return std::printf("[test_wait_gpu] skipped: no model (test_model.h) or no GPU\n"), 77;
  CHECK(gpu_sleep_on_waits());  // before the first kernel, as osmos-serve and dbench
  char ferr[FORWARD_ERR] = "";
  Gguf *g = nullptr, *eg = nullptr;
  Weights* w = test_weights(path, which, &g, &eg);
  CHECK(w);
  Forward* f = forward_create(w, w->geo.n_layers, kCtx, kBudget, ferr);
  if (!f) std::printf("forward_create: %s\n", ferr);
  CHECK(f);
  Seq* s = seq_create(f, ferr);
  CHECK(s && seq_prompt(s, kPrompt));
  std::vector<int32_t> ids(kPrompt);
  for (uint32_t i = 0; i < kPrompt; i++) ids[i] = (int32_t) (1000 + (i * 7919) % 60000);
  std::vector<Seq*> ss(kPrompt, s);
  // the prompt: one step of kPrompt tokens (its routing pairs past a layer's experts) - slept on
  double c0 = thread_cpu();
  auto t0 = std::chrono::steady_clock::now();
  CHECK(forward_batch(f, ss.data(), ids.data(), kPrompt, ferr));
  const double pw = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), pc = thread_cpu() - c0;
  // decode: a token a step - spun on
  int32_t tok = 1000;
  c0 = thread_cpu();
  t0 = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < kDecode; i++) {
    CHECK(forward_step(f, s, tok, ferr));
    tok = 1000 + (int32_t) i;
  }
  const double dw = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), dc = thread_cpu() - c0;
  std::printf("  a %u-token prompt step: %.0f ms, the forward's thread's CPU %.0f%% of it\n", kPrompt, pw * 1e3,
              100 * pc / pw);
  std::printf("  %u decode steps: %.0f ms, the thread's CPU %.0f%% of it\n", kDecode, dw * 1e3, 100 * dc / dw);
  CHECK(dc > 0.5 * dw);  // spun: busy most of it (the reads' waits sleep on the engine: never all of it)
  CHECK(pc < 0.25 * pw); // slept: the core freed while the GPU reads the prompt
  seq_destroy(s);
  forward_destroy(f);
  weights_destroy(w);
  gguf_destroy(eg);
  gguf_destroy(g);
  std::printf("[test_wait_gpu] ok\n");
  return 0;
}
