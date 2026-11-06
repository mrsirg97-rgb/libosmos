// q4x_bench <first shard> <ids> [prompt] [decode]: Qwen3.8-Flash-Next's speed - a prompt (its first `prompt` ids,
// 512 by default, steps of 256) then `decode` greedy steps (128) of one token, each timed. prints prompt tok/s, decode
// tok/s (mean and its slowest step), and an FNV-1a of the decoded ids: a change that moves a bit of the answer shows
// there (DESIGN.md, "Qwen's speed").
// build: hipcc -O2 -Iinclude --offload-arch=gfx1151 tools/bench/q4x_bench.hip -Llib -losmos_hip -losmos
//        -lhsa-runtime64 -o bin/q4x_bench
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
#include "osmos/model/fnv.h"
#include "osmos/model/gguf.h"
#include "osmos/q4x/forward.h"
#include "osmos/q4x/weights.h"
#include "osmos/run/gpu.h"
}

constexpr uint32_t kStep = 256;

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  if (argc < 3) return std::fprintf(stderr, "usage: q4x_bench <first shard> <ids> [prompt] [decode]\n"), 2;
  if (!gpu_sleep_on_waits()) return std::fprintf(stderr, "q4x_bench: the GPU's wait mode not set\n"), 1;
  const uint32_t np = argc > 3 ? (uint32_t) std::atoi(argv[3]) : 512, nd = argc > 4 ? (uint32_t) std::atoi(argv[4]) : 128;
  std::vector<int32_t> ids;
  FILE* f = std::fopen(argv[2], "r");
  for (int32_t x; f && ids.size() < np && std::fscanf(f, "%d,", &x) == 1;) ids.push_back(x);
  if (f) std::fclose(f);
  if (ids.size() != np || !np || !nd) return std::fprintf(stderr, "q4x_bench: %zu ids, %u asked\n", ids.size(), np), 2;
  char gerr[GGUF_ERR] = "", werr[Q4X_ERR] = "", ferr[Q4X_FWD_ERR] = "";
  Gguf* g = gguf_create(argv[1], gerr);
  Q4xWeights* w = g ? q4x_weights_create(g, werr) : nullptr;
  if (!w) return std::fprintf(stderr, "q4x_bench: %s%s\n", gerr, werr), 1;
  Q4xForward* fw = q4x_forward_create(w, np + nd, np + nd < kStep ? np + nd : kStep, ferr);
  Q4xSeq* s = fw ? q4x_seq_create(fw, ferr) : nullptr;
  if (!s) return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
  const double t0 = now();
  for (uint32_t at = 0; at < np; at += kStep) {
    const uint32_t k = np - at < kStep ? np - at : kStep;
    if (!q4x_batch(fw, s, ids.data() + at, k, 0, ferr)) return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
  }
  int32_t y = -1;
  const uint32_t last = (np - 1) % kStep;
  if (!q4x_greedy(fw, last, 1, &y, ferr)) return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
  const double t1 = now();
  uint64_t h = FNV_SEED;
  double worst = 0;
  for (uint32_t i = 0; i < nd; i++) {
    const double a = now();
    h = fnv(&y, sizeof y, h);
    if (!q4x_batch(fw, s, &y, 1, 0, ferr) || !q4x_greedy(fw, 0, 1, &y, ferr))
      return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
    worst = now() - a > worst ? now() - a : worst;
  }
  const double t2 = now();
  std::printf("q4x_bench: prompt %u at %.1f tok/s; decode %u at %.2f tok/s (slowest step %.1f ms); ids %016llx\n", np,
              np / (t1 - t0), nd, nd / (t2 - t1), worst * 1e3, (unsigned long long) h);
  q4x_seq_destroy(s);
  q4x_forward_destroy(fw);
  q4x_weights_destroy(w);
  gguf_destroy(g);
  return 0;
}
