// q4x_bench <first shard> <ids> [prompt] [decode] [mtp head]: Qwen3.8-Flash-Next's speed - a prompt (its first
// `prompt` ids, 512 by default, steps of 256) then `decode` greedy tokens (128): a step each, or with the MTP head
// drafted rounds (the server's: drafts after y under the engine's floors, [y, drafts] one step, the agreeing prefix
// kept, the rest rolled back, the kept seeded). prints prompt tok/s, decode tok/s (and its slowest step or round, the
// acceptance), and an FNV-1a of the decoded ids: plain and drafted the same (DESIGN.md, "Qwen's speed").
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
#include "osmos/run/engine.h"
#include "osmos/run/gpu.h"
}

constexpr uint32_t kStepDefault = 256;  // a prompt's step (Q4X_BENCH_STEP: osmos-serve's --max-tokens, 2048)

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// seconds since the epoch: tools/bench/energy.py's windows (matched to tools/power.py's samples)
static double epoch() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  if (argc < 3) return std::fprintf(stderr, "usage: q4x_bench <first shard> <ids> [prompt] [decode] [mtp head]\n"), 2;
  if (!gpu_sleep_on_waits()) return std::fprintf(stderr, "q4x_bench: the GPU's wait mode not set\n"), 1;
  const uint32_t kStep = getenv("Q4X_BENCH_STEP") ? (uint32_t) atoi(getenv("Q4X_BENCH_STEP")) : kStepDefault;
  const uint32_t np = argc > 3 ? (uint32_t) std::atoi(argv[3]) : 512, nd = argc > 4 ? (uint32_t) std::atoi(argv[4]) : 128;
  std::vector<int32_t> ids;
  FILE* f = std::fopen(argv[2], "r");
  for (int32_t x; f && ids.size() < np && std::fscanf(f, "%d,", &x) == 1;) ids.push_back(x);
  if (f) std::fclose(f);
  if (ids.size() != np || !np || !nd) return std::fprintf(stderr, "q4x_bench: %zu ids, %u asked\n", ids.size(), np), 2;
  char gerr[GGUF_ERR] = "", werr[Q4X_ERR] = "", ferr[Q4X_FWD_ERR] = "";
  Gguf* g = gguf_create(argv[1], gerr);
  /* OVERLAY: a GGUF whose tensors stand in for the model's of their names (an abliteration: q4x_weights_create) */
  Gguf* ov = getenv("OVERLAY") ? gguf_create(getenv("OVERLAY"), gerr) : nullptr;
  if (getenv("OVERLAY") && !ov) return std::fprintf(stderr, "q4x_bench: the overlay: %s\n", gerr), 1;
  Q4xWeights* w = g ? q4x_weights_create(g, ov, werr) : nullptr;
  if (!w) return std::fprintf(stderr, "q4x_bench: %s%s\n", gerr, werr), 1;
  Gguf* mg = argc > 5 ? gguf_create(argv[5], gerr) : nullptr;
  Q4xMtp* mtp = mg ? q4x_mtp_create(mg, w, werr) : nullptr;
  if (argc > 5 && !mtp) return std::fprintf(stderr, "q4x_bench: %s%s\n", gerr, werr), 1;
  /* Q4X_BENCH_CTX: the forward's context (past the native 262,144: YaRN, q4x_rope), else the run's */
  const uint32_t run = np + nd + OSMOS_MAX_BLOCK,
                 ctx = getenv("Q4X_BENCH_CTX") ? (uint32_t) atoi(getenv("Q4X_BENCH_CTX")) : run;
  if (ctx < run) return std::fprintf(stderr, "q4x_bench: Q4X_BENCH_CTX below the run's %u\n", run), 2;
  Q4xForward* fw = q4x_forward_create(w, mtp, ctx, ctx < kStep ? ctx : kStep, ferr);
  Q4xSeq* s = fw ? q4x_seq_create(fw, ferr) : nullptr;
  if (!s) return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
  const double t0 = now(), e0 = epoch();
  for (uint32_t at = 0; at < np; at += kStep) {
    const uint32_t k = np - at < kStep ? np - at : kStep;
    if (!q4x_batch(fw, s, ids.data() + at, k, 0, ferr) || (mtp && !q4x_mtp_seed(fw, s, (int32_t) at, (int32_t) (at + k), ferr)))
      return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
  }
  int32_t y = -1;
  const uint32_t last = (np - 1) % kStep;
  if (!q4x_greedy(fw, last, 1, &y, ferr))
    return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
  const double e1 = epoch();
  if (getenv("Q4X_STEP_SWEEP")) /* a step's time by its tokens: 1 .. 8, each step rolled back */
    for (uint32_t n = 1; n <= OSMOS_MAX_BLOCK; n++) {
      const int reps = 12;
      const double a = now();
      for (int r = 0; r < reps; r++)
        if (!q4x_batch(fw, s, ids.data(), n, 0, ferr) || hipDeviceSynchronize() != hipSuccess || !q4x_seq_rollback(s, n))
          return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
      std::printf("  a step of %u tokens: %.1f ms\n", n, (now() - a) / reps * 1e3);
    }
  /* the drafts' floors: the engine's (engine.h), or Q4X_BENCH_CONF / Q4X_BENCH_CUM - a sweep's */
  const float conf_min = getenv("Q4X_BENCH_CONF") ? (float) atof(getenv("Q4X_BENCH_CONF")) : OSMOS_DRAFT_CONF_MIN;
  const float cum_min = getenv("Q4X_BENCH_CUM") ? (float) atof(getenv("Q4X_BENCH_CUM")) : OSMOS_DRAFT_CUM_MIN;
  const uint32_t block = getenv("Q4X_BENCH_BLOCK") ? (uint32_t) atoi(getenv("Q4X_BENCH_BLOCK")) : Q4X_MTP_BLOCK;
  if (!block || block > OSMOS_MAX_BLOCK) return std::fprintf(stderr, "q4x_bench: Q4X_BENCH_BLOCK 1 .. 8\n"), 2;
  const double t1 = now(), e2 = epoch();
  uint64_t h = FNV_SEED;
  double worst = 0;
  uint32_t out = 0, rounds = 0, drafted = 0, accepted = 0;
  while (out < nd) {
    const double a = now();
    int32_t toks[OSMOS_MAX_BLOCK + 1] = {y}, picks[OSMOS_MAX_BLOCK + 1];
    float conf[OSMOS_MAX_BLOCK];
    const int k = mtp ? q4x_mtp_draft(fw, s, y, block, conf_min, cum_min, toks + 1, conf, ferr) : 0;
    const int32_t pos = q4x_seq_position(s);
    if (k < 0 || !q4x_batch(fw, s, toks, (uint32_t) k + 1, 0, ferr) || !q4x_greedy(fw, 0, (uint32_t) k + 1, picks, ferr))
      return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
    int keep = 1;
    while (keep <= k && toks[keep] == picks[keep - 1]) keep++;
    for (int i = 0; i < keep && out < nd; i++, out++) h = fnv(&toks[i], sizeof toks[i], h);
    y = picks[keep - 1];
    if (mtp && (!q4x_seq_rollback(s, (uint32_t) (k + 1 - keep)) || !q4x_mtp_seed(fw, s, pos, pos + keep, ferr)))
      return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
    rounds++, drafted += (uint32_t) k, accepted += (uint32_t) keep - 1;
    worst = now() - a > worst ? now() - a : worst;
  }
  const double t2 = now(), e3 = epoch();
  std::printf("window prompt %.3f %.3f %u\nwindow decode %.3f %.3f %u\n", e0, e1, np, e2, e3, nd);
  std::printf("q4x_bench: prompt %u at %.1f tok/s; decode %u at %.2f tok/s (slowest %s %.1f ms", np, np / (t1 - t0), nd,
              nd / (t2 - t1), mtp ? "round" : "step", worst * 1e3);
  if (mtp)
    std::printf("; %u rounds, %.0f%% of %u drafts accepted, %.2f tokens a round", rounds,
                drafted ? 100.0 * accepted / drafted : 0.0, drafted, (double) nd / rounds);
  std::printf("); ids %016llx\n", (unsigned long long) h);
  q4x_seq_destroy(s);
  q4x_forward_destroy(fw);
  q4x_mtp_destroy(mtp);
  gguf_destroy(mg);
  q4x_weights_destroy(w);
  gguf_destroy(g);
  return 0;
}
