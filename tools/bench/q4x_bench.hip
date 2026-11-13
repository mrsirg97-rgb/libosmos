// q4x_bench <first shard> <ids> [prompt] [decode] [drafter]: Qwen3.8-Flash-Next's speed - a prompt (its first
// `prompt` ids, 512 by default, steps of 256) then `decode` greedy tokens (128): a step each, or with a drafter (the
// MTP head, or DFlash2: its GGUF's architecture dflash2) drafted rounds (the server's: drafts after y under the engine's floors, [y, drafts] one step, the agreeing prefix
// kept, the rest rolled back, the kept seeded). prints prompt tok/s, decode tok/s (and its slowest step or round, the
// acceptance), and an FNV-1a of the decoded ids: plain and drafted the same (DESIGN.md, "Qwen's speed").
// build: hipcc -O2 -Iinclude --offload-arch=gfx1151 tools/bench/q4x_bench.hip -Llib -losmos_hip -losmos
//        -lhsa-runtime64 -o bin/q4x_bench
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/model/fnv.h"
#include "osmos/model/gguf.h"
#include "osmos/q4x/dflash.h"
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
  if (argc < 3) return std::fprintf(stderr, "usage: q4x_bench <first shard> <ids> [prompt] [decode] [drafter]\n"), 2;
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
  const GgufKv* arch = mg ? gguf_kv(mg, "general.architecture") : nullptr;
  const bool is_df = arch && arch->type == GGUF_STRING && !std::strcmp(arch->v.s, "dflash2");
  Q4xMtp* mtp = mg && !is_df ? q4x_mtp_create(mg, w, werr) : nullptr;
  Q4xDflash* df = mg && is_df ? q4x_dflash_create(mg, w, werr) : nullptr;
  const bool drafting = mtp || df;
  if (argc > 5 && !drafting) return std::fprintf(stderr, "q4x_bench: %s%s\n", gerr, werr), 1;
  /* Q4X_BENCH_CTX: the forward's context (past the native 262,144: YaRN, q4x_rope), else the run's */
  const uint32_t run = np + nd + OSMOS_MAX_BLOCK,
                 ctx = getenv("Q4X_BENCH_CTX") ? (uint32_t) atoi(getenv("Q4X_BENCH_CTX")) : run;
  if (ctx < run) return std::fprintf(stderr, "q4x_bench: Q4X_BENCH_CTX below the run's %u\n", run), 2;
  const Q4xOptions qo = {ctx, ctx < kStep ? ctx : kStep, getenv("Q4X_BENCH_STREAM") != nullptr, nullptr, 0, df};
  Q4xForward* fw = q4x_forward_create_with(w, mtp, &qo, ferr);
  // 0.7.0: the NPU beside the GPU (q4x_forward_npu): NPU=<its kernels' dir>, NPU_SHARE (NPU=1 builds)
  if (fw && std::getenv("NPU") &&
      !q4x_forward_npu(fw, std::getenv("NPU"), std::getenv("NPU_SHARE") ? std::strtof(std::getenv("NPU_SHARE"), nullptr) : OSMOS_NPU_SHARE_QWEN, ferr))
    q4x_forward_destroy(fw), fw = nullptr;
  Q4xSeq* s = fw ? q4x_seq_create(fw, ferr) : nullptr;
  if (!s) return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
  // the prompt's end told, as osmos-serve tells it (oseq_prompt): its last row's logits alone. Q4X_BENCH_ALL_ROWS: not told
  if (!getenv("Q4X_BENCH_ALL_ROWS") && !q4x_seq_prompt(s, np)) return std::fprintf(stderr, "q4x_bench: the prompt's end refused\n"), 1;
  const double t0 = now(), e0 = epoch();
  for (uint32_t at = 0; at < np; at += kStep) {
    const uint32_t k = np - at < kStep ? np - at : kStep;
    if (!q4x_batch(fw, s, ids.data() + at, k, 0, ferr) || (mtp && !q4x_mtp_seed(fw, s, (int32_t) at, (int32_t) (at + k), ferr)) ||
        (df && !q4x_dflash_seed(fw, s, (int32_t) at, (int32_t) (at + k), ferr)))
      return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
  }
  int32_t y = -1;
  const uint32_t last = (np - 1) % kStep;
  if (!q4x_greedy(fw, last, 1, &y, ferr))
    return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
  const double tp = now(), e1 = epoch(); /* the prompt's end: its time and its energy's window the same (0.7.1: the time
                                           ran on through the step sweep) */
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
  const uint32_t most = df ? (uint32_t) df->geo.block - 1 : Q4X_MTP_BLOCK;
  const uint32_t block = getenv("Q4X_BENCH_BLOCK") ? (uint32_t) atoi(getenv("Q4X_BENCH_BLOCK")) : most;
  if (!block || block > OSMOS_MAX_BLOCK) return std::fprintf(stderr, "q4x_bench: Q4X_BENCH_BLOCK 1 .. 8\n"), 2;
  const double t1 = now(), e2 = epoch();
  uint64_t h = FNV_SEED;
  std::vector<int32_t> decoded;  // Q4X_BENCH_IDS: the decoded ids written there (two runs' tokens compared)
  double worst = 0;
  uint32_t out = 0, rounds = 0, drafted = 0, accepted = 0;
  const bool trace = getenv("Q4X_BENCH_TRACE");  // each round's drafts and their probabilities, exact (%a): two runs diffed
  while (out < nd) {
    const double a = now();
    int32_t toks[OSMOS_MAX_BLOCK + 1] = {y}, picks[OSMOS_MAX_BLOCK + 1];
    float conf[OSMOS_MAX_BLOCK];
    const int k = mtp  ? q4x_mtp_draft(fw, s, y, block, conf_min, cum_min, toks + 1, conf, ferr)
                  : df ? q4x_dflash_draft(fw, s, y, block, conf_min, cum_min, toks + 1, conf, ferr)
                       : 0;
    const int32_t pos = q4x_seq_position(s);
    if (k < 0 || !q4x_batch(fw, s, toks, (uint32_t) k + 1, 0, ferr) || !q4x_greedy(fw, 0, (uint32_t) k + 1, picks, ferr))
      return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
    int keep = 1;
    while (keep <= k && toks[keep] == picks[keep - 1]) keep++;
    if (trace) {
      static std::vector<float> lg;  // the verify's logits rows, hashed: the main model's bits a round
      const uint32_t V = (uint32_t) w->geo.n_vocab;
      lg.resize((size_t) (k + 1) * V);
      if (hipDeviceSynchronize() != hipSuccess ||
          hipMemcpy(lg.data(), q4x_logits(fw), lg.size() * 4, hipMemcpyDeviceToHost) != hipSuccess)
        return std::fprintf(stderr, "q4x_bench: the trace's logits copy failed\n"), 1;
      // row 0's input is the token already taken (every run's the same): its bits must not depend on the drafts
      std::printf("round %u row0 %016llx logits %016llx\n", rounds, (unsigned long long) fnv(lg.data(), (size_t) V * 4, FNV_SEED),
                  (unsigned long long) fnv(lg.data(), lg.size() * 4, FNV_SEED));
      std::printf("round %u at %d: %d drafts, %d kept;", rounds, pos, k, keep - 1);
      for (int i = 0; i < k; i++) std::printf(" %d %a", toks[i + 1], conf[i]);
      std::printf("\n");
    }
    for (int i = 0; i < keep && out < nd; i++, out++) h = fnv(&toks[i], sizeof toks[i], h), decoded.push_back(toks[i]);
    y = picks[keep - 1];
    if (drafting && !q4x_seq_rollback(s, (uint32_t) (k + 1 - keep)))
      return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
    if ((mtp && !q4x_mtp_seed(fw, s, pos, pos + keep, ferr)) || (df && !q4x_dflash_seed(fw, s, pos, pos + keep, ferr)))
      return std::fprintf(stderr, "q4x_bench: %s\n", ferr), 1;
    rounds++, drafted += (uint32_t) k, accepted += (uint32_t) keep - 1;
    worst = now() - a > worst ? now() - a : worst;
  }
  const double t2 = now(), e3 = epoch();
  std::printf("window prompt %.3f %.3f %u\nwindow decode %.3f %.3f %u\n", e0, e1, np, e2, e3, nd);
  std::printf("q4x_bench: prompt %u at %.1f tok/s; decode %u at %.2f tok/s (slowest %s %.1f ms", np, np / (tp - t0), nd,
              nd / (t2 - t1), drafting ? "round" : "step", worst * 1e3);
  if (drafting)
    std::printf("; %u rounds, %.0f%% of %u drafts accepted, %.2f tokens a round", rounds,
                drafted ? 100.0 * accepted / drafted : 0.0, drafted, (double) nd / rounds);
  std::printf("); ids %016llx\n", (unsigned long long) h);
  if (std::getenv("NPU")) {
    uint64_t nc, np_, nw, nt;
    double us;
    q4x_forward_npu_stats(fw, &nc, &np_, &us, &nw, &nt);
    std::printf("the NPU: %llu calls, %llu parts (%llu of %llu wanted whole), %.0f ms the host waited on them\n",
                (unsigned long long) nc, (unsigned long long) np_, (unsigned long long) nt, (unsigned long long) nw, us / 1e3);
  }
  if (const char* path = getenv("Q4X_BENCH_IDS")) {
    FILE* o = std::fopen(path, "w");
    for (size_t i = 0; o && i < decoded.size(); i++) std::fprintf(o, "%d%s", decoded[i], i + 1 < decoded.size() ? "," : "\n");
    if (!o || std::fclose(o)) return std::fprintf(stderr, "q4x_bench: %s not written\n", path), 1;
  }
  q4x_seq_destroy(s);
  q4x_forward_destroy(fw);
  q4x_mtp_destroy(mtp);
  q4x_dflash_destroy(df);
  gguf_destroy(mg);
  q4x_weights_destroy(w);
  gguf_destroy(g);
  return 0;
}
