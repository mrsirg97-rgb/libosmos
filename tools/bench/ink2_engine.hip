// ink2_engine - Inkling's forward with a cache (osmos/ink2/forward.h) held to ink2_run's: one prompt's tokens in steps
// of the sizes given, every row's log-probabilities against ink2_run's (REF_OUT of the same ids, ATTN=fa), and the
// sequence's own checks - a rollback and the same tokens again the same bits, a save and load the same bits on.
//
//   ink2_engine <first shard> <ids> <steps> [ref.f32]   steps: "200,64*30,1*10" - a step's tokens, *k for k of them
//                                                     (the last one repeated to the ids' end)
//   ROOM=GiB      the experts' cache (default 32)
//   PICKS=path    a picks profile to warm by (the most picked experts first); PICKS_OUT=path: this run's written
//   DRAFT=gguf    the drafter (ink2_mtp_gguf.py's): after each step its context seeded, a draft at the next anchor
//                 (the token after the step's, no floors) printed beside the text's - its acceptance
//   CHECK=1       at each 1-token step: roll back, run it again, the logits the same bits; and once a save, a load
//                 into a new sequence and its next step the same bits
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "osmos/ink2/forward.h"
#include "osmos/ink2/mtp.h"
#include "osmos/ink2/weights.h"
#include "osmos/model/gguf.h"
}

#define CHECK(x)                                                                  \
  do {                                                                            \
    if (!(x)) {                                                                   \
      std::fprintf(stderr, "%s:%d: failed: %s (%s)\n", __FILE__, __LINE__, #x, err); \
      std::exit(1);                                                               \
    }                                                                             \
  } while (0)

static char err[512];

static std::vector<int32_t> read_ints(const char* arg) {
  std::string text = arg;
  if (FILE* f = std::fopen(arg, "rb")) {
    text.clear();
    for (int c; (c = std::fgetc(f)) != EOF;) text.push_back(c == '\n' ? ',' : (char) c);
    std::fclose(f);
  }
  std::vector<int32_t> v;
  for (char* s = std::strtok(text.data(), ","); s; s = std::strtok(nullptr, ","))
    if (*s && *s != '\r') v.push_back(std::atoi(s));
  return v;
}

struct Buf {
  std::vector<uint8_t> b;
  size_t at = 0;
};
static bool sink(void* c, const void* p, uint64_t n) {
  auto* b = (Buf*) c;
  b->b.insert(b->b.end(), (const uint8_t*) p, (const uint8_t*) p + n);
  return true;
}
static bool source(void* c, void* p, uint64_t n) {
  auto* b = (Buf*) c;
  if (b->at + n > b->b.size()) return false;
  std::memcpy(p, b->b.data() + b->at, n), b->at += n;
  return true;
}

int main(int argc, char** argv) {
  if (argc < 4) return std::fprintf(stderr, "ink2_engine <first shard> <ids> <steps> [ref.f32]\n"), 2;
  Gguf* model = gguf_create(argv[1], err);
  CHECK(model);
  Ink2Weights* W = ink2_weights_create(model, err);
  CHECK(W);
  std::vector<int32_t> ids = read_ints(argv[2]);
  const uint32_t T = (uint32_t) ids.size();
  std::vector<uint32_t> steps;
  {
    std::string sp = argv[3];
    for (char* s = std::strtok(sp.data(), ","); s; s = std::strtok(nullptr, ",")) {
      const char* star = std::strchr(s, '*');
      const uint32_t n = (uint32_t) std::atoi(s), k = star ? (uint32_t) std::atoi(star + 1) : 1;
      for (uint32_t i = 0; i < k; i++) steps.push_back(n);
    }
    uint32_t sum = 0;
    for (uint32_t n : steps) sum += n;
    while (sum < T) steps.push_back(steps.back()), sum += steps.back();
  }
  uint32_t maxs = 0;
  for (uint32_t n : steps) maxs = n > maxs ? n : maxs;
  const double room = std::getenv("ROOM") ? std::atof(std::getenv("ROOM")) : 32.0;
  const bool check = std::getenv("CHECK") && std::atoi(std::getenv("CHECK"));
  std::vector<uint64_t> picks;
  if (const char* pp = std::getenv("PICKS")) {
    picks.resize((size_t) W->geo.n_layers * W->geo.n_experts);
    FILE* pf = std::fopen(pp, "rb");
    CHECK(pf && std::fread(picks.data(), 8, picks.size(), pf) == picks.size());
    std::fclose(pf);
  }
  Gguf* dg = nullptr;
  Ink2Mtp* mtp = nullptr;
  if (const char* dp = std::getenv("DRAFT")) {
    dg = gguf_create(dp, err);
    CHECK(dg);
    mtp = ink2_mtp_create(dg, W, err);
    CHECK(mtp);
  }
  const Ink2Options o = {T + 16, maxs, (uint64_t) (room * (double) (1ull << 30)), picks.empty() ? nullptr : picks.data(),
                         mtp};
  uint32_t drafted = 0, accepted = 0, rounds = 0;
  auto t0 = std::chrono::steady_clock::now();
  Ink2Forward* f = ink2_forward_create(W, &o, err);
  CHECK(f);
  auto secs = [](auto a) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count(); };
  std::fprintf(stderr, "ink2_engine: built in %.1f s (%.2f GiB), %u tokens in %zu steps\n", secs(t0),
               (double) ink2_forward_bytes(f) / (1ull << 30), T, steps.size());
  Ink2Seq* s = ink2_seq_create(f, err);
  CHECK(s);
  const uint32_t V = (uint32_t) W->geo.n_vocab;
  FILE* ref = argc > 4 ? std::fopen(argv[4], "rb") : nullptr;
  CHECK(argc <= 4 || ref);
  std::vector<float> lg((size_t) maxs * V), again((size_t) V), rl(V);
  double kl_sum = 0, kl_max = 0;
  uint32_t rows = 0, agree = 0, checked = 0, saved = 0, same = 0;
  std::vector<double> lp(V);
  double t_prompt = 0, t_dec = 0;
  uint32_t n_dec = 0;
  for (uint32_t at = 0, si = 0; at < T; si++) {
    uint32_t n = steps[si] < T - at ? steps[si] : T - at;
    Ink2Seq* one[1] = {s};
    auto ts = std::chrono::steady_clock::now();
    CHECK(ink2_step(f, one, &n, 1, ids.data() + at, err));
    (n == 1 ? t_dec : t_prompt) += secs(ts), n_dec += n == 1;
    CHECK(hipMemcpy(lg.data(), ink2_logits(f), (size_t) n * V * 4, hipMemcpyDeviceToHost) == hipSuccess);
    if (check && n == 1 && ink2_seq_reach(s) >= 1) {  // rolled back and run again: the same bits
      CHECK(ink2_seq_rollback(s, 1));
      CHECK(ink2_step(f, one, &n, 1, ids.data() + at, err));
      CHECK(hipMemcpy(again.data(), ink2_logits(f), (size_t) V * 4, hipMemcpyDeviceToHost) == hipSuccess);
      CHECK(std::memcmp(again.data(), lg.data(), (size_t) V * 4) == 0 || (std::snprintf(err, sizeof err, "rollback at %u", at), false));
      checked++;
    }
    if (check && n == 1 && !saved && at + 1 < T) {  // saved, loaded into another sequence: its next step the same bits
      Buf b;
      CHECK(ink2_seq_save_to(s, sink, &b, err));
      CHECK(b.b.size() == ink2_seq_state_bytes(s));
      Ink2Seq* s2 = ink2_seq_create(f, err);
      CHECK(s2);
      CHECK(ink2_seq_load_from(s2, b.b.size(), source, &b, err));
      uint32_t one_n = 1;
      Ink2Seq* two[1] = {s2};
      CHECK(ink2_step(f, two, &one_n, 1, ids.data() + at + 1, err));
      CHECK(hipMemcpy(again.data(), ink2_logits(f), (size_t) V * 4, hipMemcpyDeviceToHost) == hipSuccess);
      std::vector<float> want(V);
      CHECK(ink2_step(f, one, &one_n, 1, ids.data() + at + 1, err));
      CHECK(hipMemcpy(want.data(), ink2_logits(f), (size_t) V * 4, hipMemcpyDeviceToHost) == hipSuccess);
      CHECK(std::memcmp(again.data(), want.data(), (size_t) V * 4) == 0 || (std::snprintf(err, sizeof err, "load at %u", at), false));
      CHECK(ink2_seq_rollback(s, 1));
      ink2_seq_destroy(s2);
      saved = 1;
      std::fprintf(stderr, "  save and load at %u: %.1f MB, the next step the same bits\n", at + 1, b.b.size() / 1e6);
    }
    if (mtp && at + n + 1 < T) {  // the drafter seeded with this step's positions, then a draft at the next anchor
      CHECK(ink2_seq_seed(f, s, (int32_t) at, (int32_t) (at + n), err));
      int32_t dr[8];
      float cf[8];
      const int k = ink2_seq_draft(f, s, ids[at + n], 7, 0.0f, 0.0f, dr, cf, err);
      CHECK(k >= 0);
      uint32_t acc = 0;
      while (acc < (uint32_t) k && at + n + 1 + acc < T && dr[acc] == ids[at + n + 1 + acc]) acc++;
      drafted += (uint32_t) k, accepted += acc, rounds++;
      std::printf("p %u: %u accepted  drafts [", at + n, acc);
      for (int i = 0; i < k; i++) std::printf("%s%d", i ? ", " : "", dr[i]);
      std::printf("]\n");
    }
    for (uint32_t r = 0; r < n && at + r + 1 < T; r++) {
      const float* x = &lg[(size_t) r * V];
      int32_t top = 0;
      for (uint32_t v = 1; v < V; v++) top = x[v] > x[top] ? (int32_t) v : top;
      double sum = 0;
      for (uint32_t v = 0; v < V; v++) sum += std::exp((double) x[v] - x[top]);
      const double lse = x[top] + std::log(sum);
      for (uint32_t v = 0; v < V; v++) lp[v] = (double) x[v] - lse;
      if (!ref) continue;
      CHECK(std::fread(rl.data(), 4, V, ref) == V);
      double kl = 0;
      int32_t rt = 0;
      for (uint32_t v = 0; v < V; v++) {
        rt = rl[v] > rl[rt] ? (int32_t) v : rt;
        const double p = std::exp((double) rl[v]);
        if (p > 0) kl += p * ((double) rl[v] - lp[v]);
      }
      kl = kl > 0 ? kl : 0;
      bool eq = true;  // the reference's row bit for bit: its log-probabilities from the same logits
      for (uint32_t v = 0; eq && v < V; v++) eq = (float) lp[v] == rl[v];
      same += eq;
      kl_sum += kl, kl_max = kl > kl_max ? kl : kl_max, agree += rt == top, rows++;
    }
    at += n;
  }
  std::printf("ink2_engine: %u rows", rows);
  if (rows)
    std::printf(", kld mean %.3g max %.3g against the reference, top-1 %.2f%%, %u rows its bits", kl_sum / rows, kl_max,
                100.0 * agree / rows, same);
  std::printf("; %u rollbacks the same bits; prompt steps %.1f s, %u decode steps %.2f tok/s\n", checked, t_prompt,
              n_dec, n_dec ? n_dec / t_dec : 0.0);
  if (const char* po = std::getenv("PICKS_OUT")) {
    FILE* pf = std::fopen(po, "wb");
    CHECK(pf && std::fwrite(ink2_forward_picks(f), 8, (size_t) W->geo.n_layers * W->geo.n_experts, pf) ==
                    (size_t) W->geo.n_layers * W->geo.n_experts);
    std::fclose(pf);
  }
  if (rounds) std::printf("ink2_engine: drafts %u rounds, %.2f accepted a round\n", rounds, (double) accepted / rounds);
  ink2_seq_destroy(s);
  ink2_forward_destroy(f);
  ink2_mtp_destroy(mtp);
  gguf_destroy(dg);
  ink2_weights_destroy(W);
  gguf_destroy(model);
  return 0;
}
