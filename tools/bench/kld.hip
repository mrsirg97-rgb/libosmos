// kld budget_gib step n ids.txt: a DeepSeek model's distance from a reference on a text, token by token (DESIGN.md,
// "0.6.6: the reference, and how a quant is scored") - one sequence, `step` tokens a forward_batch (the prompt's
// path: the same arithmetic for every run at the same step), each row's log-softmax in double on the host.
//   REF_OUT=path   the run is the reference: every scored row's log-probabilities written (F32, [rows][vocab])
//   REF=path       the run is a candidate: each scored row's KL(reference || run) over the whole vocabulary
//   DUMP=path      either run: a record a row - kl F32 (0 for the reference), nll F32 (the next token's), top I32 (the
//                  run's argmax, the lowest id among equals), ref_top I32 (the reference's; -1 for the reference)
//   MASK=mask.txt  0 / 1 a token (comma separated, as ids.txt): row r scored only when token r + 1 is 1 - an agent
//                  trace's assistant tokens alone; every row without it
// prints the mean KL, its median and 99th percentile, the top-1 agreement and the perplexity over the scored rows;
// the per-row dumps are tools/bench/kld_compare.py's (paired differences, splits by row).
// environment: MODEL (a first shard), PICKS (forward_create_ranked's profile), OVERLAY (weights_create_with's).
// build: hipcc -O2 -Iinclude --offload-arch=gfx1151 tools/bench/kld.hip -Llib -losmos_hip -losmos
//        -lhsa-runtime64 -o bin/kld   (after `make lib/libosmos.a lib/libosmos_hip.a`)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "bench_common.h"

extern "C" {
#include "osmos/model/gguf.h"
#include "osmos/model/weights.h"
#include "osmos/run/forward.h"
#include "osmos/run/fit.h"
#include "osmos/run/engine.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::_Exit(1);                                                             \
    }                                                                            \
  } while (0)

// the reference's file: this header, then a row of vocab F32 log-probabilities for each scored row, in order
struct RefHead {
  char magic[8];  // "osmosref"
  uint32_t vocab, n, step, rows;  // the vocabulary, the text's tokens, the step, the rows written
};
struct Rec {
  float kl, nll;
  int32_t top, ref_top;
};

static std::vector<int32_t> read_list(const char* path) {
  std::vector<int32_t> v;
  FILE* f = std::fopen(path, "r");
  CHECK(f);
  for (long x; std::fscanf(f, "%ld,", &x) == 1;) v.push_back((int32_t) x);
  std::fclose(f);
  return v;
}

int main(int argc, char** argv) {
  if (argc != 5) return std::fprintf(stderr, "usage: kld budget_gib step n ids.txt (MODEL, PICKS, OVERLAY, REF_OUT | REF, DUMP, MASK)\n"), 2;
  // budget "auto" (0.7.3): sized as osmos-serve sizes it (fit_sized: the GPU's headroom and FIT_RAM_FREE kept)
  const bool auto_budget = !std::strcmp(argv[1], "auto");
  const uint64_t budget = auto_budget ? 0 : (uint64_t) (std::atof(argv[1]) * (double) (1ull << 30));
  const uint32_t step = (uint32_t) std::atoi(argv[2]), n = (uint32_t) std::atoi(argv[3]);
  const std::vector<int32_t> ids = read_list(argv[4]);
  CHECK(step > 0 && step <= MM_MAX_TOKENS && ids.size() >= n && n > 1);
  std::vector<int32_t> mask(n, 1);
  if (const char* mp = std::getenv("MASK")) {
    mask = read_list(mp);
    CHECK(mask.size() >= n);
  }
  const char *ref_out = std::getenv("REF_OUT"), *ref_in = std::getenv("REF"), *dump = std::getenv("DUMP");
  CHECK(!(ref_out && ref_in));

  const char* mp = std::getenv("MODEL");
  CHECK(mp);
  char err[GGUF_ERR] = "", werr[WEIGHTS_ERR] = "", ferr[FORWARD_ERR] = "";
  Gguf* g = gguf_create(mp, err);
  if (!g) std::fprintf(stderr, "gguf_create: %s\n", err), std::_Exit(1);
  const char* op = std::getenv("OVERLAY");
  Gguf* ov = op ? gguf_create(op, err) : nullptr;
  if (op && !ov) std::fprintf(stderr, "%s: %s\n", op, err), std::_Exit(1);
  const WeightsOptions wo = {g, nullptr, ov};
  Weights* w = weights_create_with(&wo, werr);
  if (!w) std::fprintf(stderr, "weights_create: %s\n", werr), std::_Exit(1);
  const uint32_t V = (uint32_t) w->geo.n_vocab;

  uint32_t want = 0;  // the rows scored: those whose next token the mask keeps
  for (uint32_t r = 0; r + 1 < n; r++) want += mask[r + 1] != 0;
  FILE *ro = nullptr, *ri = nullptr, *dp = nullptr;
  if (ref_out) {
    CHECK((ro = std::fopen(ref_out, "wb")));
    RefHead h = {{'o', 's', 'm', 'o', 's', 'r', 'e', 'f'}, V, n, step, want};
    CHECK(std::fwrite(&h, sizeof h, 1, ro) == 1);
  }
  if (ref_in) {
    RefHead h;
    CHECK((ri = std::fopen(ref_in, "rb")) && std::fread(&h, sizeof h, 1, ri) == 1);
    CHECK(!std::memcmp(h.magic, "osmosref", 8) && h.vocab == V && h.n == n && h.rows == want);
    if (h.step != step) std::fprintf(stderr, "note: the reference's step %u, this run's %u\n", h.step, step);
  }
  if (dump) CHECK((dp = std::fopen(dump, "wb")));

  ForwardOptions o = {};
  // BIG (a step past MM_MAX_TOKENS, up to FORWARD_MAX_TOKENS: the server's own): a batch none of whose rows the mask
  // scores read BIG tokens at once (its logits not read; a batch's sub-batches the same bits as a step each,
  // forward.h), the rest `step` - a long text's few scored windows without the unscored rows' 4x expert reads
  const uint32_t big = std::getenv("BIG") ? (uint32_t) std::atoi(std::getenv("BIG")) : 0;
  CHECK(!big || (big >= step && big <= FORWARD_MAX_TOKENS));
  o.n_layers = (int) w->geo.n_layers, o.max_ctx = n, o.budget = budget, o.max_tokens = big ? big : step;
  if (const char* fl = std::getenv("FILL"))  // GiB of RAM a build keeps free (osmos-serve --fill): this tool's own
    o.fill_set = true, o.fill = (uint64_t) (std::atof(fl) * 1073741824.0);  // buffers beside it counted in
  std::vector<uint64_t> picks;
  if (const char* pp = std::getenv("PICKS")) {
    FILE* pf = std::fopen(pp, "rb");
    CHECK(pf);
    picks.resize((size_t) w->geo.n_layers * w->geo.n_experts);
    CHECK(std::fread(picks.data(), 8, picks.size(), pf) == picks.size());
    std::fclose(pf);
    o.picks = picks.data();
  }
  Forward* f = nullptr;
  if (auto_budget) {
    struct Ctx { const Weights* w; ForwardOptions o; Forward* f; } z = {w, o, nullptr};
    const FitBuild build = [](void* c, uint64_t b, uint64_t* built, uint64_t* room, char* err) -> bool {
      Ctx* z = (Ctx*) c;
      z->o.budget = b;
      char e[FORWARD_ERR] = "";
      z->f = forward_create_with(z->w, &z->o, e);
      if (!z->f || !forward_lane(z->f, 0, NULL, e) || !forward_room(z->f, room)) {
        if (z->f) forward_destroy(z->f), z->f = nullptr;
        return std::snprintf(err, FIT_ERR, "%s", e[0] ? e : "the room could not be read"), false;
      }
      *built = forward_budget(z->f);
      std::fprintf(stderr, "sizing: a build at %.2f GiB took %.2f, left %.2f GiB\n", b / 1073741824.0,
                   *built / 1073741824.0, *room / 1073741824.0);
      return true;
    };
    const FitUnbuild unbuild = [](void* c) {
      Ctx* z = (Ctx*) c;
      if (z->f) forward_destroy(z->f), z->f = nullptr;
    };
    uint64_t sized = 0;
    char ferr2[FIT_ERR] = "";
    if (!fit_sized(build, unbuild, &z, FIT_HEADROOM, &sized, ferr2)) std::fprintf(stderr, "fit_sized: %s\n", ferr2), std::_Exit(1);
    f = z.f;
    std::fprintf(stderr, "the budget sized to this GPU: %.2f GiB\n", sized / 1073741824.0);
  } else {
    f = forward_create_with(w, &o, ferr);
    if (!f) std::fprintf(stderr, "forward_create_with: %s\n", ferr), std::_Exit(1);
    CHECK(forward_lane(f, 0, NULL, ferr));  // as the engine and ppl.hip: a step under n_experts / n_used rows the engine's
  }
  if (const char* nd = std::getenv("NPU"))  // 0.7.0: the NPU beside the GPU (forward_npu), NPU_SHARE of a layer's pairs
    if (!forward_npu(f, nd, std::getenv("NPU_SHARE") ? std::strtof(std::getenv("NPU_SHARE"), nullptr) : OSMOS_NPU_SHARE_DS, ferr))
      std::fprintf(stderr, "forward_npu: %s\n", ferr), std::_Exit(1);
  Seq* s = seq_create(f, ferr);
  CHECK(s);
  std::vector<Seq*> ss(big ? big : step, s);
  std::vector<float> lg((size_t) step * V), rl(V);
  std::vector<double> lp(V);
  std::vector<float> kls;
  double nll = 0;
  uint32_t scored = 0, agree = 0;
  for (uint32_t at = 0, k; at < n; at += k) {
    if (big) {  // a big batch when none of its rows is scored (row r predicts token at + r + 1)
      const uint32_t kb = n - at < big ? n - at : big;
      bool any = false;
      for (uint32_t r = 0; r < kb && at + r + 1 < n && !any; r++) any = mask[at + r + 1] != 0;
      if (!any) {
        if (!forward_batch(f, ss.data(), ids.data() + at, kb, ferr))
          std::fprintf(stderr, "step at %u: %s\n", at, ferr), std::_Exit(1);
        k = kb;
        std::printf("  %u of %u tokens, %u scored\n", at + k, n, scored);
        std::fflush(stdout);
        continue;
      }
    }
    k = n - at < step ? n - at : step;
    if (!forward_batch(f, ss.data(), ids.data() + at, k, ferr))
      std::fprintf(stderr, "step at %u: %s\n", at, ferr), std::_Exit(1);
    CHECK(forward_logit_rows(f) == k);  // every row of the step kept
    CHECK(hipMemcpy(lg.data(), forward_logits(f), (size_t) k * V * 4, hipMemcpyDeviceToHost) == hipSuccess);
    for (uint32_t r = 0; r < k && at + r + 1 < n; r++) {  // row r predicts token at + r + 1
      if (!mask[at + r + 1]) continue;
      const float* x = &lg[(size_t) r * V];
      double mx = x[0], sum = 0;
      int32_t top = 0;
      for (uint32_t v = 1; v < V; v++)
        if (x[v] > x[top]) top = (int32_t) v;
      mx = x[top];
      for (uint32_t v = 0; v < V; v++) sum += std::exp((double) x[v] - mx);
      CHECK(std::isfinite(sum));
      const double lse = mx + std::log(sum);
      for (uint32_t v = 0; v < V; v++) lp[v] = (double) x[v] - lse;
      Rec rec = {0.0f, (float) -lp[ids[at + r + 1]], top, -1};
      nll -= lp[ids[at + r + 1]];
      if (ro) {
        for (uint32_t v = 0; v < V; v++) rl[v] = (float) lp[v];
        CHECK(std::fwrite(rl.data(), 4, V, ro) == V);
      }
      if (ri) {
        CHECK(std::fread(rl.data(), 4, V, ri) == V);
        double kl = 0;
        int32_t rt = 0;
        for (uint32_t v = 0; v < V; v++) {
          if (rl[v] > rl[rt]) rt = (int32_t) v;
          const double p = std::exp((double) rl[v]);
          if (p > 0) kl += p * ((double) rl[v] - lp[v]);
        }
        rec.kl = (float) std::max(kl, 0.0), rec.ref_top = rt;
        agree += rt == top;
        kls.push_back(rec.kl);
      }
      if (dp) CHECK(std::fwrite(&rec, sizeof rec, 1, dp) == 1);
      scored++;
    }
    std::printf("  %u of %u tokens, %u scored\n", at + k, n, scored);
    std::fflush(stdout);
  }
  std::printf("ppl %.4f over %u scored rows (mean log-prob %.5f), steps of %u\n", std::exp(nll / scored), scored,
              -nll / scored, step);
  if (ri && !kls.empty()) {
    double m = 0;
    for (float v : kls) m += v;
    m /= (double) kls.size();
    std::vector<double> sorted(kls.begin(), kls.end());
    std::sort(sorted.begin(), sorted.end());
    std::printf("kld mean %.6f median %.6f p99 %.6f, top-1 agreement %.2f%%\n", m, bench_quantile(sorted, 0.5),
                bench_quantile(sorted, 0.99),
                100.0 * agree / (double) kls.size());
  }
  if (ro) std::fclose(ro);
  if (ri) std::fclose(ri);
  if (dp) std::fclose(dp);
  seq_destroy(s);
  forward_destroy(f);
  weights_destroy(w);
  gguf_destroy(g);
  return 0;
}
