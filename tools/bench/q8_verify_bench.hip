// q8_verify_bench: a family's routed experts at a decode or a verify on the int8 lane as the engine runs them -
// route_tok's tables on the GPU, then q8_experts exact (x rounded, gate + up + the activation, h rounded, down) - at
// 1 .. 8 tokens, each token k distinct experts of a pool picked at random (uniform: the most distinct), cold (each
// call another of the pool's copies: past the GPU's caches). microseconds a layer and the weights' GB/s (each distinct
// expert's three matrices read once a call: the bandwidth a layer's tables could at most give back). random weights:
// speed alone.
//   q8_verify_bench <family: hy4 | qwen | small> [iters]
// REPLAY=dump.bin (OSMOS_ROUTE_DUMP's records: n_tok, k, n_exp, the picks): each size's calls the recorded picks in
// turn (a real verify's sharing), not random
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
#include "osmos/gpu/mv.h"
#include "osmos/gpu/q8.h"
#include "osmos/gpu/route.h"
#include "osmos/model/qtype.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

struct Family {
  const char* name;
  uint32_t d, ff, n_exp, k, gate_up_type, down_type;
};

static const Family kFamilies[] = {
    {"hy4", 1536, 1536, 512, 16, QTYPE_Q3_K, QTYPE_Q4_K},       // its latent MoE (Q3_K / Q4_K by layer)
    {"qwen", 2560, 640, 512, 10, QTYPE_IQ3_XXS, QTYPE_IQ4_NL},  // UD-IQ3_XXS's mix
    {"small", 4608, 2304, 384, 8, QTYPE_IQ1_S, QTYPE_IQ3_XXS},  // UD-IQ1_S's experts (IQ1 / IQ2 gate-up, IQ3 down)
};
constexpr int kCopies = 4;  // the pool's copies a call rotates through (each ~0.5-2.6 GB: none in a cache)

static uint32_t g_rng = 181;
static uint32_t next() { return g_rng = g_rng * 1664525u + 1013904223u; }

template <class T>
static T* dev(size_t n) {
  T* p = nullptr;
  CHECK(hipMalloc((void**) &p, n * sizeof(T) + 256) == hipSuccess);
  CHECK(hipMemset(p, 0, n * sizeof(T) + 256) == hipSuccess);
  return p;
}

static uint64_t row_bytes(uint32_t type, uint32_t cols) {
  const QTypeInfo* q = qtype_info(type);
  CHECK(q && cols % q->values == 0);
  return (uint64_t) cols / q->values * q->bytes;
}

// a pool's matrices of one role: n_exp of rows x cols, random bytes (speed alone)
static uint8_t* pool(uint32_t type, uint32_t n_exp, uint32_t rows, uint32_t cols, uint64_t* each) {
  *each = row_bytes(type, cols) * rows;
  const uint64_t all = *each * n_exp;
  uint8_t* p = dev<uint8_t>(all);
  std::vector<uint8_t> h(all);
  for (auto& b : h) b = (uint8_t) (next() >> 24);
  // (scales of zero would leave every sum 0: random bytes keep them random; a NaN scale moves no integer dot)
  CHECK(hipMemcpy(p, h.data(), all, hipMemcpyHostToDevice) == hipSuccess);
  return p;
}

int main(int argc, char** argv) {
  if (argc < 2) return std::fprintf(stderr, "usage: q8_verify_bench <hy4 | qwen | small> [iters]\n"), 2;
  const Family* F = nullptr;
  for (const auto& f : kFamilies)
    if (!std::strcmp(argv[1], f.name)) F = &f;
  if (!F) return std::fprintf(stderr, "q8_verify_bench: no family %s\n", argv[1]), 2;
  const int iters = argc > 2 ? std::atoi(argv[2]) : 40;
  // APART=1: no expert two tokens': every count 1 (exact's second launch all empty blocks - what count-sorted tables
  // could save at most)
  const bool apart = std::getenv("APART") && std::atoi(std::getenv("APART"));
  std::vector<std::vector<std::vector<int32_t>>> rec(MV_MAX_TOKENS + 1);  // REPLAY's records by their tokens
  if (const char* rp = std::getenv("REPLAY")) {
    FILE* rf = std::fopen(rp, "rb");
    CHECK(rf);
    for (uint32_t hd[3]; std::fread(hd, 4, 3, rf) == 3;) {
      std::vector<int32_t> p((size_t) hd[0] * hd[1]);
      CHECK(std::fread(p.data(), 4, p.size(), rf) == p.size());
      CHECK(hd[1] == F->k && hd[2] == F->n_exp);
      if (hd[0] <= MV_MAX_TOKENS) rec[hd[0]].push_back(std::move(p));
    }
    std::fclose(rf);
  }
  const bool replay = std::getenv("REPLAY") != nullptr;
  // BY_COUNT=0: exact's two launches over every entry (the tables' order unused), else route_tok's by count (the engine's)
  const bool by_count = !std::getenv("BY_COUNT") || std::atoi(std::getenv("BY_COUNT"));
  CHECK(q8_takes(F->gate_up_type) && q8_takes(F->down_type));
  const uint32_t E = F->n_exp, K = F->k, D = F->d, FF = F->ff, NT = MV_MAX_TOKENS, NK = NT * K;
  // the pools: kCopies of gate, up, down; each copy's device table [3][E] of their addresses
  uint64_t gb = 0, db = 0;
  std::vector<const void**> tabs(kCopies);
  for (int c = 0; c < kCopies; c++) {
    uint8_t *g = pool(F->gate_up_type, E, FF, D, &gb), *u = pool(F->gate_up_type, E, FF, D, &gb),
            *w = pool(F->down_type, E, D, FF, &db);
    std::vector<const void*> h(3 * (size_t) E);
    for (uint32_t e = 0; e < E; e++) h[e] = g + e * gb, h[E + e] = u + e * gb, h[2 * E + e] = w + e * db;
    tabs[c] = dev<const void*>(3 * (size_t) E);
    CHECK(hipMemcpy(tabs[c], h.data(), h.size() * sizeof(void*), hipMemcpyHostToDevice) == hipSuccess);
  }
  const uint64_t expert_bytes = 2 * gb + db;
  int32_t *idx = dev<int32_t>(NK), *gidx = dev<int32_t>(NK), *sidx = dev<int32_t>(NK), *cnt = dev<int32_t>(NK),
          *xrow = dev<int32_t>(NK), *fault = dev<int32_t>(1);
  float *wtok = dev<float>(NK), *wsel = dev<float>(NK);
  const void** wptr = dev<const void*>(3 * (size_t) NK);
  float *x = dev<float>((size_t) NT * D), *h = dev<float>((size_t) NK * FF), *y = dev<float>((size_t) NK * D),
        *xqd = dev<float>((size_t) NT * D / 32), *hqd = dev<float>((size_t) NK * FF / 32);
  int8_t *xq = dev<int8_t>((size_t) NT * D), *hq = dev<int8_t>((size_t) NK * FF);
  {
    std::vector<float> hx((size_t) NT * D), hw(NK, 1.0f / K);
    for (auto& v : hx) v = (float) ((int32_t) next() >> 8) * 1e-7f;
    CHECK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
    CHECK(hipMemcpy(wtok, hw.data(), hw.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
  }
  hipStream_t st;
  CHECK(hipStreamCreate(&st) == hipSuccess);
  std::printf("q8_verify_bench %s: d %u, ff %u, %u experts, %u a token; gate/up %s, down %s; %.2f MB an expert; "
              "%d iters, %d cold copies\n",
              F->name, D, FF, E, K, qtype_info(F->gate_up_type)->name, qtype_info(F->down_type)->name,
              expert_bytes / 1e6, iters, kCopies);
  std::printf("  tokens  distinct  route+experts us   GB/s (distinct experts' bytes)\n");
  for (uint32_t n = 1; n <= NT; n++) {
    const uint32_t nk = n * K, chunk = n < MV_MAX_TOKENS ? n : MV_MAX_TOKENS;
    if (replay && rec[n].empty()) continue;  // (no recorded call of this size)
    double us = 0, distinct = 0;
    for (int it = -2; it < iters; it++) {  // (two warm-ups)
      std::vector<int32_t> hi(nk);
      std::vector<bool> any(E, false);
      if (replay) {
        hi = rec[n][(size_t) (it + 2) % rec[n].size()];
        for (int32_t e : hi) any[e] = true;
      } else
      for (uint32_t t = 0; t < n; t++)
        for (uint32_t j = 0; j < K; j++) {
          int32_t e;
          do e = (int32_t) ((next() >> 12) % E);
          while (std::find(hi.begin() + (apart ? 0 : t * K), hi.begin() + t * K + j, e) != hi.begin() + t * K + j);
          hi[t * K + j] = e, any[e] = true;
        }
      CHECK(hipMemcpy(idx, hi.data(), nk * 4, hipMemcpyHostToDevice) == hipSuccess);
      const int c = (it + 2) % kCopies;
      const Q8Experts e = {.act = Q8_ACT_LANE, .gate_type = F->gate_up_type, .up_type = F->gate_up_type,
                           .down_type = F->down_type, .wg = wptr, .wu = wptr + nk, .wd = wptr + 2 * nk, .cnt = cnt,
                           .xrow = xrow, .xin = gidx, .n = nk, .max_cnt = chunk, .h_cnt = nullptr, .by_count = by_count,
                           .d = D, .ff = FF,
                           .x = x, .x_rows = n, .xq = xq, .xqd = xqd, .scale = wsel, .win = nullptr, .lim = 7.0f,
                           .h = h, .h_rows = nk, .hq = hq, .hqd = hqd, .y = y};
      CHECK(hipStreamSynchronize(st) == hipSuccess);
      const auto t0 = std::chrono::steady_clock::now();  // (the host's clock: launch to done, as a step sees it)
      CHECK(route_tok(idx, wtok, n, K, E, chunk, tabs[c], gidx, sidx, wsel, cnt, xrow, wptr, fault, st));
      CHECK(q8_experts(&e, st));
      CHECK(hipStreamSynchronize(st) == hipSuccess && hipGetLastError() == hipSuccess);
      int32_t hf = 0;
      CHECK(hipMemcpy(&hf, fault, 4, hipMemcpyDeviceToHost) == hipSuccess && hf == 0);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (it >= 0) us += ms * 1000.0, distinct += (double) std::count(any.begin(), any.end(), true);
    }
    us /= iters, distinct /= iters;
    std::printf("  %6u  %8.1f  %16.1f   %6.1f\n", n, distinct, us, distinct * expert_bytes / (us * 1e3));
  }
  return 0;
}
