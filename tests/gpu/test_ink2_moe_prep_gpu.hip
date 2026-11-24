// ink2_moe_prep (kernels/ink2.hip: a decode's or a verify's MoE preparation in one launch) against the three launches
// it replaces - ops_router_scored (sigmoid), q4x_expert_tiles, ink2_moe_scales - on the same inputs: every output
// the same bits (the picks, their weights, mm_group's tables entry by entry, the scales, the layer's counts), at
// 1 .. 32 tokens (to its 256 picks), Mini's 128 experts and 8 picks and a wider 256 x 16; logits with ties (an
// expert repeated), NaNs and huge magnitudes. 77 never: no model read
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern "C" {
#include "osmos/gpu/ink2_ops.h"
#include "osmos/gpu/ops.h"
#include "osmos/gpu/q4x_ops.h"
}

#define CHECK(x)                                                                 \
  do {                                                                           \
    if (!(x)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #x); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

template <class T>
static T* dev(size_t n) {
  T* p = nullptr;
  CHECK(hipMalloc((void**) &p, n * sizeof(T) + 64) == hipSuccess && hipMemset(p, 0xA5, n * sizeof(T) + 64) == hipSuccess);
  return p;
}
template <class T>
static std::vector<T> host(const T* d, size_t n) {
  std::vector<T> v(n);
  CHECK(hipMemcpy(v.data(), d, n * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess);
  return v;
}
template <class T>
static bool same(const T* a, const T* b, size_t n) {  // every byte (a NaN's bits too)
  return !std::memcmp(host(a, n).data(), host(b, n).data(), n * sizeof(T));
}

struct Out {  // one run's outputs, its tables' buffers
  int32_t *idx, *cnt, *gidx, *hidx, *where, *off;
  float *w, *sg, *su, *a;
  uint32_t *xg, *yg, *xd, *yd, *picks;
  const void** ptr[Q4X_GROUP_TENSORS];
  Q4xExpertTiles t;
};

static Out make(uint32_t P, uint32_t E, const Q4xExpertBounds& bd, const void* const* tab[Q4X_GROUP_TENSORS]) {
  Out o{};
  o.idx = dev<int32_t>(P), o.w = dev<float>(P), o.cnt = dev<int32_t>(bd.entries), o.gidx = dev<int32_t>(bd.slots);
  o.hidx = dev<int32_t>(bd.slots), o.where = dev<int32_t>(P), o.off = dev<int32_t>(3 * E);
  o.xg = dev<uint32_t>(bd.entries), o.yg = dev<uint32_t>(bd.entries), o.xd = dev<uint32_t>(bd.entries);
  o.yd = dev<uint32_t>(bd.entries), o.sg = dev<float>(bd.slots), o.su = dev<float>(bd.slots), o.a = dev<float>(P);
  o.picks = dev<uint32_t>(E);
  CHECK(hipMemset(o.picks, 0, E * 4) == hipSuccess);
  for (int j = 0; j < Q4X_GROUP_TENSORS; j++) o.ptr[j] = dev<const void*>(bd.entries), o.t.tab[j] = tab[j], o.t.ptr[j] = o.ptr[j];
  o.t.cnt = o.cnt, o.t.gidx = o.gidx, o.t.hidx = o.hidx, o.t.xg = o.xg, o.t.yg = o.yg, o.t.xd = o.xd, o.t.yd = o.yd;
  o.t.where = o.where, o.t.off = o.off;
  return o;
}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[test_ink2_moe_prep_gpu] start\n");
  std::mt19937 rng(1017);
  int cases = 0;
  for (const uint32_t E : {128u, 256u})
    for (const uint32_t K : {8u, 16u})
      for (const uint32_t n : {1u, 2u, 3u, 4u, 5u, 8u, 16u, 32u})
        for (int kind = 0; kind < 3; kind++) {  // plain, ties (one logit repeated across experts), NaNs and huge
          const uint32_t P = n * K, C = 2304, F = 1152;
          if (P > INK2_MOE_PREP_PICKS || K > E) continue;
          std::normal_distribution<float> nd(0.0f, 2.0f);
          std::vector<float> lg((size_t) n * E), bias(E), scale(3 * E);
          for (auto& x : lg) x = nd(rng);
          for (auto& x : bias) x = 0.1f * nd(rng);
          for (auto& x : scale) x = 1.0f + 0.01f * nd(rng);
          if (kind == 1)
            for (uint32_t t = 0; t < n; t++)
              for (uint32_t e = 0; e < E; e += 3) lg[(size_t) t * E + e] = lg[(size_t) t * E];
          if (kind == 2)
            for (uint32_t t = 0; t < n; t++) lg[(size_t) t * E + t % E] = NAN, lg[(size_t) t * E + (t + 5) % E] = 1e30f,
                                             lg[(size_t) t * E + (t + 9) % E] = -1e30f;
          float *d_lg = dev<float>(lg.size()), *d_bias = dev<float>(E), *d_scale = dev<float>(3 * E);
          CHECK(hipMemcpy(d_lg, lg.data(), lg.size() * 4, hipMemcpyHostToDevice) == hipSuccess);
          CHECK(hipMemcpy(d_bias, bias.data(), E * 4, hipMemcpyHostToDevice) == hipSuccess);
          CHECK(hipMemcpy(d_scale, scale.data(), 3 * E * 4, hipMemcpyHostToDevice) == hipSuccess);
          const void* const* tab[Q4X_GROUP_TENSORS];  // each tensor's experts' addresses (made up: compared, not read)
          for (int j = 0; j < Q4X_GROUP_TENSORS; j++) {
            std::vector<const void*> h(E);
            for (uint32_t e = 0; e < E; e++) h[e] = (const void*) (uintptr_t) (0x100000 * (j + 1) + 4096 * e);
            const void** d = dev<const void*>(E);
            CHECK(hipMemcpy(d, h.data(), E * sizeof(void*), hipMemcpyHostToDevice) == hipSuccess);
            tab[j] = d;
          }
          Q4xExpertBounds bd;
          q4x_expert_bounds(P, E, &bd);
          Out r = make(P, E, bd, tab), f = make(P, E, bd, tab);
          const float rs = 2.5f;
          CHECK(ops_router_scored(d_lg, d_bias, E, K, rs, OPS_SCORE_SIGMOID, n, r.idx, r.w, nullptr) &&
                q4x_expert_tiles(r.idx, P, K, E, C, F, &r.t, nullptr) &&
                ink2_moe_scales(r.idx, r.w, r.where, d_scale, E, P, r.sg, r.su, r.a, r.picks, nullptr));
          CHECK(ink2_moe_prep(d_lg, d_bias, E, K, rs, n, f.idx, f.w, C, F, &f.t, d_scale, f.sg, f.su, f.a, f.picks,
                              nullptr));
          CHECK(hipDeviceSynchronize() == hipSuccess);
          // every output the chain's - the picks first: the tables only mean the same with them
          const bool ok = same(r.idx, f.idx, P) && same(r.w, f.w, P) && same(r.cnt, f.cnt, bd.entries) &&
                          same(r.gidx, f.gidx, bd.slots) && same(r.hidx, f.hidx, bd.slots) &&
                          same(r.where, f.where, P) && same(r.off, f.off, 3 * E) && same(r.xg, f.xg, bd.entries) &&
                          same(r.yg, f.yg, bd.entries) && same(r.xd, f.xd, bd.entries) &&
                          same(r.yd, f.yd, bd.entries) && same(r.sg, f.sg, bd.slots) && same(r.su, f.su, bd.slots) &&
                          same(r.a, f.a, P) && same(r.picks, f.picks, E) &&
                          same(r.ptr[0], f.ptr[0], bd.entries) && same(r.ptr[1], f.ptr[1], bd.entries) &&
                          same(r.ptr[2], f.ptr[2], bd.entries);
          if (!ok) std::fprintf(stderr, "  E %u K %u n %u kind %d: an output differs\n", E, K, n, kind);
          CHECK(ok);
          cases++;
        }
  // past its picks it refuses (the three launches then), never part of the step
  {
    const Q4xExpertTiles t{};
    CHECK(!ink2_moe_prep(nullptr, nullptr, 128, 8, 1.0f, 33, nullptr, nullptr, 2304, 1152, &t, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr));
  }
  std::printf("  %d cases: every output the three launches' bits (picks, weights, tables, scales, counts)\n", cases);
  std::printf("[test_ink2_moe_prep_gpu] ok\n");
  return 0;
}
