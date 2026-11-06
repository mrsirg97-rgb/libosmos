// QSA (DESIGN.md, "QSA"): the indexer and the attention over its picks against llama.cpp's own tensors at 3,000
// tokens of real text, layer 3 (qwen_dump, QWEN_DUMP_LAYER=3, into ~/osmos-oracle/qwen-dump-3000) - past the 2,051
// positions below which every block is picked. the projections and the query from llama.cpp's mixed input; the
// pooled keys from its raw keys (its F16 cache: within f16's rounding); the picks from its query and pooled keys,
// against a double reference on the same (the same blocks but at a tie F32 cannot tell) and against llama.cpp's own;
// a zero query's (every score tied) the lowest blocks; the attention over the picks against double on sampled
// tokens, and against llama.cpp's; then each in pieces (the
// steps 1000 + 999 + 1 + 1000, a token at a time at the end) the bits of one run, and the picks of a token with every
// block visible the dense attention's bits. needs a gfx1151, the UD-IQ3_XXS set and the dump; skipped (77) without.
#include <algorithm>
#include <numeric>

#include "q4x_ref.h"

extern "C" {
#include "osmos/gpu/mv.h"
#include "osmos/gpu/q4x_ops.h"
}

constexpr const char* kDump = "%s/osmos-oracle/qwen-dump-3000";
constexpr uint32_t kTok = 3000, kDim = 2560, kIxH = 4, kIx = Q4X_IX_DIM, kR = 4, kBudget = 2048, kBlocks = kBudget / kR;
constexpr uint32_t kRope = 64, kRing = kR - 1 + 8, kHeads = 24, kKv = 2, kHead = 256, kQ = kHeads * kHead,
                   kKvW = kKv * kHead, kPools = kTok / kR;
constexpr float kEps = 1e-6f, kBase = 1e7f;
constexpr uint32_t kSteps[] = {1000, 999, 1, 1000};  // a prompt's pieces: one past a block, one ending inside one

static std::vector<int32_t> ints_of(const int32_t* d, size_t n) {
  std::vector<int32_t> h(n);
  CHECK(hipDeviceSynchronize() == hipSuccess && hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost) == hipSuccess);
  return h;
}

static uint32_t visible(uint32_t t) { return (t + 1) / kR; }

// the pooled keys of the dump's raw keys: in one call, then in pieces through the ring - the same bits
static float* pooled_of(const Ref& raw, const float* norm) {
  float *dr = dev_from(raw.v), *ring = dev_alloc<float>(kRing * kIx), *one = dev_alloc<float>((size_t) kPools * kIx);
  CHECK(q4x_ix_pool(dr, kTok, 0, kR, ring, kRing, kRope, norm, kEps, kBase, one, nullptr));
  const std::vector<float> whole = host_of(one, (size_t) kPools * kIx);
  float* parts = dev_alloc<float>((size_t) kPools * kIx);
  uint32_t at = 0;
  for (uint32_t n : kSteps) {
    CHECK(q4x_ix_pool(dr + (size_t) at * kIx, n, (int32_t) at, kR, ring, kRing, kRope, norm, kEps, kBase, parts,
                      nullptr));
    at += n;
  }
  CHECK(std::memcmp(host_of(parts, whole.size()).data(), whole.data(), whole.size() * 4) == 0);
  std::printf("  pooled in pieces (1000 + 999 + 1 + 1000, through the ring): the bits of one call\n");
  CHECK(!q4x_ix_pool(dr, kTok, 0, kR, ring, kR - 2, kRope, norm, kEps, kBase, one, nullptr));  // a ring too short
  return one;
}

// a token's top kBlocks blocks by (score, then the lower block) from scores in double, and the gap at the cut
static std::vector<int32_t> exact_pick(const Ref& q, const std::vector<float>& pooled, uint32_t t, double* gap) {
  const uint32_t nv = visible(t);
  std::vector<double> s(nv);
  for (uint32_t j = 0; j < nv; j++) {
    double sum = 0;
    for (uint32_t h = 0; h < kIxH; h++) {
      double dot = 0;
      for (uint32_t d = 0; d < kIx; d++) dot += (double) q.v[((size_t) t * kIxH + h) * kIx + d] * pooled[(size_t) j * kIx + d];
      sum += std::max(dot, 0.0) / std::sqrt((double) kIx);
    }
    s[j] = sum;
  }
  std::vector<int32_t> order(nv);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return s[a] > s[b]; });
  *gap = (s[order[kBlocks - 1]] - s[order[kBlocks]]) / std::max(s[order[kBlocks - 1]], 1e-30);
  std::vector<int32_t> top(order.begin(), order.begin() + kBlocks);
  std::sort(top.begin(), top.end());
  return top;
}

// the picks from the dump's query and pooled keys: against double and llama.cpp's own; then in pieces the same bits
static void picks(const Ref& q, const Ref& kpool, int32_t* sel, int32_t* npick) {
  const std::vector<float> pooled(kpool.v.begin(), kpool.v.begin() + (size_t) kPools * kIx);
  float *dq = dev_from(q.v), *dk = dev_from(pooled);
  const uint64_t sf = q4x_ix_scratch(kTok, kR);
  float* scr = dev_alloc<float>(sf);
  CHECK(q4x_ix_pick(dq, dk, kTok, kIxH, 0, kR, kBlocks, scr, sf, sel, npick, nullptr));
  const std::vector<int32_t> s = ints_of(sel, (size_t) kTok * kBlocks), np = ints_of(npick, kTok);
  const Ref ltop = ref("indexer_top_k-3", kDump);
  uint32_t past = 0, exact = 0, near = 0, llama = 0;
  double worst_gap = 0;
  for (uint32_t t = 0; t < kTok; t++) {
    const uint32_t nv = visible(t);
    CHECK(np[t] == (int32_t) std::min(nv, kBlocks));
    if (nv <= kBlocks) continue;
    past++;
    const std::vector<int32_t> mine(&s[(size_t) t * kBlocks], &s[(size_t) (t + 1) * kBlocks]);
    CHECK(std::is_sorted(mine.begin(), mine.end()) && std::adjacent_find(mine.begin(), mine.end()) == mine.end());
    double gap;
    const std::vector<int32_t> ex = exact_pick(q, pooled, t, &gap);
    if (mine == ex) {
      exact++;
    } else {  // a tie at F32's rounding alone
      near++, worst_gap = std::max(worst_gap, gap);
      CHECK(gap < kExactRel);
    }
    std::vector<int32_t> theirs(kBlocks);
    for (uint32_t i = 0; i < kBlocks; i++) theirs[i] = (int32_t) ltop.v[(size_t) t * kBlocks + i];
    std::sort(theirs.begin(), theirs.end());
    llama += theirs == mine;
  }
  std::printf("  picks past the budget: %u tokens - %u the double's blocks exactly, %u at a tie within %.1e; %u "
              "llama.cpp's\n", past, exact, near, worst_gap, llama);
  CHECK(past > 0 && exact + near == past);
  // in pieces: the same picks
  int32_t *ps = dev_alloc<int32_t>((size_t) kTok * kBlocks), *pn = dev_alloc<int32_t>(kTok);
  CHECK(hipMemcpy(ps, sel, (size_t) kTok * kBlocks * 4, hipMemcpyDeviceToDevice) == hipSuccess);
  uint32_t at = 0;
  for (uint32_t n : kSteps) {
    CHECK(q4x_ix_pick(dq + (size_t) at * kIxH * kIx, dk, n, kIxH, (int32_t) at, kR, kBlocks, scr, sf,
                      ps + (size_t) at * kBlocks, pn + at, nullptr));
    at += n;
  }
  CHECK(ints_of(pn, kTok) == np);
  const std::vector<int32_t> s2 = ints_of(ps, (size_t) kTok * kBlocks);
  for (uint32_t t = 0; t < kTok; t++)
    if (visible(t) > kBlocks) CHECK(std::memcmp(&s2[(size_t) t * kBlocks], &s[(size_t) t * kBlocks], kBlocks * 4) == 0);
  std::printf("  picked in pieces: the bits of one call\n");
  // every score tied (a zero query: each block's 0): the lowest blocks
  CHECK(hipMemset(dq, 0, q.v.size() * 4) == hipSuccess);
  CHECK(q4x_ix_pick(dq, dk, kTok, kIxH, 0, kR, kBlocks, scr, sf, ps, pn, nullptr));
  const std::vector<int32_t> tied = ints_of(ps + (size_t) (kTok - 1) * kBlocks, kBlocks);
  for (uint32_t i = 0; i < kBlocks; i++) CHECK(tied[i] == (int32_t) i);
  std::printf("  every score tied: blocks 0 .. %u, the lowest\n", kBlocks - 1);
}

// a token's key positions: its picked blocks' in order, then its tail (every block visible: 0 .. t)
static std::vector<uint32_t> keys_of(const std::vector<int32_t>& s, const std::vector<int32_t>& np, uint32_t t) {
  const uint32_t nv = visible(t);
  std::vector<uint32_t> k;
  for (int32_t i = 0; i < np[t]; i++)
    for (uint32_t m = 0; m < kR; m++) k.push_back(kR * (np[t] == (int32_t) nv ? (uint32_t) i : (uint32_t) s[(size_t) t * kBlocks + i]) + m);
  for (uint32_t p = kR * nv; p <= t; p++) k.push_back(p);
  return k;
}

int main() {
  std::printf("[test_q4x_qsa_gpu] start\n");
  int ndev = 0;
  char err[GGUF_ERR] = "";
  Ref raw;
  Gguf* gg = gguf_create(home_path(kQwenFirst).c_str(), err);
  if (!gg || !ref_read("indexer_k_raw-3", &raw, kDump) || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_q4x_qsa_gpu] skipped: no Qwen3.8 UD-IQ3_XXS, its 3,000-token dump or a GPU\n");
    return 77;
  }
  const GgufTensor *tiq, *tik;
  const void *wiq = dev_tensor(gg, "blk.3.indexer.q_proj.weight", &tiq), *wik = dev_tensor(gg, "blk.3.indexer.k_proj.weight", &tik);
  const float* qn = (const float*) dev_tensor(gg, "blk.3.indexer.q_norm.weight");
  const float* kn = (const float*) dev_tensor(gg, "blk.3.indexer.k_norm.weight");

  // the projections and the query, from llama.cpp's mixed input
  const Ref mixed = ref("hc_mixed-3", kDump), lq = ref("indexer_q-3", kDump), lk = ref("indexer_k-3", kDump);
  float *hm = dev_from(mixed.v), *iqr = dev_alloc<float>((size_t) kTok * kIxH * kIx);
  float *ikr = dev_alloc<float>((size_t) kTok * kIx), *iq = dev_alloc<float>((size_t) kTok * kIxH * kIx);
  CHECK(mv_strided(tik->type, wik, kIx, kDim, hm, kDim, kTok, ikr, kIx, nullptr));
  CHECK(mv_strided(tiq->type, wiq, kIxH * kIx, kDim, hm, kDim, kTok, iqr, kIxH * kIx, nullptr));
  held("indexer k raw (BF16 projection)", host_of(ikr, (size_t) kTok * kIx), raw);
  CHECK(q4x_ix_q(iqr, kTok, kIxH, kRope, qn, kEps, 0, kBase, iq, nullptr));
  held("indexer q normed, rotated", host_of(iq, (size_t) kTok * kIxH * kIx), lq);
  CHECK(!q4x_ix_q(iqr, kTok, kIxH, kIx + 2, qn, kEps, 0, kBase, iq, nullptr));  // a rotation past the head

  // the pooled keys from llama.cpp's raw keys
  float* pooled = pooled_of(raw, kn);
  Ref lk750;
  lk750.v.assign(lk.v.begin(), lk.v.begin() + (size_t) kPools * kIx);
  held("pooled keys (indexer_k)", host_of(pooled, (size_t) kPools * kIx), lk750, 1.0, kF16Rel);

  // the picks from llama.cpp's query and pooled keys
  int32_t *sel = dev_alloc<int32_t>((size_t) kTok * kBlocks), *npick = dev_alloc<int32_t>(kTok);
  picks(lq, lk, sel, npick);
  const std::vector<int32_t> s = ints_of(sel, (size_t) kTok * kBlocks), np = ints_of(npick, kTok);

  // the attention over the picks, from llama.cpp's q, k, v (the cache in Q8_0)
  const Ref rq = ref("Qcur-3", kDump), rk = ref("Kcur-3#1", kDump), rv = ref("Vcur-3#1", kDump), full = ref("Qcur_full-3", kDump);
  std::vector<float> gate((size_t) kTok * kQ);
  for (uint32_t t = 0; t < kTok; t++)
    for (uint32_t h = 0; h < kHeads; h++)
      std::memcpy(&gate[((size_t) t * kHeads + h) * kHead], &full.v[(size_t) t * 2 * kQ + (2 * h + 1) * kHead], kHead * 4);
  const uint64_t row = q4x_kv_row_bytes(kKvW);
  uint8_t *kc = dev_alloc<uint8_t>(kTok * row), *vc = dev_alloc<uint8_t>(kTok * row);
  CHECK(q4x_kv_put(dev_from(rk.v), dev_from(rv.v), kTok, kKvW, kc, vc, nullptr));
  std::vector<uint8_t> hk(kTok * row), hv(kTok * row);
  CHECK(hipMemcpy(hk.data(), kc, hk.size(), hipMemcpyDeviceToHost) == hipSuccess &&
        hipMemcpy(hv.data(), vc, hv.size(), hipMemcpyDeviceToHost) == hipSuccess);
  float *dq = dev_from(rq.v), *dg = dev_from(gate), *out = dev_alloc<float>((size_t) kTok * kQ);
  float* scr = dev_alloc<float>(q4x_attn_scratch(kHeads, kTok));
  const Q4xPick pick = {sel, npick, kR, kBlocks};
  CHECK(q4x_attn(dq, kc, vc, kTok, kHeads, kKv, kHead, 0, dg, &pick, scr, out, nullptr, nullptr));
  const std::vector<float> whole = host_of(out, (size_t) kTok * kQ);
  // double over each sampled token's picked keys (the cache's own values)
  auto kv = [&](const std::vector<uint8_t>& c, uint32_t p, uint32_t i) {
    const uint8_t* blk = &c[p * row + (i / Q4X_KV_BLOCK) * Q4X_KV_BLOCK_BYTES];
    _Float16 sc;
    std::memcpy(&sc, blk, 2);
    return (double) (float) sc * (double) (int8_t) blk[2 + i % Q4X_KV_BLOCK];
  };
  Ref ex;
  std::vector<float> got;
  for (uint32_t t : {100u, 2050u, 2051u, 2052u, 2500u, 2998u, 2999u}) {
    const std::vector<uint32_t> keys = keys_of(s, np, t);
    for (uint32_t h = 0; h < kHeads; h++) {
      const uint32_t g = h / (kHeads / kKv);
      std::vector<double> p(keys.size());
      double m = -INFINITY, sum = 0;
      for (size_t i = 0; i < keys.size(); i++) {
        double d = 0;
        for (uint32_t x = 0; x < kHead; x++) d += (double) rq.v[((size_t) t * kHeads + h) * kHead + x] * kv(hk, keys[i], g * kHead + x);
        p[i] = d / 16.0, m = std::max(m, p[i]);
      }
      for (double& v : p) sum += (v = std::exp(v - m));
      for (uint32_t x = 0; x < kHead; x++) {
        double o = 0;
        for (size_t i = 0; i < keys.size(); i++) o += p[i] * kv(hv, keys[i], g * kHead + x);
        const size_t at = ((size_t) t * kHeads + h) * kHead + x;
        ex.v.push_back((float) (o / sum / (1 + std::exp(-(double) gate[at])))), got.push_back(whole[at]);
      }
    }
  }
  held("  over its picks against double (sampled)", got, ex, 1.0, kF16Rel);
  held("  against llama.cpp's (attn_gated)", whole, ref("attn_gated-3", kDump), 1.0, kRelL2);
  // in pieces, and a token at a time at the end: the bits of one run
  uint32_t at = 0;
  for (uint32_t n : kSteps) {
    const Q4xPick pp = {sel + (size_t) at * kBlocks, npick + at, kR, kBlocks};
    CHECK(q4x_attn(dq + (size_t) at * kQ, kc, vc, n, kHeads, kKv, kHead, at, dg + (size_t) at * kQ, &pp, scr,
                   out + (size_t) at * kQ, nullptr, nullptr));
    at += n;
  }
  CHECK(std::memcmp(host_of(out, whole.size()).data(), whole.data(), whole.size() * 4) == 0);
  for (uint32_t t = kTok - 3; t < kTok; t++) {
    const Q4xPick pp = {sel + (size_t) t * kBlocks, npick + t, kR, kBlocks};
    CHECK(q4x_attn(dq + (size_t) t * kQ, kc, vc, 1, kHeads, kKv, kHead, t, dg + (size_t) t * kQ, &pp, scr, out,
                   nullptr, nullptr));
    CHECK(std::memcmp(host_of(out, kQ).data(), &whole[(size_t) t * kQ], kQ * 4) == 0);
  }
  // every block picked (below the budget): the dense attention's bits
  CHECK(q4x_attn(dq, kc, vc, 2048, kHeads, kKv, kHead, 0, dg, nullptr, scr, out, nullptr, nullptr));
  CHECK(std::memcmp(host_of(out, (size_t) 2048 * kQ).data(), whole.data(), (size_t) 2048 * kQ * 4) == 0);
  std::printf("  in pieces and one at a time the bits of one run; every block picked, the dense attention's bits\n");
  gguf_destroy(gg);
  std::printf("[test_q4x_qsa_gpu] ok\n");
  return 0;
}
