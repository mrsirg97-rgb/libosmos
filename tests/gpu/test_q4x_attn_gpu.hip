// Qwen's sparse attention (dense below its budget) against llama.cpp's own tensors (q4x_ref.h), layer 3 (the first
// attention layer): q's and k's norm and rotation from llama.cpp's own projections (Qcur, Kcur: F32 - within F32's
// rounding), the attention from llama.cpp's own q, k and v against the same in double (on the matrix cores: within
// f16's rounding; llama.cpp keeps its cache in F16), then the whole layer from its mixed input (attn_output). the
// tokens in two halves, and one at a time, through the cache: the bits of one run; and past a thousand positions
// (nine chunks of keys), each token against double and the batch the bits of its tokens one at a time. needs a
// gfx1151, the UD-IQ3_XXS set and the dump; skipped (77) without them.
#include "q4x_ref.h"

extern "C" {
#include "osmos/gpu/mv.h"
#include "osmos/gpu/q4x_ops.h"
}

constexpr uint32_t kTok = 256, kHalf = kTok / 2, kDim = 2560, kHeads = 24, kKv = 2, kHead = 256, kRope = 64,
                   kQ = kHeads * kHead, kKvW = kKv * kHead;
constexpr float kEps = 1e-6f, kBase = 1e7f;
constexpr uint32_t kLongCtx = 1100, kLongTok = 8;  // past eight chunks of 128 keys: the ninth's queries

// attention in double from q, k, v (each [t][head][dim]), causal, h's KV head h / 12, gated
static std::vector<float> exact_attn(const std::vector<float>& q, const std::vector<float>& k,
                                     const std::vector<float>& v, const std::vector<float>& gate) {
  std::vector<float> out((size_t) kTok * kQ);
  std::vector<double> p(kTok);
  for (uint32_t t = 0; t < kTok; t++)
    for (uint32_t h = 0; h < kHeads; h++) {
      const uint32_t g = h / (kHeads / kKv);
      const float* qt = &q[((size_t) t * kHeads + h) * kHead];
      double m = -INFINITY, sum = 0;
      for (uint32_t s = 0; s <= t; s++) {
        double d = 0;
        for (uint32_t i = 0; i < kHead; i++) d += (double) qt[i] * k[((size_t) s * kKv + g) * kHead + i];
        p[s] = d / std::sqrt((double) kHead), m = p[s] > m ? p[s] : m;
      }
      for (uint32_t s = 0; s <= t; s++) sum += (p[s] = std::exp(p[s] - m));
      for (uint32_t i = 0; i < kHead; i++) {
        double o = 0;
        for (uint32_t s = 0; s <= t; s++) o += p[s] * v[((size_t) s * kKv + g) * kHead + i];
        const size_t at = ((size_t) t * kHeads + h) * kHead + i;
        out[at] = (float) (o / sum / (1 + std::exp(-(double) gate[at])));
      }
    }
  return out;
}

// a cache of kLongCtx positions (the dump's 256 keys and values repeated), kLongTok queries at its end (the dump's
// last q's and gates): each against double, and the batch the bits of each alone
static void long_context(const float* dq, const float* dk, const float* dv, const float* dg,
                         const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                         const std::vector<float>& gate, float* scr) {
  std::vector<float> lk((size_t) kLongCtx * kKvW), lv(lk.size());
  for (uint32_t p = 0; p < kLongCtx; p++)
    std::memcpy(&lk[(size_t) p * kKvW], &k[(size_t) (p % kTok) * kKvW], kKvW * 4),
        std::memcpy(&lv[(size_t) p * kKvW], &v[(size_t) (p % kTok) * kKvW], kKvW * 4);
  float *ck = dev_from(lk), *cv = dev_from(lv), *out = dev_alloc<float>((size_t) kLongTok * kQ);
  const uint32_t p0 = kLongCtx - kLongTok;
  const float *qq = dq + (size_t) (kTok - kLongTok) * kQ, *gg = dg + (size_t) (kTok - kLongTok) * kQ;
  CHECK(q4x_attn(qq, ck, cv, kLongTok, kHeads, kKv, kHead, p0, gg, scr, out, nullptr));
  const std::vector<float> batch = host_of(out, (size_t) kLongTok * kQ);
  for (uint32_t t = 0; t < kLongTok; t++) {
    CHECK(q4x_attn(qq + (size_t) t * kQ, ck, cv, 1, kHeads, kKv, kHead, p0 + t, gg + (size_t) t * kQ, scr, out,
                   nullptr));
    CHECK(std::memcmp(host_of(out, kQ).data(), &batch[(size_t) t * kQ], kQ * 4) == 0);
  }
  // double: each query over the positions up to its own
  Ref ex, got;
  for (uint32_t t = 0; t < kLongTok; t++)
    for (uint32_t h = 0; h < kHeads; h++) {
      const uint32_t g = h / (kHeads / kKv), at = kTok - kLongTok + t, last = p0 + t;
      const float* qt = &q[((size_t) at * kHeads + h) * kHead];
      std::vector<double> p(last + 1);
      double m = -INFINITY, sum = 0;
      for (uint32_t s = 0; s <= last; s++) {
        double d = 0;
        for (uint32_t i = 0; i < kHead; i++) d += (double) qt[i] * lk[((size_t) s * kKv + g) * kHead + i];
        p[s] = d / 16.0, m = p[s] > m ? p[s] : m;
      }
      for (uint32_t s = 0; s <= last; s++) sum += (p[s] = std::exp(p[s] - m));
      for (uint32_t i = 0; i < kHead; i++) {
        double o = 0;
        for (uint32_t s = 0; s <= last; s++) o += p[s] * lv[((size_t) s * kKv + g) * kHead + i];
        const size_t gi = ((size_t) at * kHeads + h) * kHead + i;
        ex.v.push_back((float) (o / sum / (1 + std::exp(-(double) gate[gi]))));
      }
    }
  held("  past 1,000 positions (9 chunks) against exact", batch, ex, 1.0, kF16Rel);
  std::printf("  past 1,000 positions: each token's bits the batch's\n");
}

// the gates out of llama.cpp's own Qcur_full: each head's second half
static std::vector<float> gates_of(const Ref& full) {
  std::vector<float> g((size_t) kTok * kQ);
  for (uint32_t t = 0; t < kTok; t++)
    for (uint32_t h = 0; h < kHeads; h++)
      std::memcpy(&g[((size_t) t * kHeads + h) * kHead], &full.v[(size_t) t * 2 * kQ + (2 * h + 1) * kHead], kHead * 4);
  return g;
}

int main() {
  std::printf("[test_q4x_attn_gpu] start\n");
  int ndev = 0;
  Ref full;
  char err[GGUF_ERR] = "";
  Gguf* gg = gguf_create(home_path(kQwenFirst).c_str(), err);
  if (!gg || !ref_read("Qcur_full-3", &full) || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_q4x_attn_gpu] skipped: no Qwen3.8 UD-IQ3_XXS, its dump or a GPU\n");
    return 77;
  }
  CHECK(!q4x_attn(nullptr, nullptr, nullptr, 1, kHeads, 5, kHead, 0, nullptr, nullptr, nullptr, nullptr));  // 24 % 5
  CHECK(!q4x_attn(nullptr, nullptr, nullptr, 1, kHeads, kKv, kHead, OPS_Q4X_MAX_KEYS, nullptr, nullptr, nullptr,
                  nullptr));
  float* scr = dev_alloc<float>(q4x_attn_scratch(kHeads, kLongCtx));
  const GgufTensor *tq, *tk, *tv, *to;
  const void *wq = dev_tensor(gg, "blk.3.attn_q.weight", &tq), *wk = dev_tensor(gg, "blk.3.attn_k.weight", &tk);
  const void *wv = dev_tensor(gg, "blk.3.attn_v.weight", &tv), *wo = dev_tensor(gg, "blk.3.attn_output.weight", &to);
  const float* qn = (const float*) dev_tensor(gg, "blk.3.attn_q_norm.weight");
  const float* kn = (const float*) dev_tensor(gg, "blk.3.attn_k_norm.weight");

  // the norms and the rotation, from llama.cpp's own projections
  const Ref kraw = ref("Kcur-3");
  float *q = dev_alloc<float>((size_t) kTok * kQ), *gate = dev_alloc<float>((size_t) kTok * kQ);
  float *kc = dev_alloc<float>((size_t) kTok * kKvW), *vc = dev_alloc<float>((size_t) kTok * kKvW);
  CHECK(q4x_qk(dev_from(full.v), dev_from(kraw.v), kTok, kHeads, kKv, kHead, kRope, qn, kn, kEps, 0, kBase, q, gate,
               kc, nullptr));
  held("q normed, rotated (Qcur)", host_of(q, (size_t) kTok * kQ), ref("Qcur-3"), 1.0, kExactRel);
  held("k normed, rotated (Kcur)", host_of(kc, (size_t) kTok * kKvW), ref("Kcur-3#1"), 1.0, kExactRel);
  CHECK(std::memcmp(host_of(gate, (size_t) kTok * kQ).data(), gates_of(full).data(), (size_t) kTok * kQ * 4) == 0);

  // the attention from llama.cpp's own q, k, v; then in two halves through the cache: the same bits
  const Ref rq = ref("Qcur-3"), rk = ref("Kcur-3#1"), rv = ref("Vcur-3#1");
  const std::vector<float> rg = gates_of(full);
  float *dq = dev_from(rq.v), *dk = dev_from(rk.v), *dv = dev_from(rv.v), *dg = dev_from(rg);
  float* out = dev_alloc<float>((size_t) kTok * kQ);
  CHECK(q4x_attn(dq, dk, dv, kTok, kHeads, kKv, kHead, 0, dg, scr, out, nullptr));
  const std::vector<float> whole = host_of(out, (size_t) kTok * kQ);
  CHECK(q4x_attn(dq, dk, dv, kHalf, kHeads, kKv, kHead, 0, dg, scr, out, nullptr));
  CHECK(q4x_attn(dq + (size_t) kHalf * kQ, dk, dv, kHalf, kHeads, kKv, kHead, kHalf, dg + (size_t) kHalf * kQ, scr,
                 out + (size_t) kHalf * kQ, nullptr));
  CHECK(std::memcmp(host_of(out, whole.size()).data(), whole.data(), whole.size() * 4) == 0);
  Ref ex;
  ex.v = exact_attn(rq.v, rk.v, rv.v, rg);
  held("  osmos against exact (attn_gated)", whole, ex, 1.0, kF16Rel);
  held("  llama.cpp against exact (attn_gated)", ref("attn_gated-3").v, ex, 1.0, kRelL2);
  for (uint32_t t = kTok - 3; t < kTok; t++) {  // one at a time
    CHECK(q4x_attn(dq + (size_t) t * kQ, dk, dv, 1, kHeads, kKv, kHead, t, dg + (size_t) t * kQ, scr, out, nullptr));
    CHECK(std::memcmp(host_of(out, kQ).data(), &whole[(size_t) t * kQ], kQ * 4) == 0);
  }
  std::printf("  the tokens in halves, and one at a time: the bits of one run\n");
  long_context(dq, dk, dv, dg, rq.v, rk.v, rv.v, rg, scr);

  // the whole layer from its mixed input: q, k, v, the norms and rotation, the attention, attn_output
  const Ref mixed = ref("hc_mixed-3");
  float *hm = dev_from(mixed.v), *qf = dev_alloc<float>((size_t) kTok * 2 * kQ), *kr = dev_alloc<float>(kTok * kKvW);
  float* res = dev_alloc<float>((size_t) kTok * kDim);
  CHECK(mv_strided(tq->type, wq, 2 * kQ, kDim, hm, kDim, kTok, qf, 2 * kQ, nullptr));
  CHECK(mv_strided(tk->type, wk, kKvW, kDim, hm, kDim, kTok, kr, kKvW, nullptr));
  CHECK(mv_strided(tv->type, wv, kKvW, kDim, hm, kDim, kTok, vc, kKvW, nullptr));
  held("q's projection (Qcur_full)", host_of(qf, (size_t) kTok * 2 * kQ), full);
  CHECK(q4x_qk(qf, kr, kTok, kHeads, kKv, kHead, kRope, qn, kn, kEps, 0, kBase, q, gate, kc, nullptr));
  CHECK(q4x_attn(q, kc, vc, kTok, kHeads, kKv, kHead, 0, gate, scr, out, nullptr));
  CHECK(mv_strided(to->type, wo, kDim, kQ, out, kQ, kTok, res, kDim, nullptr));
  held("layer 3 whole (attn_output)", host_of(res, (size_t) kTok * kDim), ref("attn_output-3"), 1.0, kChainRel);
  gguf_destroy(gg);
  std::printf("[test_q4x_attn_gpu] ok\n");
  return 0;
}
