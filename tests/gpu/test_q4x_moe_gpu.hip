// Qwen's MoE against llama.cpp's own tensors (q4x_ref.h), layer 0, from its mixed input (hc_mixed-0#1): the router's
// logits (F32: within F32's rounding), the routing's normalized weights (the top 10 in llama.cpp's order), the routed
// experts' sum (ffn_moe_out), the shared expert (ffn_shexp) and the two together (ffn_out) - through quantized
// matmuls, so within llama.cpp's rounding - the experts grouped by expert (8 tokens a read) the bits of each slot
// alone, and the first tokens' experts' sum against the same in double from the exact weights. needs a gfx1151, the UD-IQ3_XXS set and the dump; skipped (77) without them.
#include "q4x_ref.h"

extern "C" {
#include "osmos/gpu/mv.h"
#include "osmos/gpu/q4x_ops.h"
#include "osmos/gpu/quant.h"
}

constexpr uint32_t kTok = 256, kDim = 2560, kExp = 512, kK = 10, kFf = 640, kEntries = kTok * kK, kExactTok = 16;

struct Experts {
  const void *gate, *up, *down;
  uint32_t tg, tu, td;
  uint64_t bg, bu, bd; /* an expert's bytes */
};

static Experts experts(const Gguf* g) {
  const GgufTensor *a, *b, *c;
  Experts e{dev_tensor(g, "blk.0.ffn_gate_exps.weight", &a), dev_tensor(g, "blk.0.ffn_up_exps.weight", &b),
            dev_tensor(g, "blk.0.ffn_down_exps.weight", &c), a->type, b->type, c->type, 0, 0, 0};
  e.bg = a->bytes / kExp, e.bu = b->bytes / kExp, e.bd = c->bytes / kExp;
  return e;
}

// the k experts' outputs for every slot, grouped by expert `per` tokens a read: y in the grouping's order, where[slot]
// its place
struct Routed {
  float* y;
  int32_t* where;
};

static Routed routed(const Experts& e, const float* x, const int32_t* ids, uint32_t per) {
  uint32_t multi, single;
  q4x_expert_bounds(kEntries, kExp, per, &multi, &single);
  const uint32_t bound = multi + single;
  const void** ptr = dev_alloc<const void*>((size_t) Q4X_GROUP_TENSORS * bound);
  int32_t *cnt = dev_alloc<int32_t>(bound), *row = dev_alloc<int32_t>(bound);
  int32_t *tok = dev_alloc<int32_t>(kEntries), *where = dev_alloc<int32_t>(kEntries);
  const uint32_t slots = kEntries + MV_MAX_TOKENS;  // the rows a group reads past its last
  float *xe = dev_alloc<float>((size_t) slots * kDim), *ga = dev_alloc<float>((size_t) slots * kFf);
  float *up = dev_alloc<float>((size_t) slots * kFf), *y = dev_alloc<float>((size_t) slots * kDim);
  Q4xExpertTensors t = {{e.gate, e.up, e.down}, {e.bg, e.bu, e.bd}, {ptr, ptr + bound, ptr + 2 * bound}};
  CHECK(q4x_expert_groups(ids, kEntries, kK, kExp, per, &t, cnt, row, tok, where, nullptr));
  CHECK(q4x_gather(x, tok, kEntries, kDim, xe, nullptr));
  const struct {
    uint32_t type, rows, cols;
    const void** ptr;
    const float* x;
    float* y;
  } m[] = {{e.tg, kFf, kDim, t.ptr[0], xe, ga}, {e.tu, kFf, kDim, t.ptr[1], xe, up}, {e.td, kDim, kFf, t.ptr[2], ga, y}};
  for (int i = 0; i < 3; i++) {
    if (i == 2) CHECK(q4x_swiglu(ga, up, (uint64_t) kEntries * kFf, ga, nullptr));
    CHECK(!multi || mv_group_tok(m[i].type, m[i].ptr, cnt, row, multi, per, m[i].rows, m[i].cols, m[i].x, m[i].cols,
                                 m[i].y, m[i].rows, nullptr));
    CHECK(mv_group_tok(m[i].type, m[i].ptr + multi, cnt + multi, row + multi, single, 1, m[i].rows, m[i].cols, m[i].x,
                       m[i].cols, m[i].y, m[i].rows, nullptr));
  }
  return {y, where};
}

static std::vector<float> exact(uint32_t type, const void* w, uint64_t values) {
  float* d = dev_alloc<float>(values);
  CHECK(quant_dequant(type, w, values, d, nullptr));
  return host_of(d, values);
}

int main() {
  std::printf("[test_q4x_moe_gpu] start\n");
  int ndev = 0;
  Ref mixed;
  char err[GGUF_ERR] = "";
  Gguf* g = gguf_create(home_path(kQwenFirst).c_str(), err);
  if (!g || !ref_read("hc_mixed-0#1", &mixed) || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_q4x_moe_gpu] skipped: no Qwen3.8 UD-IQ3_XXS, its dump or a GPU\n");
    return 77;
  }
  CHECK(!q4x_route(nullptr, 1, kExp, kExp + 1, nullptr, nullptr, nullptr, nullptr));
  const Experts e = experts(g);
  const GgufTensor *ts, *tsg, *tsu, *tsd;
  const void* router = dev_tensor(g, "blk.0.ffn_gate_inp.weight");
  const void* sgate = dev_tensor(g, "blk.0.ffn_gate_inp_shexp.weight", &ts);
  const void *sg = dev_tensor(g, "blk.0.ffn_gate_shexp.weight", &tsg), *su = dev_tensor(g, "blk.0.ffn_up_shexp.weight", &tsu);
  const void* sd = dev_tensor(g, "blk.0.ffn_down_shexp.weight", &tsd);

  // the router and the routing
  float *x = dev_from(mixed.v), *logits = dev_alloc<float>(kTok * kExp), *w = dev_alloc<float>(kEntries);
  int32_t *ids = dev_alloc<int32_t>(kEntries), *rows = dev_alloc<int32_t>(kEntries);
  CHECK(mv_strided(0, router, kExp, kDim, x, kDim, kTok, logits, kExp, nullptr));
  held("router (ffn_moe_logits)", host_of(logits, kTok * kExp), ref("ffn_moe_logits-0"), 1.0, kExactRel);
  CHECK(q4x_route(logits, kTok, kExp, kK, ids, w, rows, nullptr));
  held("routing weights (ffn_moe_weights_norm)", host_of(w, kEntries), ref("ffn_moe_weights_norm-0"), 1.0, kExactRel);

  // the experts, the shared expert, both
  const Routed r = routed(e, x, ids, MV_MAX_TOKENS), alone = routed(e, x, ids, 1);
  float *sh_g = dev_alloc<float>(kTok * kFf), *sh_u = dev_alloc<float>(kTok * kFf), *sh = dev_alloc<float>(kTok * kDim);
  float *sl = dev_alloc<float>(kTok), *out = dev_alloc<float>(kTok * kDim), *zero = dev_alloc<float>(kTok * kDim);
  CHECK(mv_strided(tsg->type, sg, kFf, kDim, x, kDim, kTok, sh_g, kFf, nullptr));
  CHECK(mv_strided(tsu->type, su, kFf, kDim, x, kDim, kTok, sh_u, kFf, nullptr));
  CHECK(q4x_swiglu(sh_g, sh_u, (uint64_t) kTok * kFf, sh_g, nullptr));
  CHECK(mv_strided(tsd->type, sd, kDim, kFf, sh_g, kFf, kTok, sh, kDim, nullptr));
  CHECK(mv_strided(ts->type, sgate, 1, kDim, x, kDim, kTok, sl, 1, nullptr));
  held("shared expert (ffn_shexp)", host_of(sh, kTok * kDim), ref("ffn_shexp-0"), 1.0, kChainRel);
  CHECK(hipMemset(zero, 0, kTok * kDim * 4) == hipSuccess);
  CHECK(q4x_moe_sum(alone.y, alone.where, w, kTok, kK, kDim, zero, sl, out, nullptr));
  const std::vector<float> one_by_one = host_of(out, kTok * kDim);
  CHECK(q4x_moe_sum(r.y, r.where, w, kTok, kK, kDim, zero, sl, out, nullptr));  // the routed alone: zero shared
  const std::vector<float> moe = host_of(out, kTok * kDim);
  CHECK(!std::memcmp(moe.data(), one_by_one.data(), moe.size() * 4));
  std::printf("  the experts grouped by expert (8 tokens a read): the bits of each slot alone\n");
  held("routed experts (ffn_moe_out)", moe, ref("ffn_moe_out-0"), 1.0, kChainRel);
  CHECK(q4x_moe_sum(r.y, r.where, w, kTok, kK, kDim, sh, sl, out, nullptr));
  held("layer 0's MoE whole (ffn_out)", host_of(out, kTok * kDim), ref("ffn_out-0"), 1.0, kChainRel);

  // the first tokens' routed sum in double from the exact weights (osmos's own routing and inputs)
  const std::vector<float> wg = exact(e.tg, e.gate, (uint64_t) kExp * kFf * kDim);
  const std::vector<float> wu = exact(e.tu, e.up, (uint64_t) kExp * kFf * kDim);
  const std::vector<float> wd = exact(e.td, e.down, (uint64_t) kExp * kDim * kFf);
  const std::vector<int32_t> hid = [&] {
    std::vector<int32_t> v(kEntries);
    CHECK(hipMemcpy(v.data(), ids, kEntries * 4, hipMemcpyDeviceToHost) == hipSuccess);
    return v;
  }();
  const std::vector<float> hw = host_of(w, kEntries);
  Ref ex, got, ll;
  ex.v.assign((size_t) kExactTok * kDim, 0);
  std::vector<double> act(kFf);
  for (uint32_t t = 0; t < kExactTok; t++)
    for (uint32_t j = 0; j < kK; j++) {
      const size_t id = (size_t) hid[t * kK + j];
      for (uint32_t f = 0; f < kFf; f++) {
        double a = 0, b = 0;
        for (uint32_t d = 0; d < kDim; d++) {
          a += (double) wg[(id * kFf + f) * kDim + d] * mixed.v[(size_t) t * kDim + d];
          b += (double) wu[(id * kFf + f) * kDim + d] * mixed.v[(size_t) t * kDim + d];
        }
        act[f] = a / (1 + std::exp(-a)) * b;
      }
      for (uint32_t d = 0; d < kDim; d++) {
        double o = 0;
        for (uint32_t f = 0; f < kFf; f++) o += (double) wd[(id * kDim + d) * kFf + f] * act[f];
        ex.v[(size_t) t * kDim + d] += (float) (hw[t * kK + j] * o);
      }
    }
  got.v.assign(moe.begin(), moe.begin() + (size_t) kExactTok * kDim);
  const Ref lmoe = ref("ffn_moe_out-0");
  ll.v.assign(lmoe.v.begin(), lmoe.v.begin() + (size_t) kExactTok * kDim);
  const double o = held("  osmos against exact (16 tokens)", got.v, ex, 1.0, kExactRel);
  const double l = held("  llama.cpp against exact (16 tokens)", ll.v, ex, 1.0, kChainRel);
  CHECK(o < l);
  gguf_destroy(g);
  std::printf("[test_q4x_moe_gpu] ok\n");
  return 0;
}
