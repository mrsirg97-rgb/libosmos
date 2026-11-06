// Qwen's Gated DeltaNet against llama.cpp's own tensors (q4x_ref.h), layer 0, each op from llama.cpp's own inputs:
// the conv (conv_output_silu), the gates (gate, beta_sigmoid), the delta rule (attn_output) and the gated norm
// (final_output) - their weights F32, so within F32's own rounding - then the whole layer from its mixed input
// (linear_attn_out: through quantized matmuls, llama.cpp's rounding). a sequence's state carried: the tokens in two
// halves the bits of one run. needs a gfx1151, the UD-IQ3_XXS set and the dump; skipped (77) without them.
#include "q4x_ref.h"

extern "C" {
#include "osmos/gpu/mv.h"
#include "osmos/gpu/q4x_ops.h"
}

constexpr uint32_t kTok = 256, kHalf = kTok / 2, kDim = 2560, kCh = 10240, kTaps = 4, kKHeads = 16, kVHeads = 48,
                   kHead = 128, kInner = kVHeads * kHead;
constexpr float kEps = 1e-6f;

struct Layer {
  const void *qkv, *gate, *alpha, *beta, *out;
  uint32_t t_qkv, t_gate, t_out;
  const float *conv, *dt, *a, *norm;
};

static Layer layer0(const Gguf* g) {
  const GgufTensor *q, *z, *o;
  Layer l{};
  l.qkv = dev_tensor(g, "blk.0.attn_qkv.weight", &q), l.gate = dev_tensor(g, "blk.0.attn_gate.weight", &z);
  l.out = dev_tensor(g, "blk.0.ssm_out.weight", &o);
  l.alpha = dev_tensor(g, "blk.0.ssm_alpha.weight"), l.beta = dev_tensor(g, "blk.0.ssm_beta.weight");
  l.conv = (const float*) dev_tensor(g, "blk.0.ssm_conv1d.weight");
  l.dt = (const float*) dev_tensor(g, "blk.0.ssm_dt.bias"), l.a = (const float*) dev_tensor(g, "blk.0.ssm_a");
  l.norm = (const float*) dev_tensor(g, "blk.0.ssm_norm.weight");
  l.t_qkv = q->type, l.t_gate = z->type, l.t_out = o->type;
  return l;
}

// the conv over all the tokens, then over two halves with its history carried: the same bits
static std::vector<float> conv(const Layer& l, const float* x) {
  float *hist = dev_alloc<float>((kTaps - 1) * kCh), *y = dev_alloc<float>((size_t) kTok * kCh);
  CHECK(hipMemset(hist, 0, (kTaps - 1) * kCh * 4) == hipSuccess);
  CHECK(q4x_conv(x, kTok, kCh, kTaps, 1, l.conv, hist, y, nullptr));
  const std::vector<float> whole = host_of(y, (size_t) kTok * kCh);
  CHECK(hipMemset(hist, 0, (kTaps - 1) * kCh * 4) == hipSuccess);
  CHECK(q4x_conv(x, kHalf, kCh, kTaps, 1, l.conv, hist, y, nullptr));
  CHECK(q4x_conv(x + (size_t) kHalf * kCh, kHalf, kCh, kTaps, 1, l.conv, hist, y + (size_t) kHalf * kCh, nullptr));
  CHECK(std::memcmp(host_of(y, whole.size()).data(), whole.data(), whole.size() * 4) == 0);
  CHECK(hipFree(hist) == hipSuccess && hipFree(y) == hipSuccess);
  return whole;
}

// the delta rule over all the tokens, then two halves with the state carried: the same bits
static std::vector<float> delta(const float* qkv, const float* g, const float* b) {
  const size_t ns = (size_t) kVHeads * kHead * kHead, no = (size_t) kTok * kInner;
  float *s = dev_alloc<float>(ns), *o = dev_alloc<float>(no);
  CHECK(hipMemset(s, 0, ns * 4) == hipSuccess);
  CHECK(q4x_delta(qkv, g, b, kTok, kKHeads, kVHeads, kHead, kEps, s, o, nullptr));
  const std::vector<float> whole = host_of(o, no), sw = host_of(s, ns);
  CHECK(hipMemset(s, 0, ns * 4) == hipSuccess);
  CHECK(q4x_delta(qkv, g, b, kHalf, kKHeads, kVHeads, kHead, kEps, s, o, nullptr));
  CHECK(q4x_delta(qkv + (size_t) kHalf * kCh, g + kHalf * kVHeads, b + kHalf * kVHeads, kHalf, kKHeads, kVHeads,
                  kHead, kEps, s, o + (size_t) kHalf * kInner, nullptr));
  CHECK(std::memcmp(host_of(o, no).data(), whole.data(), no * 4) == 0);
  CHECK(std::memcmp(host_of(s, ns).data(), sw.data(), ns * 4) == 0);
  CHECK(hipFree(s) == hipSuccess && hipFree(o) == hipSuccess);
  return whole;
}

int main() {
  std::printf("[test_q4x_deltanet_gpu] start\n");
  int ndev = 0;
  Ref probe;
  char err[GGUF_ERR] = "";
  Gguf* g = gguf_create(home_path(kQwenFirst).c_str(), err);
  if (!g || !ref_read("linear_attn_qkv_mixed-0", &probe) || hipGetDeviceCount(&ndev) != hipSuccess || !ndev) {
    std::printf("[test_q4x_deltanet_gpu] skipped: no Qwen3.8 UD-IQ3_XXS, its dump or a GPU\n");
    return 77;
  }
  CHECK(!q4x_conv(nullptr, 0, kCh, kTaps, 1, nullptr, nullptr, nullptr, nullptr));
  CHECK(!q4x_delta(nullptr, nullptr, nullptr, 1, 0, kVHeads, kHead, kEps, nullptr, nullptr, nullptr));
  CHECK(!q4x_delta(nullptr, nullptr, nullptr, 1, 7, kVHeads, kHead, kEps, nullptr, nullptr, nullptr));  // 48 % 7
  const Layer l = layer0(g);

  // each op from llama.cpp's own inputs
  const Ref x = ref("linear_attn_qkv_mixed-0"), mixed = ref("hc_mixed-0");
  const std::vector<float> c = conv(l, dev_from(x.v));
  held("conv (conv_output_silu)", c, ref("conv_output_silu-0"), 1.0, kExactRel);

  float *hm = dev_from(mixed.v), *al = dev_alloc<float>(kTok * kVHeads), *be = dev_alloc<float>(kTok * kVHeads);
  float *gg = dev_alloc<float>(kTok * kVHeads), *bb = dev_alloc<float>(kTok * kVHeads);
  CHECK(mv_strided(0, l.alpha, kVHeads, kDim, hm, kDim, kTok, al, kVHeads, nullptr));
  CHECK(mv_strided(0, l.beta, kVHeads, kDim, hm, kDim, kTok, be, kVHeads, nullptr));
  CHECK(q4x_ssm_gates(al, be, l.dt, l.a, kTok, kVHeads, gg, bb, nullptr));
  held("the decay (gate)", host_of(gg, kTok * kVHeads), ref("gate-0"), 1.0, kExactRel);
  held("beta (beta_sigmoid)", host_of(bb, kTok * kVHeads), ref("beta_sigmoid-0"), 1.0, kExactRel);

  const Ref rc = ref("conv_output_silu-0"), rg = ref("gate-0"), rb = ref("beta_sigmoid-0");
  const std::vector<float> d = delta(dev_from(rc.v), dev_from(rg.v), dev_from(rb.v));
  held("the delta rule (attn_output)", d, ref("attn_output-0"), 1.0, kExactRel);

  const Ref ro = ref("attn_output-0"), rz = ref("z-0");
  float* fo = dev_alloc<float>((size_t) kTok * kInner);
  CHECK(q4x_gated_norm(dev_from(ro.v), l.norm, dev_from(rz.v), kTok, kVHeads, kHead, kEps, fo, nullptr));
  held("the gated norm (final_output)", host_of(fo, (size_t) kTok * kInner), ref("final_output-0"), 1.0, kExactRel);

  // the whole layer from its mixed input: qkv and z, the conv, the gates, the rule, the norm, ssm_out
  float *qkv = dev_alloc<float>((size_t) kTok * kCh), *z = dev_alloc<float>((size_t) kTok * kInner);
  float *cv = dev_alloc<float>((size_t) kTok * kCh), *hist = dev_alloc<float>((kTaps - 1) * kCh);
  float *st = dev_alloc<float>((size_t) kVHeads * kHead * kHead), *ro2 = dev_alloc<float>((size_t) kTok * kInner);
  float* out = dev_alloc<float>((size_t) kTok * kDim);
  CHECK(hipMemset(hist, 0, (kTaps - 1) * kCh * 4) == hipSuccess);
  CHECK(hipMemset(st, 0, (size_t) kVHeads * kHead * kHead * 4) == hipSuccess);
  CHECK(mv_strided(l.t_qkv, l.qkv, kCh, kDim, hm, kDim, kTok, qkv, kCh, nullptr));
  CHECK(mv_strided(l.t_gate, l.gate, kInner, kDim, hm, kDim, kTok, z, kInner, nullptr));
  CHECK(q4x_conv(qkv, kTok, kCh, kTaps, 1, l.conv, hist, cv, nullptr));
  CHECK(q4x_delta(cv, gg, bb, kTok, kKHeads, kVHeads, kHead, kEps, st, ro2, nullptr));
  CHECK(q4x_gated_norm(ro2, l.norm, z, kTok, kVHeads, kHead, kEps, fo, nullptr));
  CHECK(mv_strided(l.t_out, l.out, kDim, kInner, fo, kInner, kTok, out, kDim, nullptr));
  held("layer 0 whole (linear_attn_out)", host_of(out, (size_t) kTok * kDim), ref("linear_attn_out-0"), 1.0,
       kChainRel);
  gguf_destroy(g);
  std::printf("[test_q4x_deltanet_gpu] ok\n");
  return 0;
}
